"""Per-table opponent model for the KRD bot, stored in the `memory` bytes (stdlib + numpy).

Self-contained (pasteable into agent.py).  Binary codec, parsed defensively:

    b"OM" + version(1) + ns(1) + int32[NG + ns*NT] (in-hand transient) + uint16[ns*NC] (counters)

With ns = 8 seats the blob is 3.5 KB, fixed size: it does not grow with hands played.
Any garbage/empty/mis-sized input -> fresh model.

API
    load(b)  -> OppModel          (never raises)
    dump(m)  -> bytes
    update(mem, events, me, state=None) -> OppModel   (mem = bytes or OppModel; never raises)
    action_freqs(mem, seat) -> dict of smoothed frequencies
    reveal_counts(mem, seat, rank=None) -> (n_hi, n_lo, n_pair) showdown reveal buckets
    reveal_posterior(mem, seat, reveal_card, prior_row[169], action=0, k=15.0) -> ndarray[169]
    class_post_to_cards(post169, reveal_card, dead=()) -> ndarray[52] over hidden card ids

Conventions: card id = rank*4 + suit (rank 0..12 = 2..A, suits "cdhs"); the 169-class order
is common.CLASS_LIST (AA, AKs, AKo, AQs, ..., 22).
Showdown bucket of a (keep=hidden, reveal) hole:  "hi" = revealed card is the higher rank,
"lo" = revealed card is the lower rank, "pair" = same rank.
"""
from array import array

import numpy as np

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_DECK = tuple(r + s for r in _RANKS for s in _SUITS)
_CID = {c: i for i, c in enumerate(_DECK)}

# ---------------------------------------------------------------- 169 classes (common.py order)
OM_CLASS_LIST = []
for _i in range(12, -1, -1):
    for _j in range(_i, -1, -1):
        if _i == _j:
            OM_CLASS_LIST.append(_RANKS[_i] * 2)
        else:
            OM_CLASS_LIST.append(_RANKS[_i] + _RANKS[_j] + "s")
            OM_CLASS_LIST.append(_RANKS[_i] + _RANKS[_j] + "o")
_CIDX = {c: i for i, c in enumerate(OM_CLASS_LIST)}
# CLS_RHS[r][h][s]: class index of ranks (r, h), s = 1 suited / 0 offsuit (pairs: both same)
CLS_RHS = np.zeros((13, 13, 2), dtype=np.int64)
for _r in range(13):
    for _h in range(13):
        hi, lo = max(_r, _h), min(_r, _h)
        if hi == lo:
            CLS_RHS[_r, _h, :] = _CIDX[_RANKS[hi] * 2]
        else:
            CLS_RHS[_r, _h, 1] = _CIDX[_RANKS[hi] + _RANKS[lo] + "s"]
            CLS_RHS[_r, _h, 0] = _CIDX[_RANKS[hi] + _RANKS[lo] + "o"]
# CLASS_HAS_RANK[r] : bool[169] classes containing rank r
CLASS_HAS_RANK = np.zeros((13, 169), dtype=bool)
for _r in range(13):
    CLASS_HAS_RANK[_r, CLS_RHS[_r].ravel()] = True


def _crude_strength():
    """Percentile 0..100 (100 = AA) of each class by a Chen-like score, combo-weighted."""
    sc = np.zeros(169)
    combos = np.zeros(169)
    for idx, name in enumerate(OM_CLASS_LIST):
        a, b = _RANKS.index(name[0]) + 2, _RANKS.index(name[1]) + 2
        if a == b:
            sc[idx] = 30 + 1.5 * a
            combos[idx] = 6
        else:
            s = a * 2 + b + (3 if name[2] == "s" else 0) + (2 if a - b == 1 else 0)
            sc[idx] = s
            combos[idx] = 4 if name[2] == "s" else 12
    order = np.argsort(sc, kind="stable")
    pct = np.zeros(169)
    cum = 0.0
    for idx in order:
        pct[idx] = 100.0 * (cum + combos[idx] / 2) / 1326.0
        cum += combos[idx]
    return pct


CLASS_PCT = _crude_strength()

# _UNIF[r]: combo-uniform distribution over classes containing rank r (with that card fixed)
_UNIF = np.zeros((13, 169))
for _r in range(13):
    for _h in range(13):
        if _h == _r:
            _UNIF[_r, CLS_RHS[_r, _h, 0]] = 3.0
        else:
            _UNIF[_r, CLS_RHS[_r, _h, 1]] += 1.0
            _UNIF[_r, CLS_RHS[_r, _h, 0]] += 3.0
    _UNIF[_r] /= _UNIF[_r].sum()  # float[169] in 0..100

# hidden card id -> class given revealed card id; and #cards sharing that class with the reveal
_CLS_OF = np.zeros((52, 52), dtype=np.int64)
_DIV = np.ones((52, 52))
for _rc in range(52):
    _r, _s = _rc >> 2, _rc & 3
    for _hc in range(52):
        _h, _hs = _hc >> 2, _hc & 3
        if _h == _r:
            _CLS_OF[_rc, _hc] = CLS_RHS[_r, _h, 0]
            _DIV[_rc, _hc] = 3.0
        elif _hs == _s:
            _CLS_OF[_rc, _hc] = CLS_RHS[_r, _h, 1]
            _DIV[_rc, _hc] = 1.0
        else:
            _CLS_OF[_rc, _hc] = CLS_RHS[_r, _h, 0]
            _DIV[_rc, _hc] = 3.0

# ---------------------------------------------------------------- layout
OM_VERSION = 1
OM_MAXSEATS = 10
# globals (int32)
G_STREET, G_LEVEL, G_NRAISE, G_AILEVEL, G_BB, G_INHAND, G_HANDS = range(7)
NG = 7
# per-seat transient (int32)
T_STK, T_START, T_BET, T_CONTRIB, T_FLAGS, T_REV = range(6)
NT = 6
F_VPIP, F_PFR, F_JAM = 1, 2, 4
# per-seat counters (uint16)
(C_HANDS, C_VPIP, C_PFR, C_JAM, C_AIOPP, C_AICALL, C_FROPP, C_FRFOLD, C_PAGG, C_PPAS,
 C_SD, C_SDHI, C_SDLO, C_SDPAIR, C_SDSUIT, C_JSD, C_JSDSTR, C_SDSTR, C_REVN, C_SPARE) = range(20)
C_HIST = 20            # 13: revealed-rank histogram (all reveals)
C_MAT = C_HIST + 13    # 169: showdown matrix [revealed rank * 13 + hidden rank]
NC = C_MAT + 169       # 202
_HDR = b"OM"
_GROUP = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4)  # revealed-rank groups: 2-5, 6-9, TJ, QK, A
_GROUP_ROWS = [np.array([g == x for x in _GROUP]) for g in range(5)]
_TRIL = np.tril(np.ones((13, 13)), -1)
_TRIU = np.triu(np.ones((13, 13)), 1)
# _BUCKET[r][h]: 0 = "hi" (h < r), 1 = "lo" (h > r), 2 = pair
_BUCKET = np.array([[0 if h < r else (1 if h > r else 2) for h in range(13)] for r in range(13)])


class OppModel:
    __slots__ = ("ns", "t", "c")

    def __init__(self, ns=8, start_stack=0):
        ns = max(2, min(OM_MAXSEATS, int(ns)))
        self.ns = ns
        self.t = array("i", [0]) * (NG + ns * NT)
        self.c = array("H", [0]) * (ns * NC)
        if start_stack:
            for s in range(ns):
                self.t[NG + s * NT + T_STK] = int(start_stack)


def fresh(state=None):
    ns, st = 8, 0
    try:
        ns = int(state.config.players)
        st = int(state.config.starting_stack)
    except Exception:
        pass
    m = OppModel(ns, st)
    if state is not None and not st:
        try:
            for si in state.seats:
                if 0 <= si.seat < m.ns:
                    m.t[NG + si.seat * NT + T_STK] = int(si.stack + si.bet)
        except Exception:
            pass
    return m


def load(b, state=None):
    """bytes -> OppModel; anything malformed -> fresh model."""
    try:
        if not b or len(b) < 4 or b[:2] != _HDR or b[2] != OM_VERSION:
            return fresh(state)
        ns = b[3]
        if not 2 <= ns <= OM_MAXSEATS:
            return fresh(state)
        ni = NG + ns * NT
        nc = ns * NC
        if len(b) != 4 + 4 * ni + 2 * nc:
            return fresh(state)
        m = OppModel.__new__(OppModel)
        m.ns = ns
        m.t = array("i")
        m.t.frombytes(bytes(b[4:4 + 4 * ni]))
        m.c = array("H")
        m.c.frombytes(bytes(b[4 + 4 * ni:]))
        return m
    except Exception:
        return fresh(state)


def dump(m):
    return _HDR + bytes((OM_VERSION, m.ns)) + m.t.tobytes() + m.c.tobytes()


def _inc(c, i, v=1):
    x = c[i] + v
    c[i] = x if x < 65535 else 65535


def _cid(card):
    if isinstance(card, str):
        return _CID.get(card, -1)
    try:
        card = int(card)
        return card if 0 <= card < 52 else -1
    except Exception:
        return -1


def _new_hand(m):
    t = m.t
    t[G_STREET] = 0
    t[G_LEVEL] = 0
    t[G_NRAISE] = 0
    t[G_AILEVEL] = 0
    t[G_BB] = 0
    t[G_INHAND] = 1
    t[G_HANDS] += 1
    for s in range(m.ns):
        o = NG + s * NT
        t[o + T_START] = t[o + T_STK]
        t[o + T_BET] = 0
        t[o + T_CONTRIB] = 0
        t[o + T_FLAGS] = 0
        t[o + T_REV] = 0


def _event(m, e):
    t, c = m.t, m.c
    kind = e.kind
    if kind == "hand":
        _new_hand(m)
        return
    if kind == "board":
        t[G_STREET] += 1
        t[G_LEVEL] = 0
        t[G_NRAISE] = 0
        t[G_AILEVEL] = 0
        for s in range(m.ns):
            t[NG + s * NT + T_BET] = 0
        return
    s = e.seat
    if s is None or not 0 <= s < m.ns:
        return
    o = NG + s * NT
    oc = s * NC
    pre = t[G_STREET] == 0
    if kind == "blind":
        a = int(e.amount or 0)
        t[o + T_BET] += a
        t[o + T_CONTRIB] += a
        if t[o + T_BET] > t[G_LEVEL]:
            t[G_LEVEL] = t[o + T_BET]
        if a > t[G_BB]:
            t[G_BB] = a
    elif kind == "reveal":
        if e.cards:
            cid = _cid(e.cards[0])
            if cid >= 0:
                t[o + T_REV] = cid + 1
                _inc(c, oc + C_HIST + (cid >> 2))
                _inc(c, oc + C_REVN)
    elif kind == "raise":
        amt = int(e.amount or 0)
        if t[G_NRAISE] > 0:
            _inc(c, oc + C_FROPP)
        if t[G_AILEVEL] and t[o + T_BET] < t[G_LEVEL]:
            _inc(c, oc + C_AIOPP)
            _inc(c, oc + C_AICALL)
        add = amt - t[o + T_BET]
        if add < 0:
            add = 0
        t[o + T_BET] = amt
        t[o + T_CONTRIB] += add
        allin = t[o + T_START] > 0 and t[o + T_CONTRIB] >= t[o + T_START]
        if pre:
            t[o + T_FLAGS] |= F_VPIP | F_PFR | (F_JAM if allin else 0)
        else:
            _inc(c, oc + C_PAGG)
        if amt > t[G_LEVEL]:
            t[G_LEVEL] = amt
            t[G_AILEVEL] = 1 if allin else 0
        t[G_NRAISE] += 1
    elif kind == "call":
        add = int(e.amount or 0)
        if t[G_NRAISE] > 0:
            _inc(c, oc + C_FROPP)
        if t[G_AILEVEL]:
            _inc(c, oc + C_AIOPP)
            _inc(c, oc + C_AICALL)
        t[o + T_BET] += add
        t[o + T_CONTRIB] += add
        if pre:
            t[o + T_FLAGS] |= F_VPIP
        else:
            _inc(c, oc + C_PPAS)
    elif kind == "fold":
        if t[G_NRAISE] > 0:
            _inc(c, oc + C_FROPP)
            _inc(c, oc + C_FRFOLD)
        if t[G_AILEVEL]:
            _inc(c, oc + C_AIOPP)
    elif kind == "check":
        if not pre:
            _inc(c, oc + C_PPAS)
    elif kind == "show":
        cards = e.cards or ()
        if len(cards) != 2:
            return
        a, b = _cid(cards[0]), _cid(cards[1])
        if a < 0 or b < 0:
            return
        rv = t[o + T_REV] - 1
        if rv == a:
            hid = b
        elif rv == b:
            hid = a
        else:  # reveal unknown: show = (keep, reveal)
            rv, hid = b, a
        r, h = rv >> 2, hid >> 2
        _inc(c, oc + C_SD)
        _inc(c, oc + (C_SDPAIR if r == h else (C_SDHI if r > h else C_SDLO)))
        if r != h and (rv & 3) == (hid & 3):
            _inc(c, oc + C_SDSUIT)
        _inc(c, oc + C_MAT + r * 13 + h)
        pct = int(CLASS_PCT[CLS_RHS[r, h, 1 if (rv & 3) == (hid & 3) else 0]])
        _inc(c, oc + C_SDSTR, pct)
        if t[o + T_FLAGS] & F_JAM:
            _inc(c, oc + C_JSD)
            _inc(c, oc + C_JSDSTR, pct)
    elif kind == "stack":
        t[o + T_STK] = int(e.amount or 0)
        f = t[o + T_FLAGS]
        _inc(c, oc + C_HANDS)
        if f & F_VPIP:
            _inc(c, oc + C_VPIP)
        if f & F_PFR:
            _inc(c, oc + C_PFR)
        if f & F_JAM:
            _inc(c, oc + C_JAM)
        t[o + T_FLAGS] = 0
        t[G_INHAND] = 0


def update(mem, events, me=None, state=None):
    """Process new events (state.events = events since our last decision). Never raises."""
    m = mem if isinstance(mem, OppModel) else load(mem, state)
    try:
        if state is not None:
            ns = int(state.config.players)
            if ns != m.ns and 2 <= ns <= OM_MAXSEATS:
                m = fresh(state)
    except Exception:
        pass
    for e in events or ():
        try:
            _event(m, e)
        except Exception:
            pass
    return m


# ---------------------------------------------------------------- queries
def _sm(x, n, p, k):
    return (x + k * p) / (n + k)


def action_freqs(mem, seat):
    """Smoothed frequencies (prior, pseudo-count k) for one seat."""
    c = mem.c
    if not 0 <= seat < mem.ns:
        o = None
    else:
        o = seat * NC
    g = (lambda i: c[o + i]) if o is not None else (lambda i: 0)
    hands = g(C_HANDS)
    sd = g(C_SD)
    out = {
        "hands": hands,
        "vpip": _sm(g(C_VPIP), hands, 0.30, 10.0),
        "pfr": _sm(g(C_PFR), hands, 0.15, 10.0),
        "jam": _sm(g(C_JAM), hands, 0.06, 20.0),
        "call_jam": _sm(g(C_AICALL), g(C_AIOPP), 0.35, 8.0),
        "fold_to_raise": _sm(g(C_FRFOLD), g(C_FROPP), 0.55, 8.0),
        "aggression": _sm(g(C_PAGG), g(C_PAGG) + g(C_PPAS), 0.40, 10.0),
        "n_sd": sd,
        "reveal_hi": _sm(g(C_SDHI), sd, 0.55, 6.0),
        "reveal_lo": _sm(g(C_SDLO), sd, 0.30, 6.0),
        "reveal_pair": _sm(g(C_SDPAIR), sd, 0.15, 6.0),
        "sd_strength": _sm(g(C_SDSTR), sd, 60.0, 4.0),
        "jam_strength": _sm(g(C_JSDSTR), g(C_JSD), 75.0, 3.0),
    }
    return out


def _mat(mem, seat):
    o = seat * NC + C_MAT
    return np.frombuffer(mem.c, dtype=np.uint16, count=169, offset=2 * o).reshape(13, 13).astype(np.float64)


def reveal_counts(mem, seat, rank=None):
    """(n_hi, n_lo, n_pair) over showdowns, for one revealed rank (0..12) or all ranks.
    E.g. reveal_counts(m, s, 12) -> how often a revealed Ace had a pair / lower hidden card."""
    if not 0 <= seat < mem.ns:
        return (0, 0, 0)
    M = _mat(mem, seat)
    rows = range(13) if rank is None else (rank,)
    hi = lo = pr = 0
    for r in rows:
        hi += int(M[r, :r].sum())
        lo += int(M[r, r + 1:].sum())
        pr += int(M[r, r])
    return hi, lo, pr


def reveal_posterior(mem, seat, reveal_card, prior_row, action=0, k=15.0, floor=0.01):
    """P(class | revealed card) for `seat`: Dirichlet blend of the equilibrium prior row with the
    seat's showdown data.  action: 0 none, 1 voluntarily raised this hand, 2 jammed.
    floor: mass of a combo-uniform distribution mixed into the prior so that classes the
    equilibrium never keeps can still pick up evidence (0 disables).
    Returns float64[169] summing to 1; classes without the revealed rank are 0."""
    r = _cid(reveal_card)
    r = r >> 2 if r >= 0 else 12
    P = np.asarray(prior_row, dtype=np.float64).reshape(169).copy()
    P[~np.isfinite(P)] = 0.0
    P[P < 0] = 0.0
    P[~CLASS_HAS_RANK[r]] = 0.0
    tot = P.sum()
    if tot <= 0:  # no prior: uniform over combos sharing the revealed card
        P = _UNIF[r].copy()
        tot = 1.0
    P /= tot
    if floor > 0:
        P = (1.0 - floor) * P + floor * _UNIF[r]
    s_idx = CLS_RHS[r, :, 1]
    o_idx = CLS_RHS[r, :, 0]
    Ph = P[s_idx] + P[o_idx]
    Ph[r] = P[o_idx[r]]  # pair counted once
    W = Ph
    if 0 <= seat < mem.ns and mem.c[seat * NC + C_SD] > 0:
        M = _mat(mem, seat)
        lo_r = (M * _TRIL).sum(1)   # per revealed rank: hidden lower  -> "hi" bucket
        hi_r = (M * _TRIU).sum(1)   # hidden higher -> "lo" bucket
        pr_r = M.diagonal()
        rows3 = np.stack((lo_r, hi_r, pr_r), 1)  # [13, 3]
        n_a = rows3.sum(0)
        n_g = rows3[_GROUP_ROWS[_GROUP[r]]].sum(0)
        bucket = _BUCKET[r]
        Pb = np.bincount(bucket, weights=Ph, minlength=3)
        n_eff = n_g + 0.3 * (n_a - n_g)
        n_eff[Pb <= 0] = 0.0
        Mb = (k * Pb + n_eff) / (k + n_eff.sum())
        C = M[r]
        Cb = np.bincount(bucket, weights=C, minlength=3)
        Pbh = Pb[bucket]
        safe = np.where(Pbh > 0, Pbh, 1.0)
        within = np.where(Pbh > 0, (k * Ph / safe + C) / (k + Cb[bucket]), 0.0)
        W = Mb[bucket] * within
    # spread each hidden rank's mass over its classes (prior shape; combos if prior empty)
    ps = P[s_idx]
    po = P[o_idx]
    den = ps + po
    fs = np.where(den > 0, ps / np.where(den > 0, den, 1.0), 0.25)
    out = np.zeros(169)
    out[s_idx] = W * fs
    out[o_idx] = W * (1.0 - fs)
    out[o_idx[r]] = W[r]
    if action:
        fr = action_freqs(mem, seat)
        if action >= 2:
            beta = np.log(0.10 / fr["jam"]) * 0.6
            beta = min(1.5, max(0.0, beta))
        else:
            beta = np.log(0.20 / fr["pfr"]) * 0.5
            beta = min(1.0, max(0.0, beta))
        if beta > 0:
            out *= np.exp(beta * (CLASS_PCT - 50.0) / 50.0)
    out[~CLASS_HAS_RANK[r]] = 0.0
    s = out.sum()
    if s <= 0 or not np.isfinite(s):
        out = CLASS_HAS_RANK[r].astype(np.float64)
        s = out.sum()
    return out / s


def class_post_to_cards(post169, reveal_card, dead=()):
    """Spread a 169-class posterior (for a seat whose revealed card is `reveal_card`) over the
    52 possible hidden card ids. dead = card ids/strings known elsewhere (board, our hand, other
    reveals). Returns float64[52] summing to 1 (uniform over live cards if everything is 0)."""
    rc = _cid(reveal_card)
    if rc < 0:
        rc = 51
    w = np.asarray(post169, dtype=np.float64)[_CLS_OF[rc]] / _DIV[rc]
    w[rc] = 0.0
    for d in dead:
        x = _cid(d)
        if x >= 0:
            w[x] = 0.0
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        w = np.ones(52)
        w[rc] = 0.0
        for d in dead:
            x = _cid(d)
            if x >= 0:
                w[x] = 0.0
        s = w.sum()
    return w / s
