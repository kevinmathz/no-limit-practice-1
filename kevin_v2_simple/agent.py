"""kevin_v2_simple: built from scratch with claude_prompt_simple.txt (keep-reveal-discard practice).

Layers: L1 mixed keep/reveal | L2 exact ICM (jam/call EV, risk premium) | L3 short-stack push/fold
(chip-EV base + ICM check) | L4 deep 3-bet-or-fold | L5 postflop Monte Carlo with call shading |
L6 opponent model in memory. Tables (tables.npz) are built offline by tmp/kv2s/build_tables.py.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy as np  # noqa: E402

from poker_agent_sdk import Action, Agent as BaseAgent  # noqa: E402

SOFT_DEADLINE_MS = 0.55
EV_MARGIN = 0.15          # tournament points
SHORT_BB = 25
K_REVEAL = 15.0
RANKS = "23456789TJQKA"
SUITS = "cdhs"
CID = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}

# ---------------------------------------------------------------- tables
_T = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.npz"))
CLS52 = _T["cls52"].astype(np.int64)
EQ1 = _T["eq1"].astype(np.float64)
HU = _T["hu"].astype(np.float64)
PRIOR = _T["prior"].astype(np.float64)                     # [13 revealed rank, 169]
CLS_HI = np.zeros(169, dtype=np.int64)
CLS_LO = np.zeros(169, dtype=np.int64)
CLS_SUITED = np.zeros(169, dtype=bool)
_k = 0
for _hi in range(12, -1, -1):
    for _lo in range(_hi, -1, -1):
        if _hi == _lo:
            CLS_HI[_k], CLS_LO[_k] = _hi, _lo
            _k += 1
        else:
            for _s in (True, False):
                CLS_HI[_k], CLS_LO[_k], CLS_SUITED[_k] = _hi, _lo, _s
                _k += 1
ORDER = np.argsort(-EQ1)                                    # strongest first
EQP = HU @ PRIOR.T                                          # [169 my, 13 opp revealed rank]
# bucket of class c for a player who revealed rank r: 0 = hidden lower, 1 = hidden higher, 2 = pair
_r = np.arange(13)[:, None]
BKT = np.where((CLS_HI[None] != _r) & (CLS_LO[None] != _r), 3,
               np.where(CLS_HI[None] == CLS_LO[None], 2, np.where(CLS_HI[None] == _r, 0, 1)))
PB = np.stack([(PRIOR * (BKT == b)).sum(1) for b in range(3)], 1)   # [13, 3]
# hidden-card weights: for revealed card c and hidden h, class prob / combos sharing c
DIV = np.where(CLS_SUITED[CLS52] | (CLS_HI[CLS52] == CLS_LO[CLS52]), 1.0, 1.0)
for _c in range(52):
    for _h in range(52):
        cl = CLS52[_c, _h]
        if CLS_HI[cl] == CLS_LO[cl]:
            DIV[_c, _h] = 3.0
        elif CLS_SUITED[cl]:
            DIV[_c, _h] = 1.0
        else:
            DIV[_c, _h] = 3.0

# ---------------------------------------------------------------- fast batch evaluator
POW5 = 5 ** np.arange(13, dtype=np.int64)
_TOP = np.zeros((6, 8192), dtype=np.int64)
_HIGH = np.full(8192, -1, dtype=np.int64)
_STR = np.full(8192, -1, dtype=np.int64)
for _m in range(8192):
    bs = [x for x in range(12, -1, -1) if _m >> x & 1]
    _HIGH[_m] = bs[0] if bs else -1
    for kk in range(6):
        v = 0
        for x in bs[:kk]:
            v = (v << 4) | x
        _TOP[kk, _m] = v << (4 * (kk - min(kk, len(bs))))
    for hh in range(12, 3, -1):
        w = 0b11111 << (hh - 4)
        if _m & w == w:
            _STR[_m] = hh
            break
    else:
        if _m & 0b1000000001111 == 0b1000000001111:
            _STR[_m] = 3


def _value_from_counts(cnt):
    """Non-flush value for one rank-count vector (list of 13)."""
    B = sum(1 << r for r in range(13) if cnt[r])
    grp = sorted(((cnt[r], r) for r in range(13) if cnt[r]), reverse=True)
    c0, g0 = grp[0]
    c1, g1 = grp[1] if len(grp) > 1 else (0, 0)
    if c0 == 4:
        return (7 << 24) | (g0 << 4) | _HIGH[B & ~(1 << g0)]
    if c0 == 3 and c1 >= 2:
        return (6 << 24) | (g0 << 4) | g1
    if _STR[B] >= 0:
        return (4 << 24) | _STR[B]
    if c0 == 3:
        return (3 << 24) | (g0 << 8) | _TOP[2, B & ~(1 << g0)]
    if c0 == 2 and c1 == 2:
        return (2 << 24) | (g0 << 8) | (g1 << 4) | _HIGH[B & ~(1 << g0) & ~(1 << g1)]
    if c0 == 2:
        return (1 << 24) | (g0 << 12) | _TOP[3, B & ~(1 << g0)]
    return _TOP[5, B]


def _build_nonflush():
    keys, vals = [], []

    def rec(r, left, cnt):
        if r == 13:
            if left == 0:
                keys.append(sum(cnt[i] * 5 ** i for i in range(13)))
                vals.append(_value_from_counts(cnt))
            return
        for c in range(min(4, left) + 1):
            cnt[r] = c
            rec(r + 1, left - c, cnt)
        cnt[r] = 0

    rec(0, 7, [0] * 13)
    keys = np.array(keys, dtype=np.int64)
    vals = np.array(vals, dtype=np.int64)
    o = np.argsort(keys)
    return keys[o], vals[o]


NF_KEYS, NF_VALS = _build_nonflush()
_FLV = np.where(_STR >= 0, (8 << 24) | _STR, (5 << 24) | _TOP[5])  # flush-suit rank mask -> value
_BIT = 1 << np.arange(13, dtype=np.int64)
_AR4 = np.arange(4)


def eval_rows(cards):
    """cards [N,7] int64 -> strength [N]."""
    r = cards >> 2
    s = cards & 3
    v = NF_VALS[np.searchsorted(NF_KEYS, POW5[r].sum(1))]
    sc = (s[:, :, None] == _AR4).sum(1)
    fl = sc.max(1) >= 5
    if fl.any():
        idx = np.nonzero(fl)[0]
        fs = sc[idx].argmax(1)
        fb = ((s[idx] == fs[:, None]) * _BIT[r[idx]]).sum(1)
        v[idx] = np.maximum(v[idx], _FLV[fb])
    return v


# ---------------------------------------------------------------- helpers
def raise_to(state, total):
    r = state.can("raise")
    if not r:
        return call_or_check(state)
    return Action.raise_to(int(max(r.min_amount, min(r.max_amount, int(total)))))


def jam(state):
    r = state.can("raise")
    if r:
        return Action.raise_to(int(r.max_amount))
    return call_or_check(state)


def call_or_check(state):
    if state.can("check"):
        return Action.check()
    if state.can("call"):
        return Action.call()
    return Action.fold()


def check_or_fold(state):
    if state.can("check"):
        return Action.check()
    if state.can("fold"):
        return Action.fold()
    f = state.legal_actions[0]
    return Action(f.kind, f.min_amount)


def auto(state):
    f = state.legal_actions[0]
    return Action(f.kind, f.min_amount)


# ---------------------------------------------------------------- L6 opponent model (memory)
# per seat: [hands, vpip, pfr, jams, calljam, calljam_opp, foldraise, foldraise_opp,
#            post_agg, post_pass, sd_hidden_lower, sd_hidden_higher, sd_pair]
NF = 13


def load_mem(memory):
    try:
        m = json.loads(memory.decode()) if memory else None
        if isinstance(m, dict) and "S" in m:
            return m
    except Exception:
        pass
    return {"S": {}, "st": {}, "rv": {}, "f": {}, "pre": 1, "ja": 0, "ra": 0}


def _seat(m, s):
    k = str(s)
    v = m["S"].get(k)
    if v is None or len(v) != NF:
        v = m["S"][k] = [0] * NF
    return v


def update_mem(m, events, me):
    for e in events:
        k = e.kind
        s = e.seat
        if k == "hand":
            m["rv"] = {}
            m["f"] = {}
            m["pre"] = 1
            m["ja"] = 0
            m["ra"] = 0
        elif k == "stack" and s is not None:
            m["st"][str(s)] = e.amount or 0
        elif k == "reveal" and s is not None and e.cards:
            m["rv"][str(s)] = e.cards[0]
            if s != me:
                _seat(m, s)[0] += 1
        elif k == "board":
            m["pre"] = 0
            m["ja"] = 0
            m["ra"] = 0
        elif s is None or s == me:
            if k in ("raise",) and s == me:
                m["ra"] = 1
            continue
        elif k in ("raise", "call", "fold", "check"):
            v = _seat(m, s)
            fl = m["f"].setdefault(str(s), [0, 0])
            if m["pre"]:
                if k in ("raise", "call") and not fl[0]:
                    v[1] += 1
                    fl[0] = 1
                if k == "raise" and not fl[1]:
                    v[2] += 1
                    fl[1] = 1
                    if (e.amount or 0) >= m["st"].get(str(s), 10 ** 9):
                        v[3] += 1
                        m["ja"] = 1
                if m["ja"] and k in ("call", "fold") and not (k == "raise"):
                    v[5] += 1
                    if k == "call":
                        v[4] += 1
            else:
                if k == "raise":
                    v[8] += 1
                elif k in ("call", "check"):
                    v[9] += 1
            if m["ra"] and k in ("fold", "call", "raise"):
                v[7] += 1
                if k == "fold":
                    v[6] += 1
            if k == "raise":
                m["ra"] = 1
        elif k == "show" and e.cards and len(e.cards) == 2:
            rv = m["rv"].get(str(s))
            if rv and rv in e.cards:
                hid = e.cards[0] if e.cards[1] == rv else e.cards[1]
                a, b = RANKS.index(rv[0]), RANKS.index(hid[0])
                v = _seat(m, s)
                v[10 + (2 if a == b else (0 if b < a else 1))] += 1
    return m


def freqs(m, s):
    v = m["S"].get(str(s)) or [0] * NF
    h = v[0]
    sm = (lambda x, n, p, k: (x + k * p) / (n + k))
    return {
        "vpip": sm(v[1], h, 0.30, 10), "pfr": sm(v[2], h, 0.15, 10), "jam": sm(v[3], h, 0.06, 20),
        "call_jam": sm(v[4], v[5], 0.35, 8), "fold_raise": sm(v[6], v[7], 0.55, 8),
        "agg": sm(v[8], v[8] + v[9], 0.40, 10), "sd": v[10:13],
    }


# ---------------------------------------------------------------- L2 ICM
def icm_hero_top3(R, h, pay):
    """Hero's ICM value for rows R [K,n] (stacks after the hand)."""
    p1, p2, p3 = pay
    K, n = R.shape
    T = R.sum(1)
    sh = R[:, h]
    S = R.copy()
    S[:, h] = 0.0
    Tm = np.maximum(T, 1e-9)
    D1 = np.maximum(T[:, None] - S, 1e-9)
    a = S / (Tm[:, None] * D1)
    v = sh * (p1 / Tm + p2 * a.sum(1))
    if p3:
        D2 = D1[:, :, None] - S[:, None, :]
        B = S[:, None, :] / np.maximum(D2, 1e-9)
        B *= (1.0 - np.eye(n))[None]
        v = v + p3 * sh * (a * B.sum(2)).sum(1)
    return v


# ---- ICM for any payout list. Places 1-3: exact closed form (icm_hero_top3).
# Places 4+: Harville's exponential-race identity (finish order = sort of Exp(1)/stack) on a fixed
# import-time noise matrix; only the (small) tail payouts are sampled, so the error stays small.
_ICM_M = 128
_ICM_E = np.random.default_rng(20261109).exponential(size=(_ICM_M, 16))


def icm_any(R, h, payouts):
    """Hero's expected payout for post-hand stack rows R [K,n]; payouts = full config list."""
    pays = [float(x) for x in payouts]
    p = (pays + [0.0, 0.0, 0.0])[:3]
    v = icm_hero_top3(R, h, (p[0], p[1], p[2]))
    if not any(pays[3:]):
        return v
    K, n = R.shape
    tail = np.zeros(n + 1)
    m = min(n, len(pays))
    tail[3:m] = pays[3:m]
    S = np.where(R > 0, R, 0.0)
    keys = _ICM_E[None, :, :n] / np.maximum(S[:, None, :], 1e-12)
    keys[np.broadcast_to((S <= 0)[:, None, :], keys.shape)] = np.inf
    ahead = (keys < keys[:, :, h:h + 1]).sum(2)
    vt = tail[np.minimum(ahead, n)].mean(1)
    return v + np.where(S[:, h] > 0, vt, 0.0)


class Ctx:
    def __init__(self, state, mem, t0):
        self.t0 = t0
        self.state = state
        self.mem = mem
        self.me = state.seat
        self.n = len(state.seats)
        self.bb = max(1, state.blinds[1])
        self.stack = [x.stack for x in state.seats]
        self.bet = [x.bet for x in state.seats]
        self.alive = [x.seat for x in state.seats if x.status != "busted"]
        self.live = [x.seat for x in state.seats if x.status in ("active", "all_in")]
        btn = state.button
        order = sorted(self.alive, key=lambda i: (i - btn) % self.n)
        self.pos = {x: k for k, x in enumerate(order)}
        self.act_order = order[:] if len(order) == 2 else order[3:] + order[:3]
        mine = self.stack[self.me] + self.bet[self.me]
        others = [self.stack[i] + self.bet[i] for i in self.live if i != self.me] or [mine]
        self.eff_bb = min(mine, max(others)) / self.bb
        pay = list(state.config.payouts) + [0, 0, 0]
        self.pay = (float(pay[0]), float(pay[1]), float(pay[2]))
        self.payouts_all = tuple(float(x) for x in state.config.payouts)
        self.payv = np.array([float(x) for x in list(state.config.payouts)[:self.n]] + [0.0] * (self.n + 1))
        v = state.variant or {}
        self.revealed = dict(v.get("revealed") or {})
        self.cache = {}

    def late(self):
        return (perf_counter() - self.t0) * 1000.0 > SOFT_DEADLINE_MS

    def values(self, R, start):
        v = icm_any(R, self.me, self.payouts_all)
        bust = R[:, self.me] <= 0
        if bust.any():
            na = (R > 0).sum(1) + ((R <= 0) & (start[None, :] > start[self.me])).sum(1)
            v = np.where(bust, self.payv[np.minimum(na, self.n)], v)
        return v


def jam_ev(ctx, pot, callers, fold_winner=None):
    me = ctx.me
    s = np.array(ctx.stack, dtype=np.float64)
    c = np.array(ctx.bet, dtype=np.float64)
    start = s + c
    k = len(callers)
    R = np.tile(s, (2 + 2 * k, 1))
    if fold_winner is None:
        cc = c.copy()
        cc[me] = -1
        fold_winner = int(np.argmax(cc))
    R[0, fold_winner] += pot
    R[1, me] += pot
    Ah = start[me]
    reach, probs = 1.0, []
    for i, (j, p, eq) in enumerate(callers):
        Aj = start[j]
        M = min(Ah, Aj)
        dead = pot - c[me] - c[j]
        R[2 + 2 * i, me] = Ah + M + dead
        R[2 + 2 * i, j] = Aj - M
        R[3 + 2 * i, me] = Ah - M
        R[3 + 2 * i, j] = Aj + M + dead
        probs.append(reach * p)
        reach *= 1.0 - p
    V = ctx.values(R, start)
    ev = reach * V[1]
    for i, (j, p, eq) in enumerate(callers):
        ev += probs[i] * (eq * V[2 + 2 * i] + (1 - eq) * V[3 + 2 * i])
    return float(ev), float(V[0])


def call_ev(ctx, pot, jammer, eq):
    me = ctx.me
    s = np.array(ctx.stack, dtype=np.float64)
    c = np.array(ctx.bet, dtype=np.float64)
    start = s + c
    Ah, Aj = start[me], start[jammer]
    M = min(Ah, Aj)
    dead = pot - c[me] - c[jammer]
    R = np.tile(s, (3, 1))
    R[0, me], R[0, jammer] = Ah + M + dead, Aj - M
    R[1, me], R[1, jammer] = Ah - M, Aj + M + dead
    R[2, jammer] += pot
    V = ctx.values(R, start)
    return float(eq * V[0] + (1 - eq) * V[1]), float(V[2])


def risk_premium(ctx, villain, amount, dead=0.0):
    me = ctx.me
    s = np.array(ctx.stack, dtype=np.float64)
    amt = min(float(amount), s[me], s[villain])
    if amt <= 0:
        return 0.0
    R = np.tile(s, (3, 1))
    R[0, me] += amt + dead
    R[0, villain] -= amt
    R[1, me] -= amt
    R[1, villain] += amt + dead
    V = ctx.values(R, s)
    d = V[0] - V[1]
    return float((V[2] - V[1]) / d - 0.5) if d > 1e-12 else 0.5


# ---------------------------------------------------------------- ranges
def rank_of(card):
    return RANKS.index(card[0])


def has_sd(ctx, s):
    v = ctx.mem["S"].get(str(s))
    return bool(v) and (v[10] + v[11] + v[12]) > 0


def opp_post(ctx, s):
    """Class distribution of seat s given its revealed card (+ its showdown reveal habits)."""
    key = ("p", s)
    if key in ctx.cache:
        return ctx.cache[key]
    card = ctx.revealed.get(s)
    if card is None:
        out = (np.full(169, 1.0 / 169), None)
    else:
        r = rank_of(card)
        P = PRIOR[r]
        if has_sd(ctx, s):
            n = ctx.mem["S"][str(s)][10:13]
            pb = PB[r].tolist()
            tot = K_REVEAL + n[0] + n[1] + n[2]
            mult = [((K_REVEAL * pb[b] + n[b]) / tot) / pb[b] if pb[b] > 0 else 0.0 for b in range(3)] + [0.0]
            P = P * np.array(mult)[BKT[r]]
            t = P.sum()
            P = P / t if t > 0 else PRIOR[r]
        out = (P, r)
    ctx.cache[key] = out
    return out


def range_top(post, frac):
    frac = min(1.0, max(0.02, frac))
    p = post[ORDER]
    cum = np.cumsum(p)
    keep = np.clip((frac - (cum - p)) / np.maximum(p, 1e-12), 0.0, 1.0)
    out = np.zeros(169)
    out[ORDER] = p * keep
    t = out.sum()
    return out / t if t > 0 else post


def eq_w(my, w):
    t = w.sum()
    return float(HU[my] @ w / t) if t > 0 else 0.5


def eq_opp(ctx, my, s):
    card = ctx.revealed.get(s)
    if card is not None and not has_sd(ctx, s):
        return float(EQP[my, rank_of(card)])
    return eq_w(my, opp_post(ctx, s)[0])


def my_class(state):
    a, b = state.cards
    return int(CLS52[CID[a], CID[b]])


def preflop_history(state):
    raisers, callers = [], []
    for e in state.history:
        if e.kind == "board":
            break
        if e.kind == "raise" and e.seat is not None:
            raisers.append(e.seat)
        elif e.kind == "call" and e.seat is not None:
            callers.append(e.seat)
    return raisers, callers


def behind(ctx, seat):
    o = ctx.act_order
    if seat not in o:
        return []
    k = o.index(seat)
    return [x for x in o[k + 1:] if x in ctx.live and ctx.stack[x] > 0]


# ---------------------------------------------------------------- L1 choose
PAIRS = ((0, 1), (0, 2), (1, 2))


def choose(state):
    c = state.cards
    ids = [CID[x] for x in c]
    best = max(PAIRS, key=lambda p: EQ1[CLS52[ids[p[0]], ids[p[1]]]])
    a, b = best
    # mixed reveal: never let the shown card decode the hidden one
    if random.random() < 0.5:
        a, b = b, a
    return Action.choose(keep=a, reveal=b)


# ---------------------------------------------------------------- L3 short stack
def call_frac(jam_bb, f):
    base = min(0.45, max(0.12, 0.10 + 1.5 / max(jam_bb, 1.0)))
    return base * min(2.0, max(0.5, f["call_jam"] / 0.35))


def short_unopened(state, ctx, my):
    bh = behind(ctx, ctx.me)
    pot = state.pot
    mine = ctx.stack[ctx.me] + ctx.bet[ctx.me]
    callers, chip = [], 0.0
    reach = 1.0
    for s in bh:
        if ctx.late():
            return EQ1[my] >= 0.62
        post, _ = opp_post(ctx, s)
        risk = min(mine, ctx.stack[s] + ctx.bet[s])
        frac = call_frac(risk / ctx.bb, freqs(ctx.mem, s))
        eq = eq_w(my, range_top(post, frac))
        callers.append((s, frac, eq))
        chip += reach * frac * (eq * (pot + risk) - (1 - eq) * risk + 0.0)
        reach *= 1 - frac
    if not callers:
        return False
    chip += reach * pot
    base = chip > 0          # chip-EV push/fold base
    ev_j, ev_f = jam_ev(ctx, pot, callers)
    gap = ev_j - ev_f
    if gap > EV_MARGIN:
        return True
    if gap < -EV_MARGIN:
        return False
    return base


def jammer_range(ctx, s, jam_bb, raised_only=False):
    post, _ = opp_post(ctx, s)
    f = freqs(ctx.mem, s)
    if raised_only:
        frac = max(0.08, f["pfr"] / max(f["vpip"], 0.05))
    else:
        frac = min(0.6, max(0.08, 0.08 + 2.5 / max(jam_bb, 1.0)))
        frac *= min(3.0, max(0.5, f["jam"] / 0.06))
    return range_top(post, min(1.0, frac))


def facing_jam(state, ctx, my, jammer, extra_callers):
    risk = min(ctx.stack[ctx.me] + ctx.bet[ctx.me], ctx.stack[jammer] + ctx.bet[jammer])
    eq = eq_w(my, jammer_range(ctx, jammer, risk / ctx.bb))
    if extra_callers:
        eq *= 0.8 ** extra_callers
    need = state.to_call / max(1, state.pot + state.to_call)
    base = eq >= need + (0.05 if extra_callers else 0.0)
    if ctx.late():
        return call_or_check(state) if base else check_or_fold(state)
    ev_c, ev_f = call_ev(ctx, state.pot, jammer, eq)
    gap = ev_c - ev_f
    go = True if gap > EV_MARGIN else (False if gap < -EV_MARGIN else base)
    return call_or_check(state) if go else check_or_fold(state)


def short_vs_raise(state, ctx, my, raiser):
    w = jammer_range(ctx, raiser, 1, raised_only=True)
    committed = ctx.bet[raiser] / max(1, ctx.stack[raiser] + ctx.bet[raiser])
    callers = [(raiser, min(0.95, 0.35 + committed), eq_w(my, w))]
    mine = ctx.stack[ctx.me] + ctx.bet[ctx.me]
    for s in behind(ctx, ctx.me):
        if s == raiser:
            continue
        risk = min(mine, ctx.stack[s] + ctx.bet[s])
        frac = call_frac(risk / ctx.bb, freqs(ctx.mem, s))
        callers.append((s, frac * 0.8, eq_w(my, range_top(opp_post(ctx, s)[0], frac))))
    if ctx.late():
        return check_or_fold(state)
    ev_j, ev_f = jam_ev(ctx, state.pot, callers, fold_winner=raiser)
    return jam(state) if ev_j > ev_f else check_or_fold(state)


# ---------------------------------------------------------------- L4 deep preflop
def gmean(xs):
    return float(np.exp(np.mean(np.log(np.maximum(xs, 1e-3))))) if xs else 0.5


def deep_preflop(state, ctx, my, raisers):
    bb = ctx.bb
    me = ctx.me
    to_call = state.to_call
    bh = behind(ctx, me)
    opps = [s for s in ctx.live if s != me]
    if not raisers:
        if not bh and to_call == 0:  # BB option
            g = gmean([eq_opp(ctx, my, s) for s in opps])
            return raise_to(state, 4 * bb) if g > 0.62 else Action.check()
        g = gmean([eq_opp(ctx, my, s) for s in bh])
        thr = 0.50 + 0.012 * max(0, len(bh) - 1)
        if ctx.pos.get(me) == 1 and len(bh) == 1:
            thr = 0.48
        limpers = sum(1 for s in opps if ctx.bet[s] >= bb and s not in bh and ctx.pos.get(s) != 2)
        if g >= thr:
            return raise_to(state, (2.5 + limpers) * bb)
        return Action.check() if state.can("check") else check_or_fold(state)
    last = raisers[-1]
    post, _ = opp_post(ctx, last)
    f = freqs(ctx.mem, last)
    nr = len(raisers)
    frac = max(0.04, f["pfr"] / max(f["vpip"], 0.05)) * (0.5 ** (nr - 1))
    eq = eq_w(my, range_top(post, min(1.0, frac * 1.5)))
    stack_total = ctx.stack[me] + ctx.bet[me]
    if ctx.late():
        return check_or_fold(state)
    rp = risk_premium(ctx, last, to_call + ctx.bet[me], dead=state.pot - ctx.bet[last] - ctx.bet[me])
    rp = min(0.15, max(-0.05, rp))
    pot_odds = to_call / max(1, state.pot + to_call)
    if eq >= 0.58 + rp:
        if nr >= 2 or 3 * ctx.bet[last] > 0.35 * stack_total:
            return jam(state)
        return raise_to(state, 3 * ctx.bet[last])
    is_bb = ctx.pos.get(me) == 2 or (len(ctx.alive) == 2 and ctx.pos.get(me) == 1)
    if is_bb and nr == 1 and pot_odds <= 0.30 and eq >= 0.50 + rp:
        return call_or_check(state)
    if nr == 1 and to_call <= 0.06 * stack_total and eq >= 0.55 + rp:
        return call_or_check(state)
    return check_or_fold(state)


def preflop(state, ctx):
    my = my_class(state)
    raisers, callers = preflop_history(state)
    me = ctx.me
    if ctx.eff_bb <= SHORT_BB:
        if not raisers:
            if state.to_call == 0 and not behind(ctx, me):
                return Action.check()
            return jam(state) if short_unopened(state, ctx, my) else check_or_fold(state)
        last = raisers[-1]
        if ctx.stack[last] == 0 or state.to_call >= ctx.stack[me]:
            extra = len(set(c for c in callers if c != me))
            return facing_jam(state, ctx, my, last, extra)
        return short_vs_raise(state, ctx, my, last)
    return deep_preflop(state, ctx, my, raisers)


# ---------------------------------------------------------------- L5 postflop
def hidden_weights(ctx, s, dead):
    card = ctx.revealed.get(s)
    if card is None:
        w = np.ones(52)
    else:
        c = CID[card]
        post, _ = opp_post(ctx, s)
        w = post[CLS52[c]] / DIV[c]
        w[c] = 0.0
    w[dead] = 0.0
    t = w.sum()
    if t <= 0:
        w = np.ones(52)
        w[dead] = 0.0
        t = w.sum()
    return w / t


def equity(ctx, my_ids, board_ids, opps, n):
    """Monte Carlo share of the pot vs opps (hidden card ~ posterior; revealed card known)."""
    dead = list(my_ids) + list(board_ids) + [CID[ctx.revealed[s]] for s in opps if s in ctx.revealed]
    k = len(opps)
    H = np.empty((n, k), dtype=np.int64)
    for i, s in enumerate(opps):
        cw = np.cumsum(hidden_weights(ctx, s, dead))
        H[:, i] = np.minimum(np.searchsorted(cw, np.random.random(n) * cw[-1]), 51)
    need = 5 - len(board_ids)
    keys = np.random.random((n, 52))
    keys[:, dead] = 2.0
    keys[np.arange(n)[:, None], H] = 2.0
    if need > 0:
        run = np.argpartition(keys, need, axis=1)[:, :need]
    else:
        run = np.empty((n, 0), dtype=np.int64)
    ok = np.ones(n, dtype=bool)
    if k > 1:
        Hs = np.sort(H, 1)
        ok &= (Hs[:, 1:] != Hs[:, :-1]).all(1)
    board = np.concatenate([np.tile(np.array(board_ids, dtype=np.int64), (n, 1)), run], 1)
    rows = [np.concatenate([np.tile(np.array(my_ids, dtype=np.int64), (n, 1)), board], 1)]
    for i, s in enumerate(opps):
        rc = CID[ctx.revealed[s]] if s in ctx.revealed else H[:, i]
        rcol = np.full((n, 1), rc, dtype=np.int64) if s in ctx.revealed else H[:, i:i + 1]
        rows.append(np.concatenate([rcol, H[:, i:i + 1], board], 1))
    V = eval_rows(np.concatenate(rows, 0)).reshape(k + 1, n)
    best = V.max(0)
    mine = V[0] == best
    share = mine / (V == best).sum(0)
    share = share[ok]
    return float(share.mean()) if len(share) else 0.5


def postflop(state, ctx):
    raisers, _ = preflop_history(state)
    opps = [s for s in ctx.live if s != ctx.me]
    me = ctx.me
    stack = ctx.stack[me]
    mybet = ctx.bet[me]
    allin_total = stack + mybet
    pot = max(1, state.pot)
    to_call = state.to_call
    rp = 0.0
    stake = max(to_call, pot)
    if opps and stake * 4 >= stack and not ctx.late():
        v = max(opps, key=lambda s: ctx.stack[s])
        rp = min(0.12, max(0.0, risk_premium(ctx, v, min(stack, ctx.stack[v]), dead=pot)))
    used = (perf_counter() - ctx.t0) * 1000.0
    n = 30 if len(opps) <= 1 else (24 if len(opps) == 2 else 16)
    if used > 0.35:
        n = 10
    eq = equity(ctx, [CID[c] for c in state.cards], [CID[c] for c in state.board], opps, n)
    nop = max(1, len(opps))
    river = len(state.board) >= 5
    if to_call > 0:
        bettors = [s for s in opps if ctx.bet[s] > mybet]
        agg = min([freqs(ctx.mem, s)["agg"] for s in bettors] or [0.4])
        size = min(1.5, to_call / max(1, pot - to_call))
        f = 1.0 - (1.0 - agg) * 0.30 * (0.5 + size)
        eq_adj = eq * min(1.05, max(0.45, f))
        need = to_call / (pot + to_call) + rp + (0.06 if to_call >= 0.3 * allin_total else 0.0)
        spr = (stack - to_call) / (pot + to_call)
        if state.can("raise") and eq_adj >= 0.72 + 0.06 * (nop - 1) + rp:
            if spr <= 1.2:
                return jam(state)
            total = mybet + to_call + int(0.8 * (pot + to_call))
            return jam(state) if total >= 0.45 * allin_total else raise_to(state, total)
        if eq_adj >= need:
            if state.can("raise") and spr <= 0.6 and eq_adj >= 0.5 + rp:
                return jam(state)
            return call_or_check(state)
        return check_or_fold(state)
    value_thr = max(0.42, 0.62 - 0.1 * (nop - 1)) + 0.5 * rp
    spr = stack / pot
    if eq >= value_thr:
        if spr <= 1.5 and eq >= 0.55:
            return jam(state)
        frac = 0.75 if (river or eq >= 0.8) else 0.62
        total = mybet + int(frac * pot)
        return jam(state) if total >= 0.5 * allin_total else raise_to(state, total)
    if nop == 1 and rp < 0.08:
        ftr = freqs(ctx.mem, opps[0])["fold_raise"]
        pr = min(0.35 if river else 0.40, 0.22 * ftr / 0.55)
        if ((river and eq < 0.15) or (not river and eq < 0.35)) and random.random() < pr:
            return raise_to(state, mybet + int((0.75 if river else 0.62) * pot))
    return check_or_fold(state)


# ---------------------------------------------------------------- agent
class Agent(BaseAgent):
    def act(self, state, memory):
        t0 = perf_counter()
        mem = load_mem(memory)
        try:
            mem = update_mem(mem, state.events, state.seat)
        except Exception:
            mem = load_mem(b"")
        try:
            action = self.decide(state, mem, t0)
        except Exception:
            action = None
        if action is None:
            try:
                action = check_or_fold(state) if state.phase == "betting" else auto(state)
            except Exception:
                action = Action("fold")
        try:
            out = json.dumps(mem, separators=(",", ":")).encode()
            if len(out) > 16000:
                out = b""
        except Exception:
            out = b""
        return action, out

    def decide(self, state, mem, t0):
        if state.phase == "choose":
            try:
                return choose(state)
            except Exception:
                return auto(state)
        if state.phase != "betting":
            return auto(state)
        if len(state.cards) != 2:
            return check_or_fold(state)
        ctx = Ctx(state, mem, t0)
        if not state.board:
            return preflop(state, ctx)
        return postflop(state, ctx)


# warm up numpy paths, then freeze import-time objects out of the GC's view
eval_rows(np.array([[0, 5, 10, 15, 20, 25, 30]], dtype=np.int64))
icm_any(np.array([[100.0, 200.0, 300.0]]), 0, (50.0, 30.0, 20.0))
icm_any(np.array([[100.0, 200.0, 300.0, 50.0]]), 0, (4.0, 3.0, 2.0, 1.0))
gc.collect()
gc.freeze()
