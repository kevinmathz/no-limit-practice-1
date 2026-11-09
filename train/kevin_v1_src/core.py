"""KRD bot core: choose policy (CFR), short-stack push/fold (CFR chart + ICM/hand-cap EV check),
deep preflop heuristic on reveal-conditioned ranges, postflop via postflop.py.

Amalgamated into krd_cfr_bot/agent.py by build_agent.py (modules rt_eval, oppmodel, tvalue,
postflop are embedded). Never import poker_engine.
"""
import os
import random
from time import perf_counter

import numpy as np

from poker_agent_sdk import Action, Agent as BaseAgent
import oppmodel as OM
import tvalue as TV
import postflop as PF

CAP_PAYOUT = "icm"        # "icm" (RULES.md) or "rank" (chip-rank places at the cap)
SHORT_BB = 25             # push/fold mode at or below this effective stack
THREEBET_EQ = 0.58
SOFT_DEADLINE_MS = 0.55  # past this, skip EV refinements and act cheaply/passively
EV_MARGIN = 0.15          # tournament points: below this |EV gap| follow the CFR chart mix
RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)
CID = {c: i for i, c in enumerate(DECK)}

# ---------------------------------------------------------------- tables (import time)
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_T = np.load(os.path.join(_DATA_DIR, "krd_tables.npz"))
PL_GRID = [int(x) for x in _T["players_grid"]]
ST_GRID = [int(x) for x in _T["stack_grid"]]
CHOOSE = _T["choose"]                 # [P,S,1755,6] uint8
PF_OPEN = _T["pf_open"]               # [P,S,8,169,2,13]
PF_CALL = _T["pf_call"]
POSTERIOR = _T["posterior"]           # [P,S,13,169]
HU_EQ = _T["hu_eq"].astype(np.float64)
EQ_RAND = _T["eq_rand"].astype(np.float64)
# direct 52^3 index (key = c0*2704 + c1*52 + c2, ids sorted descending) -> class, canonical order
COMBO_CLS = np.full(52 ** 3, -1, dtype=np.int16)
COMBO_ORD = np.zeros((52 ** 3, 3), dtype=np.int8)
COMBO_CLS[_T["combo_key"]] = _T["combo_cls"]
COMBO_ORD[_T["combo_key"]] = _T["combo_order"]
CHOOSE_ACTIONS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))

CLASS_LIST = []
for _i in range(12, -1, -1):
    for _j in range(_i, -1, -1):
        if _i == _j:
            CLASS_LIST.append(RANKS[_i] * 2)
        else:
            CLASS_LIST.append(RANKS[_i] + RANKS[_j] + "s")
            CLASS_LIST.append(RANKS[_i] + RANKS[_j] + "o")
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_LIST)}
CLS_HI = np.array([RANKS.index(c[0]) for c in CLASS_LIST])
CLS_LO = np.array([RANKS.index(c[1]) for c in CLASS_LIST])
# SIL[r, c] = 1 if a player holding class c who revealed rank r showed the LOWER card
SIL = np.array([[1 if (CLS_HI[c] != r) else 0 for c in range(169)] for r in range(13)], dtype=np.int64)
STRENGTH = EQ_RAND[:, 0]              # HU equity vs a random hand: class strength order
_ORDER = np.argsort(-STRENGTH)


def class2(a, b):
    ra, rb = RANKS.index(a[0]), RANKS.index(b[0])
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return CLASS_INDEX[RANKS[hi] * 2]
    return CLASS_INDEX[RANKS[hi] + RANKS[lo] + ("s" if a[1] == b[1] else "o")]


def rank(card):
    return RANKS.index(card[0])


def grid_idx(grid, v):
    return min(range(len(grid)), key=lambda i: abs(grid[i] - v))


# ---------------------------------------------------------------- safe action helpers
def raise_to(state, total):
    r = state.can("raise")
    if not r:
        return call_or_check(state)
    total = int(max(r.min_amount, min(r.max_amount, int(total))))
    return Action.raise_to(total)


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
    first = state.legal_actions[0]
    return Action(first.kind, first.min_amount)


# ---------------------------------------------------------------- table context
class Ctx:
    """Everything derived from the state once per decision."""

    def __init__(self, state, t0=None):
        self.t0 = perf_counter() if t0 is None else t0
        self.state = state
        self.me = state.seat
        self.n = len(state.seats)
        self.bb = max(1, state.blinds[1])
        self.stack = [s.stack for s in state.seats]
        self.bet = [s.bet for s in state.seats]
        st = {s.seat: s.status for s in state.seats}
        self.status = st
        self.alive = [s.seat for s in state.seats if s.status != "busted"]
        self.live = [s.seat for s in state.seats if s.status in ("active", "all_in")]
        self.n_players = len(self.alive)
        # positions from the button among alive seats (dealt this hand)
        order = sorted(self.alive, key=lambda i: (i - state.button) % self.n)
        if state.button in self.alive:
            order = [state.button] + [i for i in order if i != state.button]
        self.pos = {seat: k for k, seat in enumerate(order)}
        # preflop action order: HU button first; else UTG (pos 3, or button if 3-handed) ...
        if len(order) == 2:
            self.act_order = order[:]
        else:
            self.act_order = order[3:] + order[:3]
        mine = self.stack[self.me] + self.bet[self.me]
        others = [self.stack[i] + self.bet[i] for i in self.live if i != self.me] or [mine]
        self.eff_bb = min(mine, max(others)) / self.bb
        cfg = state.config
        self.pi = grid_idx(PL_GRID, self.n_players)
        self.si = grid_idx(ST_GRID, min(self.eff_bb, 40))
        self.hands_left = max(0, cfg.hand_cap - state.hand_number)
        self.payouts = tuple(float(p) for p in cfg.payouts)
        v = state.variant or {}
        self.revealed = dict(v.get("revealed") or {})
        self.cache = {}

    def late(self):
        return (perf_counter() - self.t0) * 1000.0 > SOFT_DEADLINE_MS

    def value_kwargs(self):
        cfg = self.state.config
        try:
            sched = TV.sched_from(self.state.hand_number, cfg.blinds, cfg.hands_per_level, cfg.hand_cap)
        except Exception:
            sched = None
        return dict(sched=sched, mode=CAP_PAYOUT, bb=float(self.bb), payouts=self.payouts)


def chart_pos(ctx, seat):
    return min(ctx.pos.get(seat, 0), 7)


def prior_row(ctx, rnk):
    return POSTERIOR[ctx.pi, ctx.si, rnk]


# equilibrium prior rows, restricted to classes holding the revealed rank, floored + normalised
_PRIOR = POSTERIOR.astype(np.float64) * OM.CLASS_HAS_RANK[None, None, :, :]
_PRIOR /= np.maximum(_PRIOR.sum(-1, keepdims=True), 1e-12)
_PRIOR = 0.99 * _PRIOR + 0.01 * OM._UNIF[None, None, :, :]
_TILT_X = (OM.CLASS_PCT - 50.0) / 50.0
# BUCKET169[r, c]: 0 = hidden lower than revealed rank r, 1 = hidden higher, 2 = pair, 3 = n/a
_other = np.where(CLS_HI == np.arange(13)[:, None], CLS_LO[None, :], CLS_HI[None, :])
_has = OM.CLASS_HAS_RANK
BUCKET169 = np.where(~_has, 3, np.where((CLS_HI == CLS_LO)[None, :], 2,
                                          np.where(_other < np.arange(13)[:, None], 0, 1)))
_GROUP_OF = np.array(OM._GROUP)
_K_REVEAL = 15.0


# W39[rv*13+h, rv*3+b]: showdown matrix cell -> (revealed rank, bucket) count
W39 = np.zeros((169, 39))
for _rv in range(13):
    for _h in range(13):
        W39[_rv * 13 + _h, _rv * 3 + (0 if _h < _rv else (1 if _h > _rv else 2))] = 1.0
GMAT = np.array([[1.0 if _GROUP_OF[j] == g else 0.0 for j in range(13)] for g in range(5)])
# prior mass per bucket: PB[p, s, r, b]
PB = np.stack([(np.where(BUCKET169[None, None] == b, _PRIOR, 0.0)).sum(-1) for b in range(3)], -1)


# EQP[p, s, my_cls, r]: equity vs an opponent with revealed rank r holding the equilibrium prior
EQP = np.einsum("mc,psrc->psmr", HU_EQ, _PRIOR)


def eq_opp(ctx, mem, cls, seat):
    """Equity of class `cls` vs `seat` (no action tilt); table lookup when it has no showdowns."""
    card = ctx.revealed.get(seat)
    if card is not None and not (0 <= seat < mem.ns and mem.c[seat * OM.NC + OM.C_SD] > 0):
        return float(EQP[ctx.pi, ctx.si, cls, rank(card)])
    return eq_vs(cls, opp_post(ctx, mem, seat)[0])


def seat_rows3(ctx, mem):
    """[ns, 13, 3] showdown counts (hidden lower / higher / pair) per revealed rank, all seats."""
    r3 = ctx.cache.get("rows3")
    if r3 is None:
        ns = mem.ns
        C = np.frombuffer(mem.c, dtype=np.uint16, count=ns * OM.NC)
        M = C.reshape(ns, OM.NC)[:, OM.C_MAT:OM.C_MAT + 169].astype(np.float64)
        r3 = ctx.cache["rows3"] = (M @ W39).reshape(ns, 13, 3)
    return r3


def fast_reveal_post(ctx, mem, seat, r, P):
    """Bucket-level Dirichlet blend of prior P (classes with revealed rank r) with the seat's
    showdown reveal habits (shown-high / shown-low / pair), pooled over similar ranks."""
    R3 = seat_rows3(ctx, mem)[seat]
    n_a = R3.sum(0).tolist()
    n_g = (GMAT[_GROUP_OF[r]] @ R3).tolist()
    pb = PB[ctx.pi, ctx.si, r].tolist()
    k = _K_REVEAL
    ne = [(n_g[b] + 0.3 * (n_a[b] - n_g[b])) if pb[b] > 0 else 0.0 for b in range(3)]
    tot = k + ne[0] + ne[1] + ne[2]
    mult = [((k * pb[b] + ne[b]) / tot) / pb[b] if pb[b] > 0 else 0.0 for b in range(3)] + [0.0]
    out = P * np.array(mult)[BUCKET169[r]]
    t = out.sum()
    return out / t if t > 0 else P


def freqs(ctx, mem, seat):
    f = ctx.cache.get(("f", seat))
    if f is None:
        f = ctx.cache[("f", seat)] = OM.action_freqs(mem, seat)
    return f


def opp_post(ctx, mem, seat, action=0):
    """Posterior over the 169 kept classes of `seat` given its revealed card (cached per decision)."""
    key = ("p", seat, action)
    hit = ctx.cache.get(key)
    if hit is not None:
        return hit
    card = ctx.revealed.get(seat)
    if card is None:
        out = (np.ones(169) / 169, None)
    else:
        r = rank(card)
        if 0 <= seat < mem.ns and mem.c[seat * OM.NC + OM.C_SD] > 0:
            p = fast_reveal_post(ctx, mem, seat, r, _PRIOR[ctx.pi, ctx.si, r])
        else:  # no showdown evidence yet: equilibrium prior
            p = _PRIOR[ctx.pi, ctx.si, r]
        if True:
            if action:
                fr = freqs(ctx, mem, seat)
                if action >= 2:
                    beta = min(1.5, max(0.0, np.log(0.10 / fr["jam"]) * 0.6))
                else:
                    beta = min(1.0, max(0.0, np.log(0.20 / fr["pfr"]) * 0.5))
                if beta > 0:
                    p = p * np.exp(beta * _TILT_X)
                    p = p / p.sum()
        out = (p, r)
    ctx.cache[key] = out
    return out


# ---------------------------------------------------------------- fast tournament EV

# ---- ICM for any payout list. Places 1-3: exact closed form (TV._icm_hero).
# Places 4+: Harville's exponential-race identity (finish order = sort of Exp(1)/stack) on a fixed
# import-time noise matrix; only the (small) tail payouts are sampled, so the error stays small.
_ICM_M = 128
_ICM_E = np.random.default_rng(20261109).exponential(size=(_ICM_M, 16))


def icm_any(R, h, payouts):
    """Hero's expected payout for post-hand stack rows R [K,n]; payouts = full config list."""
    pays = [float(x) for x in payouts]
    p = (pays + [0.0, 0.0, 0.0])[:3]
    v = TV._icm_hero(R, h, (p[0], p[1], p[2]))
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


def values(ctx, R, start=None):
    """Hero's tournament points for each post-hand stack row of R [K,n]."""
    if CAP_PAYOUT == "icm":  # cap settles by ICM too, so the same valuation applies at 0 hands left
        me = ctx.me
        v = icm_any(R, me, ctx.payouts)
        bust = R[:, me] <= 0
        if bust.any():
            pay = np.zeros(R.shape[1] + 1)
            m = min(R.shape[1], len(ctx.payouts))
            pay[:m] = ctx.payouts[:m]
            n_alive = (R > 0).sum(1)
            if start is not None:
                n_alive = n_alive + ((R == 0) & (start[None, :] > start[me])).sum(1)
            v = np.where(bust, pay[np.minimum(n_alive, R.shape[1])], v)
        return v
    return TV.value_rows(R, ctx.me, ctx.hands_left, start=start, **ctx.value_kwargs())


def jam_ev(ctx, pot, callers, fold_winner=None):
    """(EV jam, EV fold); callers = [(seat, p_call, eq)] in order, at most one calls."""
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
    probs = []
    reach = 1.0
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
    V = values(ctx, R, start)
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
    R[0, me] = Ah + M + dead
    R[0, jammer] = Aj - M
    R[1, me] = Ah - M
    R[1, jammer] = Aj + M + dead
    R[2, jammer] += pot
    V = values(ctx, R, start)
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
    V = values(ctx, R, s)
    d = V[0] - V[1]
    return float((V[2] - V[1]) / d - 0.5) if d > 1e-12 else 0.5


def range_top(post, frac):
    """Keep the strongest `frac` of the posterior mass (soft edge)."""
    frac = min(1.0, max(0.02, frac))
    p = post[_ORDER]
    cum = np.cumsum(p)
    keep = np.clip((frac - (cum - p)) / np.maximum(p, 1e-12), 0.0, 1.0)
    out = np.zeros(169)
    out[_ORDER] = p * keep
    s = out.sum()
    return out / s if s > 0 else post


def eq_vs(my_cls, w):
    s = w.sum()
    if s <= 0:
        return 0.5
    return float(HU_EQ[my_cls] @ w / s)


# ---------------------------------------------------------------- choose phase
def choose_action(state):
    ctx = Ctx(state)
    cards = list(state.cards)
    ids = [CID[c] for c in cards]
    srt = sorted(range(3), key=lambda k: -ids[k])
    key = ids[srt[0]] * 2704 + ids[srt[1]] * 52 + ids[srt[2]]
    cls = int(COMBO_CLS[key])
    order = COMBO_ORD[key].tolist()
    probs = CHOOSE[ctx.pi, ctx.si, cls].astype(np.float64)
    tot = probs.sum()
    if tot <= 0:
        a = 0
    else:
        x = random.random() * tot
        a = 5
        for k in range(6):
            x -= probs[k]
            if x < 0:
                a = k
                break
    kp, rp = CHOOSE_ACTIONS[a]
    keep = cards[srt[order[kp]]]
    reveal = cards[srt[order[rp]]]
    return Action.choose(keep, reveal)


# ---------------------------------------------------------------- preflop
def preflop_actions(state):
    """(raisers, callers, jammers) seats from this hand's preflop history."""
    raisers, callers = [], []
    for e in state.history:
        if e.kind == "board":
            break
        if e.kind == "raise" and e.seat is not None:
            raisers.append(e.seat)
        elif e.kind == "call" and e.seat is not None:
            callers.append(e.seat)
    return raisers, callers


def my_hand_info(state):
    v = state.variant or {}
    my_rev = v.get("my_reveal")
    c = state.cards
    if my_rev is None or my_rev not in c:
        my_rev = c[1]
    hid = c[0] if c[1] == my_rev else c[1]
    cls = class2(c[0], c[1])
    sil = 1 if rank(my_rev) < rank(hid) else 0
    return cls, sil, rank(my_rev)


def behind_seats(ctx, seat):
    """Seats still to act after `seat` in the preflop order (live, not folded)."""
    order = ctx.act_order
    if seat not in order:
        return []
    k = order.index(seat)
    return [s for s in order[k + 1:] if s in ctx.live and ctx.stack[s] > 0]


def p_call_and_range(ctx, mem, seat, my_rev_rank):
    """Prob `seat` calls our jam and the class weights of its calling range."""
    post, r = opp_post(ctx, mem, seat)
    if r is None:
        return 0.3, post
    pc = PF_CALL[ctx.pi, ctx.si, chart_pos(ctx, seat), np.arange(169), SIL[r], my_rev_rank] / 255.0
    w = post * pc
    p = float(w.sum())
    # bounded adjustment by observed call-vs-jam frequency (prior 0.35 in oppmodel)
    fr = freqs(ctx, mem, seat)
    ratio = min(2.0, max(0.5, fr["call_jam"] / 0.35))
    p = min(0.98, p * ratio)
    return p, w


def short_unopened(state, ctx, mem, cls, sil, my_rr):
    behind = behind_seats(ctx, ctx.me)
    revs = [rank(ctx.revealed[s]) for s in behind if s in ctx.revealed]
    mb = max(revs) if revs else 0
    p_chart = PF_OPEN[ctx.pi, ctx.si, chart_pos(ctx, ctx.me), cls, sil, mb] / 255.0
    callers = []
    for s in behind:
        if ctx.late():
            return random.random() < p_chart
        p, w = p_call_and_range(ctx, mem, s, my_rr)
        callers.append((s, p, eq_vs(cls, w)))
    if ctx.late():
        return random.random() < p_chart
    if not callers:  # nobody left to act (e.g. BB option): free check, jam per chart only
        return state.to_call > 0 and random.random() < p_chart
    ev_j, ev_f = jam_ev(ctx, state.pot, callers)
    return decide_mix(ev_j - ev_f, p_chart)


def decide_mix(gap, p_chart):
    """True = aggressive action. EV model decides clear spots, CFR mix decides close ones."""
    if gap > EV_MARGIN:
        return True
    if gap < -EV_MARGIN:
        return False
    return random.random() < p_chart


def facing_jam(state, ctx, mem, cls, sil, my_rr, jammer, n_callers):
    post, r = opp_post(ctx, mem, jammer, action=2)
    if r is not None:
        jpos = chart_pos(ctx, jammer)
        behind = behind_seats(ctx, jammer)
        revs = [rank(ctx.revealed[s]) for s in behind if s in ctx.revealed]
        mb = max(revs) if revs else 0
        pj = PF_OPEN[ctx.pi, ctx.si, jpos, np.arange(169), SIL[r], mb] / 255.0
        fr = freqs(ctx, mem, jammer)
        # loose jammers: blend their chart range toward their whole posterior
        loose = min(1.0, max(0.0, (fr["jam"] - 0.06) / 0.25))
        w = post * ((1 - loose) * pj + loose)
        if w.sum() <= 0:
            w = post
    else:
        w = post
    eq = eq_vs(cls, w)
    if n_callers:
        eq *= 0.8 ** n_callers  # crude multiway discount
    p_chart = PF_CALL[ctx.pi, ctx.si, chart_pos(ctx, ctx.me), cls, sil, r if r is not None else 12] / 255.0
    if n_callers:
        p_chart *= 0.5
    if ctx.late():
        return call_or_check(state) if random.random() < p_chart else check_or_fold(state)
    ev_c, ev_f = call_ev(ctx, state.pot, jammer, eq)
    return call_or_check(state) if decide_mix(ev_c - ev_f, p_chart) else check_or_fold(state)


def short_vs_raise(state, ctx, mem, cls, sil, my_rr, raiser):
    """Facing a non-all-in raise while short: re-jam or fold (call with monsters)."""
    post, r = opp_post(ctx, mem, raiser, action=1)
    fr = freqs(ctx, mem, raiser)
    w_r = range_top(post, max(0.08, fr["pfr"] / max(fr["vpip"], 0.05)))
    committed = (ctx.bet[raiser]) / max(1, ctx.stack[raiser] + ctx.bet[raiser])
    p_r = min(0.95, 0.35 + committed)
    callers = [(raiser, p_r, eq_vs(cls, w_r))]
    for s in behind_seats(ctx, ctx.me):
        if s == raiser:
            continue
        p, w = p_call_and_range(ctx, mem, s, my_rr)
        callers.append((s, p * 0.8, eq_vs(cls, w)))
    if ctx.late():
        return check_or_fold(state)
    ev_j, ev_f = jam_ev(ctx, state.pot, callers, fold_winner=raiser)
    if ev_j - ev_f > 0.0:
        return jam(state)
    return check_or_fold(state)


def deep_preflop(state, ctx, mem, cls, raisers):
    """>25bb: open/3-bet/call on equity vs reveal-conditioned ranges + ICM risk premium."""
    bb = ctx.bb
    me = ctx.me
    to_call = state.to_call
    behind = behind_seats(ctx, me)
    live_opps = [s for s in ctx.live if s != me]
    if not raisers:
        if not behind:  # BB with limpers / SB complete spots
            if to_call == 0:
                eqs = [eq_opp(ctx, mem, cls, s) for s in live_opps]
                g = float(np.exp(np.mean(np.log(np.maximum(eqs, 1e-3))))) if eqs else 0.5
                if g > 0.62:
                    return raise_to(state, 4 * bb)
                return Action.check() if state.can("check") else call_or_check(state)
        eqs = [eq_opp(ctx, mem, cls, s) for s in behind]
        g = float(np.exp(np.mean(np.log(np.maximum(eqs, 1e-3))))) if eqs else 0.5
        # stricter with more players behind
        thr = 0.50 + 0.012 * max(0, len(behind) - 1)
        if ctx.pos.get(me) == 1 and len(behind) == 1:
            thr = 0.48
        limpers = sum(1 for s in live_opps if ctx.bet[s] >= bb and s not in behind and ctx.pos.get(s) != 2)
        if g >= thr:
            return raise_to(state, (2.5 + limpers) * bb)
        if to_call == 0:
            return Action.check()
        if to_call <= bb // 2 and g >= thr - 0.04:
            return call_or_check(state)
        return check_or_fold(state)
    # facing a raise
    last = raisers[-1]
    post, _ = opp_post(ctx, mem, last, action=1)
    fr = freqs(ctx, mem, last)
    n_raises = len(raisers)
    frac = max(0.04, fr["pfr"] / max(fr["vpip"], 0.05)) * (0.5 ** (n_raises - 1))
    eq = eq_vs(cls, range_top(post, min(1.0, frac * 1.5)))
    stack_total = ctx.stack[me] + ctx.bet[me]
    try:
        if ctx.late():
            return check_or_fold(state)
        rp = risk_premium(ctx, last, to_call + ctx.bet[me], dead=state.pot - ctx.bet[last] - ctx.bet[me])
    except Exception:
        rp = 0.0
    rp = min(0.15, max(-0.05, rp))
    pot_odds = to_call / max(1, state.pot + to_call)
    if eq >= THREEBET_EQ + rp:
        if n_raises >= 2 or to_call * 3 > stack_total * 0.35:
            return jam(state)
        return raise_to(state, ctx.bet[last] * 3)
    is_bb = ctx.pos.get(me) == 2 or (ctx.n_players == 2 and ctx.pos.get(me) == 1)
    if is_bb and n_raises == 1 and pot_odds <= 0.33 and eq >= 0.47 + rp:
        return call_or_check(state)
    if eq >= 0.55 + rp and to_call <= 0.06 * stack_total and n_raises == 1:
        return call_or_check(state)
    return check_or_fold(state)


def preflop(state, ctx, mem):
    cls, sil, my_rr = my_hand_info(state)
    raisers, callers = preflop_actions(state)
    me = ctx.me
    endgame = CAP_PAYOUT == "rank" and ctx.hands_left <= 0  # last hand: exact rank payoff
    if ctx.eff_bb <= SHORT_BB or endgame:
        jammers = [s for s in raisers if ctx.stack[s] == 0]
        if not raisers:
            if state.to_call == 0 and not behind_seats(ctx, me):
                # BB option: jam or check
                pass
            return jam(state) if short_unopened(state, ctx, mem, cls, sil, my_rr) else check_or_fold(state)
        last = raisers[-1]
        if ctx.stack[last] == 0 or state.to_call >= ctx.stack[me]:
            # facing an all-in (or a raise that puts us all-in)
            ncall = sum(1 for s in callers if s != me and raisers.index(last) < len(raisers))
            return facing_jam(state, ctx, mem, cls, sil, my_rr, last, len(set(callers) - {me}) if jammers else 0)
        return short_vs_raise(state, ctx, mem, cls, sil, my_rr, last)
    return deep_preflop(state, ctx, mem, cls, raisers)


# ---------------------------------------------------------------- postflop
def postflop(state, ctx, mem):
    raisers, _ = preflop_actions(state)
    infos = []
    for s in ctx.live:
        if s == ctx.me or s not in ctx.revealed:
            continue
        post, _ = opp_post(ctx, mem, s, action=1 if s in raisers else 0)
        fr = freqs(ctx, mem, s)
        infos.append({"seat": s, "reveal": ctx.revealed[s], "post": post,
                      "agg": fr["aggression"], "fold_to_raise": fr["fold_to_raise"]})
    rp = 0.0
    try:
        opps = [s for s in ctx.live if s != ctx.me]
        if opps:
            v = max(opps, key=lambda s: ctx.stack[s])
            amt = min(ctx.stack[ctx.me], ctx.stack[v])
            stake = max(state.to_call, state.pot)
            if amt > 0 and stake * 4 >= ctx.stack[ctx.me]:
                rp = risk_premium(ctx, v, amt, dead=state.pot)
                rp = min(0.12, max(0.0, rp))
    except Exception:
        rp = 0.0
    used = (perf_counter() - ctx.t0) * 1000.0
    return PF.postflop_decide(state, list(state.cards), list(state.board), infos, rp,
                              budget_ms=max(0.05, min(0.2, 0.5 - used)))


# ---------------------------------------------------------------- agent
class Agent(BaseAgent):
    def act(self, state, memory):
        self_t0 = perf_counter()
        mem = OM.load(memory, state)
        try:
            mem = OM.update(mem, state.events, state.seat, state)
        except Exception:
            mem = OM.load(b"", state)
        try:
            action = self.decide(state, mem, self_t0)
        except Exception:
            action = None
        if action is None:
            action = fallback(state)
        try:
            out = OM.dump(mem)
        except Exception:
            out = b""
        return action, out

    def decide(self, state, mem, t0=None):
        kinds = [a.kind for a in state.legal_actions]
        if state.phase == "choose" or kinds == ["choose"]:
            try:
                return choose_action(state)
            except Exception:
                return Action("choose")
        if state.phase != "betting":
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount)
        if len(state.cards) != 2:
            return check_or_fold(state)
        ctx = Ctx(state, t0)
        if not state.board:
            return preflop(state, ctx, mem)
        return postflop(state, ctx, mem)


def fallback(state):
    try:
        if state.phase == "choose":
            return Action("choose")
        return check_or_fold(state)
    except Exception:
        return Action("fold")
