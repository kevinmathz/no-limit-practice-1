"""Keep-reveal-discard (KRD) abstract-game solver: choose policy + short-stack push/fold.

Game per cell (n players, equal stacks S bb, blinds 0.5/1, chip EV in bb):
  1. chance: 3 distinct cards per seat from one deck;
  2. every seat simultaneously picks one of common.CHOOSE_ACTIONS for its 3-card class
     (common.canon3, 1755 classes): keep one card, reveal one, discard one;
  3. push/fold in preflop order with all revealed cards public:
       opener  infoset (pos, kept169, shown_is_low, max revealed rank of players behind)
       caller  infoset (pos, kept169, shown_is_low, jammer's revealed rank)
     (other callers are ignored in the caller key; BB never opens: all-fold = BB walk);
  4. showdown among kept 2-card classes: hu_eq heads-up, normalised product of pairwise
     equities multiway (as no_limit/game/cfr/preflop_vector.py).

Algorithm: mini-batch Monte-Carlo CFR with discounting (DCFR a=1.5, b=0, g=2).
Each iteration samples a batch of deals + every seat's choose action from the current
strategy. The push/fold tree is then evaluated EXACTLY in expectation (all jammers x all
caller subsets, vectorised with numpy over the batch). That gives
  * counterfactual values for every opener / caller infoset in the batch (all seats are
    updated simultaneously; regret weighted by opponents' reach),
  * for a sub-batch, each seat's value for all 6 choose actions with the others' choose
    actions fixed at their samples (external sampling of the choose layer).
Regrets / average strategies live in flat numpy arrays indexed by integer infoset ids.

Usage:
  python solve_cfr.py --quick            # ~10 min, writes cfr_tables_quick.npz
  python solve_cfr.py                    # full run, writes cfr_tables.npz + CFR_REPORT.md
  python solve_cfr.py --cell 6 12 --secs 30   # single cell smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

NC = 169
NCH = common.N_CHOOSE          # 1755
NA = len(common.CHOOSE_ACTIONS)  # 6
NPOS = 8
NR = 13

PLAYERS_GRID = [2, 3, 4, 5, 6, 7, 8]
STACK_GRID = [3, 5, 8, 12, 17, 25, 40]   # 40 = deep cell (choose strategy for deep play)


# ----------------------------------------------------------------------------- equity
def load_equity() -> tuple[np.ndarray, str]:
    p = os.path.join(HERE, "equity.npz")
    if os.path.exists(p):
        try:
            d = np.load(p)
            eq = np.asarray(d["hu_eq"], dtype=np.float64)
            if eq.shape == (NC, NC):
                return eq, "equity.npz"
        except Exception:
            pass
    sys.path.insert(0, os.path.join(HERE, "..", "..", "no_limit"))
    from game.tables.hu_matchups import CLASSES, EQUITY  # type: ignore
    assert list(CLASSES) == common.CLASS_LIST
    return np.asarray(EQUITY, dtype=np.float64), "no_limit/game/tables/hu_matchups.py"


# ----------------------------------------------------------------------------- card tables
def build_card_tables():
    """LUT sorted-desc triple -> choose class; per (class, action): hand169, reveal rank, low."""
    lut = np.full(52 * 52 * 52, -1, dtype=np.int16)
    for cids, (cid, _order) in common._COMBO_MAP.items():
        a, b, c = cids
        lut[(a * 52 + b) * 52 + c] = cid
    hand = np.zeros((NCH, NA), dtype=np.int16)
    rrank = np.zeros((NCH, NA), dtype=np.int8)
    low = np.zeros((NCH, NA), dtype=np.int8)
    for k, t in enumerate(common.CHOOSE_CLASSES):
        for a, (kp, rp) in enumerate(common.CHOOSE_ACTIONS):
            kc, rc = t[kp], t[rp]
            hand[k, a] = common.class2(kc, rc)
            rrank[k, a] = common.rank_of(rc)
            low[k, a] = 1 if common.rank_of(rc) < common.rank_of(kc) else 0
    # deal multiplicity of each choose class (out of C(52,3)=22100)
    mult = np.zeros(NCH, dtype=np.float64)
    for cids, (cid, _o) in common._COMBO_MAP.items():
        mult[cid] += 1
    return lut, hand, rrank, low, mult


LUT, HAND, RRANK, LOW, MULT = build_card_tables()
HU_EQ, EQ_SRC = load_equity()
HU_EQ32 = HU_EQ.astype(np.float32)
CLASS_W = np.array([(6 if len(c) == 2 else 4 if c[2] == "s" else 12) for c in common.CLASS_LIST], float)
CLASS_W /= CLASS_W.sum()
EQ_VS_RANDOM = HU_EQ @ CLASS_W


def default_choose_row(k: int) -> np.ndarray:
    """Keep the best-equity 2-card hand (vs random), reveal 50/50."""
    best = max(range(3), key=lambda d: EQ_VS_RANDOM[HAND[k, [a for a in range(NA)
                                                              if 3 - sum(common.CHOOSE_ACTIONS[a]) == d][0]]])
    row = np.zeros(NA)
    for a, (kp, rp) in enumerate(common.CHOOSE_ACTIONS):
        if 3 - kp - rp == best:
            row[a] = 0.5
    return row


DEFAULT_CHOOSE = np.array([default_choose_row(k) for k in range(NCH)])


# ----------------------------------------------------------------------------- game layout
class Layout:
    def __init__(self, n: int, S: float):
        self.n, self.S = n, float(S)
        if n == 2:
            self.pos = np.array([0, 1])
            self.blind = np.array([0.5, 1.0])
        else:
            self.pos = np.array(list(range(3, n)) + [0, 1, 2])
            self.blind = np.array([0.0] * (n - 3) + [0.0, 0.5, 1.0])
        M = 1 << n
        self.M = M
        self.mem = np.array([[(m >> i) & 1 for i in range(n)] for m in range(M)], dtype=np.float64)
        cnt = self.mem.sum(1)
        dead = ((1 - self.mem) * self.blind[None, :]).sum(1)
        self.pot = cnt * self.S + dead
        self.mem32 = self.mem.astype(np.float32)
        self.pot32 = self.pot.astype(np.float32)
        self.payoff_off = (-self.mem * self.S - (1 - self.mem) * self.blind[None, :]).astype(np.float32)
        # per jammer j: global mask indices of {j} U subset(j+1..n-1) and bit matrix
        self.jm = []
        for j in range(n - 1):
            q = n - 1 - j
            subs = np.arange(1 << q)
            gm = (1 << j) | (subs << (j + 1))
            bits = ((subs[:, None] >> np.arange(q)[None, :]) & 1).astype(bool)
            self.jm.append((q, gm, bits))
        # walk payoff (everyone folds to BB)
        self.walk_pay = -self.blind.copy()
        self.walk_pay[n - 1] += self.blind.sum()


def open_index(pos, hand, low, mb):
    return ((pos * NC + hand) * 2 + low) * NR + mb


N_INFO = NPOS * NC * 2 * NR  # per table (open / call)


def evaluate(L: Layout, hand, low, rr, jam_tab, call_tab, want_cfv: bool, chunk: int):
    """hand/low/rr: [R, n] in action order. jam_tab/call_tab: flat [N_INFO] P(jam)/P(call).
    Returns EV [R, n]; with want_cfv also dict of infoset ids and cfv arrays."""
    R, n = hand.shape
    out_ev = np.empty((R, n))
    cf = None
    if want_cfv:
        cf = dict(o_idx=np.empty((R, n - 1), np.int64), o_j=np.empty((R, n - 1)), o_f=np.empty((R, n - 1)),
                  o_p=np.empty((R, n - 1)),
                  c_idx=np.empty((R, n, n), np.int64), c_c=np.zeros((R, n, n)), c_f=np.zeros((R, n, n)),
                  c_p=np.zeros((R, n, n)))
    pos = L.pos
    for s0 in range(0, R, chunk):
        s1 = min(R, s0 + chunk)
        h, lo, r = hand[s0:s1].astype(np.int64), low[s0:s1].astype(np.int64), rr[s0:s1].astype(np.int64)
        B = s1 - s0
        # max revealed rank behind each opener
        mb = np.zeros((B, n), np.int64)
        for k in range(n - 2, -1, -1):
            mb[:, k] = np.maximum(r[:, k + 1], mb[:, k + 1] if k + 1 < n - 1 else 0)
        oidx = open_index(pos[None, :n - 1], h[:, :n - 1], lo[:, :n - 1], mb[:, :n - 1])
        jam = np.zeros((B, n))
        jam[:, :n - 1] = jam_tab[oidx]
        reach = np.ones((B, n))
        reach[:, 1:] = np.cumprod(1 - jam[:, :-1], axis=1)
        first = reach * jam                    # P(k is the first jammer)
        walk = reach[:, n - 1]
        # call infosets: k facing jam from j  (k > j)
        cidx = open_index(pos[None, None, :], h[:, None, :], lo[:, None, :], r[:, :, None])  # [B, j, k]
        cprob = call_tab[cidx]
        # pairwise equities, diag 1, strength DP over masks
        E = HU_EQ32[h[:, :, None], h[:, None, :]]
        E[:, np.arange(n), np.arange(n)] = 1.0
        st = np.empty((B, L.M, n), np.float32)
        st[:, 0, :] = 1.0
        for b in range(n):
            w = 1 << b
            np.multiply(st[:, 0:w, :], E[:, None, :, b], out=st[:, w:2 * w, :])
        st *= L.mem32[None]
        tot = st.sum(2, keepdims=True)
        tot[tot == 0] = 1.0
        st /= tot
        st *= L.pot32[None, :, None]
        st += L.payoff_off[None]          # pay = mem*(share*pot - S) - (1-mem)*blind
        pay = st
        ev = walk[:, None] * L.walk_pay[None, :]
        if want_cfv:
            o_j = np.zeros((B, n - 1))
        cprob32 = cprob.astype(np.float32)
        for j in range(n - 1):
            q, gm, bits = L.jm[j]
            cj = cprob32[:, j, j + 1:]                                 # [B, q]
            pc = np.ones((B, 1), np.float32)
            for t in range(q):
                c = cj[:, t:t + 1]
                pc = np.concatenate([pc * (1 - c), pc * c], 1)        # bit t of subset = caller t
            pj = pay[:, gm, :]                                         # [B, 2^q, n]
            ev += first[:, j, None] * np.einsum("bs,bsn->bn", pc, pj)
            if want_cfv:
                o_j[:, j] = reach[:, j] * np.einsum("bs,bs->b", pc, pj[:, :, j])
                for t in range(q):
                    k = j + 1 + t
                    sel = bits[:, t]
                    # P(other callers' pattern) = pc / c_t on subsets containing t; recompute w/o t
                    c = cj[:, t:t + 1]
                    ex = np.ones((B, 1), np.float32)
                    for u in range(q):
                        if u == t:
                            ex = np.concatenate([np.zeros_like(ex), ex], 1)   # force bit t = 1
                        else:
                            cu = cj[:, u:u + 1]
                            ex = np.concatenate([ex * (1 - cu), ex * cu], 1)
                    cf["c_c"][s0:s1, j, k] = first[:, j] * np.einsum("bs,bs->b", ex[:, sel], pj[:, sel, k])
                    cf["c_f"][s0:s1, j, k] = -first[:, j] * L.blind[k]
                    cf["c_p"][s0:s1, j, k] = cprob[:, j, k]
        out_ev[s0:s1] = ev
        if want_cfv:
            cf["o_idx"][s0:s1] = oidx
            cf["o_j"][s0:s1] = o_j
            cf["o_f"][s0:s1] = -reach[:, :n - 1] * L.blind[None, :n - 1]
            cf["o_p"][s0:s1] = jam[:, :n - 1]
            cf["c_idx"][s0:s1] = cidx
    return out_ev, cf


def call_mask(n):
    jj, kk = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return kk > jj


# ----------------------------------------------------------------------------- solver
def rm(reg: np.ndarray) -> np.ndarray:
    """Regret matching along the last axis."""
    p = np.maximum(reg, 0)
    s = p.sum(-1, keepdims=True)
    k = reg.shape[-1]
    return np.where(s > 0, p / np.where(s > 0, s, 1), 1.0 / k)


class Solver:
    def __init__(self, n: int, S: float, seed: int = 0):
        self.L = Layout(n, S)
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.r_ch = np.zeros((NCH, NA))
        self.a_ch = np.zeros((NCH, NA))
        self.r_open = np.zeros((N_INFO, 2))   # [jam, fold]
        self.a_open = np.zeros((N_INFO, 2))
        self.r_call = np.zeros((N_INFO, 2))   # [call, fold]
        self.a_call = np.zeros((N_INFO, 2))
        self.v_ch = np.zeros(NCH)
        self.v_open = np.zeros(N_INFO)
        self.v_call = np.zeros(N_INFO)
        self.t = 0
        self.chunk = max(256, int(2_000_000 / ((1 << n) * n)))
        self.cmask = call_mask(n)

    # strategies
    def cur(self):
        return rm(self.r_ch), rm(self.r_open)[:, 0], rm(self.r_call)[:, 0]

    def avg(self):
        ch = self.a_ch / np.maximum(self.a_ch.sum(1, keepdims=True), 1e-300)
        ch[self.a_ch.sum(1) <= 0] = DEFAULT_CHOOSE[self.a_ch.sum(1) <= 0]
        jo = np.where(self.a_open.sum(1) > 0, self.a_open[:, 0] / np.maximum(self.a_open.sum(1), 1e-300), 0.5)
        jc = np.where(self.a_call.sum(1) > 0, self.a_call[:, 0] / np.maximum(self.a_call.sum(1), 1e-300), 0.5)
        return ch, jo, jc

    def deal(self, B):
        n = self.n
        keys = self.rng.random((B, 52))
        cards = np.argpartition(keys, 3 * n, axis=1)[:, :3 * n].reshape(B, n, 3)
        cards = -np.sort(-cards, axis=2)
        return LUT[(cards[..., 0] * 52 + cards[..., 1]) * 52 + cards[..., 2]].astype(np.int64)  # [B, n]

    def sample_actions(self, cls, ch):
        cdf = np.cumsum(ch[cls], -1)
        u = self.rng.random(cls.shape + (1,)) * cdf[..., -1:]
        return np.minimum((u > cdf).sum(-1), NA - 1)

    def iterate(self, B_bet: int, B_ch: int):
        n, L = self.n, self.L
        ch, jo, jc = self.cur()
        cls = self.deal(B_bet)
        act = self.sample_actions(cls, ch)
        hand, rr, low = HAND[cls, act], RRANK[cls, act], LOW[cls, act]
        ev, cf = evaluate(L, hand, low, rr, jo, jc, True, self.chunk)
        # --- betting regrets
        d = cf["o_j"] - cf["o_f"]
        p = cf["o_p"]
        idx = cf["o_idx"].ravel()
        r_upd = np.zeros((N_INFO, 2))
        r_upd[:, 0] = np.bincount(idx, ((1 - p) * d).ravel(), N_INFO)
        r_upd[:, 1] = np.bincount(idx, (-p * d).ravel(), N_INFO)
        cnt_o = np.bincount(idx, minlength=N_INFO).astype(float)
        m = self.cmask
        cidx = cf["c_idx"][:, m].ravel()
        dc = (cf["c_c"] - cf["c_f"])[:, m].ravel()
        pc = cf["c_p"][:, m].ravel()
        c_upd = np.zeros((N_INFO, 2))
        c_upd[:, 0] = np.bincount(cidx, (1 - pc) * dc, N_INFO)
        c_upd[:, 1] = np.bincount(cidx, -pc * dc, N_INFO)
        cnt_c = np.bincount(cidx, minlength=N_INFO).astype(float)
        # --- choose regrets on a sub-batch: every seat x every action
        Bc = min(B_ch, B_bet)
        h0, r0, l0 = hand[:Bc], rr[:Bc], low[:Bc]
        H = np.repeat(h0[:, None, None, :], n, 1).repeat(NA, 2)       # [Bc, p, a, n]
        Rr = np.repeat(r0[:, None, None, :], n, 1).repeat(NA, 2)
        Lo = np.repeat(l0[:, None, None, :], n, 1).repeat(NA, 2)
        seat = np.arange(n)
        c0 = cls[:Bc]
        H[:, seat, :, seat] = HAND[c0].transpose(1, 0, 2)
        Rr[:, seat, :, seat] = RRANK[c0].transpose(1, 0, 2)
        Lo[:, seat, :, seat] = LOW[c0].transpose(1, 0, 2)
        evv, _ = evaluate(L, H.reshape(-1, n), Lo.reshape(-1, n), Rr.reshape(-1, n), jo, jc, False, self.chunk)
        evv = evv.reshape(Bc, n, NA, n)[:, seat, :, seat].transpose(1, 0, 2)  # [Bc, p, a]
        sig = ch[c0]                                                       # [Bc, p, a]
        base = (sig * evv).sum(-1, keepdims=True)
        ch_upd = np.zeros(NCH * NA)
        flat = (c0[..., None] * NA + np.arange(NA)).ravel()
        ch_upd = np.bincount(flat, (evv - base).ravel(), NCH * NA).reshape(NCH, NA)
        cnt_ch = np.bincount(cls.ravel(), minlength=NCH).astype(float)
        # --- DCFR discount + accumulate
        self.t += 1
        t = self.t
        pos_f, neg_f = t ** 1.5 / (t ** 1.5 + 1), 0.5
        for R_, U in ((self.r_ch, ch_upd), (self.r_open, r_upd), (self.r_call, c_upd)):
            R_ *= np.where(R_ > 0, pos_f, neg_f)
            R_ += U
        g = (t / (t + 1)) ** 2
        for A_ in (self.a_ch, self.a_open, self.a_call):
            A_ *= g
        self.a_ch += cnt_ch[:, None] * ch
        self.a_open[:, 0] += cnt_o * jo
        self.a_open[:, 1] += cnt_o * (1 - jo)
        self.a_call[:, 0] += cnt_c * jc
        self.a_call[:, 1] += cnt_c * (1 - jc)
        self.v_ch += cnt_ch
        self.v_open += cnt_o
        self.v_call += cnt_c


# ----------------------------------------------------------------------------- diagnostics
def sample_eval(S_: Solver, ch, jo, jc, B, cls=None, act=None):
    if cls is None:
        cls = S_.deal(B)
    if act is None:
        act = S_.sample_actions(cls, ch)
    hand, rr, low = HAND[cls, act], RRANK[cls, act], LOW[cls, act]
    ev, cf = evaluate(S_.L, hand, low, rr, jo, jc, True, S_.chunk)
    return cls, act, ev, cf


def betting_br_sums(S_: Solver, cf):
    """Per-seat sums of cfv over infosets: returns per seat arrays (open J,F,count ; call C,F)."""
    n = S_.n
    out = []
    m = S_.cmask
    for k in range(n):
        oJ = np.zeros(N_INFO); oF = np.zeros(N_INFO); cC = np.zeros(N_INFO); cF = np.zeros(N_INFO)
        if k < n - 1:
            idx = cf["o_idx"][:, k]
            oJ = np.bincount(idx, cf["o_j"][:, k], N_INFO)
            oF = np.bincount(idx, cf["o_f"][:, k], N_INFO)
        for j in range(k):
            idx = cf["c_idx"][:, j, k]
            cC += np.bincount(idx, cf["c_c"][:, j, k], N_INFO)
            cF += np.bincount(idx, cf["c_f"][:, j, k], N_INFO)
        out.append((oJ, oF, cC, cF))
    return out


def br_gain_cv(S_: Solver, ch, jo, jc, B):
    """Cross-validated best-response gain (bb/hand) per seat for the push/fold policy:
    BR actions picked on sample A, gain measured on independent sample B."""
    _, _, _, cfA = sample_eval(S_, ch, jo, jc, B)
    _, _, _, cfB = sample_eval(S_, ch, jo, jc, B)
    sa, sb = betting_br_sums(S_, cfA), betting_br_sums(S_, cfB)
    gains, gains_in = [], []
    for k in range(S_.n):
        oJa, oFa, cCa, cFa = sa[k]
        oJb, oFb, cCb, cFb = sb[k]
        bo = oJa > oFa
        bc = cCa > cFa
        g = (np.where(bo, oJb, oFb) - (jo * oJb + (1 - jo) * oFb)).sum()
        g += (np.where(bc, cCb, cFb) - (jc * cCb + (1 - jc) * cFb)).sum()
        gi = (np.maximum(oJb, oFb) - (jo * oJb + (1 - jo) * oFb)).sum()
        gi += (np.maximum(cCb, cFb) - (jc * cCb + (1 - jc) * cFb)).sum()
        gains.append(g / B)
        gains_in.append(gi / B)
    return np.array(gains), np.array(gains_in)


def choose_values(S_: Solver, ch, jo, jc, B):
    """Sums over sample of value of each choose action per class (all seats pooled) and counts."""
    n = S_.n
    cls = S_.deal(B)
    act = S_.sample_actions(cls, ch)
    hand, rr, low = HAND[cls, act], RRANK[cls, act], LOW[cls, act]
    H = np.repeat(hand[:, None, None, :], n, 1).repeat(NA, 2)
    Rr = np.repeat(rr[:, None, None, :], n, 1).repeat(NA, 2)
    Lo = np.repeat(low[:, None, None, :], n, 1).repeat(NA, 2)
    seat = np.arange(n)
    H[:, seat, :, seat] = HAND[cls].transpose(1, 0, 2)
    Rr[:, seat, :, seat] = RRANK[cls].transpose(1, 0, 2)
    Lo[:, seat, :, seat] = LOW[cls].transpose(1, 0, 2)
    evv, _ = evaluate(S_.L, H.reshape(-1, n), Lo.reshape(-1, n), Rr.reshape(-1, n), jo, jc, False, S_.chunk)
    evv = evv.reshape(B, n, NA, n)[:, seat, :, seat].transpose(1, 0, 2)
    V = np.zeros((NCH, NA))
    for a in range(NA):
        V[:, a] = np.bincount(cls.ravel(), evv[..., a].ravel(), NCH)
    return V, B * n


def choose_br_gain_cv(S_, ch, jo, jc, B):
    Va, _ = choose_values(S_, ch, jo, jc, B)
    Vb, cnt = choose_values(S_, ch, jo, jc, B)
    best = Va.argmax(1)
    g = (Vb[np.arange(NCH), best] - (ch * Vb).sum(1)).sum() / cnt
    gi = (Vb.max(1) - (ch * Vb).sum(1)).sum() / cnt
    return g, gi


def reveal_lower_policy(ch):
    """Same hand selection (discard marginals) as ch, but always reveal the lower card."""
    out = np.zeros_like(ch)
    for k in range(NCH):
        for a, (kp, rp) in enumerate(common.CHOOSE_ACTIONS):
            # partner action: same discard, swapped keep/reveal
            b = common.CHOOSE_ACTIONS.index((rp, kp))
            tot = ch[k, a] + ch[k, b]
            if LOW[k, a] == 1 or (LOW[k, a] == 0 and LOW[k, b] == 0 and a < b):
                # a reveals the lower card (or a pair: pick one)
                out[k, a] += tot
    return out


def reveal_higher_policy(ch):
    out = np.zeros_like(ch)
    for k in range(NCH):
        for a, (kp, rp) in enumerate(common.CHOOSE_ACTIONS):
            b = common.CHOOSE_ACTIONS.index((rp, kp))
            tot = ch[k, a] + ch[k, b]
            if (LOW[k, a] == 0 and LOW[k, b] == 1) or (LOW[k, a] == 0 and LOW[k, b] == 0 and a < b):
                out[k, a] += tot
    return out


def deviant_loss(S_: Solver, ch, jo, jc, dev_ch, B, exploit: bool):
    """EV (bb/hand) lost by one seat using dev_ch instead of ch, averaged over seats.
    Without exploit: opponents keep the equilibrium push/fold. With exploit: every opponent
    switches its push/fold to a best response (computed on an independent sample)."""
    n = S_.n
    losses = []
    seats = range(n) if n <= 4 else sorted({0, n // 2, n - 2, n - 1})
    for k in seats:
        cls = S_.deal(B)
        act = S_.sample_actions(cls, ch)
        _, _, ev_eq, _ = sample_eval(S_, ch, jo, jc, B, cls, act)
        act_d = act.copy()
        act_d[:, k] = S_.sample_actions(cls[:, k], dev_ch)
        if exploit:
            clsA = S_.deal(B)
            actA = S_.sample_actions(clsA, ch)
            actA[:, k] = S_.sample_actions(clsA[:, k], dev_ch)
            _, _, _, cfA = sample_eval(S_, ch, jo, jc, B, clsA, actA)
            sums = betting_br_sums(S_, cfA)
            jo2, jc2 = jo.copy(), jc.copy()
            for q in range(n):
                if q == k:
                    continue
                oJ, oF, cC, cF = sums[q]
                # infoset ids are position specific, so only seat q's rows change
                rows = np.nonzero((oJ != 0) | (oF != 0))[0]
                if q < n - 1:
                    sel = rows[(rows // (NC * 2 * NR)) == S_.L.pos[q]]
                    jo2[sel] = (oJ[sel] > oF[sel]).astype(float)
                rows = np.nonzero((cC != 0) | (cF != 0))[0]
                sel = rows[(rows // (NC * 2 * NR)) == S_.L.pos[q]]
                jc2[sel] = (cC[sel] > cF[sel]).astype(float)
            # deviant keeps its own equilibrium push/fold rows
            pk = S_.L.pos[k]
            blk = slice(pk * NC * 2 * NR, (pk + 1) * NC * 2 * NR)
            jo2[blk], jc2[blk] = jo[blk], jc[blk]
            _, _, ev_d, _ = sample_eval(S_, ch, jo2, jc2, B, cls, act_d)
        else:
            _, _, ev_d, _ = sample_eval(S_, ch, jo, jc, B, cls, act_d)
        losses.append(ev_eq[:, k].mean() - ev_d[:, k].mean())
    return float(np.mean(losses))


def posterior_of(ch):
    post = np.zeros((NR, NC))
    w = MULT[:, None] * ch
    np.add.at(post, (RRANK.astype(np.int64), HAND.astype(np.int64)), w)
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    return post


def stats_of(ch):
    """Aggregate choose stats weighted by deal frequency."""
    w = MULT[:, None] * ch / MULT.sum()
    disc = np.zeros(NCH, dtype=bool)
    st = {}
    # P(reveal Ace | holding exactly one ace among kept 2) etc.
    keep_r = np.zeros((NCH, NA), int)
    for k, t in enumerate(common.CHOOSE_CLASSES):
        for a, (kp, rp) in enumerate(common.CHOOSE_ACTIONS):
            keep_r[k, a] = common.rank_of(t[kp])
    rev_r = RRANK.astype(int)
    pair = np.array([[CL[0] == CL[1] for CL in (common.CLASS_LIST[h] for h in HAND[k])] for k in range(NCH)])
    ace_kept2 = (keep_r == 12) | (rev_r == 12)
    one_ace = ace_kept2 & ~pair
    st["P(reveal A | exactly one A in the 2 kept, non-pair)"] = float((w * (one_ace & (rev_r == 12))).sum() / max((w * one_ace).sum(), 1e-12))
    st["P(reveal lower card | non-pair kept)"] = float((w * (~pair & (LOW == 1))).sum() / max((w * ~pair).sum(), 1e-12))
    st["P(keep a pair)"] = float((w * pair).sum())
    # P(keep pair | dealt a pair among the 3)
    trips = np.array([len({common.rank_of(c) for c in t}) == 1 for t in common.CHOOSE_CLASSES])
    has_pair = np.array([len({common.rank_of(c) for c in t}) == 2 for t in common.CHOOSE_CLASSES])
    st["P(keep the pair | dealt exactly one pair)"] = float((w[has_pair] * pair[has_pair]).sum() / max(w[has_pair].sum(), 1e-12))
    post = posterior_of(ch)
    pairs_idx = [i for i, c in enumerate(common.CLASS_LIST) if len(c) == 2]
    st["P(pair | revealed A)"] = float(post[12, pairs_idx].sum())
    st["P(AA | revealed A)"] = float(post[12, common.CLASS_INDEX["AA"]])
    st["P(pair | revealed 2)"] = float(post[0, pairs_idx].sum())
    st["P(pair | revealed r) for r=2..A"] = [round(float(post[r, pairs_idx].sum()), 3) for r in range(NR)]
    aces = [i for i, c in enumerate(common.CLASS_LIST) if c[0] == "A" or c[1] == "A"]
    st["P(kept hand contains A | revealed K)"] = float(post[11, aces].sum())
    # entropy-ish: fraction of deal mass on classes with a mixed reveal decision (both orders >10%)
    mixed = 0.0
    for k in range(NCH):
        for d in range(3):
            acts = [a for a, (kp, rp) in enumerate(common.CHOOSE_ACTIONS) if 3 - kp - rp == d]
            pa, pb = ch[k, acts[0]], ch[k, acts[1]]
            if pa + pb > 0.05 and min(pa, pb) / (pa + pb) > 0.1 and not pair[k, acts[0]]:
                mixed += MULT[k] * (pa + pb)
    st["deal mass with mixed reveal (minority >10%)"] = float(mixed / MULT.sum())
    rv = np.zeros(NR)
    np.add.at(rv, rev_r.ravel(), w.ravel())
    st["reveal rank distribution (2..A)"] = [round(float(x), 4) for x in rv]
    pr = np.zeros(NR); pt = np.zeros(NR)
    # how pairs are revealed: for kept pair of rank r, the reveal is necessarily rank r;
    # report P(keep pair rank r | dealt pair rank r + kicker)
    for k, t in enumerate(common.CHOOSE_CLASSES):
        rk = [common.rank_of(c) for c in t]
        if len(set(rk)) == 2:
            pr_r = max(set(rk), key=rk.count)
            pt[pr_r] += MULT[k]
            pr[pr_r] += MULT[k] * (ch[k] * pair[k]).sum()
    st["P(keep pair | dealt pair+kicker) by pair rank 2..A"] = [round(float(x), 3) for x in pr / np.maximum(pt, 1)]
    return st


# ----------------------------------------------------------------------------- cell driver
def batch_sizes(n):
    B_bet = {2: 40000, 3: 30000, 4: 20000, 5: 12000, 6: 8000, 7: 5000, 8: 3000}[n]
    B_ch = {2: 4000, 3: 2500, 4: 1500, 5: 1000, 6: 600, 7: 400, 8: 250}[n]
    return B_bet, B_ch


def solve_cell(args):
    n, S, secs, seed, do_diag = args
    t0 = time.time()
    sv = Solver(n, S, seed)
    B_bet, B_ch = batch_sizes(n)
    while time.time() - t0 < secs:
        sv.iterate(B_bet, B_ch)
    t_solve = time.time() - t0
    ch, jo, jc = sv.avg()
    res = dict(n=n, S=S, iters=sv.t, deals=sv.t * B_bet, choose_deals=sv.t * B_ch, secs=t_solve,
               ch=ch.astype(np.float32), jo=jo.astype(np.float32), jc=jc.astype(np.float32),
               v_ch=sv.v_ch.astype(np.float32), v_open=sv.v_open.astype(np.float32),
               v_call=sv.v_call.astype(np.float32))
    if do_diag:
        Bd = 8 * B_bet
        g, gi = br_gain_cv(sv, ch, jo, jc, Bd)
        res["br_pf_cv"] = g.tolist()
        res["br_pf_insample"] = gi.tolist()
        cg, cgi = choose_br_gain_cv(sv, ch, jo, jc, 8 * B_ch)
        res["br_choose_cv"] = float(cg)
        res["br_choose_insample"] = float(cgi)
        Bl = 2 * B_bet
        rl = reveal_lower_policy(ch)
        res["loss_reveal_lower"] = deviant_loss(sv, ch, jo, jc, rl, Bl, False)
        res["loss_reveal_lower_exploited"] = deviant_loss(sv, ch, jo, jc, rl, Bl, True)
        res["loss_reveal_higher"] = deviant_loss(sv, ch, jo, jc, reveal_higher_policy(ch), Bl, False)
        res["loss_reveal_higher_exploited"] = deviant_loss(sv, ch, jo, jc, reveal_higher_policy(ch), Bl, True)
        res["loss_default"] = deviant_loss(sv, ch, jo, jc, DEFAULT_CHOOSE, Bl, False)
        res["stats"] = stats_of(ch)
        _, _, ev, _ = sample_eval(sv, ch, jo, jc, Bd)
        res["ev_by_seat"] = ev.mean(0).tolist()
    res["total_secs"] = time.time() - t0
    save_cell(res)
    print(f"[cell n={n} S={S}] iters={sv.t} solve={t_solve:.0f}s total={res['total_secs']:.0f}s", flush=True)
    return res


CELL_DIR = os.path.join(HERE, "cells")
ARR_KEYS = ("ch", "jo", "jc", "v_ch", "v_open", "v_call")


def save_cell(res):
    os.makedirs(CELL_DIR, exist_ok=True)
    base = os.path.join(CELL_DIR, f"n{res['n']}_S{res['S']}")
    meta = {k: v for k, v in res.items() if k not in ARR_KEYS}
    np.savez_compressed(base + ".tmp.npz", meta=np.array(json.dumps(meta)), **{k: res[k] for k in ARR_KEYS})
    os.replace(base + ".tmp.npz", base + ".npz")


def load_cells(players, stacks):
    out = []
    for n in players:
        for S in stacks:
            p = os.path.join(CELL_DIR, f"n{n}_S{S}.npz")
            if os.path.exists(p):
                d = np.load(p)
                r = json.loads(str(d["meta"]))
                r.update({k: d[k] for k in ARR_KEYS})
                out.append(r)
    return out


def pack(results, players, stacks):
    P, Sg = len(players), len(stacks)
    choose = np.zeros((P, Sg, NCH, NA), np.uint8)
    pf_open = np.zeros((P, Sg, NPOS, NC, 2, NR), np.uint8)
    pf_call = np.zeros((P, Sg, NPOS, NC, 2, NR), np.uint8)
    post = np.zeros((P, Sg, NR, NC), np.float32)
    v_ch = np.zeros((P, Sg, NCH), np.float32)
    v_open = np.zeros((P, Sg, NPOS, NC, 2, NR), np.float32)
    v_call = np.zeros((P, Sg, NPOS, NC, 2, NR), np.float32)
    have = np.zeros((P, Sg), bool)
    iters = np.zeros((P, Sg), np.int32)
    for r in results:
        i, j = players.index(r["n"]), stacks.index(r["S"])
        ch = r["ch"].astype(np.float64)
        q = np.floor(ch * 255).astype(int)
        rem = 255 - q.sum(1)
        frac = ch * 255 - q
        order = np.argsort(-frac, axis=1)
        for k in range(NCH):
            q[k, order[k, :rem[k]]] += 1
        choose[i, j] = q
        pf_open[i, j] = np.round(r["jo"].reshape(NPOS, NC, 2, NR) * 255)
        pf_call[i, j] = np.round(r["jc"].reshape(NPOS, NC, 2, NR) * 255)
        post[i, j] = posterior_of(ch)
        v_ch[i, j] = r["v_ch"]
        v_open[i, j] = r["v_open"].reshape(NPOS, NC, 2, NR)
        v_call[i, j] = r["v_call"].reshape(NPOS, NC, 2, NR)
        have[i, j] = True
        iters[i, j] = r["iters"]
    # nearest-neighbour fill for missing cells (by players, then stack)
    for i in range(P):
        for j in range(Sg):
            if have[i, j]:
                continue
            cands = [(abs(players[a] - players[i]) * 10 + abs(b - j), a, b) for a in range(P) for b in range(Sg) if have[a, b]]
            _, a, b = min(cands)
            for arr in (choose, pf_open, pf_call, post):
                arr[i, j] = arr[a, b]
    return dict(players_grid=np.array(players, np.int32), stack_grid=np.array(stacks, np.int32),
                choose=choose, pf_open=pf_open, pf_call=pf_call, posterior=post,
                visits_choose=v_ch, visits_open=v_open, visits_call=v_call, iters=iters, solved=have,
                choose_actions=np.array(common.CHOOSE_ACTIONS, np.int8),
                equity_source=np.array(EQ_SRC))


def write_report(results, path, wall, quick_note=""):
    rs = sorted(results, key=lambda r: (r["n"], r["S"]))
    for r in rs:
        if "stats" not in r and "ch" in r:
            r["stats"] = stats_of(np.asarray(r["ch"], np.float64))
    L = ["# KRD CFR report", "",
         f"Generated by `solve_cfr.py`. Equity source: `{EQ_SRC}`. Wall-clock {wall/60:.1f} min. {quick_note}", "",
         "## Method", "",
         "- Abstract game per cell (n players, equal stacks S bb, blinds 0.5/1, chip EV in bb):"
         " 3-card deal -> simultaneous choose (1755 classes x 6 actions) -> push/fold with all reveals public.",
         "- Opener infoset (pos, kept169, shown_is_low, max revealed rank behind); caller infoset"
         " (pos, kept169, shown_is_low, jammer's revealed rank). BB never opens (walk).",
         "- Mini-batch MCCFR with DCFR discounting (alpha 1.5, beta 0, gamma 2). Per iteration: B_bet sampled"
         " deals with choose actions sampled from the current strategy; the push/fold tree is evaluated in exact"
         " expectation (every jammer x every caller subset) giving counterfactual values for all seats' betting"
         " infosets; on a B_ch sub-batch every seat's 6 choose actions are evaluated with the others fixed"
         " (external sampling of the choose layer).",
         "- Showdown: hu_eq heads-up, normalised product of pairwise equities multiway (no card removal between hands).",
         "",
         "## Convergence (bb per hand per seat)", "",
         "- `BR pf` = gain of a best-responding seat that changes only its push/fold (choose fixed), averaged over"
         " seats; max seat in brackets. Cross-validated (BR picked on one sample, measured on another) so"
         " sampling noise does not inflate it; in-sample (upper-biased) value after the slash.",
         "- `BR choose` = gain of a seat best-responding in the choose layer only (its push/fold fixed), CV / in-sample.",
         "- `loss lowR` = EV a single seat loses by always revealing the LOWER of its two kept cards (same hand"
         " selection as equilibrium), vs equilibrium opponents who keep their strategy; `lowR expl` = same but every"
         " opponent best-responds in push/fold to that leak. `hiR` = always reveal the higher card.",
         "- `loss dflt` = loss from the fallback policy (best-equity hand, 50/50 reveal).", "",
         "| n | S | iters | deals | solve s | BR pf CV / in | max seat | BR choose CV / in | loss lowR | lowR expl | loss hiR | hiR expl | loss dflt |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        if "br_pf_cv" not in r:
            continue
        g = np.array(r["br_pf_cv"]); gi = np.array(r["br_pf_insample"])
        L.append(f"| {r['n']} | {r['S']} | {r['iters']} | {r['deals']} | {r['secs']:.0f} | {g.mean():.4f} / {gi.mean():.4f} | {g.max():.4f} |"
                 f" {r['br_choose_cv']:.4f} / {r['br_choose_insample']:.4f} | {r['loss_reveal_lower']:.4f} |"
                 f" {r['loss_reveal_lower_exploited']:.4f} | {r['loss_reveal_higher']:.4f} | {r['loss_reveal_higher_exploited']:.4f} | {r['loss_default']:.4f} |")
    L += ["", "## Choose-policy statistics (deal-frequency weighted)", ""]
    keys = None
    for r in rs:
        if "stats" in r:
            keys = list(r["stats"].keys())
            break
    if keys:
        r0 = next(r for r in rs if "stats" in r)
        scal = [k for k in keys if not isinstance(r0["stats"][k], list)]
        L.append("| n | S | " + " | ".join(k.replace("|", "given") for k in scal) + " |")
        L.append("|---|---|" + "---|" * len(scal))
        for r in rs:
            if "stats" in r:
                L.append(f"| {r['n']} | {r['S']} | " + " | ".join(f"{r['stats'][k]:.3f}" for k in scal) + " |")
        L.append("")
        for r in rs:
            if "stats" in r and r["S"] in (8, 40) and r["n"] in (2, 6, 8):
                for k in keys:
                    if isinstance(r["stats"][k], list):
                        L.append(f"- n={r['n']} S={r['S']}: {k}: {r['stats'][k]}")
    path_tmp = path
    with open(path_tmp, "w") as f:
        f.write("\n".join(L) + "\n")
    return L


def run_grid(players, stacks, secs_fn, out, workers, diag_fn, report=None, note="", skip_done=False):
    t0 = time.time()
    jobs = [(n, S, secs_fn(n, S), 1000 * n + S, diag_fn(n, S)) for n in players for S in stacks
            if not (skip_done and os.path.exists(os.path.join(CELL_DIR, f"n{n}_S{S}.npz")))]
    jobs.sort(key=lambda j: -(j[2] + (90 if j[4] else 0)))
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(solve_cell, jobs, chunksize=1):
            pass
    results = load_cells(players, stacks)
    data = pack(results, players, stacks)
    np.savez_compressed(out, **data)
    wall = time.time() - t0
    print(f"wrote {out} in {wall/60:.1f} min ({len(results)} cells)", flush=True)
    if report:
        write_report(results, report, wall, note)
        with open(os.path.splitext(report)[0] + "_raw.json", "w") as f:
            json.dump([{k: v for k, v in r.items() if k not in ARR_KEYS} for r in results], f, indent=1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell", nargs=2, type=int)
    ap.add_argument("--secs", type=float, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip cells already saved in cells/")
    ap.add_argument("--report-from-raw", action="store_true", help="rebuild CFR_REPORT.md from CFR_REPORT_raw.json")
    a = ap.parse_args()
    if a.report_from_raw:
        results = load_cells(PLAYERS_GRID, STACK_GRID)
        wall = sum(r["total_secs"] for r in results) / max(1, a.workers)
        write_report(results, os.path.join(HERE, "CFR_REPORT.md"), wall, "(wall-clock estimated)")
        return
    if a.cell:
        r = solve_cell((a.cell[0], a.cell[1], a.secs or 30, 1, a.diag))
        for k, v in r.items():
            if not isinstance(v, np.ndarray):
                print(k, v)
        return
    if a.quick:
        secs = a.secs or 35
        run_grid(PLAYERS_GRID, STACK_GRID, lambda n, S: secs, os.path.join(HERE, "cfr_tables_quick.npz"),
                 a.workers, lambda n, S: False)
    else:
        base = a.secs or 80
        run_grid(PLAYERS_GRID, STACK_GRID,
                 lambda n, S: base * (0.55 if n <= 3 else 0.85 if n <= 5 else 1.05),
                 os.path.join(HERE, "cfr_tables.npz"), a.workers,
                 lambda n, S: (n in (2, 6) and S in (8, 40)),
                 os.path.join(HERE, "CFR_REPORT.md"), skip_done=a.resume)


if __name__ == "__main__":
    main()
