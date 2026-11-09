"""Tournament-value (T-value) layer for the KRD bot.  Self-contained: numpy + stdlib.

Hero's expected tournament points (payouts 50/30/20) for candidate post-hand stack vectors,
with the hand-cap horizon modelled.  Two cap-settlement modes:
  "icm"  : survivors at the cap are paid by Malmuth-Harville ICM (the real engine rule)
  "rank" : survivors at the cap are paid by chip rank (ties split)  -- hypothetical variant

Public API
----------
icm_batch(stacks[K,n], payouts)                 -> [K,n]  exact engine icm(), vectorised
rank_payout_batch(stacks[K,n], payouts)         -> [K,n]  chip-rank payout, ties split
sched_from(hand_number, ...)                    -> [H,2]  (sb,bb) of the hands AFTER hand_number
value_rows(rows[K,n], hero, hands_left, sched, mode, bb, start=None, hero_sigma=1) -> [K]
jam_ev(stacks, contrib, hero, pot, callers, hands_left, sched, mode, bb) -> (ev_jam, ev_fold)
call_ev(stacks, contrib, hero, pot, jammer, eq, hands_left, sched, mode, bb) -> (ev_call, ev_fold)
call_breakeven(...)                              -> equity needed to call a jam
risk_premium(stacks, hero, villain, amount, hands_left, sched, mode, bb) -> e* - 0.5
set_mc_samples(n)                                -> rebuild the fixed MC noise (analysis only)

Conventions
-----------
stacks  : chips BEHIND (not yet committed this hand), one entry per seat, 0 = busted/all-in.
contrib : chips already committed this hand per seat.   pot = total chips in the middle.
hands_left : hands that will still be dealt AFTER the current one (0 = this is the last hand).
sched   : None (constant blinds bb/2, bb) or array-like [>=hands_left, 2] of (sb, bb) for the
          remaining hands (use sched_from(hand_number)).

Horizon model (see ENDGAME_ANALYSIS.md for justification)
----------------------------------------------------------
hands_left == 0 : exact settlement (icm or rank), with place points for anyone busted.
mode "icm"      : plain ICM for every hands_left > 0 (cap settlement is ICM itself, and chips
                  are ~a martingale, so E[ICM(stacks at cap)] ~ ICM(stacks now)).  The MC
                  below can be forced with ICM_MC=True (analysis only).
mode "rank"     : Monte Carlo over a fixed, import-time pre-drawn antithetic noise matrix
                  (N_MC samples, <= NSTEP steps), blended into plain ICM as the horizon grows:
  * per step, alive player i gets a Gaussian increment with per-hand std
    min(KAPPA*bb_h, EFF_C*E_i), E_i = min(s_i, mean live opponent stack) (you can only lose
    what is covered); increments are centred over live players (chips are conserved);
  * a stack is absorbed (busts) if the step ends <= 0 or the Brownian bridge inside the step
    crosses 0 (prob exp(-2 s0 s1 / sigma^2), pre-drawn uniforms);
  * busts place below survivors (earlier / deeper = lower); survivors are ranked by chips
    with a soft rank (width SOFT_TAU * avg stack) so the value is smooth in the stacks;
  * optional blind-steal drift (STEAL, default 0);
  * ICM weight w = max(ramp(nu; NU_LO, NU_HI), ramp(H; H_BLEND_LO, H_FULL_ICM)) where
    nu = sum_h bb_h^2 / avg_stack^2 is the horizon variance in units of an average stack.
"""
from __future__ import annotations

import numpy as np

PAYOUTS = (50.0, 30.0, 20.0)
DEFAULT_BLINDS = ((10, 20), (15, 30), (25, 50), (50, 100), (75, 150), (100, 200),
                  (150, 300), (200, 400), (300, 600), (400, 800), (600, 1200), (1000, 2000))
HANDS_PER_LEVEL = 20
HAND_CAP = 250

CAP_MODE = "icm"          # "icm" (engine rule) or "rank"
KAPPA = 3.0               # deep-stack per-hand per-player stack std, in big blinds
EFF_C = 0.7               # shallow cap: per-hand std <= EFF_C * mean effective stack
N_MC = 32                 # MC samples (antithetic pairs), fixed at import
ICM_MC = False            # icm mode: False = plain ICM for every horizon (default), True = MC
NU_LO = 1.0               # relative horizon variance where blending towards ICM starts
NU_HI = 4.0               # ... and where plain ICM takes over
H_BLEND_LO = 30           # hands_left where blending towards ICM starts
H_FULL_ICM = 60           # hands_left from which plain ICM is used
STEAL = 0.0              # strength of the big-stack blind-steal drift (0 = no drift)
SOFT_TAU = 0.05           # rank-mode soft-rank width, fraction of the average stack

NSTEP = 2                 # max MC steps (sigma recomputed from current stacks); 1 if H <= 4


def _lhs_uniform(rng, shape):
    """Latin-hypercube uniforms: stratified along the sample axis (-2) for every column."""
    u = rng.random(shape)
    return (np.argsort(rng.random(shape), axis=-2) + u) / shape[-2]


def _norm_ppf(u):
    """Acklam's inverse normal CDF (|rel err| < 1.2e-9); numpy only, used at import."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    q = np.minimum(u, 1 - u)                             # lower-tail probability
    lo = q < 0.02425
    r = np.sqrt(-2 * np.log(np.where(lo, q, 0.5)))
    x_lo = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
        ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    s = q - 0.5
    t = s * s
    x_c = (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * s / \
        (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1)
    x = np.where(lo, x_lo, x_c)                          # quantile of q (<= 0) for q <= 0.5
    return np.where(u < 0.5, x, -x)


def set_mc_samples(n=32, seed=20261109):
    """(Re)build the fixed noise: antithetic LHS normals [NSTEP,n,8], bridge draws, jitter.
    Runtime uses the import-time default; analysis can call this with a large n."""
    global N_MC, _Z, _LOGU, _V
    N_MC = int(n) // 2 * 2
    rng = np.random.default_rng(seed)
    zh = _norm_ppf(_lhs_uniform(rng, (NSTEP, N_MC // 2, 8)))
    _Z = np.concatenate([zh, -zh], 1)
    _LOGU = np.log(_lhs_uniform(rng, (NSTEP, N_MC, 8)))
    _V = _lhs_uniform(rng, (N_MC, 8)) * 0.5 + 0.01


set_mc_samples(N_MC)
_AR = np.arange(9.0)


def _pay3(payouts):
    p = np.zeros(3)
    p[:min(3, len(payouts))] = payouts[:3]
    return p


def _pay_pad(payouts, n):
    p = np.zeros(n + 1)
    m = min(n, len(payouts))
    p[:m] = payouts[:m]
    return p


# ----------------------------------------------------------------------------------- ICM
_OFF = {k: 1.0 - np.eye(k) for k in range(1, 9)}        # off-diagonal float masks
_EPS = 1e-9


def icm_batch(stacks, payouts=PAYOUTS):
    """Malmuth-Harville ICM for up to 3 paid places, vectorised over rows.
    Matches poker_engine.table.icm() (zero stacks excluded, unpaid places beyond #alive)."""
    S = np.asarray(stacks, dtype=np.float64)
    if S.ndim == 1:
        return icm_batch(S[None], payouts)[0]
    K, n = S.shape
    p1, p2, p3 = _pay3(payouts)
    T = np.add.reduce(S, 1)[:, None]                     # [K,1]
    Tj = T - S                                           # T - s_j
    a = S / (np.maximum(T, _EPS) * np.maximum(Tj, _EPS))  # s_j/(T(T-s_j)); huge only if s_i=0 elsewhere
    out = S * (p1 / np.maximum(T, _EPS) + p2 * (np.add.reduce(a, 1)[:, None] - a))
    if p3 != 0.0 and n >= 3:
        D = Tj[:, :, None] - S[:, None, :]               # T - s_j - s_k
        M = (a[:, :, None] * S[:, None, :]) / np.maximum(D, _EPS) * ((D > _EPS) * _OFF[n])
        Mt = np.add.reduce(M.reshape(K, n * n), 1)[:, None]
        out += (p3 * S) * (Mt - np.add.reduce(M, 2) - np.add.reduce(M, 1))
    return out


def _icm_hero(S, h, payouts):
    """Hero column of icm_batch only (S [R,n] float64, no cancellation)."""
    p1, p2, p3 = _pay3(payouts)
    R, n = S.shape
    T = np.add.reduce(S, 1)
    sh = S[:, h]
    Tj = T[:, None] - S
    Tj[:, h] = np.inf                                    # excludes j == hero
    a = S / (np.maximum(T, _EPS)[:, None] * np.maximum(Tj, _EPS))
    v = (p1 / np.maximum(T, _EPS) + p2 * np.add.reduce(a, 1)) * sh
    if p3 != 0.0 and n >= 3:
        Sk = S * _OFF[n][h]                              # excludes k == hero
        D = Tj[:, :, None] - Sk[:, None, :]              # T - s_j - s_k  (>= s_h)
        B = Sk[:, None, :] / np.maximum(D, _EPS) * _OFF[n]
        v += (p3 * sh) * np.add.reduce(a * np.add.reduce(B, 2), 1)
    return v


# ----------------------------------------------------------------------------------- rank
def rank_payout_batch(stacks, payouts=PAYOUTS):
    """Chip-rank payout among positive stacks; ties split the tied places' payouts."""
    S = np.asarray(stacks, dtype=np.float64)
    if S.ndim == 1:
        return rank_payout_batch(S[None], payouts)[0]
    K, n = S.shape
    pay = _pay_pad(payouts, n)
    C = np.concatenate([[0.0], np.cumsum(pay)])           # C[k] = sum pay[:k]
    alive = S > 0
    gt = ((S[:, None, :] > S[:, :, None]) & alive[:, None, :]).sum(2)   # #better
    eq = ((S[:, None, :] == S[:, :, None]) & alive[:, None, :]).sum(2)  # #tied incl self
    eq = np.maximum(eq, 1)
    hi = np.minimum(gt + eq, n)
    v = (C[hi] - C[gt]) / eq
    return np.where(alive, v, 0.0)


# ----------------------------------------------------------------------------------- schedule
def sched_from(hand_number, blinds=DEFAULT_BLINDS, hands_per_level=HANDS_PER_LEVEL,
               hand_cap=HAND_CAP):
    """(sb,bb) for hands hand_number+1 .. hand_cap (hand_number = current, 1-based)."""
    hs = np.arange(hand_number + 1, hand_cap + 1)
    lv = np.minimum((hs - 1) // hands_per_level, len(blinds) - 1)
    return np.asarray(blinds, dtype=np.float64)[lv].reshape(-1, 2)


def hands_left_from(hand_number, hand_cap=HAND_CAP):
    return max(0, hand_cap - hand_number)


# ----------------------------------------------------------------------------------- value
def _settle(rows, hero, mode, pay_t):
    if mode == "rank":
        return rank_payout_batch(rows, pay_t)[:, hero]
    return _icm_hero(rows, hero, pay_t)


def _horizon(H, sched, bb):
    """Split the next H hands into chunks (1 if H <= 4 else NSTEP):
    (hands[c], rms bb[c], blinds sb+bb[c])."""
    steps = 1 if H <= 4 else min(H, NSTEP)
    cuts = [round(H * k / steps) for k in range(steps + 1)]
    nh = [float(cuts[k + 1] - cuts[k]) for k in range(steps)]
    if sched is None:
        return nh, [float(bb)] * steps, [x * 1.5 * float(bb) for x in nh]
    sb_bb = np.asarray(sched, dtype=np.float64).reshape(-1, 2)[:H]
    if len(sb_bb) < H:                                   # pad with last level
        sb_bb = np.vstack([sb_bb, np.repeat(sb_bb[-1:], H - len(sb_bb), 0)])
    rms, bl = [], []
    for k in range(steps):
        ch = sb_bb[cuts[k]:cuts[k + 1]]
        rms.append(float(np.sqrt((ch[:, 1] ** 2).mean())))
        bl.append(float(ch.sum()))
    return nh, rms, bl


def _mc(rows, hero, H, hz, mode, pay_t, hero_sigma=1.0):
    """Multi-step MC over fixed noise: Gaussian increments with stack-dependent std,
    zero-sum across live players, exact Brownian-bridge absorption inside each step."""
    nh, rms, blinds = hz
    red = np.add.reduce
    alive = rows > 0
    cols = np.nonzero(np.logical_or.reduce(alive, 0))[0]  # compress to live columns
    c8 = cols & 7
    h = int(np.searchsorted(cols, hero))
    R0 = rows[:, cols]
    K, nc = R0.shape
    T = red(R0, 1)[:, None, None]
    S = np.repeat(R0[:, None, :], N_MC, 1)               # [K,N,nc]
    live = S > 0
    lf = live.astype(np.float64)
    key = np.where(live, 0.0, -np.inf)
    hs = np.ones(nc)
    hs[h] = hero_sigma
    nst = len(nh)
    for t in range(nst):
        m = np.maximum(red(lf, 2)[:, :, None], 1.0)
        m1 = np.maximum(m - 1.0, 1.0)
        # per-hand std: min(KAPPA*bb, EFF_C * min(own stack, average live opponent stack))
        E = np.minimum(S, (red(S, 2)[:, :, None] - S) / m1)
        sig = np.minimum(KAPPA * rms[t], EFF_C * E) * (np.sqrt(nh[t]) * hs) + 1e-6
        z = _Z[t][:, c8] * lf
        z -= red(z, 2)[:, :, None] / m
        S1 = S + z * (np.sqrt(m / m1) * sig)
        if STEAL:
            S1 += (STEAL * blinds[t] / m) * (m * S / T - 1.0) * lf
        # absorbed: ends <= 0 or the Brownian bridge crosses 0  (U < exp(-2 s0 s1+ / sig^2))
        newly = live & (_LOGU[t][:, c8] * (sig * sig) < -2.0 * S * np.maximum(S1, 0.0))
        # bust key: earlier step = lower; within a step, deeper terminal = lower
        key = np.where(newly, np.minimum(S1, 0.0) - (nst - t) * T - sig * _V[:, c8], key)
        live &= ~newly
        lf = live.astype(np.float64)
        S = S1 * lf
    key = np.where(live, S, key)
    kh = key[:, :, h:h + 1]                              # [K,N,1]
    pay = _pay_pad(pay_t, nc)
    if mode == "rank":
        # soft count of players ranked above hero (self contributes 0), width SOFT_TAU*avg
        itau = (0.5 / SOFT_TAU) * red(alive, 1)[:, None, None] / T
        g = 0.5 * red(np.tanh((key - kh) * itau), 2) + 0.5 * (nc - 1)
        return red(np.interp(g, _AR[:nc + 1], pay), 1) / N_MC
    v_live = _icm_hero(S.reshape(K * N_MC, nc), h, pay_t).reshape(K, N_MC)
    g = red(key > kh, 2)
    return red(np.where(live[:, :, h], v_live, pay[g]), 1) / N_MC


def _icm_weight(R, H, hz):
    """Weight of plain ICM in the blend: grows with the horizon (hands) and with the
    relative horizon variance nu = sum bb^2 / (avg stack)^2 (how many 'average all-ins' fit)."""
    nh, rms, _ = hz
    m = np.maximum(np.add.reduce(R > 0, 1), 1)
    avg = np.add.reduce(R, 1) / m
    nu = sum(a * b * b for a, b in zip(nh, rms)) / (avg * avg)
    w_nu = np.minimum(np.maximum((nu - NU_LO) * (1.0 / (NU_HI - NU_LO)), 0.0), 1.0)
    w_h = min(max((H - H_BLEND_LO) / float(H_FULL_ICM - H_BLEND_LO), 0.0), 1.0)
    return np.maximum(w_nu, w_h)


def value_rows(rows, hero, hands_left, sched=None, mode=None, bb=2000.0, start=None,
               payouts=PAYOUTS, hero_sigma=1.0):
    """Hero's expected tournament points for each candidate post-hand stack vector.
    rows  : [K,n] post-hand stacks (0 = busted, either before or during this hand)
    start : optional [n] stacks at the start of this hand, to order same-hand busts
            (larger start stack places higher, as in the engine).
    hero_sigma : scale of hero's own future variance (1 = plays like everyone, 0 = hero folds
            to the cap; then pass hero's stack already reduced by the blinds he must post)."""
    mode = mode or CAP_MODE
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None]
    hero_alive = rows[:, hero] > 0
    if hero_alive.all():                                 # common case: no fancy indexing
        return _value_alive(rows, hero, hands_left, sched, mode, bb, payouts, hero_sigma)
    K, n = rows.shape
    pay = _pay_pad(payouts, n)
    # hero busts this hand: place = (#alive after) + 1 (+ co-busters with bigger start)
    g = (rows > 0).sum(1)
    if start is not None:
        st = np.asarray(start, dtype=np.float64)
        g = g + ((rows == 0) & (st[None, :] > st[hero])).sum(1)
    out = pay[np.minimum(g, n)]
    idx = np.nonzero(hero_alive)[0]
    if len(idx):
        out[idx] = _value_alive(rows[idx], hero, hands_left, sched, mode, bb, payouts,
                                hero_sigma)
    return out


def _value_alive(R, hero, hands_left, sched, mode, bb, payouts, hero_sigma):
    """value_rows for rows where hero still has chips (hero alone -> 1st, via the formulas)."""
    if hands_left <= 0:
        return _settle(R, hero, mode, payouts)
    if (mode != "rank" and not ICM_MC and hero_sigma == 1.0) or hands_left >= H_FULL_ICM:
        return _icm_hero(R, hero, payouts)
    H = int(hands_left)
    hz = _horizon(H, sched, bb)
    w = _icm_weight(R, H, hz)
    v = _icm_hero(R, hero, payouts)
    mc = w < 1.0
    if mc.all():
        return (1 - w) * _mc(R, hero, H, hz, mode, payouts, hero_sigma) + w * v
    if mc.any():
        v[mc] = (1 - w[mc]) * _mc(R[mc], hero, H, hz, mode, payouts, hero_sigma) \
            + w[mc] * v[mc]
    return v


# ----------------------------------------------------------------------------------- spots
def _showdown_rows(base, hero, j, Ah, Aj, dead):
    """post-hand rows when hero (total commit Ah) and j (total Aj) go to showdown."""
    M = min(Ah, Aj)
    win = base.copy()
    lose = base.copy()
    win[hero] = Ah + M + dead
    win[j] = Aj - M
    lose[hero] = Ah - M
    lose[j] = Aj + M + dead
    return win, lose


def _default_fold_winner(contrib, hero):
    c = np.array(contrib, dtype=np.float64)
    c[hero] = -1.0
    return int(np.argmax(c))


def jam_ev(stacks, contrib, hero, pot, callers, hands_left, sched=None, mode=None, bb=2000.0,
           fold_winner=None, payouts=PAYOUTS):
    """EV (tournament points) of hero jamming vs folding.
    callers : list of (seat, p_call, eq_hero_vs_seat) in action order; at most one caller
              (seat i gets to call only if all earlier ones folded).
    fold_winner : seat that collects the pot if hero folds (default: largest contributor)."""
    s = np.asarray(stacks, dtype=np.float64)
    c = np.asarray(contrib, dtype=np.float64)
    start = s + c
    Ah = s[hero] + c[hero]
    rows = []
    fw = _default_fold_winner(c, hero) if fold_winner is None else fold_winner
    fold = s.copy()
    fold[fw] += pot
    rows.append(fold)
    allfold = s.copy()
    allfold[hero] = s[hero] + pot
    rows.append(allfold)
    probs = []
    reach = 1.0
    for (j, p, eq) in callers:
        Aj = s[j] + c[j]
        dead = pot - c[hero] - c[j]
        base = s.copy()
        win, lose = _showdown_rows(base, hero, j, Ah, Aj, dead)
        rows.append(win)
        rows.append(lose)
        probs.append((reach * p, eq))
        reach *= (1.0 - p)
    V = value_rows(np.array(rows), hero, hands_left, sched, mode, bb, start, payouts)
    ev = reach * V[1]
    for k, (pr, eq) in enumerate(probs):
        ev += pr * (eq * V[2 + 2 * k] + (1 - eq) * V[3 + 2 * k])
    return float(ev), float(V[0])


def _call_vals(stacks, contrib, hero, pot, jammer, hands_left, sched, mode, bb, payouts):
    s = np.asarray(stacks, dtype=np.float64)
    c = np.asarray(contrib, dtype=np.float64)
    start = s + c
    Ah = s[hero] + c[hero]
    Aj = s[jammer] + c[jammer]
    dead = pot - c[hero] - c[jammer]
    win, lose = _showdown_rows(s.copy(), hero, jammer, Ah, Aj, dead)
    fold = s.copy()
    fold[jammer] += pot
    V = value_rows(np.array([win, lose, fold]), hero, hands_left, sched, mode, bb, start, payouts)
    return V


def call_ev(stacks, contrib, hero, pot, jammer, eq, hands_left, sched=None, mode=None,
            bb=2000.0, payouts=PAYOUTS):
    """EV of calling a jam (no players left behind) with equity eq, vs folding."""
    Vw, Vl, Vf = _call_vals(stacks, contrib, hero, pot, jammer, hands_left, sched, mode, bb,
                            payouts)
    return float(eq * Vw + (1 - eq) * Vl), float(Vf)


def call_breakeven(stacks, contrib, hero, pot, jammer, hands_left, sched=None, mode=None,
                   bb=2000.0, payouts=PAYOUTS):
    """Equity hero needs to call the jam (may be <0 or >1)."""
    Vw, Vl, Vf = _call_vals(stacks, contrib, hero, pot, jammer, hands_left, sched, mode, bb,
                            payouts)
    d = Vw - Vl
    if d <= 1e-12:                                       # result does not depend on equity
        return 0.0 if Vl >= Vf - 1e-12 else 1.0
    return float((Vf - Vl) / d)


def risk_premium(stacks, hero, villain, amount, hands_left, sched=None, mode=None, bb=2000.0,
                 dead=0.0, payouts=PAYOUTS):
    """Break-even equity minus 0.5 for hero risking `amount` against villain (each puts in
    `amount`, plus `dead` money hero wins too).  Positive = hero needs more than a flip."""
    s = np.asarray(stacks, dtype=np.float64)
    amt = min(float(amount), s[hero], s[villain])
    w = s.copy()
    l = s.copy()
    w[hero] += amt + dead
    w[villain] -= amt
    l[hero] -= amt
    l[villain] += amt + dead
    V = value_rows(np.array([w, l, s]), hero, hands_left, sched, mode, bb, s, payouts)
    d = V[0] - V[1]
    if d <= 1e-12:
        e = 0.0 if V[1] >= V[2] - 1e-12 else 1.0
    else:
        e = (V[2] - V[1]) / d
    return float(e - 0.5)
