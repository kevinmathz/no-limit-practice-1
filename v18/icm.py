"""ICM layer for v2: bust avoidance + late-game aggression. Pure stdlib, stateless.

Every decision that risks our tournament life is scored in points (Malmuth-Harville ICM over
state.config.payouts), not chips: commit only if
    p * ICM(win) + (1 - p) * ICM(lose) > ICM(fold) + margin.
Equity p comes from offline tables (equity169.json preflop, equity_post.json per v1 bucket/street,
both vs one random hand), adjusted for the number of opponents and tightened against bets.
"""
import json
import os

from poker_agent_sdk import Action

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "equity169.json")) as _f:
    EQ_PRE = json.load(_f)
with open(os.path.join(_HERE, "equity_post.json")) as _f:
    EQ_POST = {int(k): v for k, v in json.load(_f).items()}

# ---------------- tunables ----------------
LIFE_FRAC = 0.15        # a commit is "life" if losing leaves < 15% of our stack at hand start
MULTIWAY_ALPHA = 1.0    # p = eq ** (1 + ALPHA * (n_opps - 1))
TIGHT_PRE = 0.5         # vs a bet / when called preflop: p = p ** (1 + TIGHT)
TIGHT_POST = 1.2        # same, postflop (betting ranges are much stronger than random)
MARGIN = {2: 0.0, 3: 1.0, 4: 1.5, 5: 1.5, 6: 0.8, 7: 0.5, 8: 0.5}  # points per 100 paid out
LEADER_EXTRA = 3.0      # extra margin for the chip leader near the cap
FE_UNOPENED = 0.75      # per-opponent fold probability vs our preflop shove into no raise
FE_VS_BET = 0.40        # per-opponent fold probability when we shove over a bet
PUSH_BB = 10            # short-stack push/fold threshold (big blinds), any time
AGG_WINDOW = 25         # late-game aggression when hand_cap - hand_number <= this and not leader
AGG_SHOVE_BB = 15       # in aggression mode, shove instead of open when stack <= this
AGG_OPEN_BUCKET = 3     # preflop bucket (0..7) to open/raise unopened pots in aggression mode
AGG_3BET_BUCKET = 5     # preflop bucket to raise instead of call facing a raise
AGG_POST_BUCKET = 3     # postflop bucket (0..8) to bet/raise instead of check/call


# ---------------- ICM ----------------
def icm(stacks, payouts):
    """Full Malmuth-Harville: expected payout per index (used by tests)."""
    return [my_icm(stacks, i, payouts) for i in range(len(stacks))]


def my_icm(stacks, idx, payouts):
    """Expected payout of stacks[idx]. Zero stacks are treated as already busted (bottom places)."""
    if stacks[idx] <= 0:
        return 0.0
    live = [i for i, s in enumerate(stacks) if s > 0]
    npay = min(len(payouts), len(live))
    mine = stacks[idx]
    ev = 0.0

    def rec(rem_total, used, prob, place):
        nonlocal ev
        ev += prob * mine / rem_total * payouts[place]
        if place + 1 >= npay:
            return
        for j in live:
            if j == idx or j in used:
                continue
            sj = stacks[j]
            rec(rem_total - sj, used + (j,), prob * sj / rem_total, place + 1)

    rec(float(sum(stacks[i] for i in live)), (), 1.0, 0)
    return ev


def _outcome_ev(stacks, idx, payouts, n_alive):
    if stacks[idx] <= 0:  # we bust: we take the next bust place
        k = n_alive - 1
        return float(payouts[k]) if k < len(payouts) else 0.0
    return my_icm(stacks, idx, payouts)


# ---------------- helpers ----------------
def contributions(state):
    """Chips each seat has put in this hand (from the hand history)."""
    contrib = {}
    street = {}
    for e in state.history:
        k = e.kind
        if k == "board":
            street = {}
        elif k in ("blind", "ante", "call"):
            a = e.amount or 0
            contrib[e.seat] = contrib.get(e.seat, 0) + a
            if k != "ante":
                street[e.seat] = street.get(e.seat, 0) + a
        elif k == "raise":
            prev = street.get(e.seat, 0)
            contrib[e.seat] = contrib.get(e.seat, 0) + (e.amount or 0) - prev
            street[e.seat] = e.amount or 0
    return contrib


def _clamp_raise(r, total):
    return Action.raise_to(int(max(r.min_amount, min(r.max_amount, int(total)))))


def _passive(state):
    return Action.check() if state.can("check") else Action.fold()


# ---------------- main entry ----------------
def adjust(state, action, me, street, bucket, hclass):
    """Return the action to play: `action` (MCCFR sample), a vetoed passive action, or an upgrade."""
    payouts = state.config.payouts
    if not payouts:
        return action
    scale = sum(payouts) / 100.0
    bb = max(state.blinds[1], 1)
    seats = [s for s in state.seats if s.status != "busted"]
    n_alive = len(seats)
    contrib = contributions(state)
    start = {s.seat: s.stack + contrib.get(s.seat, 0) for s in seats}
    my_start = start.get(state.seat, me.stack)
    others_max = max((v for k, v in start.items() if k != state.seat), default=0)
    leader = my_start >= others_max
    hands_left = state.config.hand_cap - state.hand_number
    late = hands_left <= AGG_WINDOW
    agg = late and not leader
    margin = 0.0 if agg else MARGIN.get(n_alive, 0.5) * scale
    if late and leader:
        margin += LEADER_EXTRA * scale

    opps = [s for s in state.seats if s.seat != state.seat and s.status in ("active", "all_in")]
    if not opps:
        return action
    r = state.can("raise")
    to_call = state.to_call
    cur_max = max(s.bet for s in state.seats)
    eq = EQ_PRE.get(hclass, 0.5) if street == 0 else EQ_POST[street][bucket]
    ctx = (state, me, seats, n_alive, payouts, opps, eq, street, contrib)

    # ---- upgrades: short-stack push/fold, late-game aggression ----
    cand = action
    kind = action.kind
    unopened = street == 0 and cur_max <= bb
    if r and kind in ("fold", "check", "call"):
        stack_bb = (me.stack + me.bet) / bb
        if unopened and (stack_bb <= PUSH_BB or (agg and stack_bb <= AGG_SHOVE_BB)):
            shove = Action.raise_to(int(r.max_amount))
            if _commit_ev(ctx, shove, r) > _base_ev(ctx, kind) + margin:
                return shove
        if agg:
            pot_total = me.bet + to_call + (state.pot + to_call)
            if street == 0:
                if unopened and bucket >= AGG_OPEN_BUCKET:
                    cand = _clamp_raise(r, pot_total)
                elif kind == "call" and bucket >= AGG_3BET_BUCKET:
                    cand = _clamp_raise(r, pot_total)
            elif bucket >= AGG_POST_BUCKET:
                if kind == "check":
                    cand = _clamp_raise(r, me.bet + 0.5 * state.pot)
                elif kind == "call":
                    cand = _clamp_raise(r, pot_total)

    # ---- veto of life-risking commits ----
    for a in ((cand, action) if cand is not action else (action,)):
        if a.kind not in ("call", "raise"):
            return a
        if not _is_life(ctx, a):
            return a
        if _commit_ev(ctx, a, r) > _base_ev(ctx, "fold" if to_call > 0 else "check") + margin:
            return a
    return _passive(state)


def _villain(opps):
    return max(opps, key=lambda s: (s.bet, s.stack))


def _added(state, me, a):
    if a.kind == "call":
        return min(state.to_call, me.stack)
    if a.kind == "raise":
        return a.amount - me.bet
    return 0


def _is_life(ctx, a):
    state, me, seats, n_alive, payouts, opps, eq, street, contrib = ctx
    add = _added(state, me, a)
    if add <= 0:
        return False
    my_start = me.stack + contrib.get(state.seat, 0)
    v = max(opps, key=lambda s: s.stack + s.bet)  # the one who can hurt us most
    new_bet = me.bet + add
    refund = max(0, new_bet - (v.bet + v.stack))
    loss_stack = me.stack - add + refund
    return loss_stack < LIFE_FRAC * my_start


def _stacks(ctx):
    state, me, seats, n_alive, payouts, opps, eq, street, contrib = ctx
    st = []
    idx = vidx = 0
    v = _villain(opps)
    for s in seats:
        if s.seat == state.seat:
            idx = len(st)
        if s.seat == v.seat:
            vidx = len(st)
        x = s.stack
        if x == 0 and s.status == "all_in" and s.seat != v.seat:
            x = contrib.get(s.seat, 0)  # can still win this pot; keep them alive in the model
        st.append(x)
    return st, idx, vidx, v


def _win_prob(ctx, n_opps, tightened):
    state, me, seats, n_alive, payouts, opps, eq, street, contrib = ctx
    p = eq ** (1.0 + MULTIWAY_ALPHA * max(0, n_opps - 1))
    if tightened:
        p = p ** (1.0 + (TIGHT_PRE if street == 0 else TIGHT_POST))
    return p


def _base_ev(ctx, kind):
    """Points if we don't commit: fold (pot to the villain) or check (showdown for the pot)."""
    state, me, seats, n_alive, payouts, opps, eq, street, contrib = ctx
    st, idx, vidx, v = _stacks(ctx)
    pot = state.pot
    if kind == "fold" or state.to_call > 0:
        st2 = list(st)
        st2[vidx] += pot
        return _outcome_ev(st2, idx, payouts, n_alive)
    p = _win_prob(ctx, len(opps), False)
    w = list(st)
    w[idx] += pot
    lose = list(st)
    lose[vidx] += pot
    return p * _outcome_ev(w, idx, payouts, n_alive) + (1 - p) * _outcome_ev(lose, idx, payouts, n_alive)


def _commit_ev(ctx, a, r):
    state, me, seats, n_alive, payouts, opps, eq, street, contrib = ctx
    st, idx, vidx, v = _stacks(ctx)
    add = _added(state, me, a)
    new_bet = me.bet + add
    refund = max(0, new_bet - (v.bet + v.stack))
    vextra = max(0, min(v.stack, new_bet - v.bet))
    pot = state.pot
    win = list(st)
    win[idx] += pot + vextra
    win[vidx] -= vextra
    lose = list(st)
    lose[idx] -= add - refund
    lose[vidx] += pot + add - refund
    raising = a.kind == "raise"
    n_called = 1 if raising else len(opps)
    p = _win_prob(ctx, n_called, True)
    showdown = p * _outcome_ev(win, idx, payouts, n_alive) + (1 - p) * _outcome_ev(lose, idx, payouts, n_alive)
    if not raising:
        return showdown
    responders = [s for s in opps if s.status == "active" and s.stack > 0]
    if not responders:
        return showdown
    unopened = street == 0 and max(s.bet for s in state.seats) <= max(state.blinds[1], 1)
    fe = (FE_UNOPENED if unopened else FE_VS_BET) ** len(responders)
    now = list(st)
    now[idx] += pot
    return fe * _outcome_ev(now, idx, payouts, n_alive) + (1 - fe) * showdown
