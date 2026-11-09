"""ICM layer for v3: Malmuth-Harville ICM, equity tables, tournament-life veto, push/fold, aggression mode.

Pure stdlib, stateless (tables are constants loaded at import). All money is in tournament points.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ tunables
LIFE_FRAC = 0.15          # a call/raise leaving us below this fraction of our chips is "tournament life"
PUSHFOLD_BB = 12          # preflop at or below this many big blinds: ICM-driven push/fold
AGG_WINDOW = 0  # v23: no late-game aggression (cap points are split by ICM, so gambling late is -EV)           # hands before hand_cap in which aggression mode can turn on
# bust-aversion margin (points) by players remaining; extra when we lead near the cap
BUST_MARGIN = {2: 0.0, 3: 0.8, 4: 1.5, 5: 1.5, 6: 0.8, 7: 0.4, 8: 0.4}
LEADER_EXTRA_MARGIN = 1.0
SHADE_PRE = 0.25          # p -> p**(1+SHADE): opponent who bets/calls an all-in has a stronger-than-random range
SHADE_POST = 0.30
MULTI_K = 0.8             # multiway: p -> p**(1 + MULTI_K*(n_opp-1))
CALL_FRAC = 0.22          # assumed fraction of hands an active opponent calls our open shove with
CALL_FRAC_FACING = 0.40   # ... when we shove over a bet (the bettor continues more often)
AGG_PUSH_SCORE = 22       # aggression mode: short-stack push score threshold (normal bot: 30)
AGG_OPEN_SCORE = 24       # aggression mode: raise (not limp/call) with at least this preflop score
AGG_BLUFF_PCT = 25        # aggression mode: % of checked-to spots with air where we bet 1/3 pot

with open(os.path.join(_HERE, "equity169.json")) as _f:
    EQ169 = json.load(_f)
with open(os.path.join(_HERE, "equity_post.json")) as _f:
    EQPOST = json.load(_f)  # EQPOST[street 0..2][bucket 0..8]

RANK = {r: i + 2 for i, r in enumerate("23456789TJQKA")}


# ------------------------------------------------------------------ ICM
def icm(stacks, payouts):
    """Malmuth-Harville expected payout per index."""
    n = len(stacks)
    k = min(len(payouts), n)
    ev = [0.0] * n
    tot = float(sum(stacks))
    if tot <= 0 or k == 0:
        return ev

    def rec(used, rem, place, prob):
        pay = payouts[place]
        last = place == k - 1
        for i in range(n):
            if used >> i & 1:
                continue
            s = stacks[i]
            if s <= 0:
                continue
            pi = prob * s / rem
            ev[i] += pi * pay
            if not last:
                rec(used | (1 << i), rem - s, place + 1, pi)

    rec(0, tot, 0, 1.0)
    return ev


# ------------------------------------------------------------------ hand abstraction (copied from v1/abstraction.py)
def hand_class(c1, c2):
    r1, r2 = RANK[c1[0]], RANK[c2[0]]
    if r1 < r2:
        c1, c2, r1, r2 = c2, c1, r2, r1
    if r1 == r2:
        return c1[0] + c2[0]
    return c1[0] + c2[0] + ("s" if c1[1] == c2[1] else "o")


def _has_straight(rankset):
    rs = set(rankset)
    if 14 in rs:
        rs.add(1)
    run = 0
    for r in range(1, 15):
        run = run + 1 if r in rs else 0
        if run >= 5:
            return True
    return False


def _straight_draw(rankset):
    rs = set(rankset)
    if 14 in rs:
        rs.add(1)
    for lo in range(1, 11):
        if sum((lo + k) in rs for k in range(5)) >= 4:
            return True
    return False


def postflop_bucket(hole, board):
    """0 nothing, 1 draw, 2 weak pair, 3 top pair weak kicker, 4 top pair good kicker/overpair,
    5 two pair, 6 trips/set, 7 straight/flush, 8 full house+."""
    hr = [RANK[c[0]] for c in hole]
    br = [RANK[c[0]] for c in board]
    allr = hr + br
    cnt = {}
    for r in allr:
        cnt[r] = cnt.get(r, 0) + 1
    bcnt = {}
    for r in br:
        bcnt[r] = bcnt.get(r, 0) + 1
    suits = {}
    for c in list(hole) + list(board):
        suits[c[1]] = suits.get(c[1], 0) + 1
    flush = any(n >= 5 and any(h[1] == s for h in hole) for s, n in suits.items())
    flush_draw = len(board) < 5 and any(n == 4 and any(h[1] == s for h in hole) for s, n in suits.items())
    hole_trips = [r for r in set(hr) if cnt[r] >= 3]
    hole_pairs = [r for r in set(hr) if cnt[r] >= 2]
    if any(cnt[r] >= 4 for r in set(hr)):
        return 8
    if hole_trips and any(cnt[r] >= 2 for r in cnt if r not in hole_trips):
        return 8
    if hole_pairs and any(bcnt.get(r, 0) >= 3 for r in bcnt):
        return 8
    straight = _has_straight(allr) and not _has_straight(br)
    if flush or straight:
        return 7
    if hole_trips:
        return 6
    if len(hole_pairs) >= 2:
        return 5
    if hole_pairs:
        top = max(br)
        p = max(hole_pairs)
        if hr[0] == hr[1]:
            return 4 if p > top else 2
        if p == top:
            kicker = hr[0] if hr[1] == p else hr[1]
            return 4 if kicker >= 10 else 3
        return 2
    if flush_draw or (len(board) < 5 and _straight_draw(allr) and not _straight_draw(br)):
        return 1
    return 0


# ------------------------------------------------------------------ situation
class Sit:
    __slots__ = ("me", "S", "mybet", "total", "live", "opps", "payouts", "aggr", "leader",
                 "margin", "n", "aggressor", "pot", "ids", "stacks", "my_i")


def situation(state):
    """Cheap per-decision summary (no ICM math yet)."""
    s = Sit()
    me = None
    opps = []
    for x in state.seats:
        if x.seat == state.seat:
            me = x
        elif x.status in ("active", "all_in"):
            opps.append(x)
    s.me = me
    s.S = me.stack
    s.mybet = me.bet
    s.total = me.stack + me.bet
    s.opps = opps
    s.pot = state.pot
    s.n = state.players_remaining
    s.payouts = tuple(state.config.payouts)
    best_other = max([o.stack + o.bet for o in opps], default=0)
    s.leader = s.total >= best_other
    near_cap = (state.config.hand_cap - state.hand_number) <= AGG_WINDOW and len(opps) > 0
    s.aggr = near_cap and not s.leader
    m = BUST_MARGIN.get(s.n, 0.4)
    if s.aggr:
        m = 0.0
    elif near_cap and s.leader:
        m += LEADER_EXTRA_MARGIN
    s.margin = m
    s.aggressor = max(opps, key=lambda o: o.bet) if opps else None
    return s


# ------------------------------------------------------------------ equity
def win_prob(state, sit, n_opp, shove=False):
    """Estimated win probability if the hand goes to showdown vs n_opp opponents who put in chips.
    Heads-up equity vs a random hand (tables), raised to a power for multiway, then shaded down
    because opponents who bet / call an all-in hold stronger than random hands."""
    b = state.board
    if not b:
        eq = EQ169[hand_class(state.cards[0], state.cards[1])]
        shade = SHADE_PRE
    else:
        eq = EQPOST[len(b) - 3][postflop_bucket(state.cards, b)]
        shade = SHADE_POST
    n_opp = max(1, n_opp)
    eq = eq ** (1.0 + MULTI_K * (n_opp - 1))
    return eq ** (1.0 + shade)


def _stack_vector(state, sit, overrides):
    """ICM stacks of all non-busted seats; overrides: seat -> new stack."""
    out = []
    my_i = 0
    for x in state.seats:
        if x.status == "busted":
            continue
        if x.seat == state.seat:
            my_i = len(out)
        out.append(overrides.get(x.seat, x.stack))
    return out, my_i


def _pts(state, sit, overrides):
    st, mi = _stack_vector(state, sit, overrides)
    return icm(st, sit.payouts)[mi]


def _ev_fold(state, sit):
    """Points if we give up the pot: bettor wins it; if nobody has bet, split among live players."""
    me = state.seat
    ov = {me: sit.S}
    A = sit.aggressor
    if state.to_call > 0 and A is not None:
        ov[A.seat] = A.stack + sit.pot
    else:
        k = len(sit.opps) + 1
        share = sit.pot / k
        ov[me] = sit.S + share
        for o in sit.opps:
            ov[o.seat] = o.stack + share
    return _pts(state, sit, ov)


def ev_call(state, sit):
    """(ev_commit, ev_fold) in points for calling the current bet to showdown."""
    me = state.seat
    A = sit.aggressor
    c = state.to_call
    M = sit.mybet + c
    ex = max(0, A.bet - M)
    n_opp = sum(1 for o in sit.opps if o.bet > 0 or o.status == "all_in")
    p = win_prob(state, sit, max(1, n_opp), True)
    win = _pts(state, sit, {me: sit.S + sit.pot - ex, A.seat: A.stack + ex})
    lose = _pts(state, sit, {me: sit.S - c, A.seat: A.stack + sit.pot + c})
    return p * win + (1 - p) * lose, _ev_fold(state, sit)


def ev_shove(state, sit, commit=None):
    """(ev_commit, ev_fold) in points for putting `commit` more chips in (default: our whole stack)
    with fold equity against the opponents still to act and a single caller."""
    me = state.seat
    if commit is None or commit > sit.S:
        commit = sit.S
    T = sit.mybet + commit
    act = [o for o in sit.opps if o.status == "active"]
    if not act:
        return -1.0, 0.0
    cf = CALL_FRAC_FACING if state.to_call > 0 else CALL_FRAC
    f = (1.0 - cf) ** len(act)
    C = max(act, key=lambda o: o.stack + o.bet)
    M = min(T, C.stack + C.bet)
    r = max(0, M - sit.mybet)
    add = max(0, M - C.bet)
    p = win_prob(state, sit, 1, True)
    fold_win = _pts(state, sit, {me: sit.S + sit.pot})
    win = _pts(state, sit, {me: sit.S + sit.pot + add, C.seat: C.stack - add})
    lose = _pts(state, sit, {me: sit.S - r, C.seat: C.stack + sit.pot + r})
    ev = f * fold_win + (1 - f) * (p * win + (1 - p) * lose)
    return ev, _ev_fold(state, sit)


# ------------------------------------------------------------------ decisions
def pushfold(state, sit):
    """Short-stack preflop decision driven by ICM: True = go all-in (or call all-in)."""
    if sit.aggressor is None:
        return False
    if state.to_call >= sit.S:
        ev, base = ev_call(state, sit)
    else:
        ev, base = ev_shove(state, sit)
    return ev > base + sit.margin


def review(state, sit, action):
    """ICM veto of a tournament-life commit (fold/check instead), or aggression-mode upgrade of a
    fold to a call of an all-in. Returns the (possibly changed) action."""
    from poker_agent_sdk import Action
    kind = action.kind
    S, mybet = sit.S, sit.mybet
    if S <= 0 or sit.aggressor is None:
        return action
    total = sit.total
    tc = state.to_call
    if kind == "call" and tc > 0:
        if tc >= S or S - tc < LIFE_FRAC * total:
            ev, base = ev_call(state, sit)
            if ev > base + sit.margin:
                return action
            return Action.fold()
        return action
    if kind == "raise":
        commit = action.amount - mybet
        if S - commit >= LIFE_FRAC * total:
            return action
        if tc >= S:
            return action  # cannot happen (call is the only option); safe
        ev, base = ev_shove(state, sit, commit)
        if ev > base + sit.margin:
            return action
        if state.can("check"):
            return Action.check()
        if tc > 0 and state.can("call"):
            if tc >= S or S - tc < LIFE_FRAC * total:
                ev, base = ev_call(state, sit)
                if ev <= base + sit.margin:
                    return Action.fold()
            return Action.call()
        return Action.fold()
    if kind == "fold" and sit.aggr and tc > 0 and tc >= S and state.can("call"):
        ev, base = ev_call(state, sit)
        if ev > base:
            return Action.call()
    return action
