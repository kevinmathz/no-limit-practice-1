"""Tightening layer: much more check/fold. Applied to the bot's final action.

Goal: survive like check_or_fold, but still win chips with strong hands.
- Preflop: continue (call/raise) only with the top PRE_TOP share of hands (by equity vs random);
  widen to PRE_TOP_SHORT when we are short (<= SHORT_BB big blinds), since blinds eat us.
- Postflop: never bet/raise without a made hand >= BET_MIN bucket; facing a bet continue only with
  bucket >= CALL_MIN, or a cheap call (to_call <= CHEAP_FRAC of pot) with bucket >= 1.
- Stack-off guard: putting more than STACKOFF_FRAC of our chips in needs a premium preflop hand
  (top PRE_PREMIUM) or a postflop bucket >= STACKOFF_MIN.
- Off in late-game aggression mode (last AGG_WINDOW hands, not chip leader), so ICM aggression still works.
"""
import json
import os

from poker_agent_sdk import Action

PRE_TOP = 0.06        # continue preflop with the top 6% of hands
PRE_TOP_SHORT = 0.15  # ... top 15% when short-stacked
SHORT_BB = 15
PRE_PREMIUM = 0.04    # may stack off preflop: ~AA-QQ, AK
BET_MIN = 5           # postflop: bet/raise only with two pair or better
CALL_MIN = 4          # postflop: call a bet with top pair good kicker or better
CHEAP_FRAC = 0.25     # ... or any pair/draw if the call is <= 25% of the pot
STACKOFF_FRAC = 0.35
STACKOFF_MIN = 5      # two pair or better
AGG_WINDOW = 25

# v14: players-remaining schedule A (step table). n -> (PRE_TOP, PRE_TOP_SHORT, TOP_CONT, BET_MIN, CALL_MIN, STACKOFF_MIN)
_SCHED = {
    8: (0.08, 0.15, 0.35, 5, 4, 5),
    7: (0.08, 0.15, 0.35, 5, 4, 5),
    6: (0.06, 0.12, 0.30, 5, 4, 5),
    5: (0.04, 0.10, 0.20, 5, 5, 6),
    4: (0.04, 0.10, 0.20, 5, 5, 6),
    3: (0.15, 0.25, 0.45, 4, 3, 4),
    2: (0.30, 0.45, 0.60, 3, 2, 3),
}


def params(n):
    try:
        n = int(n)
    except Exception:
        n = 6
    return _SCHED[min(8, max(2, n))]

RANK = {r: i + 2 for i, r in enumerate("23456789TJQKA")}
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "equity169.json")) as _f:
    _EQ = json.load(_f)


def _combos(c):
    return 6 if len(c) == 2 else (4 if c.endswith("s") else 12)


# percentile from the top (0 = best hand), combo-weighted
_PCT = {}
_cum = 0.0
for _c in sorted(_EQ, key=lambda k: -_EQ[k]):
    _PCT[_c] = _cum / 1326.0
    _cum += _combos(_c)


def hand_class(c1, c2):
    r1, r2 = RANK[c1[0]], RANK[c2[0]]
    if r1 < r2:
        c1, c2, r1, r2 = c2, c1, r2, r1
    if r1 == r2:
        return c1[0] + c2[0]
    return c1[0] + c2[0] + ("s" if c1[1] == c2[1] else "o")


def postflop_bucket(hole, board):
    """Same scale as v1's abstraction: 0 nothing ... 8 full house+ (1 = draw)."""
    hr = [RANK[c[0]] for c in hole]
    br = [RANK[c[0]] for c in board]
    allr = hr + br
    cnt = {}
    for r in allr:
        cnt[r] = cnt.get(r, 0) + 1
    suits = {}
    for c in list(hole) + list(board):
        suits[c[1]] = suits.get(c[1], 0) + 1
    if any(n >= 5 and any(h[1] == s for h in hole) for s, n in suits.items()):
        return 7
    hole_pairs = [r for r in set(hr) if cnt[r] >= 2]
    trips = [r for r in set(hr) if cnt[r] >= 3]
    if trips:
        return 8 if any(cnt[r] >= 2 for r in cnt if r not in trips) else 6
    if len(hole_pairs) >= 2:
        return 5
    if hole_pairs:
        top = max(br)
        p = max(hole_pairs)
        if hr[0] == hr[1]:
            return 4 if p > top else 2
        if p == top:
            k = hr[0] if hr[1] == p else hr[1]
            return 4 if k >= 10 else 3
        return 2
    if len(board) < 5 and any(n == 4 and any(h[1] == s for h in hole) for s, n in suits.items()):
        return 1
    return 0


def _aggression_mode(state, me):
    if state.config.hand_cap - state.hand_number > AGG_WINDOW:
        return False
    return any(s.stack > me.stack for s in state.seats if s.seat != me.seat)


def _passive(state):
    if state.can("check"):
        return Action.check()
    if state.can("fold"):
        return Action.fold()
    first = state.legal_actions[0]
    return Action(first.kind, first.min_amount)


def tighten(state, action):
    if state.phase != "betting" or action.kind in ("fold", "check"):
        return action
    me = next(s for s in state.seats if s.seat == state.seat)
    if _aggression_mode(state, me):
        return action
    chips = me.stack + me.bet
    added = (action.amount - me.bet) if action.kind == "raise" else min(state.to_call, me.stack)
    stackoff = added > STACKOFF_FRAC * chips
    bb = max(state.blinds[1], 1)
    pre_top, pre_top_short, _tc, bet_min, call_min, stackoff_min = params(getattr(state, "players_remaining", 6))

    if not state.board:
        pct = _PCT.get(hand_class(state.cards[0], state.cards[1]), 1.0)
        top = pre_top_short if chips / bb <= SHORT_BB else pre_top
        if action.kind == "call" and state.to_call <= 0:
            return action
        if pct > top:
            return _passive(state)
        if stackoff and pct > PRE_PREMIUM and chips / bb > SHORT_BB:
            return _passive(state)
        return action

    b = postflop_bucket(state.cards, state.board)
    if stackoff and b < stackoff_min:
        return _passive(state)
    if action.kind == "raise":
        if b >= bet_min:
            return action
        if state.to_call > 0 and b >= call_min and state.can("call"):
            return Action.call()
        return _passive(state)
    # call
    if b >= call_min or (b >= 1 and state.to_call <= CHEAP_FRAC * state.pot):
        return action
    return _passive(state)


# ---- top-of-range filter (v12/v13): facing a bet continue only in the top min(MDF, TOP_CONT) of range ----
TOP_CONT = 0.30
import mdf as _mdf  # noqa: E402


def top_of_range(state, action):
    if state.phase != "betting" or action.kind in ("fold", "check") or state.to_call <= 0:
        return action
    me = next(s for s in state.seats if s.seat == state.seat)
    if _aggression_mode(state, me):
        return action
    nb = len(state.board)
    if nb == 0:
        key, street = hand_class(state.cards[0], state.cards[1]), -1
    else:
        key, street = postflop_bucket(state.cards, state.board), min(nb - 3, 2)
    target = min(_mdf.mdf(state.pot, state.to_call), params(getattr(state, "players_remaining", 6))[2])
    p = _mdf.defend_prob(_mdf.slot_for(key, street), target / _mdf.MDF_SCALE)
    if p >= 1.0 or (p > 0.0 and _mdf.random.random() < p):
        return action
    return _passive(state)
