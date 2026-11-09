"""v1: MCCFR policy bot.

- Choose phase (keep-reveal-discard): picks which card to reveal and which to discard at random.
- Betting: maps the real situation to the training abstraction (v1/abstraction.py) and samples
  an action from the MCCFR average strategy in v1/policy.json (trained by train/mccfr.py).
- Falls back to a simple safe rule when a situation is outside the trained tree.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json  # noqa: E402
import random  # noqa: E402

from poker_agent_sdk import Action, Agent as BaseAgent  # noqa: E402

from abstraction import (DEPTHS, MAX_RAISES, RAISE_FRAC, depth_index, info_key,  # noqa: E402
                         postflop_bucket, preflop_bucket, spr_bucket)

# Load once at import (import has its own startup allowance).
with open(os.path.join(_HERE, "policy.json")) as _f:
    POLICY = json.load(_f)
with open(os.path.join(_HERE, "preflop_buckets.json")) as _f:
    PRE = json.load(_f)

_rng = random.SystemRandom()  # copies are forked from one process; avoid identical RNG streams
STREET = {0: 0, 3: 1, 4: 2, 5: 3}


def abstract_history(state):
    """Current-street action sequence in the training alphabet (c/h/p/a).
    Folds are dropped; consecutive calls/checks and calls after a raise are collapsed,
    which maps multiway action onto the heads-up tree."""
    bets = {}
    pot = 0
    cur = 0
    hist = ""
    seat_status = {s.seat: s for s in state.seats}
    for e in state.history:
        k = e.kind
        if k == "board":
            bets = {}
            cur = 0
            hist = ""
        elif k in ("blind", "ante"):
            amt = e.amount or 0
            pot += amt
            if k == "blind":
                bets[e.seat] = bets.get(e.seat, 0) + amt
                cur = max(cur, bets[e.seat])
        elif k in ("call", "check"):
            amt = e.amount or 0
            pot += amt
            bets[e.seat] = bets.get(e.seat, 0) + amt
            if not hist or hist[-1] not in "chpa":
                hist += "c"
        elif k == "raise":
            total = e.amount
            prev = bets.get(e.seat, 0)
            to_call = cur - prev
            add = total - cur
            pot2 = pot + to_call
            s = seat_status.get(e.seat)
            if s is not None and s.status == "all_in" and s.bet == total:
                x = "a"
            else:
                x = "h" if add < 0.75 * pot2 else "p"
            hist += x
            pot += total - prev
            bets[e.seat] = total
            cur = total
    return hist


def fallback(state, strong):
    if strong and state.can("call"):
        return Action.call()
    if state.can("check"):
        return Action.check()
    if state.can("fold"):
        return Action.fold()
    first = state.legal_actions[0]
    return Action(first.kind, first.min_amount)


class Agent(BaseAgent):
    def act(self, state, memory):
        try:
            if state.phase == "choose":
                return self.choose(state), memory
            if state.phase != "betting":
                first = state.legal_actions[0]
                return Action(first.kind, first.min_amount), memory
            return self.bet(state), memory
        except Exception:
            return fallback(state, False), memory

    def choose(self, state):
        """Random: pick one card to reveal and one to discard; keep the third."""
        reveal, discard = _rng.sample(range(len(state.cards)), 2)
        keep = next(i for i in range(len(state.cards)) if i not in (reveal, discard))
        return Action.choose(keep, reveal)

    def bet(self, state):
        bb = max(state.blinds[1], 1)
        me = next(s for s in state.seats if s.seat == state.seat)
        opps = [s for s in state.seats if s.seat != state.seat and s.status in ("active", "all_in")]
        if not opps:
            return fallback(state, True)
        street = STREET.get(len(state.board), 3)
        my_total = me.stack + me.bet
        opp_total = max(s.stack + s.bet for s in opps)
        eff_start = min(my_total, opp_total)
        pot_start = state.pot - sum(s.bet for s in state.seats)

        if street == 0:
            bucket = preflop_bucket(state.cards, PRE)
            spr_b = 0
            depth_i = depth_index(eff_start / bb)
        else:
            bucket = postflop_bucket(state.cards, state.board)
            spr_b = spr_bucket(eff_start, pot_start)
            depth_i = depth_index((eff_start + pot_start / 2) / bb)

        hist = abstract_history(state)
        strong = (street == 0 and bucket >= 6) or (street > 0 and bucket >= 5)
        if sum(ch in "hpa" for ch in hist) > MAX_RAISES:
            hist = hist[:-1] + "a" if hist[-1] != "a" else hist
        probs = POLICY.get(info_key(depth_i, street, bucket, spr_b, hist))
        if probs is None:  # nearby trained situations
            for d in (depth_i - 1, depth_i + 1):
                if 0 <= d < len(DEPTHS):
                    probs = POLICY.get(info_key(d, street, bucket, spr_b, hist))
                    if probs:
                        break
        if not probs:
            return fallback(state, strong)
        return self.sample(state, probs, me)

    def sample(self, state, probs, me):
        r = state.can("raise")
        options = {}
        for a, p in probs.items():
            if a == "f":
                key = "f" if state.can("fold") else "c"
            elif a in ("h", "p", "a"):
                key = a if r else "c"
            else:
                key = "c"
            options[key] = options.get(key, 0.0) + p
        x = _rng.random() * sum(options.values())
        acc = 0.0
        choice = "c"
        for a, p in options.items():
            acc += p
            if x < acc:
                choice = a
                break
        if choice == "f":
            return Action.fold()
        if choice == "c":
            if state.can("check"):
                return Action.check()
            if state.can("call"):
                return Action.call()
            return fallback(state, False)
        if choice == "a":
            return Action.raise_to(int(r.max_amount))
        to_call = state.to_call
        total = me.bet + to_call + int(RAISE_FRAC[choice] * (state.pot + to_call))
        total = max(r.min_amount, min(r.max_amount, int(total)))
        return Action.raise_to(total)
