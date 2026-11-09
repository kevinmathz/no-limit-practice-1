"""kevin_v2_simple baseline (minute 0-5): safe, tight, legal."""
import random

from poker_agent_sdk import Action, Agent as BaseAgent

RV = {r: i for i, r in enumerate("23456789TJQKA")}


def score(a, b):
    ra, rb = RV[a[0]], RV[b[0]]
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return 20 + 2 * hi
    s = hi + 0.5 * lo + (2 if a[1] == b[1] else 0) + (1 if hi - lo == 1 else 0)
    return s


def raise_to(state, total):
    r = state.can("raise")
    if not r:
        return call_or_check(state)
    return Action.raise_to(int(max(r.min_amount, min(r.max_amount, int(total)))))


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


class Agent(BaseAgent):
    def act(self, state, memory):
        try:
            return self.decide(state), memory
        except Exception:
            try:
                if state.phase != "betting":
                    f = state.legal_actions[0]
                    return Action(f.kind, f.min_amount), memory
                return check_or_fold(state), memory
            except Exception:
                return Action("fold"), memory

    def decide(self, state):
        if state.phase == "choose":
            c = state.cards
            pairs = [(0, 1), (0, 2), (1, 2)]
            a, b = max(pairs, key=lambda p: score(c[p[0]], c[p[1]]))
            if random.random() < 0.5:
                a, b = b, a
            return Action.choose(keep=a, reveal=b)
        if state.phase != "betting":
            f = state.legal_actions[0]
            return Action(f.kind, f.min_amount)
        bb = state.blinds[1]
        me = state.seats[state.seat]
        eff = (me.stack + me.bet) / bb
        sc = score(*state.cards)
        if not state.board:
            if eff <= 12:
                return raise_to(state, me.stack + me.bet) if sc >= 18 else check_or_fold(state)
            if sc >= 26:
                return raise_to(state, max(3 * bb, 3 * (state.to_call + me.bet)))
            return check_or_fold(state)
        return check_or_fold(state) if state.to_call else Action.check()
