"""v8 = v0 + tight.py (much more check/fold). Team bot template. Plays legally, fast (<1 ms), and keeps per-table state in `memory`.

Rules this file follows (from the competition docs):
- Never rely on self/globals between calls: any copy may answer any table.
- Per-table state goes in `memory` (bytes, 64 KB cap). Parse it defensively.
- Only return offered kinds; raise totals must be ints within [min_amount, max_amount].
- 1 ms per decision by default; upload test match has NO time bank.
"""
import os

# Must be set before importing numpy/torch so local timing matches production.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402

from poker_agent_sdk import Action, Agent as BaseAgent  # noqa: E402

RANK = {r: i + 2 for i, r in enumerate("23456789TJQKA")}

# Precompute at import (import has its own startup allowance; act does not).
PREFLOP = {}
for _a in RANK:
    for _b in RANK:
        for _suited in (True, False):
            hi, lo = max(RANK[_a], RANK[_b]), min(RANK[_a], RANK[_b])
            score = hi * 2 + lo  # crude Chen-like strength, ~6..42
            if hi == lo:
                score = 30 + hi * 1.5
            if _suited and hi != lo:
                score += 3
            if hi - lo == 1:
                score += 2
            PREFLOP[(hi, lo, _suited)] = score


def preflop_score(cards):
    a, b = cards
    hi, lo = max(RANK[a[0]], RANK[b[0]]), min(RANK[a[0]], RANK[b[0]])
    return PREFLOP[(hi, lo, a[1] == b[1])]


def made_hand(cards, board):
    """0 = nothing, 1 = weak pair, 2 = top pair / overpair, 3 = two pair+ (rough)."""
    ranks = [RANK[c[0]] for c in cards]
    branks = [RANK[c[0]] for c in board]
    if not branks:
        return 0
    allr = ranks + branks
    counts = {r: allr.count(r) for r in set(allr)}
    mine = [counts[r] for r in set(ranks)]
    if max(counts.values()) >= 3 and max(mine) >= 2:
        return 3
    pairs_with_me = [r for r in set(ranks) if counts[r] >= 2]
    if len(pairs_with_me) >= 2:
        return 3
    if ranks[0] == ranks[1] and ranks[0] > max(branks):
        return 2
    if pairs_with_me and max(pairs_with_me) == max(branks):
        return 2
    if pairs_with_me or ranks[0] == ranks[1]:
        return 1
    return 0


# ---------- memory (example: count hands and opponent voluntary actions) ----------
def load_mem(memory):
    try:
        m = json.loads(memory.decode()) if memory else {}
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def update_mem(m, events, me):
    vpip = m.setdefault("vpip", {})  # seat -> [voluntary actions, hands seen]
    for e in events:
        if e.kind == "hand":
            m["hands"] = m.get("hands", 0) + 1
        elif e.kind in ("call", "raise") and e.seat is not None and e.seat != me:
            s = vpip.setdefault(str(e.seat), [0, 0])
            s[0] += 1
    return m


# ---------- safe action helpers ----------
def raise_to(state, total):
    r = state.can("raise")
    if not r:
        return call_or_check(state)
    total = int(total)
    total = max(r.min_amount, min(r.max_amount, total))
    return Action.raise_to(total)


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


class Agent(BaseAgent):
    def act(self, state, memory):
        mem = load_mem(memory)
        try:
            mem = update_mem(mem, state.events, state.seat)
            action = self.decide(state)
        except Exception:
            action = check_or_fold(state)
        out = json.dumps(mem, separators=(",", ":")).encode()
        if len(out) > 16_000:  # keep memory small; it travels inside decision time
            out = b""
        return action, out

    def decide(self, state):
        if state.phase == "choose":
            return self.choose(state)
        if state.phase != "betting":  # unknown phase: automatic choice
            first = state.legal_actions[0]
            return Action(first.kind, first.min_amount)
        return self.bet(state)

    def choose(self, state):
        """Keep-reveal-discard: keep the best 2-card combo; reveal the weaker card of it.
        TODO: smarter reveal (bluff/ICM), use opponents' reveals once known."""
        c = state.cards
        best = max(((a, b) for a in range(3) for b in range(a + 1, 3)),
                   key=lambda p: preflop_score((c[p[0]], c[p[1]])))
        a, b = best
        if RANK[c[a][0]] >= RANK[c[b][0]]:
            return Action.choose(keep=a, reveal=b)
        return Action.choose(keep=b, reveal=a)

    def bet(self, state):
        bb = state.blinds[1]
        me = next(s for s in state.seats if s.seat == state.seat)
        eff_bb = (me.stack + me.bet) / max(bb, 1)

        if not state.board:  # preflop
            score = preflop_score(state.cards)
            if eff_bb <= 12:  # short: push or fold
                if score >= 30:
                    return raise_to(state, me.stack + me.bet)
                return check_or_fold(state)
            if score >= 38:
                return raise_to(state, max(3 * bb, 3 * state.to_call + me.bet))
            if score >= 30 and state.to_call <= 3 * bb:
                return raise_to(state, 3 * bb) if state.to_call <= bb else call_or_check(state)
            if score >= 24 and state.to_call <= bb:
                return call_or_check(state)
            return check_or_fold(state)

        strength = made_hand(state.cards, state.board)
        pot = state.pot
        if strength >= 3:
            return raise_to(state, me.bet + state.to_call + pot)
        if strength == 2:
            if state.to_call == 0:
                return raise_to(state, me.bet + (2 * pot) // 3)
            if state.to_call <= pot:
                return call_or_check(state)
            return check_or_fold(state)
        if strength == 1 and state.to_call <= pot // 3:
            return call_or_check(state)
        return check_or_fold(state)


# ---- tightening layer (tight.py): much more check/fold, off in late-game aggression mode ----
import os as _os, sys as _sys  # noqa: E402
_d = _os.path.dirname(_os.path.abspath(__file__))
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import tight as _tight  # noqa: E402

_base_act = Agent.act


def _tight_act(self, state, memory):
    action, mem = _base_act(self, state, memory)
    try:
        action = _tight.tighten(state, action)
    except Exception:
        pass
    return action, mem


Agent.act = _tight_act
