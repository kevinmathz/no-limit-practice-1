"""Lock detection + lock aggression. Pure stdlib, stateless.

guaranteed_win_now(state): flop/turn/river. For every live opponent (active/all_in, not us) and every
unseen card x, our best hand (2 hole + board) >= their best (revealed card + x + board). Ties count.
Unseen = 52 - our cards - our discard - board - every revealed card (same as showdown.py).
score7 from showdown.py ranks any 5..7 cards (rank masks + counts, no 5-card enumeration needed).
"""
from poker_agent_sdk import Action

from showdown import CARD, score7

ALL_IDS = tuple(range(52))
RIVER_FRAC = 0.95   # river lock, checked to: bet ~0.95 pot
FT_FRAC = 0.75      # flop/turn lock, checked to: bet ~0.75 pot


def guaranteed_win_now(state):
    board = state.board
    if len(board) < 3 or len(state.cards) != 2:
        return False
    var = state.variant or {}
    revealed = var.get("revealed") or {}
    disc = var.get("my_discard")
    try:
        mine = [CARD[c] for c in state.cards]
        b = [CARD[c] for c in board]
        known = set(mine) | set(b)
        if disc is not None:
            known.add(CARD[disc])
        opps = []
        for s in state.seats:
            if s.seat == state.seat or s.status not in ("active", "all_in"):
                continue
            rc = revealed.get(s.seat)
            if rc is None:
                return False
            opps.append(CARD[rc])
        if not opps:
            return False
        known.update(CARD[c] for c in revealed.values())
        unseen = [x for x in ALL_IDS if x not in known]
        if not unseen:
            return False
        my = score7(mine + b)
        for rc in opps:
            base = b + [rc]
            sc = [0, 0, 0, 0]
            for c in base:
                sc[c & 3] += 1
            # x's suit matters only if it could complete a flush ("hot" suit); otherwise one card per rank
            seen_rank = 0
            for x in unseen:
                if sc[x & 3] < 4:
                    bit = 1 << (x >> 2)
                    if seen_rank & bit:
                        continue
                    seen_rank |= bit
                base.append(x)
                if score7(base) > my:
                    return False
                base.pop()
        return True
    except Exception:
        return False


def _clamp(r, total):
    return Action.raise_to(int(max(r.min_amount, min(r.max_amount, int(total)))))


def lock_action(state, stackoff_frac):
    """Returns (action, is_river) or None. Call only after guaranteed_win_now(state) is True."""
    me = next(s for s in state.seats if s.seat == state.seat)
    r = state.can("raise")
    pot = state.pot
    river = len(state.board) == 5
    if river:
        if state.to_call <= 0:
            if r:
                return _clamp(r, me.bet + RIVER_FRAC * pot), True
            return (Action.check() if state.can("check") else None), True
        if r:
            return _clamp(r, me.bet + state.to_call + (pot + state.to_call)), True
        return (Action.call() if state.can("call") else None), True
    if state.to_call <= 0:
        if r:
            return _clamp(r, me.bet + FT_FRAC * pot), False
        return None, False
    if r:
        total = int(max(r.min_amount, min(r.max_amount, me.bet + state.to_call + (pot + state.to_call))))
        chips = me.stack + me.bet
        if total <= stackoff_frac * chips:
            return Action.raise_to(total), False
    return (Action.call() if state.can("call") else None), False
