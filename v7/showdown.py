"""Guaranteed-loss river fold + a fast 7-card evaluator. Pure stdlib, stateless.

Cards are strings like "As"; internally ids 0..51 (rank*4 + suit, rank 0=2 .. 12=A).
"""
_R = "23456789TJQKA"
_S = "cdhs"
CARD = {r + s: i * 4 + j for i, r in enumerate(_R) for j, s in enumerate(_S)}
ALL_IDS = tuple(range(52))

# --- tables over 13-bit rank masks ---
_POP = [bin(m).count("1") for m in range(8192)]
_TOP5 = [0] * 8192      # top 5 set ranks packed 4 bits each (rank+1), highest first, zero padded
_STRAIGHT = [-1] * 8192  # high rank of best straight, -1 if none (wheel -> 3)
for _m in range(8192):
    _v, _n = 0, 0
    for _r in range(12, -1, -1):
        if _m >> _r & 1:
            _v = (_v << 4) | (_r + 1)
            _n += 1
            if _n == 5:
                break
    _TOP5[_m] = _v << (4 * (5 - _n))
    for _hi in range(12, 3, -1):
        _need = 0
        for _k in range(5):
            _need |= 1 << (_hi - _k)
        if _m & _need == _need:
            _STRAIGHT[_m] = _hi
            break
    if _STRAIGHT[_m] < 0 and _m & 0b1000000001111 == 0b1000000001111:
        _STRAIGHT[_m] = 3


def score7(cards):
    """cards: iterable of 7 ids. Higher score = better hand; equal = tie."""
    sm = [0, 0, 0, 0]
    cnt = [0] * 13
    for c in cards:
        r = c >> 2
        sm[c & 3] |= 1 << r
        cnt[r] += 1
    mask = sm[0] | sm[1] | sm[2] | sm[3]
    fm = 0
    for m in sm:
        if _POP[m] >= 5:
            fm = m
    if fm:
        st = _STRAIGHT[fm]
        if st >= 0:
            return (8 << 24) | st
    quad = trip = -1
    p1 = p2 = -1
    for r in range(12, -1, -1):
        n = cnt[r]
        if n == 4:
            if quad < 0:
                quad = r
        elif n == 3:
            if trip < 0:
                trip = r
            elif p1 < 0:
                p1 = r  # second trips acts as a pair
        elif n == 2:
            if p1 < 0:
                p1 = r
            elif p2 < 0:
                p2 = r
    if quad >= 0:
        return (7 << 24) | ((quad + 1) << 4) | (_TOP5[mask & ~(1 << quad)] >> 16)
    if trip >= 0 and p1 >= 0:
        return (6 << 24) | ((trip + 1) << 4) | (p1 + 1)
    if fm:
        return (5 << 24) | _TOP5[fm]
    st = _STRAIGHT[mask]
    if st >= 0:
        return (4 << 24) | st
    if trip >= 0:
        return (3 << 24) | ((trip + 1) << 8) | (_TOP5[mask & ~(1 << trip)] >> 12)
    if p2 >= 0:
        return (2 << 24) | ((p1 + 1) << 8) | ((p2 + 1) << 4) | (_TOP5[mask & ~(1 << p1) & ~(1 << p2)] >> 16)
    if p1 >= 0:
        return (1 << 24) | ((p1 + 1) << 12) | (_TOP5[mask & ~(1 << p1)] >> 8)
    return _TOP5[mask]


def guaranteed_loss(state):
    """River only, facing a bet: True if some live opponent beats us for every possible hidden card."""
    board = state.board
    if len(board) != 5 or state.to_call <= 0:
        return False
    var = state.variant
    if not var:
        return False
    revealed = var.get("revealed")
    disc = var.get("my_discard")
    if not revealed or disc is None or len(state.cards) != 2:
        return False
    try:
        mine = [CARD[c] for c in state.cards]
        b = [CARD[c] for c in board]
        known = set(mine) | set(b) | {CARD[disc]}
        opps = []
        for s in state.seats:
            if s.seat == state.seat or s.status not in ("active", "all_in"):
                continue
            rc = revealed.get(s.seat)
            if rc is None:
                continue
            opps.append(CARD[rc])
        if not opps:
            return False
        known.update(CARD[c] for c in revealed.values())
        known.update(opps)
        my = score7(mine + b)
        unseen = [x for x in ALL_IDS if x not in known]
        if not unseen:
            return False
        for rc in opps:
            base = b + [rc]
            for x in unseen:
                if score7(base + [x]) <= my:
                    break
            else:
                return True
        return False
    except Exception:
        return False
