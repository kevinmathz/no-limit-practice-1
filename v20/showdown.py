"""Guaranteed-loss river fold + a fast 7-card evaluator. Pure stdlib, stateless.

Cards are strings like "As"; internally ids 0..51 (id = rank*4 + suit, rank 0 = deuce .. 12 = ace).
Score = category << 20 | packed ranks (4 bits each); higher is better, equal is a tie.
"""
_R = "23456789TJQKA"
_S = "cdhs"
CARD = {r + s: i * 4 + j for i, r in enumerate(_R) for j, s in enumerate(_S)}

# ---- tables over 13-bit rank masks (built at import) ----
_POP = [bin(m).count("1") for m in range(8192)]
_TOP = [[0] * 8192 for _ in range(6)]  # _TOP[k][mask]: top-k set ranks packed, highest first
_STRAIGHT = [-1] * 8192                 # high rank of best straight (wheel -> 3), -1 if none
for _m in range(8192):
    _rs = [r for r in range(12, -1, -1) if _m >> r & 1]
    for _k in range(1, 6):
        _v = 0
        for _r in _rs[:_k]:
            _v = (_v << 4) | _r
        _TOP[_k][_m] = _v << (4 * (_k - min(_k, len(_rs))))
    for _hi in range(12, 3, -1):
        _need = 31 << (_hi - 4)
        if _m & _need == _need:
            _STRAIGHT[_m] = _hi
            break
    if _STRAIGHT[_m] < 0 and _m & 0b1000000001111 == 0b1000000001111:
        _STRAIGHT[_m] = 3
_TOP5, _TOP3, _TOP2, _TOP1 = _TOP[5], _TOP[3], _TOP[2], _TOP[1]


def score7(cards):
    """cards: iterable of 5-7 card ids. Returns an int; higher = better hand."""
    sm = [0, 0, 0, 0]
    cnt = [0] * 13
    for c in cards:
        r = c >> 2
        sm[c & 3] |= 1 << r
        cnt[r] += 1
    mask = sm[0] | sm[1] | sm[2] | sm[3]
    flush = 0
    for m in sm:
        if _POP[m] >= 5:
            st = _STRAIGHT[m]
            if st >= 0:
                return (8 << 20) | st
            flush = (5 << 20) | _TOP5[m]
    quad = trip = trip2 = pair = pair2 = -1
    for r in range(12, -1, -1):
        n = cnt[r]
        if n == 4:
            quad = r
        elif n == 3:
            if trip < 0:
                trip = r
            elif trip2 < 0:
                trip2 = r
        elif n == 2:
            if pair < 0:
                pair = r
            elif pair2 < 0:
                pair2 = r
    if quad >= 0:
        return (7 << 20) | (quad << 16) | (_TOP1[mask & ~(1 << quad)] << 12)
    if trip >= 0:
        p = max(pair, trip2)
        if p >= 0:
            return (6 << 20) | (trip << 16) | (p << 12)
    if flush:
        return flush
    st = _STRAIGHT[mask]
    if st >= 0:
        return (4 << 20) | st
    if trip >= 0:
        return (3 << 20) | (trip << 16) | (_TOP2[mask & ~(1 << trip)] << 8)
    if pair2 >= 0:
        return (2 << 20) | (pair << 16) | (pair2 << 12) | (_TOP1[mask & ~(1 << pair) & ~(1 << pair2)] << 8)
    if pair >= 0:
        return (1 << 20) | (pair << 16) | (_TOP3[mask & ~(1 << pair)] << 4)
    return _TOP5[mask]


def opponent_locks(my_score, base, unseen):
    """True if base (6 ids: board + opponent's revealed card) + x beats my_score for every x in unseen.
    The suit of x only matters when base already holds 4+ cards of that suit, so equal results are
    cached by (rank, flush-relevant suit). Exits at the first x that does not strictly beat us."""
    suit_n = [0, 0, 0, 0]
    for c in base:
        suit_n[c & 3] += 1
    seen = {}
    cards = list(base)
    cards.append(0)
    for x in unseen:
        s = x & 3
        key = x if suit_n[s] >= 4 else -1 - (x >> 2)  # rank-only key when suit is irrelevant
        if key in seen:
            continue
        cards[6] = x
        if score7(cards) <= my_score:
            return False
        seen[key] = True
    return True


def guaranteed_loss(state):
    """River only, facing a bet: True if some live opponent with a revealed card beats us for every
    possible hidden card (ties don't count). Returns False on anything unexpected."""
    board = state.board
    if len(board) != 5 or state.to_call <= 0:
        return False
    var = state.variant or {}
    revealed = var.get("revealed") or {}
    if not revealed:
        return False
    hole = [CARD[c] for c in state.cards]
    b = [CARD[c] for c in board]
    my_score = score7(hole + b)
    dead = set(hole)
    dead.update(b)
    d = var.get("my_discard")
    if d:
        dead.add(CARD[d])
    rev = {}
    for seat, c in revealed.items():
        if c:
            cid = CARD[c]
            dead.add(cid)
            rev[int(seat)] = cid
    unseen = [x for x in range(52) if x not in dead]
    for s in state.seats:
        if s.seat == state.seat or s.status not in ("active", "all_in"):
            continue
        cid = rev.get(s.seat)
        if cid is None:
            continue
        if opponent_locks(my_score, b + [cid], unseen):
            return True
    return False
