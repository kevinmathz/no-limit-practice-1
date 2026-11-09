"""Fast pure-Python 5/6/7-card hand evaluator (KRD runtime bot; self-contained, stdlib only).

Card ids: id = rank_idx*4 + suit_idx (rank 0..12 = 2..A, suit 0..3 = c,d,h,s), same as common.py.

Score encoding (identical to no_limit/game/evaluator.py; larger is better):
    category << 20 | k1 << 16 | k2 << 12 | k3 << 8 | k4 << 4 | k5
category 0 = high card .. 8 = straight flush.

Method: every card maps to KEY[c] = (5**rank << 16) | (1 << 4*suit).  The sum of the keys of
the hand holds a base-5 rank-count signature in the high bits (unique while every rank count
is <= 4) and four 4-bit suit counters in the low 16 bits.
  * flush test: (sum + 0x3333) & 0x8888 != 0  <=> some suit has >= 5 cards
    (no carries: a nibble holds at most 7).
  * no flush: one dict lookup RANK_SCORE[sum >> 16] (built at import for every rank
    multiset of 5..7 cards).
  * flush: rank mask of the flush suit -> STRAIGHT_TOP / TOP5 tables.  With <= 7 cards a
    flush can never coexist with quads or a full house, so the flush score is final.

Public API
    ev(cards)            -> score; cards = iterable of ids or "As" strings (5..7 cards)
    ev_ids(cards)        -> score for a list/tuple of ids (fastest general entry point)
    ev_key(keysum, ids)  -> score given precomputed sum(KEY[c] for c in ids)
    KEY                  -> per-card key list for incremental sums
    category(score)
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)
CARD_ID = {c: i for i, c in enumerate(DECK)}

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)


def _build_mask_tables():
    straight_top = [-1] * 8192
    top5 = [0] * 8192
    for mask in range(8192):
        best = -1
        for top in range(12, 3, -1):
            if (mask >> (top - 4)) & 0x1F == 0x1F:
                best = top
                break
        if best < 0 and (mask & 0x100F) == 0x100F:  # wheel
            best = 3
        straight_top[mask] = best
        packed = 0
        n = 0
        for r in range(12, -1, -1):
            if (mask >> r) & 1:
                packed = (packed << 4) | r
                n += 1
                if n == 5:
                    break
        top5[mask] = packed << (4 * (5 - n))
    return straight_top, top5


STRAIGHT_TOP, TOP5 = _build_mask_tables()

# flush path: for a suit s, score of the flush given its rank mask
FLUSH_SCORE = [0] * 8192
for _m in range(8192):
    _t = STRAIGHT_TOP[_m]
    FLUSH_SCORE[_m] = (STRAIGHT_FLUSH << 20) | (_t << 16) if _t >= 0 else (FLUSH << 20) | TOP5[_m]


def _score_counts(counts):
    """Non-flush score from a 13-long rank count list (same rules as no_limit evaluator)."""
    rank_mask = 0
    quads = -1
    trips = []
    pairs = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n:
            rank_mask |= 1 << r
        if n == 4:
            if quads < 0:
                quads = r
            else:
                trips.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
    if quads >= 0:
        kicker = TOP5[rank_mask & ~(1 << quads)] >> 16
        return (QUADS << 20) | (quads << 16) | (kicker << 12)
    if trips:
        t = trips[0]
        second = trips[1] if len(trips) > 1 else -1
        if pairs and pairs[0] > second:
            second = pairs[0]
        if second >= 0:
            return (FULL_HOUSE << 20) | (t << 16) | (second << 12)
    st = STRAIGHT_TOP[rank_mask]
    if st >= 0:
        return (STRAIGHT << 20) | (st << 16)
    if trips:
        t = trips[0]
        kick = TOP5[rank_mask & ~(1 << t)] >> 12
        return (TRIPS << 20) | (t << 16) | (kick & 0xFF) << 8
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = TOP5[rank_mask & ~(1 << p1) & ~(1 << p2)] >> 16
        return (TWO_PAIR << 20) | (p1 << 16) | (p2 << 12) | (kicker << 8)
    if pairs:
        p = pairs[0]
        kick = TOP5[rank_mask & ~(1 << p)] >> 8
        return (PAIR << 20) | (p << 16) | ((kick & 0xFFF) << 4)
    return (HIGH_CARD << 20) | TOP5[rank_mask]


def _build_rank_table():
    table = {}
    counts = [0] * 13
    pw = [5 ** r for r in range(13)]

    def rec(r, left, key, total):
        if r < 0:
            if total >= 5:
                table[key] = _score_counts(counts)
            return
        for n in range(0, min(4, left) + 1):
            counts[r] = n
            rec(r - 1, left - n, key + n * pw[r], total + n)
        counts[r] = 0

    rec(12, 7, 0, 0)
    return table


RANK_SCORE = _build_rank_table()
KEY = [(5 ** (c >> 2) << 16) | (1 << (4 * (c & 3))) for c in range(52)]
BIT = [1 << (c >> 2) for c in range(52)]
SUIT = [c & 3 for c in range(52)]
_FLUSH_SUIT_OF = {}  # low 16 bits (+0x3333 & 0x8888) -> suit with >=5 cards
for _s in range(4):
    _FLUSH_SUIT_OF[0x8 << (4 * _s)] = _s


def _flush(ids, sk):
    f = (sk + 0x3333) & 0x8888
    s = _FLUSH_SUIT_OF[f]
    m = 0
    for c in ids:
        if c & 3 == s:
            m |= 1 << (c >> 2)
    return FLUSH_SCORE[m]


def ev_key(k, ids):
    """Score from a precomputed key sum `k` of the card ids `ids` (5..7 cards)."""
    if (k + 0x3333) & 0x8888:
        return _flush(ids, k & 0xFFFF)
    return RANK_SCORE[k >> 16]


def ev_ids(ids):
    k = 0
    for c in ids:
        k += KEY[c]
    if (k + 0x3333) & 0x8888:
        return _flush(ids, k & 0xFFFF)
    return RANK_SCORE[k >> 16]


def ev7(a, b, c, d, e, f, g):
    """Seven ids as positional args (the hot path)."""
    K = KEY
    k = K[a] + K[b] + K[c] + K[d] + K[e] + K[f] + K[g]
    if (k + 0x3333) & 0x8888:
        return _flush((a, b, c, d, e, f, g), k & 0xFFFF)
    return RANK_SCORE[k >> 16]


def ev(cards):
    """Score 5..7 cards given as ids or strings."""
    ids = [CARD_ID[c] if isinstance(c, str) else c for c in cards]
    return ev_ids(ids)


def category(score):
    return score >> 20


if __name__ == "__main__":
    import random
    import time
    rng = random.Random(1)
    hands = [rng.sample(range(52), 7) for _ in range(100000)]
    t0 = time.perf_counter()
    for h in hands:
        ev_ids(h)
    t1 = time.perf_counter()
    for h in hands:
        ev7(*h)
    t2 = time.perf_counter()
    print(f"table entries {len(RANK_SCORE)}")
    print(f"ev_ids: {len(hands) / (t1 - t0) / 1000:.0f} evals/ms; ev7: {len(hands) / (t2 - t1) / 1000:.0f} evals/ms")
