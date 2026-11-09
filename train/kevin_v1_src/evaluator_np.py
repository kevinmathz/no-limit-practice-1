"""Vectorised numpy 7-card poker hand evaluator.

    eval7(cards: int ndarray [N, 7] of card ids (common.py: id = rank_idx*4 + suit_idx))
        -> int64 ndarray [N]; higher is better, equal hands get equal values.

Score layout: category << 20 | five 4-bit rank nibbles (most significant first).
Categories: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush,
6 full house, 7 quads, 8 straight flush.

Implementation (fast path, `eval7`):
  * non-flush value = RANK_TABLE[sum of additive per-rank keys]. The keys
    (0, 1, 5, 22, ... 1479181; as in SKPokerEval) give a unique sum for each of
    the 49205 7-card rank multisets; max sum 7.8M, so the table is ~31 MB int32.
  * flush detection by summing per-suit nibbles (16**suit); only the ~3% flush
    hands get a 13-bit suit mask looked up in TOPK / STRAIGHT.
RANK_TABLE is built at import from `_eval7_generic` (a histogram + bitmask
evaluator, also fully vectorised), which takes ~1 s.

Run as a script to validate against the practice engine's `cards.best_hand`
and benchmark throughput.
"""
from __future__ import annotations

import numpy as np

_NMASK = 1 << 13
_POW2 = (1 << np.arange(13)).astype(np.int64)


def _build_tables():
    """TOPK[k][mask]: the k highest set ranks of `mask` packed as k nibbles (left-aligned
    to 5 nibbles so different k never collide within one category). STRAIGHT[mask]:
    high rank (3..12; 3 = wheel 5-high) of the best straight in mask, or -1."""
    topk = np.zeros((6, _NMASK), dtype=np.int64)
    straight = np.full(_NMASK, -1, dtype=np.int64)
    for m in range(_NMASK):
        ranks = [r for r in range(12, -1, -1) if m >> r & 1]
        for k in range(1, 6):
            v = 0
            for p in range(5):
                v = v << 4 | (ranks[p] if p < k and p < len(ranks) else 0)
            topk[k, m] = v
        for hi in range(12, 3, -1):
            win = 0x1F << (hi - 4)
            if m & win == win:
                straight[m] = hi
                break
        else:
            if m & 0x100F == 0x100F:  # A-2-3-4-5
                straight[m] = 3
    return topk, straight


TOPK, STRAIGHT = _build_tables()
HIGHBIT = (TOPK[1] >> 16).astype(np.int64)  # rank of the highest set bit (0 if mask empty)


def _eval7_generic(cards: np.ndarray, flushes: bool = True) -> np.ndarray:
    """Histogram/bitmask evaluator for [N, 7] card ids (slower; builds RANK_TABLE).
    With flushes=False suits are ignored."""
    cards = np.asarray(cards, dtype=np.int64)
    n = cards.shape[0]
    ranks = cards >> 2
    suits = cards & 3
    rows = np.arange(n, dtype=np.int64)[:, None]

    counts = np.bincount((rows * 13 + ranks).ravel(), minlength=n * 13).reshape(n, 13)
    scounts = np.bincount((rows * 4 + suits).ravel(), minlength=n * 4).reshape(n, 4)
    bits = _POW2[ranks]
    allmask = np.bitwise_or.reduce(bits, axis=1)

    m4 = (counts == 4).astype(np.int64) @ _POW2
    m3 = (counts == 3).astype(np.int64) @ _POW2
    m2 = (counts == 2).astype(np.int64) @ _POW2
    m1 = (counts == 1).astype(np.int64) @ _POW2

    # Default: high card.
    score = TOPK[5][allmask]

    # Pair (exactly one pair, no trips).
    hb2 = HIGHBIT[m2]
    sel = (m2 != 0) & (m3 == 0) & (m4 == 0)
    pair = (1 << 20) | (hb2 << 16) | (TOPK[3][m1] >> 4)
    score = np.where(sel, pair, score)

    # Two pair (2 or 3 pairs; best two + best remaining kicker).
    m2b = m2 & ~(np.int64(1) << hb2)
    hb2b = HIGHBIT[m2b]
    two_sel = sel & (m2b != 0)
    rest = allmask & ~(np.int64(1) << hb2) & ~(np.int64(1) << hb2b)
    twop = (2 << 20) | (hb2 << 16) | (hb2b << 12) | (HIGHBIT[rest] << 8)
    score = np.where(two_sel, twop, score)

    # Trips (one trips, no pair) / full house.
    hb3 = HIGHBIT[m3]
    trip_sel = (m3 != 0) & (m4 == 0)
    fh_pairmask = (m3 & ~(np.int64(1) << hb3)) | m2
    is_fh = trip_sel & (fh_pairmask != 0)
    trips = (3 << 20) | (hb3 << 16) | (TOPK[2][m1] >> 4)
    full = (6 << 20) | (hb3 << 16) | (HIGHBIT[fh_pairmask] << 12)
    score = np.where(trip_sel, np.where(is_fh, full, trips), score)

    # Quads.
    hb4 = HIGHBIT[m4]
    quad = (7 << 20) | (hb4 << 16) | (HIGHBIT[allmask & ~(np.int64(1) << hb4)] << 12)
    score = np.where(m4 != 0, quad, score)

    # Straight (beats trips/two pair/pair; cannot coexist with quads/FH in 7 cards
    # only if those are better, so only overwrite categories < 4).
    st = STRAIGHT[allmask]
    score = np.where((st >= 0) & (score < (4 << 20)), (4 << 20) | (st << 16), score)

    # Flush / straight flush (a flush in 7 cards excludes quads and full house).
    fsuit = np.argmax(scounts, axis=1)
    has_flush = scounts[np.arange(n), fsuit] >= 5
    if flushes and has_flush.any():
        fmask = np.bitwise_or.reduce(np.where(suits == fsuit[:, None], bits, 0), axis=1)
        sf = STRAIGHT[fmask]
        fl = np.where(sf >= 0, (8 << 20) | (sf << 16), (5 << 20) | TOPK[5][fmask])
        score = np.where(has_flush, fl, score)
    return score


# ------------------------------------------------------------------ fast path
_RANK_KEYS = np.array([0, 1, 5, 22, 98, 453, 2031, 8698, 22854, 83661, 262349, 636345, 1479181],
                      dtype=np.int32)
CARD_RKEY = _RANK_KEYS[np.arange(52) >> 2]                        # additive rank key per card
CARD_SKEY = (np.int32(1) << (4 * (np.arange(52) & 3))).astype(np.int32)  # suit nibble per card
CARD_BIT = (np.int32(1) << (np.arange(52) >> 2)).astype(np.int32)


def _build_rank_table() -> np.ndarray:
    from itertools import combinations_with_replacement
    ms = np.array([c for c in combinations_with_replacement(range(13), 7)
                   if max(c.count(r) for r in set(c)) <= 4], dtype=np.int64)
    vals = _eval7_generic(ms * 4, flushes=False)  # suits irrelevant with flushes=False
    keys = _RANK_KEYS[ms].sum(axis=1)
    assert len(np.unique(keys)) == len(ms), "rank keys not unique"
    table = np.zeros(int(keys.max()) + 1, dtype=np.int32)
    table[keys] = vals
    return table


RANK_TABLE = _build_rank_table()
FLUSH_TABLE = np.where(STRAIGHT >= 0, (8 << 20) | (STRAIGHT << 16), (5 << 20) | TOPK[5]).astype(np.int32)


def eval7(cards: np.ndarray) -> np.ndarray:
    """Evaluate [N, 7] card ids; returns int64 [N] strengths (higher wins, ties equal)."""
    cards = np.asarray(cards)
    score = RANK_TABLE[CARD_RKEY[cards].sum(axis=1)].astype(np.int64)
    sk = CARD_SKEY[cards].sum(axis=1)
    fl = np.flatnonzero((sk + 0x3333) & 0x8888)  # some suit nibble >= 5
    if fl.size:
        fc = cards[fl]
        nib = (sk[fl, None] >> np.array([0, 4, 8, 12])) & 15
        fsuit = np.argmax(nib, axis=1)
        fmask = np.bitwise_or.reduce(np.where((fc & 3) == fsuit[:, None], CARD_BIT[fc], 0), axis=1)
        score[fl] = FLUSH_TABLE[fmask]
    return score


# ------------------------------------------------------------------ validation
def _validate(n: int = 20000, seed: int = 1) -> int:
    """Compare pairwise ordering with the practice engine's best_hand. Returns #mismatches."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "/sdk")
    from poker_engine.cards import DECK, best_hand

    rng = np.random.default_rng(seed)
    hands = np.argsort(rng.random((n, 52)), axis=1)[:, :7]
    # Bias half the sample toward rare categories: draw from a small rank/suit subset.
    k = n // 2
    small = np.argsort(rng.random((k, 24)), axis=1)[:, :7]
    pool = rng.permuted(np.tile(np.arange(52), (k, 1)), axis=1)[:, :24]
    hands[:k] = np.take_along_axis(pool, small, axis=1)

    mine = eval7(hands)
    assert np.array_equal(mine, _eval7_generic(hands)), "fast and generic paths disagree"
    ref = [best_hand([DECK[c] for c in h]) for h in hands]
    # Rank the reference tuples to integers so we can compare orderings vectorised.
    uniq = sorted(set(ref))
    pos = {t: i for i, t in enumerate(uniq)}
    ref_rank = np.array([pos[t] for t in ref])
    bad = 0
    perm = rng.permutation(n)
    for a, b in ((np.arange(n), perm), (np.arange(n - 1), np.arange(1, n))):
        bad += int(np.sum(np.sign(mine[a] - mine[b]) != np.sign(ref_rank[a] - ref_rank[b])))
    # Strongest check: the full sort induced by both must agree (all pairs).
    order = np.argsort(ref_rank, kind="stable")
    m_sorted = mine[order]
    r_sorted = ref_rank[order]
    d_m, d_r = np.sign(np.diff(m_sorted)), np.sign(np.diff(r_sorted))
    bad += int(np.sum(d_m != d_r))
    cats = np.bincount(mine >> 20, minlength=9)
    print(f"validated {n} hands vs best_hand: mismatches={bad}; category counts={cats.tolist()}")
    return bad


def _bench(n: int = 1_000_000, seed: int = 2) -> float:
    import time
    rng = np.random.default_rng(seed)
    hands = np.argsort(rng.random((n, 52)), axis=1)[:, :7]
    eval7(hands[:1000])
    t = time.perf_counter()
    eval7(hands)
    dt = time.perf_counter() - t
    print(f"eval7: {n} hands in {dt:.3f}s = {n / dt / 1e6:.2f}M evals/s")
    return n / dt


if __name__ == "__main__":
    assert _validate() == 0
    _bench()
