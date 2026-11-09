"""Offline tables for kevin_v2_simple (written fresh).

Outputs <bot>/tables.npz:
  cls52[52,52] int16   class index (0..168) of two card ids (rank*4+suit; rank 0..12 = 2..A)
  eq1[169]     float32 all-in equity vs one random hand
  hu[169,169]  float32 all-in equity of class i vs class j
  prior[13,169] float32 P(kept class | revealed rank) for a "keep best pair, mixed reveal" player
Run: python3 build_tables.py [samples_per_pair]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

OUT = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "/kevin_v2_simple/tables.npz"
RANKS = "23456789TJQKA"

# ---------------------------------------------------------------- classes
CLASSES = []          # (hi, lo, suited) ; pairs suited=False
for hi in range(12, -1, -1):
    for lo in range(hi, -1, -1):
        if hi == lo:
            CLASSES.append((hi, lo, False))
        else:
            CLASSES.append((hi, lo, True))
            CLASSES.append((hi, lo, False))
CIDX = {c: i for i, c in enumerate(CLASSES)}
CLS52 = np.zeros((52, 52), dtype=np.int16)
for a in range(52):
    for b in range(52):
        ra, rb = a >> 2, b >> 2
        hi, lo = max(ra, rb), min(ra, rb)
        CLS52[a, b] = CIDX[(hi, lo, hi != lo and (a & 3) == (b & 3))]
COMBOS = [[] for _ in range(169)]
for a in range(52):
    for b in range(a + 1, 52):
        COMBOS[CLS52[a, b]].append((a, b))
COMBOS = [np.array(c, dtype=np.int64) for c in COMBOS]

# ---------------------------------------------------------------- evaluator
def _bits(m):
    return [r for r in range(12, -1, -1) if m >> r & 1]


STRAIGHT = np.full(8192, -1, dtype=np.int64)
TOP = np.zeros((6, 8192), dtype=np.int64)       # TOP[k][mask] = top-k ranks packed (4 bits each)
HIGH = np.full(8192, -1, dtype=np.int64)
for m in range(8192):
    for h in range(12, 3, -1):
        w = 0b11111 << (h - 4)
        if m & w == w:
            STRAIGHT[m] = h
            break
    else:
        if m & 0b1000000001111 == 0b1000000001111:
            STRAIGHT[m] = 3
    bs = _bits(m)
    HIGH[m] = bs[0] if bs else -1
    for k in range(6):
        v = 0
        for r in bs[:k]:
            v = (v << 4) | r
        v <<= 4 * (k - min(k, len(bs)))
        TOP[k, m] = v


def eval7(cards):
    """cards [N,7] int card ids -> int64 strength (higher better)."""
    cards = np.asarray(cards, dtype=np.int64)
    N = cards.shape[0]
    r = cards >> 2
    s = cards & 3
    cnt = np.zeros((N, 13), dtype=np.int64)
    np.add.at(cnt, (np.arange(N)[:, None], r), 1)
    B = ((cnt > 0) * (1 << np.arange(13))).sum(1)
    sc = np.zeros((N, 4), dtype=np.int64)
    np.add.at(sc, (np.arange(N)[:, None], s), 1)
    fsuit = sc.argmax(1)
    isflush = sc.max(1) >= 5
    FB = (((s == fsuit[:, None]) * (1 << r)).astype(np.int64))
    FBm = np.zeros(N, dtype=np.int64)
    for k in range(7):
        FBm |= FB[:, k]
    key = cnt * 16 + np.arange(13)
    order = np.argsort(-key, axis=1)
    g0 = order[:, 0]
    g1 = order[:, 1]
    c0 = cnt[np.arange(N), g0]
    c1 = cnt[np.arange(N), g1]
    val = np.zeros(N, dtype=np.int64)
    cat = np.zeros(N, dtype=np.int64)
    # high card
    cat[:] = 0
    val[:] = TOP[5, B]
    # pair
    m = (c0 == 2) & (c1 < 2)
    cat[m] = 1
    val[m] = (g0[m] << 12) | TOP[3, B[m] & ~(1 << g0[m])]
    # two pair
    m = (c0 == 2) & (c1 == 2)
    rest = B & ~(1 << g0) & ~(1 << g1)
    cat[m] = 2
    val[m] = (g0[m] << 8) | (g1[m] << 4) | HIGH[rest[m]]
    # trips
    m = (c0 == 3) & (c1 < 2)
    cat[m] = 3
    val[m] = (g0[m] << 8) | TOP[2, B[m] & ~(1 << g0[m])]
    # straight
    st = STRAIGHT[B]
    m = (st >= 0) & (cat < 4)
    cat[m] = 4
    val[m] = st[m]
    # flush
    m = isflush & (cat < 5)
    cat[m] = 5
    val[m] = TOP[5, FBm[m]]
    # full house
    m = (c0 == 3) & (c1 >= 2)
    cat[m] = 6
    val[m] = (g0[m] << 4) | g1[m]
    # quads
    m = c0 == 4
    cat[m] = 7
    val[m] = (g0[m] << 4) | HIGH[B[m] & ~(1 << g0[m])]
    # straight flush
    sf = STRAIGHT[FBm]
    m = isflush & (sf >= 0)
    cat[m] = 8
    val[m] = sf[m]
    return (cat << 24) | val


def verify(n=20000):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "/sdk")
    from poker_engine.cards import best_hand, DECK
    rng = np.random.default_rng(1)
    hands = np.array([rng.choice(52, 7, replace=False) for _ in range(n)])
    mine = eval7(hands)
    ref = [best_hand([DECK[c] for c in h]) for h in hands]
    bad = 0
    for k in range(n - 1):
        a = int(mine[k] > mine[k + 1]) - int(mine[k] < mine[k + 1])
        b = int(ref[k] > ref[k + 1]) - int(ref[k] < ref[k + 1])
        bad += a != b
    return bad


# ---------------------------------------------------------------- equity
def _pair_eq(args):
    i, samples, seed = args
    rng = np.random.default_rng(seed)
    row = np.zeros(169, dtype=np.float32)
    ci = COMBOS[i]
    for j in range(169):
        if j < i:
            continue
        cj = COMBOS[j]
        a = ci[rng.integers(len(ci), size=samples * 2)]
        b = cj[rng.integers(len(cj), size=samples * 2)]
        ok = (a[:, 0] != b[:, 0]) & (a[:, 0] != b[:, 1]) & (a[:, 1] != b[:, 0]) & (a[:, 1] != b[:, 1])
        a, b = a[ok][:samples], b[ok][:samples]
        n = len(a)
        if n == 0:
            row[j] = 0.5
            continue
        dead = np.zeros((n, 52), dtype=bool)
        dead[np.arange(n)[:, None], a] = True
        dead[np.arange(n)[:, None], b] = True
        keys = rng.random((n, 52))
        keys[dead] = 2.0
        board = np.argpartition(keys, 5, axis=1)[:, :5]
        va = eval7(np.concatenate([a, board], 1))
        vb = eval7(np.concatenate([b, board], 1))
        row[j] = ((va > vb).sum() + 0.5 * (va == vb).sum()) / n
    return i, row


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    t0 = time.time()
    print("evaluator mismatches vs engine:", verify(), flush=True)
    x = np.random.default_rng(2).integers(0, 52, (200000, 7))
    t = time.time(); eval7(x); print(f"eval speed {200000 / (time.time() - t) / 1e6:.2f} M/s", flush=True)
    hu = np.zeros((169, 169), dtype=np.float32)
    with Pool(min(7, os.cpu_count())) as pool:
        for i, row in pool.imap_unordered(_pair_eq, [(i, samples, 1000 + i) for i in range(169)]):
            hu[i, i:] = row[i:]
    for i in range(169):
        for j in range(i):
            hu[i, j] = 1.0 - hu[j, i]
        hu[i, i] = 0.5
    w = np.array([len(c) for c in COMBOS], dtype=np.float64)
    eq1 = (hu * w[None, :]).sum(1) / w.sum()
    # prior: keep best pair by eq1, reveal mixed 50/50 (pairs: either card, same rank)
    rng = np.random.default_rng(3)
    prior = np.zeros((13, 169))
    D = 300000
    keys = rng.random((D, 52))
    deal = np.argpartition(keys, 3, axis=1)[:, :3]
    pairs = [(0, 1), (0, 2), (1, 2)]
    cl = np.stack([CLS52[deal[:, a], deal[:, b]] for a, b in pairs], 1)
    best = eq1[cl].argmax(1)
    kc = cl[np.arange(D), best]
    pa = np.array(pairs)[best]
    c1 = deal[np.arange(D), pa[:, 0]]
    c2 = deal[np.arange(D), pa[:, 1]]
    np.add.at(prior, (c1 >> 2, kc), 0.5)
    np.add.at(prior, (c2 >> 2, kc), 0.5)
    prior /= prior.sum(1, keepdims=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, cls52=CLS52, eq1=eq1.astype(np.float32), hu=hu, prior=prior.astype(np.float32))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s; AA vs KK {hu[0, CIDX[(11, 11, False)]]:.3f} "
          f"AKo vs 22 {hu[CIDX[(12, 11, False)], 168]:.3f} eq1 AA {eq1[0]:.3f} 72o {eq1[CIDX[(5, 0, False)]]:.3f}")


if __name__ == "__main__":
    main()
