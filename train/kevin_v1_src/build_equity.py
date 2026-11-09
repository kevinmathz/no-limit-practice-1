"""Build preflop all-in equity tables for the KRD pipeline -> equity.npz.

Contents of equity.npz
----------------------
hu_eq        float32 [169,169]  equity of CLASS_LIST[i] vs CLASS_LIST[j] (ties = 1/2).
                                Monte Carlo over uniformly drawn non-conflicting combo pairs
                                of the two classes + random boards. hu_eq[i,j] + hu_eq[j,i] = 1,
                                diagonal = 0.5 exactly.
hu_n         int    samples per off-diagonal pair (std err <= 0.5/sqrt(hu_n)).
eq_vs_random float32 [169,7]    equity share (ties split) of class i vs k = 1..7 random hands.
mw_triples   int    [M,3]       3-way class triples (dealt as random hands, i.e. combo-weighted).
mw_true      float32 [M,3]      Monte Carlo 3-way equity of each seat.
mw_approx    float32 [M,3]      normalised product of pairwise equities
                                (p_i ~ prod_j hu_eq[i,j]; no_limit/game/cfr/preflop_vector.py).
mw_mae       float  mean |mw_true - mw_approx| over all seats;  mw_maxe: max.
mw_note      str    one-line human summary of the multiway check.

Usage: python3 build_equity.py [--hu-n 36000] [--rand-n 30000] [--mw-m 300] [--mw-n 20000]
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import time

import numpy as np

from common import CLASS_LIST, class2
from evaluator_np import eval7

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "equity.npz")
HU_REF = "/mnt/c/Users/kevin/onedrive/documents/no_limit/game/tables/hu_matchups.py"
NC = len(CLASS_LIST)

# All 2-card combos per class: COMBOS[c] -> int array [k, 2]  (k = 6 / 4 / 12).
_by_class: list[list] = [[] for _ in range(NC)]
for _a in range(52):
    for _b in range(_a + 1, 52):
        _by_class[class2(_a, _b)].append((_a, _b))
COMBOS = [np.array(x, dtype=np.int64) for x in _by_class]


def deal_rest(rng, dead: np.ndarray, k: int) -> np.ndarray:
    """For each row of dead [n,d] card ids, deal k distinct cards from the other 52-d."""
    n = dead.shape[0]
    keys = rng.random((n, 52))
    np.put_along_axis(keys, dead, 2.0, axis=1)
    return np.argpartition(keys, k, axis=1)[:, :k]


def showdown_share(hands: np.ndarray, board: np.ndarray) -> np.ndarray:
    """hands [n,p,2], board [n,5] -> pot share [n,p] (ties split)."""
    n, p, _ = hands.shape
    seven = np.concatenate([hands, np.broadcast_to(board[:, None, :], (n, p, 5))], axis=2)
    s = eval7(seven.reshape(n * p, 7)).reshape(n, p)
    win = s == s.max(axis=1, keepdims=True)
    return win / win.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ HU table
def _hu_task(args):
    pairs, n, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for i, j in pairs:
        a, b = COMBOS[i], COMBOS[j]
        both = np.concatenate([np.repeat(a, len(b), 0), np.tile(b, (len(a), 1))], axis=1)
        ok = (both[:, :2, None] != both[:, None, 2:]).all(axis=(1, 2))
        both = both[ok]
        hole = both[rng.integers(len(both), size=n)]
        board = deal_rest(rng, hole, 5)
        share = showdown_share(hole.reshape(n, 2, 2), board)
        out.append((i, j, float(share[:, 0].mean())))
    return out


def build_hu(pool, n: int, seed: int = 11) -> np.ndarray:
    pairs = [(i, j) for i in range(NC) for j in range(i + 1, NC)]
    batch = 64
    tasks = [(pairs[k:k + batch], n, seed + k) for k in range(0, len(pairs), batch)]
    eq = np.full((NC, NC), 0.5, dtype=np.float64)
    for res in pool.imap_unordered(_hu_task, tasks):
        for i, j, e in res:
            eq[i, j], eq[j, i] = e, 1.0 - e
    return eq.astype(np.float32)


# ------------------------------------------------------------------ vs random opponents
def _rand_task(args):
    c, n, seed = args
    rng = np.random.default_rng(seed)
    row = np.zeros(7)
    for k in range(1, 8):
        hero = COMBOS[c][rng.integers(len(COMBOS[c]), size=n)]
        rest = deal_rest(rng, hero, 2 * k + 5)
        hands = np.concatenate([hero[:, None, :], rest[:, :2 * k].reshape(n, k, 2)], axis=1)
        row[k - 1] = showdown_share(hands, rest[:, 2 * k:])[:, 0].mean()
    return c, row


def build_vs_random(pool, n: int, seed: int = 23) -> np.ndarray:
    out = np.zeros((NC, 7), dtype=np.float32)
    for c, row in pool.imap_unordered(_rand_task, [(c, n, seed + c) for c in range(NC)]):
        out[c] = row
    return out


# ------------------------------------------------------------------ multiway check
def _mw_task(args):
    triple, n, seed = args
    rng = np.random.default_rng(seed)
    got, need = [], n
    while need > 0:
        m = int(need * 1.6) + 64
        h = np.stack([COMBOS[c][rng.integers(len(COMBOS[c]), size=m)] for c in triple], axis=1)
        flat = np.sort(h.reshape(m, 6), axis=1)
        ok = (np.diff(flat, axis=1) != 0).all(axis=1)
        got.append(h[ok][:need])
        need -= len(got[-1])
    hands = np.concatenate(got)
    board = deal_rest(rng, hands.reshape(n, 6), 5)
    return showdown_share(hands, board).mean(axis=0)


def pairwise_product(hu: np.ndarray, triple) -> np.ndarray:
    """p_i ~ prod_{j != i} hu[c_i, c_j], normalised (preflop_vector.py:237-258)."""
    s = np.array([np.prod([hu[triple[i], triple[j]] for j in range(3) if j != i])
                  for i in range(3)], dtype=np.float64)
    return s / s.sum()


def build_mw(pool, hu: np.ndarray, m: int, n: int, seed: int = 37):
    rng = np.random.default_rng(seed)
    triples = []
    for _ in range(m):  # deal 3 random hands -> combo-weighted class triple
        d = rng.permutation(52)[:6]
        triples.append(tuple(class2(d[2 * p], d[2 * p + 1]) for p in range(3)))
    triples = np.array(triples, dtype=np.int64)
    true = np.array(pool.map(_mw_task, [(t, n, seed + 1 + k) for k, t in enumerate(triples)]))
    approx = np.array([pairwise_product(hu, t) for t in triples])
    return triples, true.astype(np.float32), approx.astype(np.float32)


# ------------------------------------------------------------------ main
def load_reference():
    spec = importlib.util.spec_from_file_location("hu_matchups_ref", HU_REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(mod.CLASSES) == CLASS_LIST, "class order mismatch with hu_matchups"
    return np.array(mod.EQUITY, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hu-n", type=int, default=36000)
    ap.add_argument("--rand-n", type=int, default=30000)
    ap.add_argument("--mw-m", type=int, default=300)
    ap.add_argument("--mw-n", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    t0 = time.perf_counter()
    times = {}
    with mp.Pool(a.procs) as pool:
        t = time.perf_counter()
        hu = build_hu(pool, a.hu_n)
        times["hu_eq"] = time.perf_counter() - t
        t = time.perf_counter()
        vr = build_vs_random(pool, a.rand_n)
        times["eq_vs_random"] = time.perf_counter() - t
        t = time.perf_counter()
        triples, true, approx = build_mw(pool, hu, a.mw_m, a.mw_n)
        times["mw_check"] = time.perf_counter() - t

    err = np.abs(true - approx)
    mw_mae, mw_maxe = float(err.mean()), float(err.max())
    seat0 = np.abs(true[:, 0] - approx[:, 0])
    mw_note = (f"3-way normalised pairwise-product approx vs MC ({a.mw_m} dealt triples, "
               f"{a.mw_n} samples each, MC se~{0.5 / np.sqrt(a.mw_n):.4f}): "
               f"mean abs err {mw_mae:.4f}, max {mw_maxe:.4f}")
    np.savez(a.out, hu_eq=hu, hu_n=a.hu_n, eq_vs_random=vr, mw_triples=triples,
             mw_true=true, mw_approx=approx, mw_mae=mw_mae, mw_maxe=mw_maxe, mw_note=mw_note,
             class_list=np.array(CLASS_LIST))

    ref = load_reference()
    off = ~np.eye(NC, dtype=bool)
    d = np.abs(hu.astype(np.float64) - ref)[off]
    print("=" * 70)
    print(f"timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in times.items())
          + f", total {time.perf_counter() - t0:.1f}s ({a.procs} procs)")
    print(f"hu_eq: {a.hu_n} samples/pair, max std err {0.5 / np.sqrt(a.hu_n):.4f}; "
          f"AA vs KK {hu[0, 25]:.4f}, AKo vs 22 {hu[2, 168]:.4f}")
    print(f"vs hu_matchups.EQUITY (1500 samples): max abs diff {d.max():.4f}, "
          f"mean abs diff {d.mean():.4f}")
    print(f"eq_vs_random: AA {np.round(vr[0].astype(float), 3).tolist()}; 72o {np.round(vr[CLASS_LIST.index('72o')].astype(float), 3).tolist()}")
    print(mw_note)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
