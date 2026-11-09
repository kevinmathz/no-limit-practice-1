"""Tests for tvalue.py.  Run:  python3 test_tvalue.py   (or pytest test_tvalue.py)

Timing asserts are scaled by a machine-speed factor (a reference numpy op vs its typical
0.5 us), so they stay meaningful on a loaded box; set TV_STRICT_TIMING=1 for raw limits.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tvalue as tv  # noqa: E402

ENGINE_SDK = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "/sdk"


def _engine_icm():
    try:
        sys.path.insert(0, ENGINE_SDK)
        from poker_engine.table import icm  # read-only import, test only
        return icm
    except Exception:                                    # verbatim copy as a fallback
        def icm(stacks, payouts):
            ev = [0.0] * len(stacks)

            def rec(remaining, prob, place):
                if place >= len(payouts) or not remaining:
                    return
                tot = sum(stacks[i] for i in remaining)
                for i in remaining:
                    p = prob * stacks[i] / tot
                    ev[i] += p * payouts[place]
                    rec([j for j in remaining if j != i], p, place + 1)

            rec([i for i, s in enumerate(stacks) if s > 0], 1.0, 0)
            return ev
        return icm


# ------------------------------------------------------------------ correctness
def test_icm_matches_engine():
    icm = _engine_icm()
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in range(400):
        n = int(rng.integers(1, 9))
        S = rng.integers(0, 6000, size=(5, n)).astype(float)
        S[rng.random((5, n)) < 0.3] = 0.0
        S[1] = 0.0
        S[1, 0] = 16000.0                                # one player has everything
        if n >= 2:
            S[2] = 0.0
            S[2, :2] = (7000.0, 9000.0)                  # exactly two alive
        S[3] = 0.0                                       # nobody alive
        for pay in [(50, 30, 20), (50, 30), (50,)]:
            A = tv.icm_batch(S, pay)
            for k in range(5):
                ref = np.array(icm(list(S[k]), pay))
                worst = max(worst, float(np.abs(A[k] - ref).max()))
                for h in range(n):
                    if S[k, h] > 0:
                        worst = max(worst, abs(tv._icm_hero(S[k:k + 1], h, pay)[0] - ref[h]))
    assert worst < 1e-9, worst


def test_icm_1d_and_sum():
    v = tv.icm_batch(np.array([8000, 4000, 2500, 1500.0]))
    assert abs(v.sum() - 100) < 1e-9
    assert np.all(np.diff(v) < 0)


def test_rank_payout_ties():
    R = tv.rank_payout_batch(np.array([[100, 100, 50, 0], [100, 50, 50, 0], [5, 5, 5, 5],
                                       [0, 300, 0, 200]], float))
    assert np.allclose(R[0], [40, 40, 20, 0])
    assert np.allclose(R[1], [50, 25, 25, 0])
    assert np.allclose(R[2], [25, 25, 25, 25])
    assert np.allclose(R[3], [0, 50, 0, 30])


def test_last_hand_rank_mode():
    rows = np.array([[8000, 4000, 2500, 1500, 0], [8000, 4000, 1500, 2500, 0],
                     [8000, 5500, 2500, 0, 0]], float)
    v = tv.value_rows(rows, 3, 0, mode="rank", bb=2000)
    assert np.allclose(v, [0.0, 20.0, 0.0])             # 4th of 4 / 3rd of 4 / busted with 3 left
    v2 = tv.value_rows(rows, 0, 0, mode="rank")
    assert np.allclose(v2, [50, 50, 50])


def test_last_hand_icm_mode_is_engine_settlement():
    icm = _engine_icm()
    rows = np.array([[8000, 4000, 2500, 1500, 0, 0], [9500, 4000, 2500, 0, 0, 0]], float)
    for h in range(3):
        v = tv.value_rows(rows, h, 0, mode="icm")
        for k in range(2):
            alive = [i for i in range(6) if rows[k, i] > 0]
            ref = icm([rows[k, i] for i in alive], (50, 30, 20)[:len(alive)])
            assert abs(v[k] - ref[alive.index(h)]) < 1e-9


def test_bust_places():
    rows = np.array([[0, 9000, 7000, 0], [0, 16000, 0, 0], [0, 6000, 5000, 5000]], float)
    assert np.allclose(tv.value_rows(rows, 0, 0, mode="icm"), [20, 30, 0])
    # co-bust: hero (start 3000) and seat 3 (start 1000) bust in the same hand -> hero 3rd
    start = np.array([3000, 6000, 6000, 1000.0])
    assert tv.value_rows(rows[:1], 0, 5, start=start)[0] == 20
    start2 = np.array([1000, 6000, 6000, 3000.0])
    assert tv.value_rows(rows[:1], 0, 5, start=start2)[0] == 0


def _monotone(mode, H, tol):
    base = np.array([6000, 4000, 3000, 2000, 1000.0])
    xs = np.arange(-900, 4001, 200)
    rows = np.array([base + np.array([-x, 0, 0, 0, x]) for x in xs])   # hero = seat 4 gains
    v = tv.value_rows(rows, 4, H, sched=None, mode=mode, bb=2000)
    assert np.all(np.diff(v) > -tol), (mode, H, np.round(v, 2))
    return v


def test_monotonicity():
    _monotone("icm", 0, 1e-12)
    _monotone("icm", 7, 1e-12)
    _monotone("rank", 0, 1e-12)
    for H in (1, 2, 4, 8, 20):
        _monotone("rank", H, 0.35)                       # MC: monotone up to sampling noise


def test_horizon_limits_and_smoothness():
    S = np.array([[8000, 4000, 2500, 1500.0]])
    icmv = tv.icm_batch(S)[0]
    for h in range(4):
        assert abs(tv.value_rows(S, h, 100, mode="rank")[0] - icmv[h]) < 1e-9
        assert abs(tv.value_rows(S, h, 7, mode="icm")[0] - icmv[h]) < 1e-9
    # rank mode: H=1 already far from the hard step function, values in (0,50)
    v1 = [tv.value_rows(S, h, 1, mode="rank")[0] for h in range(4)]
    assert 5 < v1[3] < 20 and 40 < v1[0] < 50 and abs(sum(v1) - 100) < 1.0


def test_jam_ev_consistency():
    stacks = np.array([6500, 3500, 2000, 0, 0, 0, 0, 0.0])
    contrib = np.array([0, 1000, 2000, 0, 0, 0, 0, 0.0])
    pot = 3000.0
    for mode in ("icm", "rank"):
        ej, ef = tv.jam_ev(stacks, contrib, 0, pot, [(1, 0.0, 0.5), (2, 0.0, 0.5)], 0,
                           mode=mode, bb=2000)
        win = np.array([9500, 3500, 2000, 0, 0, 0, 0, 0.0])
        assert abs(ej - tv.value_rows(win[None], 0, 0, mode=mode)[0]) < 1e-9
        # calling with 100% equity = winning the showdown
        ec, _ = tv.call_ev(np.array([0, 3500, 2000, 0, 0, 0, 0, 0.0]),
                           np.array([6500, 1000, 2000, 0, 0, 0, 0, 0.0]), 2, 9500.0, 0, 1.0, 0,
                           mode=mode)
        # hero (BB, 4000 total) vs jammer (6500): matched 4000, dead SB 1000
        won = np.array([2500, 3500, 9000, 0, 0, 0, 0, 0.0])
        assert abs(ec - tv.value_rows(won[None], 2, 0, mode=mode)[0]) < 1e-9


def test_risk_premium_sign():
    S = np.array([8000, 4000, 2500, 1500.0])
    # under ICM a coinflip never gains: risk premium > 0 for every pair
    for h in range(4):
        for v in range(4):
            if h != v:
                assert tv.risk_premium(S, h, v, 1000, 0, mode="icm") > 0
    # rank mode, last hand: the 4th-place short stack needs much less than 50% to flip
    assert tv.risk_premium(S, 3, 2, 1500, 0, mode="rank") < -0.4


# ------------------------------------------------------------------ timing
def _best_us(f, *a, reps=300):
    f(*a)
    best = 1e18
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(reps):
            f(*a)
        best = min(best, (time.perf_counter() - t) / reps * 1e6)
    return best


def _slowdown():
    if os.environ.get("TV_STRICT_TIMING"):
        return 1.0
    x = np.ones((16, 8))
    return max(1.0, _best_us(np.add, x, x, reps=3000) / 0.5)


def test_timing():
    f = _slowdown()
    rng = np.random.default_rng(3)
    S16 = rng.integers(1, 5000, size=(16, 8)).astype(float)
    t_icm = _best_us(tv.icm_batch, S16)
    rows = np.array([[8000, 4000, 2500, 1500, 0, 0, 0, 0]] * 6, float)
    rows[:, 0] += np.arange(6) * 300
    rows[:, 3] -= np.arange(6) * 200
    t_icm_rows = _best_us(tv.value_rows, rows, 0, 5, None, "icm", 2000)
    t_rank_rows = _best_us(tv.value_rows, rows, 0, 2, None, "rank", 2000)
    print(f"slowdown x{f:.1f}  icm_batch K16n8 {t_icm:.0f}us  value_rows icm K6 {t_icm_rows:.0f}us"
          f"  rank-MC K6 {t_rank_rows:.0f}us")
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    if load > 0.75 and not os.environ.get("TV_STRICT_TIMING"):
        print(f"  (load/cpu = {load:.1f}: wall-clock asserts skipped; see perf instruction"
              " counts in ENDGAME_ANALYSIS.md)")
        return
    assert t_icm < 40 * f
    assert t_icm_rows < 60 * f
    assert t_rank_rows < 150 * f


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
