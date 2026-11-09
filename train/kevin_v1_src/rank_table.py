"""Engine Table variant whose hand-cap settlement pays by chip RANK instead of ICM.

At the hand cap, survivors are ordered by stack; the payout for each place goes to the
player holding that rank (50/30/20 by default). Tied stacks split the payouts of the places
they jointly occupy and share the best of those place numbers. Busted players are scored
exactly like the engine. The engine itself is not modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "")
for _p in (str(REPO / "sdk"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poker_engine.table import Table  # noqa: E402


def rank_payouts(stacks, payouts):
    """stacks: list of survivor stacks. Returns (points, places) lists aligned with stacks."""
    n = len(stacks)
    order = sorted(range(n), key=lambda i: -stacks[i])
    pts = [0.0] * n
    plc = [0] * n
    k = 0
    while k < n:
        j = k
        while j + 1 < n and stacks[order[j + 1]] == stacks[order[k]]:
            j += 1
        group = order[k:j + 1]
        pool = sum(payouts[p] for p in range(k, j + 1) if p < len(payouts))
        for i in group:
            pts[i] = pool / len(group)
            plc[i] = k + 1
        k = j + 1
    return pts, plc


class RankTable(Table):
    def results(self):
        alive = self.alive()
        if len(alive) <= 1:
            return super().results()
        points = [0.0] * self.n
        places = [0] * self.n
        for k, i in enumerate(self.bust_order):
            place = self.n - k
            places[i] = place
            points[i] = float(self.payouts[place - 1]) if place - 1 < len(self.payouts) else 0.0
        pts, plc = rank_payouts([self.stacks[i] for i in alive], self.payouts[: len(alive)])
        for k, i in enumerate(alive):
            points[i] = pts[k]
            places[i] = plc[k]
        return points, places


if __name__ == "__main__":
    assert rank_payouts([100, 300, 200], (50, 30, 20)) == ([20.0, 50.0, 30.0], [3, 1, 2])
    assert rank_payouts([300, 300, 100, 50], (50, 30, 20)) == ([40.0, 40.0, 20.0, 0.0], [1, 1, 3, 4])
    assert rank_payouts([5, 5, 5], (50, 30, 20)) == ([100 / 3] * 3, [1, 1, 1])
    print("rank_table OK")
