"""Merge offline outputs into the bot's data file: no-limit-practice-1/kevin_v1/krd_tables.npz.

  python3 build_bundle.py [cfr_tables.npz|cfr_tables_quick.npz]
"""
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

OUT = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "/kevin_v1/krd_tables.npz"


def main():
    cfr_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cfr_tables.npz")
    if not os.path.exists(cfr_path):
        cfr_path = os.path.join(HERE, "cfr_tables_quick.npz")
    cfr = np.load(cfr_path)
    eq_path = os.path.join(HERE, "equity.npz")
    if os.path.exists(eq_path):
        eq = np.load(eq_path)
        hu_eq = eq["hu_eq"].astype(np.float32)
        eq_rand = eq["eq_vs_random"].astype(np.float32)
    else:
        sys.path.insert(0, "/mnt/c/Users/kevin/onedrive/documents/no_limit")
        from game.tables.hu_matchups import EQUITY
        from game.tables.preflop_equity import PREFLOP_EQUITY
        hu_eq = np.array(EQUITY, dtype=np.float32)
        eq_rand = np.array([PREFLOP_EQUITY[c] for c in common.CLASS_LIST], dtype=np.float32)

    # 3-card combo -> (class id, canonical order): key = c0*2704 + c1*52 + c2, ids sorted desc
    keys, cls, orders = [], [], []
    for combo in combinations(range(52), 3):
        ids = tuple(sorted(combo, reverse=True))
        cid, order = common._COMBO_MAP[ids]
        keys.append(ids[0] * 2704 + ids[1] * 52 + ids[2])
        cls.append(cid)
        orders.append(order)
    out = dict(
        players_grid=cfr["players_grid"].astype(np.int16),
        stack_grid=cfr["stack_grid"].astype(np.int16),
        choose=cfr["choose"].astype(np.uint8),
        pf_open=cfr["pf_open"].astype(np.uint8),
        pf_call=cfr["pf_call"].astype(np.uint8),
        posterior=cfr["posterior"].astype(np.float32),
        hu_eq=hu_eq,
        eq_rand=eq_rand,
        combo_key=np.array(keys, dtype=np.int32),
        combo_cls=np.array(cls, dtype=np.int16),
        combo_order=np.array(orders, dtype=np.int8),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB from", cfr_path)
    for k, v in out.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
