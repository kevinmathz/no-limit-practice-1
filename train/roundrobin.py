"""All bots vs each other: each table seats the --fixed bots + the rest drawn at random from the pool.
Usage: python train/roundrobin.py --tables 3000 --seed 1 v0 v2 v3 ...
"""
import argparse, random, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "sdk"))
import poker_agent as pa  # noqa: E402
from poker_engine.table import Table, SeatPlayer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("bots", nargs="+")
ap.add_argument("--tables", type=int, default=3000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--fixed", nargs="*", default=["examples/check_or_fold"], help="bots seated at every table")
a = ap.parse_args()
pool = a.bots
names = pool + a.fixed
cls = {b: pa.load_agent_class(b) for b in names}
copies = {b: [cls[b]() for _ in range(2)] for b in names}
stats = {b: pa.Stats() for b in names}
pts = {b: [] for b in names}
places = {b: Counter() for b in names}
rng = random.Random(a.seed)
for t in range(a.tables):
    seated = rng.sample(pool, 8 - len(a.fixed)) + a.fixed
    rng.shuffle(seated)
    table = Table([SeatPlayer(copies[b], stats[b]) for b in seated], f"t{t}", random.Random(rng.random()))
    p, pl = table.run()
    for i, b in enumerate(seated):
        pts[b].append(p[i]); places[b][pl[i]] += 1
print(f"{'bot':<14}{'tables':>7}{'pts':>7}{'1st':>6}{'2nd':>6}{'3rd':>6}{'out':>6}{'fails':>7}{'p99ms':>7}")
for b in sorted(names, key=lambda b: -sum(pts[b]) / max(1, len(pts[b]))):
    n = len(pts[b]); c = places[b]; ts = sorted(stats[b].times) or [0]
    out = sum(v for k, v in c.items() if k > 3) / n
    print(f"{b.replace('examples/',''):<14}{n:>7}{sum(pts[b])/n:>7.1f}{c[1]/n:>6.0%}{c[2]/n:>6.0%}{c[3]/n:>6.0%}{out:>6.0%}"
          f"{sum(stats[b].failed.values()):>7}{ts[int(len(ts)*0.99)]:>7.3f}")
