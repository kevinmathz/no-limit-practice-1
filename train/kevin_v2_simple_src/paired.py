"""Paired comparison: same seeds and seatings, seat 'A' replaced by 'B'. Usage:
   paired.py A B --field f1 f2 ... --tables N --seed S --procs P"""
import sys, random, argparse, statistics
from multiprocessing import Pool
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + ""
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/sdk")
ap = argparse.ArgumentParser(); ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("--field", nargs="+", required=True)
ap.add_argument("--tables", type=int, default=400); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--hand-cap", type=int, default=None); ap.add_argument("--hpl", type=int, default=None)
a = ap.parse_args()

def run(args):
    who, lo, hi = args
    import poker_agent as PA
    from poker_engine import Table
    from poker_engine.table import SeatPlayer
    folders = [who] + [f if f.startswith("/") else REPO + "/" + f for f in a.field]
    cls = [PA.load_agent_class(f) for f in folders]
    out = []
    for t in range(lo, hi):
        rng = random.Random(a.seed * 100003 + t)
        order = list(range(len(folders))); rng.shuffle(order)
        stats = PA.Stats()
        players = [SeatPlayer([cls[k](), cls[k]()], stats) for k in order]
        st = {k: v for k, v in (("hand_cap", a.hand_cap), ("hands_per_level", a.hpl)) if v}
        pts, _ = Table(players, f"t{t}", random.Random(rng.random()), st).run()
        seat = order.index(0)
        out.append((t, pts[seat], sum(stats.failed.values())))
    return out

if __name__ == "__main__":
    ch = [(lo, min(a.tables, lo + max(1, a.tables // (a.procs * 3)))) for lo in range(0, a.tables, max(1, a.tables // (a.procs * 3)))]
    res = {}
    with Pool(a.procs) as p:
        for who in (a.A, a.B):
            r = []
            for part in p.map(run, [(who, lo, hi) for lo, hi in ch]):
                r += part
            res[who] = dict((t, v) for t, v, _ in r)
    ts = sorted(res[a.A])
    d = [res[a.B][t] - res[a.A][t] for t in ts]
    m = statistics.mean(d); se = statistics.stdev(d) / len(d) ** 0.5
    print(f"tables={len(ts)}  A={statistics.mean(res[a.A].values()):.2f}  B={statistics.mean(res[a.B].values()):.2f}  B-A={m:+.2f} ± {1.96*se:.2f} (95%)")
