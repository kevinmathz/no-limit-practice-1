"""Learn v18's players-remaining looseness schedule by cross-entropy search over simulated tables.
Candidate = multiplier per players_remaining n in 2..8 (log-space Gaussian). Fitness = v18's avg points
over N tables vs a fixed field of our best bots (+ check_or_fold), with common random seeds per generation.
Usage: python train/learn_schedule.py --minutes 7
"""
import argparse, json, math, multiprocessing as mp, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
FIELD = ["v12", "v12", "v10", "v13", "v17", "examples/check_or_fold", "v14"]
NS = list(range(2, 9))
_cache = {}


def _setup():
    if _cache:
        return _cache
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "sdk"))
    os.chdir(ROOT)
    import poker_agent as pa
    from poker_engine.table import Table, SeatPlayer
    names = ["v18"] + FIELD
    cls = {b: pa.load_agent_class(b) for b in set(names)}
    _cache.update(pa=pa, Table=Table, SeatPlayer=SeatPlayer, cls=cls,
                  tight=pa._modules[(ROOT / "v18" / "agent.py").resolve()]._tight)
    return _cache


def evaluate(args):
    sched, tables, seed = args
    c = _setup()
    c["tight"].SCHED = {int(k): v for k, v in sched.items()}
    rng = random.Random(seed)
    tot = 0.0
    for t in range(tables):
        seated = ["v18"] + FIELD
        rng.shuffle(seated)
        players = [c["SeatPlayer"]([c["cls"][b]()], c["pa"].Stats()) for b in seated]
        pts, _ = c["Table"](players, f"t{t}", random.Random(rng.random())).run()
        tot += pts[seated.index("v18")]
    return tot / tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=7)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--tables", type=int, default=500)
    a = ap.parse_args()
    mu = [0.0] * len(NS)
    sigma = [0.6] * len(NS)
    rng = random.Random(0)
    end = time.time() + a.minutes * 60
    gen = 0
    with mp.Pool(os.cpu_count()) as pool:
        while time.time() < end:
            gen += 1
            cands = [[m + s * rng.gauss(0, 1) for m, s in zip(mu, sigma)] for _ in range(a.pop - 1)] + [list(mu)]
            seed = 1000 + gen  # common random numbers within a generation
            jobs = [({n: round(math.exp(x), 3) for n, x in zip(NS, c)}, a.tables, seed) for c in cands]
            fit = pool.map(evaluate, jobs)
            order = sorted(range(len(cands)), key=lambda i: -fit[i])
            elite = [cands[i] for i in order[: max(2, a.pop // 4)]]
            mu = [sum(e[k] for e in elite) / len(elite) for k in range(len(NS))]
            sigma = [max(0.08, math.sqrt(sum((e[k] - mu[k]) ** 2 for e in elite) / len(elite))) for k in range(len(NS))]
            print(f"gen {gen}: best {fit[order[0]]:.2f}  mean-candidate {fit[-1]:.2f}  "
                  f"mu={ {n: round(math.exp(x), 2) for n, x in zip(NS, mu)} }", flush=True)
        final = {n: round(math.exp(x), 3) for n, x in zip(NS, mu)}
        base = {n: 1.0 for n in NS}
        res = pool.map(evaluate, [(final, 250, 9000 + k) for k in range(16)] + [(base, 250, 9000 + k) for k in range(16)])
    learned, baseline = sum(res[:16]) / 16, sum(res[16:]) / 16
    print(f"validation (4000 tables, same seeds): learned {learned:.2f} vs v12 schedule (all 1.0) {baseline:.2f}")
    (ROOT / "v18" / "schedule.json").write_text(json.dumps({str(k): v for k, v in final.items()}))
    print("wrote v18/schedule.json", final)


if __name__ == "__main__":
    main()
