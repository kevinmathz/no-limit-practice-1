"""True act() cost: record real states, time each 15x, keep the min. Usage: bench.py BOTDIR [tables]"""
import sys, time, random, collections
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + ""
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/sdk")
import poker_agent as PA
from poker_engine import Table
from poker_engine.table import SeatPlayer
A = PA.load_agent_class(sys.argv[1])
rec = []
class R(A):
    def act(self, s, m):
        rec.append((s, m)); return A.act(self, s, m)
opp = ["v19", "v13", "v12", "kevin_v1", "examples/check_or_fold", "v21" if __import__("os").path.isdir(REPO + "/v21") else "v10", "v18"]
for t in range(int(sys.argv[2]) if len(sys.argv) > 2 else 10):
    players = [SeatPlayer([R()], PA.Stats())] + [SeatPlayer([PA.load_agent_class(REPO + "/" + o)()], PA.Stats()) for o in opp]
    Table(players, f"t{t}", random.Random(t), {"time_bank_ms": 1e9}).run()
ag = A(); by = collections.defaultdict(list)
for s, m in rec:
    b = 1e9
    for _ in range(15):
        t = time.perf_counter(); ag.act(s, m); b = min(b, time.perf_counter() - t)
    br = s.phase if s.phase != "betting" else ("pre" if not s.board else "post")
    by[br].append(b * 1e3); by["ALL"].append(b * 1e3)
for k, v in sorted(by.items()):
    v.sort(); n = len(v)
    print(f"{k:7s} n={n:5d} p50={v[n//2]:.3f} p99={v[int(n*.99)]:.3f} p99.8={v[min(n-1,int(n*.998))]:.3f} max={v[-1]:.3f} ms")
