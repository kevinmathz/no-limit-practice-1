"""Noise-robust act() cost: record real states, re-time each 15x, keep the min."""
import sys, time, random, collections
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + ""
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/sdk")
import poker_agent as PA
from poker_engine import Table
from poker_engine.table import SeatPlayer
bot = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else REPO + "/kevin_v1"
A = PA.load_agent_class(bot)
rec = []
class Rec(A):
    def act(self, state, memory):
        rec.append((state, memory))
        return A.act(self, state, memory)
opp = sys.argv[3].split(",") if len(sys.argv) > 3 else ["check_or_fold", "min_raiser", "random_bot", "calling_station", "random_bot", "min_raiser", "calling_station"]
folders = [None] + [REPO + "/examples/" + x if "/" not in x else x for x in opp]
copies = [[Rec()]] + [[PA.load_agent_class(f)()] for f in folders[1:]]
stats = [PA.Stats() for _ in folders]
for t in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
    players = [SeatPlayer(copies[k], stats[k]) for k in range(len(folders))]
    Table(players, f"t{t}", random.Random(t), {"time_bank_ms": 1e9}).run()
ag = A()
by = collections.defaultdict(list)
for st, mem in rec:
    best = 1e9
    for _ in range(15):
        t = time.perf_counter(); ag.act(st, mem); best = min(best, time.perf_counter() - t)
    br = st.phase if st.phase != "betting" else ("pre" if not st.board else "post")
    by[br].append(best * 1000); by["ALL"].append(best * 1000)
for k, v in sorted(by.items()):
    v = sorted(v); print(f"   {k} p99.8={v[min(len(v)-1, int(len(v)*.998))]:.3f}")
    v.sort(); print(f"{k:8s} n={len(v):5d} p50={v[len(v)//2]:.3f} p90={v[int(len(v)*.9)]:.3f} p99={v[int(len(v)*.99)]:.3f} max={v[-1]:.3f} ms")
if len(sys.argv) > 4 and sys.argv[4] == "prof":
    import cProfile, pstats
    posts = [(s, m) for s, m in rec if s.phase == "betting" and s.board]
    pr = cProfile.Profile(); pr.enable()
    for _ in range(5):
        for s, m in posts: ag.act(s, m)
    pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(18)
if len(sys.argv) > 4 and sys.argv[4] == "slowpre":
    import cProfile, pstats
    core = sys.modules["krd_core"]
    tim = []
    for s, m in rec:
        if s.phase == "betting" and not s.board:
            b = 1e9
            for _ in range(5):
                t = time.perf_counter(); ag.act(s, m); b = min(b, time.perf_counter() - t)
            tim.append((b, s, m))
    tim.sort(key=lambda x: -x[0])
    worst = tim[:25]
    for b, s, m in worst[:6]:
        ctx = core.Ctx(s)
        print(f"{b*1000:.3f}ms eff_bb={ctx.eff_bb:.1f} players={ctx.n_players} to_call={s.to_call} hist_raises={sum(e.kind=='raise' for e in s.history)}")
    pr = cProfile.Profile(); pr.enable()
    for _ in range(20):
        for b, s, m in worst: ag.act(s, m)
    pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(14)
