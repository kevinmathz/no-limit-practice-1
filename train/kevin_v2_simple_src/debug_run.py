"""Run tables and report any exception swallowed inside kevin_v2_simple's decide()."""
import sys, random, traceback, collections
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + ""
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/sdk")
import poker_agent as PA
from poker_engine import Table
from poker_engine.table import SeatPlayer
A = PA.load_agent_class(REPO + "/kevin_v2_simple")
errs = collections.Counter(); sample = {}
orig = A.decide
def dec(self, state, mem, t0):
    try:
        return orig(self, state, mem, t0)
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()
        key = tb[-1] + " @ " + [l for l in tb if "agent.py" in l][-1].strip()
        errs[key] += 1; sample.setdefault(key, "\n".join(tb[-6:]))
        raise
A.decide = dec
opp = sys.argv[2].split(",") if len(sys.argv) > 2 else ["examples/check_or_fold", "examples/min_raiser", "examples/random_bot", "examples/calling_station", "v19", "v13", "kevin_v1"]
for t in range(int(sys.argv[1])):
    players = [SeatPlayer([A()], PA.Stats())] + [SeatPlayer([PA.load_agent_class(REPO + "/" + o)()], PA.Stats()) for o in opp]
    Table(players, f"t{t}", random.Random(t), {"time_bank_ms": 1e9}).run()
print("swallowed exceptions:", sum(errs.values()))
for k, v in errs.most_common(5):
    print(v, k); print(sample[k])
