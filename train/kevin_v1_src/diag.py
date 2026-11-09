"""Attribute our chip results per hand to the kind of first voluntary decision we made."""
import sys, random, collections
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + ""
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/sdk")
import poker_agent as PA
from poker_engine import Table
from poker_engine.table import SeatPlayer
HERE = str(__import__("pathlib").Path(__file__).resolve().parent / "opponents") + "/"
A = PA.load_agent_class(REPO + "/kevin_v1")
core = sys.modules["krd_core"]
log = {}
class Rec(A):
    def act(self, state, memory):
        a, m = A.act(self, state, memory)
        if state.phase == "betting":
            ctx = core.Ctx(state)
            key = (state.config.table_id, state.hand_number)
            street = "pre" if not state.board else "post"
            short = "S" if ctx.eff_bb <= core.SHORT_BB else "D"
            kind = a.kind
            if kind == "raise" and state.can("raise") and a.amount == state.can("raise").max_amount: kind = "jam"
            if kind in ("call", "raise", "jam") and key not in log:
                me = state.seats[state.seat]; frac = state.to_call / max(1, me.stack + me.bet); log[key] = f"{street}-{short}-{kind}" + ("-vsraise" if state.to_call > state.blinds[1] and street == "pre" else "") + (("-big" if frac > 0.25 else "-small") if kind == "call" and street == "pre" else "")
        return a, m
opp = ["tag_krd", "reveal_high", "reveal_random", "jammer", "icm_nit", "reveal_reader", "maniac"]
res = collections.defaultdict(lambda: [0, 0, 0])
for t in range(int(sys.argv[1])):
    players = [SeatPlayer([Rec()], PA.Stats())] + [SeatPlayer([PA.load_agent_class(HERE + o)()], PA.Stats()) for o in opp]
    tb = Table(players, f"t{t}", random.Random(t), {"time_bank_ms": 1e9})
    orig = tb.play_hand
    def ph(tb=tb, orig=orig):
        s0 = tb.stacks[0]
        orig()
        k = log.get((tb.table_id, tb.hand_number), "none")
        r = res[k]; r[0] += 1; r[1] += (tb.stacks[0] - s0) / max(1, tb.s["blinds"][tb.level()][1])
        r[2] += tb.stacks[0] == 0 and s0 > 0
    tb.play_hand = ph
    tb.run()
for k, (n, bb, bust) in sorted(res.items(), key=lambda x: x[1][1]):
    print(f"{k:28s} n={n:5d} total={bb:9.1f}bb per={bb/n:7.2f}bb busts={bust}")
