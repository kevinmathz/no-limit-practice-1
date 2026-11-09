"""Tests for rt_eval / oppmodel / postflop (A5).  Run: python3 test_oppmodel.py
(plain asserts; no pytest needed).  Uses the practice engine (local only) for real event streams.
"""
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.abspath(os.path.join(HERE, "..", "..", "no-limit-practice-1"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PRACTICE, "sdk"))
sys.path.insert(0, PRACTICE)

import numpy as np  # noqa: E402

from poker_agent_sdk import Action, Config, Event, LegalAction, SeatInfo, State  # noqa: E402
from poker_engine.cards import DECK, best_hand  # noqa: E402

import rt_eval  # noqa: E402
import oppmodel as om  # noqa: E402
import postflop as pf  # noqa: E402

RESULTS = []


def report(name, ok, info=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0.0


# ------------------------------------------------------------------ rt_eval
def test_eval():
    rng = random.Random(7)
    rows = []
    for n in [5, 6, 7] * 2000 + [7] * 4000:
        h = rng.sample(DECK, n)
        rows.append((rt_eval.ev(h), best_hand(h)))
    rows.sort(key=lambda x: x[0])
    bad = sum(1 for (a, x), (b, y) in zip(rows, rows[1:]) if (a < b) != (x < y) or (a == b) != (x == y))
    bad += sum(1 for a, x in rows if (a >> 20) != x[0])
    report("rt_eval matches cards.best_hand", bad == 0, f"({len(rows)} hands, {bad} mismatches)")
    hands = [rng.sample(range(52), 7) for _ in range(100000)]
    t0 = time.perf_counter()
    for h in hands:
        rt_eval.ev_ids(h)
    t1 = time.perf_counter()
    report("rt_eval speed", True, f"{len(hands) / (t1 - t0) / 1000:.0f} evals/ms "
           f"(load avg {os.getloadavg()[0]:.1f} on {os.cpu_count()} cpus)")


# ------------------------------------------------------------------ synthetic stream
def E(kind, seat=None, amount=None, cards=()):
    return Event(kind, seat, amount, tuple(cards))


def test_synthetic():
    m = om.OppModel(8, 2000)
    ev = [E("hand", amount=1), E("blind", 1, 10), E("blind", 2, 20)]
    ev += [E("reveal", s, cards=(c,)) for s, c in [(0, "As"), (1, "Kd"), (2, "7c"), (3, "2h"), (4, "Ah")]]
    ev += [E("raise", 3, 60),        # seat 3 opens (pfr)
           E("raise", 4, 2000),      # seat 4 jams (all-in raise)
           E("fold", 5), E("fold", 6), E("fold", 7),
           E("call", 0, 2000),       # seat 0 calls the jam
           E("fold", 1),             # folds to raise, folds to jam
           E("fold", 2),
           E("fold", 3),
           E("board", cards=("Qs", "Jd", "3c")), E("board", cards=("9h",)), E("board", cards=("4s",)),
           E("show", 0, cards=("Ad", "As")),   # pair, reveal As
           E("show", 4, cards=("5c", "Ah")),   # revealed A high, hidden 5
           E("win", 0, 4090)]
    stacks = {0: 4090, 1: 1990, 2: 1980, 3: 1940, 4: 0, 5: 2000, 6: 2000, 7: 2000}
    ev += [E("stack", s, a) for s, a in stacks.items()]
    # deliver in random chunks, round-tripping through bytes each time (like act() calls)
    b = om.dump(m)
    i = 0
    rng = random.Random(3)
    while i < len(ev):
        j = i + rng.randint(1, 4)
        b = om.dump(om.update(b, ev[i:j], 0))
        i = j
    m = om.load(b)
    f4, f0, f1, f3 = (om.action_freqs(m, s) for s in (4, 0, 1, 3))
    c = lambda s, k: m.c[s * om.NC + k]  # noqa: E731
    ok = (c(4, om.C_JAM) == 1 and c(4, om.C_PFR) == 1 and c(3, om.C_PFR) == 1 and c(3, om.C_JAM) == 0
          and c(0, om.C_AICALL) == 1 and c(0, om.C_AIOPP) == 1 and c(0, om.C_VPIP) == 1
          and c(1, om.C_FRFOLD) == 1 and c(1, om.C_AIOPP) == 1 and c(1, om.C_VPIP) == 0
          and all(c(s, om.C_HANDS) == 1 for s in range(8))
          and c(0, om.C_SDPAIR) == 1 and c(4, om.C_SDHI) == 1 and c(4, om.C_JSD) == 1
          and c(0, om.C_MAT + 12 * 13 + 12) == 1 and c(4, om.C_MAT + 12 * 13 + 3) == 1
          and c(4, om.C_HIST + 12) == 1 and c(3, om.C_HIST + 0) == 1
          and om.reveal_counts(m, 4, 12) == (1, 0, 0) and om.reveal_counts(m, 0) == (0, 0, 1)
          and m.t[om.NG + 4 * om.NT + om.T_STK] == 0)
    report("synthetic stream counters", ok,
           f"(jam4={f4['jam']:.3f} calljam0={f0['call_jam']:.3f} f2r1={f1['fold_to_raise']:.3f})")
    # garbage -> fresh
    good = True
    for g in (b"", b"xx", b"OM\x01\x08" + b"\x00" * 5, os.urandom(3000), b"OM\x09\x08" + b"\0" * 3500, None):
        mm = om.load(g)
        good &= isinstance(mm, om.OppModel) and sum(mm.c) == 0
        mm = om.update(g, ev, 0)
        good &= mm.c[4 * om.NC + om.C_PFR] == 1 and mm.c[0 * om.NC + om.C_HANDS] == 1
    report("garbage memory -> fresh model", good)


# ------------------------------------------------------------------ posterior
def test_posterior():
    rng = np.random.default_rng(0)
    m = om.OppModel(8, 2000)
    ok = True
    worst = 0.0
    for trial in range(200):
        rc = DECK[rng.integers(52)]
        prior = rng.random(169) * (rng.random(169) < 0.6)
        post = om.reveal_posterior(m, 1, rc, prior, action=int(rng.integers(3)))
        r = rt_eval.CARD_ID[rc] >> 2
        ok &= post.shape == (169,) and abs(post.sum() - 1) < 1e-9 and (post >= 0).all()
        ok &= post[~om.CLASS_HAS_RANK[r]].sum() == 0
        if trial < 50:  # no data, no action -> equals renormalised prior
            p0 = prior * om.CLASS_HAS_RANK[r]
            if p0.sum() > 0:
                p0 = p0 / p0.sum()
                q = om.reveal_posterior(m, 1, rc, prior, action=0, floor=0.0)
                worst = max(worst, float(np.abs(q - p0).max()))
    report("posterior shape/normalisation/impossible=0", ok)
    report("posterior == prior without data", worst < 1e-9, f"(max diff {worst:.2e})")
    # evidence moves it, but bounded by k=15: seat 2 always reveals a K with a hidden A (lo)
    prior = np.ones(169)
    base = om.reveal_posterior(m, 2, "Kd", prior)
    shows = []
    for n in (5, 15, 60, 250):
        mm = om.OppModel(8, 2000)
        for _ in range(n):
            om.update(mm, [E("hand"), E("reveal", 2, cards=("Kd",)), E("show", 2, cards=("Ac", "Kd")),
                           E("stack", 2, 2000)], 0)
        p = om.reveal_posterior(mm, 2, "Kd", prior)
        lo = sum(p[om.CLS_RHS[11, 12, s]] for s in (0, 1))
        shows.append(round(float(lo), 3))
    b_lo = sum(base[om.CLS_RHS[11, 12, s]] for s in (0, 1))
    mono = all(a < b for a, b in zip(shows, shows[1:])) and shows[0] > b_lo
    report("posterior grows with evidence (AK given K revealed)", mono, f"prior {b_lo:.3f} -> {shows}")
    w = om.class_post_to_cards(base, "Kd", ("As", "Kd", "2c"))
    ok = w.shape == (52,) and abs(w.sum() - 1) < 1e-9 and w[rt_eval.CARD_ID["As"]] == 0 \
        and w[rt_eval.CARD_ID["Kd"]] == 0
    report("class_post_to_cards", ok)
    ts = []
    for _ in range(300):
        t0 = time.perf_counter_ns()
        om.reveal_posterior(mm, 2, "Kd", prior, action=2)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    report("reveal_posterior timing", True, f"p50 {pct(ts, .5):.3f} ms p99 {pct(ts, .99):.3f} ms")


# ------------------------------------------------------------------ engine runs
TIMES = {"act": [], "post": [], "mem": []}
PRIOR = np.ones(169)


class ModelBot:
    """Maintains the opponent model in memory; uses postflop_decide after the flop."""

    def act(self, state, memory):
        t0 = time.perf_counter_ns()
        m = om.update(memory, state.events, state.seat, state)
        t1 = time.perf_counter_ns()
        if state.phase == "choose":
            c = state.cards
            ranks = ["23456789TJQKA".index(x[0]) for x in c]
            o = sorted(range(3), key=lambda i: -ranks[i])
            act = Action.choose(o[0], o[1])
        elif not state.board:
            r = state.can("raise")
            if state.to_call <= 3 * state.blinds[1]:
                act = pf.call_or_check(state)
                if r and random.random() < 0.15:
                    act = pf.raise_to(state, 3 * state.blinds[1])
            else:
                act = pf.check_or_fold(state) if random.random() < 0.6 else pf.call_or_check(state)
        else:
            infos = []
            rev = state.variant.get("revealed", {})
            for s in state.seats:
                if s.seat == state.seat or s.status not in ("active", "all_in"):
                    continue
                rc = rev.get(s.seat)
                if rc is None:
                    continue
                post = om.reveal_posterior(m, s.seat, rc, PRIOR)
                fr = om.action_freqs(m, s.seat)
                infos.append({"seat": s.seat, "reveal": rc, "post": post,
                              "agg": fr["aggression"], "fold_to_raise": fr["fold_to_raise"]})
            tp = time.perf_counter_ns()
            act = pf.postflop_decide(state, state.cards, state.board, infos, 0.0)
            TIMES["post"].append((time.perf_counter_ns() - tp) / 1e6)
        t2 = time.perf_counter_ns()
        out = om.dump(m)
        t3 = time.perf_counter_ns()
        TIMES["mem"].append((t1 - t0 + t3 - t2) / 1e6)
        TIMES["act"].append((t3 - t0) / 1e6)
        return act, out


def _load_example(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"ex_{name}", os.path.join(PRACTICE, "examples", name, "agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent


def independent_counts(events, ns):
    """Ground truth from the raw event list: hands, vpip, pfr, showdowns, reveals."""
    hands = [0] * ns
    vpip = [0] * ns
    pfr = [0] * ns
    sd = [0] * ns
    revn = [0] * ns
    bad_link = 0
    pre = True
    fv, fp, rev = set(), set(), {}
    for e in events:
        if e.kind == "hand":
            pre, fv, fp, rev = True, set(), set(), {}
        elif e.kind == "board":
            pre = False
        elif e.kind in ("call", "raise") and pre:
            fv.add(e.seat)
            if e.kind == "raise":
                fp.add(e.seat)
        elif e.kind == "reveal":
            revn[e.seat] += 1
            rev[e.seat] = e.cards[0]
        elif e.kind == "show":
            sd[e.seat] += 1
            bad_link += rev.get(e.seat) != e.cards[1]
        elif e.kind == "stack":
            hands[e.seat] += 1
            vpip[e.seat] += e.seat in fv
            pfr[e.seat] += e.seat in fp
    return hands, vpip, pfr, sd, revn, bad_link


def test_engine(n_tables=6):
    from poker_engine import Table
    from poker_engine.table import SeatPlayer
    from poker_agent import Stats
    names = ["calling_station", "min_raiser", "random_bot", "check_or_fold", "random_bot", "calling_station",
             "min_raiser"]
    classes = [_load_example(n) for n in names]
    rng = random.Random(11)
    all_ok = True
    max_mem = 0
    fails = 0
    timeouts = 0
    errs = []
    decisions = 0
    hands_played = []
    for t in range(n_tables):
        stats = [Stats() for _ in range(8)]
        seat_of_model = rng.randrange(8)
        players = []
        k = 0
        for s in range(8):
            if s == seat_of_model:
                players.append(SeatPlayer([ModelBot(), ModelBot()], stats[s]))
            else:
                players.append(SeatPlayer([classes[k]()], stats[s]))
                k += 1
        table = Table(players, f"t{t}", random.Random(rng.random()))
        table.run()
        hands_played.append(table.hand_number)
        st = stats[seat_of_model]
        fails += sum(v for kk, v in st.failed.items() if kk != "timeout")
        timeouts += st.failed.get("timeout", 0)
        errs += [e for e in st.errors if not e.startswith("timeout")]
        decisions += st.decisions
        mem = table.memory[seat_of_model]
        max_mem = max(max_mem, len(mem))
        seen = table.seen[seat_of_model]
        # 1) incremental (through memory bytes) == one-shot on the same events
        ref = om.update(om.OppModel(8, 2000), table.events[:seen], seat_of_model)
        same = om.dump(ref) == mem
        # 2) one-shot over the whole table vs independent ground truth
        full = om.update(om.OppModel(8, 2000), table.events, seat_of_model)
        h, v, p, sd, rv, bad = independent_counts(table.events, 8)
        c = lambda s, i: full.c[s * om.NC + i]  # noqa: E731
        truth = all(c(s, om.C_HANDS) == h[s] and c(s, om.C_VPIP) == v[s] and c(s, om.C_PFR) == p[s]
                    and c(s, om.C_SD) == sd[s] and c(s, om.C_REVN) == rv[s]
                    and c(s, om.C_JAM) <= c(s, om.C_PFR) and c(s, om.C_AICALL) <= c(s, om.C_AIOPP)
                    and c(s, om.C_FRFOLD) <= c(s, om.C_FROPP)
                    and sum(full.c[s * om.NC + om.C_MAT: s * om.NC + om.C_MAT + 169]) == sd[s]
                    for s in range(8)) and bad == 0
        all_ok &= same and truth
        if not (same and truth):
            print("   table", t, "incremental==oneshot", same, "truth", truth, "badlink", bad)
    report("engine streams: incremental==one-shot, counts==ground truth", all_ok,
           f"({n_tables} tables, hands {hands_played})")
    report("engine: no failed decisions (postflop legal)", fails == 0, f"({decisions} decisions, {fails} illegal/exception {errs[:3]}, {timeouts} timeouts)")
    report("memory size < 8 KB after full tables", max_mem < 8192, f"({max_mem} bytes)")
    # print an example model summary
    s = (seat_of_model + 1) % 8
    fr = om.action_freqs(full, s)
    print("   example seat", s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fr.items()})


# ------------------------------------------------------------------ synthetic postflop fuzz
def random_state(rng):
    n = rng.randint(2, 8)
    deck = list(DECK)
    rng.shuffle(deck)
    nb = rng.choice((3, 4, 5))
    board = tuple(deck.pop() for _ in range(nb))
    me = rng.randrange(n)
    seats = []
    live = []
    for s in range(n):
        status = rng.choice(("active", "active", "active", "folded", "all_in"))
        if s == me:
            status = "active"
        stack = 0 if status == "all_in" else rng.randint(1, 6000)
        bet = rng.choice((0, 0, rng.randint(1, 3000)))
        seats.append(SeatInfo(s, stack, bet, status, (deck.pop(),)))
        if status in ("active", "all_in") and s != me:
            live.append(s)
    if not live:
        live = [(me + 1) % n]
        seats[live[0]] = SeatInfo(live[0], 500, 100, "active", seats[live[0]].shown_cards)
    current = max(x.bet for x in seats)
    my = seats[me]
    to_call = min(current - my.bet, my.stack)
    la = []
    if to_call > 0:
        la += [LegalAction("fold"), LegalAction("call", to_call, to_call)]
    else:
        la.append(LegalAction("check"))
    others = [s for s in live if seats[s].status == "active"]
    if my.stack > to_call and others:
        mx = my.bet + my.stack
        la.append(LegalAction("raise", min(current + rng.choice((20, 100, 400)), mx), mx))
    cards = (deck.pop(), deck.pop())
    pot = sum(x.bet for x in seats) + rng.randint(0, 5000)
    cfg = Config("fz", me, "keep-reveal-discard", n, 2000, ((10, 20),), 20, 250, (50, 30, 20), 1, 100, 65536)
    st = State(seat=me, cards=cards, board=board, pot=pot, to_call=to_call, legal_actions=la, history=[],
               phase="betting", seats=seats, button=0, blind_level=0, blinds=(10, 20),
               hands_until_next_level=5, hand_number=10, players_remaining=n, bust_order=[], events=[],
               config=cfg, time_bank_ms=100.0,
               variant={"revealed": {s: seats[s].shown_cards[0] for s in range(n)}})
    infos = []
    for s in live:
        kind = rng.random()
        info = {"seat": s, "reveal": seats[s].shown_cards[0], "agg": rng.random(), "fold_to_raise": rng.random()}
        if kind < 0.4:
            info["post"] = np.random.default_rng(rng.randrange(1 << 30)).random(169)
        elif kind < 0.8:
            info["hidden"] = np.random.default_rng(rng.randrange(1 << 30)).random(52)
        infos.append(info)
    return st, cards, board, infos


def legal(st, a):
    la = st.can(a.kind)
    if la is None:
        return False
    if a.kind == "raise":
        return isinstance(a.amount, int) and la.min_amount <= a.amount <= la.max_amount
    return True


def test_postflop_fuzz(n=3000):
    rng = random.Random(5)
    bad = 0
    ts = {1: [], 2: [], 3: []}
    cpu = []
    kinds = {}
    for _ in range(n):
        st, cards, board, infos = random_state(rng)
        rp = rng.choice((0.0, 0.0, 0.05, 0.15))
        c0 = time.thread_time_ns()
        t0 = time.perf_counter_ns()
        a = pf.postflop_decide(st, cards, board, infos, rp)
        dt = (time.perf_counter_ns() - t0) / 1e6
        cpu.append((time.thread_time_ns() - c0) / 1e6)
        ts[min(3, len(infos))].append(dt)
        bad += not legal(st, a)
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    report("postflop fuzz legality", bad == 0, f"({n} states, {bad} illegal, {kinds})")
    allt = ts[1] + ts[2] + ts[3]
    p99 = pct(allt, .99)
    info = " ".join(f"{k}opp p50 {pct(v, .5):.3f}/p99 {pct(v, .99):.3f}" for k, v in ts.items() if v)
    report("postflop timing p99 < 0.45 ms (fuzz, wall)", p99 < 0.45, f"(all p99 {p99:.3f} ms; {info})")
    report("postflop timing p99 < 0.45 ms (fuzz, thread CPU)", pct(cpu, .99) < 0.45,
           f"(cpu p50 {pct(cpu, .5):.3f} p99 {pct(cpu, .99):.3f} ms)")


def test_equity_sanity():
    # nut flush on river vs anything: 1.0; AA vs revealed 2 on dry board ~ high
    w = np.ones(52)
    ids = pf._ids
    e1 = pf.equity(ids(["As", "Ks"]), ids(["Qs", "Js", "Ts", "2d", "3c"]), [rt_eval.CARD_ID["2h"]], [w])
    e2 = pf.equity(ids(["Ah", "Ad"]), ids(["7c", "8d", "Kh"]), [rt_eval.CARD_ID["2h"]], [w], 4000)
    e3 = pf.equity(ids(["2c", "3d"]), ids(["Kc", "Kd", "9h", "5s"]), [rt_eval.CARD_ID["Ks"]], [w], 4000)
    ok = e1 == 1.0 and 0.85 < e2 < 0.99 and e3 < 0.1
    report("equity sanity", ok, f"(royal {e1:.3f}, AA vs 2x {e2:.3f}, 23 vs revealed K on KK board {e3:.3f})")


def main():
    test_eval()
    test_synthetic()
    test_posterior()
    test_equity_sanity()
    test_postflop_fuzz()
    test_engine()
    for k in ("act", "post", "mem"):
        v = TIMES[k]
        print(f"   engine {k:4s}: n={len(v)} p50 {pct(v, .5):.3f} ms p99 {pct(v, .99):.3f} ms max {max(v):.3f} ms")
    p99 = pct(TIMES["post"], .99)
    report("postflop timing p99 < 0.45 ms (engine)", p99 < 0.45, f"({p99:.3f} ms)")
    nf = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nf}/{len(RESULTS)} passed")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
