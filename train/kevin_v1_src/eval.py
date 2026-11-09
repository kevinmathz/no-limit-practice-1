#!/usr/bin/env python3
"""Fixed-seed, multiprocess benchmark for KRD bots (mirrors poker_agent.py `play`).

    python eval.py BOT1 BOT2 ... [--tables 1000] [--seed 0] [--procs 6] [--settle icm|rank]
                   [--hand-cap N] [--json]
    python eval.py BOT_A opp1 opp2 ... --compare BOT_A BOT_B   # paired A->B swap, same seeds

Seating / table seeds are generated exactly like poker_agent.py play() (one master
random.Random(seed): shuffle the order, then rng.random() as the table seed), so with --procs 1
the numbers match `poker_agent.py play ... --seed S` for stateless bots. Tables are split into
chunks over a process pool; results do not depend on --procs (each table has its own seed;
agent objects are per process, which only matters for bots that cheat with self/global state).

Per folder: avg_points +- 95% CI (normal approx over tables), places histogram, decisions,
failed decisions (by kind), and act() p50/p99/max ms.
Duplicate folders (same agent.py path) also get one aggregate row: the per-table mean over
that bot's seats, so the CI respects within-table correlation.

--compare A B: runs the field twice on identical seeds/seatings, once with A and once with B
in every seat A holds. Reports both fields and the paired per-table difference (B - A, mean
over the swapped seats) with a 95% CI. Note: decks match only until the two runs diverge
(the engine draws the agent copy from the table RNG per decision), so pairing removes the
seating/first-hands variance but not all of it.

Timing caveat: with --procs > 1, processes compete for cores (WSL/Windows scheduling, turbo,
hyperthreads), so p99/max act ms are noisier and higher than a single-process run. Use
--procs 1 for timing-critical checks. Percentiles come from a log histogram (~5% bins);
max is exact. Timeouts are judged by the engine (decision_ms + allowance, time bank) inside
each worker, so heavy contention can in principle inflate failed decisions of kind 'timeout'.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
REPO = Path(str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "")
for _p in (str(HERE), str(REPO), str(REPO / "sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poker_agent import Stats, load_agent_class  # noqa: E402

NBINS = 400
BIN_PER_DECADE = 50
BIN_LO = 1e-4  # ms


def _bin(ms):
    if ms <= BIN_LO:
        return 0
    return min(NBINS - 1, int(BIN_PER_DECADE * math.log10(ms / BIN_LO)) + 1)


def _bin_hi(b):
    return BIN_LO * 10 ** (b / BIN_PER_DECADE)


def _pct(hist, q):
    tot = sum(hist)
    if not tot:
        return 0.0
    target = min(tot - 1, int(tot * q))
    acc = 0
    for b, c in enumerate(hist):
        acc += c
        if acc > target:
            return _bin_hi(b)
    return _bin_hi(NBINS - 1)


def schedule(nseats, tables, seed):
    """Same draws as poker_agent.play(): per table, shuffle order then rng.random()."""
    rng = random.Random(seed)
    out = []
    for t in range(tables):
        order = list(range(nseats))
        rng.shuffle(order)
        out.append((t, order, rng.random()))
    return out


def _table_cls(settle):
    if settle == "rank":
        from rank_table import RankTable
        return RankTable
    from poker_engine import Table
    return Table


def _run_field(folders, sched, settings, settle):
    from poker_engine.table import SeatPlayer
    Table = _table_cls(settle)
    stats = [Stats() for _ in folders]
    copies = [[load_agent_class(f)() for _ in range(2)] for f in folders]
    pts = [[] for _ in folders]
    places = [Counter() for _ in folders]
    hist = [[0] * NBINS for _ in folders]
    tmax = [0.0] * len(folders)
    for t, order, tseed in sched:
        players = [SeatPlayer(copies[k], stats[k]) for k in order]
        table = Table(players, f"t{t}", random.Random(tseed), settings)
        p, plc = table.run()
        for seat, k in enumerate(order):
            pts[k].append(p[seat])
            places[k][plc[seat]] += 1
        for k, st in enumerate(stats):  # fold timings into histograms, keep memory flat
            h = hist[k]
            for ms in st.times:
                h[_bin(ms)] += 1
                if ms > tmax[k]:
                    tmax[k] = ms
            st.times.clear()
    return [dict(points=pts[k], places=dict(places[k]), decisions=stats[k].decisions,
                 failed=dict(stats[k].failed), errors=stats[k].errors, hist=hist[k],
                 tmax=tmax[k], notes=sorted(stats[k].notes)) for k in range(len(folders))]


def _worker(args):
    fields, sched, settings, settle = args
    return [_run_field(f, sched, settings, settle) for f in fields]


def _merge(parts):
    """parts: list over chunks of per-seat dicts (chunks in table order)."""
    out = []
    for k in range(len(parts[0])):
        m = dict(points=[], places=Counter(), decisions=0, failed=Counter(), errors=[],
                 hist=[0] * NBINS, tmax=0.0, notes=set())
        for part in parts:
            r = part[k]
            m["points"] += r["points"]
            m["places"].update(r["places"])
            m["decisions"] += r["decisions"]
            m["failed"].update(r["failed"])
            if len(m["errors"]) < 5:
                m["errors"] += r["errors"][: 5 - len(m["errors"])]
            m["hist"] = [a + b for a, b in zip(m["hist"], r["hist"])]
            m["tmax"] = max(m["tmax"], r["tmax"])
            m["notes"].update(r["notes"])
        out.append(m)
    return out


def mean_ci(xs):
    n = len(xs)
    if not n:
        return 0.0, 0.0
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, 1.96 * math.sqrt(var / n)


def _row(name, seats, merged_seats):
    """Build a report row from one or more seats (aggregate = per-table mean over seats)."""
    ntab = len(merged_seats[0]["points"])
    per_table = [sum(m["points"][t] for m in merged_seats) / len(merged_seats) for t in range(ntab)]
    mu, ci = mean_ci(per_table)
    places, failed, hist = Counter(), Counter(), [0] * NBINS
    dec, tmax, errors, notes = 0, 0.0, [], set()
    for m in merged_seats:
        places.update(m["places"])
        failed.update(m["failed"])
        hist = [a + b for a, b in zip(hist, m["hist"])]
        dec += m["decisions"]
        tmax = max(tmax, m["tmax"])
        errors += m["errors"]
        notes |= set(m["notes"])
    return {
        "folder": name, "seats": seats, "tables": ntab,
        "avg_points": round(mu, 3), "ci95": round(ci, 3),
        "places": {int(k): v for k, v in sorted(places.items())},
        "decisions": dec, "failed_decisions": sum(failed.values()), "failed_by_kind": dict(failed),
        "act_ms_p50": round(_pct(hist, 0.50), 4), "act_ms_p99": round(_pct(hist, 0.99), 4),
        "act_ms_max": round(tmax, 4), "sample_errors": errors[:5], "notes": sorted(notes),
        "_per_table": per_table,
    }


def report_field(folders, merged):
    rows = [_row(f, [k], [merged[k]]) for k, f in enumerate(folders)]
    groups = {}
    for k, f in enumerate(folders):
        groups.setdefault(str((Path(f) / "agent.py").resolve()), []).append(k)
    aggs = []
    for path, ks in groups.items():
        if len(ks) > 1:
            aggs.append(_row(folders[ks[0]] + f" (x{len(ks)} aggregate)", ks, [merged[k] for k in ks]))
    return rows, aggs


def _strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def print_rows(title, rows, aggs):
    print(f"\n== {title}")
    hdr = f"{'folder':<34}{'avg_pts':>8}{'±ci95':>8}  {'places 1..8':<34}{'fail':>5}{'p50ms':>8}{'p99ms':>8}{'maxms':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows + aggs:
        n = max([8] + list(r["places"]))
        pl = " ".join(str(r["places"].get(i, 0)) for i in range(1, n + 1))
        name = r["folder"] if len(r["folder"]) <= 33 else "…" + r["folder"][-32:]
        print(f"{name:<34}{r['avg_points']:>8.2f}{r['ci95']:>8.2f}  {pl:<34}{r['failed_decisions']:>5}"
              f"{r['act_ms_p50']:>8.3f}{r['act_ms_p99']:>8.3f}{r['act_ms_max']:>8.2f}")
    for r in rows + aggs:
        if r["sample_errors"]:
            print(f"  errors {r['folder']}: {r['sample_errors']}")


def run(fields, tables, seed, procs, settings, settle):
    nseats = len(fields[0])
    sched = schedule(nseats, tables, seed)
    procs = max(1, min(procs, tables))
    if procs == 1:
        parts = [_worker((fields, sched, settings, settle))]
    else:
        nchunks = min(tables, procs * 4)
        size = math.ceil(tables / nchunks)
        chunks = [sched[i:i + size] for i in range(0, tables, size)]
        with ProcessPoolExecutor(max_workers=procs) as ex:
            parts = list(ex.map(_worker, [(fields, c, settings, settle) for c in chunks]))
    return [_merge([p[i] for p in parts]) for i in range(len(fields))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="bot folders, one per seat (2-8)")
    ap.add_argument("--tables", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--settle", choices=["icm", "rank"], default="icm")
    ap.add_argument("--hand-cap", type=int, default=None)
    ap.add_argument("--variant", choices=["keep-reveal-discard", "reference-holdem"], default=None)
    ap.add_argument("--decision-ms", type=int, default=None)
    ap.add_argument("--time-bank-ms", type=int, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not 2 <= len(a.folders) <= 8:
        sys.exit("need 2 to 8 folders (one per seat)")
    settings = {k: v for k, v in dict(hand_cap=a.hand_cap, variant=a.variant, decision_ms=a.decision_ms,
                                      time_bank_ms=a.time_bank_ms).items() if v is not None}
    folders = list(a.folders)
    fields = [folders]
    if a.compare:
        A, B = a.compare
        ra = (Path(A) / "agent.py").resolve()
        swap = [k for k, f in enumerate(folders) if (Path(f) / "agent.py").resolve() == ra]
        if not swap:
            sys.exit(f"--compare: {A} is not one of the seats")
        fields.append([B if k in swap else f for k, f in enumerate(folders)])
    res = run(fields, a.tables, a.seed, a.procs, settings, a.settle)

    out = {"tables": a.tables, "seed": a.seed, "settle": a.settle, "settings": settings}
    rows, aggs = report_field(fields[0], res[0])
    out["results"] = _strip(rows)
    out["aggregates"] = _strip(aggs)
    if a.compare:
        rows_b, aggs_b = report_field(fields[1], res[1])
        out["results_B"] = _strip(rows_b)
        out["aggregates_B"] = _strip(aggs_b)
        diffs = []
        for t in range(a.tables):
            pa = sum(res[0][k]["points"][t] for k in swap) / len(swap)
            pb = sum(res[1][k]["points"][t] for k in swap) / len(swap)
            diffs.append(pb - pa)
        mu, ci = mean_ci(diffs)
        ma, _ = mean_ci([sum(res[0][k]["points"][t] for k in swap) / len(swap) for t in range(a.tables)])
        mb, _ = mean_ci([sum(res[1][k]["points"][t] for k in swap) / len(swap) for t in range(a.tables)])
        out["compare"] = {"A": A, "B": B, "seats": swap, "avg_A": round(ma, 3), "avg_B": round(mb, 3),
                          "diff_B_minus_A": round(mu, 3), "ci95": round(ci, 3)}
    if a.json:
        print(json.dumps(out, indent=2))
        return
    print(f"tables={a.tables} seed={a.seed} settle={a.settle} settings={settings} procs={a.procs}")
    print_rows("field" + (" A" if a.compare else ""), rows, aggs)
    if a.compare:
        print_rows("field B", rows_b, aggs_b)
        c = out["compare"]
        print(f"\nPAIRED {c['B']} vs {c['A']} (seats {c['seats']}): avg A={c['avg_A']:.3f}  B={c['avg_B']:.3f}  "
              f"diff B-A = {c['diff_B_minus_A']:+.3f} ± {c['ci95']:.3f} (95% CI)")


if __name__ == "__main__":
    main()
