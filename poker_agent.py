#!/usr/bin/env python3
"""MOCK of the kit's `poker-agent` command:  check | play | pack.

  python poker_agent.py check bot/agent.py
  python poker_agent.py play bot examples/check_or_fold examples/min_raiser --tables 200
  python poker_agent.py pack bot out.zip
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sdk"))

BANNED = {"poker_engine", "pokerkit"}
ALLOWED_3P = {"numpy", "scipy", "pandas", "sklearn", "torch", "lightgbm", "xgboost",
              "onnxruntime", "numba", "eval7", "poker_agent_sdk", "poker_agent_protocol"}
LIMITS = dict(zip_bytes=50 * 2**20, unpacked=200 * 2**20, entries=1000, agent_py=2**20,
              path_len=512, depth=16)


# ---------------- check ----------------
def _check_source(name, src, errors, warnings):
    try:
        tree = ast.parse(src, filename=name)
    except SyntaxError as e:
        errors.append(f"{name}: syntax error: {e}")
        return None
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods = [node.module]
        for m in mods:
            top = m.split(".")[0]
            if top in BANNED:
                errors.append(f"{name}: imports {top} (local-only, upload rejected)")
            elif top not in sys.stdlib_module_names and top not in ALLOWED_3P:
                warnings.append(f"{name}: imports {top}, not in the allowed package list")
    return tree


def check(path: str) -> int:
    p = Path(path)
    errors, warnings = [], []
    if p.suffix == ".zip":
        if p.stat().st_size > LIMITS["zip_bytes"]:
            errors.append("zip over 50 MB")
        with zipfile.ZipFile(p) as z:
            infos = z.infolist()
            if len(infos) > LIMITS["entries"]:
                errors.append("over 1,000 entries")
            if sum(i.file_size for i in infos) > LIMITS["unpacked"]:
                errors.append("over 200 MB unpacked")
            names = [i.filename for i in infos]
            for n in names:
                if len(n) > LIMITS["path_len"] or n.count("/") >= LIMITS["depth"]:
                    errors.append(f"path too long/deep: {n}")
            if "agent.py" not in names:
                errors.append("no agent.py at zip root")
            for n in names:
                if n.endswith(".py"):
                    src = z.read(n)
                    if n == "agent.py" and len(src) > LIMITS["agent_py"]:
                        errors.append("agent.py over 1 MB")
                    tree = _check_source(n, src.decode("utf-8", "replace"), errors, warnings)
                    if n == "agent.py" and tree is not None:
                        _check_agent_class(tree, errors)
    else:
        src = p.read_bytes()
        if len(src) > LIMITS["agent_py"]:
            errors.append("agent.py over 1 MB")
        tree = _check_source(p.name, src.decode("utf-8", "replace"), errors, warnings)
        if tree is not None:
            _check_agent_class(tree, errors)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("check: FAILED" if errors else "check: OK")
    return 1 if errors else 0


def _check_agent_class(tree, errors):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Agent":
            if any(isinstance(f, ast.FunctionDef) and f.name == "act" for f in node.body):
                return
            errors.append("class Agent has no act method")
            return
    errors.append("agent.py defines no class Agent")


# ---------------- pack ----------------
def pack(src_dir: str, out: str) -> int:
    d = Path(src_dir).resolve()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(d.rglob("*")):
            rel = f.relative_to(d)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            if f.is_symlink() or f.is_dir():
                continue
            zi = zipfile.ZipInfo(rel.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, f.read_bytes())
    print(f"packed {out}")
    return check(out)


# ---------------- play ----------------
class Stats:
    def __init__(self):
        self.decisions = 0
        self.failed = Counter()
        self.errors = []
        self.times = []
        self.notes = set()

    def fail(self, kind, msg):
        self.failed[kind] += 1
        if len(self.errors) < 5:
            self.errors.append(f"{kind}: {msg}")


_modules = {}


def load_agent_class(folder: str):
    path = (Path(folder) / "agent.py").resolve()
    if path not in _modules:
        spec = importlib.util.spec_from_file_location(f"bot_{len(_modules)}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # "import once", like production
        _modules[path] = mod
    return _modules[path].Agent


def play(folders, tables, seed, settings):
    from poker_engine import Table
    from poker_engine.table import SeatPlayer

    if not 2 <= len(folders) <= 8:
        sys.exit("play needs 2 to 8 folders (one per seat)")
    rng = random.Random(seed)
    stats = [Stats() for _ in folders]
    copies = [[load_agent_class(f)() for _ in range(2)] for f in folders]  # two copies per seat
    points = [[] for _ in folders]
    places = [Counter() for _ in folders]
    for t in range(tables):
        order = list(range(len(folders)))
        rng.shuffle(order)  # anonymized, random seating
        players = [SeatPlayer(copies[k], stats[k]) for k in order]
        table = Table(players, f"t{t}", random.Random(rng.random()), settings)
        pts, plc = table.run()
        for seat, k in enumerate(order):
            points[k].append(pts[seat])
            places[k][plc[seat]] += 1
    out = []
    for k, f in enumerate(folders):
        ts = sorted(stats[k].times) or [0.0]
        out.append({
            "seat": k,
            "folder": f,
            "avg_points": round(sum(points[k]) / len(points[k]), 3),
            "places": dict(sorted(places[k].items())),
            "decisions": stats[k].decisions,
            "failed_decisions": sum(stats[k].failed.values()),
            "failed_by_kind": dict(stats[k].failed),
            "act_ms_p50": round(ts[len(ts) // 2], 4),
            "act_ms_p99": round(ts[min(len(ts) - 1, int(len(ts) * 0.99))], 4),
            "act_ms_max": round(ts[-1], 4),
            "sample_errors": stats[k].errors,
            "notes": sorted(stats[k].notes),
        })
    print(json.dumps(out, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser(prog="poker_agent.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("file")
    p = sub.add_parser("play")
    p.add_argument("folders", nargs="+")
    p.add_argument("--tables", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-history", action="store_true", help="accepted for kit compatibility")
    p.add_argument("--decision-ms", type=int, default=None)
    p.add_argument("--time-bank-ms", type=int, default=None)
    p.add_argument("--starting-stack", type=int, default=None)
    p.add_argument("--hand-cap", type=int, default=None)
    p.add_argument("--variant", choices=["keep-reveal-discard", "reference-holdem"], default=None)
    k = sub.add_parser("pack")
    k.add_argument("dir")
    k.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(check(a.file))
    if a.cmd == "pack":
        sys.exit(pack(a.dir, a.out))
    settings = {key: v for key, v in dict(
        decision_ms=a.decision_ms, time_bank_ms=a.time_bank_ms,
        starting_stack=a.starting_stack, hand_cap=a.hand_cap, variant=a.variant).items() if v is not None}
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    sys.exit(play(a.folders, a.tables, a.seed, settings))


if __name__ == "__main__":
    main()
