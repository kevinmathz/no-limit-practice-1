"""Train the v1 policy with External-Sampling MCCFR (Lanctot et al. 2009), the same algorithm as
OpenSpiel's external_sampling_mccfr and slumbot2019's MCCFR, with regret-matching+ and linear
averaging. Ported to a compact pure-Python heads-up NLHE abstraction for speed.

Game abstraction (see v1/abstraction.py):
  - heads-up, equal stacks of DEPTHS bb; one tree per depth
  - actions: fold, check/call, raise 1/2 pot, raise pot, all-in (max 3 raises/street)
  - cards: 8 preflop equity buckets, 9 postflop made-hand buckets, imperfect recall
  - utility: chips

Usage:  python train/mccfr.py --minutes 10      (writes v1/policy.json)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOT = os.environ.get("CFR_BOT", "v1")  # bot folder whose abstraction.py to train, and where policy is written
sys.path.insert(0, str(ROOT / BOT))
sys.path.insert(0, str(ROOT / "sdk"))

from abstraction import (ACTIONS, DEPTHS, MAX_RAISES, N_PRE, RAISE_FRAC, hand_class,  # noqa: E402
                         info_key, postflop_bucket, spr_bucket)

CACHE = ROOT / "train" / "cache"
SB, BB = 1, 2


# ---------------- preflop equity table + deal pool ----------------
def _all_classes():
    from poker_engine.cards import DECK
    classes = {}
    for i in range(52):
        for j in range(i + 1, 52):
            classes.setdefault(hand_class(DECK[i], DECK[j]), (DECK[i], DECK[j]))
    return classes


def _equity_job(args):
    name, cards, trials, seed = args
    from poker_engine.cards import DECK, best_hand
    rng = random.Random(seed)
    rest = [c for c in DECK if c not in cards]
    score = 0.0
    for _ in range(trials):
        s = rng.sample(rest, 7)
        a = best_hand(list(cards) + s[2:])
        b = best_hand(s[:7])
        score += 1.0 if a > b else 0.5 if a == b else 0.0
    return name, score / trials


def build_pre_table(pool, trials=3000):
    classes = _all_classes()
    res = dict(pool.map(_equity_job, [(n, c, trials, k) for k, (n, c) in enumerate(classes.items())]))
    combos = {n: (6 if len(n) == 2 else 4 if n.endswith("s") else 12) for n in res}
    order = sorted(res, key=lambda n: res[n])  # weakest first
    # bucket masses, weakest -> strongest; top bucket = premium ~4%
    masses = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    assert len(masses) == N_PRE
    cuts, acc = [], 0.0
    for m in masses:
        acc += m
        cuts.append(acc * 1326)
    table, run = {}, 0
    for n in order:
        run += combos[n] / 2  # midpoint
        table[n] = next(b for b, c in enumerate(cuts) if run <= c + 1e-9)
        run += combos[n] / 2
    return table, res


def _deal_job(args):
    n, seed, pre = args
    from poker_engine.cards import DECK, best_hand
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d = rng.sample(DECK, 9)
        h0, h1, board = d[0:2], d[2:4], d[4:9]
        row = [pre[hand_class(*h0)], pre[hand_class(*h1)]]
        for k in (3, 4, 5):
            row += [postflop_bucket(h0, board[:k]), postflop_bucket(h1, board[:k])]
        a, b = best_hand(h0 + board), best_hand(h1 + board)
        row.append(0 if a > b else 1 if b > a else 2)
        out.append(tuple(row))
    return out


def build_deals(pool, pre, n=400_000):
    jobs = 64
    parts = pool.map(_deal_job, [(n // jobs, 1000 + k, pre) for k in range(jobs)])
    return [d for p in parts for d in p]


# ---------------- game ----------------
# state = (street, hist, nraises, actor, c0, c1, s0, s1, b0, b1, spr_b)
def legal(st):
    street, hist, nr, me, c0, c1, s0, s1, b0, b1, sb_ = st
    stacks = (s0, s1)
    bets = (b0, b1)
    opp = 1 - me
    to_call = bets[opp] - bets[me]
    acts = []
    if to_call > 0:
        acts.append("f")
    acts.append("c")
    if stacks[me] > to_call and stacks[opp] > 0:
        if nr < MAX_RAISES:
            pot2 = c0 + c1 + to_call
            for x in RAISE_FRAC:
                add = max(int(RAISE_FRAC[x] * pot2), BB)
                if to_call + add < stacks[me]:
                    acts.append(x)
        acts.append("a")
    return acts


def raise_add(st, x):
    street, hist, nr, me, c0, c1, s0, s1, b0, b1, sb_ = st
    bets = (b0, b1)
    to_call = bets[1 - me] - bets[me]
    return to_call, max(int(RAISE_FRAC[x] * (c0 + c1 + to_call)), BB)


def apply(st, a):
    """Returns ('T', winner_or_None_for_showdown, c0, c1) for terminals, else a new state."""
    street, hist, nr, me, c0, c1, s0, s1, b0, b1, spr_b = st
    c = [c0, c1]
    s = [s0, s1]
    b = [b0, b1]
    opp = 1 - me
    to_call = b[opp] - b[me]
    if a == "f":
        return ("T", opp, c0, c1)
    if a == "c":
        amt = min(to_call, s[me])
        c[me] += amt
        s[me] -= amt
        b[me] += amt
        closes = len(hist) > 0
        if not closes:
            return (street, hist + "c", nr, opp, c[0], c[1], s[0], s[1], b[0], b[1], spr_b)
        if street == 3 or s[0] == 0 or s[1] == 0:
            return ("T", None, c[0], c[1])
        return (street + 1, "", 0, 1, c[0], c[1], s[0], s[1], 0, 0,
                spr_bucket(min(s[0], s[1]), c[0] + c[1]))
    if a == "a":
        need = s[me]
    else:
        tc, add = raise_add(st, a)
        need = tc + add
    c[me] += need
    s[me] -= need
    b[me] += need
    return (street, hist + a, nr + 1, opp, c[0], c[1], s[0], s[1], b[0], b[1], spr_b)


def utility(term, deal, trav):
    _, winner, c0, c1 = term
    c = (c0, c1)
    if winner is None:
        w = deal[8]
        if w == 2:
            return 0.0
        winner = w
    return float(c[1 - trav]) if winner == trav else -float(c[trav])


BUCKET_IDX = ((0, 1), (2, 3), (4, 5), (6, 7))


class Trainer:
    def __init__(self, depth_i, deals, seed):
        self.depth_i = depth_i
        self.deals = deals
        self.rng = random.Random(seed)
        self.R = {}
        self.S = {}
        self.t = 0
        stack = DEPTHS[depth_i] * BB
        self.root = (0, "", 0, 0, SB, BB, stack - SB, stack - BB, SB, BB, 0)

    def walk(self, st, deal, trav):
        if st[0] == "T":
            return utility(st, deal, trav)
        street, hist, _, me = st[0], st[1], st[2], st[3]
        acts = legal(st)
        bucket = deal[BUCKET_IDX[street][me]]
        key = info_key(self.depth_i, street, bucket, st[10], hist) + "|" + "".join(acts)
        reg = self.R.get(key)
        if reg is None:
            reg = self.R[key] = [0.0] * len(acts)
        pos = [r if r > 0 else 0.0 for r in reg]
        tot = sum(pos)
        n = len(acts)
        strat = [p / tot for p in pos] if tot > 0 else [1.0 / n] * n
        if me == trav:
            utils = [self.walk(apply(st, a), deal, trav) for a in acts]
            node = sum(p * u for p, u in zip(strat, utils))
            for i in range(n):
                v = reg[i] + utils[i] - node
                reg[i] = v if v > 0 else 0.0  # regret-matching+
            return node
        ss = self.S.get(key)
        if ss is None:
            ss = self.S[key] = [0.0] * n
        w = self.t
        for i in range(n):
            ss[i] += w * strat[i]  # linear averaging
        x = self.rng.random()
        acc = 0.0
        for i in range(n):
            acc += strat[i]
            if x < acc:
                break
        return self.walk(apply(st, acts[i]), deal, trav)

    def run(self, seconds):
        end = time.time() + seconds
        deals = self.deals
        while time.time() < end:
            for _ in range(200):
                self.t += 1
                deal = deals[self.rng.randrange(len(deals))]
                self.walk(self.root, deal, 0)
                self.walk(self.root, deal, 1)
        return self


def _train_job(args):
    depth_i, seconds, seed = args
    with open(CACHE / "deals.pkl", "rb") as f:
        deals = pickle.load(f)
    tr = Trainer(depth_i, deals, seed).run(seconds)
    return tr.t, tr.S


def merge_policy(results):
    """Average strategy per infoset (summing linear-weighted sums across workers)."""
    total = {}
    for _, S in results:
        for k, v in S.items():
            t = total.get(k)
            if t is None:
                total[k] = list(v)
            else:
                for i in range(len(v)):
                    t[i] += v[i]
    # keys are "<infoset>|<legal actions>"; keep the most-visited action set per infoset
    best = {}
    for k, v in total.items():
        base, acts = k.rsplit("|", 1)
        w = sum(v)
        if w > 0 and (base not in best or w > best[base][0]):
            best[base] = (w, acts, v)
    return {base: {a: round(x / w, 3) for a, x in zip(acts, v) if x / w >= 0.001}
            for base, (w, acts, v) in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--workers-per-depth", type=int, default=max(1, (os.cpu_count() or 4) // len(DEPTHS)))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with mp.Pool(os.cpu_count()) as pool:
        if a.rebuild or not (CACHE / "deals.pkl").exists():
            t0 = time.time()
            pre, eq = build_pre_table(pool)
            (ROOT / BOT / "preflop_buckets.json").write_text(json.dumps(pre, sort_keys=True))
            deals = build_deals(pool, pre)
            with open(CACHE / "deals.pkl", "wb") as f:
                pickle.dump(deals, f)
            print(f"built preflop table + {len(deals)} deals in {time.time() - t0:.0f}s", flush=True)
        jobs = [(d, a.minutes * 60, 7 + 100 * d + w)
                for d in range(len(DEPTHS)) for w in range(a.workers_per_depth)]
        print(f"training {len(jobs)} workers for {a.minutes} min ...", flush=True)
        results = pool.map(_train_job, jobs)
    iters = sum(t for t, _ in results)
    policy = merge_policy(results)
    (ROOT / BOT / "policy.json").write_text(json.dumps(policy, separators=(",", ":"), sort_keys=True))
    print(f"done: {iters:,} iterations, {len(policy):,} infosets -> {BOT}/policy.json", flush=True)


if __name__ == "__main__":
    main()
