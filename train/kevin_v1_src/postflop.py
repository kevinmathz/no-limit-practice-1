"""Postflop decision for the KRD bot (stdlib + numpy; uses rt_eval + oppmodel helpers).

postflop_decide(state, my_cards, board, opp_infos, risk_premium=0.0) -> Action

opp_infos: one dict per live opponent (still in the hand, folded seats excluded):
    {"seat": int,
     "reveal": "As",                     # their public card
     "hidden": ndarray[52]  (optional)   # P(hidden card id); OR
     "post":   ndarray[169] (optional)   # class posterior (oppmodel.reveal_posterior)
     "agg": float (optional, postflop aggression 0..1, default 0.4),
     "fold_to_raise": float (optional, default 0.55)}
If neither "hidden" nor "post" is given, the hidden card is uniform over live cards.

Equity: river = exact over each opponent's hidden-card distribution (opponents treated as
independent); flop/turn = Monte Carlo over hidden cards + runouts with a time-bounded sample
count (default <= 30 samples; fewer when multiway).  The opponents' revealed cards are part of
their hands, so "their reveal pairs the board" etc. is accounted for exactly.
"""
import random
from bisect import bisect_right
from time import perf_counter

import numpy as np

from poker_agent_sdk import Action
from rt_eval import KEY, RANK_SCORE, FLUSH_SCORE, CARD_ID
from oppmodel import class_post_to_cards

_FSUIT = {0x8: 0, 0x80: 1, 0x800: 2, 0x8000: 3}


# ---------- safe action helpers (from the template) ----------
def raise_to(state, total):
    r = state.can("raise")
    if not r:
        return call_or_check(state)
    total = int(total)
    total = max(r.min_amount, min(r.max_amount, total))
    return Action.raise_to(total)


def call_or_check(state):
    if state.can("check"):
        return Action.check()
    if state.can("call"):
        return Action.call()
    return Action.fold()


def check_or_fold(state):
    if state.can("check"):
        return Action.check()
    if state.can("fold"):
        return Action.fold()
    first = state.legal_actions[0]
    return Action(first.kind, first.min_amount)


# ---------- evaluation helpers ----------
def _score(k, ids):
    if (k + 0x3333) & 0x8888:
        s = _FSUIT[(k + 0x3333) & 0x8888]
        m = 0
        for c in ids:
            if c & 3 == s:
                m |= 1 << (c >> 2)
        return FLUSH_SCORE[m]
    return RANK_SCORE[k >> 16]


def _ids(cards):
    return [CARD_ID[c] if isinstance(c, str) else int(c) for c in cards]


def _hidden_weights(info, dead):
    h = info.get("hidden")
    if h is None:
        p = info.get("post")
        if p is not None:
            h = class_post_to_cards(p, info["reveal"], ())
        else:
            h = np.ones(52)
    w = np.array(h, dtype=np.float64).reshape(52)
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    w[dead] = 0.0
    s = w.sum()
    if s <= 0:
        w = np.ones(52)
        w[dead] = 0.0
        s = w.sum()
    return w / s


def equity(my_ids, board_ids, opp_reveals, opp_weights, n_samples=30, rng=random, budget_ms=None):
    """My share of the pot at showdown vs all opponents. opp_weights: list of float[52].
    budget_ms: wall-clock cap for the Monte Carlo loop (always >= 8 samples)."""
    nb = len(board_ids)
    nop = len(opp_reveals)
    if nop == 0:
        return 1.0
    dead = set(my_ids) | set(board_ids) | set(opp_reveals)
    kb = 0
    for c in board_ids:
        kb += KEY[c]
    kme = kb + KEY[my_ids[0]] + KEY[my_ids[1]]
    if nb == 5 and nop <= 2:  # exact per opponent, independence across opponents
        my_ids7 = list(my_ids) + list(board_ids)
        s0 = _score(kme, my_ids7)
        eq = 1.0
        for rv, w in zip(opp_reveals, opp_weights):
            kr = kb + KEY[rv]
            win = tie = 0.0
            tot = 0.0
            wl = w.tolist()
            for h in np.flatnonzero(w).tolist():
                if h in dead:
                    continue
                p = wl[h]
                tot += p
                so = _score(kr + KEY[h], board_ids + [rv, h])
                if s0 > so:
                    win += p
                elif s0 == so:
                    tie += p
            if tot > 0:
                eq *= (win + 0.5 * tie) / tot
        return eq
    cums = []
    for w in opp_weights:
        cs = np.cumsum(w)
        cums.append((cs / cs[-1]).tolist())
    need = 5 - nb
    rnd = rng.random
    acc = 0.0
    n_ok = 0
    t_end = perf_counter() + budget_ms / 1000.0 if budget_ms else None
    for it in range(n_samples):
        if t_end is not None and it >= 8 and perf_counter() > t_end:
            break
        used = set(dead)
        hids = []
        ok = True
        for cs in cums:
            for _try in range(6):
                h = bisect_right(cs, rnd())
                if h > 51:
                    h = 51
                if h not in used:
                    break
            else:
                ok = False
                break
            used.add(h)
            hids.append(h)
        if not ok:
            continue
        run = []
        while len(run) < need:
            c = int(rnd() * 52)
            if c not in used:
                used.add(c)
                run.append(c)
        kr = 0
        for c in run:
            kr += KEY[c]
        full_board = board_ids + run
        s0 = _score(kme + kr, my_ids + full_board)
        best = -1
        nt = 0
        for rv, h in zip(opp_reveals, hids):
            so = _score(kb + kr + KEY[rv] + KEY[h], full_board + [rv, h])
            if so > best:
                best, nt = so, 1
            elif so == best:
                nt += 1
        n_ok += 1
        if s0 > best:
            acc += 1.0
        elif s0 == best:
            acc += 1.0 / (nt + 1)
    if n_ok == 0:
        return 1.0 / (nop + 1)
    return acc / n_ok


CALL_SHADE = 0.30
STACKOFF_MARGIN = 0.06


def _n_samples(nop):
    return 30 if nop <= 1 else (24 if nop == 2 else max(10, 60 // (nop + 1)))


def postflop_decide(state, my_cards, board, opp_infos, risk_premium=0.0, rng=random, budget_ms=0.25):
    """budget_ms caps the equity Monte Carlo loop (wall clock)."""
    try:
        return _decide(state, my_cards, board, opp_infos, risk_premium, rng, budget_ms)
    except Exception:
        return check_or_fold(state)


def _decide(state, my_cards, board, opp_infos, risk_premium, rng, budget_ms):
    t0 = perf_counter()
    my_ids = _ids(my_cards)
    board_ids = _ids(board)
    revs = [CARD_ID[o["reveal"]] if isinstance(o["reveal"], str) else int(o["reveal"])
            for o in opp_infos]
    dead = my_ids + board_ids + revs
    ws = [_hidden_weights(o, dead) for o in opp_infos]
    nop = len(opp_infos)
    left = None if budget_ms is None else max(0.02, budget_ms - (perf_counter() - t0) * 1000.0)
    eq = equity(my_ids, board_ids, revs, ws, _n_samples(nop), rng, left)

    me = None
    for s in state.seats:
        if s.seat == state.seat:
            me = s
            break
    stack = me.stack if me is not None else 0
    mybet = me.bet if me is not None else 0
    pot = max(1, state.pot)
    to_call = state.to_call
    river = len(board_ids) >= 5
    rp = max(0.0, float(risk_premium))
    allin_total = stack + mybet

    if to_call > 0:
        # who is betting into us: the most passive bettor's aggression shades our equity
        agg = 0.4
        bettors = [o for o in opp_infos
                   if any(s.seat == o.get("seat") and s.bet > mybet for s in state.seats)]
        if bettors:
            agg = min(float(o.get("agg", 0.4)) for o in bettors)
        size = min(1.5, to_call / max(1, pot - to_call))
        # a bettor's range is stronger than its preflop range: shade harder vs passive players
        f = 1.0 - (1.0 - agg) * CALL_SHADE * (0.5 + size)
        eq_adj = max(0.0, min(1.0, eq * max(0.45, min(1.05, f))))
        need = to_call / (pot + to_call) + rp
        if to_call >= 0.3 * allin_total:
            need += STACKOFF_MARGIN
        spr = (stack - to_call) / (pot + to_call)
        raise_thr = 0.72 + 0.06 * (nop - 1) + rp
        if state.can("raise") and eq_adj >= raise_thr:
            if spr <= 1.2:
                return raise_to(state, allin_total)
            total = mybet + to_call + int(0.8 * (pot + to_call))
            if total >= 0.45 * allin_total:
                total = allin_total
            return raise_to(state, total)
        if eq_adj >= need:
            if state.can("raise") and spr <= 0.6 and eq_adj >= 0.5 + rp:
                return raise_to(state, allin_total)
            return call_or_check(state)
        return check_or_fold(state)

    # checked to us (or we open the action)
    fair = 1.0 / (nop + 1)
    value_thr = max(0.42, 0.62 - 0.1 * (nop - 1)) + 0.5 * rp
    spr = stack / pot
    if eq >= value_thr:
        if spr <= 1.5 and eq >= max(0.55, value_thr):
            return raise_to(state, allin_total)
        frac = 0.75 if (river or eq >= 0.8) else 0.62
        total = mybet + int(frac * pot)
        if total >= 0.5 * allin_total:
            total = allin_total
        return raise_to(state, total)
    if nop == 1 and rp < 0.08:
        ftr = float(opp_infos[0].get("fold_to_raise", 0.55))
        if river:
            # bluff with the bottom of range at roughly b/(1+2b) of the value-bet weight
            if eq < 0.15 and rng.random() < min(0.35, 0.22 * ftr / 0.55):
                return raise_to(state, mybet + int(0.75 * pot))
        elif eq < fair * 0.7:
            if rng.random() < min(0.4, 0.22 * ftr / 0.55):
                return raise_to(state, mybet + int(0.62 * pot))
        elif eq < value_thr and rng.random() < 0.3:  # semi-bluff / protection
            return raise_to(state, mybet + int(0.62 * pot))
    return check_or_fold(state)
