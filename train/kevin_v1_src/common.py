"""Shared conventions for the keep-reveal-discard (KRD) bot pipeline.

Everything offline (equity, CFR, eval) and the runtime bot MUST agree on these.

Cards
-----
- String form "As": rank char from RANKS, suit char from SUITS (same as the practice engine).
- Int form: card_id = rank_idx * 4 + suit_idx, rank_idx 0..12 (2..A), suit_idx 0..3 (c,d,h,s).
  DECK[card_id] == string form (identical to sdk/poker_engine/cards.DECK order).

169 two-card classes
--------------------
CLASS_LIST is the no_limit order (AA, AKs, AKo, AQs, ..., 22) so it lines up with
no_limit/game/tables/hu_matchups.EQUITY[i][j] (equity of class i vs class j).

3-card choose classes
---------------------
A 3-card hand is canonicalised up to suit isomorphism: over all 24 suit permutations take the
lexicographically smallest sorted tuple of card ids (sorted descending by id). The canonical
tuple's card ORDER defines positions 0,1,2 (descending id => descending rank).
CHOOSE_ACTIONS lists (keep_pos, reveal_pos) pairs; the third position is discarded.
`canon3(cards)` returns (class_id, order) where order[p] = index into the ORIGINAL cards of the
card sitting at canonical position p.
"""
from __future__ import annotations

from itertools import combinations, permutations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)
CARD_ID = {c: i for i, c in enumerate(DECK)}


def rank_of(cid: int) -> int:
    return cid >> 2


def suit_of(cid: int) -> int:
    return cid & 3


# ---------------------------------------------------------------- 169 classes
CLASS_LIST: list[str] = []
for _i in range(12, -1, -1):
    for _j in range(_i, -1, -1):
        if _i == _j:
            CLASS_LIST.append(RANKS[_i] * 2)
        else:
            CLASS_LIST.append(RANKS[_i] + RANKS[_j] + "s")
            CLASS_LIST.append(RANKS[_i] + RANKS[_j] + "o")
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_LIST)}


def class2(c1: int, c2: int) -> int:
    """169-class index of two card ids."""
    r1, r2 = rank_of(c1), rank_of(c2)
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return CLASS_INDEX[RANKS[hi] * 2]
    return CLASS_INDEX[RANKS[hi] + RANKS[lo] + ("s" if suit_of(c1) == suit_of(c2) else "o")]


def class2_str(a: str, b: str) -> int:
    return class2(CARD_ID[a], CARD_ID[b])


# ---------------------------------------------------------------- 3-card classes
CHOOSE_ACTIONS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))  # (keep_pos, reveal_pos)
_SUIT_PERMS = list(permutations(range(4)))


def _canon_tuple(cids):
    best = None
    for perm in _SUIT_PERMS:
        t = tuple(sorted(((c & ~3) | perm[c & 3] for c in cids), reverse=True))
        if best is None or t < best:
            best = t
    return best


def _build():
    classes: list[tuple] = []
    index: dict[tuple, int] = {}
    combo_map: dict[tuple, tuple] = {}  # sorted-desc original ids -> (class_id, order)
    for combo in combinations(range(52), 3):
        cids = tuple(sorted(combo, reverse=True))
        best = None
        best_perm = None
        for perm in _SUIT_PERMS:
            mapped = [(c & ~3) | perm[c & 3] for c in cids]
            order = sorted(range(3), key=lambda k: -mapped[k])
            t = tuple(mapped[k] for k in order)
            if best is None or t < best:
                best, best_perm = t, order
        if best not in index:
            index[best] = len(classes)
            classes.append(best)
        combo_map[cids] = (index[best], tuple(best_perm))
    return classes, index, combo_map


CHOOSE_CLASSES, _CHOOSE_INDEX, _COMBO_MAP = _build()
N_CHOOSE = len(CHOOSE_CLASSES)


def canon3(cards) -> tuple[int, tuple]:
    """cards: 3 card ids or strings (any order). Returns (class_id, order) where order[p] is
    the index into `cards` of the card at canonical position p."""
    ids = [CARD_ID[c] if isinstance(c, str) else c for c in cards]
    srt = sorted(range(3), key=lambda k: -ids[k])
    cid, order = _COMBO_MAP[tuple(ids[k] for k in srt)]
    return cid, tuple(srt[p] for p in order)


def choose_to_cards(cards, action_idx: int):
    """Map a CHOOSE_ACTIONS index to (keep_card, reveal_card, discard_card) of `cards`."""
    _, order = canon3(cards)
    kp, rp = CHOOSE_ACTIONS[action_idx]
    dp = 3 - kp - rp
    return cards[order[kp]], cards[order[rp]], cards[order[dp]]


if __name__ == "__main__":
    print("choose classes:", N_CHOOSE)
    print(canon3(["As", "Ah", "Kd"]), canon3(["Kc", "Ad", "As"]))
    print(choose_to_cards(["As", "Ah", "Kd"], 0))
