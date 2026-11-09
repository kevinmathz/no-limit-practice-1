"""Cards and a pure-Python 7-card hand evaluator (local only)."""
from collections import Counter
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)
RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}


def _rank5(cards):
    rs = sorted((RANK_VAL[c[0]] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(rs), reverse=True)
    straight = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight = 5
    groups = sorted(Counter(rs).items(), key=lambda x: (x[1], x[0]), reverse=True)
    pattern = [c for _, c in groups]
    ordered = [r for r, _ in groups]
    if straight and flush:
        return (8, straight)
    if pattern[0] == 4:
        return (7, *ordered)
    if pattern == [3, 2]:
        return (6, *ordered)
    if flush:
        return (5, *rs)
    if straight:
        return (4, straight)
    if pattern[0] == 3:
        return (3, *ordered)
    if pattern == [2, 2, 1]:
        return (2, *ordered)
    if pattern[0] == 2:
        return (1, *ordered)
    return (0, *rs)


def best_hand(cards):
    """Higher tuple = better hand. Works for 5-7 cards."""
    return max(_rank5(c) for c in combinations(cards, 5))
