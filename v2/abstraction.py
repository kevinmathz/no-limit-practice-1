"""Card + betting abstraction shared by the MCCFR trainer (train/) and the v1 bot at runtime.

Must stay pure-stdlib and fast (< ~50 us per call): the bot has a 1 ms decision limit.
"""
RANK = {r: i + 2 for i, r in enumerate("23456789TJQKA")}

# ---------- betting abstraction ----------
ACTIONS = "fchpa"  # fold, check/call, raise half-pot, raise pot, all-in
RAISE_FRAC = {"h": 0.5, "p": 1.0}
MAX_RAISES = 3  # per street; after that only f/c/a
DEPTHS = (10, 20, 40, 80)  # training effective stacks, in big blinds

N_PRE = 8   # preflop buckets
N_POST = 9  # postflop buckets


def depth_index(eff_bb):
    return min(range(len(DEPTHS)), key=lambda i: abs(DEPTHS[i] - eff_bb) / DEPTHS[i])


def spr_bucket(eff_stack, pot):
    r = eff_stack / max(pot, 1)
    if r < 1:
        return 0
    if r < 3:
        return 1
    if r < 8:
        return 2
    return 3


def info_key(depth_i, street, bucket, spr_b, hist):
    return f"{depth_i}|{street}|{bucket}|{spr_b}|{hist}"


# ---------- card abstraction ----------
def hand_class(c1, c2):
    """169-class name, e.g. 'AKs', 'T9o', '77'."""
    r1, r2 = RANK[c1[0]], RANK[c2[0]]
    if r1 < r2:
        c1, c2, r1, r2 = c2, c1, r2, r1
    if r1 == r2:
        return c1[0] + c2[0]
    return c1[0] + c2[0] + ("s" if c1[1] == c2[1] else "o")


def preflop_bucket(cards, pre_table):
    """pre_table: hand_class -> bucket 0..N_PRE-1 (built from equity vs random)."""
    return pre_table[hand_class(cards[0], cards[1])]


def _has_straight(rankset):
    rs = set(rankset)
    if 14 in rs:
        rs.add(1)
    run = 0
    for r in range(1, 15):
        run = run + 1 if r in rs else 0
        if run >= 5:
            return True
    return False


def _straight_draw(rankset):
    rs = set(rankset)
    if 14 in rs:
        rs.add(1)
    for lo in range(1, 11):  # any 4 of 5 consecutive ranks
        if sum((lo + k) in rs for k in range(5)) >= 4:
            return True
    return False


def postflop_bucket(hole, board):
    """0 nothing, 1 draw, 2 weak pair, 3 top pair weak kicker, 4 top pair good kicker / overpair,
    5 two pair, 6 trips/set, 7 straight/flush, 8 full house+.  Only counts made hands that use a hole card."""
    hr = [RANK[c[0]] for c in hole]
    br = [RANK[c[0]] for c in board]
    allr = hr + br
    cnt = {}
    for r in allr:
        cnt[r] = cnt.get(r, 0) + 1
    bcnt = {}
    for r in br:
        bcnt[r] = bcnt.get(r, 0) + 1

    # flush (needs a hole card of the suit)
    suits = {}
    for c in list(hole) + list(board):
        suits[c[1]] = suits.get(c[1], 0) + 1
    flush = any(n >= 5 and any(h[1] == s for h in hole) for s, n in suits.items())
    flush_draw = len(board) < 5 and any(n == 4 and any(h[1] == s for h in hole) for s, n in suits.items())

    # full house / quads using a hole card
    hole_trips = [r for r in set(hr) if cnt[r] >= 3]
    hole_pairs = [r for r in set(hr) if cnt[r] >= 2]
    if any(cnt[r] >= 4 for r in set(hr)):
        return 8
    if hole_trips and any(cnt[r] >= 2 for r in cnt if r not in hole_trips):
        return 8
    if hole_pairs and any(bcnt.get(r, 0) >= 3 for r in bcnt):
        return 8
    straight = _has_straight(allr) and not _has_straight(br)
    if flush or straight:
        return 7
    if hole_trips:
        return 6
    if len(hole_pairs) >= 2:
        return 5
    if hole_pairs:
        top = max(br)
        p = max(hole_pairs)
        if hr[0] == hr[1]:  # pocket pair
            if p > top:
                return 4
            return 2
        if p == top:
            kicker = hr[0] if hr[1] == p else hr[1]
            return 4 if kicker >= 10 else 3
        return 2
    if flush_draw or (len(board) < 5 and _straight_draw(allr) and not _straight_draw(br)):
        return 1
    return 0
