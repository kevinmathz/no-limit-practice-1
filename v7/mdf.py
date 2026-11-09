"""Minimum defense frequency (MDF) floor and bluff cap for v6. Pure stdlib, tables built at import.

MDF = P / (P + B), where B = state.to_call (the bet we face) and P = the pot before that bet.
state.pot already includes every chip committed this hand (bettor's bet too), so
P = state.pot - to_call and MDF = (pot - to_call) / pot.

Defend set: our range's buckets sorted strongest-first by equity (equity169 / equity_post);
take buckets until cumulative frequency reaches MDF * MDF_SCALE; the boundary bucket defends
with probability (target - cum_before) / freq so the whole range continues exactly MDF.
"""
import bisect
import json
import os
import random

MDF_SCALE = 1.0   # <1 defends less
BLUFF_CAP = True  # cap bucket-0/1 bets to the indifference bluff share alpha = bet/(pot+2*bet)
BLUFF_BUCKETS = (0, 1)
VALUE_MIN = 3     # buckets >= this count as value bets for the bluff ratio

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "bucket_freq.json")) as _f:
    _BF = json.load(_f)
with open(os.path.join(_HERE, "equity169.json")) as _f:
    _EQ169 = json.load(_f)
with open(os.path.join(_HERE, "equity_post.json")) as _f:
    _EQPOST = json.load(_f)


def _order(keys, eq, freq):
    """keys strongest-first -> {key: (cum_before, freq)}, normalized to total 1."""
    ks = sorted(keys, key=lambda k: -eq[k])
    tot = sum(freq[k] for k in ks) or 1.0
    out, cum = {}, 0.0
    for k in ks:
        f = freq[k] / tot
        out[k] = (cum, f)
        cum += f
    return out


PRE = _order(list(_EQ169), _EQ169, _BF["preflop"])
POST = [_order(list(range(9)), _EQPOST[s], _BF["post"][s]) for s in range(3)]
BLUFF_FREQ = [sum(_BF["post"][s][b] for b in BLUFF_BUCKETS) for s in range(3)]
VALUE_FREQ = [sum(_BF["post"][s][b] for b in range(VALUE_MIN, 9)) for s in range(3)]


def mdf(pot, to_call):
    if to_call <= 0:
        return 0.0
    return max(0.0, (pot - to_call) / max(pot, 1))


def defend_prob(slot, m):
    """slot = (cum_before, freq) of our bucket; m = MDF target. Prob. our hand must continue."""
    cum, f = slot
    t = m * MDF_SCALE
    if cum + f <= t:
        return 1.0
    if cum >= t or f <= 0:
        return 0.0
    return (t - cum) / f


def slot_for(cls_or_bucket, street):
    """street -1 preflop (key = 169 class), 0..2 flop/turn/river (key = bucket 0..8)."""
    return PRE[cls_or_bucket] if street < 0 else POST[street][cls_or_bucket]


def should_defend(cls_or_bucket, street, pot, to_call):
    p = defend_prob(slot_for(cls_or_bucket, street), mdf(pot, to_call))
    return p >= 1.0 or (p > 0.0 and random.random() < p)


def bluff_allowed(street, pot, bet):
    """Bucket-0/1 bet of size `bet` into `pot`: allow with prob p so that
    p*bluff/(p*bluff+value) = alpha = bet/(pot+2*bet)."""
    if not BLUFF_CAP or bet <= 0:
        return True
    a = bet / (pot + 2 * bet)
    bf, vf = BLUFF_FREQ[street], VALUE_FREQ[street]
    if bf <= 0:
        return True
    p = a * vf / ((1 - a) * bf)
    return p >= 1.0 or random.random() < p
