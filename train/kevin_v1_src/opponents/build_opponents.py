#!/usr/bin/env python3
"""Generate opponents/<name>/agent.py from _base_agent.py + a STYLE dict (standalone files).

    python opponents/build_opponents.py

template_bot/agent.py is a verbatim copy of the repo's bot/agent.py and is NOT generated here.
Chen scale reference: AA 20, KK 16, QQ 14, JJ 12, AKs 12, AQs 11, TT/AKo/KQs 10, 99/AQo 9, 22 5.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

TAG = dict(
    reveal="low", naive_reveals=True, reader=False, maniac=False,
    pf_bb=15, pf_jam=dict(early=9, mid=8, late=6, sb=5, bb=8), pf_short_bonus=1.5, pf_call=9,
    open=dict(early=9, mid=8, late=6, sb=7, bb=10), open_size=2.5,
    threebet=10, flat=8, flat_frac=0.08, fourbet=12,
    value_cat=2, aggr=1.0, bubble_players=0, bubble_tighten=0,
)

STYLES = {
    "tag_krd": dict(TAG),
    "reveal_high": dict(TAG, reveal="high"),
    "reveal_random": dict(TAG, reveal="random"),
    "jammer": dict(
        TAG, pf_bb=20, pf_jam=dict(early=5, mid=4, late=3, sb=2, bb=5), pf_short_bonus=2,
        pf_call=7, call_any_pair_ace=True,
        open=dict(early=8, mid=7, late=5, sb=6, bb=9), open_size=3,
        threebet=10, flat=7, flat_frac=0.1, fourbet=11, aggr=1.0),
    "icm_nit": dict(
        TAG, pf_bb=10, pf_jam=dict(early=11, mid=10, late=10, sb=10, bb=11), pf_short_bonus=1,
        pf_call=12, open=dict(early=11, mid=11, late=10, sb=11, bb=13),
        threebet=13, flat=11, flat_frac=0.06, fourbet=14,
        value_cat=3, aggr=0.3, bubble_players=4, bubble_tighten=2),
    "reveal_reader": dict(TAG, naive_reveals=False, reader=True),
    "maniac": dict(TAG, reveal="random", maniac=True, aggr=1.6),
}


def main():
    base = (HERE / "_base_agent.py").read_text()
    marker = 'STYLE = {"name": "__NAME__"}  # __STYLE__'
    assert marker in base
    for name, style in STYLES.items():
        st = dict(style, name=name)
        src = base.replace(marker, "STYLE = " + repr(st))
        d = HERE / name
        d.mkdir(exist_ok=True)
        (d / "agent.py").write_text(src)
        print("wrote", d / "agent.py")


if __name__ == "__main__":
    main()
