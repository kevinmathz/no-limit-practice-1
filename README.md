# no-limit-practice-1

A **mock** of the Sequoia No Limit 2026 starter kit, running our practice variant **keep-reveal-discard** (see `RULES.md`), built from the public "Build and submit" page so we can practice before the real kit is released. The interfaces match the published docs. The defaults in `RULES.md` are guesses.

## Quick start (no installs needed, Python 3.10+)

```bash
python poker_agent.py check v0/agent.py
python poker_agent.py play v0 examples/check_or_fold examples/min_raiser --tables 200
python poker_agent.py play v0 v0 examples/random_bot examples/calling_station \
       examples/min_raiser examples/check_or_fold examples/random_bot examples/calling_station --tables 200
python poker_agent.py pack v0 out.zip
```

`play` takes one folder per seat (2 to 8 seats) and prints JSON per seat. The fields are `avg_points`, `places`, `failed_decisions` (broken down by kind), and act time percentiles: p50, p99 and max, in ms. To replay the same randomness, pass `--seed N`. You can override the table defaults with `--decision-ms`, `--time-bank-ms`, `--starting-stack`, `--hand-cap` and `--variant reference-holdem`.

## Layout

| Path | What |
| --- | --- |
| `v0/agent.py` | **Our bot.** It is a template with safe action helpers and a memory example. |
| `examples/` | Opponents. `check_or_fold` and `min_raiser` mirror the real kit; `calling_station` and `random_bot` are extras. |
| `sdk/poker_agent_sdk/` | The mock SDK: `Agent`, `Action`, `State`, `LegalAction`, `Event`, `SeatInfo` and `Config`. |
| `sdk/poker_engine/` | The local simulator: sit-and-go tables, side pots, ICM at the hand cap and timing. **Bots must not import it.** |
| `poker_agent.py` | A mock of the `poker-agent` command, with `check`, `play` and `pack`. |

## Bots

| Folder | What |
| --- | --- |
| `v0/` | Rule-based bot (previously `bot/`). It keeps its best two cards and reveals the weaker one, uses a hand score before the flop, and judges made hands after the flop. |
| `v1/` | MCCFR bot. It reveals and discards at random, then bets from a strategy trained by `train/mccfr.py` (external-sampling MCCFR on heads-up hold'em with grouped hands and fixed bet sizes). Retrain with `python train/mccfr.py --minutes 10`. |
| `v2/` | v1 plus a river guaranteed-loss fold (`showdown.py`) and an ICM layer (`icm.py`). The ICM layer vetoes stack-risking commits unless they gain expected points, and plays aggressively in the last 25 hands when not the chip leader. |
| `v3/` | v0 plus the same river fold and ICM layer. |
| `v4/` | v2 with every policy action under 8% removed and the rest renormalized. |
| `v5/` | v1 with raise sizes by pot fraction (1/3, 2/3, 1, 1.5, 2.5 pot, all-in), each with its own frequency. Undertrained: 4.7M iterations. Retrain with `CFR_BOT=v5 python train/mccfr.py --minutes N`. |
| `v6/` | v3 plus a minimum-defense-frequency floor (`mdf.py`: defend the strongest MDF share of the range against a bet) and a bluff cap. |
| `v7/` | v3 with exact MDF: facing a bet, continue with exactly the top MDF share of its own (self-calibrated) range, never more or less. |
| `v8/`-`v11/` | v0, v3, v4, v7 plus `tight.py`: preflop top 10% only (top 20% when short), postflop bet with top pair good kicker+, call with top pair+, stack-off guard. Off during late-game aggression. |

Results of all bots plus check_or_fold over 2,000 8-seat tables (seed 41). "Out of money" means finishing 4th-8th. "Overfold" is the mean shortfall of the continue rate below MDF; `play` now reports it for every seat.

| Bot | Avg points | 1st | Out of money | Overfold |
| --- | --- | --- | --- | --- |
| check_or_fold | 26.7 | 0% | 0% | 0.52 |
| v3 | 15.0 | 23% | 63% | 0.21 |
| v0 | 13.6 | 24% | 69% | 0.18 |
| v4 | 11.3 | 10% | 63% | 0.22 |
| v2 | 10.8 | 8% | 63% | 0.23 |
| v6 | 10.6 | 18% | 75% | 0.12 |
| v5 | 6.3 | 10% | 83% | 0.20 |
| v1 | 5.6 | 7% | 84% | 0.17 |

Round-robin of all bots except v1 (`python train/roundrobin.py v0 v2 ... --tables 3000`; check_or_fold at every table, 7 random others): check_or_fold 20.6, v10 17.1, v9 13.8, v8 13.0, v11 12.9, v7 12.7, v0 11.5, v3 11.4, v4 9.2, v6 8.9, v2 8.1, v5 6.2 average points.

## Moving to the real kit

1. Copy `v0/agent.py` into the kit as `agent.py`.
2. Run the real `poker-agent check agent.py`, then the real `play`.
3. Fix any field-name differences against the real SDK, and read the real `RULES.md` for the variant.

The real kit has `agent.py` at its root. Here the bot lives in `v0/`, so `pack` zips only the bot.

## Collaboration

- Push straight to `main` in small commits, and pull before you start.
- One person owns `v0/agent.py`. The other works on tests, opponents and analysis.
- Before every commit, run `check` and a 200-table `play`. You need **0 failed decisions** and a p99 well under 1 ms.
