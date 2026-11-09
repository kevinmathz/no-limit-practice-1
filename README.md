# no-limit-practice-1

A **mock** of the Sequoia No Limit 2026 starter kit, running our practice variant **keep-reveal-discard** (see `RULES.md`), built from the public "Build and submit" page so we can practice before the real kit is released. The interfaces match the published docs. The defaults in `RULES.md` are guesses.

## Quick start (no installs needed, Python 3.10+)

```bash
python poker_agent.py check bot/agent.py
python poker_agent.py play bot examples/check_or_fold examples/min_raiser --tables 200
python poker_agent.py play bot bot examples/random_bot examples/calling_station \
       examples/min_raiser examples/check_or_fold examples/random_bot examples/calling_station --tables 200
python poker_agent.py pack bot out.zip
```

`play` takes one folder per seat (2 to 8 seats) and prints JSON per seat. The fields are `avg_points`, `places`, `failed_decisions` (broken down by kind), and act time percentiles: p50, p99 and max, in ms. To replay the same randomness, pass `--seed N`. You can override the table defaults with `--decision-ms`, `--time-bank-ms`, `--starting-stack`, `--hand-cap` and `--variant reference-holdem`.

## Layout

| Path | What |
| --- | --- |
| `bot/agent.py` | **Our bot.** It is a template with safe action helpers and a memory example. |
| `examples/` | Opponents. `check_or_fold` and `min_raiser` mirror the real kit; `calling_station` and `random_bot` are extras. |
| `sdk/poker_agent_sdk/` | The mock SDK: `Agent`, `Action`, `State`, `LegalAction`, `Event`, `SeatInfo` and `Config`. |
| `sdk/poker_engine/` | The local simulator: sit-and-go tables, side pots, ICM at the hand cap and timing. **Bots must not import it.** |
| `poker_agent.py` | A mock of the `poker-agent` command, with `check`, `play` and `pack`. |

## Moving to the real kit

1. Copy `bot/agent.py` into the kit as `agent.py`.
2. Run the real `poker-agent check agent.py`, then the real `play`.
3. Fix any field-name differences against the real SDK, and read the real `RULES.md` for the variant.

The real kit has `agent.py` at its root. Here the bot lives in `bot/`, so `pack` zips only the bot.

## Collaboration

- Push straight to `main` in small commits, and pull before you start.
- One person owns `bot/agent.py`. The other works on tests, opponents and analysis.
- Before every commit, run `check` and a 200-table `play`. You need **0 failed decisions** and a p99 well under 1 ms.
