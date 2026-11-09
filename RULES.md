# Rules (mock: reference no-limit hold'em)

The real variant will be announced at the event. The defaults below are **our guesses**. Bots should always read `state.config` and never hard-code these values.

## Format

- Sit-and-go tables of 2 to 10 seats. A round uses 8 seats, and the docs' example uses 3.
- Everyone starts with 1,500 chips. Blinds rise every 10 hands: 10/20, 15/30, 25/50, 50/100 and so on (see `sdk/poker_engine/table.py`).
- The hand cap is 150. If the table reaches it, the remaining players are scored by ICM on their stacks.
- Payouts are 50/30/20 points for 1st, 2nd and 3rd. A round may draw different payouts per table, so read `state.config.payouts`.
- Scoring is average points across tables.

## Betting

- Actions are `fold`, `check`, `call` and `raise_to(total)`, where `total` is your whole bet for the current betting round.
- Check and call are never offered together. Fold is not offered when checking is free.
- The minimum raise is the last full raise increment, or the big blind if there hasn't been one. A raise can go up to all-in.
- Heads-up, the button posts the small blind and acts first preflop.
- Every player who reaches showdown shows their hand; there is no mucking.

## Limits (from the real docs)

- `decision_ms` is 1. `time_bank_ms` is 100 per table. The upload test match has **no time bank**.
- `memory_bytes` is 64 KB. Memory starts as `b""` at each table.
- A failed decision (a timeout, an exception or an illegal action) plays a fallback: check if free, otherwise fold.
