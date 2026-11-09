# Rules (practice variant: keep-reveal-discard)

## Keep-reveal-discard (default variant; `--variant reference-holdem` for plain hold'em)

1. Blinds are posted, then each player is dealt **3 cards**.
2. **Choose phase** (`state.phase == "choose"`, only legal action `choose`): every player picks one card to **keep** (private), one to **reveal** (public), one to **discard** (hidden, out of play). Choices are simultaneous: nobody sees others' reveals until all have chosen.
   - Return `Action.choose(keep, reveal)`; each is a card string (`"As"`) or an index 0-2 into `state.cards`.
   - `Action("choose")` (or a failed decision) takes the automatic choice: keep card 0, reveal card 1, discard card 2.
3. Reveals arrive as `reveal` events (`seat`, `cards=(card,)`), and in `state.variant`: `{"revealed": {seat: card}, "my_reveal": card, "my_discard": card}`. `seats[*].shown_cards` holds the revealed card.
4. Play continues as normal no-limit hold'em with your 2 cards (kept + revealed): preflop betting, flop, turn, river, showdown.


The real variant will be announced at the event. The defaults below are **our guesses**. Bots should always read `state.config` and never hard-code these values.

## Format

- Sit-and-go tables of 2 to 8 seats.
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
