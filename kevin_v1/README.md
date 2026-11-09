# kevin_v1: CFR + tournament-EV bot for keep-reveal-discard

`agent.py` is **generated**. Edit the sources in `../train/kevin_v1_src/` and rebuild (see its README):
```
cd ../train/kevin_v1_src
python3 build_bundle.py cfr_tables.npz   # tables -> krd_tables.npz
python3 build_agent.py                   # sources -> agent.py
```

## Layers
1. **Choose (keep/reveal/discard).**
   - A CFR-balanced mixed strategy over 1755 suit-isomorphic 3-card classes.
   - Solved per player count (2–8) × effective stack (3–40bb) in `tmp/krd/solve_cfr.py`.
   - Reveals are mixed, so opponents can't read the hidden card from the shown card. A lone Ace is shown only 17–65% of the time, depending on the spot.
2. **Opponent model (in `memory`, 3.4 KB).**
   - Counts each seat's showdown reveal habits: shown-high, shown-low, pair.
   - Also tracks VPIP, PFR, jam rate, call-vs-jam and aggression.
   - The posterior over an opponent's hidden card is a Dirichlet blend of the CFR prior with that evidence (pseudo-count 15). Exploitation therefore grows only with data.
3. **Preflop, effective stack ≤25bb.**
   - The CFR push/fold chart is conditioned on revealed ranks.
   - It is checked by the exact tournament-points EV of jamming or calling. That EV uses ICM and the hand-cap horizon, and calling ranges come from the chart plus the opponent model.
   - Clear EV spots are decided by EV; close ones follow the CFR mix.
4. **Preflop, deep.**
   - Equity is computed against reveal-conditioned ranges.
   - It opens, or 3-bets/folds (no loose flat calls), with an ICM risk premium.
5. **Postflop.**
   - Monte Carlo equity against each opponent's posterior hidden card. Their revealed card is exact.
   - Equity is shaded against bettors, with a stack-off margin.

## Cap payout switch
`CAP_PAYOUT` in `tmp/krd/core.py`:
- `"icm"` (default, per RULES.md). At the cap, ICM is linear with 2 survivors and concave otherwise. So the last hand is **not** a shove spot: a pure coinflip never gains points.
- `"rank"`: chip-rank places pay 50/30/20. On the last hand every preflop decision goes through the exact rank-payoff EV, so a trailing stack shoves and the leader protects. Measured gain with hand-cap 60 and rank settlement: +1.6 ± 1.3 points.

## Results (practice engine, avg points per table, field mean = 12.5)

| Field | Score |
|---|---|
| vs team bots v12/v10/v13/v3/v17/v0 + check_or_fold | **18.5–21.1** (best: v13 13.6, v10 13.3–13.8, v12 12.6–13.3) |
| vs tmp/krd/opponents (tag, reveal readers, jammer, nit, maniac) | **18.0** (next: reveal_reader 16.7) |
| vs the original template, paired | +8.5 ± 1.7 |
| reveal-exploit field (reveal_reader ×2, reveal_high ×2, …) | 16.7, best seat |

## Timing
- On a clean benchmark (minimum of 15 timings per recorded state), act takes p50 0.1 ms and p99 0.56 ms.
- On a loaded WSL machine, even a trivial loop gets ~11 ms preemption spikes, so check timing on a quiet box.
- Import-time objects are `gc.freeze()`d so garbage collection never pauses act.
