# kevin_v1 sources (the bot in `../../kevin_v1/` is GENERATED from these)

`kevin_v1/agent.py` is a single file with these modules embedded. `kevin_v1/krd_tables.npz` is
built from the solver and equity outputs. Edit here, then rebuild. All paths resolve relative to
the repo, so the scripts run from any checkout.

```
python3 train/kevin_v1_src/build_agent.py            # core/tvalue/oppmodel/postflop/rt_eval -> kevin_v1/agent.py
python3 train/kevin_v1_src/build_bundle.py cfr_tables.npz   # needs cfr_tables.npz (see below) -> kevin_v1/krd_tables.npz
python3 poker_agent.py check kevin_v1/agent.py
```

| File | Role |
|---|---|
| `core.py` | Bot logic: choose policy, push/fold + exact tournament-EV check, deep 3-bet-or-fold, postflop glue, any-payout ICM, soft deadline |
| `tvalue.py` | Tournament value: exact top-3 ICM batch, jam/call EV, risk premium, rank-mode settlement (`CAP_MODE`) |
| `oppmodel.py` | Binary per-table opponent model in `memory` (VPIP/PFR/jam/call-jam/aggression, reveal habits, posteriors) |
| `postflop.py`, `rt_eval.py` | Time-budgeted Monte Carlo postflop decisions and the fast pure-Python evaluator |
| `common.py` | Card ids, the 169-class order, 1,755 canonical 3-card classes and the choose-action mapping |
| `evaluator_np.py`, `build_equity.py` | numpy 7-card evaluator; builds `equity.npz` (169×169 HU equity, included here, about 8 min on 6 cores) |
| `solve_cfr.py` | Mini-batch discounted CFR for choose + push/fold per (players × stack) cell; per-cell checkpoints in `cells/`. Quick run: `--quick` (about 6 min); full run: about 11 min on 6 cores. `cfr_tables*.npz` (4 MB) aren't committed: rerun to regenerate. |
| `build_agent.py`, `build_bundle.py` | Amalgamator (embeds modules, adds `gc.freeze()`, a literal `class Agent`) and table bundler |
| `eval.py`, `rank_table.py`, `opponents/` | Paired benchmark harness (`--compare A B`, `--settle icm|rank`, `--hand-cap`) and test opponents |
| `bench_states.py`, `diag.py` | True-cost timing (minimum of 15 runs per recorded state) and chip-attribution leak finder |
| `test_tvalue.py`, `test_oppmodel.py` | Unit tests. The timing asserts fail on a loaded machine; that's expected. |
| `CFR_REPORT.md`, `ENDGAME_ANALYSIS.md`, `SPEC.md` | Solver convergence and strategy findings; ICM/hand-cap endgame analysis; shared conventions |

Notes:
- `CAP_PAYOUT` in `core.py` is `"icm"`, matching the official rules (the cap settles by ICM). `"rank"` is only for a chip-rank cap.
- ICM handles any payout list: places 1–3 exact; places 4+ use the exponential-race identity on fixed samples, accurate to about 0.1–0.2 points on average.
