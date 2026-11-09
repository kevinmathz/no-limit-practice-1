# KRD bot pipeline: shared spec

Goal: a tournament bot for the practice variant **keep-reveal-discard** in
`/mnt/c/Users/kevin/onedrive/documents/no-limit-practice-1`. Read these files there:

- `RULES.md`
- `sdk/poker_agent_sdk/__init__.py` (State/Action/Event)
- `sdk/poker_engine/table.py` (engine)
- `poker_agent.py` (check/play)
- `bot/agent.py` (template)

## The game
- 8-max sit-and-go. 2000 chips each.
- Blinds 10/20, rising every 20 hands. The schedule is in `state.config.blinds`.
- Hand cap 250. Payouts are 50/30/20. At the cap, survivors are paid by ICM on stacks.
- Each hand, every player gets 3 cards and **simultaneously** picks:
  - keep (private),
  - reveal (public, a `reveal` event before preflop betting),
  - discard (hidden).
- Then normal NLHE with the 2 cards (kept + revealed).
- The choose phase happens after blinds are posted, before preflop betting.
- At showdown, everyone remaining shows (`show` events carry the 2-card hole = (keep, reveal)).

## Runtime constraints (the final bot)
- Single `agent.py` plus data files beside it.
- stdlib + numpy only.
- `act()` < 1 ms (target p99 < 0.6 ms).
- No state in self/globals between calls; per-table state lives in `memory` bytes (<16 KB).
- Never import `poker_engine`.

## Conventions
Use `tmp/krd/common.py` for:
- card ids,
- the 169-class order (= `no_limit/game/tables/hu_matchups.CLASSES`),
- 3-card choose classes (1755), `CHOOSE_ACTIONS`, `canon3`.

Positions in charts count **from the button**: 0 = button, 1 = SB, 2 = BB, 3 = first to act preflop (UTG) ... . Heads-up: 0 = button/SB, 1 = BB.

The 13-valued "rank" dims use rank_idx 0..12 = 2..A.

## Scope rule for all workers
- Write ONLY inside `/mnt/c/Users/kevin/onedrive/documents/tmp/krd/`.
- Everything else, including the `no_limit` and `no-limit-practice-1` repos, is read-only.
- No git operations.

## Files and owners
| Owner | Files | Contract |
| --- | --- | --- |
| A1 | `evaluator_np.py`, `build_equity.py` -> `equity.npz` | `hu_eq[169,169]` float32, `mw_note` |
| A2 | `solve_cfr.py` -> `cfr_tables_quick.npz`, `cfr_tables.npz`, `CFR_REPORT.md` | see A2 task |
| A3 | `tvalue.py`, `test_tvalue.py`, `ENDGAME_ANALYSIS.md` | see A3 task |
| A4 | `opponents/<name>/agent.py`, `eval.py`, `rank_table.py` | see A4 task |
| A5 | `rt_eval.py`, `oppmodel.py`, `postflop.py`, `test_oppmodel.py` | see A5 task |
