# Instructions for AI coding assistants

This repo is a mock of the Sequoia No Limit 2026 kit. The team bot is `v0/agent.py`.

## Hard rules for bot code. Breaking these gets the bot rejected or makes it lose.

- **Imports:** only the stdlib, `poker_agent_sdk`, `poker_agent_protocol`, numpy, scipy, pandas, scikit-learn, torch (CPU), lightgbm, xgboost, onnxruntime, numba and eval7. **Never** import `poker_engine` or `pokerkit`.
- **Speed:** `act` must return in under 1 ms. Do heavy work, such as tables and model loads plus one warm-up prediction, at import time.
- **State:** don't keep state in `self` or globals between calls. Per-table state goes in the `memory` bytes (64 KB), parsed defensively. Never carry information between tables.
- **Legality:** return only offered kinds. `raise_to` totals must be ints in `[min_amount, max_amount]`. The engine does not clamp them, and floats are rounded down.
- **Variant-proofing:** choose from `state.legal_actions`. Ignore event kinds you don't handle.
- **Sandbox:** no threads at import (only the importing thread survives the fork), no network and no file writes outside `/tmp`.

## Workflow

- Before committing, run `python poker_agent.py check v0/agent.py`, then a 200-table `play` against the examples. You need 0 failed decisions and a p99 under 1 ms.
- Keep changes small and measurable. Compare `avg_points` with a fixed `--seed`.
