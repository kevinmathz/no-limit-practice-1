# Pokerbot playbook: lessons from the practice run

Written for a Claude agent starting bot work on a **new variant** of the Sequoia No Limit tournament. Read it alongside the official kit (`README.md`, `RULES.md`, `AGENTS.md` and the SDK), which is more complete than our mock. Where the two disagree, **the kit wins**.

---

## ⚡ SPEED IS THE TOP PRIORITY

The build window is short: 60 minutes, then scored rounds every 20 minutes with rising weights. **A good bot uploaded now beats a great bot uploaded late.** A team with no active bot scores 0 for that round.

- **Upload a safe bot in the first 10 minutes:** legal, never crashes, tight. Then improve it.
- **Estimate before you start.** If a task will take 5 minutes or more, tell the user first and give a time estimate. Anything over about 30 minutes is almost certainly out of the question; cut its scope instead (fewer tables, smaller search, skip tuning).
- **Parallelize.** Use one subagent per bot version, each writing only its own `vN/` folder. Then run one round-robin to decide.
- **Pick fast methods.**
  - Rules, tightness, ICM and cheap exact checks give big gains in minutes.
  - CFR needs much more compute than the window allows. v1 took 10 minutes to train and lost to rules.
  - A cross-entropy parameter search takes about 7 minutes on 16 cores and gave +1.3–1.5 points.
- **Keep runtime fast too:** p99 decision time under 0.2 ms against a 1 ms limit, with no time bank in the upload test. Do heavy work at import.
- **Don't wait on anything you don't need.** Run long jobs in the background, keep building, and commit and upload at every milestone.

## Top 8 bots (latest)

From a 3,000-table match (seed 171) with all 8 at one table. "Out" means finishing 4th–8th.

| Rank | Bot | Lineage | Avg points | 1st | 2nd | 3rd | Out |
|---|---|---|---|---|---|---|---|
| 1 | **v21** | v19, with deep preflop stack-offs only with QQ+/AK | **17.2** | 21% | 15% | 10% | 53% |
| 2 | v22 | v19, with stack-offs only with KK+ | 15.7 | 15% | 18% | 14% | 53% |
| 3 | v19 | v13 + learned players-remaining schedule | 15.5 | 23% | 10% | 6% | 61% |
| 4 | v13 | rule line: v0 + ICM + river fold + tight + lock | 13.0 | 21% | 8% | 4% | 68% |
| 5 | v18 | v12 + learned schedule | 11.5 | 7% | 15% | 16% | 62% |
| 6 | v12 | CFR line, tight + lock | 11.4 | 6% | 16% | 15% | 62% |
| 7 | v10 | CFR line, tight | 11.1 | 7% | 16% | 13% | 64% |
| 8 | check_or_fold | baseline | 4.7 | 0% | 2% | 20% | 78% |

- **Fixing the stack-off leak gave +1.7 points (v19 → v21).** v19's "premium" range was the top 4% of hands by equity against a random hand, which ranks 99–77 above AK. A sampled hand showed v19 putting 86 big blinds in with 99 against QQ. **Use an explicit hand set for stack-offs, not an equity percentile.**
- **v23** (v21 with late-game aggression off) scored 16.7 against v21's 16.9 in a separate 3,000-table run (seed 181), the same within noise. See §4.2.
- **Timeouts:** 3–10 per bot over 3,000 tables, all single decisions of 100–130 ms that exhausted the time bank. They are most likely Windows CPU spikes. Check on Linux.
- **Recommended submission: v21,** or v23 for simpler logic near the cap.

### Previous top 8 (seed 161, before v21)
v19 16.4, v13 14.7, v18 13.6, v10 13.2, v12 12.6, v20 12.4, v15 11.6, check_or_fold 5.5.

## 0. TL;DR (what actually moved the score)

1. **Scoring is about placing, not chips.** Payouts are 50/30/20 points for 1st/2nd/3rd and 0 for 4th–8th, averaged over thousands of 8-player tables. **Busting is the dominant loss.** A bot that only checks or folds (`check_or_fold`) beat every "smart" bot we wrote until we made them very tight. It never wins a table, but it slides into 2nd or 3rd while the others knock each other out.
2. **Tightness gave the biggest gains, by far.** Wrapping any bot in a tightening layer gave +1.5 to +8 points: play only the top ~6–10% of hands preflop, bet two pair or better, call top pair (good kicker) or better, and add a guard against committing the stack. Do this first.
3. **ICM-aware commits and a river "can't win" fold helped a bit.** The river check uses the variant's public information. Bet hard with a lock (a hand no unseen card can beat) and fold when the hand is drawing dead.
4. **CFR (MCCFR) on a heads-up abstraction lost to simple rules at 8-max.** Its low-probability "noise" actions and its heads-up ranges played multi-way got it busted early: 86% of seats, half of them by hand 6. Pruning actions under 8%, then adding ICM and tightness, made it competitive (v10/v12), but most of that gain came from the tightness, not the CFR.
5. **Clever theory hurt in a field of passive bots.** MDF-based defending (v6, v7) and hand-tuned schedules by players remaining (v14–v17) did no better or worse in this field. **Measure everything against `check_or_fold` and your current best, and keep only changes that win in simulation.**
6. **Process.** Build the safe template and the harness first, then tighten, add ICM, then add variant-specific edges. Ask the user before any step that takes 5 minutes or more. Use a fixed seed with 2,000+ tables per comparison; noise is about ±0.3–0.5 points.

---

## 1. The platform: hard constraints that break bots

These come from the official docs, and our mock enforces them. Check the new kit's README for changes.

- **Interface:** `class Agent(poker_agent_sdk.Agent): def act(self, state, memory) -> (Action, bytes)`.
  - Actions: `Action.fold()`, `check()`, `call()`, `raise_to(total)`, where `total` is the **whole bet for this betting round**.
  - Variants may add kinds and phases (ours added `choose`). Always pick from `state.legal_actions`.
  - For the automatic choice, return `Action(first.kind, first.min_amount)` with `first = state.legal_actions[0]`.
- **Time:** 1 ms per decision by default, with a 100 ms time bank per table. **The upload test match has no bank:** 20 hands against a check/fold bot, and the upload is rejected if more than 10% of decisions fail. Do heavy work at import (tables, model loads, one warm-up prediction). Our bots run at a p99 of 0.03–0.2 ms in pure Python.
- **Rare long decisions:** Windows sometimes produced 1–100 ms decisions under CPU load. Two runs logged a "timeout, bank exhausted" at 102 ms. Keep `act` far below 1 ms and avoid big allocations in it.
- **State between calls:** `self` and globals are unreliable, because copies are forked and any copy may answer any table. Per-table state goes in `memory` bytes (64 KB, truncated), parsed defensively. Carrying information between tables is against the rules. Constants loaded at import are fine.
- **Legality:** return only offered kinds. Raise totals must be **ints** within `[min_amount, max_amount]`; floats are rounded down, and out-of-range amounts are not clamped and count as failed decisions. Clamp every raise with a helper (see `v0/agent.py: raise_to`).
- **Imports:** stdlib, `poker_agent_sdk`, `poker_agent_protocol`, numpy, scipy, pandas, scikit-learn, torch (CPU), lightgbm, xgboost, onnxruntime, numba and eval7. **Never import `poker_engine` or `pokerkit`**, not even in an unused file in the zip.
- **Packaging:** upload `agent.py`, or a zip with `agent.py` at the root plus the files it reads. A bot folder must be self-contained.
  - Helper modules sit next to `agent.py`. Add `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` before importing them.
  - Load data files by path relative to `__file__`.
  - The bot directory is read-only and not executable; `/tmp` is the only writable place.
- **Threads:** set `OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=1` before importing numpy or torch. Threads started at import don't survive the fork.
- **Wrap everything:** put `try/except` around every bot layer, with a fallback (check if free, else fold, else the automatic choice). One exception costs one decision. Crashes cost restarts.

---

## 2. Format facts that drive strategy (2026 regional; verify for finals)

- **Table:** 8 players, 2,000-chip stacks, blinds start 10/20 and rise every 20 hands, hand cap 250.
- **End of table:** at the cap, the survivors split the remaining points by **ICM** on their stacks.
- **Payouts:** 50/30/20 by default, but read `state.config.payouts`, since rounds may draw others.
- **Busting together:** when several players bust in the same hand, the one with more chips at the start of the hand places higher.
- **Scoring:** each round's score is average points per table over at least 5,000 tables. There are 6 scored rounds with weights 1×–6×, 20 minutes apart. The first 60 minutes are build time.
- **Read `state.config`** and never hard-code: decision_ms, time_bank_ms, memory_bytes, the blinds schedule, hand_cap, payouts.
- **Consequence:** this is a **survival tournament**. Finishing 4th is worth the same as 8th. With ICM, losing chips hurts more than winning them helps, so when in doubt, fold.

---

## 3. Scaffolding to build first (about 30 minutes)

Mirror what we built in `pokerbot-practice`, but put the real kit's engine underneath.

1. **A safe template bot**, like `v0/agent.py`:
   - legal-action helpers (`raise_to` clamp, `call_or_check`, `check_or_fold`);
   - defensive memory JSON;
   - a `try/except` fallback;
   - a phase dispatch: `if state.phase != "betting"` → the variant's handler.
2. **Baseline opponents:** `check_or_fold`, `min_raiser`, `calling_station`, `random_bot`. **Always keep `check_or_fold` in your test fields.** Weak spewing bots inflate how much survival is worth, so also test against your own strong bots.
3. **Harness** (the real kit has `poker-agent play`):
   - fixed seeds;
   - per seat: average points, a place histogram, out-of-money %, failed decisions by kind, and act-time p50/p99/max;
   - a round-robin script (`train/roundrobin.py`): a pool of bots, 7 drawn at random per table, plus fixed seats.
4. **Module isolation:** if you load several bots in one Python process, isolate their same-named helper modules (`abstraction`, `icm`, `tight`, ...). Otherwise every bot silently uses the first one loaded. **This bug corrupted several of our comparisons.** See `poker_agent.py: load_agent_class`.
5. **Version folders:** use `v0/`, `v1/`, …, each self-contained. Write a one-line description in the docstring on line 1 of each `agent.py`, and keep a results table in the README.

---

## 4. Layers that worked, in order of value

Each layer wraps the previous one's action. The order matters.

```
act(state):
  if phase is variant-specific  -> variant handler (e.g. keep/reveal/discard)
  1. guaranteed-loss fold (river)        # showdown.py
  2. lock aggression (can't lose now)    # lock.py: bet ~pot, bypasses filters on the river
  3. base action (rules or CFR policy)
  4. ICM review / veto                   # icm.py
  5. tight filter + top-of-range filter  # tight.py (skipped in late-game aggression mode)
```

### 4.1 Tight layer (`v12/tight.py`): the biggest win
- **Preflop:** continue only with the top `PRE_TOP` share of hands by equity against a random hand. Use 6% (v12/v13) or 10% (v8–v11). Widen to 15–20% when at 15 big blinds or less.
- **Postflop:** bet or raise only with bucket 5+ (two pair or better). Call only with bucket 4+, or bucket 1+ when the call is at most 25% of the pot.
- **Stack-off guard:** committing more than 35% of the stack needs a premium preflop hand (top 4%) or two pair or better.
- **Top-of-range filter:** facing a bet, continue only with the top `min(MDF, 30%)` of the range.
- **Late-game aggression:** in the last 25 hands before the cap, when not the chip leader, the filter is skipped.
- **Effect:** v4 went from 9.2 to v10 at 17.1 points in the round-robin; v3 went from 11.4 to v9 at 13.8.

### 4.2 ICM (`v3/icm.py`)
- **Valuation:** Malmuth-Harville ICM over the non-busted stacks with `state.config.payouts`. Computing it for 8 players takes about 0.05 ms.
- **Tournament-life decisions:** a call of an all-in, or any commitment that leaves less than 15% of the stack. Commit only if p·ICM(win) + (1−p)·ICM(lose) > ICM(fold) + margin.
  - p comes from equity tables (169 preflop classes, postflop buckets per street) against a random hand.
  - p is raised to the power `1 + 0.8·(opponents−1)`, then shaded down further when facing a bet.
- **Bust-aversion margin, in points:** 1.5 with 4–5 players left, 0.8 with 3 or 6, 0.4 with 7–8, and 0 heads-up.
- **Late-game aggression:** in the last 25 hands before the cap, when not the chip leader, the margin is 0 and play widens. The chip leader near the cap plays tighter.
- **Correction: late-game aggression isn't needed.** At the hand cap, points are split by ICM on stacks. A short stack that survives already gets its ICM share, and gambling to catch up is negative in expected points because ICM is concave. With the gentle blinds, **72% of tables reach the 250-hand cap** (median final hand 250), so this matters. v23 (v21 with aggression off) scored 16.7 against v21's 16.9 over 3,000 tables, the same within noise. **Default: no late-game aggression.** Keep the plain ICM veto all the way to the cap.
- **Effect:** about −7 percentage points in the bust rate against v0 alone, and +1.3 points head-to-head.

### 4.3 Public-information checks (variant-specific; ours revealed one hole card)
- **Guaranteed-loss fold** (`showdown.py`, river only):
  - if some live opponent beats us for **every** unseen card x (their hand = revealed card + x + board), fold to any bet;
  - unseen = deck − our cards − our discard − board − all revealed cards;
  - a tie is not a loss.
- **Lock aggression** (`lock.py`):
  - if no unseen card lets any opponent beat or tie past us right now, bet about pot;
  - on the river it's a true lock: bet 0.95× pot, raise to about pot, and skip the filters;
  - it fires on about 10% of postflop checks.
- **Fast evaluator:** a table-driven 7-card scorer (`score7`) with an early exit at the first card that changes the answer, deduplicated by rank (suits only matter when a flush is possible). Worst case with 7 opponents: p99 0.46 ms.
- **For a new variant, ask:** what information does it make public, and can it prove a loss or a win? Those are free, exact edges.

### 4.4 Pruning a CFR policy
- Remove actions with probability under 8% and renormalize. v2 went from 8.3 to v4 at 9.4 points.
- **Why:** 28.5% of v1's busts came from actions its policy gave under 10%.

---

## 5. What didn't work (and why) — don't repeat it blindly

| Idea | Result | Why |
|---|---|---|
| Heads-up MCCFR policy used directly at 8-max (v1) | worst bot, about 6 points | Heads-up ranges played multi-way. Stacks over 80 big blinds were treated as 80. Low-probability shoves. Raise sizes lumped into "half pot", "pot" and all-in. Busted at a median of hand 6. |
| More raise sizes, as pot fractions (v5) | worse than v1 | Twice the infosets, but trained for only 4.7M iterations. Undertrained. |
| MDF as a floor (v6) or exact MDF (v7) | −2 to −5 points against v3 | The field barely bluffs, so defending at MDF pays off value bets. MDF assumes a balanced opponent; ours were exploitable and passive. If you use it, **calibrate the range table to your own range in that spot** (v7 continued 82% against a 66% target until it was calibrated). |
| Tightness schedules hand-picked by players remaining (v14–v17) | −0.3 to −1.8 points | Extra tightness on the bubble (4–5 left) lost chips to the blinds, and loosening in the top 3 gave back ICM. → Learn schedules from simulation instead (§6). |
| Random keep/reveal/discard (v1) | throws away the best card | In variants with card choices, use a cheap heuristic (keep the best 2-card combination, reveal the weaker card). |

---

## 6. Learning parameters quickly (cross-entropy search over simulation)

`train/learn_schedule.py` tunes a looseness multiplier for each player count (2–8) on v12's tight layer:
- the cross-entropy method, 16 candidates per generation, 500 tables per candidate;
- common random seeds within a generation, so candidates are compared on the same deals;
- 16 cores;
- a 4,000-table validation against the untuned baseline at the end.

Use the same pattern for any constant: thresholds, ICM margins, the aggression window. **It is optimal only against the field it was trained on.** Keep the field diverse (your best bots, a CFR bot and `check_or_fold`), and accept a change only if it wins on fresh seeds.

**Result:** after 4 generations in 7 minutes, the learned multipliers were:

| Players remaining | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| Multiplier | 0.74 | 0.37 | 1.58 | 1.41 | 2.08 | 0.69 | 3.34 |

- **Pattern:** tight at 7–8 players, looser at 4–6 (steal blinds while others are scared of the bubble), tight at 3, very loose heads-up.
- This is roughly the **opposite** of our hand-picked schedules (v14–v17), which explains why those lost.
- **Validation** on the same 4,000 seeds: 13.91 vs 12.63 for v12's flat schedule.
- **Fresh seed, top-bot field:** v18 14.2 vs v12 12.7.

---

## 7. CFR notes (if you try it again)

- **Trainer:** `train/mccfr.py`, external-sampling MCCFR (Lanctot et al. 2009, the same algorithm as OpenSpiel and slumbot2019) with regret-matching+ and linear averaging, in pure Python. It ran about 25k iterations/s on 16 cores; v1 got 15M iterations in 10 minutes.
- **Abstraction:**
  - heads-up, stack depths of 10/20/40/80 big blinds;
  - 8 preflop equity buckets and 9 postflop made-hand buckets (imperfect recall);
  - key = depth|street|bucket|spr|current-street history.
- **Deals:** sampled from a precomputed pool, 400k deals with buckets and the showdown winner. Training needs no evaluator.
- **Infosets with different legal action sets:** key the infoset by legal-action set as well, then keep the most-visited set.
- **Convergence:** two seeds differ by about 4% (total variation, visit-weighted), worst preflop. The bot fell back to rules in 1% of decisions.
- **Before trusting it at 8-max:** train on **points or ICM utility**, not chips. Add multi-way or position context. Cap stack-offs. Prune actions under 8%. Otherwise the tightness and ICM layers do the real work.
- GPUs don't help this trainer; it's a pure-Python tree walk. A CPU cluster (MIT ORCD) helps for long runs.

---

## 8. Workflow tips for a time-boxed event

- **Estimate every task.** Tell the user before anything that takes 5 minutes or more, and never plan more than about 30 minutes. Run independent builds in parallel subagents, each creating only its own `vN/` folder, then compare everything in one round-robin.
- **Always compare** with a fixed seed and 2,000–4,000 tables, against `check_or_fold` plus your current best. Report points, 1st %, out-of-money (4th–8th) % and failed decisions.
- **Gate changes:** check passes, 0 failed decisions, p99 well under 1 ms. Ship only a version that beats the current best in the round-robin.
- **Commit after each milestone.** Upload early (one upload per 30 s). Every scored round snapshots the active bot when it starts, and a missing bot scores 0.
- **Hand histories** are available after each round. Mine them for opponent tendencies, which are aggregate only, since there's no cross-table memory.

---

## 9. Final standings (practice variant: keep-reveal-discard, gentle blinds)

Top-8 match, 3,000 tables: **v12 15.8**, v10 15.7, check_or_fold 14.8, v13 12.5, v11 10.8, v9 10.8, v7 10.5, v0 9.2.

Round-robin, 4,000 tables (check_or_fold at every table): v12 15.2, v17 14.9, v10 14.8, v16 14.5, v13 13.9, v14 13.0, check_or_fold 12.8, v15 12.6, v11 11.1, v9 10.5, v8 10.4, v7 9.7, v0 9.0.

7 copies of v12 against 1 check_or_fold, 2,000 tables: v12 13.7 per seat, check_or_fold 4.3 (it never finishes 1st or 2nd, 3rd 21% of the time). Tight bots eliminate the check/fold exploit.

**Recommended starting point for a new variant:** copy v12's layered structure (tight + ICM + public-info checks), replace the variant-specific pieces, add the baselines and harness, and run the parameter search from §6.

---

## 10. Bot catalog (every version we built)

All bots are in `pokerbot-practice/`. "Pts" is average points per table in the latest round-robin that included the bot. The 1st/2nd/3rd payout is 50/30/20.

| Bot | Built from | What it adds | Best result | Verdict |
|---|---|---|---|---|
| `examples/check_or_fold` | — | Checks when free, otherwise folds | 18.3–21.7 pts. It never finishes 1st and busts at a median of hand 103. | **The benchmark to beat.** It exploits fields that bust each other. |
| `examples/min_raiser`, `calling_station`, `random_bot` | — | Spewy baselines | 1.6–7 pts | Only good for smoke tests. They make survival look more valuable than it is. |
| **v0** (was `bot/`) | — | Rule bot: preflop hand score, push/fold at 12 big blinds or less, made-hand postflop logic. Keeps the best 2 of 3 cards and reveals the weaker. | 9.0–14.0 pts | Strong early baseline, but loose. Busts 69–81% of the time. |
| **v1** | — | Heads-up external-sampling MCCFR policy (`train/mccfr.py`, 15M iterations). Random reveal/discard. | 5.6–6.2 pts | Worst real bot. Busts in 86% of seats, half by hand 6. |
| v2 | v1 | + river guaranteed-loss fold + ICM veto and late aggression | 7.4–10.8 | Better than v1, still weak. |
| v3 | v0 | + river guaranteed-loss fold + ICM veto and late aggression | 10.2–15.0 | Small gain over v0. Fewer busts. |
| v4 | v2 | Policy actions under 8% removed | 9.2–11.3 | +1 point over v2. |
| v5 | v1 | Pot-fraction raise sizes (1/3, 2/3, 1, 1.5, 2.5 pot, all-in) | 6.2–6.3 | Undertrained, with 4.7M iterations for twice the infosets. Not worth it under time pressure. |
| v6 | v3 | MDF floor (never fold more than 1−MDF) + bluff cap | 7.8–10.6 | Worse. Defending against a passive field pays off its value bets. |
| v7 | v3 | Exact MDF (continue with exactly the top MDF of a self-calibrated range) | 9.5–12.7 | Worse than v3 in most runs. |
| v8 | v0 | + `tight.py` (top 10% preflop, bet with top pair good kicker+, call with top pair+, stack-off guard) | 10.4–13.0 | +1.5 over v0. |
| v9 | v3 | + `tight.py` | 10.5–13.8 | +2.4 over v3. |
| **v10** | v4 | + `tight.py` | **14.8–18.9** | **The biggest jump** (+8 over v4). First bot to rival check_or_fold. |
| v11 | v7 | + `tight.py` | 10.8–12.9 | Okay. |
| **v12** | v10 | Tighter (top 6% preflop, bet with two pair+, call with top pair good kicker+, top-of-range filter at min(MDF, 30%)) + `lock.py` (bet about pot when no unseen card beats us) | **15.2–17.7** | **Best overall.** Beats check_or_fold head-to-head. As 7 copies against 1 check_or_fold: 13.7 vs 4.3. |
| v13 | v9 | Same changes as v12 | 12.5–14.4 | Wins the most 1sts (20–22%) but busts 62–69%. |
| v14 | v13 | Step schedule by players remaining (tightest at 4–5 left, loosest in the top 3) | 11.7–13.0 | Slightly worse than v13. |
| v15 | v13 | Smooth schedule + doubled ICM margin on the bubble | 12.6–12.9 | About even with v13. |
| v16 | v10 | Step schedule | 12.8–14.5 | Slightly worse than v10. |
| v17 | v10 | Smooth schedule | 13.1–14.9 | About even with v10. |
| **v18** | v12 | Looseness multiplier per player count, **learned** by a cross-entropy search over simulation (tight at 7–8 players, loose at 4–6, very loose heads-up) | **14.2** in the top-bot field, 1.5 above v12 | **Best of the v10/v12 line.** The learned schedule beat every hand-picked one. |

**Lineages:**
- **CFR line:** v1 (MCCFR policy) → v2 (+ river fold + ICM) → v4 (prune actions under 8%) → v10 (+ tight) → v12 (tighter + top-of-range + lock) → v18 (+ learned players-remaining schedule).
- **Rule line:** v0 (rule bot) → v3 (+ river fold + ICM) → v9 (+ tight) → v13 (tighter + top-of-range + lock) → v14/v15 (+ hand-picked schedules).

**Best bots depend on the field.** Fields with spewy or loose bots favour the survival line: v18, then v12 and v10. Fields of strong, tight bots favour v13 (16.6 points, 27% 1st), because once nobody spews, **winning tables matters more than surviving**. Submit the line that wins in the field you expect, or blend the two. Both come from the CFR line, but the tightness, ICM and lock layers do most of the work.

---

## 11. Log of the key runs (newest first)

All runs are 8-seat sit-and-gos. "Out" means finishing 4th–8th.

0. **Top-bot field, 3,000 tables (seed 141):**
   - Field: v18, v12, v10, v13, v17, v15, v16 and check_or_fold.
   - Results: **v13 16.6 (1st 27%)**, v18 14.2, v15 14.2, v10 13.1, v12 12.7, v16 12.7, v17 11.8, check_or_fold 4.6.
   - In a field with no spewy bots, check_or_fold collapses and the aggressive tight bots win.
   - 0–6 timeouts per bot came from rare decisions over 100 ms, most likely Windows hiccups.
1. **v12 ×7 vs check_or_fold, 2,000 tables (gentle blinds):** v12 13.7 per seat (1st 14%, out 60%); check_or_fold 4.3 (3rd 21%, out 79%).
2. **Round-robin, 4,000 tables, check_or_fold plus 7 random from the pool (seed 121):**
   - v12 15.2, v17 14.9, v10 14.8, v16 14.5, v13 13.9, v14 13.0, check_or_fold 12.8, v15 12.6, v11 11.1, v9 10.5, v8 10.4, v7 9.7, v0 9.0.
   - A few timeouts (1–2 per bot) came from Windows CPU-load spikes over 100 ms.
3. **Top 8, 3,000 tables (seed 111):** v12 15.8 (1st 4%, out 44%), v10 15.7, check_or_fold 14.8 (out 34%), v13 12.5 (1st 20%), v11 10.8, v9 10.8, v7 10.5, v0 9.2.
4. **Round-robin, 3,000 tables (seed 101), gentle blinds:** check_or_fold 18.3, v12 17.7, v10 15.9, v13 14.4, v7 12.0, v9 11.7, v11 11.4, v0 10.9, v8 10.6, v3 10.2, v4 10.0, v6 7.8, v2 7.4.
5. **Top 8, 2,000 tables (seed 81), gentle blinds:** check_or_fold 21.7, v10 18.9, v11 11.0, v9 10.8, v8 10.4, v7 9.5, v3 8.9, v0 8.8.
6. **Round-robin, 3,000 tables (seed 71), original steep blinds that went up to 1000/2000:** check_or_fold 20.6, v10 17.1, v9 13.8, v8 13.0, v11 12.9, v7 12.7, v0 11.5, v3 11.4, v4 9.2, v6 8.9, v2 8.1, v5 6.2.
7. **All of v0–v6 plus check_or_fold, 2,000 tables (seed 41), after the module-isolation fix:** check_or_fold 26.7, v3 15.0, v0 13.6, v4 11.3, v2 10.8, v6 10.6, v5 6.3, v1 5.6.
8. **v1 bust analysis, 2,000 tables, mixed field:**
   - v1 busts in 86.5% of seats; the median bust hand is 6, and 78% happen at the first blind level.
   - 70% of busts are preflop; in 71%, v1 shoved and got called.
   - 28.5% of the busting actions had a policy probability under 10%.
   - Fallback rate 1%.
9. **v1's postflop play, 100 hands:**
   - It bets half pot almost always and bluffs air 39% of the time.
   - It folds air to a bet 24 of 26 times and calls with any piece 31 of 42.
   - It lost 46 big blinds postflop.
10. **v0 vs v1 early, 1,000 tables:** heads-up v0 41.4 vs v1 38.6. With 4 of each, v0 14.6 vs v1 10.4.
11. **Example-bot field, original blinds:** check_or_fold took 2nd 98% of the time. The spewy bots busted at a median of hands 1–4.

**How much to trust a difference:** about ±0.15 points between seeds on 2,000-table runs of the same field. Across different fields, rankings in the middle shift by 1–2 places. Trust gaps above about 1 point.

---

## 12. Key takeaways (checklist for the next variant)

1. **Read the scoring first.** If only the top 3 places pay, survival dominates. Build `check_or_fold` on day zero and make it the bar to beat.
2. **Start tight and loosen only when simulation proves it pays.** Preflop top 6–10%, value bets only, and a stack-off guard. This alone beat every clever strategy we tried.
3. **Put ICM on every commitment that risks the stack.** Use equity tables built at import plus Malmuth-Harville over `config.payouts`.
4. **Exploit what the variant makes public.** Prove losses (fold) and locks (bet pot) by enumerating the unseen cards. The checks are cheap and exact.
5. **Use CFR only as a base policy, never alone.**
   - Heads-up abstractions misplay 8-max.
   - Prune actions under 8%.
   - Train on point utility if you can.
   - It needs much more compute than a 60-minute build window allows unless the abstraction is tiny.
6. **Rankings depend on the field.** Against spewy fields, survival (v12/v18) wins. Against tight, strong fields, table-winning aggression (v13) wins. Test against several fields and against the real opponents' hand histories after round 1.
7. **Distrust theory that assumes balanced opponents** (MDF, GTO defense). Measure against the real field.
8. **Tune constants by search, not by hand.** Hand-picked schedules (v14–v17) lost. Use the cross-entropy method with common random seeds, then validate on fresh seeds.
9. **Engineering first:**
   - legal-action helpers and `try/except` fallbacks;
   - per-table memory only;
   - p99 under 0.2 ms;
   - self-contained bot folders;
   - module isolation in the local harness;
   - 0 failed decisions before every upload.
10. **Use a parallel, time-boxed workflow.** One agent per bot version, an up-front time estimate (ask the user at 5+ minutes, nothing over ~30), a single round-robin to decide, and a commit after each milestone.
11. **The official kit's README, RULES and AGENTS override this playbook.** Re-check limits, payouts, blinds and the variant's new phases and actions before reusing any code here.

---

## 13. Final thoughts: v19 and the path that led to it

### What v19 is
(Update: **v21** = v19 with stack-offs limited to QQ+/AK is now best at 17.2. Everything below still applies.)

v19 was our best bot: 16.4–17.0 points in top-bot fields, wins 26% of tables, p99 0.17 ms. It's a **rule-based** bot with five layers stacked on top. It has no CFR policy.

1. **Base (v0):** a rule bot.
   - Preflop: a hand score from high card, kicker, suitedness and connectedness, with push/fold at 12 big blinds or less.
   - Postflop: made-hand categories.
   - Choose phase: keep the best 2 of 3 cards and reveal the weaker one.
2. **Safety (v3):**
   - Malmuth-Harville ICM veto on every commitment that risks the stack (`icm.py`).
   - Late-game aggression when not the chip leader near the hand cap.
   - River guaranteed-loss fold from opponents' revealed cards (`showdown.py`).
3. **Tightness (v9):** `tight.py`, with a premium-only preflop range, value-only postflop play and a stack-off guard.
4. **Tighter, plus exact edges (v13):**
   - preflop top 6%, bet two pair or better, call top pair (good kicker) or better;
   - continue facing a bet only with the top `min(MDF, 30%)` of range;
   - `lock.py`: when no unseen card lets anyone beat us, bet about pot (0.95× pot on the river, skipping every filter).
5. **Learned schedule (v19):** a looseness multiplier for each number of players remaining, found by a 7-minute cross-entropy search over simulated tables (`train/learn_schedule.py`, 3 generations):

| Players remaining | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| Multiplier on the preflop range and TOP_CONT | 0.75 | 0.32 | 0.46 | 1.18 | 1.59 | 0.96 | 2.74 |

In words: **very tight while 6–7 players remain** (let the others bust each other), **open up near the bubble with 4–5 left** (steal from players scared of busting), stay about normal at 3, and **play very loose heads-up**.

### The path that led to it, and what each step taught us

| Step | Result | Lesson |
|---|---|---|
| v0 rule bot vs v1 heads-up MCCFR | v0 won | A heads-up CFR abstraction misplays 8-max. Simple rules are a strong start. |
| `check_or_fold` beat everything | 20–27 points with 0% wins | **This is a survival tournament.** Busting is the main loss. |
| v1 bust analysis | 86% bust, half by hand 6 | The losses came from early stack-offs and low-probability actions. |
| ICM + river fold (v2/v3) | small gains | Necessary, but not enough alone. |
| MDF defending (v6/v7) | worse | Theory that assumes balanced opponents hurts against a passive field. |
| **Tight layer (v8–v11)** | **+1.5 to +8 points** | **The biggest single win.** |
| Tighter + lock (v12/v13) | beat check_or_fold | Exact public-information edges are cheap and safe. |
| Hand-picked schedules (v14–v17) | no gain | Intuition about the bubble was wrong. |
| **Learned schedules (v18/v19)** | **+1 to +1.7 points** | **Search beats intuition.** The learned shape was the opposite of our guess. |
| Fields without spewy bots | check_or_fold collapses to about 5; v13/v19 win | Rankings depend on the field. Table-winning aggression matters once nobody gives chips away. |

### What to do first on a new variant (about 60 minutes)
1. **Minutes 0–10:** scaffold the bot, the baselines (`check_or_fold` first) and the harness. Upload a safe, tight bot.
2. **Minutes 10–25:** port v19's layers. Rewrite the variant-specific parts: card buckets and equity tables, the choose or other new phases, and the public-information checks (provable loss and provable lock). Keep ICM and the tight layer as they are.
3. **Minutes 25–40:** run the cross-entropy search over the tight constants and the players-remaining schedule on 16 cores. Tell the user the time estimate first.
4. **Minutes 40–60:** run a round-robin against `check_or_fold`, a spewy field and a tight field. Upload the winner. After round 1, read the hand histories to see which kind of field we're in, then pick the survival line (v18-style) or the winning line (v19-style).

### Open ideas we didn't have time for
- **Longer schedule searches:** run 10+ generations, and also learn the ICM margins and the aggression window. Each search had only 3–4 generations.
- **v20 (the CFR line with a learned schedule):** rerun on uncontended cores.
- **Opponent modeling within a table, using `memory`:** track each seat's VPIP and aggression and loosen against nits.
- **Use opponents' revealed cards in equity estimates:** today we use them only for the exact lock and fold checks.
- **Retrain CFR on point (ICM) utility** with multi-way context.
