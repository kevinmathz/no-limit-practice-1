# Endgame analysis: hand-cap settlement, last-hand jams, horizon effects

All numbers come from `tvalue.py`. The analysis runs used `set_mc_samples(512)`; the runtime default is 32. Setup: 1000/2000 blinds, 16,000 chips in play, payouts 50/30/20. Two cap-settlement modes:

- `icm`: the real engine rule. Survivors at hand 250 are paid by Malmuth-Harville ICM.
- `rank`: the hypothetical variant. Survivors are paid 50/30/20 by chip rank, and ties split.

"Eq needed" means the equity hero needs **when called** for the action to have at least the EV of folding. As a reference for these equity numbers:

- A random hand vs a random hand has 0.50.
- Any two cards vs a top-30% calling range has about 0.36–0.40.
- Any two cards vs a top-10% range has about 0.30–0.33.
- A top-20% hand vs a random hand has about 0.60–0.65.

`any` means even 0% equity is enough, because of fold equity or dead money. `never` means even 100% equity is not enough. `nan` means hero was already all-in from posting the BB.

## TL;DR

1. **Under the real rule (ICM at the cap), the last hand is not special.** The value of a stack at the cap equals its ICM value now. So there is no "freeroll" and no reason to gamble on hand 250.
   - A no-dead-money coinflip always loses tournament EV. The break-even equity for an all-in flip is **0.51–0.75** in every pair we tested.
   - The intuition "unless you are dominating, go all in on the very last hand" is **wrong under ICM**.
2. **At 1000/2000 everybody has at most 4.5 bb, so pot odds dominate.** Dead blinds and fold equity make many jams and calls correct, regardless of the cap:
   - BB calls of a jam need only 0.10–0.40 equity.
   - A 1000–1500 stack jamming into a big BB needs 0.36–0.46 when called, and `any` once the BB folds 50% of the time.
3. **The cap creates "locked-in" value.** A player who will not have to post a blind before the cap can fold to it.
   - That is worth **+1.5 to +8 points** over continuing to gamble normally, because the others keep busting each other while he sits.
   - Medium stacks gain the most.
   - Rule: in the last *m* hands (m = players left), when your next blind falls after the cap, play only clearly +ICM spots (bigger risk premium).
4. **Under rank settlement the last hand is purely ordinal:**
   - The chip leader never risks the lead (`never`).
   - A player out of the money jams or calls **any two**, because 0 is already locked in.
   - Paid players flip only if winning moves them up more than losing moves them down.
   - With 3–10 hands left, the 4th-of-4 short stack should **wait**: it has option value from others' busts and the last-hand shot, and needs 0.68–0.83 to flip earlier.
   - Stacks just above the short stack should attack it: they need only 0.19–0.37 to jam into it.

## 1. Model (what `value_rows` computes) and why

- **`hands_left == 0`.** The value is the exact engine settlement, and it is tested to 1e-9 against `poker_engine.table.icm`.
  - Busted rows pay place points. A hero who busts gets place = (#alive after) + 1. When `start` stacks are given, co-busters with a bigger starting stack place above hero, as in the engine.
  - In rank mode the payout is the chip rank, with ties splitting the tied places.
- **ICM mode with `hands_left > 0`: plain ICM.** The cap payout *is* ICM of the final stacks.
  - If chips are roughly a martingale (fair play), then E[ICM(S_cap)] ≈ ICM(S_now). The gap is only the curvature of ICM along the path.
  - We built the MC below for ICM mode too (`ICM_MC=True`). It moves values by 0–3 points in either direction (for example, 4A at H=3 gives 38.1/23.3/19.8/18.9 vs ICM 37.3/27.8/21.1/13.8). That difference mostly reflects the model's assumptions (Gaussian increments, a fixed variance rule), not better information.
  - So the runtime uses exact ICM: it is the cap rule itself, it is fast and smooth, and it matches the far-horizon limit.
- **Rank mode with `hands_left > 0`: Monte Carlo.** The settlement is a step function, so the horizon matters a lot (50/30/20/0 at H=0 vs about 47/25/18/10 at H=1). The MC works as follows:
  - **Fixed noise.** The noise is pre-drawn at import: antithetic Latin-hypercube normals. Because of common random numbers across rows, comparing jam/fold/call rows is low-noise.
  - **Per-hand std.** Player i's per-hand std is `min(KAPPA*bb, EFF_C*min(s_i, mean opp stack))`, with KAPPA=3 bb deep and EFF_C=0.7 shallow. The idea is that you can only win or lose what is covered. A 0.75-bb stack that is all-in about once an orbit with roughly triple-or-nothing has a per-hand std of about 0.7·s.
  - **Steps.** The std is recomputed from the current stacks at each step: 1 step if H ≤ 4, 2 steps otherwise.
  - **Zero-sum increments.** Increments are centred over the live players.
  - **Absorption.** A stack is absorbed at 0 either at the end of a step or through the exact Brownian-bridge crossing probability exp(−2·s0·s1/σ²) inside a step. One step therefore already captures the risk of busting along the way.
  - **Places.** Busts place below survivors, earlier and deeper busts lower. Survivors are ranked with a soft rank of width 0.05·(average stack), so the value is continuous in the stacks.
  - **Blend into ICM.** The MC value is blended into plain ICM as the horizon variance grows, with weight `ramp(nu; 1, 4)`, where nu = Σ bb_h² / avg_stack² is "how many average all-ins fit in the horizon". The hand-count ramp runs over 30..60.
    - At 1000/2000 with 4 players: nu = 0.25 per hand. So H ≤ 4 is pure MC and H ≥ 16 is pure ICM.
    - This reflects that, over a long horizon, the tournament resolves by busts and the settlement rule stops mattering.
- **Blind drift.** This is optional (`STEAL`, default 0). A proportional "big stacks steal blinds" drift was tried: it was far too strong at 1000/2000, because 20 hands of blinds is 60,000 chips. At these levels a short stack posting all-in is roughly chip-EV neutral (dead money), so the blind effect is carried by variance and absorption, not by drift.
- **Locked-in analysis.** `hero_sigma=0` freezes hero (folds to the cap) while the others keep their variance. To account for blinds hero must post, deduct them from his stack first.

Knobs to calibrate from self-play logs (A4/A5): `KAPPA`, `EFF_C`, `NU_LO/NU_HI`, and `STEAL`.

## 2. All-in flip for the smaller stack, no dead money: break-even equity (rows = hero, cols = villain)

In ICM mode the table is the same for every horizon. Under ICM nobody should ever take a pure flip: every entry is above 0.5.

```
ICM (any H)                          rank, H=0 (last hand)               rank, H=3
4A [8000,4000,2500,1500]
 8000:   -   0.56 0.53 0.52           -   never any  any                 -   0.72 0.70 0.75
 4000: 0.75   -   0.58 0.54          0.60  -   never never              0.53  -   0.37 0.29
 2500: 0.71 0.67   -   0.54          0.67 0.67  -   0.80                0.66 0.68  -   0.26
 1500: 0.63 0.60 0.56   -            any  any  any   -                  0.77 0.83 0.81  -
5 [6000,4000,3000,2000,1000]
 6000:   -  0.61 0.57 0.54 0.52       -  never never never any          -  0.65 0.65 0.67 0.72
 4000: 0.69   -  0.60 0.55 0.52      0.60  -  0.60 0.75 never           0.53  -  0.53 0.51 0.55
 3000: 0.66 0.63   -  0.56 0.52      0.40 0.50  -  0.67 0.73            0.52 0.52  -  0.52 0.47
 2000: 0.61 0.59 0.57   -  0.52      any  any  any   -  any
 1000: 0.56 0.55 0.54 0.53   -       any  any  any  any   -
3A [9000,4500,2500]
 9000:   -  0.53 0.51                 -  never any                       -  0.62 0.69
 4500: 0.63   -  0.52                0.33  -  never                     0.48  -  0.40
 2500: 0.57 0.53   -                 any  any   -                       0.54 0.57  -
```

Reading the table:

- **ICM.** The middle stacks pay the biggest premium, especially against the leader (4000 vs 8000 needs 0.75). The smallest stacks and the leader pay the least (0.51–0.56).
- **Rank, last hand.** A stack that is already out of the money (1500 in 4A; 2000 and 1000 in the 5-player case) flips with anything. The leader flips only against a stack that cannot overtake it (8000 vs 2500 is `any`). Paid middle stacks need 0.6–0.8, or `never` when losing would drop them to unpaid.
  - Example: in the 5-player case, the 1000 stack flipping vs the 6000 stack only reaches a 2-way tie for 4th, so it is a gamble without upside. Doubling against 3000 instead creates a 3-way tie for 3rd, which pays 6.7.
- **Rank, H=3.** The pattern turns around for the short stack: it now needs 0.77–0.83. Busting now forfeits about 16 points of "wait for others to bust, or gamble on the last hand" value. Stacks just above it want the confrontation (0.26–0.37).

## 3. Open-jam into the BB (BB posted 2000, SB dead 1000)

Each cell is `eq needed if always called / eq needed if BB calls 50%`.

```
ICM, any H                                      4A [8000,4000,2500,1500]
 8000:      -       0.52/0.22  0.46/any   0.40/0.40      (1500 BB is already all-in: a "call")
 4000:  0.71/0.53       -      0.52/0.09  0.43/0.43
 2500:  0.65/0.35  0.61/0.27       -      0.43/0.43
 1500:  0.54/0.01  0.50/any   0.46/any        -
ICM, any H                                      6 [5000,3000,3000,2500,1500,1000]
 5000:      -     0.52/0.15 0.52/0.15 0.49/0.05 0.42/0.42 0.36/0.36
 3000:  0.57/0.24     -     0.55/0.19 0.51/0.08 0.42/0.42 0.37/0.37
 2500:  0.54/0.14 0.52/0.10 0.52/0.10     -     0.43/0.43 0.37/0.37
 1500:  0.46/any  0.44/any  0.44/any  0.44/any      -     0.37/0.37
 1000:  0.40/any  0.38/any  0.38/any  0.38/any  0.37/0.37     -
```

How the rank-mode thresholds change with the horizon, for 4A with hero jamming into the given BB:

| hero → BB | ICM | rank H=0 | H=1 | H=3 | H=5 | H=10 | H=20 (≈ICM) |
|---|---|---|---|---|---|---|---|
| 8000 → 4000 | 0.52/0.22 | never | 0.82/0.63 | 0.66/0.34 | 0.64/0.35 | 0.62/0.36 | 0.51/0.21 |
| 4000 → 8000 | 0.71/0.53 | 0.60/0.20 | 0.50/0.15 | 0.49/0.15 | 0.55/0.24 | 0.58/0.31 | 0.71/0.53 |
| 4000 → 2500 | 0.52/0.09 | never | 0.47/any | 0.35/any | 0.33/any | 0.32/any | 0.57/0.10 |
| 2500 → 1500 | 0.43 | 0.33 | 0.32 | 0.21 | 0.18 | 0.26 | 0.44 |
| 1500 → 8000 | 0.54/0.01 | any | 0.44/any | 0.68/0.28 | 0.77/0.47 | 0.75/0.44 | 0.54/0.01 |

In rank mode with 5 or 6 players, the out-of-the-money short stacks jam much more freely, even before the last hand. In the 6-player case at H=1, 1000/1500 into anyone needs 0.04–0.19. At H=3, the 1000 stack needs 0.28–0.32.

## 4. BB calling a jam (BB posted 2000, SB dead 1000)

```
ICM (any H)                           rank H=0                            rank H=3
4A  8000:   -   0.29 0.10 any          -   any never any                 -   0.37 0.14 any
    4000: 0.52   -   0.13 any         0.40  -   0.50 any                0.38  -   0.08 any
    2500: 0.22 0.20   -   any         any  any   -   any                0.46 0.32  -   any
5   6000:   -   0.33 0.19 any any      -   0.50 0.17 any any             -   0.26 0.17 any any
    4000: 0.42   -   0.22 any any     0.20  -   0.20 any any            0.28  -   0.15 any any
    3000: 0.27 0.26   -   any any     any  any   -   any any            0.23 0.17  -   any any
```

- **ICM.** The BB has already posted 2000, so calls are cheap: almost any two vs short jammers, and 0.2–0.4 vs big jammers. The exception is the 4000 stack in 4A facing the 8000 leader, which needs 0.52.
- **Rank, last hand.** The leader folds when a loss would hand the lead to the jammer (8000 vs a 2500 jam: `never`). Otherwise calls are wide.

## 5. Hand-cap horizon vs normal ICM: "locked-in" stacks

Values below are expected points. "play" means hero keeps normal variance (MC). "freefold" means hero is not due a blind before the cap and folds every hand while the others keep gambling. In this table ICM mode uses the MC (`ICM_MC=True`) to measure the effect.

```
ICM mode                      now   H=1 play/free   H=2 play/free   H=3 play/free
4A 8000                       37.3   37.7 / 38.2     38.1 / 39.0     38.2 / 39.7
4A 4000                       27.8   25.5 / 29.1     25.1 / 30.3     23.2 / 31.2
4A 2500                       21.1   20.7 / 23.5     20.3 / 25.2     20.3 / 26.6
4A 1500                       13.8   16.1 / 17.6     16.5 / 20.0     18.4 / 22.4
5  3000                       20.7   19.4 / 22.0     18.7 / 23.6     17.8 / 24.8
6  2500                       16.6   16.0 / 17.7     15.9 / 19.0     15.7 / 19.7
rank mode
4A 4000                       30.0   25.1 / 28.1     24.7 / 27.3     23.0 / 27.6
4A 1500                        0.0   10.5 /  9.4     13.5 / 14.0     16.5 / 17.2
6  1500                        0.0    5.4 /  1.2      8.1 /  3.7      9.7 /  7.3
```

What the table shows:

- **Folding to the cap is worth the most in the middle.** It is worth +3.6 to +8 points for middle stacks over 1–3 hands. The table's other stacks move up when somebody busts, and a locked-in player keeps his stack.
- **Normal ICM has no such option.** With a long horizon you must keep posting, so the lock-in value is an effect of the hand cap alone.
- **Blinds cost lock-in value.** Charging the folder the average blinds, (sb+bb)/m per hand, erases the benefit after about m hands. In 4A at H=5, folding with 4000 is worth 15.2 vs 23.6 when playing. So the lock-in only applies when **your next forced blind comes after the cap**.
- **Rank mode differs for the out-of-the-money short stack.** For it, folding is worth about the same as playing (4A 1500) or much less (5 and 6 players): it has to gamble at some point, and the last hand is the cheapest time.

## 6. Rules of thumb for the bot

Let `H = hands_left`, `m = players left`, and `nb =` number of hands until hero's next SB/BB.

1. **Always value chips with `value_rows`.** In ICM mode, decide jams and calls by `jam_ev` / `call_ev` with ICM. That includes the very last hand: no "last-hand gamble" logic. In particular, never take a pure no-dead-money flip under ICM.
2. **Lock-in (ICM mode).** If `nb > H` (no blind left before the cap), add a lock-in premium: require **+0.05–0.10 equity** over the `call_breakeven` / jam break-even. Middle stacks use the high end, leader and micro-stacks the low end. Equivalently, compare against `value_rows(..., hero_sigma=0)` for the fold row instead of plain ICM. If `nb ≤ H` and hero has ≤ 1.5 bb, ignore lock-in: you will be blinded into a worse spot.
3. **Short stack at 1000/2000 (≤ 1 bb), ICM mode.** Jam any two into a big BB when first in: the break-even is 0.36–0.46 if always called and `any` with 50% folds. Call any two from the BB vs a short jam.
4. **Medium stack facing the leader, ICM mode.** This has the largest risk premium: needs 0.71 to jam into the leader's BB if always called, and 0.52 to call the leader's jam from the BB in 4A. Steal from the leader only with real fold equity.
5. **Leader, ICM mode.** Pressure the middle stacks: jam into a 2500–3000 BB needs 0.46–0.52 called and about 0.1 with 50% folds. Call short jams wide (0.10–0.30).
6. **Rank mode, last hand (`H=0`).** Compute the result exactly with `value_rows(mode="rank")`. In practice:
   - Leader: never risks the lead. Fold, or call only when even a loss keeps it first.
   - Unpaid stacks: jam or call any two.
   - Paid stacks: act only when winning moves them up and losing does not drop them out of the money.
7. **Rank mode, last 1–10 hands.**
   - 4th of 4 (one unpaid seat): do not flip early unless forced (needs about 0.7–0.8). Take the gamble on the last hand, or when your BB forces it.
   - Stacks just above the short stack: attack it (0.2–0.35).
   - Leader: tighten as the cap approaches (0.66 at H=3 rising to 0.82 at H=1).
8. **Mid-horizon (`H ≥ ~16` at 1000/2000, or earlier levels).** The modes converge to plain ICM, so use ICM.

## 7. Implementation notes and performance

- **API.** `icm_batch`, `rank_payout_batch`, `value_rows`, `jam_ev`, `call_ev`, `call_breakeven`, `risk_premium`, `sched_from`, and `set_mc_samples`. Everything is numpy + stdlib with no imports from the engine, so it pastes into `agent.py` as-is (the import-time noise build takes a few ms: 8 ms on the loaded box).
- **Accuracy.** `icm_batch` matches `engine.icm` to < 1e-12, including zero stacks, one-player rows, two-player rows, and payout tuples of length 1–3.
- **Timing.** The box was at load 12–19 on 8 cores during this work (other workers' solvers), so wall-clock was 3–8× inflated. `perf stat` user-space instruction counts per call:
  - `icm_batch` with K=16, n=8: about 0.36 M.
  - `value_rows`, ICM path, K=6: about 0.24 M.
  - `value_rows`, rank MC, K=6, 4 alive: about 1.2 M.
  - `jam_ev`, rank, H=2, 2 callers: about 1.3 M.
  
  At a typical 7–12 G instructions/s on an idle core, that is roughly 30–50 µs for `icm_batch` (target 40), 20–35 µs for the ICM `value_rows`, and 100–170 µs for the rank MC (target 150). The worst case is K=16 × 8 alive on the MC path: about 2× that.
- **MC noise.** With 32 samples, the rank-mode value has about 0.2 pts of jitter per 50 chips (second differences). Common random numbers make row comparisons much steadier than that. For offline analysis, use `set_mc_samples(512)`.
- **Tests.** `test_tvalue.py`:
  - ICM matches the engine.
  - Rank ties.
  - Last hand in both modes.
  - Bust places and co-bust ordering.
  - Monotonicity: exact in ICM and rank at H=0, within MC noise (0.35) for rank at H>0.
  - Horizon limits.
  - `jam_ev` / `call_ev` consistency.
  - ICM risk premium > 0.
  - Load-aware timing.
