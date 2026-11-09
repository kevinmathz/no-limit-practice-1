# kevin_v2_simple sources

`kevin_v2_simple/agent.py` is hand-written (not generated). It was built from scratch in about
8 minutes by following `train/claude_prompt_simple.txt`, with no CFR and no code from kevin_v1.
These scripts build its table file and test it.

```
python3 train/kevin_v2_simple_src/build_tables.py 2500   # fresh evaluator check + hu[169,169], eq1, prior, cls52 -> kevin_v2_simple/tables.npz (about 40 s)
python3 train/kevin_v2_simple_src/bench.py $PWD/kevin_v2_simple 10      # true-cost timing (min of 15 per state; p99.8 gate 0.8 ms)
python3 train/kevin_v2_simple_src/paired.py <A> <B> --field v21 v19 ... --tables 1000 --procs 7 --hand-cap 200 --hpl 25
python3 train/kevin_v2_simple_src/debug_run.py 15        # reports exceptions swallowed by the bot's fallback
```

`base/agent.py` is the minute-5 safe baseline from the prompt (kept for paired comparisons).

Results: arena 23.4 points (kevin_v1 21.3, v21 13.1). Paired: +11.7 ± 1.7 vs the baseline,
−0.8 ± 2.0 vs kevin_v1 (a tie), +5.7 ± 1.8 vs v21; +4.4 ± 1.9 vs v21 under the official
200-hand cap with 25 hands per level.
