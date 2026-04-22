# final_results

I park **“done” evaluation artifacts** here—the numbers and plots I’d actually put in a report or slide deck.

## What’s inside

- **`eval/eval_raw.csv`** — Per-episode rows from my main eval sweep.
- **`eval/eval_summary.csv`** — Aggregated stats derived from the raw file.
- **`eval/plots/`** — A few bar / comparison PNGs plus a small `README.txt` describing plot order.

## How I use it

After a big `src.evaluate` or `final_eval` run, I copy or symlink outputs here so I’m not hunting through `outputs/`. `src/plot_final_eval.py` can also target `final_report/plots/` for paper figures—that’s parallel to this folder.
