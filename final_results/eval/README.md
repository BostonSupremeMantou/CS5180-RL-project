# final_results/eval

This is literally where I drop the **CSV + plot bundle** for my headline comparison across policies.

## What’s here

- **`eval_raw.csv`** — One row per episode with returns, IoU-ish metrics, cost, consistency, policy name, etc.
- **`eval_summary.csv`** — Means / stderrs I computed from the raw table.
- **`plots/`** — `01_return_comparison.png`, `02_iou_comparison.png`, … plus `plot_order_table.csv` if I need to reorder slides.

## How I use it

I point `src/plot_final_eval.py --input-csv` at `eval_raw.csv` when I want fresher figures under `final_report/plots/`. The PNGs in `plots/` are the “frozen” versions I already liked.
