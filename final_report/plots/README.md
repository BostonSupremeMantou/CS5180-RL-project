# final_report/plots

I stash **publication-ready figures** here—mostly matplotlib exports I `\includegraphics` from `final_report.tex`.

## What’s inside

- **`01_*.png` … `11_*.png`** — Numbered charts (learning curves, bar comparisons, Pareto scatters, etc.).
- **`README_ordered.md`** — The order I actually talk about them in the write-up.
- **`README.txt`**, **`plot_order_table.csv`** — Quick notes / machine-readable ordering if I re-slideshow.

## How I use it

When I update eval CSVs, I rerun `src/plot_final_eval.py` with `--out-dir final_report/plots` and then skim this folder before recompiling the PDF.
