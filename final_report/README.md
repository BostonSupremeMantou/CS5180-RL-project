# final_report

I keep my **AAAI-style paper source** and the **figure assets** that go with it in this directory.

## What’s inside

- **`final_report.tex`** — Main LaTeX entry; pulls in `aaai2026.sty` / `aaai2026.bst` next to it.
- **`plots/`** — Curated PNGs for the paper (see `plots/README_ordered.md` for the narrative order I use).
- **`drafts/`** — Scratch Markdown / checklists while I’m writing (some paths may be gitignored).
- **`references/`** — PDFs, HTML mirrors, and notes I collected for citations—not all of it ships with the paper.

## How I build it

```bash
cd final_report
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
```

I regenerate plots from repo root with `python src/plot_final_eval.py ...` when the underlying CSV changes.
