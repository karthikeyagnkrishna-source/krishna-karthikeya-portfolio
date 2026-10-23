# Overleaf source package

Main file: `main.tex`

## Compile

Upload the entire folder (or the provided ZIP) to Overleaf, set `main.tex` as the main document, and compile with pdfLaTeX. The bibliography uses BibTeX; Overleaf runs it automatically.

Local equivalent:

```bash
latexmk -pdf main.tex
```

## Structure

- `main.tex` - document style, title page, and section order
- `sections/problem.tex` - research question, novelty, and system overview
- `sections/physics.tex` - contact law, single-leg identification, and four-leg coupling
- `sections/learning.tex` - cross-modal architecture, uncertainty, losses, and training
- `sections/twin_planning.tex` - registered local twin, online update, and physics-aware MPPI
- `sections/global_mapping.tex` - persistent world map, A*/D* routing, U-shape benchmark, and why time fields are optional
- `sections/evaluation.tex` - experiments, metrics, baselines, ablations, and gates
- `sections/execution.tex` - 26-week plan, resources, risks, and decision case
- `sections/appendix.tex` - algorithms, tensor schema, dataset format, and mentor checklist
- `references.bib` - bibliography

The report contains no external figures; all architecture diagrams are generated directly in LaTeX with TikZ.
