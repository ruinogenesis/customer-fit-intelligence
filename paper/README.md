# paper

*Customer Fit Intelligence: A Theory-Constrained Neural Network with Exact
Explanations for Customer Relationships in Food Delivery* (T808, revised after
two outside reviews in T810): 8 pages in an IEEE conference layout, 19
references.

- `main.pdf`: the paper, built from this folder.
- `main.tex`: its source. Build it here with `pdflatex main.tex`, three times so
  the references settle; any TeX Live with the standard packages will do, and
  so will Overleaf (upload the folder as a project).
- `SUPPLEMENT.md`: every setting the page limit leaves out, and the full
  tables behind the paper's numbers.
- `figures.py`: draws the data figures, `fig_*.pdf`, from `notebooks/results`
  (`fig_kappa.pdf` is for the supplement; the paper quotes its interval). From
  the repository root, with `python -m pip install -e "fitcore[lab]"`, run
  `python paper/figures.py`.
- `shot_matrix.png`, `shot_drivers.png`: two dashboard pages on dev, shown in
  the supplement.
- `SOURCES.md`: where every table, figure and number comes from.

When a result file changes, rerun `figures.py`, update the numbers that
`SOURCES.md` ties to that file (in `main.tex` and `SUPPLEMENT.md`), and
rebuild `main.pdf`.
