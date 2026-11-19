# paper

*Customer Fit Intelligence: A Theory-Constrained Neural Network with Exact
Explanations for Customer Relationships in Food Delivery* (T808): 8 pages in an
IEEE conference layout, 15 references.

- `main.pdf`: the paper, built from this folder.
- `main.tex`: its source. Build it here with `pdflatex main.tex`, three times so
  the references settle; any TeX Live with the standard packages will do, and
  so will Overleaf (upload the folder as a project).
- `figures.py`: draws the five data figures, `fig_*.pdf`, from
  `notebooks/results`, and cuts Fig. 8's panels, `fig_dash_*.png`, from the
  screenshots. From the repository root, with
  `python -m pip install -e "fitcore[lab]"`, run `python paper/figures.py`.
- `shot_matrix.png`, `shot_drivers.png`: the two dashboard pages on dev that
  Fig. 8 shows.
- `SOURCES.md`: where every table, figure and number comes from.

When a result file changes, rerun `figures.py`, update the numbers that
`SOURCES.md` ties to that file, and rebuild `main.pdf`.
