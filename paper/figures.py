"""The paper's data figures, drawn from the notebooks' result files (T808).

Run from the repository root with fitcore's lab extra installed:

    python paper/figures.py

It reads notebooks/results and writes fig_*.pdf beside this file: small data
(t803_small_data.csv), one order's contributions to Fit, explained by fitcore
with the T112 model (t112_model.json), planted couriers found in the top
three (t804_drivers.csv), Olist review by days early or late
(t805_review_by_gap.csv) and the profile likelihood of kappa
(t805_kappa_profile.csv). The text is set in TeX Gyre Termes when it is
installed, else in STIX, which comes with matplotlib.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent
RESULTS = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "notebooks" / "results"
)

ACCENT = "#2a6fdb"  # the one accent: what the figure's finding is about
MUTED = "#9a9a92"  # comparisons
INK = "#1f1f1f"
QUIET = "#5f5f5a"
GRID = "#e4e4df"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": QUIET,
        "axes.labelcolor": QUIET,
        "xtick.color": QUIET,
        "ytick.color": QUIET,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
    }
)
WIDTH = 3.45  # inches: one IEEE column


def frame(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def drivers() -> None:
    r = pd.read_csv(RESULTS / "t804_drivers.csv")
    p = r[(r.measure == "precision at 3") & (r.by == "courier")]
    mean = p.pivot_table(index="size", columns="method", values="value")
    orders = r.groupby("size")["orders"].mean()
    sizes = list(mean.index)
    xs = range(len(sizes))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.15))
    frame(ax)
    style = {
        "excess": (ACCENT, "-", "Excess (GET /drivers)"),
        "fit": (MUTED, "-", "Lowest average Fit"),
        "drag": (MUTED, "--", "Raw contribution"),
    }
    for method, (colour, dash, name) in style.items():
        ys = mean[method].to_numpy()
        ax.plot(
            xs,
            ys,
            color=colour,
            linestyle=dash,
            linewidth=1.4,
            marker="o",
            markersize=3.6,
            zorder=3 if method == "excess" else 2,
        )
        ax.annotate(
            name,
            (xs[-1], ys[-1]),
            xytext=(6, -1 if method != "drag" else 3),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=7.5,
        )
    for x, y, fitv in zip(xs, mean["excess"], mean["fit"], strict=True):
        ax.annotate(
            f"{y:.0%}",
            (x, y),
            xytext=(0, 6 if y > fitv else -10),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=7,
        )
    chance = 3 / 60
    ax.axhline(chance, color=QUIET, linestyle=":", linewidth=0.8)
    ax.annotate(
        "Chance (3 of 60)",
        (xs[-1], chance),
        xytext=(6, -4),
        bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"},
        textcoords="offset points",
        va="center",
        color=QUIET,
        fontsize=7.5,
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{s}\n({round(orders[s], -1):,.0f})" for s in sizes])
    ax.set_xlabel("Customers (orders) per population")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlim(-0.25, len(sizes) - 1 + 0.25)
    ax.set_ylabel("Planted couriers in top three")
    fig.subplots_adjust(left=0.13, right=0.70, top=0.97, bottom=0.25)
    fig.savefig(OUT / "fig_drivers.pdf")
    plt.close(fig)


def olist() -> None:
    r = pd.read_csv(RESULTS / "t805_review_by_gap.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    frame(ax)
    ax.axvline(0, color=QUIET, linestyle=":", linewidth=0.8)
    ax.annotate(
        "Promised date",
        (0, 1.0),
        xytext=(3, 2),
        textcoords="offset points",
        color=QUIET,
        fontsize=7.5,
    )
    ax.plot(
        r.gap,
        r.fitted_symmetric,
        color=MUTED,
        linestyle="--",
        linewidth=1.2,
        label="Symmetric gap",
    )
    ax.plot(
        r.gap,
        r.fitted_asymmetric,
        color=MUTED,
        linewidth=1.4,
        label=r"Asymmetric gap, $\kappa$ = 7.0",
    )
    ax.errorbar(
        r.gap,
        r.review,
        yerr=r.ci95,
        fmt="o",
        color=ACCENT,
        markersize=3.4,
        elinewidth=1.0,
        capsize=0,
        label="Observed mean, 95% CI",
        zorder=3,
    )
    ax.set_xlim(-40, 40)
    ax.set_ylim(0.9, 5.05)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_xticklabels(["−30", "−20", "−10", "0", "+10", "+20", "+30"])
    ax.set_xlabel("Days early (+) or late (−) against the promise")
    ax.set_ylabel("Mean review (stars)")
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        frameon=False,
        fontsize=7,
        handlelength=2.2,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.18)
    fig.savefig(OUT / "fig_olist.pdf")
    plt.close(fig)


EXAMPLE = {"x1": -6.0, "x2": 10.0, "x3": -0.01, "x5": 0.5, "x6": -0.5}
NAMES = {
    "x1": "$x_1$ speed: 6 min late",
    "x2": "$x_2$ app ease: 10 s faster",
    "x3": "$x_3$ fee load: 1 point higher",
    "x4": "$x_4$ fidelity: not measured",
    "x5": "$x_5$ courier: +0.5 star",
    "x6": "$x_6$ food: \u22120.5 star",
    "x7": "$x_7$ personalisation: not measured",
    "x8": "$x_8$ recovery: no case",
}
GOOD, BAD = "#3f8f5a", "#c0504d"


def example() -> None:
    from fitcore.model import ModelParams
    from fitcore.scoring import explain, score

    params = ModelParams.from_json((RESULTS / "t112_model.json").read_bytes())
    before, after = score(params, {}).fit, score(params, EXAMPLE).fit
    phi = explain(params, EXAMPLE).contributions["fit"]
    keys = [k for k in sorted(phi, key=lambda k: phi[k]) if k in EXAMPLE]
    keys += [k for k in phi if k not in EXAMPLE]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ys = list(range(len(keys)))[::-1]
    for y, k in zip(ys, keys, strict=True):
        v = phi[k]
        ax.barh(y, v, height=0.6, color=(GOOD if v > 0 else BAD), alpha=0.8)
        if v != 0:
            ax.annotate(
                f"{v:+.3f}".replace("-", "\u2212"),
                (v, y),
                xytext=(3 if v > 0 else -3, 0),
                textcoords="offset points",
                va="center",
                ha="left" if v > 0 else "right",
                color=INK,
                fontsize=7,
            )
    ax.axvline(0, color=QUIET, linewidth=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[k] for k in keys], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.33, 0.13)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1])
    ax.set_xticklabels(["\u22120.3", "\u22120.2", "\u22120.1", "0", "+0.1"])
    ax.set_xlabel("Contribution to Fit")
    ax.annotate(
        f"Fit: {before:.3f} as expected,\n{after:.3f} for this order",
        (-0.32, ys[-2]),
        ha="left",
        va="center",
        color=QUIET,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.47, right=0.98, top=0.98, bottom=0.2)
    fig.savefig(OUT / "fig_example.pdf")
    plt.close(fig)


LR_95 = 3.841458820694124 / 2.0


def kappa_profile() -> None:
    k = pd.read_csv(RESULTS / "t805_kappa_profile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.75), sharey=True)
    panels = (
        ("delivery", "Lateness", (1.0, 13.0)),
        ("fee", "Fee load", (0.0, 1.6)),
    )
    for ax, (kind, title, (lo, hi)) in zip(axes, panels, strict=True):
        frame(ax)
        g = k[k.kind == kind].sort_values("kappa")
        g = g[(g.kappa >= lo) & (g.kappa <= hi)]
        drop = g.train_nll - g.train_nll.min()
        best = float(g.kappa.iloc[int(drop.to_numpy().argmin())])
        inside = g.kappa[drop <= LR_95]
        ax.axvspan(inside.min(), inside.max(), color=ACCENT, alpha=0.12, lw=0)
        ax.axvline(1.0, color=QUIET, linestyle=":", linewidth=0.8)
        ax.axhline(LR_95, color=QUIET, linestyle="--", linewidth=0.6)
        ax.plot(g.kappa, drop, color=ACCENT, linewidth=1.3)
        ax.set_ylim(0, 25)
        ax.set_xlim(lo, hi)
        ax.set_title(
            f"{title}: $\\kappa$ = {best:g} ({inside.min():g} to {inside.max():g})",
            fontsize=7.5,
            color=INK,
        )
        ax.set_xlabel(r"$\kappa$")
    axes[0].set_ylabel("Log-likelihood drop")
    axes[1].annotate(
        "$\\kappa$ = 1",
        (1.0, 22),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=7,
        color=QUIET,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.24, wspace=0.12)
    fig.savefig(OUT / "fig_kappa.pdf")
    plt.close(fig)


def small_data() -> None:
    r = pd.read_csv(RESULTS / "t803_small_data.csv")
    a = r[r.measure.str.startswith("auc")]
    mean = a.pivot_table(
        index=["measure", "customers"], columns="model", values="value"
    )
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.1))
    style = {
        "CFI (15 paths)": (ACCENT, "-", "CFI"),
        "logistic regression": (MUTED, "-", "Logistic regression"),
        "unconstrained network": (MUTED, "--", "Unconstrained network"),
    }
    for ax, (measure, title) in zip(
        axes,
        (("auc churn 30d", "Churn 30 d"), ("auc reorder 14d", "Reorder 14 d")),
        strict=True,
    ):
        frame(ax)
        m = mean.loc[measure]
        for model, (colour, dash, name) in style.items():
            ax.plot(
                m.index,
                m[model],
                color=colour,
                linestyle=dash,
                linewidth=1.4,
                marker="o",
                markersize=3,
                label=name,
                zorder=3 if model.startswith("CFI") else 2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(m.index))
        ax.set_xticklabels([str(c) for c in m.index], fontsize=7)
        ax.minorticks_off()
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xlabel("Training customers")
    axes[0].set_ylabel("Held-out AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        columnspacing=1.2,
        bbox_to_anchor=(0.55, 1.0),
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.8, bottom=0.22, wspace=0.32)
    fig.savefig(OUT / "fig_small_data.pdf")
    plt.close(fig)


if __name__ == "__main__":
    small_data()
    drivers()
    olist()
    example()
    kappa_profile()
    print(f"wrote 5 figures to {OUT}")
