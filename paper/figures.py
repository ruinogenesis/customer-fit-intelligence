"""The paper's data figures, drawn from the notebooks' result files (T808).

Run from the repository root with fitcore's lab extra installed:

    python paper/figures.py

It reads notebooks/results and writes fig_*.pdf beside this file: small data
(t803_small_data.csv), one order's contributions to Fit, explained by fitcore
with the T112 model (t112_model.json), planted couriers found in the top
three, with standard errors over 20 populations (t804_drivers.csv), Olist
review by days early or late, all reviews and those written before and after
delivery (t805_review_by_gap.csv), and the profile likelihood of kappa
(t805_kappa_profile.csv), with its 95% interval between grid points, which
the paper quotes (the figure itself is for the supplement since T810). The
text is set in TeX Gyre Termes when it is installed, else in STIX, which
comes with matplotlib.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    couriers = r[r.by == "courier"]
    p = couriers[couriers.measure == "precision at 3"]
    per_population = {
        method: p[p.method == method].pivot_table(
            index="size", columns="seed", values="value"
        )
        for method in ("excess", "fit", "drag")
    }
    # A stricter hit (T810): in the top three and named for courier conduct.
    keys = ["size", "seed", "group"]
    excess = couriers[couriers.method == "excess"]
    rank = excess[excess.measure == "rank"].set_index(keys)["value"]
    first = excess[excess.measure == "right first driver"].set_index(keys)["value"]
    hit = ((rank <= 3) & (first.reindex(rank.index) == 1)).astype(float)
    per_population["strict"] = hit.groupby(["size", "seed"]).mean().unstack("seed")
    orders = r.groupby("size")["orders"].mean()
    sizes = list(per_population["excess"].index)
    xs = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    frame(ax)
    style = {
        "excess": (ACCENT, "-", "o", "Excess (GET /drivers)", -0.06),
        "strict": (ACCENT, ":", "o", "Excess, naming courier conduct", 0.06),
        "fit": (MUTED, "-", "s", "Lowest average Fit", 0.0),
        "drag": (MUTED, "--", "^", "Raw contribution", 0.0),
    }
    for method, (colour, dash, marker, name, shift) in style.items():
        values = per_population[method]
        mean = values.mean(axis=1).to_numpy()
        se = (values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])).to_numpy()
        ax.errorbar(
            xs + shift,
            mean,
            yerr=se,
            color=colour,
            linestyle=dash,
            linewidth=1.3,
            marker=marker,
            markersize=3.4,
            markerfacecolor="white" if method == "strict" else colour,
            elinewidth=0.8,
            capsize=0,
            label=name,
            zorder=3 if colour == ACCENT else 2,
        )
    chance = 3 / 60
    ax.axhline(
        chance, color=QUIET, linestyle=":", linewidth=0.8, label="Chance (3 of 60)"
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{s}\n({round(orders[s], -1):,.0f})" for s in sizes])
    ax.set_xlabel("Customers (orders) per population")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlim(-0.3, len(sizes) - 1 + 0.3)
    ax.set_ylabel("Planted couriers in top three")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in labels if name != "Chance (3 of 60)"]
    order.append(labels.index("Chance (3 of 60)"))
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=2.4,
        columnspacing=1.0,
        labelspacing=0.25,
        bbox_to_anchor=(0.55, 1.0),
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.79, bottom=0.235)
    fig.savefig(OUT / "fig_drivers.pdf")
    plt.close(fig)


def olist() -> None:
    r = pd.read_csv(RESULTS / "t805_review_by_gap.csv")
    before = r.assign(n=(r.before_share * r.orders).round())
    before = before[(before.n >= 30) & (before.gap < 0)]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    frame(ax)
    ax.axvline(0, color=QUIET, linestyle=":", linewidth=0.8)
    ax.annotate(
        "Promised date",
        (0, 4.95),
        xytext=(3, 0),
        textcoords="offset points",
        va="top",
        color=QUIET,
        fontsize=7.5,
    )
    # Fitted means are drawn at each bin's mean gap only: a line between
    # bins would invent a shape between them.
    ax.plot(
        r.gap,
        r.fitted_asymmetric,
        linestyle="none",
        marker="_",
        markersize=8,
        markeredgewidth=1.4,
        color=MUTED,
        label=r"Fitted, asymmetric gap ($\kappa$ = 7.0)",
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
        label="All reviews",
        zorder=3,
    )
    ax.errorbar(
        r.gap + 0.9,
        r.review_after,
        yerr=r.ci95_after,
        fmt="o",
        color=INK,
        markerfacecolor="white",
        markersize=3.4,
        elinewidth=0.8,
        capsize=0,
        label="Written after delivery",
        zorder=4,
    )
    ax.plot(
        before.gap - 0.9,
        before.review_before,
        linestyle="none",
        marker="v",
        markersize=3.4,
        color=MUTED,
        label="Written before delivery",
        zorder=2,
    )
    ax.set_xlim(-40, 40)
    ax.set_ylim(0.9, 5.05)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_xticklabels(["\u221230", "\u221220", "\u221210", "0", "+10", "+20", "+30"])
    ax.set_xlabel("Days early (+) or late (\u2212) against the promise")
    ax.set_ylabel("Mean review (stars)")
    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index(name)
        for name in (
            "All reviews",
            "Written after delivery",
            "Written before delivery",
            r"Fitted, asymmetric gap ($\kappa$ = 7.0)",
        )
    ]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.18)
    fig.savefig(OUT / "fig_olist.pdf")
    plt.close(fig)


EXAMPLE = {"x1": -6.0, "x2": 10.0, "x3": -0.01, "x5": 0.5, "x6": -0.5}
NAMES = {
    "x1": "$x_1$ speed: 6 min late",
    "x2": "$x_2$ app ease: 10 s faster",
    "x3": "$x_3$ fee load: +1% of basket",
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
    fig, ax = plt.subplots(figsize=(WIDTH, 1.85))
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
    ax.set_xlim(-0.40, 0.14)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1])
    ax.set_xticklabels(["\u22120.3", "\u22120.2", "\u22120.1", "0", "+0.1"])
    ax.set_xlabel("Contribution to Fit")
    ax.annotate(
        f"Fit: {before:.3f} as expected,\n{after:.3f} for this order",
        (-0.39, ys[-2]),
        ha="left",
        va="center",
        color=QUIET,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.47, right=0.98, top=0.98, bottom=0.2)
    fig.savefig(OUT / "fig_example.pdf")
    plt.close(fig)


LR_95 = 3.841458820694124 / 2.0


def lr_interval(kappa: pd.Series, drop: pd.Series) -> tuple[float, float]:
    """Where the log-likelihood drop crosses LR_95 on each side of the best kappa,
    by linear interpolation between the profiled grid points (T809)."""
    x, y = kappa.to_numpy(), drop.to_numpy()
    best = int(y.argmin())
    left = next(i for i in range(best, 0, -1) if y[i - 1] > LR_95)
    right = next(i for i in range(best, len(y) - 1) if y[i + 1] > LR_95)
    low = x[left - 1] + (x[left] - x[left - 1]) * (y[left - 1] - LR_95) / (
        y[left - 1] - y[left]
    )
    high = x[right] + (x[right + 1] - x[right]) * (LR_95 - y[right]) / (
        y[right + 1] - y[right]
    )
    return float(low), float(high)


def kappa_profile() -> None:
    k = pd.read_csv(RESULTS / "t805_kappa_profile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.75), sharey=True)
    panels = (
        ("delivery", "Lateness", (0.0, 13.0), ".1f"),
        ("fee", "Fee load", (0.0, 1.6), ".2f"),
    )
    for ax, (kind, title, (lo, hi), f) in zip(axes, panels, strict=True):
        frame(ax)
        g = k[k.kind == kind].sort_values("kappa")
        g = g[(g.kappa >= lo) & (g.kappa <= hi)]
        drop = g.train_nll - g.train_nll.min()
        best = float(g.kappa.iloc[int(drop.to_numpy().argmin())])
        low, high = lr_interval(g.kappa, drop)
        print(f"{kind}: kappa {best:.2f}, 95% interval {low:.3f} to {high:.3f}")
        ax.axvspan(low, high, color=ACCENT, alpha=0.12, lw=0)
        ax.axvline(1.0, color=QUIET, linestyle=":", linewidth=0.8)
        ax.axhline(LR_95, color=QUIET, linestyle="--", linewidth=0.6)
        ax.plot(g.kappa, drop, color=ACCENT, linewidth=1.3)
        ax.set_ylim(0, 25)
        ax.set_xlim(lo, hi)
        ax.set_title(
            f"{title}: $\\kappa$ = {best:{f}} ({low:{f}} to {high:{f}})",
            fontsize=7.5,
            color=INK,
        )
        ax.set_xlabel(r"$\kappa$")
    axes[0].set_ylabel("Log-likelihood drop")
    for ax in axes:
        ax.annotate(
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
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.15))
    style = {
        "CFI (15 paths)": (ACCENT, "-", "o", "white", "CFI"),
        "CFI, outcome labels only": (
            ACCENT,
            "--",
            "o",
            "white",
            "CFI, outcomes only",
        ),
        "logistic regression": (INK, "-", "s", INK, "LR"),
        "logistic regression, surplus and shortfall": (
            INK,
            "--",
            "s",
            "white",
            "LR, surplus/shortfall",
        ),
        "unconstrained network": (MUTED, "-", "^", MUTED, "Network"),
    }
    for ax, (measure, title) in zip(
        axes,
        (("auc churn 30d", "Churn 30 d"), ("auc reorder 14d", "Reorder 14 d")),
        strict=True,
    ):
        frame(ax)
        m = mean.loc[measure]
        for model, (colour, dash, marker, face, name) in style.items():
            ax.plot(
                m.index,
                m[model],
                color=colour,
                linestyle=dash,
                linewidth=1.2,
                marker=marker,
                markersize=3.4,
                markerfacecolor=colour if model == "CFI (15 paths)" else face,
                label=name,
                zorder=3 if model.startswith("CFI") else 2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(m.index))
        ax.set_xticklabels([f"{c:,}" for c in m.index], fontsize=7)
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
        handlelength=2.2,
        columnspacing=1.0,
        bbox_to_anchor=(0.54, 1.0),
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.73, bottom=0.21, wspace=0.32)
    fig.savefig(OUT / "fig_small_data.pdf")
    plt.close(fig)


if __name__ == "__main__":
    small_data()
    drivers()
    olist()
    example()
    kappa_profile()
    print(f"wrote 6 figures to {OUT}")
