#!/usr/bin/env python3
"""Draw the figures in results/figures/ from the stored result tables.

    python scripts/make_figures.py [--results DIR]

convergence_ladder.png  6x6 support ladder: rank-6 sector values and uniform
                        means at ranks 4-6 versus the support cutoff, with the
                        conditional H(4) envelope on the reported value.
calibration.png         small-lattice residuals A - exact at rank 6 versus the
                        cutoff, with the calibration band +-R.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubbard259 import error  # noqa: E402

def HEX(code):
    return "#" + code


SECTOR_COLORS = {"PP": HEX("2a78d6"), "AP": HEX("eb6834"), "PA": HEX("1baf7a"), "AA": HEX("eda100")}
INK, MUTED, GRID = HEX("0b0b0b"), HEX("52514e"), HEX("e4e3df")
RANK_STYLE = {4: (HEX("9a9893"), ":"), 5: (HEX("6d6b66"), "--"), 6: (INK, "-")}
CUTS = (4, 5, 6)
PNG_META = {"Software": None}
LABEL_DY = {"AP": 6, "PA": -6}  # AP and PA nearly coincide at 6x6


def style(ax):
    ax.set_facecolor(HEX("fcfcfb"))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(CUTS)
    ax.set_xlim(3.8, 6.55)


def ladder_figure(h66, budget, out):
    recs = h66["records"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(HEX("fcfcfb"))
    for b in error.SECTORS:
        ys = [next(r["A"] for r in recs if r["boundary"] == b and r["rank"] == 6 and r["cutoff"] == c) for c in CUTS]
        a1.plot(CUTS, ys, color=SECTOR_COLORS[b], lw=2, marker="o", ms=6, label=b)
        a1.annotate(b, (6, ys[-1]), xytext=(8, LABEL_DY.get(b, 0)), textcoords="offset points", va="center", fontsize=9, color=INK)
    style(a1)
    a1.set_title("6x6, rank 6: boundary sectors", fontsize=10, color=INK, loc="left")
    a1.set_xlabel("connected preparation-support cutoff c", fontsize=9, color=MUTED)
    a1.set_ylabel("A", fontsize=9, color=MUTED)
    a1.legend(frameon=False, fontsize=8, loc="upper left")

    means = h66["uniform_means"]
    for rank in (4, 5, 6):
        ys = [next(m["A_mean"] for m in means if m["rank"] == rank and m["cutoff"] == c) for c in CUTS]
        color, ls = RANK_STYLE[rank]
        a2.plot(CUTS, ys, color=color, ls=ls, lw=2, marker="o", ms=6, label=f"rank {rank}")
    A = budget["estimate"]["A"]
    E = budget["reported_envelope"]
    a2.errorbar([6.25], [A], yerr=[[E], [E]], fmt="s", ms=7, color=INK, capsize=5, lw=2)
    a2.annotate(f"r6c6 mean {A:.10f}\n\u00b1{E:.7f} (H(4) + E_num)", (6.25, A), xytext=(-12, -42),
                textcoords="offset points", ha="right", fontsize=8, color=INK)
    style(a2)
    a2.set_title("6x6: uniform mean over four sectors", fontsize=10, color=INK, loc="left")
    a2.set_xlabel("connected preparation-support cutoff c", fontsize=9, color=MUTED)
    a2.set_ylabel("A (uniform mean)", fontsize=9, color=MUTED)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out, metadata=PNG_META, facecolor=fig.get_facecolor())
    plt.close(fig)


def calibration_figure(small, budget, out):
    R = budget["calibration"]["R"]
    panels = (("4x2", "mixed"), ("4x4", "mixed"), ("8x2", "fp64"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=150, sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(HEX("fcfcfb"))
    for ax, (lat, arith) in zip(axes, panels):
        ax.axhspan(-R, R, color=HEX("d9e6f7"), lw=0, label="\u00b1R")
        ax.axhline(0, color=MUTED, lw=0.8)
        for b in error.SECTORS:
            ys = [next(r["residual"] for r in small["records"] if r["lattice"] == lat and r["boundary"] == b
                       and r["rank"] == 6 and r["cutoff"] == c and r["arithmetic"] == arith) for c in CUTS]
            ax.plot(CUTS, ys, color=SECTOR_COLORS[b], lw=2, marker="o", ms=6, label=b)
        style(ax)
        role = "calibration" if lat in error.CALIBRATION_LATTICES else "validation"
        ax.set_title(f"{lat} ({role}), rank 6", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("support cutoff c", fontsize=9, color=MUTED)
    axes[0].set_ylabel("A - exact", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
    fig.savefig(out, metadata=PNG_META, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=error.results_dir())
    args = ap.parse_args(argv)
    figs = args.results / "figures"
    figs.mkdir(exist_ok=True)
    h66 = error.load_json(args.results / "hierarchy_6x6.json")
    small = error.load_json(args.results / "hierarchy_small.json")
    budget = error.load_json(args.results / "error_budget.json")
    ladder_figure(h66, budget, figs / "convergence_ladder.png")
    calibration_figure(small, budget, figs / "calibration.png")
    print(f"wrote {figs / 'convergence_ladder.png'} and {figs / 'calibration.png'}")


if __name__ == "__main__":
    main()
