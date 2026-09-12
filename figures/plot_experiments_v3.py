#!/usr/bin/env python3
"""Present published B1, B2 and A1/A2 values without new estimation.

Input is the hash-bound public projection of accepted numerical snapshots.
The accepted interval belongs to the endpoint contrast, not to curve points;
this presentation therefore deliberately draws no confidence band.
B2 uses only secondary.per_track from the bound b2_analysis.json snapshot,
including its published marginal and paired-contrast intervals.
A1/A2 uses only cells from the bound a1a2_analysis.json snapshot. Its A1
points are R1-certified latent medians and its bars are admissible published
intervals; A2 uses the published decided fractions and Wilson intervals.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from analysis.check_package import input_snapshot

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter


BUDGETS = [16, 32, 64, 128, 256]
B2_LABELS = {
    "Qwen3-8B-bf16": "Qwen3-8B bf16",
    "Qwen3-8B-4bit": "Qwen3-8B 4-bit",
    "Qwen3-14B-bf16": "Qwen3-14B bf16",
    "Qwen3-14B-4bit": "Qwen3-14B 4-bit",
    "Qwen3-32B-bf16": "Qwen3-32B bf16",
    "Qwen3-30B-A3B-bf16": "Qwen3-30B-A3B bf16",
    "Qwen3-235B-A22B-4bit": "Qwen3-235B-A22B 4-bit",
    "gemma-3-12b-it-bf16": "Gemma-3-12B-it bf16",
    "glm-4-9b-chat-hf-bf16": "GLM-4-9B-chat bf16",
    "glm-4-9b-chat-hf-4bit": "GLM-4-9B-chat 4-bit",
}


def plot_a1a2(root, parser):
    document = input_snapshot(root,"a1a2_analysis.json")
    cells = document["cells"]
    expected = {
        "random_3sat": [50, 100, 150, 200],
        "pigeonhole": [7, 9, 11, 12],
        "tseitin": [40, 60, 80, 100],
    }
    if len(cells) != 12 or {(row["family"], row["size"]) for row in cells} != {
        (family, size) for family, sizes in expected.items() for size in sizes
    }:
        parser.error("A1/A2 presentation requires the twelve frozen cells")

    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 3.05))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18,
                        top=0.765, wspace=0.37)
    for family, label, color, marker, line_style in (
        ("random_3sat", "Random 3-CNF", "#244b75", "o", "-"),
        ("pigeonhole", "Pigeonhole", "#bc5b40", "s", "--"),
        ("tseitin", "Tseitin", "#48796d", "^", "-."),
    ):
        rows = sorted((row for row in cells if row["family"] == family),
                      key=lambda row: row["n_vars"])
        x = [row["n_vars"] for row in rows]
        medians = [row["a1"]["median"]
                   if row["a1"]["certified_under_r1"] is True
                   and row["a1"]["median"] is not None else float("nan")
                   for row in rows]
        fractions = [row["a2"]["fraction"] for row in rows]
        style = dict(color=color, marker=marker, linestyle=line_style,
                     linewidth=1.2, markersize=4.8, markeredgewidth=0.9,
                     markerfacecolor="white", label=label)
        left.plot(x, medians, **style)
        right.plot(x, fractions, **style)
        if family == "pigeonhole":
            # Deterministic singleton formulas have no sampling intervals.
            continue
        for row in rows:
            a1, a2 = row["a1"], row["a2"]
            if (a1["certified_under_r1"] is True
                    and a1["median"] is not None
                    and a1["interval_admissible"] is True
                    and a1["ci"] is not None):
                lo, hi = a1["ci"]
                # Differences only render the stored interval endpoints.
                left.errorbar(row["n_vars"], a1["median"],
                              yerr=[[a1["median"] - lo], [hi - a1["median"]]],
                              fmt="none", color=color, capsize=2.2,
                              elinewidth=0.9, capthick=0.9)
            if a2["wilson"] is not None:
                lo, hi = a2["wilson"]
                right.errorbar(row["n_vars"], a2["fraction"],
                               yerr=[[a2["fraction"] - lo],
                                     [hi - a2["fraction"]]],
                               fmt="none", color=color, capsize=2.2,
                               elinewidth=0.9, capthick=0.9)

    left.set_yscale("log")
    left.set_ylim(20, 6e7)
    left.set_ylabel("All-instance median conflicts", fontsize=9)
    right.set_ylim(-0.055, 1.065)
    right.set_ylabel("Decided fraction", fontsize=9)
    right.set_yticks([0, 0.25, 0.5, 0.75, 1],
                     ["0", "0.25", "0.50", "0.75", "1.00"])
    left.set_title(r"(a) A1: R1-certified, $3\times10^8$ budget", loc="left",
                   fontsize=9.3, pad=8)
    right.set_title(r"(b) A2: $10^6$ conflict budget", loc="left",
                    fontsize=9.3, pad=8)
    for ax in (left, right):
        ax.set_xlim(43, 207)
        ax.set_xticks([50, 100, 150, 200])
        ax.set_xlabel("Number of variables", fontsize=9, labelpad=6)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.0),
               ncol=3, frameon=False, fontsize=9, handlelength=2.3,
               columnspacing=2.0)
    output = root / "figures/a1a2_censoring_v3.pdf"
    fig.savefig(output, metadata={
        "Title": "A1 certified latent medians and A2 decided fractions",
        "Author": "",
        "Subject": "Presentation of published R1-certified medians and admissible intervals; no new estimation",
        "Creator": "figures/plot_experiments_v3.py",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


def plot_b2(root, parser):
    published_rows = input_snapshot(root,"b2_analysis.json")["secondary"]["per_track"]
    by_track = {row["track"]: row for row in published_rows}
    if len(published_rows) != 10 or set(by_track) != set(B2_LABELS):
        parser.error("B2 presentation requires the ten frozen tracks")
    rows = [by_track[track] for track in B2_LABELS]
    positions = list(range(len(rows)))

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7, 3.7), sharey=True,
        gridspec_kw={"width_ratios": [1, 0.94]},
    )
    fig.subplots_adjust(left=0.26, right=0.985, bottom=0.15,
                        top=0.84, wspace=0.18)
    for family, offset, color, label in (
        ("clean", -0.12, "#244b75", "Clean"),
        ("legacy", 0.12, "#bc5b40", "Legacy"),
    ):
        points = [row[family] for row in rows]
        # These distances merely draw the stored interval endpoints.
        errors = [[row[family] - row[f"{family}_ci95"][0] for row in rows],
                  [row[f"{family}_ci95"][1] - row[family] for row in rows]]
        left.errorbar(points, [p + offset for p in positions], xerr=errors,
                      fmt="o", markersize=3.8, capsize=2, elinewidth=0.9,
                      color=color, label=label)

    differences = [row["delta"] for row in rows]
    difference_errors = [[row["delta"] - row["delta_ci95"][0] for row in rows],
                         [row["delta_ci95"][1] - row["delta"] for row in rows]]
    right.errorbar(differences, positions, xerr=difference_errors,
                   fmt="o", markersize=3.8, capsize=2, elinewidth=0.9,
                   color="#454545")
    right.axvline(0, color="#8d8d8d", linestyle="--", linewidth=0.75)
    left.set_yticks(positions, list(B2_LABELS.values()), fontsize=8.7)
    left.set_ylim(9.65, -0.65)
    left.set_xlim(-0.035, 1.035)
    left.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
    right.set_xlim(-0.30, 0.80)
    right.set_xticks([-0.25, 0, 0.25, 0.5, 0.75], ["-0.25", "0", "0.25", "0.50", "0.75"])
    left.set_xlabel("Recovery rate", fontsize=9)
    right.set_xlabel("Paired difference", fontsize=9)
    left.set_title("(a) Clean and legacy", fontsize=9.5, loc="left", pad=23)
    right.set_title("(b) Legacy minus clean", fontsize=9.5, loc="left", pad=23)
    left.legend(loc="lower left", bbox_to_anchor=(0.24, 1.015), ncol=2,
                frameon=False, fontsize=8, handlelength=1.3,
                columnspacing=1.0, borderaxespad=0)
    for ax in (left, right):
        ax.tick_params(axis="x", labelsize=8.3)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
        ax.set_axisbelow(True)
    right.tick_params(axis="y", labelleft=False)
    output = root / "figures/b2_tracks_v3.pdf"
    fig.savefig(output, metadata={
        "Title": "B2 published per-track recovery and paired differences",
        "Author": "",
        "Subject": "Presentation of stored per-track points and intervals; no new aggregation",
        "Creator": "figures/plot_experiments_v3.py",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--a1a2-only", action="store_true",
                        help="Render A1/A2 without rewriting the B1/B2 figures")
    args = parser.parse_args()
    root = args.root.resolve()
    curve = input_snapshot(root,"b1_analysis.json")["curve"]
    budgets = [point["q"] for point in curve]
    rates = [point["fnr"] for point in curve]
    if budgets != BUDGETS:
        parser.error("B1 curve must contain exactly the five frozen budgets")

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    if args.a1a2_only:
        plot_a1a2(root, parser)
        return
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.25, top=0.94)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(BUDGETS))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(13.5, 303)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.plot(budgets, rates, color="#244b75", linewidth=1.7,
            marker="o", markersize=5.2, markeredgewidth=0.8,
            markeredgecolor="white")
    ax.set_xlabel("Distinct target calls per canary and model state", labelpad=7)
    ax.set_ylabel("Found-nothing rate", labelpad=7)

    output = root / "figures/b1_nested_budget_v3.pdf"
    fig.savefig(output, metadata={
        "Title": "B1 nested-budget found-nothing rate",
        "Author": "",
        "Subject": "Presentation of the five published B1 curve points; no new aggregation",
        "Creator": "figures/plot_experiments_v3.py",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")
    plot_b2(root, parser)
    plot_a1a2(root, parser)


if __name__ == "__main__":
    main()
