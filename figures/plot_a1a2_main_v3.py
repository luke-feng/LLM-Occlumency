#!/usr/bin/env python3
"""Compact main-text layout of the published A1/A2 family-size results.

The data-selection and interval-rendering loop is identical to plot_a1a2
in figures/plot_experiments_v3.py. Only layout and titles differ. Every
plotted median, decided fraction and interval is read from the byte-bound
analysis snapshot. No solver, model, raw-record reader or estimator runs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


A1A2_SHA256 = "67c679d2ae3f0cfe20dacf42ee5c8f71fd1ef7db3e1a32b99b102d2f4ed334b3"


def plot_a1a2(root, parser):
    source = root / "data/evidence_v3/a1a2_analysis.json"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != A1A2_SHA256:
        parser.error(f"A1/A2 source SHA-256 mismatch: {source}")
    document = json.loads(raw)
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

    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 2.60))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.175,
                        top=0.81, wspace=0.37)
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
    left.set_title(r"(a) A1, $3\times10^8$ conflict budget", loc="left",
                   fontsize=9.3, pad=6)
    right.set_title(r"(b) A2, $10^6$ conflict budget", loc="left",
                    fontsize=9.3, pad=6)
    for ax in (left, right):
        ax.set_xlim(43, 207)
        ax.set_xticks([50, 100, 150, 200])
        ax.set_xlabel("Number of variables", fontsize=9, labelpad=6)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.005),
               ncol=3, frameon=False, fontsize=9, handlelength=2.0,
               columnspacing=1.5)
    output = root / "figures/a1a2_main_v3.pdf"
    fig.savefig(output, metadata={
        "Title": "A1 decision costs and A2 decided fractions",
        "Author": "",
        "Subject": "Compact presentation of published R1 medians and admitted intervals; no new estimation",
        "Creator": "figures/plot_a1a2_main_v3.py",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
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
    plot_a1a2(root, parser)


if __name__ == "__main__":
    main()
