#!/usr/bin/env python3
"""Present the frozen public B1 curve and B4 decisions without new estimation.

Only the byte-bound public B1 and B4 snapshots are read. B1 points and its
endpoint bootstrap range are stored fields; B4 entries are counts of recorded
labels, checked against the published scalar witness counts. The original
four-panel figure and its source remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter


SOURCES = {
    "b1_analysis.json": "fc8055490003b29d3cf5afc5a7869e3b1309e29d47010077ea81d61572dd1905",
    "b4_analysis.json": "6f83bb9d3f8b0aedaf71fe26d41b2c2cfd665faacc24fa43658a24b0ff3ad1a3",
}
BUDGETS = [16, 32, 64, 128, 256]
MODELS = ["Qwen3-8B-bf16", "Qwen3-32B-bf16"]
STATES = ["base", "L2", "L3", "L4"]
BLUE = "#244b75"
INK = "#26323d"
GREY = "#757d85"
UNSAFE = "#984b45"


def read_bound(root, filename):
    raw = (root / "data/evidence_v3" / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES[filename]:
        raise ValueError(f"Snapshot digest mismatch: {filename}")
    return json.loads(raw)


def display_counts(b4):
    """Count existing labels only; do not estimate rates or uncertainty."""
    if b4["counts"] != {"SAFE": 33, "UNSAFE": 31, "UNKNOWN": 0}:
        raise ValueError("Unexpected B4 decision counts")
    if len(b4["cells"]) != 64 or any(c["evaluations"] != 4096 for c in b4["cells"]):
        raise ValueError("B4 must contain 64 completed 4096-evaluation targets")
    if any(type(c["witness_count"]) is not int
           or not 0 <= c["witness_count"] <= c["evaluations"]
           or (c["witness_count"] == 0) != (c["outcome"] == "SAFE")
           for c in b4["cells"]):
        raise ValueError("B4 public witness counts must agree with the recorded decisions")
    counts = {}
    for model in MODELS:
        for state in STATES:
            cells = [c for c in b4["cells"] if c["model"] == model and c["state"] == state]
            if len(cells) != 8 or len({c["canary"] for c in cells}) != 8:
                raise ValueError("Each B4 model-state row must have eight distinct canaries")
            if any(c["outcome"] not in {"SAFE", "UNSAFE"} for c in cells):
                raise ValueError("B4 requires completed SAFE/UNSAFE decisions")
            counts[model, state] = tuple(sum(c["outcome"] == verdict for c in cells)
                                         for verdict in ("SAFE", "UNSAFE"))
    if sum(sum(row) for row in counts.values()) != 64:
        raise ValueError("B4 displayed rows must account for all 64 targets")
    if tuple(sum(counts[model, "base"][i] for model in MODELS) for i in (0, 1)) != (16, 0):
        raise ValueError("B4 base rows must contain 16 SAFE and zero UNSAFE targets")
    if tuple(sum(counts[model, state][i] for model in MODELS for state in STATES[1:])
             for i in (0, 1)) != (17, 31):
        raise ValueError("B4 adapted rows must contain 17 SAFE and 31 UNSAFE targets")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    b1 = read_bound(root, "b1_analysis.json")
    b4 = read_bound(root, "b4_analysis.json")
    if b1["budget_grid"] != BUDGETS or [p["q"] for p in b1["curve"]] != BUDGETS:
        raise ValueError("Unexpected B1 budget grid")
    if b1["claim"]["status"] != "available" or b1["n_runs"] != 57 or b1["n_admitted_cells"] != 423:
        raise ValueError("Unexpected B1 population or unavailable endpoint contrast")
    counts = display_counts(b4)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#78818a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7, 2.5))
    ax1 = fig.add_axes([0.09, 0.205, 0.365, 0.58])
    ax4 = fig.add_axes([0.555, 0.125, 0.425, 0.64])
    fig.text(0.09, 0.975, "(a) B1  Expanding search", fontsize=10, va="top")
    fig.text(0.09, 0.897, "57 runs, equally weighted", fontsize=8.5, va="top", color=GREY)
    fig.text(0.555, 0.975, "(b) B4  Complete-domain decisions", fontsize=10, va="top")
    fig.text(0.555, 0.897, "Greedy; fixed order and batches of 8", fontsize=8.2, va="top", color=GREY)
    fig.text(0.555, 0.837, "4,096 prompts/target; 8 canaries per state", fontsize=8.2, va="top", color=GREY)

    rates = [100 * p["fnr"] for p in b1["curve"]]
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_locator(FixedLocator(BUDGETS))
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xlim(BUDGETS[0] / 1.17, BUDGETS[-1] * 1.22)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel("Mean unrecovered canaries (%)", labelpad=5)
    ax1.grid(axis="y", color="#e4e8eb", linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.plot(BUDGETS, rates, "-o", color=BLUE, linewidth=1.65,
             markersize=4.4, markeredgecolor="white", markeredgewidth=0.6)
    ax1.set_xlabel("Distinct target calls", labelpad=4)
    ax1.annotate(f"{rates[0]:.1f}%", (16, rates[0]), xytext=(0, 7),
                 textcoords="offset points", ha="left", fontsize=9.3)
    ax1.annotate(f"{rates[-1]:.1f}%", (256, rates[-1]), xytext=(-1, -13),
                 textcoords="offset points", ha="right", fontsize=9.3)
    d, (lo, hi) = b1["claim"]["D"], b1["claim"]["ci95"]
    ax1.text(0.035, 0.07,
             f"Change {100*d:.1f} pp\n95% paired run-bootstrap range\n[{100*lo:.1f}, {100*hi:.1f}]",
             transform=ax1.transAxes, fontsize=8.2, linespacing=1.25)

    # The separate count matrix avoids equating B4 decisions with B1 rates.
    # Base shading is a categorical distinction, not an estimated quantity.
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.5, 8.8)
    ax4.axis("off")
    for x, label, color in [(0.15, "Model (bf16)", INK), (0.45, "State", INK),
                            (0.69, "SAFE", BLUE), (0.9, "UNSAFE", UNSAFE)]:
        ax4.text(x, 8.4, label, fontsize=8.3, ha="center", va="center", color=color)
    ax4.axhline(7.95, color="#78818a", linewidth=0.65)
    for model_index, model in enumerate(MODELS):
        top = 7.35 - 4.25 * model_index
        ax4.text(0.15, top - 1.35, model.removesuffix("-bf16").replace("Qwen3-", "Qwen3-\n"),
                 fontsize=8.6, ha="center", va="center", linespacing=1.2)
        for state_index, state in enumerate(STATES):
            y = top - 0.9 * state_index
            if state == "base":
                ax4.axhspan(y - 0.41, y + 0.41, xmin=0.32, xmax=1,
                            color="#edf1f5", zorder=0)
            weight = "bold" if state == "base" else "normal"
            ax4.text(0.45, y, state, fontsize=8.5, ha="center", va="center", fontweight=weight)
            for x, count, color in zip((0.69, 0.9), counts[model, state], (BLUE, UNSAFE)):
                ax4.text(x, y, str(count), fontsize=9, ha="center", va="center",
                         fontweight=weight, color=color)
    ax4.axhline(3.75, color="#ccd3d9", linewidth=0.65)
    fig.text(0.555, 0.06, "Counts: 16 base + 48 adapted; 0 UNKNOWN", fontsize=8.2, color=GREY)

    output = args.output or root / "figures/main_evidence_b1_b4_focused.pdf"
    fig.savefig(output, metadata={
        "Title": "B1 budget extension and B4 fixed-domain decisions",
        "Author": "",
        "Subject": "Stored B1 curve and endpoint range; grouped B4 label counts; no re-estimation",
        "Creator": "figures/plot_main_evidence_b1_b4_focused.py",
        "CreationDate": None,
        "ModDate": None,
    })
    if args.png:
        fig.savefig(args.png, dpi=220)
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
