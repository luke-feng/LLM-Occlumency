#!/usr/bin/env python3
"""Draw the accepted B1--B4 results without estimating new quantities.

Every data input is byte-bound. Curves, points and bootstrap ranges are
read from published fields. B4 groups existing cell decisions by model
and state, without estimating rates or uncertainty.
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
    "b2_analysis.json": "f84d58f91f208459661ab6c328f96932e9d8c30f4e55ca2556cb8e74d5d68f16",
    "b3_analysis.json": "40c16b9b5fa8b270dc8138f03ded75b30f9d2eca08a4ce1604299c290791bb00",
    "b4_analysis.json": "6f83bb9d3f8b0aedaf71fe26d41b2c2cfd665faacc24fa43658a24b0ff3ad1a3",
}
BLUE = "#244b75"
INK = "#26323d"
GREY = "#757d85"
UNSAFE = "#984b45"
BUDGETS = [16, 32, 64, 128, 256]
MODELS = ["Qwen3-8B-bf16", "Qwen3-32B-bf16"]
STATES = ["base", "L2", "L3", "L4"]


def read_bound(root, filename):
    raw = (root / "data/evidence_v3" / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES[filename]:
        raise ValueError(f"Snapshot digest mismatch: {filename}")
    return json.loads(raw)


def budget_axis(ax, budgets, ylabel):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(budgets))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(budgets[0] / 1.17, budgets[-1] * 1.22)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel, labelpad=5)
    ax.grid(axis="y", color="#e4e8eb", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    b1, b2, b3, b4 = [
        read_bound(root, name) for name in SOURCES
    ]
    if [p["q"] for p in b1["curve"]] != BUDGETS:
        raise ValueError("Unexpected B1 budget grid")
    if [p["q"] for p in b3["prefix"]] != BUDGETS[:-1]:
        raise ValueError("Unexpected B3 budget grid")
    if b1["claim"]["status"] != "available" or b3["claim"]["status"] != "available":
        raise ValueError("A primary result is unavailable")
    if not b2["contrast"]["paired"]:
        raise ValueError("B2 contrast must be the published paired contrast")
    if b3["prefix"][-1]["R"] != b3["claim"]["R"] or b3["prefix"][-1]["ci95"] != b3["claim"]["ci95"]:
        raise ValueError("B3 endpoint does not match its primary result")
    if b4["counts"] != {"SAFE": 33, "UNSAFE": 31, "UNKNOWN": 0}:
        raise ValueError("Unexpected B4 decision counts")
    if len(b4["cells"]) != 64 or any(c["evaluations"] != 4096 for c in b4["cells"]):
        raise ValueError("B4 must contain 64 completed 4096-evaluation cells")
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
                raise ValueError("Each B4 model-state row must contain eight distinct canaries")
            if any(c["outcome"] not in {"SAFE", "UNSAFE"} for c in cells):
                raise ValueError("B4 display requires complete SAFE/UNSAFE decisions")
            counts[model, state] = tuple(sum(c["outcome"] == outcome for c in cells)
                                         for outcome in ("SAFE", "UNSAFE"))
    if sum(sum(row) for row in counts.values()) != 64:
        raise ValueError("B4 displayed rows must account for all 64 cells")
    if tuple(sum(counts[model, "base"][i] for model in MODELS) for i in (0, 1)) != (16, 0):
        raise ValueError("B4 base rows must contain 16 SAFE and zero UNSAFE cells")
    if tuple(sum(counts[model, state][i] for model in MODELS for state in STATES[1:])
             for i in (0, 1)) != (17, 31):
        raise ValueError("B4 adapted rows must contain 17 SAFE and 31 UNSAFE cells")

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#78818a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7, 4.55))
    ax1 = fig.add_axes([0.09, 0.615, 0.36, 0.26])
    ax2 = fig.add_axes([0.65, 0.615, 0.325, 0.26])
    ax3 = fig.add_axes([0.09, 0.145, 0.36, 0.26])
    ax4 = fig.add_axes([0.555, 0.095, 0.42, 0.315])

    for x, y, title, subtitle in [
        (0.09, 0.975, "(a) B1  Expanding search", "57 runs / run-equal"),
        (0.555, 0.975, "(b) B2  Starting-template families", "128 issued calls / track-equal"),
        (0.09, 0.505, "(c) B3  Alternative proposer", "Known-secret feedback / track-equal"),
        (0.555, 0.505, "(d) B4  Complete-domain decisions",
         "4,096 prompts/cell / 8 canaries per state"),
    ]:
        fig.text(x, y, title, fontsize=10, va="top")
        fig.text(x, y - 0.048, subtitle, fontsize=8.8, va="top", color=GREY)

    rates = [100 * p["fnr"] for p in b1["curve"]]
    budget_axis(ax1, BUDGETS, "Unrecovered canaries (%)")
    ax1.plot(BUDGETS, rates, "-o", color=BLUE, linewidth=1.65,
             markersize=4.4, markeredgecolor="white", markeredgewidth=0.6)
    ax1.set_xlabel("Distinct target calls", labelpad=4)
    ax1.annotate(f"{rates[0]:.1f}%", (16, rates[0]), xytext=(0, 7),
                 textcoords="offset points", ha="left", fontsize=9.6)
    ax1.annotate(f"{rates[-1]:.1f}%", (256, rates[-1]), xytext=(-1, -13),
                 textcoords="offset points", ha="right", fontsize=9.6)
    d, (lo, hi) = b1["claim"]["D"], b1["claim"]["ci95"]
    ax1.text(0.035, 0.08, f"Change {100*d:.1f} pp\n95% bootstrap range\n[{100*lo:.1f}, {100*hi:.1f}]",
             transform=ax1.transAxes, fontsize=8.5, linespacing=1.25)

    for y, key, color, marker in [(1, "primary", BLUE, "o"), (0, "comparator", GREY, "D")]:
        point = b2[key]["point"] * 100
        lo, hi = [v * 100 for v in b2[key]["ci95"]]
        ax2.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt=marker,
                     color=color, capsize=3, elinewidth=1.3, markersize=5)
        ax2.annotate(f"{point:.1f}%", (point, y), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=9.6)
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_ylim(-1.30, 1.8)
    ax2.set_yticks([1, 0], ["Clean\n(screened starts)", "Legacy\n(hint-containing)"])
    ax2.tick_params(axis="y", labelsize=8.3)
    ax2.tick_params(axis="y", length=0, pad=6)
    ax2.spines["left"].set_visible(False)
    ax2.grid(axis="x", color="#e4e8eb", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("Recovered canaries (%)", labelpad=4)
    delta = b2["contrast"]["point"]
    lo, hi = b2["contrast"]["ci95"]
    ax2.text(0.04, 0.025, f"Legacy minus clean {100*delta:.1f} pp\n95% bootstrap range [{100*lo:.1f}, {100*hi:.1f}]",
             transform=ax2.transAxes, fontsize=8.0, linespacing=1.3)

    prefix = b3["prefix"]
    q3 = [p["q"] for p in prefix]
    r3 = [100 * p["R"] for p in prefix]
    err3 = [[100 * (p["R"] - p["ci95"][0]) for p in prefix],
            [100 * (p["ci95"][1] - p["R"]) for p in prefix]]
    budget_axis(ax3, q3, "Recovered canaries (%)")
    ax3.set_ylim(0, 60)
    ax3.set_yticks([0, 20, 40, 60])
    ax3.errorbar(q3, r3, yerr=err3, fmt="-o", color=BLUE,
                 ecolor="#8299b0", elinewidth=1, capsize=3,
                 linewidth=1.65, markersize=4.4,
                 markeredgecolor="white", markeredgewidth=0.6)
    ax3.set_xlabel("Unique target calls", labelpad=4)
    ax3.text(0.04, 0.84, f"128 calls  {100*b3['claim']['R']:.1f}%",
             transform=ax3.transAxes, fontsize=9.6)
    ax3.text(0.04, 0.69, "95% bootstrap ranges", transform=ax3.transAxes,
             fontsize=8.5, color=GREY)

    # A count matrix keeps the complete-domain unit distinct from the recovery
    # percentages in (a)--(c). Shading marks base rows, not an estimated quantity.
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.5, 8.8)
    ax4.axis("off")
    for x, label, color in [(0.15, "Model", INK), (0.45, "State", INK),
                            (0.69, "SAFE", BLUE), (0.9, "UNSAFE", UNSAFE)]:
        ax4.text(x, 8.4, label, fontsize=8.5, ha="center", va="center", color=color)
    ax4.axhline(7.95, color="#78818a", linewidth=0.65)
    for model_index, model in enumerate(MODELS):
        top = 7.35 - 4.25 * model_index
        ax4.text(0.15, top - 1.35, model.removesuffix("-bf16").replace("Qwen3-", "Qwen3-\n"),
                 fontsize=8.8, ha="center", va="center", linespacing=1.25)
        for state_index, state in enumerate(STATES):
            y = top - 0.9 * state_index
            if state == "base":
                ax4.axhspan(y - 0.41, y + 0.41, xmin=0.32, xmax=1,
                            color="#edf1f5", zorder=0)
            weight = "bold" if state == "base" else "normal"
            ax4.text(0.45, y, state, fontsize=8.5, ha="center", va="center", fontweight=weight)
            for x, count, color in zip((0.69, 0.9), counts[model, state], (BLUE, UNSAFE)):
                ax4.text(x, y, str(count), fontsize=9.2, ha="center", va="center",
                         fontweight=weight, color=color)
    ax4.axhline(3.75, color="#ccd3d9", linewidth=0.65)
    ax4.text(0.5, -0.42, "Cell counts", fontsize=8.5, ha="center", va="top", color=GREY)

    output = args.output or root / "figures/main_evidence_b1_b4_v3.pdf"
    fig.savefig(output, metadata={
        "Title": "B1-B4 budgeted discovery, protocol sensitivity and fixed-domain decisions",
        "Author": "",
        "Subject": "Published curves and bootstrap ranges, grouped complete-domain cell counts; no re-estimation",
        "Creator": "figures/plot_main_evidence_b1_b4_v3.py",
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
