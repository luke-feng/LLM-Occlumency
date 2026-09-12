#!/usr/bin/env python3
"""Present the published B1 trajectory and B2 track estimates together.

This reads three hash-bound public numerical projections. It performs no estimation,
resampling, model calls, or experiment-repository imports. B1 has no
pointwise intervals. B2 track rows are descriptive point pairs, while the
separated overall row uses the stored primary and comparator intervals.
The two contrast annotations use their own published paired intervals.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter


BUDGETS = [16, 32, 64, 128, 256]
TRACK_LABELS = {
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
CLEAN = "#244b75"
LEGACY = "#bc5b40"


def load_snapshot(root, name, parser):
    return input_snapshot(root,name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    b1 = load_snapshot(root, "b1_analysis.json", parser)
    b2 = load_snapshot(root, "b2_analysis.json", parser)
    decomposition = load_snapshot(root, "b1b2_decomposition.json", parser)
    if [point["q"] for point in b1["curve"]] != BUDGETS:
        parser.error("B1 must contain exactly the five published budget points")
    rows = b2["secondary"]["per_track"]
    by_track = {row["track"]: row for row in rows}
    if len(rows) != 10 or set(by_track) != set(TRACK_LABELS):
        parser.error("B2 must contain exactly the ten published tracks")
    if b1["claim"]["status"] != "available" or b2["contrast"]["paired"] is not True:
        parser.error("Published paired contrasts must be available")

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7, 3.35))
    # The right panel includes its full precision-qualified track labels.
    left = fig.add_axes([0.085, 0.30, 0.285, 0.535])
    right = fig.add_axes([0.705, 0.30, 0.277, 0.535])

    rates = [point["fnr"] for point in b1["curve"]]
    left.set_xscale("log", base=2)
    left.xaxis.set_major_locator(FixedLocator(BUDGETS))
    left.xaxis.set_major_formatter(ScalarFormatter())
    left.xaxis.set_minor_locator(NullLocator())
    track_rows = decomposition["b1"]["tracks"]
    if len(track_rows) != 9 or decomposition["b1"]["budget_grid"] != BUDGETS:
        parser.error("Decomposition must contain nine tracks at the same budgets")
    for row in track_rows:
        if row["track"] == "glm-4-9b-chat-hf-bf16":
            continue
        left.plot(BUDGETS, [p["fnr"]["float"] for p in row["curve"]],
                  color="#b3bbc3", linewidth=0.85, alpha=0.8, zorder=1)
    glm = next(row for row in track_rows if row["track"] == "glm-4-9b-chat-hf-bf16")
    left.plot(BUDGETS, [p["fnr"]["float"] for p in glm["curve"]],
              color="#a35435", linewidth=1.15, linestyle="--", zorder=2)
    left.plot(BUDGETS, rates, color="#344052", linewidth=1.6,
              marker="o", markersize=4.5, markeredgecolor="white",
              markeredgewidth=0.7, zorder=3)
    left.set_xlim(13, 320)
    left.set_ylim(0, 1)
    left.set_yticks([0, 0.25, 0.5, 0.75, 1],
                   ["0", "0.25", "0.50", "0.75", "1.00"])
    left.set_xlabel("Distinct target calls per canary", labelpad=6, fontsize=8.7)
    left.set_ylabel("Found-nothing rate", labelpad=6)
    left.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    left.set_axisbelow(True)
    left.legend(handles=[
        Line2D([], [], color="#344052", linewidth=1.6, label="Primary (run-equal)"),
        Line2D([], [], color="#b3bbc3", linewidth=0.9, label="Other tracks"),
        Line2D([], [], color="#a35435", linewidth=1.15, linestyle="--", label="GLM-4-9B bf16"),
    ], loc="lower left", frameon=False, fontsize=7.0, handlelength=1.7,
        borderaxespad=0.3, labelspacing=0.3)
    left.annotate(f"{100 * rates[0]:.1f}%", (BUDGETS[0], rates[0]),
                  xytext=(1, 7), textcoords="offset points", fontsize=8.3,
                  ha="left", va="bottom")
    left.annotate(f"{100 * rates[-1]:.1f}%", (BUDGETS[-1], rates[-1]),
                  xytext=(-1, -10), textcoords="offset points", fontsize=8.3,
                  ha="right", va="top")

    positions = list(range(10))
    for position, track in zip(positions, TRACK_LABELS):
        row = by_track[track]
        # The short connector pairs two stored estimates, not a confidence bar.
        right.plot([row["clean"], row["legacy"]],
                   [position - 0.10, position + 0.10],
                   color="#c3c3c3", linewidth=0.8, zorder=1)
        right.plot(row["clean"], position - 0.10, marker="o", markersize=3.7,
                   color=CLEAN, linestyle="none", zorder=3)
        right.plot(row["legacy"], position + 0.10, marker="D", markersize=3.3,
                   color=LEGACY, markerfacecolor="white", markeredgewidth=0.9,
                   linestyle="none", zorder=3)

    overall_position = 11
    right.axhspan(10.35, 11.65, facecolor="#f0f2f4", edgecolor="none", zorder=0)
    right.axhline(10.25, color="#aeb5bc", linewidth=0.65)
    for key, offset, color, marker, face in (
        ("primary", -0.18, CLEAN, "o", CLEAN),
        ("comparator", 0.18, LEGACY, "D", "white"),
    ):
        point = b2[key]["point"]
        lo, hi = b2[key]["ci95"]
        # These differences only render the published marginal CI endpoints.
        right.errorbar(point, overall_position + offset,
                       xerr=[[point - lo], [hi - point]], fmt=marker,
                       markersize=4.0, color=color, markerfacecolor=face,
                       markeredgewidth=0.9, capsize=2.3, capthick=0.9,
                       elinewidth=1.15, zorder=4)
    right.set_yticks(positions + [overall_position],
                      list(TRACK_LABELS.values()) + ["Overall (track-equal)"],
                      fontsize=8.0)
    right.get_yticklabels()[-1].set_fontweight("bold")
    right.set_ylim(11.85, -0.75)
    right.set_xlim(-0.025, 1.025)
    right.set_xticks([0, 0.25, 0.5, 0.75, 1],
                     ["0", "0.25", "0.50", "0.75", "1.00"])
    right.set_xlabel("Recovery at 128 issued calls", labelpad=6)
    right.tick_params(axis="y", length=0, pad=5)
    right.grid(axis="x", color="#e4e4e4", linewidth=0.6)
    right.set_axisbelow(True)
    right.spines["left"].set_visible(False)

    fig.text(0.085, 0.96, "(a) B1: nested-budget audit", fontsize=9.5, va="top")
    fig.text(0.445, 0.96, "(b) B2: clean and legacy families", fontsize=9.5, va="top")
    fig.text(0.085, 0.885, "57 runs, run-averaged", fontsize=8.2, va="center")
    fig.text(0.445, 0.885, "59 runs, 10 tracks", fontsize=8.2, va="center")
    right.legend(handles=[
        Line2D([], [], color=CLEAN, marker="o", linestyle="none",
               markersize=4, label="Clean"),
        Line2D([], [], color=LEGACY, marker="D", linestyle="none",
               markerfacecolor="white", markersize=3.6, label="Legacy"),
    ], loc="lower center", bbox_to_anchor=(0.48, 1.035),
        ncol=2, frameon=False, fontsize=8.2, handletextpad=0.45,
        columnspacing=1.2, borderaxespad=0)

    d = b1["claim"]["D"]
    dlo, dhi = b1["claim"]["ci95"]
    fig.text(0.085, 0.068,
             f"$D = {100 * d:.1f}$ pp\n95% CI [{100 * dlo:.1f}, {100 * dhi:.1f}]",
             fontsize=8.7, va="bottom", linespacing=1.4)
    delta = b2["contrast"]["point"]
    low, high = b2["contrast"]["ci95"]
    fig.text(0.49, 0.068,
             f"Paired $\\Delta = {100 * delta:.1f}$ pp (legacy - clean)\n"
             f"95% CI [{100 * low:.1f}, {100 * high:.1f}]",
             fontsize=8.7, va="bottom", linespacing=1.4)

    output = root / "figures/main_evidence_v3.pdf"
    fig.savefig(output, metadata={
        "Title": "B1 nested audits and B2 clean-versus-legacy recovery",
        "Author": "",
        "Subject": "Accepted B1 primary and post hoc track curves; B2 track points and overall intervals; no re-estimation",
        "Creator": "figures/plot_main_evidence_v3.py",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
