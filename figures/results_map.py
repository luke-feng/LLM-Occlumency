"""Figure 1: result map, quantifier ladder x decoding regime."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
root = parser.parse_args().root.resolve()

STRUCT = "#1C3A78"; LEAK = "#CE2020"; BLOCK = "#5C6880"
VAULT  = "#E9EFFB"; REDBG = "#FBECEC"; GOLD = "#CEA028"; GOLDBG = "#FBF3DD"

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

# limits are set to the drawn extent, not to round numbers, so the figure
# carries no dead margin; figsize is scaled with them to keep the box aspect
fig, ax = plt.subplots(figsize=(8.87, 2.83))
ax.set_xlim(0.018, 0.962); ax.set_ylim(-0.036, 0.995); ax.axis("off")

DX, SX = 0.315, 0.765      # column centers
BW = 0.37                   # box width
R1, R2, R3 = 0.775, 0.42, 0.09
BH = 0.235

def box(cx, cy, lines, fc, ec, dashed=False, bh=BH, bw=BW):
    p = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                       boxstyle="round,pad=0.008,rounding_size=0.018",
                       fc=fc, ec=ec, lw=1.4,
                       ls=(0, (4, 3)) if dashed else "-", zorder=3)
    ax.add_patch(p)
    n = len(lines)
    for k, (txt, fs, col, bold) in enumerate(lines):
        yy = cy + (n - 1) * 0.042 - k * 0.084
        ax.text(cx, yy, txt, fontsize=fs, color=col, ha="center",
                va="center", zorder=4,
                fontweight="bold" if bold else "normal")

def arrow(cx, y0, y1, label, lcol, fs=8.6):
    ax.annotate("", xy=(cx, y1), xytext=(cx, y0),
                arrowprops=dict(arrowstyle="-|>", color=BLOCK, lw=1.3,
                                mutation_scale=13), zorder=2)
    ax.text(cx + 0.012, (y0 + y1) / 2, label, fontsize=fs, color=lcol,
            ha="left", va="center", style="italic", zorder=4)

# column headers
ax.text(DX, 0.965, "deterministic decoding  ($m=0$)", fontsize=11,
        color=STRUCT, ha="center", va="center", fontweight="bold")
ax.text(SX, 0.965, r"stochastic decoding  ($m>0$)", fontsize=11,
        color=STRUCT, ha="center", va="center", fontweight="bold")

# row labels
ax.text(0.058, R1, "attacker\nfind one\nwitness", fontsize=9, color=LEAK,
        ha="center", va="center", style="italic")
ax.text(0.058, R2, "defender certify\nall prompts meet\nthe leak bound", fontsize=8.6, color=STRUCT,
        ha="center", va="center", style="italic")
ax.text(0.058, R3, "designer\nchoose a defense\nfor all prompts", fontsize=9, color=GOLD,
        ha="center", va="center", style="italic")

# deterministic column
box(DX, R1, [("JAILBREAK-EXISTS", 10, LEAK, True),
             (r"$\exists p:\ \mathrm{leak}(p)=1$", 10.5, "black", False),
             ("NP-complete  (Thm 1)", 9.3, LEAK, False)], REDBG, LEAK)
box(DX, R2, [("SAFE $=$ CERTIFY$_0$", 10, STRUCT, True),
             (r"$\forall p:\ \mathrm{leak}(p)=0$", 10.5, "black", False),
             ("co-NP-complete  (Thm 2)", 9.3, STRUCT, False)], VAULT, STRUCT)
box(DX, R3, [("DEFEND", 10, GOLD, True),
             (r"$\exists d:\ \mathrm{Adm}(d)\wedge\forall p:\ \mathrm{leak}_d(p)=0$",
              10.0, "black", False),
             (r"$\Sigma_2^p$-complete  (Thm 5)", 9.3, GOLD, False)], GOLDBG, GOLD)

# stochastic column
box(SX, R1, [("BREACH", 10, LEAK, True),
             (r"$\exists p,r:\ \mathrm{leak}(p,r)=1$", 10.5, "black", False),
             (r"NP-complete  (Thm 1)", 9, LEAK, False)],
    REDBG, LEAK)
box(SX, R2, [("CERTIFY", 10, STRUCT, True),
             (r"$\forall p:\ q(p)\leq\tau$", 10.5, "black", False),
             (r"co-NP$^{\mathrm{PP}}$-complete  (Thm 4)", 9.3, STRUCT, False)], VAULT, STRUCT)
box(SX, R3, [("STOCHASTIC DEFENSE DESIGN\nnot formalized in this paper",
              9, BLOCK, False)], "white", BLOCK, dashed=True, bh=0.175)

# arrows down the ladder
gap = BH/2 + 0.008
arrow(DX, R1-gap, R2+gap, "from  $\\exists$  to  $\\forall$  over all prompts", BLOCK)
arrow(DX, R2-gap, R3+gap, "add  $\\exists d$  with  $|d|\\leq\\beta$", BLOCK)
# Counting is a separate result, not an arrow suggesting that BREACH and
# threshold CERTIFY are complements (they may both hold for positive tau).
ax.text(SX, (R1 + R2) / 2,
        "LeakCount: #P-complete\nexact $q(p)$: #P-hard  (Thm 3)",
        fontsize=7.8, color=BLOCK, ha="center", va="center", style="italic")
arrow(SX, R2-gap, R3+0.098, "", BLOCK)

fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
# crop to the drawn content; a shorter figure also costs less vertical space
# on the page, which the 9-page limit cares about
fig.savefig(root / "figures/results_map.pdf", bbox_inches="tight", pad_inches=0.05,
            metadata={"Title":"Complexity result map","Author":"",
                      "Creator":"figures/results_map.py","CreationDate":None,"ModDate":None})
print("saved")
