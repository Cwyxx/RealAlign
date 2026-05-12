"""Horizontal stacked bar chart for Human Evaluation Win/Tie/Lose rates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.linewidth"] = 0.8

COMPARISONS = [
    "vs. Diffusion-DPO",
    "vs. SD-3.5-M",
]
WIN  = np.array([56.67, 63.33])
TIE  = np.array([10.00, 16.67])
LOSE = np.array([33.33, 20.00])

C_WIN, C_TIE, C_LOSE = "#2E8B57", "#BDBDBD", "#D55E00"

fig, ax = plt.subplots(figsize=(6.0, 1.9))

y = np.arange(len(COMPARISONS))
bar_h = 0.55

ax.barh(y, WIN,  height=bar_h, color=C_WIN,  edgecolor="white", linewidth=0.8, label="Win")
ax.barh(y, TIE,  height=bar_h, left=WIN,         color=C_TIE,  edgecolor="white", linewidth=0.8, label="Tie")
ax.barh(y, LOSE, height=bar_h, left=WIN + TIE,   color=C_LOSE, edgecolor="white", linewidth=0.8, label="Lose")

def annotate(values, lefts, color="white"):
    for yi, v, l in zip(y, values, lefts):
        if v < 4:
            continue
        ax.text(l + v / 2, yi, f"{v:.2f}%", va="center", ha="center",
                color=color, fontsize=9, fontweight="bold")

annotate(WIN,  np.zeros_like(WIN))
annotate(TIE,  WIN, color="#333333")
annotate(LOSE, WIN + TIE)

ax.set_yticks(y)
ax.set_yticklabels(COMPARISONS, fontsize=10)
ax.set_xlim(0, 100)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels([f"{t}%" for t in [0, 20, 40, 60, 80, 100]], fontsize=9)
ax.set_xlabel("Human Preference (%)", fontsize=10)

ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=3, width=0.8)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#888888")

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=10,
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=1.6,
)

plt.tight_layout()

out_dir = Path(__file__).resolve().parent
for ext in ("pdf", "png", "svg"):
    fig.savefig(out_dir / f"human_eval_winrate.{ext}", dpi=300, bbox_inches="tight")
print("Saved figures to:", out_dir)
