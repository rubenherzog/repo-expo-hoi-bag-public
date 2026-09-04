#!/usr/bin/env python3
"""Generate a standalone exposome domain legend figure."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oinfo_bag_ladder.make_boss_figure import DOMAIN_ORDER, DOMAIN_COLORS, LEGEND_FONTSIZE

OUTPUT_ROOT = Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
OUTPUT_DIR = OUTPUT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "exposome_domain_legend.png"


def build_exposome_domain_legend():
    labels = [dom for dom in DOMAIN_ORDER if dom in DOMAIN_COLORS]
    handles = [Patch(facecolor=DOMAIN_COLORS[dom], edgecolor="black", label=dom) for dom in labels]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.axis("off")

    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        frameon=True,
        framealpha=0.9,
        borderpad=0.6,
        handlelength=1.5,
        handleheight=1.2,
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)

    fig.savefig(OUTPUT_FILE, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved legend figure: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_exposome_domain_legend()
