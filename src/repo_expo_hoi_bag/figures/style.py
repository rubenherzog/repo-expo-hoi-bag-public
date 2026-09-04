"""The single source of truth for every publication figure style."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

for candidate in (
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
):
    try:
        if Path(candidate).exists():
            fm.fontManager.addfont(candidate)
    except OSError:
        pass

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"],
        # mathtext defaults to the "dejavusans" fontset, which embeds DejaVu into
        # any figure carrying a $...$ label even when every other glyph is Arial.
        # "custom" routes mathtext through font.sans-serif above, so a panel
        # labelled R$^2$ stays single-font and fully editable in Illustrator.
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        # Pinned so the unset default ("cursive") cannot fall back to DejaVu and
        # reintroduce a second font through the calligraphic slot.
        "mathtext.cal": "sans",
    }
)

# Merged sensitivity figures carry one row per BAG, structural before functional
# (CLAUDE.md 3), and label the first panel of each row "<letter>. <description>"
# left-aligned -- the plot_fig2_grid_v3 convention. No suptitle.
BAG_ROW_ORDER = ("structural", "functional")
BAG_LABELS = {"structural": "Structural BAG", "functional": "Functional BAG", "combined": "Combined BAG"}
# Compact BAG tags for Source Data sheet names (xlsx caps names at 31 chars).
BAG_SHORT = {"structural": "struct", "functional": "func", "combined": "comb"}
ROW_LETTERS = ("a", "b", "c", "d", "e", "f", "g", "h")

SYN_COLOR = "#1B6B2E"
RED_COLOR = "#4B0082"
SINGLE_COLOR = "#E07B00"
NULL_COLOR = "#AAAAAA"
RUNG_ORDER = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
RUNG_LABELS = ("OLS", "d1", "d2", "d3")
# The only user-visible spelling of a model level. The analysis names these
# levels OLS / depth-1 / depth-2 / depth-3 and never says "rung" or "XGB", so no
# figure axis, tick, legend or caption may use those words either.
LEVEL_LABELS = {"ols": "OLS", "xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
LEVEL_AXIS_LABEL = "Model level"
RUNG_COLOR = {"ols": "#666666", "xgb_tree_d1": "#E6AB02", "xgb_tree_d2": "#66A61E", "xgb_tree_d3": "#CC00CC"}
FS, FS_TK, GRID_FS, GRID_FS_TK = 22, 20, 15, 13


def style_axis(axis):
    """Apply the project-wide clean-axis convention."""
    axis.spines[["top", "right"]].set_visible(False)
    return axis


def save_figure(figure, name: str, output_dir: Path) -> tuple[Path, Path, Path]:
    """Save all figure formats to the configured external output root."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_dir / f"{name}.{extension}" for extension in ("pdf", "svg", "png"))
    for output in outputs:
        figure.savefig(output, dpi=300, bbox_inches="tight")
    return outputs  # type: ignore[return-value]
