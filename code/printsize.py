"""Print sizes for the paper's figures.

Each figure is drawn at the width at which the manuscript prints it, so that LaTeX
does not rescale it and the font sizes set here are the printed sizes.  The text
block of the manuscript (12pt article, default geometry, letter paper) is 430pt wide.
Figures are saved without a tight bounding box so the PDF keeps exactly this size.
"""

TEXTWIDTH_IN = 430.0 / 72.27   # 5.95 inches


def width(fraction: float) -> float:
    """Figure width in inches for \\includegraphics[width=<fraction>\\textwidth]."""
    return fraction * TEXTWIDTH_IN


# No printed text below 8pt: ticks, legends, and in-figure notes at 8pt, labels at 9pt.
RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.bbox": "standard",
}
