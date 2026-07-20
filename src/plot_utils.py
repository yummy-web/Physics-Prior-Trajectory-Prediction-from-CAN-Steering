"""plot_utils.py — unified plotting style (English labels; publication-ready).
- 300 DPI, larger fonts for two-column IEEE figures.
- save() strips all titles (axes titles + suptitle) so figures carry NO title
  (journals use captions instead), per submission requirement.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "lines.linewidth": 2.0,
})

STRIP_TITLES = True  # journals use figure captions; no in-figure titles


def _strip(fig):
    if not STRIP_TITLES:
        return
    for ax in fig.get_axes():
        ax.set_title("")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")


def save(fig, name):
    _strip(fig)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [figure] {path}")
    return path
