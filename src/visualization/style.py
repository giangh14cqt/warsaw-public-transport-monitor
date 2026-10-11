"""
Academic Publication Theme & Matplotlib Styling for Thesis Exhibits.
"""

from contextlib import contextmanager
from typing import Generator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical color palette for Warsaw Transit exhibits
COLORS = {
    "primary": "#1f77b4",       # ZTM Blue
    "secondary": "#ff7f0e",     # Warning Orange
    "success": "#2ca02c",       # Dedicated ROW Green
    "danger": "#d62728",        # Delay Red
    "purple": "#9467bd",        # Counterfactual Purple
    "gray": "#7f7f7f",          # Neutral Gray
    "tram": "#d62728",          # Tram Red
    "bus": "#1f77b4",           # Bus Blue
    "background_grid": "#e0e0e0",
}


def set_publication_style():
    """Apply high-resolution academic styling to Matplotlib global rcParams."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


@contextmanager
def publication_theme() -> Generator[None, None, None]:
    """Context manager for temporary academic figure styling."""
    old_params = plt.rcParams.copy()
    try:
        set_publication_style()
        yield
    finally:
        plt.rcParams.update(old_params)
