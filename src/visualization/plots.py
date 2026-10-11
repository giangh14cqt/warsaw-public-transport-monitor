"""
Reusable High-Resolution Visualizations for xAI Diagnostics & Policy Exhibits.
Decouples Matplotlib figure rendering from mathematical and algorithmic modules.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.visualization.style import COLORS, set_publication_style

logger = logging.getLogger(__name__)


def _ensure_dir(path: Union[str, Path]):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)


def plot_shap_beeswarm(
    explanation: Any,
    output_path: Union[str, Path],
    max_display: int = 15,
) -> str:
    """Generate and save global TreeSHAP Beeswarm summary plot."""
    _ensure_dir(output_path)
    set_publication_style()

    plt.figure(figsize=(10, 7), dpi=300)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(
        "Warsaw Transit Delay Telemetry: Global TreeSHAP Feature Attribution",
        fontsize=12,
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved global TreeSHAP summary plot to: {output_path}")
    return str(output_path)


def plot_shap_waterfall(
    explanation_row: Any,
    output_path: Union[str, Path],
    sample_idx: int = 0,
    max_display: int = 10,
) -> str:
    """Generate and save local TreeSHAP Waterfall plot for an individual observation."""
    _ensure_dir(output_path)
    set_publication_style()

    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.waterfall(explanation_row, max_display=max_display, show=False)
    plt.title(f"Local Delay Decomposition (Observation #{sample_idx})", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved local waterfall plot to: {output_path}")
    return str(output_path)


def plot_shap_grouped_importance(
    grouped_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> str:
    """Generate and save policy domain grouped SHAP importance horizontal bar chart."""
    _ensure_dir(output_path)
    set_publication_style()

    plt.figure(figsize=(8, 4.5), dpi=300)
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["purple"]]
    bars = plt.barh(
        grouped_df["group"],
        grouped_df["relative_impact_pct"],
        color=colors[: len(grouped_df)],
        alpha=0.85,
        edgecolor="black",
    )
    plt.xlabel("Relative Delay Attribution (%)", fontsize=11)
    plt.title("TreeSHAP Grouped Policy Domain Importance", fontsize=12, pad=12)
    plt.gca().invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved grouped SHAP importance plot to: {output_path}")
    return str(output_path)


def plot_ale_curve(
    feature_name: str,
    ale_df: pd.DataFrame,
    tipping_points: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate and save a publication-quality 1D ALE diagnostic chart."""
    if output_path:
        _ensure_dir(output_path)
    set_publication_style()

    fig, (ax_main, ax_rug) = plt.subplots(
        2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [5, 1]}, sharex=True, dpi=300
    )

    x = ale_df["grid_val"].to_numpy()
    y = ale_df["ale_centered"].to_numpy()

    # Main ALE curve
    ax_main.plot(x, y, color=COLORS["primary"], lw=2.5, label="1D Accumulated Local Effect (ALE)")
    ax_main.axhline(0, color="gray", linestyle="--", alpha=0.5, lw=1)

    # Highlight freezing point for temperature
    if "temp" in feature_name.lower() and x.min() <= 0 <= x.max():
        ax_main.axvline(
            0, color=COLORS["danger"], linestyle=":", lw=1.8, label="Freezing Boundary (0°C)"
        )

    # Highlight tipping points
    if tipping_points:
        for pt in tipping_points:
            tx = pt["threshold_value"]
            ty = pt["ale_effect_at_threshold"]
            ax_main.axvline(
                tx, color=COLORS["secondary"], linestyle="--", lw=1.5, label=f"Tipping Point ({tx:.2f})"
            )
            ax_main.scatter([tx], [ty], color=COLORS["secondary"], s=60, zorder=5)
            ax_main.annotate(
                f"{pt['threshold_value']:.1f}",
                xy=(tx, ty),
                xytext=(tx + 0.05 * (x.max() - x.min()), ty + 1.0),
                fontsize=9,
                fontweight="bold",
                color="#d95f02",
                arrowprops=dict(arrowstyle="->", color=COLORS["secondary"], lw=1),
            )

    ax_main.set_ylabel("Accumulated Delay Effect (seconds)", fontsize=11)
    ax_main.set_title(
        f"Non-Linear Delay Response: 1D ALE for {feature_name.replace('_', ' ').title()}",
        fontsize=12,
        pad=10,
    )
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(loc="best", framealpha=0.9, fontsize=9)

    # Density rug
    ax_rug.hist(x, bins=len(x), color="gray", alpha=0.5, edgecolor="none")
    ax_rug.set_yticks([])
    ax_rug.set_xlabel(f"{feature_name.replace('_', ' ').title()}", fontsize=11)
    ax_rug.set_ylabel("Density", fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved ALE chart to: {output_path}")
        return str(output_path)
    else:
        plt.close("all")
        return ""


def plot_interaction_buffering(
    shock_feature: str,
    x_grid: np.ndarray,
    preds_mixed: np.ndarray,
    preds_dedicated: np.ndarray,
    output_path: Union[str, Path],
) -> str:
    """Plot 2D interaction curve comparing predicted delay escalation under shock_feature."""
    _ensure_dir(output_path)
    set_publication_style()

    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(
        x_grid, preds_mixed, color=COLORS["danger"], lw=2.5, label="Shared Mixed Traffic (ROW = 0)"
    )
    plt.plot(
        x_grid,
        preds_dedicated,
        color=COLORS["success"],
        lw=2.5,
        linestyle="--",
        label="Dedicated Right-of-Way (ROW = 1)",
    )

    # Fill buffering gap
    plt.fill_between(
        x_grid,
        preds_dedicated,
        preds_mixed,
        where=(preds_mixed >= preds_dedicated),
        color=COLORS["success"],
        alpha=0.15,
        label="Infrastructure Delay Buffering Zone",
    )

    clean_shock = shock_feature.replace("_", " ").title()
    plt.xlabel(f"{clean_shock}", fontsize=11)
    plt.ylabel("Expected Delay Output (seconds)", fontsize=11)
    plt.title(
        f"Infrastructure Buffering Interaction: {clean_shock} vs Dedicated ROW",
        fontsize=12,
        pad=12,
    )
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved interaction plot to: {output_path}")
    return str(output_path)


def plot_counterfactual_case_study(
    original_row: pd.Series,
    counterfactual_row: pd.Series,
    actionable_features: List[str],
    output_path: Union[str, Path],
) -> str:
    """Generate and save local counterfactual case study visualization."""
    _ensure_dir(output_path)
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    y_pos = np.arange(len(actionable_features))

    orig_vals = [float(original_row.get(f, 0.0)) for f in actionable_features]
    cf_vals = [float(counterfactual_row.get(f, 0.0)) for f in actionable_features]

    width = 0.35
    ax.barh(y_pos - width / 2, orig_vals, width, label="Observed Severe Incident", color=COLORS["danger"], alpha=0.8)
    ax.barh(y_pos + width / 2, cf_vals, width, label="Counterfactual On-Time Remedy", color=COLORS["success"], alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ").title() for f in actionable_features])
    ax.set_xlabel("Feature Magnitude / Operational Setting")
    ax.set_title("Actionable Counterfactual Remedy for Severe Delay Incident")
    ax.legend(loc="best")
    plt.tight_layout()

    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close("all")
    logger.info(f"Counterfactual case study saved to {output_path}")
    return str(output_path)


def plot_counterfactual_policy_summary(
    summary_dict: Dict[str, Any],
    output_path: Union[str, Path],
) -> str:
    """Plot comparative bar chart comparing mean delay reduction and on-time recovery rates."""
    _ensure_dir(output_path)
    set_publication_style()

    savings = summary_dict["mean_delay_reduction_seconds"]
    recovery = summary_dict["on_time_recovery_rates"]

    policy_names = [
        "Dedicated ROW\nUpgrade",
        "Headway\nRegularization",
        "Upstream Delay\nAbsorption",
        "Transit Signal\nPriority (TSP)",
        "Joint Multimodal\nRemedy",
    ]
    keys = [
        "dedicated_row_upgrade",
        "headway_regularization",
        "upstream_delay_absorption",
        "transit_signal_priority",
        "joint_multimodal_remedy",
    ]

    savings_vals = [savings.get(k, 0.0) for k in keys]
    recovery_vals = [recovery.get(k, 0.0) * 100 for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Mean Delay Reduction (Seconds)
    bar_colors = ["#0275d8", "#5bc0de", "#f0ad4e", "#9b59b6", "#5cb85c"]
    bars1 = ax1.bar(
        policy_names, savings_vals, color=bar_colors, edgecolor="black", alpha=0.85, width=0.55
    )
    ax1.set_ylabel("Mean Delay Reduction (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title("Average Delay Attenuation by Intervention", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.set_ylim(0, max(max(savings_vals) * 1.25, 1.0) if savings_vals else 5.0)

    for bar in bars1:
        h = bar.get_height()
        offset = max(h * 0.04, 0.1)
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"+{h:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Panel 2: On-Time Recovery Rate (%)
    bars2 = ax2.bar(
        policy_names, recovery_vals, color=bar_colors, edgecolor="black", alpha=0.85, width=0.55
    )
    ax2.set_ylabel("On-Time Restoration Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title(
        r"Probability of Achieving On-Time Status ($\Delta t \leq 120$s)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.set_ylim(0, 115)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 2.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.suptitle(
        f"Empirical Policy Efficacy: Transit Delay Counterfactual Remedies (N = {summary_dict.get('total_incidents_analyzed', 0)})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close("all")
    logger.info(f"Counterfactual policy summary chart saved to {output_path}")
    return str(output_path)
