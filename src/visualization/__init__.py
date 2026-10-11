"""
Warsaw Transit Telemetry & Attribution System: Visualization Layer.
Provides publication-grade styling and visualization generators for xAI diagnostics and model benchmarks.
"""

from src.visualization.plots import (
    plot_ale_curve,
    plot_counterfactual_case_study,
    plot_counterfactual_policy_summary,
    plot_interaction_buffering,
    plot_shap_beeswarm,
    plot_shap_grouped_importance,
    plot_shap_waterfall,
)
from src.visualization.style import publication_theme, set_publication_style

__all__ = [
    "set_publication_style",
    "publication_theme",
    "plot_shap_beeswarm",
    "plot_shap_waterfall",
    "plot_shap_grouped_importance",
    "plot_ale_curve",
    "plot_interaction_buffering",
    "plot_counterfactual_case_study",
    "plot_counterfactual_policy_summary",
]
