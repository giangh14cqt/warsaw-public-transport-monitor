"""
Explainable AI (xAI) Diagnostics Package.
Provides TreeSHAP, ALE curves, interaction analyses, and counterfactual diagnostics.
"""

from src.xai.shap_explainer import TreeSHAPExplainer
from src.xai.ale_curves import ALENonLinearDetector
from src.xai.interactions import InteractionQuantifier
from src.xai.counterfactuals import CounterfactualDiagnostics

__all__ = [
    "TreeSHAPExplainer",
    "ALENonLinearDetector",
    "InteractionQuantifier",
    "CounterfactualDiagnostics",
]
