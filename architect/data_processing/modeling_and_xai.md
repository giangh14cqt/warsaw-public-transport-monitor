# Modeling Suite & Explainable AI (xAI) Architecture

This document details the econometric and machine learning modeling suite, the purged temporal block validation framework, and the explainable AI diagnostic pipeline.

---

## 🛡️ Purged Temporal Block Cross-Validation Framework

Transit delay data exhibits strong autocorrelation. Standard $K$-fold cross-validation or random train-test splitting introduces catastrophic **data leakage** because downstream stops on the same vehicle run appear in both train and test partitions.

To prevent leakage, [`src/models/validation.py`](file:///Users/truonggiangdo/Data/LearningMaterials/UW/Thesis/DelayTraffic/playground/src/models/validation.py) implements **Purged Temporal Block Splitting** with an **Adaptive Embargo Buffer**:

```mermaid
gantt
    title Purged Temporal Block Cross-Validation with Embargo Buffer
    dateFormat X
    axisFormat %s

    section Fold 1
    Train Block (e.g., 60% of time) :done, 0, 60
    Embargo Buffer (Purged Gap)     :crit, 60, 65
    Validation Block (e.g., 20%)    :active, 65, 85
    Test Block (Holdout)            :85, 100

    section Fold 2 (Expanding)
    Train Block (e.g., 70% of time) :done, 0, 70
    Embargo Buffer (Purged Gap)     :crit, 70, 75
    Validation Block (e.g., 15%)    :active, 75, 90
    Test Block (Holdout)            :90, 100
```

### Mathematical Zero-Leakage Guarantee
The system verifies that for every train index $i$ and validation index $j$:

$$\max_{i \in \text{Train}} \left(t_i\right) + \text{Embargo} \le \min_{j \in \text{Val}} \left(t_j\right)$$

- **Adaptive Embargo Scaling**: For standard datasets, the embargo is fixed at 15–30 minutes (covering maximum vehicle transit time across Warsaw). For short sample slices ($<4$ hours), the splitter dynamically scales the embargo to prevent fold starvation while maintaining the mathematical zero-leakage guarantee verified via `verify_no_leakage()`.

---

## 📈 Model Architecture & Benchmarking Tier

```mermaid
flowchart TD
    FeatureMart[("Feature Mart<br/>(Stratified Cohort or Pooled)")] --> Splitter["Purged Temporal Block Splitter"]
    
    Splitter --> TrainSet["Train Set (Chronological T_0 ... T_train)"]
    Splitter --> ValSet["Validation Set (T_val ... T_end)"]

    subgraph EconometricBaseline ["1. Econometric TWFE Baseline (src/models/twfe.py)"]
        direction TB
        TWFE_Model["Two-Way Fixed Effects Panel Regression<br/><code>y_{r,t} = β X_{r,t} + α_r + γ_t + ε_{r,t}</code>"]
        TWFE_SE["Clustered Standard Errors (by route/line)"]
        TWFE_Out["Linear Coefficients & Baseline Metrics"]
        TWFE_Model --> TWFE_SE --> TWFE_Out
    end

    subgraph GBDTSuite ["2. Nonlinear GBDT Suite (src/models/gbm.py)"]
        direction TB
        LGBM["LightGBM Regressor<br/>• Gradient-based One-Side Sampling (GOSS)<br/>• Fast histogram binning<br/>• L1/L2 regularization"]
        CatB["CatBoost Regressor<br/>• Symmetric oblivious trees<br/>• Native categorical target encoding<br/>• Robust to correlated exogenous shocks"]
        GBM_Out["GBDT Model Artifacts (.joblib)"]
        LGBM --> GBM_Out
        CatB --> GBM_Out
    end

    TrainSet --> TWFE_Model
    TrainSet --> LGBM
    TrainSet --> CatB

    ValSet --> EvalEngine["Comprehensive Evaluation Suite (src/models/metrics.py)<br/>• MAE (Mean Absolute Error)<br/>• RMSE (Root Mean Squared Error)<br/>• MedAE (Median Absolute Error)<br/>• WAPE (Weighted Absolute Percentage Error)<br/>• R² (Explained Variance)"]

    TWFE_Out --> EvalEngine
    GBM_Out --> EvalEngine

    EvalEngine --> CompTable[("reports/tables/benchmark_comparison.csv")]
    EvalEngine --> CohortTable[("reports/tables/benchmark_by_cohort.csv")]

    %% Styles
    classDef dataStyle fill:#2d3748,stroke:#4a5568,color:#edf2f7;
    classDef twfeStyle fill:#1a365d,stroke:#2b6cb0,color:#bee3f8;
    classDef gbmStyle fill:#702459,stroke:#b83280,color:#fed7e2;
    classDef repStyle fill:#744210,stroke:#ecc94b,color:#ffffb0;

    class FeatureMart,Splitter,TrainSet,ValSet dataStyle;
    class TWFE_Model,TWFE_SE,TWFE_Out twfeStyle;
    class LGBM,CatB,GBM_Out gbmStyle;
    class EvalEngine,CompTable,CohortTable repStyle;
```

---

## 🔍 Explainable AI (xAI) Diagnostic Pipeline

```mermaid
flowchart LR
    subgraph TrainedModels ["Trained GBDT Model"]
        M["LightGBM / CatBoost"]
    end

    subgraph xAIEngines ["xAI Diagnostic Engines (src/xai/)"]
        direction TB
        SHAP["<b>TreeSHAP (shap_explainer.py)</b><br/>• Global feature importance (beeswarm)<br/>• Local instance attribution (waterfall)<br/>• Exact Shapley values"]
        ALE["<b>Accumulated Local Effects (ale_curves.py)</b><br/>• 1D non-linear response curves<br/>• 2D interaction contour surfaces<br/>• Immune to feature correlation bias"]
        Inter["<b>Interactions (interactions.py)</b><br/>• Friedman's H-statistic<br/>• Quantifies non-linear synergy"]
        DiCE["<b>DiCE Counterfactuals (counterfactuals.py)</b><br/>• Genetic optimization<br/>• Minimum feature change for delay recovery"]
    end

    subgraph Figures ["Publication Visualizations (src/visualization/)"]
        direction TB
        F_Shap[("shap_beeswarm.png<br/>shap_waterfall.png")]
        F_Ale[("ale_temperature.png<br/>ale_precipitation.png")]
        F_Inter[("interaction_matrix.png")]
        F_Dice[("counterfactual_paths.png")]
    end

    M --> SHAP --> F_Shap
    M --> ALE --> F_Ale
    M --> Inter --> F_Inter
    M --> DiCE --> F_Dice
```

### Diagnostic Capabilities
1. **TreeSHAP**: Measures the exact additive attribution of each exogenous feature (e.g., rainfall, wind, corridor speed limit) relative to baseline upstream delay.
2. **Accumulated Local Effects (ALE)**: Unlike Partial Dependence Plots (PDP) which extrapolate over impossible feature combinations (e.g., snow at $+30^\circ\text{C}$), ALE computes pure marginal effects conditional on the local feature distribution.
3. **Friedman's $H$-Statistic**: Discovers non-linear compounding effects—such as heavy rain exacerbating delays on narrow single-lane corridors ($H > 0.15$).
4. **DiCE Counterfactuals**: Provides urban transport planners with actionable questions: *"What minimal headway or schedule buffer adjustment would have prevented this trip from exceeding 3 minutes of delay under 15 mm/h rainfall?"*
