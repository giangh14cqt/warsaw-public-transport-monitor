---
description: Run purged temporal block validation, train econometric TWFE baseline, and evaluate gradient boosting models
---

# Modeling & Evaluation Workflow (`/eval-models`)

This workflow executes Phase 4 modeling POC benchmarks:
1. Purged temporal block splitting (Weeks 1–3 Train, Week 4 Val, Week 5 Test).
2. Two-Way Fixed Effects (TWFE) panel regression baseline.
3. Non-linear Gradient Boosting (LightGBM / CatBoost).
4. Generates model benchmark metrics (RMSE, MAE, R²).

## Step 1: Run Model Benchmark Suite
```bash
python3 -m src.models.benchmark
```

## Step 2: Generate xAI Attributions
```bash
python3 -m src.xai.shap_explainer
```
