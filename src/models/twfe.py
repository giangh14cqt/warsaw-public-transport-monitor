"""
Econometric Baseline: Two-Way Fixed Effects (TWFE) Panel Regression.
Estimates linear elasticities of transit delay drivers with entity (route/corridor)
and time (hour-of-day) fixed effects:
    y_{ist} = alpha_i + lambda_t + beta * X_{ist} + epsilon_{ist}

Features:
- High-performance within-transformation (demeaning) via pyhdfe / Frisch-Waugh-Lovell.
- Automatic collinearity and absorbed-effect detection (e.g., peak-hour absorbed by hour FE).
- Cluster-robust (HC1 / HC3 / robust) standard errors and t-statistics.
- Automatic point elasticity calculation: epsilon_k = beta_k * (mean(X_k) / mean(y)).
- Robust out-of-sample fixed-effects prediction for validation and test splits.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TWFEBaselineModel:
    """
    Two-Way Fixed Effects panel regression estimator.
    Absorbs entity (route_id) and temporal (hour_of_day) fixed effects.
    """

    def __init__(
        self,
        target_col: str = "delta_t_run",
        entity_col: str = "route_id",
        time_col: str = "hour_of_day",
        cov_type: str = "cluster",
    ):
        """
        Initialize TWFE Baseline Model.

        Parameters
        ----------
        target_col : str
            Dependent variable (default 'delta_t_run' or 'arrival_delay_seconds').
        entity_col : str
            Entity fixed effect column (default 'route_id').
        time_col : str
            Temporal fixed effect column (default 'hour_of_day').
        cov_type : str
            Covariance estimator for standard errors: 'cluster', 'HC1', 'HC3', or 'nonrobust'.
        """
        self.target_col = target_col
        self.entity_col = entity_col
        self.time_col = time_col
        self.cov_type = cov_type

        self.feature_cols: List[str] = []
        self.active_features: List[str] = []
        self.absorbed_features: List[str] = []
        self.coefficients: Dict[str, float] = {}
        self.standard_errors: Dict[str, float] = {}
        self.p_values: Dict[str, float] = {}
        self.t_stats: Dict[str, float] = {}
        self.elasticities: Dict[str, float] = {}
        self.conf_int_lower: Dict[str, float] = {}
        self.conf_int_upper: Dict[str, float] = {}
        self.std_coefficients: Dict[str, float] = {}

        self.grand_mean: float = 0.0
        self.entity_effects: Dict[Any, float] = {}
        self.time_effects: Dict[Any, float] = {}
        self.fitted_summary: Dict[str, Any] = {}
        self.ols_results = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sample_weight: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Fit Two-Way Fixed Effects model using within-transformation demeaning.
        Supports sample weights (WLS) for stratified inverse-probability sampling.

        Parameters
        ----------
        df : pd.DataFrame
            Training panel dataset.
        feature_cols : List[str]
            Candidate exogenous covariates.

        Returns
        -------
        Dict[str, Any]
            Estimation summary dictionary.
        """
        if df.empty:
            raise ValueError("Training DataFrame is empty.")

        # Resolve target column
        target = self.target_col
        if target not in df.columns:
            if "delta_t_run" in df.columns:
                target = "delta_t_run"
            elif "arrival_delay_seconds" in df.columns:
                target = "arrival_delay_seconds"
            else:
                raise KeyError(f"Neither '{self.target_col}' nor default targets found in columns.")
        self.target_col = target

        # Filter out entity and time cols, and deterministically collinear variables from candidate features
        cols_to_exclude = {self.target_col, self.entity_col, self.time_col}
        if self.time_col == "hour_of_day":
            cols_to_exclude.update(["is_peak_hour", "is_morning_peak", "is_evening_peak"])
        # day_of_week is a discrete cyclic indicator; linear slope extrapolates dangerously out-of-sample
        cols_to_exclude.add("day_of_week")
        # Prevent physical collinearity between straight-line length and polyline path length
        if "segment_length_meters" in feature_cols and "actual_path_length_meters" in feature_cols:
            cols_to_exclude.add("segment_length_meters")
        # Prevent collinear functional forms of instantaneous rainfall in linear specification
        if "precipitation_mm" in feature_cols:
            cols_to_exclude.update(["is_raining", "rain_intensity_category"])

        candidate_features = [
            c
            for c in feature_cols
            if c in df.columns and c not in cols_to_exclude
        ]

        # Prepare working copy and impute missing feature values with 0.0 (consistent with predict)
        cols_to_keep = [self.target_col] + candidate_features
        if self.entity_col in df.columns and self.entity_col not in cols_to_keep:
            cols_to_keep.append(self.entity_col)
        if self.time_col in df.columns and self.time_col not in cols_to_keep:
            cols_to_keep.append(self.time_col)
        if "inverse_sampling_weight" in df.columns and "inverse_sampling_weight" not in cols_to_keep:
            cols_to_keep.append("inverse_sampling_weight")

        data = df[cols_to_keep].copy()
        if sample_weight is not None:
            data["_explicit_weight"] = np.asarray(sample_weight, dtype=float)
        for col in candidate_features:
            if col in data.columns and data[col].isna().any():
                data[col] = data[col].fillna(0.0)
        data = data.dropna(subset=[self.target_col]).copy()
        if len(data) == 0:
            raise ValueError("No valid rows remaining after dropping missing values.")

        # Clean booleans and numeric casting
        valid_features = []
        for col in candidate_features:
            if data[col].dtype == bool:
                data[col] = data[col].astype(float)
                valid_features.append(col)
            elif np.issubdtype(data[col].dtype, np.number):
                valid_features.append(col)
            else:
                try:
                    data[col] = data[col].astype(float)
                    valid_features.append(col)
                except (ValueError, TypeError):
                    logger.warning(f"Dropping non-numeric feature '{col}' from TWFE regression.")

        self.feature_cols = valid_features

        y = data[self.target_col].to_numpy(dtype=float)
        weights = None
        if "_explicit_weight" in data.columns:
            weights = data["_explicit_weight"].to_numpy(dtype=float)
        elif "inverse_sampling_weight" in data.columns:
            weights = data["inverse_sampling_weight"].to_numpy(dtype=float)

        mean_y = float(np.average(y, weights=weights)) if weights is not None else float(np.mean(y))
        self.grand_mean = mean_y

        has_entity = self.entity_col in data.columns and data[self.entity_col].nunique() > 1
        has_time = self.time_col in data.columns and data[self.time_col].nunique() > 1

        y_resid = y.copy()
        X_resid = np.zeros((len(data), len(self.feature_cols)), dtype=float)

        if self.feature_cols:
            X_raw = data[self.feature_cols].to_numpy(dtype=float)

            if has_entity or has_time:
                fe_cols = []
                if has_entity:
                    fe_cols.append(self.entity_col)
                if has_time:
                    fe_cols.append(self.time_col)

                try:
                    import pyhdfe

                    algo = pyhdfe.create(data[fe_cols], drop_singletons=False)
                    w_col = weights[:, None] if weights is not None else None
                    y_resid = algo.residualize(y[:, None], weights=w_col).ravel()
                    X_resid = algo.residualize(X_raw, weights=w_col)
                except Exception as e:
                    logger.warning(
                        f"pyhdfe residualization failed ({e}); falling back to sequential demeaning."
                    )
                    y_resid = y.copy()
                    X_resid = X_raw.copy()
                    if has_entity:
                        y_resid -= (
                            data.groupby(self.entity_col)[self.target_col]
                            .transform("mean")
                            .to_numpy(float)
                        )
                        for idx, c in enumerate(self.feature_cols):
                            X_resid[:, idx] -= (
                                data.groupby(self.entity_col)[c].transform("mean").to_numpy(float)
                            )
                    if has_time:
                        y_resid -= (
                            data.groupby(self.time_col)[self.target_col]
                            .transform("mean")
                            .to_numpy(float)
                        )
                        for idx, c in enumerate(self.feature_cols):
                            X_resid[:, idx] -= (
                                data.groupby(self.time_col)[c].transform("mean").to_numpy(float)
                            )
            else:
                y_resid = y - mean_y
                X_resid = (
                    X_raw - np.average(X_raw, axis=0, weights=weights)
                    if weights is not None
                    else X_raw - np.mean(X_raw, axis=0)
                )

        # Detect active vs absorbed features (features whose variance after residualization is < 1e-6)
        self.active_features = []
        self.absorbed_features = []
        active_indices = []

        for idx, col in enumerate(self.feature_cols):
            res_var = float(np.var(X_resid[:, idx]))
            if res_var > 1e-6:
                self.active_features.append(col)
                active_indices.append(idx)
            else:
                self.absorbed_features.append(col)
                self.coefficients[col] = 0.0
                self.standard_errors[col] = 0.0
                self.p_values[col] = 1.0
                self.t_stats[col] = 0.0
                self.elasticities[col] = 0.0
                logger.info(
                    f"Feature '{col}' absorbed by fixed effects (var={res_var:.1e}); omitted from OLS."
                )

        if self.active_features:
            X_cand = X_resid[:, active_indices]

            # Prune collinear columns via QR decomposition with column pivoting
            try:
                import scipy.linalg as la

                _, R, P = la.qr(X_cand, mode="economic", pivoting=True)
                diag_R = np.abs(np.diag(R))
                tol = max(X_cand.shape) * np.spacing(np.max(diag_R)) if len(diag_R) > 0 else 1e-10
                rank = int(np.sum(diag_R > max(tol, 1e-7)))
                if rank < len(active_indices):
                    keep_cand_indices = sorted(P[:rank])
                    dropped_cand_indices = [idx for idx in range(len(active_indices)) if idx not in keep_cand_indices]
                    for d_idx in dropped_cand_indices:
                        col = self.active_features[d_idx]
                        self.absorbed_features.append(col)
                        self.coefficients[col] = 0.0
                        self.standard_errors[col] = 0.0
                        self.p_values[col] = 1.0
                        self.t_stats[col] = 0.0
                        self.elasticities[col] = 0.0
                        logger.info(f"Feature '{col}' dropped due to linear collinearity with other regressors.")

                    self.active_features = [self.active_features[i] for i in keep_cand_indices]
                    active_indices = [active_indices[i] for i in keep_cand_indices]
            except Exception as e:
                logger.debug(f"QR collinearity check skipped: {e}")

            X_active = X_resid[:, active_indices]

            # Fit OLS/WLS via statsmodels with absorbed FE degrees-of-freedom adjustment
            try:
                import statsmodels.api as sm

                if weights is not None:
                    ols_model = sm.WLS(y_resid, X_active, weights=weights)
                else:
                    ols_model = sm.OLS(y_resid, X_active)

                n_fe_dof = (int(data[self.entity_col].nunique()) if has_entity else 0) + (
                    max(0, int(data[self.time_col].nunique()) - (1 if has_entity else 0))
                    if has_time
                    else 0
                )
                ols_model.df_resid = float(max(1, len(y_resid) - X_active.shape[1] - n_fe_dof))

                if self.cov_type == "cluster" and has_entity:
                    res = ols_model.fit(
                        cov_type="cluster",
                        cov_kwds={"groups": data[self.entity_col].astype(str).to_numpy()},
                    )
                else:
                    cov_type = self.cov_type if self.cov_type in ["HC1", "HC3"] else "nonrobust"
                    res = ols_model.fit(cov_type=cov_type)
                self.ols_results = res
                beta = res.params
                bse = res.bse
                pvals = res.pvalues
                tvalues = res.tvalues
                conf_int = res.conf_int()
            except Exception as e:
                logger.warning(f"Statsmodels OLS/WLS error ({e}), using numpy least squares fallback.")
                if weights is not None:
                    w_sqrt = np.sqrt(weights)[:, None]
                    beta, _, _, _ = np.linalg.lstsq(X_active * w_sqrt, y_resid * np.sqrt(weights), rcond=None)
                else:
                    beta, _, _, _ = np.linalg.lstsq(X_active, y_resid, rcond=None)
                bse = np.zeros(len(beta))
                pvals = np.ones(len(beta))
                tvalues = np.zeros(len(beta))
                conf_int = np.column_stack([beta - 1.96 * bse, beta + 1.96 * bse])

            # Store active coefficients and compute elasticities & standardized coefficients
            std_y = float(np.std(y))
            for local_idx, col in enumerate(self.active_features):
                b = float(beta[local_idx])
                se = float(bse[local_idx])
                pval = float(pvals[local_idx])
                tstat = float(tvalues[local_idx])
                c_low = float(conf_int[local_idx, 0])
                c_high = float(conf_int[local_idx, 1])

                mean_x = float(data[col].mean())
                std_x = float(np.std(data[col]))
                elasticity = float(b * (mean_x / mean_y)) if abs(mean_y) > 1e-6 else 0.0
                std_coef = float(b * (std_x / std_y)) if std_y > 1e-6 else 0.0

                self.coefficients[col] = round(b, 6)
                self.standard_errors[col] = round(se, 6)
                self.p_values[col] = round(pval, 6)
                self.t_stats[col] = round(tstat, 4)
                self.elasticities[col] = round(elasticity, 6)
                self.std_coefficients[col] = round(std_coef, 6)
                self.conf_int_lower[col] = round(c_low, 6)
                self.conf_int_upper[col] = round(c_high, 6)
        else:
            logger.info("All exogenous features absorbed by fixed effects; relying on pure TWFE.")

        # Recover fixed effects: r = y - sum(beta * X_active)
        if self.active_features:
            active_betas = np.array([self.coefficients[c] for c in self.active_features])
            r = y - data[self.active_features].to_numpy(dtype=float) @ active_betas
        else:
            r = y.copy()

        data["_pe_residual"] = r
        if weights is not None:
            base_grand = float(np.average(r, weights=weights))
        else:
            base_grand = float(np.mean(r))
        self.grand_mean = base_grand

        data["_pe_residual"] = r - base_grand
        self.entity_effects = {}
        self.time_effects = {}

        if has_entity or has_time:
            ent_eff = {k: 0.0 for k in data[self.entity_col].unique()} if has_entity else {}
            time_eff = {k: 0.0 for k in data[self.time_col].unique()} if has_time else {}
            n_iter = 3 if (has_entity and has_time) else 1
            for _ in range(n_iter):
                if has_entity:
                    rem_ent = data["_pe_residual"].to_numpy(dtype=float) - (
                        data[self.time_col].map(time_eff).fillna(0.0).to_numpy(dtype=float)
                        if has_time
                        else 0.0
                    )
                    if weights is not None:
                        data["_w_res"] = rem_ent * weights
                        data["_w"] = weights
                        ent_sums = data.groupby(self.entity_col)[["_w_res", "_w"]].sum()
                        ent_eff = (ent_sums["_w_res"] / ent_sums["_w"].replace(0, np.nan)).fillna(0.0).to_dict()
                    else:
                        data["_rem_ent"] = rem_ent
                        ent_eff = data.groupby(self.entity_col)["_rem_ent"].mean().fillna(0.0).to_dict()
                if has_time:
                    rem_time = data["_pe_residual"].to_numpy(dtype=float) - (
                        data[self.entity_col].map(ent_eff).fillna(0.0).to_numpy(dtype=float)
                        if has_entity
                        else 0.0
                    )
                    if weights is not None:
                        data["_w_res"] = rem_time * weights
                        data["_w"] = weights
                        time_sums = data.groupby(self.time_col)[["_w_res", "_w"]].sum()
                        time_eff = (time_sums["_w_res"] / time_sums["_w"].replace(0, np.nan)).fillna(0.0).to_dict()
                    else:
                        data["_rem_time"] = rem_time
                        time_eff = data.groupby(self.time_col)["_rem_time"].mean().fillna(0.0).to_dict()
            for tmp_col in ["_w_res", "_w", "_rem_ent", "_rem_time"]:
                if tmp_col in data.columns:
                    data.drop(columns=[tmp_col], inplace=True)
            self.entity_effects = ent_eff
            self.time_effects = time_eff

        # Compute within and overall R-squared
        y_fitted = self.predict(data)
        ss_res = float(np.sum((y - y_fitted) ** 2))
        ss_tot = float(np.sum((y - mean_y) ** 2))
        overall_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

        if self.active_features:
            ss_res_within = float(np.sum((y_resid - X_active @ beta) ** 2))
            ss_tot_within = float(np.sum(y_resid**2))
            within_r2 = (
                float(1.0 - (ss_res_within / ss_tot_within)) if ss_tot_within > 1e-9 else 0.0
            )
        else:
            within_r2 = 0.0

        self.fitted_summary = {
            "rsquared": round(overall_r2, 4),
            "within_rsquared": round(within_r2, 4),
            "nobs": int(len(data)),
            "n_entities": int(len(self.entity_effects)),
            "n_time_periods": int(len(self.time_effects)),
            "active_features": self.active_features,
            "absorbed_features": self.absorbed_features,
            "cov_type": self.cov_type,
            "coefficients": self.coefficients,
            "standard_errors": self.standard_errors,
            "pvalues": self.p_values,
            "elasticities": self.elasticities,
            "std_coefficients": self.std_coefficients,
        }

        logger.info(
            f"Fitted TWFE panel model: Overall R²={overall_r2:.4f}, "
            f"Within R²={within_r2:.4f}, N={len(data)}, "
            f"Active Features={len(self.active_features)}"
        )
        return self.fitted_summary

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate out-of-sample predictions:
            y_hat = grand_mean + alpha_i + lambda_t + sum(beta_k * X_k)
        """
        if df.empty:
            return np.array([], dtype=float)

        n = len(df)
        preds = np.full(n, self.grand_mean, dtype=float)

        # Add entity fixed effect alpha_i if available
        if self.entity_col in df.columns and self.entity_effects:
            entity_vals = (
                df[self.entity_col].map(self.entity_effects).fillna(0.0).to_numpy(dtype=float)
            )
            preds += entity_vals

        # Add time fixed effect lambda_t if available
        if self.time_col in df.columns and self.time_effects:
            time_vals = df[self.time_col].map(self.time_effects).fillna(0.0).to_numpy(dtype=float)
            preds += time_vals

        # Add active linear covariate contributions
        for col in self.active_features:
            b = self.coefficients.get(col, 0.0)
            if col in df.columns:
                val = df[col].fillna(0.0)
                if val.dtype == bool:
                    val = val.astype(float)
                preds += b * val.to_numpy(dtype=float)

        return preds

    def get_elasticity_table(self) -> pd.DataFrame:
        """
        Generate structured DataFrame summarizing coefficients, standard errors,
        t-statistics, p-values, 95% confidence intervals, elasticities, and standardized coefficients.
        """
        records = []
        for feat in self.active_features:
            b = self.coefficients.get(feat, 0.0)
            se = self.standard_errors.get(feat, 0.0)
            t = self.t_stats.get(feat, 0.0)
            p = self.p_values.get(feat, 1.0)
            elas = self.elasticities.get(feat, 0.0)
            std_c = self.std_coefficients.get(feat, 0.0)
            ci_low = self.conf_int_lower.get(feat, b - 1.96 * se)
            ci_high = self.conf_int_upper.get(feat, b + 1.96 * se)

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            records.append(
                {
                    "feature": feat,
                    "coefficient": b,
                    "std_error": se,
                    "t_stat": t,
                    "p_value": p,
                    "ci_lower_95": ci_low,
                    "ci_upper_95": ci_high,
                    "elasticity": elas,
                    "std_coefficient": std_c,
                    "significance": sig,
                }
            )

        for feat in self.absorbed_features:
            records.append(
                {
                    "feature": feat,
                    "coefficient": 0.0,
                    "std_error": 0.0,
                    "t_stat": 0.0,
                    "p_value": 1.0,
                    "ci_lower_95": 0.0,
                    "ci_upper_95": 0.0,
                    "elasticity": 0.0,
                    "std_coefficient": 0.0,
                    "significance": "[absorbed]",
                }
            )

        return pd.DataFrame(records)

    def summary(self) -> str:
        """Format an econometric summary table as an ASCII string."""
        lines = [
            "=" * 78,
            "Two-Way Fixed Effects (TWFE) Panel Regression Baseline",
            f"Dependent Variable: {self.target_col}",
            f"Entity Effect: {self.entity_col} ({self.fitted_summary.get('n_entities', 0)} groups)",
            f"Time Effect:   {self.time_col} ({self.fitted_summary.get('n_time_periods', 0)} periods)",
            f"No. Observations: {self.fitted_summary.get('nobs', 0)} | Covariance: {self.cov_type}",
            f"Overall R²: {self.fitted_summary.get('rsquared', 0.0):.4f} | Within R²: {self.fitted_summary.get('within_rsquared', 0.0):.4f}",
            "-" * 78,
            f"{'Feature':<28} {'Coef':>10} {'Std.Err':>10} {'t-stat':>8} {'P>|t|':>8} {'Elas':>9} {'Sig':>10}",
            "-" * 78,
        ]
        table = self.get_elasticity_table()
        for _, row in table.iterrows():
            lines.append(
                f"{row['feature']:<28} "
                f"{row['coefficient']:>10.4f} "
                f"{row['std_error']:>10.4f} "
                f"{row['t_stat']:>8.2f} "
                f"{row['p_value']:>8.4f} "
                f"{row['elasticity']:>9.4f} "
                f"{row['significance']:>10}"
            )
        lines.append("=" * 78)
        lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
        lines.append("=" * 78)
        return "\n".join(lines)
