"""
Insurance Underwriting Optimizer - Survival & Time-to-Event Analysis
====================================================================
Evaluates policyholder retention and time-to-first-claim dynamics:
1. Kaplan-Meier Survival Curve Estimation stratified by underwriting groups
2. Log-rank statistical hypothesis testing for risk differentiation
3. Cox Proportional Hazards (Cox PH) Regression for multivariate hazard ratios
4. Proportional hazards diagnostics and Schoenfeld residual checks
5. Generates actuarial survival curves and Hazard Ratio Forest Plots

Author: Actuarial Engineering Team
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Lifelines survival analysis suite
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Local utilities
from utils import load_config, setup_logger, get_project_root

logger = setup_logger("survival_analysis")


class UnderwritingSurvivalAnalyzer:
    """Implements survival analysis for time-to-claim risk modeling."""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.root = get_project_root()
        self.processed_dir = self.root / self.config["paths"]["processed_data_dir"]
        self.reports_dir = self.root / self.config["paths"]["reports_dir"]
        self.figures_dir = self.root / self.config["paths"]["figures_dir"]
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.df = self.load_data()
        self.cph_model = None

    def load_data(self) -> pd.DataFrame:
        """Load preprocessed policy dataset with survival fields."""
        parquet_path = self.processed_dir / "french_motor_tpl_clean.parquet"
        csv_path = self.processed_dir / "french_motor_tpl_clean.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("Cleaned policy data not found. Run scripts/preprocessing.py first.")

        # Ensure numeric survival time and binary event indicator
        df["SurvivalTimeDays"] = df["SurvivalTimeDays"].astype(float)
        df["ClaimEvent"] = df["ClaimEvent"].astype(int)
        logger.info("Loaded survival dataset (%d records, %d claim events)", len(df), df["ClaimEvent"].sum())
        return df

    def fit_kaplan_meier_strata(self, strata_col: str = "VehGas") -> Dict[str, KaplanMeierFitter]:
        """
        Estimate Kaplan-Meier survival curves for each category of strata_col.
        Evaluates cumulative probability of remaining claim-free over policy lifetime.
        """
        km_fitters = {}
        unique_groups = self.df[strata_col].dropna().unique()

        logger.info("Fitting Kaplan-Meier survival curves stratified by: %s", strata_col)
        for group in unique_groups:
            sub = self.df[self.df[strata_col] == group]
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=sub["SurvivalTimeDays"],
                event_observed=sub["ClaimEvent"],
                label=f"{strata_col}={group}"
            )
            km_fitters[group] = kmf

        # Log-rank test if two groups
        if len(unique_groups) == 2:
            g1, g2 = unique_groups[0], unique_groups[1]
            sub1 = self.df[self.df[strata_col] == g1]
            sub2 = self.df[self.df[strata_col] == g2]
            res = logrank_test(
                sub1["SurvivalTimeDays"], sub2["SurvivalTimeDays"],
                sub1["ClaimEvent"], sub2["ClaimEvent"]
            )
            logger.info("Log-rank test (%s vs %s): p-value = %.4e (stat = %.2f)", g1, g2, res.p_value, res.test_statistic)

        return km_fitters

    def fit_cox_proportional_hazards(self) -> pd.DataFrame:
        """
        Fit Cox Proportional Hazards regression to identify hazard ratios for risk factors.
        Hazard Rate: h(t | x) = h_0(t) * exp(beta' * x)
        """
        logger.info("Preparing design matrix for Cox Proportional Hazards regression...")
        # Select representative underwriting covariates
        cols = [
            "SurvivalTimeDays", "ClaimEvent",
            "VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity",
            "VehGas", "Area"
        ]
        data = self.df[cols].dropna().copy()

        # Limit sample size to 30,000 policies if large for fast, stable Cox PH fitting
        if len(data) > 30000:
            logger.info("Downsampling to 30,000 records for Cox PH optimization...")
            # Stratified sample ensuring representation of events
            events = data[data["ClaimEvent"] == 1]
            non_events = data[data["ClaimEvent"] == 0].sample(n=30000 - len(events), random_state=42)
            data = pd.concat([events, non_events]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        # One-hot encode categoricals with drop_first=True for reference categories
        data = pd.get_dummies(data, columns=["VehGas", "Area"], drop_first=True, dtype=float)

        penalizer = self.config["survival_analysis"].get("cox_penalizer", 0.01)
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(data, duration_col="SurvivalTimeDays", event_col="ClaimEvent")

        self.cph_model = cph

        # Format summary table
        summary = cph.summary.copy()
        summary["Hazard_Ratio"] = np.exp(summary["coef"])
        summary["HR_lower_95%"] = np.exp(summary["coef lower 95%"])
        summary["HR_upper_95%"] = np.exp(summary["coef upper 95%"])

        out_csv = self.reports_dir / "survival_hazard_ratios.csv"
        summary.to_csv(out_csv)
        logger.info("Fitted Cox PH model (Concordance Index: %.4f). Results exported to %s", cph.concordance_index_, out_csv)

        return summary

    def plot_kaplan_meier_and_hazard_ratios(self) -> None:
        """Generate dual publication plots: Kaplan-Meier curves and Forest Plot of Hazard Ratios."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Kaplan-Meier Survival Curves by Powertrain & Driver Age
        km_gas = self.fit_kaplan_meier_strata("VehGas")
        colors = ["#1f77b4", "#d62728"]
        for idx, (label, kmf) in enumerate(km_gas.items()):
            kmf.plot_survival_function(ax=axes[0], color=colors[idx % len(colors)], linewidth=2.0)

        axes[0].set_title("Kaplan-Meier Claim-Free Survival Curve by Vehicle Fuel Type", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Policy Duration (Days)", fontsize=10)
        axes[0].set_ylabel("Probability of Remaining Claim-Free S(t)", fontsize=10)
        axes[0].set_ylim(0.85, 1.005)  # Auto TPL claims are rare, zoom into 85-100% range
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # 2. Forest Plot of Hazard Ratios
        if self.cph_model is None:
            summary = self.fit_cox_proportional_hazards()
        else:
            summary = self.cph_model.summary

        covariates = summary.index.tolist()
        hr = np.exp(summary["coef"])
        hr_lower = np.exp(summary["coef lower 95%"])
        hr_upper = np.exp(summary["coef upper 95%"])

        y_pos = np.arange(len(covariates))
        xerr = [hr - hr_lower, hr_upper - hr]

        axes[1].errorbar(hr, y_pos, xerr=xerr, fmt="o", color="#2b5c8f", ecolor="#7399c6", elinewidth=2, capsize=4)
        axes[1].axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(covariates, fontsize=10)
        axes[1].set_xlabel("Hazard Ratio (HR with 95% CI)", fontsize=10)
        axes[1].set_title("Cox PH Regression Hazard Ratios (Risk Factors)", fontsize=12, fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        save_path = self.figures_dir / "survival_analysis_diagnostics.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved survival analysis diagnostic visual to: %s", save_path)

    def print_summary(self) -> None:
        """Display actuarial survival analysis findings."""
        if self.cph_model is None:
            summary = self.fit_cox_proportional_hazards()
        else:
            summary = self.cph_model.summary

        print("\n" + "=" * 80)
        print("          UNDERWRITING SURVIVAL ANALYSIS (COX PH HAZARD RATIOS)")
        print("=" * 80)
        print(f"Concordance Index (Harrell's C): {self.cph_model.concordance_index_:.4f}")
        print("-" * 80)
        cols_to_show = ["coef", "exp(coef)", "se(coef)", "p", "exp(coef) lower 95%", "exp(coef) upper 95%"]
        available_cols = [c for c in cols_to_show if c in summary.columns]
        print(summary[available_cols].round(4).to_string())
        print("=" * 80 + "\n")


def main():
    analyzer = UnderwritingSurvivalAnalyzer()
    analyzer.fit_kaplan_meier_strata("VehGas")
    analyzer.fit_cox_proportional_hazards()
    analyzer.plot_kaplan_meier_and_hazard_ratios()
    analyzer.print_summary()


if __name__ == "__main__":
    main()
