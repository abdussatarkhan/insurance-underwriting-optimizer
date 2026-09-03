"""
Insurance Underwriting Optimizer - Actuarial GLM Pricing Engine
===============================================================
Implements statistical pricing models for personal lines motor insurance:
1. Tweedie GLM (Compound Poisson-Gamma with power parameter p in (1, 2))
2. Classical Two-Part Frequency-Severity Decomposition:
   - Claim Frequency: Poisson GLM with log(Exposure) offset
   - Claim Severity: Gamma GLM with log link on non-zero claims
3. Out-of-Sample Performance Evaluation:
   - Deviance, Mean Absolute Error, Normalized Gini Coefficient, Lift Curves
4. Production Model Serialization and Coefficient Export

Author: Actuarial Engineering Team
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Tweedie, Poisson, Gamma
from statsmodels.genmod.families.links import Log
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Local utilities
from utils import load_config, setup_logger, get_project_root, gini_coefficient, calculate_actual_vs_expected

logger = setup_logger("pricing_model")


class ActuarialPricingEngine:
    """Trains and compares Tweedie GLM and Two-Part Frequency/Severity models."""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.root = get_project_root()
        self.processed_dir = self.root / self.config["paths"]["processed_data_dir"]
        self.models_dir = self.root / self.config["paths"]["models_dir"]
        self.figures_dir = self.root / self.config["paths"]["figures_dir"]
        self.reports_dir = self.root / self.config["paths"]["reports_dir"]
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.df = self.load_data()
        self.train_df = None
        self.test_df = None
        self.tweedie_model = None
        self.freq_model = None
        self.sev_model = None

    def load_data(self) -> pd.DataFrame:
        """Load cleaned French Motor TPL policy records."""
        parquet_path = self.processed_dir / "french_motor_tpl_clean.parquet"
        csv_path = self.processed_dir / "french_motor_tpl_clean.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("Clean policy data not found. Run scripts/preprocessing.py first.")

        # Ensure types and clean columns
        df["PurePremium"] = df["ClaimAmountCapped"] / df["Exposure"]
        return df

    def prepare_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified or random train/test split per config."""
        test_size = self.config["preprocessing"].get("test_size", 0.20)
        seed = self.config["project"].get("random_seed", 42)

        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=seed)
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        logger.info("Split dataset into Train (%d records) and Test (%d records)", len(train_df), len(test_df))
        return self.train_df, self.test_df

    def fit_tweedie_glm(self) -> Any:
        """
        Fit compound Poisson-Gamma Tweedie GLM directly on PurePremium.
        Tweedie variance power p = 1.55, weights = Exposure.
        """
        if self.train_df is None:
            self.prepare_train_test_split()

        p = self.config["pricing_glm"].get("tweedie_p", 1.55)
        logger.info("Fitting Tweedie GLM (var_power=%.2f, link=log) on training set...", p)

        formula = (
            "PurePremium ~ VehPower + VehAge + DrivAge + BonusMalus + LogDensity "
            "+ C(Area) + C(VehGas)"
        )

        family = Tweedie(var_power=p, link=Log())
        model = smf.glm(
            formula=formula,
            data=self.train_df,
            family=family,
            freq_weights=self.train_df["Exposure"]
        )
        res = model.fit(maxiter=100)
        self.tweedie_model = res

        joblib.dump(res, self.models_dir / "tweedie_pure_premium_model.joblib")
        logger.info("Tweedie GLM converged. Log-Likelihood: %.2f, AIC: %.2f", res.llf, res.aic)
        return res

    def fit_two_part_frequency_severity(self) -> Tuple[Any, Any]:
        """
        Fit classical two-part actuarial model:
        1. Frequency: Poisson GLM on ClaimNbClean with offset=log(Exposure)
        2. Severity: Gamma GLM on ClaimAmountCapped / ClaimNbClean for policies with claims > 0
        """
        if self.train_df is None:
            self.prepare_train_test_split()

        logger.info("Fitting Frequency Poisson GLM with offset=log(Exposure)...")
        freq_formula = (
            "ClaimNbClean ~ VehPower + VehAge + DrivAge + BonusMalus + LogDensity "
            "+ C(Area) + C(VehGas)"
        )
        freq_family = Poisson(link=Log())
        freq_model = smf.glm(
            formula=freq_formula,
            data=self.train_df,
            family=freq_family,
            offset=np.log(self.train_df["Exposure"])
        ).fit(maxiter=100)
        self.freq_model = freq_model

        # Severity model on policies with observed claims
        claim_train = self.train_df[self.train_df["ClaimNbClean"] > 0].copy()
        claim_train["AvgSeverity"] = claim_train["ClaimAmountCapped"] / claim_train["ClaimNbClean"]

        logger.info("Fitting Severity Gamma GLM on %d claims...", len(claim_train))
        sev_formula = (
            "AvgSeverity ~ VehPower + VehAge + DrivAge + BonusMalus + LogDensity "
            "+ C(Area) + C(VehGas)"
        )
        sev_family = Gamma(link=Log())
        sev_model = smf.glm(
            formula=sev_formula,
            data=claim_train,
            family=sev_family,
            freq_weights=claim_train["ClaimNbClean"]
        ).fit(maxiter=100)
        self.sev_model = sev_model

        joblib.dump(freq_model, self.models_dir / "freq_poisson_model.joblib")
        joblib.dump(sev_model, self.models_dir / "sev_gamma_model.joblib")
        logger.info("Two-Part GLM successfully trained and serialized.")
        return freq_model, sev_model

    def evaluate_models_on_test(self) -> pd.DataFrame:
        """Score test dataset and evaluate actuarial lift and Gini metrics."""
        if self.tweedie_model is None:
            self.fit_tweedie_glm()
        if self.freq_model is None:
            self.fit_two_part_frequency_severity()

        test = self.test_df.copy()

        # Tweedie prediction: Expected pure premium per unit of exposure
        test["Pred_PurePremium_Tweedie"] = self.tweedie_model.predict(test)
        test["Pred_Loss_Tweedie"] = test["Pred_PurePremium_Tweedie"] * test["Exposure"]

        # Two-Part prediction: (Expected Annual Frequency) * (Expected Severity) * Exposure
        pred_freq_annual = self.freq_model.predict(test.assign(Exposure=1.0))
        pred_sev = self.sev_model.predict(test)
        test["Pred_PurePremium_TwoPart"] = pred_freq_annual * pred_sev
        test["Pred_Loss_TwoPart"] = test["Pred_PurePremium_TwoPart"] * test["Exposure"]

        actual_loss = test["ClaimAmountCapped"]
        exposure = test["Exposure"]

        # Gini Coefficients
        gini_tweedie = gini_coefficient(actual_loss, test["Pred_Loss_Tweedie"], exposure=exposure)
        gini_twopart = gini_coefficient(actual_loss, test["Pred_Loss_TwoPart"], exposure=exposure)

        # Actual vs Expected
        ae_tweedie = calculate_actual_vs_expected(actual_loss, test["Pred_Loss_Tweedie"])
        ae_twopart = calculate_actual_vs_expected(actual_loss, test["Pred_Loss_TwoPart"])

        # Mean Absolute Error
        mae_tweedie = np.average(np.abs(actual_loss - test["Pred_Loss_Tweedie"]), weights=exposure)
        mae_twopart = np.average(np.abs(actual_loss - test["Pred_Loss_TwoPart"]), weights=exposure)

        eval_summary = pd.DataFrame({
            "Metric": ["Normalized Gini", "Actual-to-Expected (A/E)", "Weighted MAE (€)", "AIC"],
            "Tweedie_GLM": [gini_tweedie, ae_tweedie, mae_tweedie, self.tweedie_model.aic],
            "Two_Part_FreqSev": [gini_twopart, ae_twopart, mae_twopart, self.freq_model.aic + self.sev_model.aic]
        })

        out_csv = self.reports_dir / "pricing_model_evaluation.csv"
        eval_summary.to_csv(out_csv, index=False)
        self.test_df = test
        return eval_summary

    def plot_pricing_diagnostics(self) -> None:
        """Plot Decile Lift Chart and Lorenz Curves."""
        if "Pred_PurePremium_Tweedie" not in self.test_df.columns:
            self.evaluate_models_on_test()

        test = self.test_df.copy()

        # Create deciles by predicted Tweedie pure premium
        test["Decile"] = pd.qcut(test["Pred_PurePremium_Tweedie"], q=10, labels=False) + 1
        decile_stats = test.groupby("Decile").agg(
            Exposure=("Exposure", "sum"),
            ActualLoss=("ClaimAmountCapped", "sum"),
            PredLoss=("Pred_Loss_Tweedie", "sum")
        ).reset_index()

        decile_stats["Actual_PurePrem"] = decile_stats["ActualLoss"] / decile_stats["Exposure"]
        decile_stats["Pred_PurePrem"] = decile_stats["PredLoss"] / decile_stats["Exposure"]
        overall_pure_prem = test["ClaimAmountCapped"].sum() / test["Exposure"].sum()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Decile Lift Chart
        width = 0.35
        x = decile_stats["Decile"]
        axes[0].bar(x - width/2, decile_stats["Actual_PurePrem"], width, label="Observed Pure Premium", color="#1f77b4")
        axes[0].bar(x + width/2, decile_stats["Pred_PurePrem"], width, label="Tweedie Predicted Pure Premium", color="#ff7f0e", alpha=0.85)
        axes[0].axhline(y=overall_pure_prem, color="red", linestyle="--", label="Portfolio Average (€)")
        axes[0].set_title("10-Decile Pricing Lift Chart (Risk Segmentation)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Risk Decile (1 = Safest, 10 = Riskiest)", fontsize=10)
        axes[0].set_ylabel("Pure Premium (€ / Policy Year)", fontsize=10)
        axes[0].set_xticks(x)
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle=":", alpha=0.5)

        # 2. Lorenz Curves
        order = np.argsort(test["Pred_Loss_Tweedie"].values)
        ordered_exp = test["Exposure"].values[order]
        ordered_actual = test["ClaimAmountCapped"].values[order]

        cum_exp = np.cumsum(ordered_exp) / np.sum(ordered_exp)
        cum_loss = np.cumsum(ordered_actual) / np.sum(ordered_actual)

        cum_exp = np.insert(cum_exp, 0, 0.0)
        cum_loss = np.insert(cum_loss, 0, 0.0)

        axes[1].plot(cum_exp, cum_loss, label="Tweedie GLM Lorenz Curve", color="#2b5c8f", linewidth=2.2)
        axes[1].plot([0, 1], [0, 1], label="Line of Equality (Random)", color="gray", linestyle="--")
        axes[1].set_title("Lorenz Curve of Risk Differentiation", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Cumulative Share of Exposure (Policies)", fontsize=10)
        axes[1].set_ylabel("Cumulative Share of Losses (€)", fontsize=10)
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        save_path = self.figures_dir / "pricing_model_performance.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved pricing diagnostic visual to: %s", save_path)

    def print_model_report(self) -> None:
        """Print executive evaluation table and regression coefficients."""
        summary = self.evaluate_models_on_test()
        print("\n" + "=" * 70)
        print("          ACTUARIAL PRICING MODEL VALIDATION (TEST SET)")
        print("=" * 70)
        print(summary.to_string(index=False))
        print("-" * 70)
        print("Tweedie GLM Significant Relativities (Exponentiated Betas):")
        params = np.exp(self.tweedie_model.params)
        for name, val in params.items():
            print(f"  {name:25s}: {val:.4f}")
        print("=" * 70 + "\n")


def main():
    engine = ActuarialPricingEngine()
    engine.fit_tweedie_glm()
    engine.fit_two_part_frequency_severity()
    engine.evaluate_models_on_test()
    engine.plot_pricing_diagnostics()
    engine.print_model_report()


if __name__ == "__main__":
    main()
