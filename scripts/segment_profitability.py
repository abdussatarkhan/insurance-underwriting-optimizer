"""
Insurance Underwriting Optimizer - Segment Profitability & Rate Adequacy Engine
==============================================================================
Conducts portfolio loss ratio decomposition and underwriting optimization:
1. Multi-dimensional risk cohort segmentation (Driver Age x Bonus-Malus x Vehicle Power)
2. Non-parametric Bootstrap Confidence Intervals for cohort loss ratios
3. Flags the worst 20% unprofitable underwriting sub-segments
4. Actuarial Rate Adequacy Indications:
   Indicated Rate Change = (Observed LR / Target LR) - 1.0
5. Actionable underwriting recommendations: rate surcharges, non-renewal, or deductible adjustments

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

# Local utilities
from utils import load_config, setup_logger, get_project_root, calculate_loss_ratio

logger = setup_logger("segment_profitability")


class SegmentProfitabilityOptimizer:
    """Analyzes segment combined ratios and derives rate indication remedies."""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.root = get_project_root()
        self.processed_dir = self.root / self.config["paths"]["processed_data_dir"]
        self.reports_dir = self.root / self.config["paths"]["reports_dir"]
        self.figures_dir = self.root / self.config["paths"]["figures_dir"]
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.df = self.load_data()
        self.target_lr = self.config["profitability"].get("target_loss_ratio", 0.65)
        self.expense_ratio = self.config["profitability"].get("expense_ratio", 0.25)
        self.breakeven_lr = self.config["profitability"].get("breakeven_loss_ratio", 0.75)

    def load_data(self) -> pd.DataFrame:
        """Load preprocessed French Motor TPL dataset."""
        parquet_path = self.processed_dir / "french_motor_tpl_clean.parquet"
        csv_path = self.processed_dir / "french_motor_tpl_clean.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("Cleaned policy data not found. Run scripts/preprocessing.py first.")

        return df

    def compute_bootstrap_ci(
        self,
        losses: np.ndarray,
        premiums: np.ndarray,
        n_boot: int = 1000,
        ci_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute non-parametric bootstrap confidence interval for aggregate loss ratio.
        """
        n_obs = len(losses)
        if n_obs < 15 or np.sum(premiums) <= 0:
            point_est = np.sum(losses) / np.sum(premiums) if np.sum(premiums) > 0 else np.nan
            return point_est, point_est, point_est

        point_est = np.sum(losses) / np.sum(premiums)
        np.random.seed(42)
        indices = np.random.randint(0, n_obs, size=(n_boot, n_obs))

        # Vectorized bootstrap calculation
        boot_losses = np.take(losses, indices).sum(axis=1)
        boot_premiums = np.take(premiums, indices).sum(axis=1)
        boot_ratios = boot_losses / np.maximum(1e-6, boot_premiums)

        alpha = (1.0 - ci_level) / 2.0
        lower_ci = np.percentile(boot_ratios, alpha * 100)
        upper_ci = np.percentile(boot_ratios, (1.0 - alpha) * 100)

        return float(point_est), float(lower_ci), float(upper_ci)

    def analyze_underwriting_segments(self) -> pd.DataFrame:
        """
        Segment portfolio by core underwriting risk cells:
        [DrivAgeGroup, BonusMalusClass, VehGas]
        """
        logger.info("Computing segment profitability across underwriting cohorts...")
        n_boot = self.config["profitability"].get("bootstrap_replicates", 500)
        ci_level = self.config["profitability"].get("confidence_interval", 0.95)

        segment_cols = ["DrivAgeGroup", "BonusMalusClass", "VehGas"]
        grouped = self.df.groupby(segment_cols, observed=False)

        segment_rows = []
        for segment_keys, group in grouped:
            if len(group) == 0:
                continue

            losses = group["ClaimAmountCapped"].values
            premiums = group["EarnedPremium"].values
            exposure = group["Exposure"].sum()
            claim_counts = group["ClaimNbClean"].sum()

            lr, lr_lower, lr_upper = self.compute_bootstrap_ci(
                losses, premiums, n_boot=n_boot, ci_level=ci_level
            )

            # Combined Operating Ratio = Loss Ratio + Expense Ratio
            combined_ratio = lr + self.expense_ratio

            # Indicated Rate Change = (LR / Target LR) - 1.0
            indicated_rate_change = (lr / self.target_lr) - 1.0

            segment_name = f"{segment_keys[0]} | {segment_keys[1]} | {segment_keys[2]}"

            segment_rows.append({
                "Segment": segment_name,
                "DrivAgeGroup": str(segment_keys[0]),
                "BonusMalusClass": str(segment_keys[1]),
                "VehGas": str(segment_keys[2]),
                "Policies": len(group),
                "EarnedExposure": round(exposure, 2),
                "TotalEarnedPrem": round(np.sum(premiums), 2),
                "TotalIncurredLoss": round(np.sum(losses), 2),
                "ClaimCount": int(claim_counts),
                "FrequencyPerYear": round(claim_counts / max(0.01, exposure), 4),
                "LossRatio": round(lr, 4),
                "LR_CI_Lower": round(lr_lower, 4),
                "LR_CI_Upper": round(lr_upper, 4),
                "CombinedRatio": round(combined_ratio, 4),
                "IndicatedRateChange": round(indicated_rate_change, 4)
            })

        df_seg = pd.DataFrame(segment_rows)
        # Filter cohorts with reasonable statistical credibility (e.g. >= 20 policies)
        df_seg = df_seg[df_seg["Policies"] >= 20].sort_values("LossRatio", ascending=False).reset_index(drop=True)

        # Identify worst 20% segments by loss ratio
        worst_cutoff = int(np.ceil(len(df_seg) * self.config["profitability"].get("worst_segment_percentile", 0.20)))
        df_seg["SegmentRank"] = np.arange(1, len(df_seg) + 1)
        df_seg["IsWorst20Pct"] = df_seg["SegmentRank"] <= worst_cutoff

        # Classify Underwriting Action
        def assign_action(row):
            if row["LossRatio"] > 1.20:
                return "Non-Renew / Strict Underwriting Exclusion"
            elif row["LossRatio"] > self.breakeven_lr:
                return f"Apply Rate Surcharge (+{min(0.25, row['IndicatedRateChange']):.1%}) & Higher Deductible"
            elif row["LossRatio"] < 0.50:
                return "Prime Growth Segment - Expand Marketing"
            else:
                return "Maintain Current Rate Level (Adequate)"

        df_seg["RecommendedAction"] = df_seg.apply(assign_action, axis=1)

        out_csv = self.reports_dir / "segment_profitability_audit.csv"
        df_seg.to_csv(out_csv, index=False)
        logger.info("Saved segment profitability audit (%d cohorts) to %s", len(df_seg), out_csv)

        return df_seg

    def plot_profitability_landscape(self) -> None:
        """Visualize segment loss ratios with bootstrap confidence intervals and action classes."""
        df_seg = self.analyze_underwriting_segments()
        top_cohorts = pd.concat([df_seg.head(10), df_seg.tail(10)]).reset_index(drop=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # 1. Segment Loss Ratios with 95% Bootstrap Error Bars
        y_pos = np.arange(len(top_cohorts))
        lrs = top_cohorts["LossRatio"] * 100
        err_lower = (top_cohorts["LossRatio"] - top_cohorts["LR_CI_Lower"]) * 100
        err_upper = (top_cohorts["LR_CI_Upper"] - top_cohorts["LossRatio"]) * 100
        xerr = [np.maximum(0, err_lower), np.maximum(0, err_upper)]

        colors = ["#d9534f" if is_w else "#2b5c8f" for is_w in top_cohorts["IsWorst20Pct"]]

        axes[0].errorbar(lrs, y_pos, xerr=xerr, fmt="o", color="black", ecolor=colors, elinewidth=2, capsize=4)
        axes[0].scatter(lrs, y_pos, c=colors, s=70, zorder=3)
        axes[0].axvline(x=self.target_lr * 100, color="green", linestyle="--", linewidth=1.5, label=f"Target LR ({self.target_lr:.0%})")
        axes[0].axvline(x=self.breakeven_lr * 100, color="orange", linestyle=":", linewidth=1.5, label=f"Breakeven LR ({self.breakeven_lr:.0%})")

        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(top_cohorts["Segment"], fontsize=9)
        axes[0].set_xlabel("Segment Loss Ratio (%) with 95% Bootstrap CI", fontsize=10)
        axes[0].set_title("Top 10 Highest vs Lowest Loss Ratio Cohorts", fontsize=12, fontweight="bold")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, linestyle=":", alpha=0.4)

        # 2. Indicated Rate Change Distribution & Cumulative Impact
        sns.histplot(df_seg["IndicatedRateChange"] * 100, bins=25, kde=True, ax=axes[1], color="#2b5c8f")
        axes[1].axvline(x=0.0, color="black", linestyle="-", linewidth=1.2)
        axes[1].axvline(x=15.0, color="red", linestyle="--", label="+15% Rate Surcharge Line")
        axes[1].set_title("Distribution of Actuarially Indicated Rate Adjustments", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Indicated Rate Change (%)", fontsize=10)
        axes[1].set_ylabel("Number of Cohorts", fontsize=10)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle=":", alpha=0.4)

        plt.tight_layout()
        save_path = self.figures_dir / "segment_profitability.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved segment profitability visual to: %s", save_path)

    def print_executive_actions(self) -> None:
        """Display summary of worst 20% segments and recommended rate revisions."""
        df_seg = self.analyze_underwriting_segments()
        worst = df_seg[df_seg["IsWorst20Pct"]]

        worst_prem = worst["TotalEarnedPrem"].sum()
        worst_loss = worst["TotalIncurredLoss"].sum()
        worst_lr = worst_loss / worst_prem

        print("\n" + "=" * 85)
        print("          UNDERWRITING OPTIMIZATION: WORST 20% SEGMENT REMEDIATION")
        print("=" * 85)
        print(f"Total Cohorts Evaluated:          {len(df_seg)}")
        print(f"Worst 20% Cohort Count:           {len(worst)}")
        print(f"Worst 20% Earned Premium:         €{worst_prem:,.2f} ({worst_prem/df_seg['TotalEarnedPrem'].sum():.1%} of portfolio)")
        print(f"Worst 20% Incurred Losses:        €{worst_loss:,.2f} ({worst_loss/df_seg['TotalIncurredLoss'].sum():.1%} of total losses)")
        print(f"Worst 20% Blended Loss Ratio:     {worst_lr:.2%} (vs Target {self.target_lr:.0%})")
        print("-" * 85)
        print("Top 5 Critical Action Segments:")
        print(worst[["Segment", "Policies", "LossRatio", "CombinedRatio", "IndicatedRateChange", "RecommendedAction"]].head(5).to_string(index=False))
        print("=" * 85 + "\n")


def main():
    optimizer = SegmentProfitabilityOptimizer()
    optimizer.print_executive_actions()
    optimizer.plot_profitability_landscape()


if __name__ == "__main__":
    main()
