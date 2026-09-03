"""
Insurance Underwriting Optimizer - Actuarial Loss Reserving Engine
==================================================================
Implements classical and modern actuarial loss reserving models:
1. Mack Chain Ladder Method (Development Technique)
2. Bornhuetter-Ferguson (BF) Method (Credibility-weighted prior expectation)
3. Cape Cod / Stanard-Bühlmann Method
4. IBNR (Incurred But Not Reported) and Total Reserve Derivations
5. Comparative Actuarial Diagnostics and Capital Reserve Reporting

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
from utils import load_config, setup_logger, get_project_root
from cohort_analysis import LossTriangleAnalyzer

logger = setup_logger("loss_reserving")


class ActuarialReservingEngine:
    """Computes IBNR reserves across classical actuarial reserving methodologies."""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.root = get_project_root()
        self.reports_dir = self.root / self.config["paths"]["reports_dir"]
        self.figures_dir = self.root / self.config["paths"]["figures_dir"]
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = LossTriangleAnalyzer(config_path)
        self.triangle_df = self.analyzer.load_data()
        self.cum_triangle = self.analyzer.build_cumulative_triangle()
        self.dev_summary = self.analyzer.compute_development_factor_summary()

    def get_accident_year_premiums(self) -> pd.Series:
        """Extract net earned premium by accident year."""
        prem_col = self.config["reserving"].get("premium_metric", "EarnedPremNet_")
        ay_prems = self.triangle_df.groupby("AccidentYear")[prem_col].first()
        return ay_prems

    def chain_ladder_reserving(self) -> pd.DataFrame:
        """
        Mack Chain Ladder Reserving:
        Projects future cumulative losses by applying development factors:
        Ultimate_i = Latest_Paid_i * CDF_i
        IBNR_i = Ultimate_i - Latest_Paid_i
        """
        tri = self.cum_triangle.copy()
        n_rows, n_cols = tri.shape
        cdfs = self.dev_summary["CDF_to_Ultimate"].values
        tail_factor = self.config["reserving"].get("tail_factor", 1.002)

        accident_years = tri.index.tolist()
        latest_paid = []
        latest_lag = []
        cdf_applied = []
        cl_ultimate = []

        for i in range(n_rows):
            row = tri.iloc[i].dropna()
            last_lag = row.index[-1]
            last_val = row.values[-1]

            lag_idx = last_lag - 1
            if lag_idx < len(cdfs):
                selected_cdf = cdfs[lag_idx]
            else:
                selected_cdf = tail_factor

            ultimate = last_val * selected_cdf
            latest_paid.append(last_val)
            latest_lag.append(last_lag)
            cdf_applied.append(selected_cdf)
            cl_ultimate.append(ultimate)

        premiums = self.get_accident_year_premiums().reindex(accident_years).values

        df_cl = pd.DataFrame({
            "AccidentYear": accident_years,
            "EarnedPremium": premiums,
            "LatestLag": latest_lag,
            "PaidToDate": latest_paid,
            "CDF": cdf_applied,
            "PercentEmerged": [1.0 / c for c in cdf_applied],
            "CL_Ultimate": cl_ultimate,
            "CL_IBNR": np.array(cl_ultimate) - np.array(latest_paid),
            "CL_LossRatio": np.array(cl_ultimate) / premiums
        })

        return df_cl

    def bornhuetter_ferguson_reserving(self, prior_loss_ratio: float = None) -> pd.DataFrame:
        """
        Bornhuetter-Ferguson Reserving:
        Combines actual emergence to date with prior expected un-emerged losses:
        Ultimate_i = Latest_Paid_i + Premium_i * ELR * (1 - 1 / CDF_i)
        IBNR_i = Premium_i * ELR * (1 - 1 / CDF_i)
        """
        df_cl = self.chain_ladder_reserving()
        if prior_loss_ratio is None:
            prior_loss_ratio = self.config["reserving"].get("selected_loss_ratio", 0.685)

        unemerged_pct = 1.0 - (1.0 / df_cl["CDF"])
        expected_unemerged_loss = df_cl["EarnedPremium"] * prior_loss_ratio * unemerged_pct

        df_bf = df_cl.copy()
        df_bf["PriorLossRatio"] = prior_loss_ratio
        df_bf["BF_IBNR"] = expected_unemerged_loss
        df_bf["BF_Ultimate"] = df_bf["PaidToDate"] + df_bf["BF_IBNR"]
        df_bf["BF_LossRatio"] = df_bf["BF_Ultimate"] / df_bf["EarnedPremium"]

        return df_bf

    def cape_cod_reserving(self) -> pd.DataFrame:
        """
        Cape Cod / Stanard-Bühlmann Method:
        Derives the prior loss ratio endogenously from the triangle's observed emergence:
        ELR_CapeCod = sum(PaidToDate) / sum(EarnedPremium * (1 / CDF))
        """
        df_bf = self.bornhuetter_ferguson_reserving()
        used_premium = df_bf["EarnedPremium"] * df_bf["PercentEmerged"]
        total_paid = df_bf["PaidToDate"].sum()
        total_used_premium = used_premium.sum()

        cape_cod_elr = total_paid / total_used_premium if total_used_premium > 0 else 0.65
        logger.info("Endogenously calculated Cape Cod Expected Loss Ratio: %.4f", cape_cod_elr)

        df_cc = df_bf.copy()
        unemerged_pct = 1.0 - (1.0 / df_cc["CDF"])
        df_cc["CapeCod_ELR"] = cape_cod_elr
        df_cc["CapeCod_IBNR"] = df_cc["EarnedPremium"] * cape_cod_elr * unemerged_pct
        df_cc["CapeCod_Ultimate"] = df_cc["PaidToDate"] + df_cc["CapeCod_IBNR"]
        df_cc["CapeCod_LossRatio"] = df_cc["CapeCod_Ultimate"] / df_cc["EarnedPremium"]

        return df_cc

    def compare_reserving_methods(self) -> pd.DataFrame:
        """Synthesize comparative table across reserving techniques."""
        df_res = self.cape_cod_reserving()

        summary = pd.DataFrame({
            "AccidentYear": df_res["AccidentYear"],
            "EarnedPremium": df_res["EarnedPremium"],
            "PaidToDate": df_res["PaidToDate"],
            "PercentEmerged": df_res["PercentEmerged"],
            "CL_IBNR": df_res["CL_IBNR"],
            "BF_IBNR": df_res["BF_IBNR"],
            "CapeCod_IBNR": df_res["CapeCod_IBNR"],
            "CL_Ultimate": df_res["CL_Ultimate"],
            "BF_Ultimate": df_res["BF_Ultimate"],
            "CapeCod_Ultimate": df_res["CapeCod_Ultimate"],
            "CL_LR": df_res["CL_LossRatio"],
            "BF_LR": df_res["BF_LossRatio"],
            "CapeCod_LR": df_res["CapeCod_LossRatio"]
        })

        out_csv = self.reports_dir / "loss_reserving_summary.csv"
        summary.to_csv(out_csv, index=False)
        logger.info("Exported actuarial reserve summary to: %s", out_csv)
        return summary

    def plot_reserves_comparison(self, save_path: str = None) -> None:
        """Visualize comparative IBNR and Ultimate Loss estimates by method."""
        summary = self.compare_reserving_methods()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. IBNR Reserves by Accident Year
        x = np.arange(len(summary["AccidentYear"]))
        width = 0.25

        axes[0].bar(x - width, summary["CL_IBNR"] / 1e3, width, label="Chain Ladder", color="#1f77b4")
        axes[0].bar(x, summary["BF_IBNR"] / 1e3, width, label="Bornhuetter-Ferguson", color="#ff7f0e")
        axes[0].bar(x + width, summary["CapeCod_IBNR"] / 1e3, width, label="Cape Cod", color="#2ca02c")

        axes[0].set_title("IBNR Reserves by Accident Year & Methodology", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Accident Year", fontsize=10)
        axes[0].set_ylabel("IBNR Reserve ($ Thousands)", fontsize=10)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(summary["AccidentYear"], rotation=45)
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # 2. Implied Ultimate Loss Ratios by Method
        axes[1].plot(summary["AccidentYear"], summary["CL_LR"] * 100, marker="o", linewidth=2, label="Chain Ladder LR%", color="#1f77b4")
        axes[1].plot(summary["AccidentYear"], summary["BF_LR"] * 100, marker="s", linewidth=2, label="Bornhuetter-Ferguson LR%", color="#ff7f0e")
        axes[1].plot(summary["AccidentYear"], summary["CapeCod_LR"] * 100, marker="^", linewidth=2, label="Cape Cod LR%", color="#2ca02c")
        axes[1].axhline(y=65.0, color="gray", linestyle=":", label="Underwriting Target (65%)")

        axes[1].set_title("Implied Ultimate Loss Ratio Comparison", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Accident Year", fontsize=10)
        axes[1].set_ylabel("Ultimate Loss Ratio (%)", fontsize=10)
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        if save_path is None:
            save_path = self.figures_dir / "loss_reserving_comparison.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved reserving comparison figure to: %s", save_path)

    def print_executive_summary(self) -> None:
        """Display actuarial reserve audit table."""
        summary = self.compare_reserving_methods()

        total_paid = summary["PaidToDate"].sum()
        total_prem = summary["EarnedPremium"].sum()

        cl_ibnr_total = summary["CL_IBNR"].sum()
        bf_ibnr_total = summary["BF_IBNR"].sum()
        cc_ibnr_total = summary["CapeCod_IBNR"].sum()

        cl_ult_total = summary["CL_Ultimate"].sum()
        bf_ult_total = summary["BF_Ultimate"].sum()
        cc_ult_total = summary["CapeCod_Ultimate"].sum()

        print("\n" + "=" * 80)
        print("                 ACTUARIAL LOSS RESERVING AUDIT SUMMARY")
        print("=" * 80)
        print(f"Total Cumulative Paid Losses to Date:  ${total_paid:,.2f}")
        print(f"Total Net Earned Premium:             ${total_prem:,.2f}")
        print("-" * 80)
        print(f"Chain Ladder Total IBNR:              ${cl_ibnr_total:,.2f} (Ultimate: ${cl_ult_total:,.2f}, LR: {cl_ult_total/total_prem:.1%})")
        print(f"Bornhuetter-Ferguson Total IBNR:      ${bf_ibnr_total:,.2f} (Ultimate: ${bf_ult_total:,.2f}, LR: {bf_ult_total/total_prem:.1%})")
        print(f"Cape Cod Total IBNR:                  ${cc_ibnr_total:,.2f} (Ultimate: ${cc_ult_total:,.2f}, LR: {cc_ult_total/total_prem:.1%})")
        print("=" * 80 + "\n")


def main():
    reserver = ActuarialReservingEngine()
    reserver.print_executive_summary()
    reserver.plot_reserves_comparison()


if __name__ == "__main__":
    main()
