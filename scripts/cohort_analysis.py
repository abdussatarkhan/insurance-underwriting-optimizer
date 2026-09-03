"""
Insurance Underwriting Optimizer - Actuarial Cohort & Triangle Analysis
=======================================================================
Constructs and evaluates loss development triangles:
1. Pivots long-format loss transactions into accident year by development lag matrices
2. Calculates empirical link ratios (age-to-age development factors)
3. Computes volume-weighted, simple arithmetic, and medians across development periods
4. Derives Cumulative Development Factors (CDF to Ultimate)
5. Generates actuarial development curve visualizations

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

logger = setup_logger("cohort_analysis")


class LossTriangleAnalyzer:
    """Performs empirical development analysis on policy accident year cohorts."""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.root = get_project_root()
        self.processed_dir = self.root / self.config["paths"]["processed_data_dir"]
        self.figures_dir = self.root / self.config["paths"]["figures_dir"]
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.triangle_df = None
        self.loss_triangle = None

    def load_data(self) -> pd.DataFrame:
        """Load processed CAS Schedule P loss data."""
        parquet_path = self.processed_dir / "loss_development_triangles.parquet"
        csv_path = self.processed_dir / "loss_development_triangles.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("Processed loss development data not found. Run scripts/preprocessing.py first.")

        # Aggregate across industry or select primary benchmark carrier
        if "GRCODE" in df.columns:
            # If multiple carriers, aggregate to industry level or pick modal carrier
            target_carrier = df["GRCODE"].value_counts().index[0]
            df_subset = df[df["GRCODE"] == target_carrier].copy()
            logger.info("Selected benchmark carrier GRCODE=%s (%d records)", target_carrier, len(df_subset))
            self.triangle_df = df_subset
        else:
            self.triangle_df = df
        return self.triangle_df

    def build_cumulative_triangle(self, value_col: str = "CumPaidLoss_") -> pd.DataFrame:
        """
        Pivot long data into upper triangular matrix:
        Rows = Accident Year (AY), Columns = Development Lag (in years)
        """
        if self.triangle_df is None:
            self.load_data()

        logger.info("Constructing cumulative loss triangle on column: %s", value_col)
        triangle = self.triangle_df.pivot(
            index="AccidentYear",
            columns="DevelopmentLag",
            values=value_col
        )
        # Ensure rows and cols are sorted
        triangle = triangle.sort_index(axis=0).sort_index(axis=1)
        self.loss_triangle = triangle
        return triangle

    def compute_age_to_age_factors(self) -> pd.DataFrame:
        """
        Compute empirical link ratios f_{i, j} = C_{i, j+1} / C_{i, j}.
        """
        if self.loss_triangle is None:
            self.build_cumulative_triangle()

        tri = self.loss_triangle.values
        n_rows, n_cols = tri.shape
        link_ratios = np.full((n_rows, n_cols - 1), np.nan)

        for i in range(n_rows):
            for j in range(n_cols - 1):
                c_curr = tri[i, j]
                c_next = tri[i, j + 1]
                if not np.isnan(c_curr) and not np.isnan(c_next) and c_curr > 0:
                    link_ratios[i, j] = c_next / c_curr

        cols = [f"{j}-{j+1}" for j in range(1, n_cols)]
        df_link = pd.DataFrame(link_ratios, index=self.loss_triangle.index, columns=cols)
        return df_link

    def compute_development_factor_summary(self) -> pd.DataFrame:
        """
        Compute standard actuarial link ratio averages:
        - Volume-weighted average (Chain Ladder standard)
        - Simple arithmetic average
        - Latest 3-year weighted average
        - Selected Factors and CDF to Ultimate
        """
        if self.loss_triangle is None:
            self.build_cumulative_triangle()

        tri = self.loss_triangle.values
        n_rows, n_cols = tri.shape
        cols = [f"Lag_{j}_to_{j+1}" for j in range(1, n_cols)]

        volume_weighted = []
        simple_avg = []
        med_avg = []

        for j in range(n_cols - 1):
            curr_vals = []
            next_vals = []
            ratios = []
            for i in range(n_rows):
                c_curr = tri[i, j]
                c_next = tri[i, j + 1]
                if not np.isnan(c_curr) and not np.isnan(c_next) and c_curr > 0:
                    curr_vals.append(c_curr)
                    next_vals.append(c_next)
                    ratios.append(c_next / c_curr)

            if len(curr_vals) > 0 and sum(curr_vals) > 0:
                vw = sum(next_vals) / sum(curr_vals)
                sa = np.mean(ratios)
                med = np.median(ratios)
            else:
                vw = 1.0
                sa = 1.0
                med = 1.0

            volume_weighted.append(vw)
            simple_avg.append(sa)
            med_avg.append(med)

        summary_df = pd.DataFrame({
            "Development_Step": cols,
            "Volume_Weighted": volume_weighted,
            "Simple_Average": simple_avg,
            "Median": med_avg
        })

        # Selected factors (actuary picks volume-weighted by default)
        summary_df["Selected_Factor"] = summary_df["Volume_Weighted"]

        # Tail factor to ultimate
        tail_factor = self.config["reserving"].get("tail_factor", 1.002)

        # Cumulative Development Factor (CDF) backwards from tail
        factors = summary_df["Selected_Factor"].tolist()
        cdfs = []
        running_cdf = tail_factor
        for f in reversed(factors):
            running_cdf *= f
            cdfs.append(running_cdf)
        cdfs.reverse()

        summary_df["CDF_to_Ultimate"] = cdfs
        summary_df["Percent_Emerged"] = 1.0 / summary_df["CDF_to_Ultimate"]

        return summary_df

    def plot_development_patterns(self, save_path: str = None) -> None:
        """Generate publication-ready visualization of loss development trends."""
        if self.loss_triangle is None:
            self.build_cumulative_triangle()

        summary = self.compute_development_factor_summary()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Cumulative Paid Loss Trajectories by Accident Year
        palette = sns.color_palette("viridis", n_colors=len(self.loss_triangle))
        for idx, (ay, row) in enumerate(self.loss_triangle.iterrows()):
            valid_mask = ~row.isna()
            axes[0].plot(
                row.index[valid_mask],
                row.values[valid_mask] / 1e3,
                marker="o",
                label=f"AY {ay}",
                color=palette[idx],
                linewidth=1.8
            )

        axes[0].set_title("Cumulative Loss Development Trajectory by Accident Year", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Development Lag (Years)", fontsize=10)
        axes[0].set_ylabel("Cumulative Paid Losses ($ Thousands)", fontsize=10)
        axes[0].legend(ncol=2, fontsize=8, loc="lower right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. Link Ratios and Cumulative Emergence Curve
        lags = [int(step.split("_")[1]) for step in summary["Development_Step"]]
        emergence_pct = summary["Percent_Emerged"] * 100.0

        ax2 = axes[1]
        ax2.bar(lags, summary["Selected_Factor"] - 1.0, color="#2b5c8f", alpha=0.7, width=0.5, label="Incremental Factor (f - 1)")
        ax2.set_xlabel("Development Lag (Years)", fontsize=10)
        ax2.set_ylabel("Factor Increment (f - 1.0)", fontsize=10, color="#2b5c8f")
        ax2.tick_params(axis="y", labelcolor="#2b5c8f")
        ax2.grid(True, linestyle=":", alpha=0.4)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(lags, emergence_pct, color="#d9534f", marker="s", linewidth=2.2, label="% Ultimate Emerged")
        ax2_twin.set_ylabel("% of Ultimate Losses Emerged", fontsize=10, color="#d9534f")
        ax2_twin.tick_params(axis="y", labelcolor="#d9534f")
        ax2_twin.set_ylim(20, 105)

        axes[1].set_title("Age-to-Age Development Factors & Cumulative Emergence", fontsize=12, fontweight="bold")

        plt.tight_layout()
        if save_path is None:
            save_path = self.figures_dir / "loss_development_patterns.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved cohort development figure to: %s", save_path)

    def print_triangle_report(self) -> None:
        """Display formatted triangle and development factors."""
        tri = self.build_cumulative_triangle()
        link_df = self.compute_age_to_age_factors()
        summary = self.compute_development_factor_summary()

        print("\n" + "=" * 75)
        print("          ACTUARIAL CUMULATIVE LOSS TRIANGLE (C_{i, j})")
        print("=" * 75)
        print(tri.round(1).to_string())

        print("\n" + "=" * 75)
        print("          EMPIRICAL AGE-TO-AGE LINK RATIOS (f_{i, j})")
        print("=" * 75)
        print(link_df.round(3).to_string())

        print("\n" + "=" * 75)
        print("          SELECTED DEVELOPMENT FACTORS & CDF TO ULTIMATE")
        print("=" * 75)
        print(summary.to_string(index=False))
        print("=" * 75 + "\n")


def main():
    analyzer = LossTriangleAnalyzer()
    analyzer.print_triangle_report()
    analyzer.plot_development_patterns()


if __name__ == "__main__":
    main()
