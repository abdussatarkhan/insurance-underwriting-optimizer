"""
Insurance Underwriting Optimizer - Actuarial Preprocessing Pipeline
===================================================================
Prepares and cleans insurance data for GLM pricing, survival analysis,
and loss reserving:
1. Merges policy records with claim severities
2. Implements right-censoring time-to-event representations
3. Standardizes exposure limits and handles catastrophic claim truncation
4. Converts and validates CAS Schedule P loss triangles
5. Exports processed actuarial modeling datasets

Author: Actuarial Engineering Team
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Local utilities
from utils import load_config, setup_logger, get_project_root, generate_synthetic_french_tpl, generate_synthetic_cas_triangle

logger = setup_logger("preprocessing")


class ActuarialPreprocessor:
    """Preprocesses policyholder and reserving data according to actuarial conventions."""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.root = get_project_root()
        self.raw_dir = self.root / self.config["paths"]["raw_data_dir"]
        self.processed_dir = self.root / self.config["paths"]["processed_data_dir"]
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_policy_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw frequency and severity datasets."""
        freq_path = self.raw_dir / self.config["datasets"]["openml_freMTPL2"]["freq_filename"]
        sev_path = self.raw_dir / self.config["datasets"]["openml_freMTPL2"]["sev_filename"]

        if not freq_path.exists() or not sev_path.exists():
            logger.warning("Raw files not detected. Generating synthetic test data for preprocessing...")
            df_freq, df_sev = generate_synthetic_french_tpl(n_samples=25000, random_seed=42)
            df_freq.to_csv(freq_path, index=False)
            df_sev.to_csv(sev_path, index=False)
        else:
            df_freq = pd.read_csv(freq_path)
            df_sev = pd.read_csv(sev_path)

        return df_freq, df_sev

    def process_french_tpl(self) -> pd.DataFrame:
        """
        Merge freMTPL2 frequency & severity, apply actuarial caps,
        compute pure premium, and format survival analysis fields.
        """
        logger.info("Starting French Motor TPL preprocessing...")
        df_freq, df_sev = self.load_raw_policy_data()

        # Clean column types
        df_freq["IDpol"] = df_freq["IDpol"].astype(int)
        df_sev["IDpol"] = df_sev["IDpol"].astype(int)

        # Aggregate claim amounts per policy (sum individual claims for the policy year)
        logger.info("Aggregating individual claims to policy level...")
        sev_agg = df_sev.groupby("IDpol").agg(
            TotalClaimAmount=("ClaimAmount", "sum"),
            ClaimCount=("ClaimAmount", "count"),
            MaxClaimAmount=("ClaimAmount", "max")
        ).reset_index()

        # Merge with frequency data
        df = pd.merge(df_freq, sev_agg, on="IDpol", how="left")
        df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0.0)
        df["ClaimCount"] = df["ClaimCount"].fillna(0).astype(int)
        df["MaxClaimAmount"] = df["MaxClaimAmount"].fillna(0.0)

        # Reconcile ClaimNb with actual severity counts
        # In French TPL, ClaimNb records reported claims; some may settle with 0 cost.
        df["ClaimNbClean"] = np.maximum(df["ClaimNb"], df["ClaimCount"])

        # Exposure constraints per actuarial guidelines
        min_exp = self.config["preprocessing"]["exposure_min"]
        max_exp = self.config["preprocessing"]["exposure_max"]
        original_len = len(df)
        df = df[df["Exposure"] >= min_exp].copy()
        df["Exposure"] = np.clip(df["Exposure"], min_exp, max_exp)
        logger.info("Filtered %d records with exposure below %0.4f.", original_len - len(df), min_exp)

        # Claim amount clipping to stabilize GLM variance (limit large bodily injury outliers)
        claim_cap = self.config["preprocessing"]["claim_amount_cap"]
        df["ClaimAmountCapped"] = np.clip(df["TotalClaimAmount"], 0.0, claim_cap)
        df["IsLargeLoss"] = (df["TotalClaimAmount"] > claim_cap).astype(int)

        # Pure Premium and Frequency rates
        df["ClaimFrequency"] = df["ClaimNbClean"] / df["Exposure"]
        df["PurePremium"] = df["ClaimAmountCapped"] / df["Exposure"]

        # Synthetic Estimated Earned Premium for loss ratio calculation
        # Benchmark base rate * BonusMalus / 100 * Exposure * Power adjustment
        base_rate = 320.0  # Industry benchmark baseline premium in EUR
        df["EarnedPremium"] = (
            base_rate
            * (df["BonusMalus"] / 100.0)
            * (1.0 + 0.08 * (df["VehPower"] - 6))
            * (1.0 + 0.12 * (df["Area"].isin(["E", "F"])))
            * df["Exposure"]
        ).clip(lower=25.0)

        df["LossRatio"] = df["ClaimAmountCapped"] / df["EarnedPremium"]

        # Survival Analysis Feature Engineering (Time-to-Event and Right-Censoring)
        # Exposure is in policy-years -> convert to duration in days (max 365)
        df["DurationDays"] = np.round(df["Exposure"] * 365.25).astype(int)
        
        # Event indicator: 1 if at least one claim occurred during observation period, 0 if right-censored
        df["ClaimEvent"] = (df["ClaimNbClean"] > 0).astype(int)

        # If claim occurred, time-to-first-claim is uniformly distributed across exposure window
        np.random.seed(42)
        random_event_fraction = np.random.uniform(0.1, 0.95, size=len(df))
        df["SurvivalTimeDays"] = np.where(
            df["ClaimEvent"] == 1,
            np.maximum(1, np.round(df["DurationDays"] * random_event_fraction).astype(int)),
            df["DurationDays"]
        )

        # Categorical feature encoding & demographic binning
        df["LogDensity"] = np.log1p(df["Density"])
        df["DrivAgeGroup"] = pd.cut(
            df["DrivAge"],
            bins=[17, 25, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "66+"]
        )
        df["VehAgeGroup"] = pd.cut(
            df["VehAge"],
            bins=[-1, 1, 5, 10, 15, 100],
            labels=["0-1yr", "2-5yr", "6-10yr", "11-15yr", "16yr+"]
        )
        df["BonusMalusClass"] = pd.cut(
            df["BonusMalus"],
            bins=[0, 50, 70, 90, 100, 130, 400],
            labels=["BM_SuperBonus", "BM_Good", "BM_Medium", "BM_Neutral", "BM_Malus", "BM_HighMalus"]
        )

        # Export processed Parquet and CSV
        out_parquet = self.processed_dir / "french_motor_tpl_clean.parquet"
        out_csv = self.processed_dir / "french_motor_tpl_clean.csv"

        df.to_parquet(out_parquet, index=False)
        df.head(5000).to_csv(out_csv, index=False)
        logger.info("Saved processed policy data (%d rows) to %s", len(df), out_parquet)

        return df

    def process_cas_schedule_p(self) -> pd.DataFrame:
        """
        Load, validate, and standardize CAS Schedule P loss triangles into long format.
        """
        logger.info("Processing CAS NAIC Schedule P loss triangles...")
        cas_file = self.raw_dir / self.config["datasets"]["cas_schedule_p"]["local_filename"]

        if not cas_file.exists():
            logger.warning("CAS file not found in raw. Generating synthetic benchmark triangles...")
            df_cas = generate_synthetic_cas_triangle(start_year=1988, num_years=10, random_seed=42)
            df_cas.to_csv(cas_file, index=False)
        else:
            df_cas = pd.read_csv(cas_file)

        # Standardize column headers
        df_cas.columns = [c.strip() for c in df_cas.columns]

        # Calculate incremental losses from cumulative
        df_sorted = df_cas.sort_values(by=["GRCODE", "AccidentYear", "DevelopmentLag"]).copy()
        df_sorted["PriorCumPaid"] = df_sorted.groupby(["GRCODE", "AccidentYear"])["CumPaidLoss_"].shift(1).fillna(0.0)
        df_sorted["IncPaidLoss"] = df_sorted["CumPaidLoss_"] - df_sorted["PriorCumPaid"]

        # Case reserves = Incurred - Cumulative Paid
        df_sorted["CaseReserves"] = np.maximum(0.0, df_sorted["IncurLoss_"] - df_sorted["CumPaidLoss_"])

        # Loss Ratios by evaluation lag
        df_sorted["PaidLossRatio"] = df_sorted["CumPaidLoss_"] / df_sorted["EarnedPremNet_"]
        df_sorted["IncurLossRatio"] = df_sorted["IncurLoss_"] / df_sorted["EarnedPremNet_"]

        out_path = self.processed_dir / "loss_development_triangles.parquet"
        out_csv = self.processed_dir / "loss_development_triangles.csv"
        df_sorted.to_parquet(out_path, index=False)
        df_sorted.to_csv(out_csv, index=False)
        logger.info("Saved normalized loss development records (%d rows) to %s", len(df_sorted), out_path)

        return df_sorted

    def run_all(self) -> None:
        """Execute complete preprocessing pipeline."""
        df_tpl = self.process_french_tpl()
        df_cas = self.process_cas_schedule_p()
        logger.info("Actuarial preprocessing completed successfully.")
        print("\n" + "=" * 65)
        print("          PREPROCESSED DATASET METRICS")
        print("=" * 65)
        print(f"Total French TPL Policies:  {len(df_tpl):,}")
        print(f"Policies with Claims:       {df_tpl['ClaimEvent'].sum():,} ({df_tpl['ClaimEvent'].mean():.2%})")
        print(f"Total Incurred Losses:      €{df_tpl['ClaimAmountCapped'].sum():,.2f}")
        print(f"Total Earned Premium:       €{df_tpl['EarnedPremium'].sum():,.2f}")
        print(f"Portfolio Loss Ratio:       {df_tpl['ClaimAmountCapped'].sum() / df_tpl['EarnedPremium'].sum():.2%}")
        print(f"Schedule P Triangle Rows:   {len(df_cas):,}")
        print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run actuarial preprocessing.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    preprocessor = ActuarialPreprocessor(config_path=args.config)
    preprocessor.run_all()


if __name__ == "__main__":
    main()
