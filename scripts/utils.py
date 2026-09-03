"""
Insurance Underwriting Optimizer - Utility Functions & Actuarial Metrics
========================================================================
Provides shared utilities for:
- Configuration and directory management
- Actuarial loss ratio and performance metrics
- Triangle display formatting and table generation
- Synthetic data generators for French Motor TPL and CAS Schedule P
  (used for unit testing, offline demonstrations, and reproducible runs)
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Parameters
    ----------
    config_path : Optional[str]
        Path to config.yaml. If None, resolves from default location.
        
    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure a standardized console logger.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    log_level : int
        Logging level (default: logging.INFO).
        
    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create all necessary output and data directories if they do not exist."""
    root = get_project_root()
    paths = config.get("paths", {})
    for _, rel_path in paths.items():
        dir_path = root / rel_path
        dir_path.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Actuarial and Performance Metrics
# =====================================================================

def calculate_loss_ratio(
    losses: Union[pd.Series, np.ndarray],
    earned_premium: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate the aggregate Loss Ratio: Total Losses / Total Earned Premium.
    
    Parameters
    ----------
    losses : pd.Series or np.ndarray
        Incurred or paid claims in monetary units.
    earned_premium : pd.Series or np.ndarray
        Earned premium corresponding to the exposure period.
        
    Returns
    -------
    float
        Aggregate loss ratio (e.g. 0.68 -> 68%).
    """
    total_loss = np.sum(losses)
    total_prem = np.sum(earned_premium)
    if total_prem <= 0:
        return np.nan
    return float(total_loss / total_prem)


def calculate_actual_vs_expected(
    actual: Union[pd.Series, np.ndarray],
    expected: Union[pd.Series, np.ndarray],
    weights: Optional[Union[pd.Series, np.ndarray]] = None
) -> float:
    """
    Compute Actuarial A/E (Actual-to-Expected) Ratio.
    """
    if weights is not None:
        act = np.sum(actual * weights)
        exp = np.sum(expected * weights)
    else:
        act = np.sum(actual)
        exp = np.sum(expected)
        
    if exp == 0:
        return np.nan
    return float(act / exp)


def gini_coefficient(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    exposure: Optional[Union[pd.Series, np.ndarray]] = None
) -> float:
    """
    Calculate the Normalized Gini Coefficient for insurance pricing models.
    Measures the model's ability to rank-order risk relative to the perfect model.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if exposure is None:
        exposure = np.ones_like(y_true, dtype=float)
    else:
        exposure = np.asarray(exposure, dtype=float)

    def lorenz_curve_auc(true_vals, pred_vals, weights):
        order = np.argsort(pred_vals)
        ordered_weights = weights[order]
        ordered_loss = (true_vals * weights)[order]
        
        cum_weights = np.cumsum(ordered_weights) / np.sum(ordered_weights)
        cum_loss = np.cumsum(ordered_loss) / np.sum(ordered_loss)
        
        cum_weights = np.insert(cum_weights, 0, 0.0)
        cum_loss = np.insert(cum_loss, 0, 0.0)
        
        return np.trapz(cum_loss, cum_weights)

    auc_model = lorenz_curve_auc(y_true, y_pred, exposure)
    auc_perfect = lorenz_curve_auc(y_true, y_true, exposure)
    auc_random = 0.5

    gini_model = 2.0 * (auc_random - auc_model)
    gini_perfect = 2.0 * (auc_random - auc_perfect)
    
    if abs(gini_perfect) < 1e-9:
        return 0.0
    return float(gini_model / gini_perfect)


# =====================================================================
# Synthetic Dataset Generators for Testing & Offline Execution
# =====================================================================

def generate_synthetic_french_tpl(
    n_samples: int = 15000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic French Motor TPL dataset mimicking freMTPL2freq and freMTPL2sev.
    Ensures zero-external-dependency functionality for offline testing and demos.
    """
    np.random.seed(random_seed)
    policy_ids = np.arange(100000, 100000 + n_samples)
    
    # Exposure: Beta distribution shifted between 0.05 and 1.0 year
    exposure = np.clip(np.random.beta(2.5, 1.2, size=n_samples), 0.05, 1.0)
    
    # Driver demographics
    driv_age = np.random.choice(
        np.arange(18, 86),
        size=n_samples,
        p=np.exp(-0.5 * ((np.arange(18, 86) - 42) / 14) ** 2) / np.sum(np.exp(-0.5 * ((np.arange(18, 86) - 42) / 14) ** 2))
    )
    
    veh_age = np.random.geometric(p=0.14, size=n_samples) - 1
    veh_age = np.clip(veh_age, 0, 25)
    
    veh_power = np.random.choice([4, 5, 6, 7, 8, 9, 10, 11, 12], size=n_samples, p=[0.15, 0.22, 0.25, 0.18, 0.10, 0.05, 0.03, 0.01, 0.01])
    bonus_malus = np.clip(np.random.lognormal(mean=4.15, sigma=0.28, size=n_samples), 50, 250).astype(int)
    
    area = np.random.choice(["A", "B", "C", "D", "E", "F"], size=n_samples, p=[0.15, 0.20, 0.30, 0.18, 0.12, 0.05])
    veh_brand = np.random.choice([f"B{i}" for i in range(1, 13)], size=n_samples)
    veh_gas = np.random.choice(["Regular", "Diesel"], size=n_samples, p=[0.48, 0.52])
    region = np.random.choice([f"R{i}" for i in [11, 24, 31, 41, 52, 72, 82, 93]], size=n_samples)
    
    density = np.exp(np.random.normal(loc=5.5, scale=1.8, size=n_samples))
    
    # Frequency risk score (log-linear GLM rate)
    log_lambda = (
        -2.3
        + 0.008 * (bonus_malus - 100)
        + 0.04 * (veh_power - 6)
        - 0.015 * np.maximum(0, driv_age - 25)
        + 0.45 * (driv_age < 25)
        + 0.02 * veh_age
        + 0.12 * (area == "E") + 0.25 * (area == "F")
        + 0.08 * (veh_gas == "Diesel")
        + np.log(exposure)
    )
    lambda_param = np.exp(log_lambda)
    
    # Claim counts via Poisson
    claim_nb = np.random.poisson(lam=lambda_param)
    
    df_freq = pd.DataFrame({
        "IDpol": policy_ids,
        "ClaimNb": claim_nb,
        "Exposure": np.round(exposure, 4),
        "Area": area,
        "VehPower": veh_power,
        "VehAge": veh_age,
        "DrivAge": driv_age,
        "BonusMalus": bonus_malus,
        "VehBrand": veh_brand,
        "VehGas": veh_gas,
        "Density": np.round(density, 1),
        "Region": region
    })
    
    # Generate Claim Amounts for claims > 0
    sev_records = []
    claim_policies = df_freq[df_freq["ClaimNb"] > 0]
    for _, row in claim_policies.iterrows():
        c_nb = int(row["ClaimNb"])
        for _ in range(c_nb):
            # Heavy-tailed claim severity: Lognormal mixture with Pareto tail
            if np.random.rand() < 0.96:
                # Typical attritional loss
                cost = np.random.lognormal(mean=6.9, sigma=1.0)
            else:
                # Large bodily injury claim (Pareto tail)
                cost = 10000.0 * (np.random.pareto(a=1.8) + 1.0)
            sev_records.append({
                "IDpol": int(row["IDpol"]),
                "ClaimAmount": round(float(np.clip(cost, 50.0, 150000.0)), 2)
            })
            
    df_sev = pd.DataFrame(sev_records)
    if df_sev.empty:
        df_sev = pd.DataFrame(columns=["IDpol", "ClaimAmount"])
        
    return df_freq, df_sev


def generate_synthetic_cas_triangle(
    start_year: int = 1988,
    num_years: int = 10,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic CAS Schedule P loss triangles matching NAIC standard reporting.
    """
    np.random.seed(random_seed)
    years = list(range(start_year, start_year + num_years))
    
    records = []
    base_ultimate = 45000.0
    loss_inflation = 0.04
    
    # Cumulative reporting pattern (Lags 1 to 10)
    true_cum_pattern = np.array([0.38, 0.65, 0.79, 0.88, 0.93, 0.96, 0.98, 0.99, 0.997, 1.00])
    
    for i, ay in enumerate(years):
        # Premium and ultimate losses with trend
        earned_prem = (base_ultimate * 1.35) * ((1.0 + loss_inflation) ** i) * np.random.uniform(0.95, 1.05)
        ult_loss = earned_prem * np.random.uniform(0.62, 0.76)
        
        for lag in range(1, num_years + 1):
            dev_year = ay + lag - 1
            if dev_year <= (start_year + num_years - 1):
                # Paid losses up to lag
                ratio = true_cum_pattern[lag - 1] * np.random.uniform(0.97, 1.03)
                ratio = min(ratio, 1.0)
                cum_paid = ult_loss * ratio
                
                # Incurred losses (paid + case reserves)
                incur_ratio = min(1.0, true_cum_pattern[lag - 1] ** 0.45 * np.random.uniform(0.98, 1.02))
                incur_loss = ult_loss * incur_ratio
                bulk_loss = max(0.0, ult_loss - incur_loss)
                
                records.append({
                    "GRCODE": 10001,
                    "GRNAME": "Benchmark Mutual Insurance Co",
                    "AccidentYear": ay,
                    "DevelopmentYear": dev_year,
                    "DevelopmentLag": lag,
                    "IncurLoss_": round(incur_loss, 2),
                    "CumPaidLoss_": round(cum_paid, 2),
                    "BulkLoss_": round(bulk_loss, 2),
                    "EarnedPremDIR_": round(earned_prem * 1.05, 2),
                    "EarnedPremCeded_": round(earned_prem * 0.05, 2),
                    "EarnedPremNet_": round(earned_prem, 2)
                })
                
    return pd.DataFrame(records)
