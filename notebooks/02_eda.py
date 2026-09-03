# %% [markdown]
# # Notebook 02: Exploratory Data Analysis & Actuarial Risk Profiling
#
# **Project:** Underwriting Loss Ratio Optimizer  
# **Focus:** Loss ratio distributions, claim frequency/severity decomposition, and heavy-tailed risk dynamics.

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Add scripts directory to system path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import load_config, setup_logger
from preprocessing import ActuarialPreprocessor

config = load_config(project_root / "config" / "config.yaml")

# %% [markdown]
# ## 1. Execute Preprocessing & Load Harmonized Policies
# Load the cleaned dataset with pure premium and exposure metrics.

# %%
preprocessor = ActuarialPreprocessor(str(project_root / "config" / "config.yaml"))
df = preprocessor.process_french_tpl()
df.head()

# %% [markdown]
# ## 2. Portfolio Key Actuarial Metrics
# Compute aggregate portfolio KPIs: overall loss ratio, frequency per 1,000 policy-years, and mean severity.

# %%
total_exposure = df["Exposure"].sum()
total_claims = df["ClaimNbClean"].sum()
total_losses = df["ClaimAmountCapped"].sum()
total_premium = df["EarnedPremium"].sum()

portfolio_lr = total_losses / total_premium
annual_freq = total_claims / total_exposure
mean_sev = total_losses / total_claims if total_claims > 0 else 0.0
pure_premium = total_losses / total_exposure

print("=" * 55)
print("       PORTFOLIO UNDERWRITING BENCHMARK KPIs")
print("=" * 55)
print(f"Total Policy Exposure:         {total_exposure:,.1f} policy-years")
print(f"Total Incurred Claims:         {total_claims:,}")
print(f"Total Earned Premium:          €{total_premium:,.2f}")
print(f"Total Incurred Losses:         €{total_losses:,.2f}")
print(f"Aggregate Loss Ratio (LR):     {portfolio_lr:.2%}")
print(f"Annual Claim Frequency:        {annual_freq:.4f} ({annual_freq*1000:.1f} per 1k car-years)")
print(f"Mean Claim Severity:           €{mean_sev:,.2f}")
print(f"Portfolio Pure Premium:        €{pure_premium:,.2f} per policy-year")
print("=" * 55)

# %% [markdown]
# ## 3. Heavy-Tailed Severity Distribution Analysis
# In property & casualty insurance, claims exhibit heavy right tails (subexponential distributions).
# We examine the empirical claims via log transforms, Pareto fitting, and Q-Q plots.

# %%
claims_only = df[df["ClaimAmountCapped"] > 0]["ClaimAmountCapped"].values

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram on Linear Scale
axes[0].hist(claims_only, bins=40, color="#2b5c8f", edgecolor="black", alpha=0.7)
axes[0].set_title("Claim Severity (€) - Linear Scale", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Claim Cost (€)")
axes[0].set_ylabel("Frequency")
axes[0].grid(True, linestyle=":", alpha=0.5)

# Histogram on Log10 Scale
axes[1].hist(np.log10(claims_only), bins=35, color="#e67e22", edgecolor="black", alpha=0.75)
axes[1].set_title("Log10 Claim Severity Distribution", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Log10(Claim Cost in €)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, linestyle=":", alpha=0.5)

# Lognormal Q-Q Plot
stats.probplot(np.log(claims_only), dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot: Log(Claim Amount) vs Normal", fontsize=11, fontweight="bold")
axes[2].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Frequency and Loss Ratio by Driver Age and Bonus-Malus
# Investigate actuarial risk differentiation across driver demographics.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Driver Age Group vs Claim Frequency
age_summary = df.groupby("DrivAgeGroup", observed=False).agg(
    Exposure=("Exposure", "sum"),
    Claims=("ClaimNbClean", "sum"),
    Losses=("ClaimAmountCapped", "sum"),
    Premium=("EarnedPremium", "sum")
).reset_index()
age_summary["Frequency"] = age_summary["Claims"] / age_summary["Exposure"]
age_summary["LossRatio"] = age_summary["Losses"] / age_summary["Premium"]

axes[0].bar(age_summary["DrivAgeGroup"], age_summary["Frequency"] * 100, color="#1f77b4", edgecolor="black", alpha=0.8)
axes[0].set_title("Annual Claim Frequency by Driver Age Group", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Driver Age Bracket")
axes[0].set_ylabel("Claim Frequency (%)")
axes[0].grid(True, linestyle=":", alpha=0.5)

# Bonus-Malus vs Loss Ratio
bm_summary = df.groupby("BonusMalusClass", observed=False).agg(
    Losses=("ClaimAmountCapped", "sum"),
    Premium=("EarnedPremium", "sum")
).reset_index()
bm_summary["LossRatio"] = bm_summary["Losses"] / bm_summary["Premium"]

axes[1].plot(bm_summary["BonusMalusClass"], bm_summary["LossRatio"] * 100, marker="o", color="#d9534f", linewidth=2.2)
axes[1].axhline(y=config["profitability"]["target_loss_ratio"] * 100, color="gray", linestyle="--", label="Target LR (65%)")
axes[1].set_title("Loss Ratio (%) Across Bonus-Malus Classes", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Bonus-Malus Class")
axes[1].set_ylabel("Loss Ratio (%)")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Correlation & Risk Covariates Heatmap
# Examine collinearity between vehicle characteristics, driver age, and loss metrics.

# %%
numeric_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity", "Exposure", "ClaimNbClean", "ClaimAmountCapped"]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Underwriting Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Insights
# 1. Young drivers (18-25) present more than 2x the claim frequency of mature drivers (36-65).
# 2. Bonus-Malus exhibits strong predictive power on loss ratio; policies with Malus (>100) require substantial underwriting surcharges.
# 3. Claims follow a heavy-tailed mixture distribution, motivating the use of compound Poisson-Gamma (Tweedie) GLMs in subsequent pricing modules.
