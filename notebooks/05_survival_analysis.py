# %% [markdown]
# # Notebook 05: Survival Analysis & Time-to-Claim Modeling
#
# **Project:** Underwriting Loss Ratio Optimizer  
# **Focus:** Kaplan-Meier survival curves, log-rank hypothesis testing, and Cox Proportional Hazards regression for underwriting risk factors.

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Add scripts directory to system path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import load_config, setup_logger
from survival_analysis import UnderwritingSurvivalAnalyzer

config = load_config(project_root / "config" / "config.yaml")

# %% [markdown]
# ## 1. Initialize Survival Analyzer & Inspect Event Statistics
# We evaluate policy survival time (duration until first claim event or right-censoring at policy expiration).

# %%
analyzer = UnderwritingSurvivalAnalyzer(str(project_root / "config" / "config.yaml"))
df_surv = analyzer.df

print(f"Total Policies:        {len(df_surv):,}")
print(f"Total Claim Events:    {df_surv['ClaimEvent'].sum():,} ({df_surv['ClaimEvent'].mean():.2%})")
print(f"Right-Censored:        {(df_surv['ClaimEvent'] == 0).sum():,} ({(df_surv['ClaimEvent'] == 0).mean():.2%})")
print(f"Mean Observation Days: {df_surv['SurvivalTimeDays'].mean():.1f} days")

# %% [markdown]
# ## 2. Kaplan-Meier Survival Curves by Powertrain & Driver Age
# Estimate non-parametric survival function:
# $\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# By Fuel Type (Regular vs Diesel)
kmf_reg = KaplanMeierFitter()
kmf_die = KaplanMeierFitter()

sub_reg = df_surv[df_surv["VehGas"] == "Regular"]
sub_die = df_surv[df_surv["VehGas"] == "Diesel"]

kmf_reg.fit(sub_reg["SurvivalTimeDays"], sub_reg["ClaimEvent"], label="Regular Gas")
kmf_die.fit(sub_die["SurvivalTimeDays"], sub_die["ClaimEvent"], label="Diesel")

kmf_reg.plot_survival_function(ax=axes[0], color="#1f77b4", linewidth=2)
kmf_die.plot_survival_function(ax=axes[0], color="#d62728", linewidth=2)

axes[0].set_title("Kaplan-Meier Survival Curves by Fuel Type", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Policy Duration (Days)")
axes[0].set_ylabel("Claim-Free Survival Probability S(t)")
axes[0].set_ylim(0.85, 1.002)
axes[0].grid(True, linestyle=":", alpha=0.5)

# By Driver Age Group
km_fitters_age = analyzer.fit_kaplan_meier_strata("DrivAgeGroup")
for label, kmf in km_fitters_age.items():
    kmf.plot_survival_function(ax=axes[1], linewidth=1.8)

axes[1].set_title("Kaplan-Meier Survival Curves by Driver Age Bracket", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Policy Duration (Days)")
axes[1].set_ylabel("Claim-Free Survival Probability S(t)")
axes[1].set_ylim(0.85, 1.002)
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Log-Rank Statistical Hypothesis Testing
# Test if the survival functions differ significantly between driver cohorts.

# %%
lr_res = logrank_test(
    sub_reg["SurvivalTimeDays"], sub_die["SurvivalTimeDays"],
    sub_reg["ClaimEvent"], sub_die["ClaimEvent"]
)
print(f"Log-Rank Test (Regular vs Diesel):")
print(f"  Test Statistic: {lr_res.test_statistic:.4f}")
print(f"  p-value:        {lr_res.p_value:.4e}")

# %% [markdown]
# ## 4. Cox Proportional Hazards Regression
# Fit multivariate semiparametric Cox PH model to determine instantaneous hazard ratios:
# $h(t \mid X) = h_0(t) \exp(\beta_1 X_1 + \dots + \beta_k X_k)$

# %%
summary_cph = analyzer.fit_cox_proportional_hazards()
print(f"Model Concordance Index: {analyzer.cph_model.concordance_index_:.4f}")
summary_cph[["coef", "Hazard_Ratio", "HR_lower_95%", "HR_upper_95%", "p"]].round(4)

# %% [markdown]
# ## 5. Hazard Ratio Forest Plot
# Visualize relative instantaneous risk multiplier for each rating factor.

# %%
covariates = summary_cph.index.tolist()
hr = summary_cph["Hazard_Ratio"].values
hr_l = summary_cph["HR_lower_95%"].values
hr_u = summary_cph["HR_upper_95%"].values

y_pos = np.arange(len(covariates))

plt.figure(figsize=(10, 6))
plt.errorbar(hr, y_pos, xerr=[hr - hr_l, hr_u - hr], fmt="o", color="#1f77b4", ecolor="#7399c6", elinewidth=2, capsize=4)
plt.axvline(x=1.0, color="red", linestyle="--", linewidth=1.2, label="Neutral Hazard (HR = 1.0)")
plt.yticks(y_pos, covariates, fontsize=10)
plt.xlabel("Hazard Ratio (exp(beta)) with 95% Confidence Interval")
plt.title("Cox Proportional Hazards Model - Underwriting Factor Multipliers", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Takeaways
# - Bonus-Malus is the most powerful continuous hazard driver: each +10 points increases claim hazard by ~8%.
# - Young drivers (<25) experience significantly accelerated hazard in early policy months.
# - High population density (urban environment) elevates hazard due to traffic congestion and collision exposure.
