# %% [markdown]
# # Notebook 06: Tweedie GLM Pricing & Segment Optimization
#
# **Project:** Underwriting Loss Ratio Optimizer  
# **Focus:** Tweedie Compound Poisson-Gamma GLM, Frequency-Severity modeling, Gini evaluation, and rate adequacy optimization.

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add scripts directory to system path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import load_config, setup_logger, gini_coefficient
from pricing_model import ActuarialPricingEngine
from segment_profitability import SegmentProfitabilityOptimizer

config = load_config(project_root / "config" / "config.yaml")

# %% [markdown]
# ## 1. Train Actuarial GLM Pricing Models
# Fit Tweedie GLM on pure premium and Frequency x Severity two-part model.

# %%
engine = ActuarialPricingEngine(str(project_root / "config" / "config.yaml"))
tweedie_res = engine.fit_tweedie_glm()
freq_res, sev_res = engine.fit_two_part_frequency_severity()

print(f"Tweedie GLM AIC: {tweedie_res.aic:,.2f}")
print(f"Poisson Freq GLM AIC: {freq_res.aic:,.2f}, Gamma Sev GLM AIC: {sev_res.aic:,.2f}")

# %% [markdown]
# ## 2. Out-of-Sample Performance Comparison
# Evaluate Normalized Gini, Actual-to-Expected (A/E) ratios, and weighted MAE on unseen test policies.

# %%
eval_df = engine.evaluate_models_on_test()
eval_df

# %% [markdown]
# ## 3. Decile Lift Chart & Risk Sorting Capacity
# A strong actuarial model concentrates losses heavily in the top deciles while maintaining low losses in Decile 1-3.

# %%
engine.plot_pricing_diagnostics()

# Generate inline lift visualization
test = engine.test_df
test["Decile"] = pd.qcut(test["Pred_PurePremium_Tweedie"], q=10, labels=False) + 1
decile_summary = test.groupby("Decile").agg(
    Exposure=("Exposure", "sum"),
    ActualLoss=("ClaimAmountCapped", "sum"),
    PredLoss=("Pred_Loss_Tweedie", "sum")
).reset_index()

decile_summary["Actual_PurePrem"] = decile_summary["ActualLoss"] / decile_summary["Exposure"]
decile_summary["Pred_PurePrem"] = decile_summary["PredLoss"] / decile_summary["Exposure"]

plt.figure(figsize=(12, 5))
x = decile_summary["Decile"]
width = 0.35
plt.bar(x - width/2, decile_summary["Actual_PurePrem"], width, label="Actual Observed Pure Premium", color="#1f77b4")
plt.bar(x + width/2, decile_summary["Pred_PurePrem"], width, label="Predicted Pure Premium (Tweedie)", color="#ff7f0e", alpha=0.85)
plt.title("10-Decile Lift Chart (Test Set Risk Differentiation)", fontsize=12, fontweight="bold")
plt.xlabel("Tweedie Predicted Risk Decile")
plt.ylabel("Pure Premium (€)")
plt.xticks(x)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Portfolio Segment Profitability & Rate Adequacy
# Deconstruct loss ratios across multi-dimensional risk cells with 95% Bootstrap Confidence Intervals.

# %%
optimizer = SegmentProfitabilityOptimizer(str(project_root / "config" / "config.yaml"))
df_seg = optimizer.analyze_underwriting_segments()
optimizer.print_executive_actions()

# %% [markdown]
# ## 5. Worst 20% Unprofitable Segments & Action Plan
# Inspect the top loss ratio segments and review indicated rate revisions.

# %%
worst_segments = df_seg[df_seg["IsWorst20Pct"]]
print("Top 10 High-Deficit Underwriting Segments:")
worst_segments[["Segment", "Policies", "LossRatio", "CombinedRatio", "IndicatedRateChange", "RecommendedAction"]].head(10)

# %% [markdown]
# ## Strategic Portfolio Recommendations
# 1. **Immediate Surcharge**: Apply a +15% to +25% rate adjustment on drivers under 25 with Malus (>100) operating diesel vehicles in high-density regions.
# 2. **Deductible Optimization**: Mandate a €500 higher collision deductible for High-Malus segments to suppress high attritional claim frequency.
# 3. **Target Retention**: Maintain competitive discounts (up to 8%) on Super-Bonus drivers (>10 years claim-free) to safeguard profitable low-loss volume.
