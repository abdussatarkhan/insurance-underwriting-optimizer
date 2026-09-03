# %% [markdown]
# # Notebook 03: Loss Triangle Cohort Development Analysis
#
# **Project:** Underwriting Loss Ratio Optimizer  
# **Domain:** Actuarial Reserving & Cohort Emergence  
# **Dataset:** CAS NAIC Schedule P Private Passenger Auto
#
# This notebook constructs cumulative and incremental loss triangles, computes empirical
# link ratios (age-to-age development factors), and analyzes cohort emergence trajectories.

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

from utils import load_config, setup_logger
from preprocessing import ActuarialPreprocessor
from cohort_analysis import LossTriangleAnalyzer

config = load_config(project_root / "config" / "config.yaml")

# %% [markdown]
# ## 1. Build Cumulative Loss Triangle
# Pivot historical transactions into accident year by development lag dimensions.

# %%
analyzer = LossTriangleAnalyzer(str(project_root / "config" / "config.yaml"))
cum_tri = analyzer.build_cumulative_triangle()

print("Cumulative Paid Loss Triangle ($ Thousands):")
cum_tri_display = (cum_tri / 1e3).round(1)
cum_tri_display

# %% [markdown]
# ## 2. Empirical Link Ratios (Age-to-Age Development Factors)
# $f_{i, j} = \frac{C_{i, j+1}}{C_{i, j}}$

# %%
link_factors = analyzer.compute_age_to_age_factors()
print("Empirical Link Ratios:")
link_factors.round(4)

# %% [markdown]
# ## 3. Actuarial Development Factors & CDF to Ultimate
# Compute Volume-Weighted, Simple Average, and Cumulative Development Factors (CDF).

# %%
summary_df = analyzer.compute_development_factor_summary()
summary_df.round(4)

# %% [markdown]
# ## 4. Visualize Cohort Development Patterns
# Plot cumulative loss trajectories and percentage emergence by development lag.

# %%
analyzer.plot_development_patterns()

# Display inline plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Cumulative Curves
palette = sns.color_palette("tab10", n_colors=len(cum_tri))
for idx, (ay, row) in enumerate(cum_tri.iterrows()):
    valid = ~row.isna()
    axes[0].plot(row.index[valid], row.values[valid] / 1e3, marker="o", label=f"AY {ay}", color=palette[idx])
axes[0].set_title("Paid Loss Emergence by Accident Year", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Development Lag (Years)")
axes[0].set_ylabel("Paid Losses ($ Thousands)")
axes[0].legend(ncol=2, fontsize=8)
axes[0].grid(True, linestyle=":", alpha=0.5)

# Emergence Percentage
lags = [int(s.split("_")[1]) for s in summary_df["Development_Step"]]
axes[1].plot(lags, summary_df["Percent_Emerged"] * 100, marker="s", color="#d9534f", linewidth=2.2)
axes[1].set_title("Cumulative % of Ultimate Loss Emerged", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Development Lag (Years)")
axes[1].set_ylabel("% Emerged")
axes[1].set_ylim(20, 105)
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion
# - Losses mature to over 90% emergence by Lag 5.
# - Tail factor of 1.002 captures remaining post-10-year development.
# - Link ratios establish the foundational inputs for Chain Ladder and Bornhuetter-Ferguson reserving in Notebook 04.
