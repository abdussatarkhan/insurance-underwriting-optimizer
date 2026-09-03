# %% [markdown]
# # Notebook 01: Insurance Data Ingestion and Source Validation
#
# **Project:** Underwriting Loss Ratio Optimizer  
# **Domain:** Actuarial Science & P&C Insurance  
# **Datasets:**
# - French Motor Third-Party Liability (freMTPL2freq, freMTPL2sev)
# - CAS NAIC Schedule P Loss Reserving Triangles (Private Passenger Auto)
#
# This notebook validates the automated data ingestion pipelines, verifies the raw data schema,
# handles missing values, and checks exposure distributions.

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

from utils import load_config, setup_logger, generate_synthetic_french_tpl, generate_synthetic_cas_triangle
from data_collection import DataCollector

logger = setup_logger("nb01_data_ingestion")
config = load_config(project_root / "config" / "config.yaml")

# %% [markdown]
# ## 1. Automated Dataset Acquisition
# Fetch or generate the raw insurance datasets via `DataCollector`.

# %%
collector = DataCollector(str(project_root / "config" / "config.yaml"))

# Run ingestion for both datasets
collector.fetch_openml_french_tpl(force=False, max_samples=25000)
collector.fetch_cas_schedule_p(force=False)
collector.print_dataset_summaries()

# %% [markdown]
# ## 2. French Motor TPL Dataset Inspection (freMTPL2)
# Load raw frequency and severity tables and analyze policy characteristics.

# %%
raw_dir = project_root / config["paths"]["raw_data_dir"]
freq_file = raw_dir / config["datasets"]["openml_freMTPL2"]["freq_filename"]
sev_file = raw_dir / config["datasets"]["openml_freMTPL2"]["sev_filename"]

df_freq = pd.read_csv(freq_file)
df_sev = pd.read_csv(sev_file)

print(f"Policy Frequency Records: {len(df_freq):,}")
print(f"Severity Claims Records:  {len(df_sev):,}")

# %%
df_freq.info()

# %%
df_freq.describe().T

# %% [markdown]
# ## 3. Exposure and Claim Count Diagnostics
# In actuarial ratemaking, policies with zero or near-zero exposure must be handled with care.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Exposure Distribution
axes[0].hist(df_freq["Exposure"], bins=30, color="#2b5c8f", edgecolor="black", alpha=0.7)
axes[0].set_title("Distribution of Policy Exposure (Years)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Exposure (Policy-Years)")
axes[0].set_ylabel("Count of Policies")
axes[0].grid(True, linestyle=":", alpha=0.5)

# Claim Count Distribution (Reported Claims)
claim_counts = df_freq["ClaimNb"].value_counts().sort_index()
axes[1].bar(claim_counts.index, claim_counts.values, color="#d9534f", edgecolor="black", alpha=0.8)
axes[1].set_title("Reported Claim Counts (ClaimNb)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Number of Claims in Policy Period")
axes[1].set_ylabel("Count of Policies (Log Scale)")
axes[1].set_yscale("log")
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. CAS NAIC Schedule P Loss Reserving Triangles
# Inspect the multi-carrier loss triangle dataset and check development lag consistency.

# %%
cas_file = raw_dir / config["datasets"]["cas_schedule_p"]["local_filename"]
df_cas = pd.read_csv(cas_file)
df_cas.columns = [c.strip() for c in df_cas.columns]

print(f"Total CAS Schedule P records: {len(df_cas):,}")
print(f"Accident Years covered: {df_cas['AccidentYear'].min()} to {df_cas['AccidentYear'].max()}")
print(f"Development Lags: {sorted(df_cas['DevelopmentLag'].unique())}")
df_cas.head(10)

# %%
# Verify incremental paid vs cumulative paid logic
df_sample = df_cas[df_cas["GRCODE"] == df_cas["GRCODE"].iloc[0]].sort_values(["AccidentYear", "DevelopmentLag"])
df_sample[["AccidentYear", "DevelopmentLag", "IncurLoss_", "CumPaidLoss_", "BulkLoss_", "EarnedPremNet_"]].head(10)

# %% [markdown]
# ## Conclusion & Next Steps
# - Data sources are validated, properly typed, and cached in `data/raw/`.
# - In Notebook 02, we conduct comprehensive Exploratory Data Analysis (EDA) focusing on loss ratios, claim frequency, severity distributions, and heavy-tailed risk behavior.
