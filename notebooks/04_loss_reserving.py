# %% [markdown]
# # Notebook 04: Actuarial Loss Reserving & IBNR Estimation
#
# **Project:** Underwriting Loss Ratio Optimizer  
# **Focus:** Mack Chain Ladder, Bornhuetter-Ferguson (BF), Cape Cod methods, and IBNR reserve adequacy.

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to system path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(project_root / "scripts"))

from utils import load_config, setup_logger
from loss_reserving import ActuarialReservingEngine

config = load_config(project_root / "config" / "config.yaml")

# %% [markdown]
# ## 1. Execute Reserving Engine
# Initialize the actuarial reserving engine and compute reserves across methods.

# %%
engine = ActuarialReservingEngine(str(project_root / "config" / "config.yaml"))
df_reserves = engine.compare_reserving_methods()
df_reserves

# %% [markdown]
# ## 2. Reserving Comparative Summary
# Evaluate aggregate IBNR obligations and ultimate loss ratio across techniques.

# %%
total_paid = df_reserves["PaidToDate"].sum()
total_prem = df_reserves["EarnedPremium"].sum()

cl_ibnr = df_reserves["CL_IBNR"].sum()
bf_ibnr = df_reserves["BF_IBNR"].sum()
cc_ibnr = df_reserves["CapeCod_IBNR"].sum()

cl_ult = df_reserves["CL_Ultimate"].sum()
bf_ult = df_reserves["BF_Ultimate"].sum()
cc_ult = df_reserves["CapeCod_Ultimate"].sum()

res_summary = pd.DataFrame({
    "Method": ["Mack Chain Ladder", "Bornhuetter-Ferguson", "Cape Cod (Stanard-Bühlmann)"],
    "Total_IBNR ($)": [cl_ibnr, bf_ibnr, cc_ibnr],
    "Total_Ultimate ($)": [cl_ult, bf_ult, cc_ult],
    "Ultimate_Loss_Ratio": [cl_ult / total_prem, bf_ult / total_prem, cc_ult / total_prem],
    "Reserve_to_Paid_Ratio": [cl_ibnr / total_paid, bf_ibnr / total_paid, cc_ibnr / total_paid]
})
res_summary

# %% [markdown]
# ## 3. IBNR Distribution by Accident Year
# Newer accident years have the highest reserve uncertainty and largest IBNR requirements.

# %%
x = np.arange(len(df_reserves["AccidentYear"]))
width = 0.25

plt.figure(figsize=(14, 6))
plt.bar(x - width, df_reserves["CL_IBNR"] / 1e3, width, label="Chain Ladder", color="#1f77b4")
plt.bar(x, df_reserves["BF_IBNR"] / 1e3, width, label="Bornhuetter-Ferguson", color="#ff7f0e")
plt.bar(x + width, df_reserves["CapeCod_IBNR"] / 1e3, width, label="Cape Cod", color="#2ca02c")

plt.title("IBNR Reserves by Accident Year and Actuarial Technique", fontsize=12, fontweight="bold")
plt.xlabel("Accident Year", fontsize=10)
plt.ylabel("IBNR Reserve ($ Thousands)", fontsize=10)
plt.xticks(x, df_reserves["AccidentYear"])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Ultimate Loss Ratio Progression
# Compare the implied ultimate loss ratios across accident years against the 65% target.

# %%
plt.figure(figsize=(12, 5))
plt.plot(df_reserves["AccidentYear"], df_reserves["CL_LR"] * 100, marker="o", label="Chain Ladder LR%", color="#1f77b4", linewidth=2)
plt.plot(df_reserves["AccidentYear"], df_reserves["BF_LR"] * 100, marker="s", label="Bornhuetter-Ferguson LR%", color="#ff7f0e", linewidth=2)
plt.plot(df_reserves["AccidentYear"], df_reserves["CapeCod_LR"] * 100, marker="^", label="Cape Cod LR%", color="#2ca02c", linewidth=2)
plt.axhline(y=65.0, color="gray", linestyle="--", label="Target Loss Ratio (65%)")
plt.title("Ultimate Loss Ratio Comparison by Accident Year", fontsize=12, fontweight="bold")
plt.xlabel("Accident Year")
plt.ylabel("Ultimate Loss Ratio (%)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Actuarial Finding
# - Chain Ladder produces higher volatility for the most immature accident year due to sensitivity to early payments.
# - Bornhuetter-Ferguson dampens this volatility by incorporating the a priori expected loss ratio.
# - The recommended management reserve is the Cape Cod estimate, providing an objective balance.
