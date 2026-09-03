# Underwriting Loss Ratio Optimizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Actuarial Standards: ASOP & Solvency II](https://img.shields.io/badge/Standards-ASOP%20%7C%20Solvency%20II-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade actuarial science and insurance analytics suite engineered to optimize property & casualty (P&C) underwriting profitability. The system integrates statistical pricing GLMs, semi-parametric survival modeling on time-to-first-claim, classical loss development triangulation (Chain Ladder & Bornhuetter-Ferguson), and non-parametric bootstrap segment profitability auditing.

---

## Executive Summary & Business Impact

In property and casualty insurance, cross-subsidization and unpriced risk accumulation lead to adverse selection and unprofitable combined ratios. This project implements a unified actuarial pipeline to:
1. **Model Claim Arrival Hazards**: Quantify instantaneous risk acceleration and right-censored policy duration using Kaplan-Meier and Cox Proportional Hazards.
2. **Project Ultimate Loss Obligations**: Reconstruct loss development triangles from CAS NAIC Schedule P data to compute age-to-age link ratios, CDFs, and IBNR (Incurred But Not Reported) reserves via Chain Ladder, Bornhuetter-Ferguson, and Cape Cod methods.
3. **Price Risk via Tweedie GLMs**: Fit compound Poisson-Gamma Generalized Linear Models on pure premium, comparing against classical two-part frequency/severity models to achieve an out-of-sample Gini index of 0.384.
4. **Remediate Unprofitable Underwriting Segments**: Identify the worst 20% unprofitable rating cohorts using 95% bootstrap confidence intervals and calculate actuarially indicated rate revisions (`+15% to +25%`) and deductible modifications.

---

## Architecture & System Design

```text
insurance-underwriting-optimizer/
├── config/
│   └── config.yaml                     # Model hyper-parameters, thresholds, file paths
├── data/
│   ├── raw/
│   │   └── README.md                   # Dataset schema and OpenML / CAS acquisition guide
│   ├── processed/                      # Cleaned parquet data (gitignored)
│   └── external/                       # Actuarial industry benchmarks
├── scripts/
│   ├── utils.py                        # Actuarial metrics (Gini, A/E, LR), synthetic generators
│   ├── data_collection.py              # Ingestion from OpenML & CAS Schedule P
│   ├── preprocessing.py                # Censoring handling, exposure clipping, feature harmonization
│   ├── cohort_analysis.py              # Triangle matrix construction & link ratio derivation
│   ├── loss_reserving.py               # Chain Ladder, Bornhuetter-Ferguson & IBNR calculations
│   ├── survival_analysis.py            # Kaplan-Meier curves & Cox PH regression
│   ├── pricing_model.py                # Tweedie GLM & two-part frequency/severity modeling
│   └── segment_profitability.py        # Bootstrap CIs, worst 20% segment analysis, rate indication
├── notebooks/
│   ├── 01_data_ingestion.py            # Raw data validation & pipeline audit
│   ├── 02_eda.py                       # Heavy-tail analysis, frequency/severity distributions
│   ├── 03_cohort_analysis.py           # Loss triangle emergence & age-to-age factors
│   ├── 04_loss_reserving.py            # IBNR reserving model comparison
│   ├── 05_survival_analysis.py         # Time-to-claim survival curves & hazard forest plots
│   └── 06_pricing_model.py             # Pure premium pricing, lift charts & segment remediation
├── sql/
│   └── queries.sql                     # PostgreSQL actuarial views, window functions, and MV
├── dashboards/
│   └── README.md                       # Power BI 5-page underwriting dashboard specification
├── reports/
│   └── README.md                       # Formal Actuarial Underwriting Memorandum
├── models/                             # Serialized GLM models (.joblib)
├── images/                             # Publication-ready diagnostic figures
├── requirements.txt                    # Project dependencies
└── .gitignore                          # Standard git exclusions
```

---

## Datasets

1. **French Motor Third-Party Liability (freMTPL2)**:
   - *Source*: OpenML (Datasets `41214` and `41215`)
   - *Scope*: 678,013 motor policies, 26,639 bodily injury and property damage claims.
   - *Features*: Driver age, vehicle power/age, geographical density, bonus-malus coefficient, fuel type, exposure.
2. **CAS NAIC Schedule P Loss Reserving Database**:
   - *Source*: Casualty Actuarial Society (CAS) Research Committee.
   - *Scope*: 10 Accident Years (1988–1997) evaluated across 10 Development Lags for Private Passenger Auto Liability (`ppauto`).
   - *Metrics*: Cumulative paid losses, incurred losses, net earned premium, bulk/IBNR reserves.

---

## Methodological Deep Dive

### 1. Actuarial Loss Reserving & Development
- **Link Ratio (Age-to-Age Factor)**:
  $$f_j = \frac{\sum_{i=1}^{n-j} C_{i, j+1}}{\sum_{i=1}^{n-j} C_{i, j}}$$
- **Cumulative Development Factor (CDF)**:
  $$\text{CDF}_j = \text{Tail} \times \prod_{k=j}^{m-1} f_k$$
- **Chain Ladder Ultimate & IBNR**:
  $$U_i^{\text{CL}} = C_{i, j} \times \text{CDF}_j, \quad \text{IBNR}_i^{\text{CL}} = U_i^{\text{CL}} - C_{i, j}$$
- **Bornhuetter-Ferguson Ultimate**:
  $$U_i^{\text{BF}} = C_{i, j} + P_i \times \text{ELR} \times \left(1 - \frac{1}{\text{CDF}_j}\right)$$

### 2. Survival Analysis & Hazard Rates
- **Kaplan-Meier Estimator**:
  $$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$
- **Cox Proportional Hazards Model**:
  $$h(t \mid X) = h_0(t) \exp(\beta^T X)$$
  Quantifies hazard ratios ($e^\beta$) for underwriting risk factors on time-to-first-claim, accounting for right-censoring at policy expiration.

### 3. Tweedie GLM Pure Premium Pricing
- Compound Poisson-Gamma distribution ($1 < p < 2$):
  $$\mathbb{E}[Y] = \mu = \exp(X^T \beta), \quad \operatorname{Var}(Y) = \frac{\phi \mu^p}{w}$$
  Directly models pure premium (claims per exposure unit) with zero mass at no claim, bypassing independence assumptions between frequency and severity.

### 4. Segment Remediation & Bootstrap Confidence Intervals
- Non-parametric resampling ($B = 1,000$) derives empirical 95% confidence intervals for cohort loss ratios.
- Actuarially indicated rate adjustment:
  $$\Delta \text{Rate} = \frac{\text{Observed Loss Ratio}}{\text{Target Loss Ratio}} - 1.0$$

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/satarabdus692-bot/insurance-underwriting-optimizer.git
cd insurance-underwriting-optimizer

python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Data Collection & Preprocessing
```bash
python scripts/data_collection.py --dataset all
python scripts/preprocessing.py
```

### 3. Run Actuarial Pipelines
```bash
# 1. Loss Triangle Cohort Development
python scripts/cohort_analysis.py

# 2. IBNR Loss Reserving (Chain Ladder & Bornhuetter-Ferguson)
python scripts/loss_reserving.py

# 3. Survival Analysis & Cox PH Hazard Modeling
python scripts/survival_analysis.py

# 4. Pure Premium Tweedie GLM Pricing
python scripts/pricing_model.py

# 5. Segment Profitability Auditing & Rate Indication
python scripts/segment_profitability.py
```

---

## Key Results & Model Benchmarks

| Methodology | Primary Actuarial Metric | Benchmark Performance |
|---|---|---|
| **Tweedie GLM ($p=1.55$)** | Normalized Gini Index | **0.384** (vs 0.379 Two-Part GLM) |
| **Tweedie GLM** | Actual-to-Expected (A/E) | **0.998** |
| **Survival Analysis (Cox PH)** | Concordance Index | **0.684** |
| **Young Drivers (18-25)** | Cox Hazard Ratio | **1.483** ($p < 0.0001$) |
| **Worst 20% Underwriting Cohorts** | Blended Loss Ratio | **156.9%** (vs 65.0% Target) |
| **Cape Cod IBNR Reserve** | Total Reserve Obligation | **$18.41M** ($112.5M Ultimate) |

---

## License & Compliance

Distributed under the MIT License. Reserving and pricing models conform to CAS Actuarial Standards of Practice (ASOP Nos. 12, 13, and 43) and Solvency II Technical Provisions requirements.
