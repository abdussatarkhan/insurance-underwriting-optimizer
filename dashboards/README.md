# Executive Underwriting & Actuarial Power BI Dashboard

## 1. Overview & Architecture

The **Underwriting Loss Ratio Optimizer Dashboard** is a multi-tier Business Intelligence and Actuarial analytics platform developed in Power BI. It connects directly to PostgreSQL reporting views and processed Parquet data models to provide portfolio managers, pricing actuaries, and underwriters with end-to-end transparency into loss development, underwriting profitability, and rate adequacy.

```
┌────────────────────────────────────────────────────────┐
│                   PostgreSQL Database                  │
│       (Materialized Views: mv_underwriting_exec)       │
└──────────────────────────┬─────────────────────────────┘
                           │ DirectQuery / Scheduled Import
┌──────────────────────────▼─────────────────────────────┐
│             Power BI Star Schema Data Model            │
│  - Dim_Driver      - Dim_Vehicle      - Dim_Geography  │
│  - Dim_Calendar    - Fact_Policies    - Fact_Reserves  │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│                 5-Page Interactive Report              │
│  [Page 1: Portfolio Overview]                          │
│  [Page 2: Actuarial Loss Reserving & IBNR]             │
│  [Page 3: Segment Profitability & Rate Adequacy]       │
│  [Page 4: Survival & Time-to-Event Dynamics]           │
│  [Page 5: Tweedie Pricing GLM & Decile Lift]           │
└────────────────────────────────────────────────────────┘
```

---

## 2. Dashboard Pages Specification

### Page 1: Portfolio Executive Overview
* **KPI Scorecards**:
  - `Total Policies`: 678,013
  - `Earned Car-Years`: 365,420
  - `Gross Earned Premium`: €116.9M
  - `Incurred Losses`: €75.4M
  - `Aggregate Loss Ratio`: 64.5% *(Target: 65.0%)*
  - `Combined Operating Ratio`: 89.5% *(Breakeven: 100.0%)*
* **Visuals**:
  - **Loss Ratio Gauge**: Colored bands (Green: <65%, Amber: 65-75%, Red: >75%).
  - **Choropleth Map**: Loss ratio and claim density across French Administrative Regions (R11 to R94).
  - **Trend Slicer**: Filterable by Accident Year, Vehicle Fuel Type, and Rating Territory.

### Page 2: Actuarial Loss Reserving & IBNR Triangles
* **Interactive Triangle Matrix**:
  - Heatmap showing cumulative paid losses across Accident Years (1988–1997) and Development Lags (1–10).
* **Reserve Technique Comparison Visual**:
  - Clustered column chart comparing Mack Chain Ladder, Bornhuetter-Ferguson, and Cape Cod IBNR reserves.
* **Emergence & Link Ratio Line Chart**:
  - Selected Volume-Weighted development factors and cumulative percent emerged curve.

### Page 3: Segment Profitability & Underwriting Remediation
* **Cohort Bubble Chart**:
  - **X-axis**: Earned Exposure (Policy-years).
  - **Y-axis**: Segment Loss Ratio (%).
  - **Bubble Size**: Total Incurred Losses (€).
  - **Color**: Profitability Category (Green: Profitable, Orange: Borderline, Red: Unprofitable).
* **Worst 20% Pareto Action Table**:
  - Lists the worst-performing segments with 95% Bootstrap Confidence Intervals.
  - Displays Actuarial Indicated Rate Adjustment (`+15% to +25%`) and recommended underwriting guideline modifications.

### Page 4: Survival Analysis & Policy Lifecycle
* **Kaplan-Meier Survival Curves**:
  - Interactive survival function $S(t)$ comparing claim-free retention across driver age cohorts and vehicle types.
* **Hazard Rate Progression**:
  - Cumulative hazard plot illustrating peak collision risk timing during initial policy months.
* **Log-Rank Significance Card**:
  - p-value readout confirming statistical risk separation.

### Page 5: Tweedie Pricing GLM & Decile Lift
* **10-Decile Lift Chart**:
  - Compares Model Predicted Pure Premium vs Observed Actual Pure Premium across 10 risk deciles.
* **Lorenz Curve & Gini Indicator**:
  - Actuarial Lorenz curve displaying risk sorting power (Model Gini: 0.38 vs Random: 0.00).
* **Rating Factor Relativities Table**:
  - Exponentiated multiplicative GLM rating multipliers for driver age, vehicle power, area, and bonus-malus.

---

## 3. Core DAX Measure Definitions

```dax
// 1. Total Earned Premium
Total Earned Premium = 
SUM(Fact_Policies[EarnedPremium])

// 2. Total Incurred Losses (Capped)
Total Incurred Loss = 
SUM(Fact_Policies[ClaimAmountCapped])

// 3. Aggregate Loss Ratio
Loss Ratio % = 
DIVIDE([Total Incurred Loss], [Total Earned Premium], BLANK())

// 4. Combined Operating Ratio (Assuming 25% Expense Ratio)
Combined Operating Ratio % = 
[Loss Ratio %] + 0.25

// 5. Indicated Rate Revision (Target LR = 65%)
Indicated Rate Revision % = 
VAR TargetLR = 0.65
RETURN
DIVIDE([Loss Ratio %], TargetLR) - 1.0

// 6. Chain Ladder IBNR Reserve
CL IBNR Reserve = 
SUM(Fact_Reserves[CL_IBNR])

// 7. Cape Cod Ultimate Loss
Cape Cod Ultimate = 
SUM(Fact_Reserves[CapeCod_Ultimate])

// 8. Bootstrap CI Lower (Segment Level)
Segment LR Lower 95% = 
AVERAGE(Fact_Segments[LR_CI_Lower])

// 9. Bootstrap CI Upper (Segment Level)
Segment LR Upper 95% = 
AVERAGE(Fact_Segments[LR_CI_Upper])
```

---

## 4. Row-Level Security (RLS) & Governance

The data model enforces dynamic Row-Level Security based on underwriter operating territories:
* **Regional Underwriting Directors**: Access to assigned administrative regions (`Region = USERNAME()`).
* **Chief Actuary & Portfolio Committee**: Unrestricted global visibility across all regions, reserving triangles, and capital allocation models.
