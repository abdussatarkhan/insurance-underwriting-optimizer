-- ============================================================================
-- Insurance Underwriting Optimizer: PostgreSQL Production Queries
-- ============================================================================
-- Provides database logic for:
-- 1. Loss ratio and combined ratio aggregation by underwriting cohorts
-- 2. Loss development triangulation using SQL window functions
-- 3. Policyholder exposure and claim frequency deciling (NTILE)
-- 4. Materialized views for executive Power BI underwriting dashboards
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Portfolio Underwriting KPI Summary Table
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS view_underwriting_portfolio_kpis;
CREATE OR REPLACE VIEW view_underwriting_portfolio_kpis AS
SELECT
    COUNT(DISTINCT IDpol) AS total_policies,
    ROUND(SUM(Exposure)::numeric, 2) AS total_earned_car_years,
    SUM(ClaimNbClean) AS total_claims_count,
    ROUND(SUM(ClaimAmountCapped)::numeric, 2) AS total_incurred_loss_eur,
    ROUND(SUM(EarnedPremium)::numeric, 2) AS total_earned_premium_eur,
    ROUND((SUM(ClaimAmountCapped) / NULLIF(SUM(EarnedPremium), 0))::numeric, 4) AS portfolio_loss_ratio,
    ROUND((SUM(ClaimNbClean) / NULLIF(SUM(Exposure), 0))::numeric, 4) AS annual_claim_frequency,
    ROUND((SUM(ClaimAmountCapped) / NULLIF(SUM(ClaimNbClean), 0))::numeric, 2) AS average_claim_severity_eur,
    ROUND((SUM(ClaimAmountCapped) / NULLIF(SUM(Exposure), 0))::numeric, 2) AS empirical_pure_premium_eur
FROM french_motor_tpl_clean;


-- ----------------------------------------------------------------------------
-- 2. Segment Profitability & Actuarial Rate Adequacy Indications
-- Identifies cohorts where Loss Ratio > 75% Breakeven threshold
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS view_segment_profitability;
CREATE OR REPLACE VIEW view_segment_profitability AS
WITH segment_metrics AS (
    SELECT
        DrivAgeGroup,
        BonusMalusClass,
        VehGas,
        Area,
        COUNT(IDpol) AS policy_count,
        ROUND(SUM(Exposure)::numeric, 2) AS earned_exposure,
        ROUND(SUM(EarnedPremium)::numeric, 2) AS total_earned_premium,
        ROUND(SUM(ClaimAmountCapped)::numeric, 2) AS total_incurred_loss,
        SUM(ClaimNbClean) AS total_claims,
        -- Loss Ratio = Total Incurred Losses / Total Earned Premium
        ROUND((SUM(ClaimAmountCapped) / NULLIF(SUM(EarnedPremium), 0))::numeric, 4) AS loss_ratio,
        -- Frequency = Claims / Exposure
        ROUND((SUM(ClaimNbClean) / NULLIF(SUM(Exposure), 0))::numeric, 4) AS claim_frequency
    FROM french_motor_tpl_clean
    GROUP BY DrivAgeGroup, BonusMalusClass, VehGas, Area
    HAVING COUNT(IDpol) >= 25 -- Actuarial credibility threshold
)
SELECT
    DrivAgeGroup,
    BonusMalusClass,
    VehGas,
    Area,
    policy_count,
    earned_exposure,
    total_earned_premium,
    total_incurred_loss,
    total_claims,
    loss_ratio,
    claim_frequency,
    -- Combined Operating Ratio (Assuming 25% administrative & brokerage expense ratio)
    ROUND((loss_ratio + 0.25)::numeric, 4) AS combined_operating_ratio,
    -- Actuarial Indicated Rate Revision: (Observed LR / Target LR 0.65) - 1.0
    ROUND(((loss_ratio / 0.65) - 1.0)::numeric, 4) AS indicated_rate_change,
    -- Rank Segments by Profitability Deficit
    DENSE_RANK() OVER (ORDER BY loss_ratio DESC) AS deficit_rank,
    -- Remedial Action Classification
    CASE
        WHEN loss_ratio >= 1.20 THEN 'CRITICAL DEFICIT: Non-Renew or Strict Underwriting Exception'
        WHEN loss_ratio >= 0.75 THEN 'UNPROFITABLE: Mandate +15% to +25% Surcharge & Higher Deductible'
        WHEN loss_ratio <= 0.50 THEN 'HIGHLY PROFITABLE: Target for Growth & Marketing Retention'
        ELSE 'ADEQUATE: Maintain Current Base Rates'
    END AS underwriting_action
FROM segment_metrics;


-- ----------------------------------------------------------------------------
-- 3. Top 20% Worst Performing Underwriting Cells (Pareto Action List)
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS view_worst_20pct_segments;
CREATE OR REPLACE VIEW view_worst_20pct_segments AS
WITH ranked_cohorts AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY loss_ratio DESC) AS quintile_rank
    FROM view_segment_profitability
)
SELECT
    DrivAgeGroup,
    BonusMalusClass,
    VehGas,
    Area,
    policy_count,
    earned_exposure,
    total_earned_premium,
    total_incurred_loss,
    loss_ratio,
    combined_operating_ratio,
    indicated_rate_change,
    underwriting_action
FROM ranked_cohorts
WHERE quintile_rank = 1 -- Worst 20% quintile
ORDER BY loss_ratio DESC;


-- ----------------------------------------------------------------------------
-- 4. Actuarial Loss Triangle SQL Development Engine
-- Reconstructs cumulative losses and derives link ratios using LAG window functions
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS view_actuarial_link_ratios;
CREATE OR REPLACE VIEW view_actuarial_link_ratios AS
WITH triangle_base AS (
    SELECT
        AccidentYear,
        DevelopmentLag,
        CumPaidLoss_,
        IncurLoss_,
        EarnedPremNet_,
        LAG(CumPaidLoss_, 1) OVER (
            PARTITION BY AccidentYear 
            ORDER BY DevelopmentLag
        ) AS prior_lag_paid_loss
    FROM cas_schedule_p_ppauto
    WHERE GRCODE = (SELECT GRCODE FROM cas_schedule_p_ppauto GROUP BY GRCODE ORDER BY COUNT(*) DESC LIMIT 1)
)
SELECT
    AccidentYear,
    DevelopmentLag,
    prior_lag_paid_loss,
    CumPaidLoss_,
    IncurLoss_,
    -- Incremental Paid Loss = Current Lag Paid - Prior Lag Paid
    ROUND((CumPaidLoss_ - COALESCE(prior_lag_paid_loss, 0))::numeric, 2) AS inc_paid_loss,
    -- Case Reserves = Incurred - Cumulative Paid
    ROUND(GREATEST(0, IncurLoss_ - CumPaidLoss_)::numeric, 2) AS case_reserves,
    -- Empirical Age-to-Age Development Factor (Link Ratio)
    CASE 
        WHEN prior_lag_paid_loss IS NOT NULL AND prior_lag_paid_loss > 0 
        THEN ROUND((CumPaidLoss_ / prior_lag_paid_loss)::numeric, 4)
        ELSE NULL 
    END AS empirical_link_ratio,
    -- Paid Loss Ratio to Net Earned Premium
    ROUND((CumPaidLoss_ / NULLIF(EarnedPremNet_, 0))::numeric, 4) AS paid_loss_ratio
FROM triangle_base
ORDER BY AccidentYear, DevelopmentLag;


-- ----------------------------------------------------------------------------
-- 5. Volume-Weighted Industry Link Ratios & Emergence Pattern
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS view_volume_weighted_factors;
CREATE OR REPLACE VIEW view_volume_weighted_factors AS
WITH development_steps AS (
    SELECT
        t1.DevelopmentLag AS from_lag,
        t2.DevelopmentLag AS to_lag,
        SUM(t1.CumPaidLoss_) AS sum_curr_loss,
        SUM(t2.CumPaidLoss_) AS sum_next_loss
    FROM cas_schedule_p_ppauto t1
    INNER JOIN cas_schedule_p_ppauto t2
        ON t1.AccidentYear = t2.AccidentYear
        AND t1.DevelopmentLag + 1 = t2.DevelopmentLag
        AND t1.GRCODE = t2.GRCODE
    GROUP BY t1.DevelopmentLag, t2.DevelopmentLag
)
SELECT
    from_lag,
    to_lag,
    CONCAT('Lag ', from_lag, ' to ', to_lag) AS development_step,
    sum_curr_loss,
    sum_next_loss,
    ROUND((sum_next_loss / NULLIF(sum_curr_loss, 0))::numeric, 4) AS volume_weighted_factor
FROM development_steps
ORDER BY from_lag;


-- ----------------------------------------------------------------------------
-- 6. Risk Deciling (NTILE) for Pure Premium Pricing Model Calibration
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS view_pricing_risk_deciles;
CREATE OR REPLACE VIEW view_pricing_risk_deciles AS
WITH scored_policies AS (
    SELECT
        IDpol,
        Exposure,
        ClaimNbClean,
        ClaimAmountCapped,
        EarnedPremium,
        BonusMalus,
        VehPower,
        DrivAge,
        -- Compute Decile based on empirical risk factor combinations
        NTILE(10) OVER (
            ORDER BY (BonusMalus * (1.0 + 0.1 * VehPower) / GREATEST(18, DrivAge))
        ) AS risk_decile
    FROM french_motor_tpl_clean
)
SELECT
    risk_decile,
    COUNT(IDpol) AS policy_count,
    ROUND(SUM(Exposure)::numeric, 2) AS decile_exposure,
    SUM(ClaimNbClean) AS total_claims,
    ROUND(SUM(ClaimAmountCapped)::numeric, 2) AS total_losses,
    ROUND(SUM(EarnedPremium)::numeric, 2) AS total_premium,
    ROUND((SUM(ClaimAmountCapped) / NULLIF(SUM(Exposure), 0))::numeric, 2) AS empirical_pure_premium,
    ROUND((SUM(ClaimAmountCapped) / NULLIF(SUM(EarnedPremium), 0))::numeric, 4) AS decile_loss_ratio,
    ROUND((SUM(ClaimNbClean) / NULLIF(SUM(Exposure), 0))::numeric, 4) AS claim_frequency
FROM scored_policies
GROUP BY risk_decile
ORDER BY risk_decile;


-- ----------------------------------------------------------------------------
-- 7. Materialized View for Power BI Executive Dashboard
-- Refreshed nightly for high-performance dashboard querying
-- ----------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS mv_underwriting_executive_dashboard;
CREATE MATERIALIZED VIEW mv_underwriting_executive_dashboard AS
SELECT
    f.Region,
    f.Area,
    f.VehBrand,
    f.VehGas,
    f.DrivAgeGroup,
    f.BonusMalusClass,
    COUNT(f.IDpol) AS policy_count,
    ROUND(SUM(f.Exposure)::numeric, 2) AS earned_car_years,
    SUM(f.ClaimNbClean) AS claim_count,
    ROUND(SUM(f.ClaimAmountCapped)::numeric, 2) AS incurred_losses_eur,
    ROUND(SUM(f.EarnedPremium)::numeric, 2) AS earned_premium_eur,
    ROUND((SUM(f.ClaimAmountCapped) / NULLIF(SUM(f.EarnedPremium), 0))::numeric, 4) AS loss_ratio,
    ROUND(((SUM(f.ClaimAmountCapped) / NULLIF(SUM(f.EarnedPremium), 0)) + 0.25)::numeric, 4) AS combined_ratio,
    ROUND(((SUM(f.ClaimAmountCapped) / NULLIF(SUM(f.EarnedPremium), 0) / 0.65) - 1.0)::numeric, 4) AS indicated_rate_adjustment
FROM french_motor_tpl_clean f
GROUP BY 
    f.Region,
    f.Area,
    f.VehBrand,
    f.VehGas,
    f.DrivAgeGroup,
    f.BonusMalusClass;

CREATE INDEX idx_mv_exec_region ON mv_underwriting_executive_dashboard(Region);
CREATE INDEX idx_mv_exec_loss_ratio ON mv_underwriting_executive_dashboard(loss_ratio);
