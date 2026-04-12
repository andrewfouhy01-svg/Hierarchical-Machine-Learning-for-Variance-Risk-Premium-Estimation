#===============================================================================
# RV_Feature_Engineering_Enhanced.R
# Enhanced Feature Engineering for Forward Realised Volatility Prediction
# With Integration of VIX Model Predictions (Classification & Regression)
#
# Author: Andrew Fouhy
# Date: January 2026
#
#===============================================================================
# THEORETICAL FRAMEWORK
#===============================================================================
#
# 1. OBJECTIVE
# ------------
# Predict RV_{t+1, t+23} (22-day forward RV starting tomorrow) incorporating:
#   (a) Historical volatility features with proper embargo
#   (b) VIX model predictions as informative signals
#   (c) Advanced microstructure and regime features
#
# 2. VIX-RV RELATIONSHIP
# ----------------------
# Under Q-measure: VIX_t^2 ≈ E^Q[(1/T) ∫_t^{t+T} σ_s^2 ds]
# Under P-measure: RV_t = (1/T) ∑_{i=1}^n r_i^2 (realised estimator)
#
# The Variance Risk Premium: VRP_t = VIX_t^2 - E^P[RV_{t,t+T}]
#
# Our VIX predictions provide:
#   - Classification P(VIX_{t+1} > VIX_t): directional signal
#   - Regression E[VIX_{t+1}]: level forecast with uncertainty
#
# 3. INFORMATION HIERARCHY (CRITICAL FOR LEAKAGE PREVENTION)
# ----------------------------------------------------------
# At prediction time t, we observe:
#   - Prices up to and including day t
#   - VIX_t (implied vol for [t, t+22])
#   - Our VIX model's prediction for VIX_{t+1}, made using info up to t
#
# We predict: Direction of RV_{t+1,t+23} vs RV_{t,t+22}
#
# VALID FEATURES:
#   - Lagged RV (≥22 days lag to avoid overlap)
#   - Returns, volume, range (any lag ≥1)
#   - VIX predictions made with info ≤ t
#
# INVALID FEATURES:
#   - Contemporaneous RV (overlaps with target)
#   - Future information of any kind
#
# 4. VIX PREDICTION INTEGRATION STRATEGY
# --------------------------------------
# Option A: Walk-forward OOS predictions (preferred for deployment)
# Option B: Stacked CV predictions (preferred for training)
#
# We use Option B: the OOS predictions from CPCV ensure each observation's
# VIX prediction came from a model that never trained on that observation.
#
#===============================================================================

#===============================================================================
# 0. SETUP AND CONFIGURATION
#===============================================================================

source("Setup.R")

cat_progress(paste(rep("=", 80), collapse = ""))
cat_progress("ENHANCED RV FEATURE ENGINEERING")
cat_progress("With VIX Model Predictions Integration")
cat_progress(paste(rep("=", 80), collapse = ""))

# Configuration
CONFIG <- list(
  # Core parameters
  rv_window = 22L,                 # VIX convention: 22 trading days
  embargo_days = 22L,              # Minimum lag for RV features (no overlap)
  
  # VIX prediction features
  vix_pred_lookback = c(5L, 10L, 22L),  # Windows for prediction-based features
  
  # Correlation thresholds
  suspicious_cor_threshold = 0.5,
  extreme_cor_threshold = 0.8,
  
  # Missing data
  na_threshold_pct = 0.20,
  
  # Numerical stability
  eps = 1e-10,
  
  # Feature engineering
  har_lags = c(1L, 5L, 22L),       # HAR-RV style lags (daily, weekly, monthly)
  mom_windows = c(5L, 10L, 22L, 63L),
  
  # Regime detection
  n_regimes = 3L,                  # For HMM-based features
  
  # Random seed
  seed = 42L
)

set.seed(CONFIG$seed)

#===============================================================================
# 1. LOAD BASE DATA AND VIX MODEL PREDICTIONS
#===============================================================================

cat_progress("Loading base data and VIX model predictions...")

# Load RV data
rv_data <- readRDS("data/data_with_rv.rds")
setDT(rv_data)

# Load split info
split_info <- readRDS("results/models/split_info.rds")
split_date <- split_info$split_date

# Load VIX model predictions (OOS from CV)
vix_class_cv_preds <- tryCatch(
  readRDS("results/models/xgb_classification_cv_predictions.rds"),
  error = function(e) {
    cat_progress("WARNING: Classification CV predictions not found")
    NULL
  }
)

vix_class_test_preds <- tryCatch(
  readRDS("results/models/xgb_classification_predictions.rds"),
  error = function(e) {
    cat_progress("WARNING: Classification test predictions not found")
    NULL
  }
)

vix_reg_cv_preds <- tryCatch(
  readRDS("results/models/xgb_regression_ar1_cv_predictions.rds"),
  error = function(e) {
    cat_progress("WARNING: Regression CV predictions not found")
    NULL
  }
)

vix_reg_test_preds <- tryCatch(
  readRDS("results/models/xgb_regression_ar1_predictions.rds"),
  error = function(e) {
    cat_progress("WARNING: Regression test predictions not found")
    NULL
  }
)

# Verify required columns in base data
required_cols <- c("date", "close", "open", "high", "low", "volume",
                   "log_return", "vix_close", "rv_cc", "rv_daily", 
                   "rv_weekly", "rv_monthly")
missing_cols <- setdiff(required_cols, names(rv_data))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

cat_progress(sprintf("Base data: %d observations from %s to %s",
                     nrow(rv_data), min(rv_data$date), max(rv_data$date)))
cat_progress(sprintf("Train/test split: %s", split_date))

#===============================================================================
# 2. MERGE VIX PREDICTIONS INTO BASE DATA
#===============================================================================

cat_progress("Merging VIX model predictions...")

# Combine CV (training period) and test predictions
# Classification predictions
if (!is.null(vix_class_cv_preds) && !is.null(vix_class_test_preds)) {
  setDT(vix_class_cv_preds)
  setDT(vix_class_test_preds)
  
  # Aggregate CV predictions (each date may appear in multiple folds)
  # Use mean probability across folds for robustness
  vix_class_agg <- vix_class_cv_preds[, .(
    vix_pred_prob = mean(pred_prob, na.rm = TRUE),
    vix_pred_prob_sd = sd(pred_prob, na.rm = TRUE),
    vix_pred_n_folds = .N
  ), by = date]
  
  # Add test predictions
  test_class <- vix_class_test_preds[, .(
    date,
    vix_pred_prob = pred_prob,
    vix_pred_prob_sd = NA_real_,
    vix_pred_n_folds = 1L
  )]
  
  vix_class_all <- rbind(vix_class_agg, test_class)
  vix_class_all <- vix_class_all[order(date)]
  
  # Merge into base data
  rv_data <- merge(rv_data, vix_class_all, by = "date", all.x = TRUE)
  cat_progress(sprintf("  Classification predictions merged: %d dates", 
                       sum(!is.na(rv_data$vix_pred_prob))))
} else {
  rv_data[, `:=`(vix_pred_prob = NA_real_, vix_pred_prob_sd = NA_real_)]
  cat_progress("  No classification predictions available")
}





names(vix_reg_cv_preds)
head(vix_reg_cv_preds$pred[is.na(as.numeric(vix_reg_cv_preds$pred))])
names(vix_reg_test_preds)


# Regression predictions
if (!is.null(vix_reg_cv_preds) && !is.null(vix_reg_test_preds)) {
  setDT(vix_reg_cv_preds)
  setDT(vix_reg_test_preds)
  
  # Aggregate CV predictions
  vix_reg_agg <- vix_reg_cv_preds[, .(
    vix_pred_level = mean(as.numeric(predicted_vix), na.rm = TRUE),
    vix_pred_level_sd = sd(as.numeric(predicted_vix), na.rm = TRUE),
    vix_pred_error_mean = mean(as.numeric(error_vix), na.rm = TRUE),
    vix_pred_error_sd = sd(as.numeric(error_vix), na.rm = TRUE)
  ), by = date]
  
  # Add test predictions with CI
  if ("pi_95_lower" %in% names(vix_reg_test_preds)) {
    test_reg <- vix_reg_test_preds[, .(
      date,
      vix_pred_level = as.numeric(predicted_vix),
      vix_pred_level_sd = NA_real_,
      vix_pred_error_mean = as.numeric(error_level),
      vix_pred_error_sd = NA_real_,
      vix_pred_ci95_lower = as.numeric(pi_95_lower),
      vix_pred_ci95_upper = as.numeric(pi_95_upper),
      vix_pred_ci90_lower = as.numeric(pi_90_lower),
      vix_pred_ci90_upper = as.numeric(pi_90_upper)
    )]
  } else {
    test_reg <- vix_reg_test_preds[, .(
      date,
      vix_pred_level = as.numeric(predicted_vix),
      vix_pred_level_sd = NA_real_,
      vix_pred_error_mean = as.numeric(error_level),
      vix_pred_error_sd = NA_real_
    )]
  }
  
  vix_reg_all <- rbind(vix_reg_agg, test_reg, fill = TRUE)
  vix_reg_all <- vix_reg_all[order(date)]
  
  # Merge into base data
  rv_data <- merge(rv_data, vix_reg_all, by = "date", all.x = TRUE)
  cat_progress(sprintf("  Regression predictions merged: %d dates",
                       sum(!is.na(rv_data$vix_pred_level))))
} else {
  rv_data[, `:=`(vix_pred_level = NA_real_, vix_pred_level_sd = NA_real_)]
  cat_progress("  No regression predictions available")
}

#===============================================================================
# 3. CREATE TARGET VARIABLE
#===============================================================================

cat_progress("Creating target variable: RV direction...")

# Target: Direction of 22-day forward RV
# RV_{t+1, t+23} vs RV_{t, t+22}
rv_data[, rv_cc_next := shift(rv_cc, -1L, type = "lead")]

# Classification target
rv_data[, target_rv_direction := as.integer(rv_cc_next > rv_cc)]

# Regression targets for auxiliary analysis
rv_data[, rv_change := rv_cc_next - rv_cc]
rv_data[, rv_change_pct := (rv_cc_next - rv_cc) / (rv_cc + CONFIG$eps) * 100]
rv_data[, rv_log_change := log(rv_cc_next + CONFIG$eps) - log(rv_cc + CONFIG$eps)]

# Summary
target_summary <- rv_data[!is.na(target_rv_direction), .(
  n = .N,
  pct_up = mean(target_rv_direction) * 100,
  mean_change = mean(rv_change, na.rm = TRUE),
  sd_change = sd(rv_change, na.rm = TRUE),
  mean_change_pct = mean(rv_change_pct, na.rm = TRUE)
)]

cat_progress(sprintf("Target: %d obs, Up=%.1f%%, Mean change=%.3f (%.2f%%)",
                     target_summary$n, target_summary$pct_up, 
                     target_summary$mean_change, target_summary$mean_change_pct))

#===============================================================================
# 4. CREATE VIX PREDICTION FEATURES
#===============================================================================

cat_progress("Creating VIX prediction-based features...")

# 4.1 Core prediction features (lagged by 1 day for causality)
# The prediction at t is for VIX_{t+1}, so we lag it to use at t+1
rv_data[, `:=`(
  # Classification: probability of VIX increase
  vix_pred_prob_lag1 = shift(vix_pred_prob, 1L),
  vix_pred_prob_lag2 = shift(vix_pred_prob, 2L),
  vix_pred_prob_lag5 = shift(vix_pred_prob, 5L),
  
  # Prediction uncertainty (from CV fold variation)
  vix_pred_prob_uncertainty = shift(vix_pred_prob_sd, 1L),
  
  # Regression: predicted level
  vix_pred_level_lag1 = shift(vix_pred_level, 1L),
  vix_pred_level_lag2 = shift(vix_pred_level, 2L),
  vix_pred_level_lag5 = shift(vix_pred_level, 5L)
)]

# 4.2 Prediction error features (how well did our model do recently?)
# Realised prediction error: actual VIX - predicted VIX (lagged appropriately)
rv_data[, vix_pred_error_realised := vix_close - shift(vix_pred_level, 1L)]
rv_data[, `:=`(
  # Rolling prediction accuracy
  vix_pred_mae_5d = frollmean(abs(vix_pred_error_realised), n = 5L, align = "right"),
  vix_pred_mae_22d = frollmean(abs(vix_pred_error_realised), n = 22L, align = "right"),
  
  # Rolling prediction bias
  vix_pred_bias_5d = frollmean(vix_pred_error_realised, n = 5L, align = "right"),
  vix_pred_bias_22d = frollmean(vix_pred_error_realised, n = 22L, align = "right"),
  
  # Rolling directional accuracy (did prob > 0.5 match direction?)
  vix_actual_direction = as.integer(vix_close > shift(vix_close, 1L))
)]

rv_data[, vix_pred_dir_correct := as.integer(
  (vix_pred_prob_lag1 > 0.5) == vix_actual_direction
)]
rv_data[, `:=`(
  vix_pred_dir_acc_5d = frollmean(vix_pred_dir_correct, n = 5L, align = "right"),
  vix_pred_dir_acc_22d = frollmean(vix_pred_dir_correct, n = 22L, align = "right")
)]

# 4.3 Prediction vs market features
rv_data[, `:=`(
  # Predicted change vs actual VIX
  vix_pred_vs_actual = vix_pred_level_lag1 - shift(vix_close, 1L),
  
  # Standardised prediction (z-score of prediction relative to recent predictions)
  vix_pred_level_ma22 = frollmean(vix_pred_level, n = 22L, align = "right"),
  vix_pred_level_sd22 = frollapply(vix_pred_level, n = 22L, FUN = sd, align = "right")
)]
rv_data[, vix_pred_zscore := (shift(vix_pred_level, 1L) - shift(vix_pred_level_ma22, 1L)) / 
          (shift(vix_pred_level_sd22, 1L) + CONFIG$eps)]

# 4.4 Classification probability features
rv_data[, `:=`(
  # Entropy of prediction (uncertainty measure)
  # H = -p*log(p) - (1-p)*log(1-p)
  vix_pred_entropy = -shift(vix_pred_prob, 1L) * log(shift(vix_pred_prob, 1L) + CONFIG$eps) -
    (1 - shift(vix_pred_prob, 1L)) * log(1 - shift(vix_pred_prob, 1L) + CONFIG$eps),
  
  # Extreme predictions (high confidence)
  vix_pred_high_conf = as.integer(abs(shift(vix_pred_prob, 1L) - 0.5) > 0.3),
  
  # Rolling probability statistics
  vix_pred_prob_ma5 = frollmean(vix_pred_prob, n = 5L, align = "right"),
  vix_pred_prob_ma22 = frollmean(vix_pred_prob, n = 22L, align = "right")
)]

# 4.5 Confidence interval features (if available)
if ("vix_pred_ci95_lower" %in% names(rv_data)) {
  rv_data[, `:=`(
    # CI width (uncertainty)
    vix_pred_ci95_width = shift(vix_pred_ci95_upper - vix_pred_ci95_lower, 1L),
    vix_pred_ci90_width = shift(vix_pred_ci90_upper - vix_pred_ci90_lower, 1L),
    
    # Normalised CI width
    vix_pred_ci95_width_pct = shift((vix_pred_ci95_upper - vix_pred_ci95_lower) / 
                                      vix_pred_level, 1L) * 100,
    
    # Is actual within CI? (lagged for analysis)
    vix_in_ci95 = as.integer(vix_close >= shift(vix_pred_ci95_lower, 1L) & 
                               vix_close <= shift(vix_pred_ci95_upper, 1L)),
    vix_in_ci90 = as.integer(vix_close >= shift(vix_pred_ci90_lower, 1L) & 
                               vix_close <= shift(vix_pred_ci90_upper, 1L))
  )]
  
  # Rolling CI coverage
  rv_data[, `:=`(
    vix_ci95_coverage_22d = frollmean(vix_in_ci95, n = 22L, align = "right"),
    vix_ci90_coverage_22d = frollmean(vix_in_ci90, n = 22L, align = "right")
  )]
}

# 4.6 Classification-Regression agreement features
rv_data[, `:=`(
  # Do class and reg agree on direction?
  vix_pred_agree = as.integer(
    (shift(vix_pred_prob, 1L) > 0.5) == 
      (shift(vix_pred_level, 1L) > shift(vix_close, 1L))
  ),
  
  # Strength of agreement (probability × predicted change)
  vix_pred_strength = shift(vix_pred_prob, 1L) * 
    (shift(vix_pred_level, 1L) - shift(vix_close, 1L)) / 
    (shift(vix_close, 1L) + CONFIG$eps)
)]

cat_progress(sprintf("Created %d VIX prediction features",
                     sum(grepl("^vix_pred_", names(rv_data)))))

#===============================================================================
# 5. CREATE LAGGED RV FEATURES (NO LEAKAGE)
#===============================================================================

cat_progress("Creating lagged RV features (embargo >= 22 days)...")

# 5.1 Core lagged RV values
rv_data[, `:=`(
  rv_lag_22 = shift(rv_cc, 22L),
  rv_lag_44 = shift(rv_cc, 44L),
  rv_lag_66 = shift(rv_cc, 66L),
  rv_lag_126 = shift(rv_cc, 126L),
  rv_lag_252 = shift(rv_cc, 252L)
)]

# 5.2 HAR-RV components (lagged with embargo)
rv_data[, `:=`(
  rv_daily_lag_1 = shift(rv_daily, 1L),
  rv_daily_lag_5 = shift(rv_daily, 5L),
  rv_daily_lag_22 = shift(rv_daily, 22L),
  rv_weekly_lag_22 = shift(rv_weekly, 22L),
  rv_monthly_lag_22 = shift(rv_monthly, 22L)
)]

# 5.3 Rolling RV statistics
rv_data[, `:=`(
  rv_ma_22_lag22 = frollmean(rv_lag_22, n = 22L, align = "right"),
  rv_ma_63_lag22 = frollmean(rv_lag_22, n = 63L, align = "right"),
  rv_ma_126_lag22 = frollmean(rv_lag_22, n = 126L, align = "right"),
  rv_ma_252_lag22 = frollmean(rv_lag_22, n = 252L, align = "right"),
  
  rv_sd_22_lag22 = frollapply(rv_lag_22, n = 22L, FUN = sd, align = "right"),
  rv_sd_63_lag22 = frollapply(rv_lag_22, n = 63L, FUN = sd, align = "right"),
  rv_sd_252_lag22 = frollapply(rv_lag_22, n = 252L, FUN = sd, align = "right")
)]

# 5.4 RV z-score and percentile (mean-reversion signals)
rv_data[, rv_zscore_lag22 := (rv_lag_22 - rv_ma_252_lag22) / (rv_sd_252_lag22 + CONFIG$eps)]

rv_data[, rv_pctrank_252_lag22 := frollapply(
  rv_lag_22, n = 252L,
  FUN = function(x) rank(x)[length(x)] / length(x),
  align = "right"
)]

# 5.5 RV term structure
rv_data[, `:=`(
  rv_slope_22_44 = rv_lag_22 - rv_lag_44,
  rv_slope_22_66 = rv_lag_22 - rv_lag_66,
  rv_slope_44_126 = rv_lag_44 - rv_lag_126,
  rv_curve = (rv_lag_22 - 2 * rv_lag_44 + rv_lag_66)  # Curvature
)]

# 5.6 RV momentum and acceleration
rv_data[, `:=`(
  rv_momentum_22 = (rv_lag_22 - rv_lag_44) / (rv_lag_44 + CONFIG$eps),
  rv_momentum_44 = (rv_lag_22 - rv_lag_66) / (rv_lag_66 + CONFIG$eps),
  rv_momentum_126 = (rv_lag_22 - rv_lag_126) / (rv_lag_126 + CONFIG$eps)
)]

rv_data[, rv_accel := rv_momentum_22 - shift(rv_momentum_22, 22L)]

# 5.7 Volatility of volatility (vol-of-vol)
rv_data[, `:=`(
  rv_vov_22 = rv_sd_22_lag22 / (rv_ma_22_lag22 + CONFIG$eps),
  rv_vov_63 = rv_sd_63_lag22 / (rv_ma_63_lag22 + CONFIG$eps)
)]

# 5.8 RV skewness and kurtosis (rolling)
rv_data[, rv_skew_63 := frollapply(rv_lag_22, n = 63L, FUN = function(x) {
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (s < 1e-10) return(NA_real_)
  mean(((x - m) / s)^3, na.rm = TRUE)
}, align = "right")]

rv_data[, rv_kurt_63 := frollapply(rv_lag_22, n = 63L, FUN = function(x) {
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (s < 1e-10) return(NA_real_)
  mean(((x - m) / s)^4, na.rm = TRUE) - 3  # Excess kurtosis
}, align = "right")]

cat_progress(sprintf("Created %d lagged RV features",
                     sum(grepl("^rv_(lag|ma|sd|zscore|pctrank|slope|curve|momentum|accel|vov|skew|kurt)", 
                               names(rv_data)))))

#===============================================================================
# 6. CREATE SPX RETURN FEATURES
#===============================================================================

cat_progress("Creating SPX return features...")

# 6.1 Return lags
rv_data[, `:=`(
  ret_lag_1 = shift(log_return, 1L),
  ret_lag_2 = shift(log_return, 2L),
  ret_lag_3 = shift(log_return, 3L),
  ret_lag_5 = shift(log_return, 5L),
  ret_lag_10 = shift(log_return, 10L),
  ret_lag_22 = shift(log_return, 22L)
)]

# 6.2 Cumulative returns
rv_data[, `:=`(
  ret_cum_5 = frollsum(log_return, n = 5L, align = "right"),
  ret_cum_10 = frollsum(log_return, n = 10L, align = "right"),
  ret_cum_22 = frollsum(log_return, n = 22L, align = "right"),
  ret_cum_63 = frollsum(log_return, n = 63L, align = "right"),
  ret_cum_126 = frollsum(log_return, n = 126L, align = "right"),
  ret_cum_252 = frollsum(log_return, n = 252L, align = "right")
)]

# 6.3 Rolling volatility (from daily returns)
rv_data[, `:=`(
  ret_vol_5 = frollapply(log_return, n = 5L, FUN = sd, align = "right") * sqrt(252),
  ret_vol_10 = frollapply(log_return, n = 10L, FUN = sd, align = "right") * sqrt(252),
  ret_vol_22 = frollapply(log_return, n = 22L, FUN = sd, align = "right") * sqrt(252),
  ret_vol_63 = frollapply(log_return, n = 63L, FUN = sd, align = "right") * sqrt(252),
  ret_vol_252 = frollapply(log_return, n = 252L, FUN = sd, align = "right") * sqrt(252)
)]

# 6.4 Absolute returns (volatility proxy)
rv_data[, abs_ret := abs(log_return)]
rv_data[, `:=`(
  abs_ret_ma_5 = frollmean(abs_ret, n = 5L, align = "right"),
  abs_ret_ma_22 = frollmean(abs_ret, n = 22L, align = "right"),
  abs_ret_ma_63 = frollmean(abs_ret, n = 63L, align = "right")
)]

# 6.5 Signed returns (leverage effect)
rv_data[, `:=`(
  ret_neg = pmin(log_return, 0),
  ret_pos = pmax(log_return, 0)
)]

rv_data[, `:=`(
  ret_neg_cum_5 = frollsum(ret_neg, n = 5L, align = "right"),
  ret_neg_cum_22 = frollsum(ret_neg, n = 22L, align = "right"),
  ret_pos_cum_5 = frollsum(ret_pos, n = 5L, align = "right"),
  ret_pos_cum_22 = frollsum(ret_pos, n = 22L, align = "right"),
  ret_asymmetry_22 = abs(frollsum(ret_neg, n = 22L, align = "right")) / 
    (abs(frollsum(ret_pos, n = 22L, align = "right")) + CONFIG$eps)
)]

# 6.6 Extreme returns
rv_data[, ret_sd_22 := frollapply(log_return, n = 22L, FUN = sd, align = "right")]
rv_data[, ret_zscore := log_return / (shift(ret_sd_22, 1L) + CONFIG$eps)]

rv_data[, `:=`(
  n_extreme_up_22 = frollsum(as.numeric(ret_zscore > 2), n = 22L, align = "right"),
  n_extreme_down_22 = frollsum(as.numeric(ret_zscore < -2), n = 22L, align = "right"),
  max_ret_22 = frollapply(log_return, n = 22L, FUN = max, align = "right"),
  min_ret_22 = frollapply(log_return, n = 22L, FUN = min, align = "right"),
  ret_range_22 = frollapply(log_return, n = 22L, FUN = max, align = "right") -
    frollapply(log_return, n = 22L, FUN = min, align = "right")
)]

# 6.7 Return momentum
rv_data[, `:=`(
  ret_ma_5 = frollmean(log_return, n = 5L, align = "right"),
  ret_ma_22 = frollmean(log_return, n = 22L, align = "right"),
  ret_ma_63 = frollmean(log_return, n = 63L, align = "right"),
  ret_ma_cross_5_22 = frollmean(log_return, n = 5L, align = "right") - 
    frollmean(log_return, n = 22L, align = "right"),
  ret_ma_cross_22_63 = frollmean(log_return, n = 22L, align = "right") - 
    frollmean(log_return, n = 63L, align = "right")
)]

# 6.8 Return autocorrelation
rv_data[, ret_autocor_22 := frollapply(log_return, n = 22L, FUN = function(x) {
  if (length(x) < 5) return(NA_real_)
  cor(x[-length(x)], x[-1], use = "complete.obs")
}, align = "right")]

cat_progress(sprintf("Created %d return features",
                     sum(grepl("^ret_|^abs_ret_|^n_extreme", names(rv_data)))))

#===============================================================================
# 7. CREATE INTRADAY RANGE FEATURES
#===============================================================================

cat_progress("Creating intraday range-based features...")

# 7.1 Daily range measures
rv_data[, `:=`(
  daily_range = (high - low) / close,
  daily_range_pct = (high - low) / low * 100,
  daily_gap = (open - shift(close, 1L)) / shift(close, 1L) * 100
)]

# 7.2 Parkinson volatility estimator
rv_data[, parkinson_var := (1 / (4 * log(2))) * (log(high / low))^2]
rv_data[, `:=`(
  parkinson_vol_5 = sqrt(frollmean(parkinson_var, n = 5L, align = "right") * 252) * 100,
  parkinson_vol_22 = sqrt(frollmean(parkinson_var, n = 22L, align = "right") * 252) * 100,
  parkinson_vol_63 = sqrt(frollmean(parkinson_var, n = 63L, align = "right") * 252) * 100
)]

# 7.3 Garman-Klass volatility estimator
rv_data[, gk_var := 0.5 * (log(high / low))^2 - (2 * log(2) - 1) * (log(close / open))^2]
rv_data[, `:=`(
  gk_vol_5 = sqrt(pmax(frollmean(gk_var, n = 5L, align = "right"), 0) * 252) * 100,
  gk_vol_22 = sqrt(pmax(frollmean(gk_var, n = 22L, align = "right"), 0) * 252) * 100
)]

# 7.4 Rogers-Satchell volatility estimator (drift-independent)
rv_data[, rs_var := log(high / close) * log(high / open) + log(low / close) * log(low / open)]
rv_data[, `:=`(
  rs_vol_5 = sqrt(pmax(frollmean(rs_var, n = 5L, align = "right"), 0) * 252) * 100,
  rs_vol_22 = sqrt(pmax(frollmean(rs_var, n = 22L, align = "right"), 0) * 252) * 100
)]

# 7.5 Range statistics
rv_data[, `:=`(
  range_ma_5 = frollmean(daily_range, n = 5L, align = "right"),
  range_ma_22 = frollmean(daily_range, n = 22L, align = "right"),
  range_sd_22 = frollapply(daily_range, n = 22L, FUN = sd, align = "right"),
  range_ratio_5_22 = frollmean(daily_range, n = 5L, align = "right") /
    (frollmean(daily_range, n = 22L, align = "right") + CONFIG$eps)
)]

# 7.6 Gap analysis
rv_data[, `:=`(
  gap_ma_22 = frollmean(abs(daily_gap), n = 22L, align = "right"),
  n_gap_up_22 = frollsum(as.numeric(daily_gap > 0.5), n = 22L, align = "right"),
  n_gap_down_22 = frollsum(as.numeric(daily_gap < -0.5), n = 22L, align = "right")
)]

cat_progress(sprintf("Created %d range-based features",
                     sum(grepl("^daily_|^parkinson_|^gk_|^rs_|^range_|^gap_", names(rv_data)))))

#===============================================================================
# 8. CREATE VOLUME FEATURES
#===============================================================================

cat_progress("Creating volume features...")

rv_data[, `:=`(
  volume_ma_5 = frollmean(volume, n = 5L, align = "right"),
  volume_ma_22 = frollmean(volume, n = 22L, align = "right"),
  volume_ma_63 = frollmean(volume, n = 63L, align = "right"),
  volume_ratio_5_22 = frollmean(volume, n = 5L, align = "right") / 
    (frollmean(volume, n = 22L, align = "right") + CONFIG$eps),
  volume_ratio_22_63 = frollmean(volume, n = 22L, align = "right") / 
    (frollmean(volume, n = 63L, align = "right") + CONFIG$eps)
)]

# Volume-volatility relationship
rv_data[, `:=`(
  vol_x_range = volume * daily_range,
  vol_x_absret = volume * abs_ret
)]

rv_data[, `:=`(
  vol_range_ma_22 = frollmean(vol_x_range, n = 22L, align = "right"),
  vol_absret_ma_22 = frollmean(vol_x_absret, n = 22L, align = "right")
)]

# Rolling volume-return correlation (using TTR or manual)
rv_data[, vol_ret_cor_22 := {
  n <- 22L
  out <- rep(NA_real_, .N)
  for (i in n:.N) {
    idx <- (i - n + 1):i
    out[i] <- cor(volume[idx], abs(log_return)[idx], use = "complete.obs")
  }
  out
}]

cat_progress(sprintf("Created %d volume features",
                     sum(grepl("^volume_|^vol_x_|^vol_ret_|^vol_range|^vol_absret", names(rv_data)))))

#===============================================================================
# 9. CREATE VOLATILITY REGIME FEATURES
#===============================================================================

cat_progress("Creating volatility regime features...")

# 9.1 RV regime indicators
rv_data[, `:=`(
  rv_regime_pct = rv_pctrank_252_lag22,
  rv_regime_high = as.integer(rv_pctrank_252_lag22 > 0.75),
  rv_regime_low = as.integer(rv_pctrank_252_lag22 < 0.25),
  rv_regime_extreme = as.integer(rv_pctrank_252_lag22 > 0.90 | rv_pctrank_252_lag22 < 0.10)
)]

# 9.2 Regime transitions
rv_data[, `:=`(
  rv_regime_high_lag1 = shift(rv_regime_high, 1L),
  rv_regime_high_lag5 = shift(rv_regime_high, 5L)
)]

rv_data[, regime_change := as.integer(rv_regime_high != shift(rv_regime_high, 1L))]
rv_data[is.na(regime_change), regime_change := 0L]

# 9.3 VIX-based regime features (lagged)
rv_data[, `:=`(
  vix_lag_1 = shift(vix_close, 1L),
  vix_lag_5 = shift(vix_close, 5L),
  vix_lag_22 = shift(vix_close, 22L),
  vix_ma_5 = frollmean(vix_close, n = 5L, align = "right"),
  vix_ma_22 = frollmean(vix_close, n = 22L, align = "right"),
  vix_ma_63 = frollmean(vix_close, n = 63L, align = "right")
)]

rv_data[, `:=`(
  vix_slope = shift(vix_close, 1L) - shift(vix_ma_22, 1L),
  vix_zscore = (shift(vix_close, 1L) - shift(vix_ma_63, 1L)) / 
    (frollapply(vix_close, n = 63L, FUN = sd, align = "right") + CONFIG$eps)
)]

rv_data[, vix_pctrank_252 := frollapply(
  shift(vix_close, 1L), n = 252L,
  FUN = function(x) rank(x)[length(x)] / length(x),
  align = "right"
)]

# 9.4 VIX term structure proxy (VIX vs short-term vol)
rv_data[, vix_vs_rv_lag22 := shift(vix_close, 1L) - rv_lag_22]

cat_progress(sprintf("Created %d regime features",
                     sum(grepl("^rv_regime|^regime_|^vix_lag|^vix_ma|^vix_slope|^vix_zscore|^vix_pctrank|^vix_vs_rv", 
                               names(rv_data)))))

#===============================================================================
# 10. CREATE VIX-RV INTERACTION FEATURES
#===============================================================================

cat_progress("Creating VIX-RV interaction features...")

# 10.1 VIX prediction × RV features
rv_data[, `:=`(
  # Predicted VIX vs lagged RV (VRP proxy from prediction)
  pred_vrp_proxy = vix_pred_level_lag1 - rv_lag_22,
  
  # Predicted change scaled by RV regime
  pred_rv_interaction = vix_pred_prob_lag1 * rv_zscore_lag22,
  
  # Prediction confidence × volatility level
  pred_conf_vol = abs(vix_pred_prob_lag1 - 0.5) * rv_lag_22
)]

# 10.2 Historical VRP-like features
rv_data[, vix_rv_spread_lag22 := shift(vix_close, 22L) - rv_lag_22]

# Second: create rolling statistics from it
rv_data[, `:=`(
  vrp_ma_22 = frollmean(vix_rv_spread_lag22, n = 22L, align = "right"),
  vrp_ma_63 = frollmean(vix_rv_spread_lag22, n = 63L, align = "right")
)]

# 10.3 Return-volatility interactions
rv_data[, `:=`(
  ret_vol_interaction = ret_cum_5 * ret_vol_22,
  ret_neg_vol_interaction = ret_neg_cum_22 * rv_lag_22,
  ret_neg_vix_interaction = ret_neg_cum_22 * shift(vix_close, 1L)
)]

# 10.4 Range-RV ratios
rv_data[, `:=`(
  range_rv_ratio = parkinson_vol_22 / (rv_lag_22 + CONFIG$eps),
  gk_rv_ratio = gk_vol_22 / (rv_lag_22 + CONFIG$eps)
)]

# 10.5 Volatility term structure
rv_data[, `:=`(
  rv_ratio_short_long = rv_lag_22 / (rv_lag_66 + CONFIG$eps),
  vol_ratio_5_63 = ret_vol_5 / (ret_vol_63 + CONFIG$eps),
  vol_ratio_22_252 = ret_vol_22 / (ret_vol_252 + CONFIG$eps)
)]

cat_progress(sprintf("Created %d interaction features",
                     sum(grepl("pred_vrp|pred_rv_|pred_conf|vix_rv_spread|vrp_ma|ret_vol_interaction|
                               ret_neg_vol|ret_neg_vix|range_rv|gk_rv|rv_ratio|vol_ratio", names(rv_data)))))

#===============================================================================
# 11. CREATE ADVANCED FEATURES
#===============================================================================

cat_progress("Creating advanced features...")

# 11.1 Volatility surprise (actual vs expected)
rv_data[, vol_surprise := rv_daily_lag_1 - shift(ret_vol_22, 1L)]

# 11.2 Information ratio features
rv_data[, `:=`(
  sharpe_22 = ret_cum_22 / (ret_vol_22 + CONFIG$eps) * sqrt(252 / 22),
  sharpe_63 = ret_cum_63 / (ret_vol_63 + CONFIG$eps) * sqrt(252 / 63)
)]

# 11.3 Drawdown features
rv_data[, `:=`(
  cum_max = cummax(close),
  drawdown = (close - cummax(close)) / cummax(close)
)]

rv_data[, `:=`(
  drawdown_lag1 = shift(drawdown, 1L),
  max_drawdown_22 = frollapply(drawdown, n = 22L, FUN = min, align = "right"),
  max_drawdown_63 = frollapply(drawdown, n = 63L, FUN = min, align = "right")
)]

# 11.4 Volatility persistence (GARCH-like)
rv_data[, vol_persistence := frollapply(rv_daily, n = 63L, FUN = function(x) {
  if (length(x) < 10) return(NA_real_)
  tryCatch({
    cor(x[-length(x)], x[-1], use = "complete.obs")
  }, error = function(e) NA_real_)
}, align = "right")]

# 11.5 Day-of-week effects (encode as features)
rv_data[, dow := as.integer(format(date, "%u"))]
rv_data[, `:=`(
  is_monday = as.integer(dow == 1),
  is_friday = as.integer(dow == 5)
)]

# 11.6 Month-of-year seasonality
rv_data[, month := as.integer(format(date, "%m"))]
rv_data[, `:=`(
  is_jan = as.integer(month == 1),
  is_oct = as.integer(month == 10),
  is_q4 = as.integer(month >= 10)
)]

# 11.7 Time since regime change
rv_data[, days_in_regime := {
  regime_group <- cumsum(c(1, diff(rv_regime_high) != 0))
  ave(seq_len(.N), regime_group, FUN = seq_along)
}]

cat_progress("Created advanced features")

#===============================================================================
# 12. COMPILE FEATURE SET
#===============================================================================

cat_progress("Compiling final feature set...")

# Columns to exclude from features
exclude_cols <- c(
  # Identifiers
  "date",
  
  # Target variables
  "target_rv_direction", "rv_cc_next", "rv_change", "rv_change_pct", "rv_log_change",
  
  # Contemporaneous RV (would be leakage)
  "rv_cc", "rv_parkinson", "rv_gk", "rv_rs", "rv_yz",
  "rv_daily", "rv_weekly", "rv_monthly",
  
  # Raw price data
  "open", "high", "low", "close", "volume", "adj_close",
  "log_return", "simple_return",
  
  # VIX/VRP contemporaneous (leakage)
  "vix_open", "vix_high", "vix_low", "vix_close",
  "vix_return", "vix_change",
  "vrp_cc", "vrp_parkinson", "vrp_gk", "vrp_rs", "vrp_yz",
  "vrp_var_cc", "vrp_var_parkinson",
  
  # Raw VIX predictions (use lagged versions)
  "vix_pred_prob", "vix_pred_prob_sd", "vix_pred_n_folds",
  "vix_pred_level", "vix_pred_level_sd", "vix_pred_error_mean", "vix_pred_error_sd",
  "vix_pred_ci95_lower", "vix_pred_ci95_upper", 
  "vix_pred_ci90_lower", "vix_pred_ci90_upper",
  "vix_pred_level_ma22", "vix_pred_level_sd22",
  "vix_actual_direction", "vix_pred_dir_correct",
  "vix_in_ci95", "vix_in_ci90",
  
  # Intermediate calculations
  "abs_ret", "ret_neg", "ret_pos", "parkinson_var", "gk_var", "rs_var",
  "ret_sd_22", "ret_zscore", "regime_change",
  "vol_x_range", "vol_x_absret", "cum_max", "drawdown", "dow", "month"
)

feature_cols <- setdiff(names(rv_data), exclude_cols)

# Remove any remaining non-numeric columns
numeric_check <- sapply(rv_data[, ..feature_cols], is.numeric)
feature_cols <- feature_cols[numeric_check]

cat_progress(sprintf("Initial feature count: %d", length(feature_cols)))

#===============================================================================
# 13. FEATURE CATEGORISATION
#===============================================================================

cat_progress("Categorising features...")

feature_categories <- list(
  # VIX prediction features (from our models)
  vix_predictions = feature_cols[grepl("^vix_pred_", feature_cols)],
  
  # Lagged RV features
  rv_lagged = feature_cols[grepl("^rv_lag_|^rv_ma_|^rv_sd_|rv_zscore|rv_pctrank|rv_slope|rv_curve|rv_momentum|rv_accel|rv_vov|rv_skew|rv_kurt|^rv_daily_lag|^rv_weekly_lag|^rv_monthly_lag", feature_cols)],
  
  # SPX return features
  returns = feature_cols[grepl("^ret_lag|^ret_cum|^ret_vol|^ret_ma|^ret_neg_cum|^ret_pos_cum|ret_asymmetry|^ret_range|ret_ma_cross|ret_autocor", feature_cols)],
  
  # Extreme return features
  extremes = feature_cols[grepl("n_extreme|max_ret|min_ret", feature_cols)],
  
  # Absolute return features
  abs_returns = feature_cols[grepl("^abs_ret_ma", feature_cols)],
  
  # Intraday range features
  range_based = feature_cols[grepl("^daily_range|^parkinson_vol|^gk_vol|^rs_vol|^range_ma|^range_sd|^range_ratio", feature_cols)],
  
  # Gap features
  gaps = feature_cols[grepl("^daily_gap|^gap_ma|^n_gap", feature_cols)],
  
  # Volume features
  volume = feature_cols[grepl("^volume_|^vol_ret_cor|^vol_range|^vol_absret", feature_cols)],
  
  # Regime features
  regime = feature_cols[grepl("^rv_regime|^vix_pctrank|days_in_regime", feature_cols)],
  
  # VIX context features (lagged)
  vix_context = feature_cols[grepl("^vix_lag|^vix_ma|^vix_slope|^vix_zscore|^vix_vs_rv", feature_cols)],
  
  # Interaction features
  interactions = feature_cols[grepl("pred_vrp|pred_rv_|pred_conf|vix_rv_spread|vrp_ma|ret_vol_interaction|ret_neg_vol|ret_neg_vix|range_rv|gk_rv|rv_ratio|vol_ratio|vol_surprise", feature_cols)],
  
  # Advanced features
  advanced = feature_cols[grepl("sharpe|drawdown|vol_persistence|is_monday|is_friday|is_jan|is_oct|is_q4", feature_cols)]
)

# Print summary
cat("\nFeature Categories:\n")
total_categorised <- 0
for (cat_name in names(feature_categories)) {
  n_feats <- length(feature_categories[[cat_name]])
  total_categorised <- total_categorised + n_feats
  cat(sprintf("  %-20s: %3d features\n", cat_name, n_feats))
}
uncategorised <- setdiff(feature_cols, unlist(feature_categories))
cat(sprintf("\nCategorised: %d / %d\n", total_categorised, length(feature_cols)))
if (length(uncategorised) > 0) {
  cat(sprintf("Uncategorised: %d\n", length(uncategorised)))
  cat(sprintf("  %s\n", paste(head(uncategorised, 10), collapse = ", ")))
}

#===============================================================================
# 14. HANDLE MISSING VALUES
#===============================================================================

cat_progress("Handling missing values...")

# Check NA rates
na_rates <- sapply(rv_data[, ..feature_cols], function(x) mean(is.na(x)))
high_na_features <- names(na_rates[na_rates > CONFIG$na_threshold_pct])

if (length(high_na_features) > 0) {
  cat_progress(sprintf("Removing %d features with >%.0f%% NA values",
                       length(high_na_features), CONFIG$na_threshold_pct * 100))
  cat(sprintf("  Removed: %s\n", paste(head(high_na_features, 10), collapse = ", ")))
  if (length(high_na_features) > 10) cat(sprintf("  ... and %d more\n", length(high_na_features) - 10))
  
  feature_cols <- setdiff(feature_cols, high_na_features)
  
  # Update categories
  for (cat_name in names(feature_categories)) {
    feature_categories[[cat_name]] <- intersect(feature_categories[[cat_name]], feature_cols)
  }
}

# Remove rows with NA in target
n_before <- nrow(rv_data)
rv_data <- rv_data[!is.na(target_rv_direction)]
cat_progress(sprintf("Removed %d rows with NA target", n_before - nrow(rv_data)))

#===============================================================================
# 15. LEAKAGE VERIFICATION
#===============================================================================

cat_progress("Performing rigorous leakage verification...")

# 15.1 Check for forbidden patterns
forbidden_patterns <- list(
  "Contemporaneous RV" = "^rv_(cc|parkinson|gk|rs|yz|daily|weekly|monthly)$",
  "Contemporaneous VIX" = "^vix_(close|open|high|low)$",
  "Contemporaneous VRP" = "^vrp_",
  "Target variables" = "^target_|_next$",
  "Future actuals" = "actual$",
  "Raw predictions" = "^vix_pred_(prob|level)$"
)

leakage_found <- FALSE
for (check_name in names(forbidden_patterns)) {
  matches <- grep(forbidden_patterns[[check_name]], feature_cols, value = TRUE)
  if (length(matches) > 0) {
    cat(sprintf("LEAKAGE DETECTED - %s: %s\n", check_name, paste(matches, collapse = ", ")))
    feature_cols <- setdiff(feature_cols, matches)
    leakage_found <- TRUE
  }
}

if (leakage_found) {
  cat_progress("Leaky features removed")
} else {
  cat_progress("✓ No forbidden patterns detected")
}

# 15.2 Check correlation with target
cat_progress("Computing feature-target correlations...")

direction_cors <- sapply(feature_cols, function(x) {
  if (is.numeric(rv_data[[x]])) {
    cor(rv_data[[x]], rv_data$target_rv_direction, use = "complete.obs")
  } else {
    NA_real_
  }
})

# Flag suspicious correlations
suspicious <- names(direction_cors[abs(direction_cors) > CONFIG$suspicious_cor_threshold & 
                                     !is.na(direction_cors)])
extreme <- names(direction_cors[abs(direction_cors) > CONFIG$extreme_cor_threshold & 
                                  !is.na(direction_cors)])

if (length(suspicious) > 0) {
  cat("\nFeatures with |cor| > 0.3 (review for leakage):\n")
  for (f in head(suspicious, 20)) {
    cat(sprintf("  %s: r = %.4f\n", f, direction_cors[f]))
  }
}

if (length(extreme) > 0) {
  cat("\nCRITICAL: Features with |cor| > 0.7 (likely leakage - removing):\n")
  for (f in extreme) {
    cat(sprintf("  %s: r = %.4f\n", f, direction_cors[f]))
  }
  feature_cols <- setdiff(feature_cols, extreme)
}

cat_progress(sprintf("Final feature count after leakage check: %d", length(feature_cols)))

#===============================================================================
# 16. TRAIN/TEST SPLIT
#===============================================================================

cat_progress("Applying train/test split...")

# Select columns for output
output_cols <- c("date", "target_rv_direction", "rv_change", "rv_change_pct",
                 "rv_cc", "rv_cc_next", "vix_close", feature_cols)

# Handle any missing output columns gracefully
output_cols <- intersect(output_cols, names(rv_data))

features_rv <- rv_data[, ..output_cols]

train_rv <- features_rv[date <= split_date]
test_rv <- features_rv[date > split_date]

cat_progress(sprintf("Training set: %d observations (%s to %s)",
                     nrow(train_rv), min(train_rv$date), max(train_rv$date)))
cat_progress(sprintf("Test set: %d observations (%s to %s)",
                     nrow(test_rv), min(test_rv$date), max(test_rv$date)))

# Class balance
train_balance <- table(train_rv$target_rv_direction)
test_balance <- table(test_rv$target_rv_direction)

cat_progress(sprintf("Training: Down=%d (%.1f%%), Up=%d (%.1f%%)",
                     train_balance["0"], 100 * train_balance["0"] / sum(train_balance),
                     train_balance["1"], 100 * train_balance["1"] / sum(train_balance)))
cat_progress(sprintf("Test: Down=%d (%.1f%%), Up=%d (%.1f%%)",
                     test_balance["0"], 100 * test_balance["0"] / sum(test_balance),
                     test_balance["1"], 100 * test_balance["1"] / sum(test_balance)))

#===============================================================================
# 17. FEATURE STATISTICS REPORT
#===============================================================================

cat_progress("Computing feature statistics...")

# Feature statistics for report
feature_stats <- data.table(
  Feature = feature_cols,
  Mean = sapply(feature_cols, function(x) mean(train_rv[[x]], na.rm = TRUE)),
  SD = sapply(feature_cols, function(x) sd(train_rv[[x]], na.rm = TRUE)),
  Min = sapply(feature_cols, function(x) min(train_rv[[x]], na.rm = TRUE)),
  Max = sapply(feature_cols, function(x) max(train_rv[[x]], na.rm = TRUE)),
  NA_Pct = sapply(feature_cols, function(x) mean(is.na(train_rv[[x]])) * 100),
  Target_Cor = direction_cors[feature_cols]
)

feature_stats <- feature_stats[order(-abs(Target_Cor))]

#===============================================================================
# 18. SAVE RESULTS
#===============================================================================

cat_progress("Saving enhanced RV feature engineering results...")

# Create output directories
rv_dirs <- c("data/rv_forward", "results/models/rv_forward",
             "results/tables/rv_forward", "results/figures/rv_forward")
for (d in rv_dirs) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# Save datasets
saveRDS(features_rv, "data/rv_forward/features_rv_full.rds")
saveRDS(train_rv, "data/rv_forward/features_rv_train.rds")
saveRDS(test_rv, "data/rv_forward/features_rv_test.rds")
saveRDS(feature_cols, "data/rv_forward/feature_columns_rv.rds")
saveRDS(feature_categories, "data/rv_forward/feature_categories_rv.rds")

# Save configuration
target_config_rv <- list(
  target_name = "target_rv_direction",
  target_type = "classification",
  target_definition = "RV_{t+1,t+23} > RV_{t,t+22}",
  rv_estimator = "close-to-close",
  rv_window = CONFIG$rv_window,
  embargo_days = CONFIG$embargo_days,
  class_labels = c("0" = "RV Down", "1" = "RV Up"),
  split_date = split_date,
  train_n = nrow(train_rv),
  test_n = nrow(test_rv),
  n_features = length(feature_cols),
  scale_pos_weight = as.numeric(train_balance["0"] / train_balance["1"]),
  leakage_verified = TRUE,
  vix_predictions_integrated = TRUE,
  config = CONFIG
)
saveRDS(target_config_rv, "results/models/rv_forward/target_config_rv.rds")

# Save correlation analysis
target_correlations_rv <- data.table(
  Feature = feature_cols,
  Correlation = direction_cors[feature_cols]
)
target_correlations_rv <- target_correlations_rv[order(-abs(Correlation))]
write.csv(target_correlations_rv, "results/tables/rv_forward/feature_target_correlations_rv_enhanced.csv",
          row.names = FALSE)

# Save feature statistics
write.csv(feature_stats, "results/tables/rv_forward/feature_statistics_rv_enhanced.csv", row.names = FALSE)

# Save feature summary by category
feature_summary_rv <- data.table(
  Category = names(feature_categories),
  N_Features = sapply(feature_categories, length)
)
write.csv(feature_summary_rv, "results/tables/rv_forward/feature_summary_rv_enhanced.csv", row.names = FALSE)


#===============================================================================
# END OF SCRIPT
#===============================================================================