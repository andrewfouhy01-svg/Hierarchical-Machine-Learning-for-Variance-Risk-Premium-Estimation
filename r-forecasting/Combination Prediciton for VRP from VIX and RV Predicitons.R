################################################################################
# VRP_Comparison_Analysis.R
# Variance Risk Premium Prediction and Comparison
#
# OBJECTIVE:
# ----------
# Combine predictions from:
#   1. XGBoost VIX regression: E[VIX_{t+1} | F_t]
#   2. XGBoost RV regression:  E[RV_{t+1,t+23} | F_t]
#
# To construct predicted VRP:
#   VRP_pred_{t+1} = VIX_pred_{t+1} - RV_pred_{t+1,t+23}
#
# Compare against actual VRP:
#   VRP_actual_{t+1} = VIX_actual_{t+1} - RV_actual_{t+1,t+23}
#
# ANALYSIS:
# ---------
# 1. VRP forecast accuracy metrics
# 2. Decomposition: VIX forecast error vs RV forecast error
# 3. Trading signal generation
# 4. Mincer-Zarnowitz efficiency tests
# 5. Regime-conditional analysis
# 6. Economic value assessment
#
################################################################################

#------------------------------------------------------------------
# 0. SETUP
#------------------------------------------------------------------

source("Setup.R")

cat_progress(paste(rep("=", 70), collapse = ""))
cat_progress("VRP Comparison Analysis: Combining VIX and RV Forecasts")
cat_progress(paste(rep("=", 70), collapse = ""))

# Create output directories
vrp_dirs <- c("results/models/vrp_comparison",
              "results/tables/vrp_comparison",
              "results/figures/vrp_comparison")
for (d in vrp_dirs) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

#------------------------------------------------------------------
# 1. LOAD MODEL PREDICTIONS
#------------------------------------------------------------------

cat_progress("Loading model predictions...")

# VIX regression predictions (predicting VIX_{t+1})
vix_preds <- readRDS("results/models/xgb_regression_ar1_predictions.rds")
setDT(vix_preds)

# RV regression predictions (predicting RV_{t+1,t+23})
rv_preds <- readRDS("results/models/rv_regression/xgb_rv_regression_predictions.rds")
setDT(rv_preds)

cat_progress(sprintf("VIX predictions: %d observations (%s to %s)",
                     nrow(vix_preds), min(vix_preds$date), max(vix_preds$date)))
cat_progress(sprintf("RV predictions: %d observations (%s to %s)",
                     nrow(rv_preds), min(rv_preds$date), max(rv_preds$date)))

#------------------------------------------------------------------
# 2. MERGE PREDICTIONS AND ALIGN DATES
#------------------------------------------------------------------
names(vix_preds)


cat_progress("Merging and aligning predictions...")

# Standardise column names
# VIX predictions: date, actual (=VIX_{t+1}), predicted (=VIX_pred_{t+1})
setnames(vix_preds, c("actual_vix", "predicted_vix"), c("vix_actual", "vix_predicted"), skip_absent = TRUE)

# RV predictions: date, actual_rv (=RV_{t+1,t+23}), predicted_rv (=RV_pred_{t+1,t+23})
setnames(rv_preds, c("actual_rv", "predicted_rv"), c("rv_actual", "rv_predicted"), skip_absent = TRUE)

# Select relevant columns from each
vix_cols <- intersect(names(vix_preds), c("date", "vix_actual", "vix_predicted", "current_vix", "error"))
rv_cols <- intersect(names(rv_preds), c("date", "rv_actual", "rv_predicted", "vix_current", "forecast_error"))

vix_dt <- vix_preds[, ..vix_cols]
rv_dt <- rv_preds[, ..rv_cols]

# Rename error columns to avoid confusion
if ("error" %in% names(vix_dt)) setnames(vix_dt, "error", "vix_error")
if ("forecast_error" %in% names(rv_dt)) setnames(rv_dt, "forecast_error", "rv_error")

# Merge on date (inner join to ensure alignment)
vrp_dt <- merge(vix_dt, rv_dt, by = "date", all = FALSE)

cat_progress(sprintf("Merged dataset: %d observations (%s to %s)",
                     nrow(vrp_dt), min(vrp_dt$date), max(vrp_dt$date)))

#------------------------------------------------------------------
# 3. CONSTRUCT VRP PREDICTIONS
#------------------------------------------------------------------

cat_progress("Constructing VRP predictions...")

# VRP in volatility space: VIX - RV
vrp_dt[, `:=`(
  # Predicted VRP (from two models)
  vrp_predicted = vix_predicted - rv_predicted,
  
  # Actual VRP (realised)
  vrp_actual = vix_actual - rv_actual
)]

# VRP in variance space: (VIX^2 - RV^2) / 100
vrp_dt[, `:=`(
  vrp_var_predicted = (vix_predicted^2 - rv_predicted^2) / 100,
  vrp_var_actual = (vix_actual^2 - rv_actual^2) / 100
)]

# VRP forecast error
vrp_dt[, `:=`(
  vrp_error = vrp_actual - vrp_predicted,
  vrp_var_error = vrp_var_actual - vrp_var_predicted
)]

# Alternative VRP estimates for comparison
# 1. Using actual VIX_{t+1} with predicted RV
vrp_dt[, vrp_vix_actual_rv_pred := vix_actual - rv_predicted]

# 2. Using predicted VIX_{t+1} with actual RV
vrp_dt[, vrp_vix_pred_rv_actual := vix_predicted - rv_actual]

# 3. Naive VRP (using VIX at t as forecast for both, i.e., VRP = 0 by construction)
# Get VIX at time t (current_vix from VIX predictions)
if ("current_vix" %in% names(vrp_dt)) {
  vrp_dt[, vix_t := current_vix]
} else if ("vix_current" %in% names(vrp_dt)) {
  vrp_dt[, vix_t := vix_current]
}

cat_progress("VRP measures constructed:")
cat_progress("  - vrp_predicted: VIX_pred - RV_pred (full model)")
cat_progress("  - vrp_actual: VIX_actual - RV_actual (realised)")
cat_progress("  - vrp_var_*: variance space equivalents")

#------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
#------------------------------------------------------------------

cat_progress("Computing descriptive statistics...")

compute_moments <- function(x, name) {
  x <- x[!is.na(x)]
  n <- length(x)
  data.table(
    Variable = name,
    N = n,
    Mean = mean(x),
    SD = sd(x),
    Skewness = moments::skewness(x),
    Kurtosis = moments::kurtosis(x),
    Min = min(x),
    Q25 = quantile(x, 0.25),
    Median = median(x),
    Q75 = quantile(x, 0.75),
    Max = max(x),
    Pct_Positive = 100 * mean(x > 0)
  )
}

vrp_descriptive <- rbindlist(list(
  compute_moments(vrp_dt$vrp_predicted, "VRP Predicted"),
  compute_moments(vrp_dt$vrp_actual, "VRP Actual"),
  compute_moments(vrp_dt$vrp_error, "VRP Forecast Error"),
  compute_moments(vrp_dt$vix_predicted, "VIX Predicted"),
  compute_moments(vrp_dt$vix_actual, "VIX Actual"),
  compute_moments(vrp_dt$rv_predicted, "RV Predicted"),
  compute_moments(vrp_dt$rv_actual, "RV Actual")
))

cat("\n=== Descriptive Statistics ===\n")
print(vrp_descriptive[, .(Variable, N, Mean, SD, Skewness, Pct_Positive)])

#------------------------------------------------------------------
# 5. VRP FORECAST ACCURACY METRICS
#------------------------------------------------------------------

cat_progress("Computing VRP forecast accuracy metrics...")

# Remove NAs
valid_idx <- !is.na(vrp_dt$vrp_predicted) & !is.na(vrp_dt$vrp_actual)
vrp_valid <- vrp_dt[valid_idx]

# Basic regression metrics for VRP
vrp_residuals <- vrp_valid$vrp_actual - vrp_valid$vrp_predicted
vrp_rmse <- sqrt(mean(vrp_residuals^2))
vrp_mae <- mean(abs(vrp_residuals))
vrp_me <- mean(vrp_residuals)  # Mean error (bias)
vrp_r2 <- 1 - sum(vrp_residuals^2) / sum((vrp_valid$vrp_actual - mean(vrp_valid$vrp_actual))^2)

# QLIKE loss (heteroskedasticity-robust)
# Need to handle negative VRP values - use MSE for signed quantities
vrp_mse <- mean(vrp_residuals^2)

# Directional accuracy: did we predict the sign of VRP correctly?
vrp_sign_accuracy <- mean(sign(vrp_valid$vrp_predicted) == sign(vrp_valid$vrp_actual))

# Directional accuracy for VRP change
vrp_valid[, vrp_pred_change := vrp_predicted - shift(vrp_predicted, 1L)]
vrp_valid[, vrp_actual_change := vrp_actual - shift(vrp_actual, 1L)]
vrp_change_dir_acc <- mean(sign(vrp_valid$vrp_pred_change) == sign(vrp_valid$vrp_actual_change), na.rm = TRUE)

# Compile metrics
vrp_metrics <- data.table(
  Metric = c("RMSE", "MAE", "Mean Error (Bias)", "R²", "MSE",
             "Sign Accuracy", "Change Direction Accuracy"),
  Value = c(vrp_rmse, vrp_mae, vrp_me, vrp_r2, vrp_mse,
            vrp_sign_accuracy, vrp_change_dir_acc)
)

cat("\n=== VRP Forecast Accuracy ===\n")
print(vrp_metrics, digits = 4)

#------------------------------------------------------------------
# 6. DECOMPOSITION: VIX VS RV CONTRIBUTION TO VRP ERROR
#------------------------------------------------------------------

cat_progress("Decomposing VRP forecast error...")

# VRP_error = VRP_actual - VRP_predicted
#           = (VIX_actual - RV_actual) - (VIX_pred - RV_pred)
#           = (VIX_actual - VIX_pred) - (RV_actual - RV_pred)
#           = VIX_error - RV_error

vrp_valid[, `:=`(
  vix_error_contrib = vix_actual - vix_predicted,
  rv_error_contrib = rv_actual - rv_predicted
)]

# Verify decomposition
vrp_valid[, vrp_error_check := vix_error_contrib - rv_error_contrib]

# Variance decomposition
var_vrp_error <- var(vrp_valid$vrp_error, na.rm = TRUE)
var_vix_error <- var(vrp_valid$vix_error_contrib, na.rm = TRUE)
var_rv_error <- var(vrp_valid$rv_error_contrib, na.rm = TRUE)
cov_errors <- cov(vrp_valid$vix_error_contrib, vrp_valid$rv_error_contrib, use = "complete.obs")

# Var(VRP_error) = Var(VIX_error) + Var(RV_error) - 2*Cov(VIX_error, RV_error)
var_decomposition <- data.table(
  Component = c("Var(VRP_error)", "Var(VIX_error)", "Var(RV_error)", 
                "2*Cov(VIX_error, RV_error)", "Sum (check)"),
  Value = c(var_vrp_error, var_vix_error, var_rv_error, 
            2 * cov_errors, var_vix_error + var_rv_error - 2 * cov_errors)
)

# Percentage contribution
vix_pct_contrib <- (var_vix_error / var_vrp_error) * 100
rv_pct_contrib <- (var_rv_error / var_vrp_error) * 100
cov_pct_contrib <- (-2 * cov_errors / var_vrp_error) * 100

error_decomp_summary <- data.table(
  Source = c("VIX Forecast Error", "RV Forecast Error", "Covariance Term"),
  Variance = c(var_vix_error, var_rv_error, -2 * cov_errors),
  Pct_of_VRP_Var = c(vix_pct_contrib, rv_pct_contrib, cov_pct_contrib),
  RMSE = c(sqrt(var_vix_error), sqrt(var_rv_error), NA)
)

cat("\n=== VRP Error Decomposition ===\n")
print(error_decomp_summary, digits = 4)

# Correlation between VIX and RV forecast errors
error_correlation <- cor(vrp_valid$vix_error_contrib, vrp_valid$rv_error_contrib, use = "complete.obs")
cat(sprintf("\nCorrelation(VIX_error, RV_error): %.4f\n", error_correlation))

#------------------------------------------------------------------
# 7. MINCER-ZARNOWITZ REGRESSION FOR VRP
#------------------------------------------------------------------

cat_progress("Running Mincer-Zarnowitz regressions...")

# MZ regression: VRP_actual = alpha + beta * VRP_predicted + epsilon
# Efficient forecast: alpha = 0, beta = 1

mz_vrp <- lm(vrp_actual ~ vrp_predicted, data = vrp_valid)
mz_vrp_summary <- summary(mz_vrp)

mz_alpha <- coef(mz_vrp)[1]
mz_beta <- coef(mz_vrp)[2]

# HAC standard errors (Newey-West)
nw_vcov <- sandwich::NeweyWest(mz_vrp, lag = floor(nrow(vrp_valid)^(1/3)), prewhite = FALSE)
mz_se_hac <- sqrt(diag(nw_vcov))

# t-statistics with HAC
t_alpha_hac <- mz_alpha / mz_se_hac[1]
t_beta_hac <- (mz_beta - 1) / mz_se_hac[2]

# Wald test for joint efficiency: H0: alpha = 0, beta = 1
wald_test <- car::linearHypothesis(mz_vrp, c("(Intercept) = 0", "vrp_predicted = 1"), 
                                   vcov = nw_vcov)

mz_results <- data.table(
  Parameter = c("Alpha (Intercept)", "Beta (Slope)"),
  Estimate = c(mz_alpha, mz_beta),
  SE_OLS = mz_vrp_summary$coefficients[, 2],
  SE_HAC = mz_se_hac,
  t_HAC = c(t_alpha_hac, t_beta_hac),
  p_HAC = c(2 * pt(-abs(t_alpha_hac), df = nrow(vrp_valid) - 2),
            2 * pt(-abs(t_beta_hac), df = nrow(vrp_valid) - 2))
)

cat("\n=== Mincer-Zarnowitz Regression: VRP ===\n")
cat("VRP_actual = alpha + beta * VRP_predicted + epsilon\n")
cat("Efficient forecast: alpha = 0, beta = 1\n\n")
print(mz_results, digits = 4)
cat(sprintf("\nMZ R²: %.4f\n", mz_vrp_summary$r.squared))
cat(sprintf("Wald test (alpha=0, beta=1): F = %.4f, p = %.4f\n", 
            wald_test$F[2], wald_test$`Pr(>F)`[2]))

# Separate MZ regressions for VIX and RV forecasts
mz_vix <- lm(vix_actual ~ vix_predicted, data = vrp_valid)
mz_rv <- lm(rv_actual ~ rv_predicted, data = vrp_valid)

mz_comparison <- data.table(
  Model = c("VRP", "VIX", "RV"),
  Alpha = c(coef(mz_vrp)[1], coef(mz_vix)[1], coef(mz_rv)[1]),
  Beta = c(coef(mz_vrp)[2], coef(mz_vix)[2], coef(mz_rv)[2]),
  R2 = c(summary(mz_vrp)$r.squared, summary(mz_vix)$r.squared, summary(mz_rv)$r.squared)
)

cat("\n=== MZ Comparison: VRP vs Components ===\n")
print(mz_comparison, digits = 4)

#------------------------------------------------------------------
# 8. BENCHMARK COMPARISON
#------------------------------------------------------------------

cat_progress("Comparing against benchmarks...")

# Benchmark 1: Naive VRP (VRP_t as forecast for VRP_{t+1})
vrp_valid[, vrp_lag_1 := shift(vrp_actual, 1L)]
naive_residuals <- vrp_valid$vrp_actual - vrp_valid$vrp_lag_1
naive_rmse <- sqrt(mean(naive_residuals^2, na.rm = TRUE))

# Benchmark 2: Historical mean VRP
hist_mean_vrp <- mean(vrp_valid$vrp_actual, na.rm = TRUE)
mean_residuals <- vrp_valid$vrp_actual - hist_mean_vrp
mean_rmse <- sqrt(mean(mean_residuals^2, na.rm = TRUE))

# Benchmark 3: AR(1) model for VRP
ar1_vrp <- lm(vrp_actual ~ vrp_lag_1, data = vrp_valid)
ar1_fitted <- predict(ar1_vrp, vrp_valid)
ar1_residuals <- vrp_valid$vrp_actual - ar1_fitted
ar1_rmse <- sqrt(mean(ar1_residuals^2, na.rm = TRUE))

# Benchmark 4: Zero VRP assumption
zero_residuals <- vrp_valid$vrp_actual - 0
zero_rmse <- sqrt(mean(zero_residuals^2, na.rm = TRUE))

benchmark_comparison <- data.table(
  Model = c("XGBoost (VIX + RV)", "Random Walk (VRP_{t-1})", "AR(1)", 
            "Historical Mean", "Zero VRP"),
  RMSE = c(vrp_rmse, naive_rmse, ar1_rmse, mean_rmse, zero_rmse),
  Improvement_vs_RW = c((naive_rmse - vrp_rmse) / naive_rmse * 100,
                        0,
                        (naive_rmse - ar1_rmse) / naive_rmse * 100,
                        (naive_rmse - mean_rmse) / naive_rmse * 100,
                        (naive_rmse - zero_rmse) / naive_rmse * 100)
)

cat("\n=== Benchmark Comparison ===\n")
print(benchmark_comparison, digits = 4)

# Diebold-Mariano test: XGBoost vs Random Walk
dm_test <- function(e1, e2, h = 1) {
  d <- e1^2 - e2^2
  d <- d[!is.na(d)]
  n <- length(d)
  d_bar <- mean(d)
  var_d <- var(d) / n
  dm_stat <- d_bar / sqrt(var_d)
  p_value <- 2 * pnorm(-abs(dm_stat))
  list(statistic = dm_stat, p_value = p_value)
}

dm_vs_rw <- dm_test(vrp_residuals, naive_residuals[!is.na(naive_residuals)])
dm_vs_ar1 <- dm_test(vrp_residuals, ar1_residuals[!is.na(ar1_residuals)])

cat(sprintf("\nDiebold-Mariano vs Random Walk: stat = %.4f, p = %.4f\n", 
            dm_vs_rw$statistic, dm_vs_rw$p_value))
cat(sprintf("Diebold-Mariano vs AR(1): stat = %.4f, p = %.4f\n", 
            dm_vs_ar1$statistic, dm_vs_ar1$p_value))

#------------------------------------------------------------------
# 9. REGIME-CONDITIONAL ANALYSIS
#------------------------------------------------------------------

cat_progress("Analysing regime-conditional performance...")

# Define VIX regimes
vrp_valid[, vix_regime := cut(vix_t,
                              breaks = c(0, 15, 20, 25, 30, Inf),
                              labels = c("Very Low (<15)", "Low (15-20)", 
                                         "Medium (20-25)", "High (25-30)", 
                                         "Very High (>30)"))]

# Define VRP regimes (by actual VRP percentile)
vrp_percentiles <- quantile(vrp_valid$vrp_actual, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
vrp_valid[, vrp_regime := cut(vrp_actual,
                              breaks = c(-Inf, vrp_percentiles[1], vrp_percentiles[2], 
                                         vrp_percentiles[3], Inf),
                              labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)"))]

# Performance by VIX regime
regime_perf_vix <- vrp_valid[!is.na(vix_regime), .(
  N = .N,
  Mean_VRP_Pred = mean(vrp_predicted, na.rm = TRUE),
  Mean_VRP_Actual = mean(vrp_actual, na.rm = TRUE),
  RMSE = sqrt(mean((vrp_actual - vrp_predicted)^2, na.rm = TRUE)),
  MAE = mean(abs(vrp_actual - vrp_predicted), na.rm = TRUE),
  Bias = mean(vrp_actual - vrp_predicted, na.rm = TRUE),
  Sign_Acc = mean(sign(vrp_predicted) == sign(vrp_actual), na.rm = TRUE)
), by = vix_regime][order(vix_regime)]

# Performance by VRP regime
regime_perf_vrp <- vrp_valid[!is.na(vrp_regime), .(
  N = .N,
  Mean_VRP_Pred = mean(vrp_predicted, na.rm = TRUE),
  Mean_VRP_Actual = mean(vrp_actual, na.rm = TRUE),
  RMSE = sqrt(mean((vrp_actual - vrp_predicted)^2, na.rm = TRUE)),
  MAE = mean(abs(vrp_actual - vrp_predicted), na.rm = TRUE),
  Bias = mean(vrp_actual - vrp_predicted, na.rm = TRUE)
), by = vrp_regime][order(vrp_regime)]

cat("\n=== Performance by VIX Regime ===\n")
print(regime_perf_vix, digits = 4)

cat("\n=== Performance by VRP Quartile ===\n")
print(regime_perf_vrp, digits = 4)

#------------------------------------------------------------------
# 10. TRADING SIGNAL ANALYSIS
#------------------------------------------------------------------

cat_progress("Generating and analysing trading signals...")

# Trading logic based on VRP predictions:
# - High predicted VRP (VIX overpriced vs RV) → Sell volatility (short VIX, sell straddles)
# - Low predicted VRP (VIX underpriced vs RV) → Buy volatility (long VIX, buy straddles)

# Signal thresholds (based on historical distribution)
vrp_median <- median(vrp_valid$vrp_predicted, na.rm = TRUE)
vrp_q25 <- quantile(vrp_valid$vrp_predicted, 0.25, na.rm = TRUE)
vrp_q75 <- quantile(vrp_valid$vrp_predicted, 0.75, na.rm = TRUE)

vrp_valid[, `:=`(
  # Basic signal: long vol if VRP_pred < 0, short vol if VRP_pred > 0
  signal_sign = fifelse(vrp_predicted > 0, -1L, 1L),
  
  # Threshold signal: only trade if VRP_pred outside middle 50%
  signal_threshold = fifelse(vrp_predicted > vrp_q75, -1L,
                             fifelse(vrp_predicted < vrp_q25, 1L, 0L)),
  
  # Z-score signal (need rolling mean/sd)
  vrp_pred_mean_252 = frollmean(vrp_predicted, n = 252L, align = "right"),
  vrp_pred_sd_252 = frollapply(vrp_predicted, n = 252L, FUN = sd, align = "right")
)]

vrp_valid[, vrp_pred_zscore := (vrp_predicted - vrp_pred_mean_252) / (vrp_pred_sd_252 + 1e-10)]
vrp_valid[, signal_zscore := fifelse(vrp_pred_zscore > 1, -1L,
                                     fifelse(vrp_pred_zscore < -1, 1L, 0L))]

# Evaluate signals using actual VRP
# If signal = -1 (short vol) and actual VRP > 0 → correct
# If signal = 1 (long vol) and actual VRP < 0 → correct

vrp_valid[, `:=`(
  correct_sign = as.integer((signal_sign == -1 & vrp_actual > 0) | 
                              (signal_sign == 1 & vrp_actual < 0)),
  correct_threshold = as.integer((signal_threshold == -1 & vrp_actual > 0) | 
                                   (signal_threshold == 1 & vrp_actual < 0) |
                                   signal_threshold == 0),
  correct_zscore = as.integer((signal_zscore == -1 & vrp_actual > 0) | 
                                (signal_zscore == 1 & vrp_actual < 0) |
                                signal_zscore == 0)
)]

# Signal performance
signal_perf <- data.table(
  Signal_Type = c("Sign-based", "Threshold (Q25/Q75)", "Z-score (±1σ)"),
  Trades = c(sum(vrp_valid$signal_sign != 0, na.rm = TRUE),
             sum(vrp_valid$signal_threshold != 0, na.rm = TRUE),
             sum(vrp_valid$signal_zscore != 0, na.rm = TRUE)),
  Accuracy = c(mean(vrp_valid$correct_sign, na.rm = TRUE),
               mean(vrp_valid$correct_threshold[vrp_valid$signal_threshold != 0], na.rm = TRUE),
               mean(vrp_valid$correct_zscore[vrp_valid$signal_zscore != 0], na.rm = TRUE))
)

cat("\n=== Trading Signal Performance ===\n")
print(signal_perf, digits = 4)

# Confusion matrix for sign-based signal
confusion <- table(Predicted = vrp_valid$signal_sign, 
                   Actual = sign(vrp_valid$vrp_actual))
cat("\n=== Confusion Matrix (Sign-based Signal) ===\n")
print(confusion)

#------------------------------------------------------------------
# 11. TEMPORAL STABILITY ANALYSIS
#------------------------------------------------------------------

cat_progress("Analysing temporal stability...")

# Rolling window performance (63-day ≈ 3 months)
window_size <- 63
rolling_metrics <- data.frame()

for (i in window_size:nrow(vrp_valid)) {
  window_idx <- (i - window_size + 1):i
  y_w <- vrp_valid$vrp_actual[window_idx]
  pred_w <- vrp_valid$vrp_predicted[window_idx]
  
  rmse_w <- sqrt(mean((y_w - pred_w)^2, na.rm = TRUE))
  r2_w <- 1 - sum((y_w - pred_w)^2, na.rm = TRUE) / sum((y_w - mean(y_w, na.rm = TRUE))^2)
  sign_acc_w <- mean(sign(pred_w) == sign(y_w), na.rm = TRUE)
  
  rolling_metrics <- rbind(rolling_metrics, data.frame(
    date = vrp_valid$date[i],
    rmse = rmse_w,
    r2 = r2_w,
    sign_accuracy = sign_acc_w
  ))
}

setDT(rolling_metrics)

mean_rolling_rmse <- mean(rolling_metrics$rmse)
sd_rolling_rmse <- sd(rolling_metrics$rmse)
degradation_threshold <- mean_rolling_rmse + 2 * sd_rolling_rmse
degradation_periods <- rolling_metrics[rmse > degradation_threshold]

cat(sprintf("\nRolling RMSE: mean = %.4f, sd = %.4f\n", mean_rolling_rmse, sd_rolling_rmse))
cat(sprintf("Degradation periods (RMSE > %.4f): %d days\n", 
            degradation_threshold, nrow(degradation_periods)))

#------------------------------------------------------------------
# 12. BOOTSTRAP CONFIDENCE INTERVALS
#------------------------------------------------------------------

cat_progress("Computing bootstrap confidence intervals...")

set.seed(42)
n_bootstrap <- 10000

bootstrap_vrp_rmse <- numeric(n_bootstrap)
bootstrap_vrp_r2 <- numeric(n_bootstrap)
bootstrap_sign_acc <- numeric(n_bootstrap)

for (b in 1:n_bootstrap) {
  boot_idx <- sample(1:nrow(vrp_valid), replace = TRUE)
  y_boot <- vrp_valid$vrp_actual[boot_idx]
  pred_boot <- vrp_valid$vrp_predicted[boot_idx]
  
  bootstrap_vrp_rmse[b] <- sqrt(mean((y_boot - pred_boot)^2))
  ss_res <- sum((y_boot - pred_boot)^2)
  ss_tot <- sum((y_boot - mean(y_boot))^2)
  bootstrap_vrp_r2[b] <- 1 - ss_res / ss_tot
  bootstrap_sign_acc[b] <- mean(sign(pred_boot) == sign(y_boot))
}

rmse_ci <- quantile(bootstrap_vrp_rmse, c(0.025, 0.975))
r2_ci <- quantile(bootstrap_vrp_r2, c(0.025, 0.975))
sign_acc_ci <- quantile(bootstrap_sign_acc, c(0.025, 0.975))

cat("\n=== Bootstrap 95% Confidence Intervals ===\n")
cat(sprintf("RMSE: [%.4f, %.4f]\n", rmse_ci[1], rmse_ci[2]))
cat(sprintf("R²: [%.4f, %.4f]\n", r2_ci[1], r2_ci[2]))
cat(sprintf("Sign Accuracy: [%.4f, %.4f]\n", sign_acc_ci[1], sign_acc_ci[2]))

#------------------------------------------------------------------
# 13. VISUALISATIONS
#------------------------------------------------------------------

cat_progress("Generating visualisations...")

# 13.1 Predicted vs Actual VRP
pdf("results/figures/vrp_comparison/01_vrp_pred_vs_actual.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

# Scatter plot
plot(vrp_valid$vrp_predicted, vrp_valid$vrp_actual,
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = "Predicted VRP", ylab = "Actual VRP",
     main = sprintf("VRP: Predicted vs Actual (R² = %.4f)", vrp_r2))
abline(0, 1, col = "red", lwd = 2)
abline(mz_vrp, col = "blue", lty = 2)
abline(h = 0, v = 0, col = "grey", lty = 3)
legend("topleft", c("Perfect forecast", sprintf("MZ: α=%.2f, β=%.2f", mz_alpha, mz_beta)),
       col = c("red", "blue"), lty = c(1, 2), cex = 0.8)

# Time series
plot(vrp_valid$date, vrp_valid$vrp_actual, type = "l", col = "black", lwd = 0.5,
     xlab = "Date", ylab = "VRP",
     main = "VRP Time Series: Predicted vs Actual")
lines(vrp_valid$date, vrp_valid$vrp_predicted, col = "blue", lwd = 0.5)
abline(h = 0, col = "red", lty = 2)
legend("topright", c("Actual", "Predicted"), col = c("black", "blue"), lwd = 1, cex = 0.8)

# Error distribution
hist(vrp_valid$vrp_error, breaks = 50, col = "lightblue", probability = TRUE,
     main = "VRP Forecast Error Distribution", xlab = "Error")
curve(dnorm(x, mean = vrp_me, sd = sqrt(vrp_mse)), add = TRUE, col = "red", lwd = 2)
abline(v = 0, col = "red", lty = 2)
abline(v = vrp_me, col = "blue", lty = 2)

# Q-Q plot
qqnorm(vrp_valid$vrp_error, pch = 16, cex = 0.5, main = "Q-Q Plot: VRP Errors")
qqline(vrp_valid$vrp_error, col = "red")

dev.off()

# 13.2 Component analysis
pdf("results/figures/vrp_comparison/02_component_analysis.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

# VIX: predicted vs actual
plot(vrp_valid$vix_predicted, vrp_valid$vix_actual,
     pch = 16, cex = 0.5, col = rgb(1, 0, 0, 0.3),
     xlab = "Predicted VIX", ylab = "Actual VIX",
     main = sprintf("VIX Forecast (R² = %.4f)", summary(mz_vix)$r.squared))
abline(0, 1, col = "red", lwd = 2)
abline(mz_vix, col = "blue", lty = 2)

# RV: predicted vs actual
plot(vrp_valid$rv_predicted, vrp_valid$rv_actual,
     pch = 16, cex = 0.5, col = rgb(0, 0.5, 0, 0.3),
     xlab = "Predicted RV", ylab = "Actual RV",
     main = sprintf("RV Forecast (R² = %.4f)", summary(mz_rv)$r.squared))
abline(0, 1, col = "red", lwd = 2)
abline(mz_rv, col = "blue", lty = 2)

# Error contribution scatter
plot(vrp_valid$vix_error_contrib, vrp_valid$rv_error_contrib,
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = "VIX Forecast Error", ylab = "RV Forecast Error",
     main = sprintf("Error Correlation: r = %.4f", error_correlation))
abline(h = 0, v = 0, col = "red", lty = 2)
abline(lm(rv_error_contrib ~ vix_error_contrib, data = vrp_valid), col = "blue", lty = 2)

# Error decomposition bar chart
barplot(c(vix_pct_contrib, rv_pct_contrib, abs(cov_pct_contrib)),
        names.arg = c("VIX Error", "RV Error", "|Cov Term|"),
        main = "VRP Error Variance Decomposition",
        ylab = "% of Total Variance",
        col = c("coral", "lightgreen", "lightblue"))

dev.off()

# 13.3 Regime analysis
pdf("results/figures/vrp_comparison/03_regime_analysis.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

# RMSE by VIX regime
barplot(regime_perf_vix$RMSE, names.arg = regime_perf_vix$vix_regime,
        las = 2, cex.names = 0.7,
        main = "VRP Forecast RMSE by VIX Regime",
        ylab = "RMSE", col = rainbow(nrow(regime_perf_vix)))

# Sign accuracy by VIX regime
barplot(regime_perf_vix$Sign_Acc * 100, names.arg = regime_perf_vix$vix_regime,
        las = 2, cex.names = 0.7,
        main = "VRP Sign Accuracy by VIX Regime",
        ylab = "Accuracy (%)", col = rainbow(nrow(regime_perf_vix)))
abline(h = 50, col = "red", lty = 2)

# Bias by VIX regime
barplot(regime_perf_vix$Bias, names.arg = regime_perf_vix$vix_regime,
        las = 2, cex.names = 0.7,
        main = "VRP Forecast Bias by VIX Regime",
        ylab = "Bias (Actual - Predicted)", col = rainbow(nrow(regime_perf_vix)))
abline(h = 0, col = "red", lty = 2)

# VRP predicted vs actual by regime
boxplot(vrp_error ~ vix_regime, data = vrp_valid,
        las = 2, cex.axis = 0.7,
        main = "VRP Forecast Error by VIX Regime",
        ylab = "Error", col = rainbow(nlevels(vrp_valid$vix_regime)))
abline(h = 0, col = "red", lty = 2)

dev.off()

# 13.4 Temporal stability
pdf("results/figures/vrp_comparison/04_temporal_stability.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

# Rolling RMSE
plot(rolling_metrics$date, rolling_metrics$rmse, type = "l",
     col = "blue", lwd = 1,
     main = "Rolling 63-day VRP RMSE", xlab = "Date", ylab = "RMSE")
abline(h = mean_rolling_rmse, col = "green", lty = 2)
abline(h = degradation_threshold, col = "red", lty = 2)
legend("topright", c("RMSE", "Mean", "2σ threshold"),
       col = c("blue", "green", "red"), lty = c(1, 2, 2), cex = 0.8)

# Rolling R²
plot(rolling_metrics$date, rolling_metrics$r2, type = "l",
     col = "blue", lwd = 1,
     main = "Rolling 63-day VRP R²", xlab = "Date", ylab = "R²")
abline(h = mean(rolling_metrics$r2), col = "green", lty = 2)

# Rolling sign accuracy
plot(rolling_metrics$date, rolling_metrics$sign_accuracy * 100, type = "l",
     col = "blue", lwd = 1, ylim = c(30, 80),
     main = "Rolling 63-day VRP Sign Accuracy", xlab = "Date", ylab = "Accuracy (%)")
abline(h = 50, col = "red", lty = 2)
abline(h = mean(rolling_metrics$sign_accuracy) * 100, col = "green", lty = 2)

# Cumulative forecast error
cumulative_error <- cumsum(vrp_valid$vrp_error)
plot(vrp_valid$date, cumulative_error, type = "l",
     col = "blue", lwd = 1,
     main = "Cumulative VRP Forecast Error", xlab = "Date", ylab = "Cumulative Error")
abline(h = 0, col = "red", lty = 2)

dev.off()

# 13.5 Trading signals
pdf("results/figures/vrp_comparison/05_trading_signals.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

# VRP predicted with signals
plot(vrp_valid$date, vrp_valid$vrp_predicted, type = "l",
     col = "blue", lwd = 0.5,
     main = "VRP Predicted with Trading Signals", xlab = "Date", ylab = "Predicted VRP")
abline(h = c(vrp_q25, vrp_q75), col = "orange", lty = 2)
abline(h = 0, col = "red", lty = 2)

# Signal histogram
hist(vrp_valid$vrp_predicted, breaks = 50, col = "lightblue",
     main = "Distribution of Predicted VRP", xlab = "Predicted VRP")
abline(v = c(vrp_q25, vrp_q75), col = "orange", lwd = 2)
abline(v = 0, col = "red", lwd = 2)

# Accuracy by signal strength
vrp_valid[, pred_abs := abs(vrp_predicted)]
vrp_valid[, pred_quintile := cut(pred_abs, breaks = quantile(pred_abs, probs = seq(0, 1, 0.2), na.rm = TRUE),
                                 include.lowest = TRUE, labels = c("Q1", "Q2", "Q3", "Q4", "Q5"))]

acc_by_strength <- vrp_valid[!is.na(pred_quintile), .(
  Accuracy = mean(sign(vrp_predicted) == sign(vrp_actual), na.rm = TRUE)
), by = pred_quintile][order(pred_quintile)]

barplot(acc_by_strength$Accuracy * 100, names.arg = acc_by_strength$pred_quintile,
        main = "Sign Accuracy by Prediction Confidence",
        xlab = "Quintile of |VRP_pred|", ylab = "Accuracy (%)",
        col = "lightblue")
abline(h = 50, col = "red", lty = 2)

# Cumulative P&L (simplified: assume unit bet, P&L = signal * actual_VRP)
vrp_valid[, pnl := -signal_sign * vrp_actual]  # Short vol when VRP high
vrp_valid[, cumulative_pnl := cumsum(fifelse(is.na(pnl), 0, pnl))]

plot(vrp_valid$date, vrp_valid$cumulative_pnl, type = "l",
     col = "darkgreen", lwd = 1,
     main = "Cumulative P&L (Sign-based Strategy)", 
     xlab = "Date", ylab = "Cumulative P&L (vol points)")
abline(h = 0, col = "red", lty = 2)

dev.off()

#------------------------------------------------------------------
# 14. SAVE RESULTS
#------------------------------------------------------------------

cat_progress("Saving results...")

# Save merged predictions
saveRDS(vrp_dt, "results/models/vrp_comparison/vrp_combined_predictions.rds")
writexl::write_xlsx(as.data.frame(vrp_dt), "results/tables/vrp_comparison/vrp_combined_predictions.xlsx")

# Save metrics and summaries
write.csv(vrp_metrics, "results/tables/vrp_comparison/vrp_forecast_metrics.csv", row.names = FALSE)
write.csv(vrp_descriptive, "results/tables/vrp_comparison/vrp_descriptive_stats.csv", row.names = FALSE)
write.csv(error_decomp_summary, "results/tables/vrp_comparison/vrp_error_decomposition.csv", row.names = FALSE)
write.csv(mz_results, "results/tables/vrp_comparison/vrp_mincer_zarnowitz.csv", row.names = FALSE)
write.csv(mz_comparison, "results/tables/vrp_comparison/mz_comparison_components.csv", row.names = FALSE)
write.csv(benchmark_comparison, "results/tables/vrp_comparison/vrp_benchmark_comparison.csv", row.names = FALSE)
write.csv(regime_perf_vix, "results/tables/vrp_comparison/vrp_regime_performance_vix.csv", row.names = FALSE)
write.csv(regime_perf_vrp, "results/tables/vrp_comparison/vrp_regime_performance_vrp.csv", row.names = FALSE)
write.csv(signal_perf, "results/tables/vrp_comparison/vrp_signal_performance.csv", row.names = FALSE)
write.csv(rolling_metrics, "results/tables/vrp_comparison/vrp_rolling_metrics.csv", row.names = FALSE)

# Compile full results
vrp_comparison_results <- list(
  # Data
  predictions = vrp_dt,
  
  # Metrics
  forecast_metrics = vrp_metrics,
  descriptive_stats = vrp_descriptive,
  
  # Decomposition
  error_decomposition = error_decomp_summary,
  var_decomposition = var_decomposition,
  error_correlation = error_correlation,
  
  # Mincer-Zarnowitz
  mz_vrp = list(alpha = mz_alpha, beta = mz_beta, r2 = mz_vrp_summary$r.squared,
                se_hac = mz_se_hac, wald_test = wald_test),
  mz_comparison = mz_comparison,
  
  # Benchmarks
  benchmark_comparison = benchmark_comparison,
  dm_tests = list(vs_rw = dm_vs_rw, vs_ar1 = dm_vs_ar1),
  
  # Regime analysis
  regime_perf_vix = regime_perf_vix,
  regime_perf_vrp = regime_perf_vrp,
  
  # Trading signals
  signal_performance = signal_perf,
  signal_thresholds = c(q25 = vrp_q25, median = vrp_median, q75 = vrp_q75),
  
  # Temporal stability
  rolling_metrics = rolling_metrics,
  degradation_periods = degradation_periods,
  
  # Bootstrap CIs
  bootstrap_ci = list(
    rmse = rmse_ci,
    r2 = r2_ci,
    sign_accuracy = sign_acc_ci
  )
)

saveRDS(vrp_comparison_results, "results/models/vrp_comparison/vrp_comparison_full_results.rds")

#------------------------------------------------------------------
# 15. FINAL SUMMARY
#------------------------------------------------------------------

cat_progress(paste(rep("=", 70), collapse = ""))
cat_progress("VRP COMPARISON ANALYSIS COMPLETE")
cat_progress(paste(rep("=", 70), collapse = ""))

cat("\n========== VRP FORECAST PERFORMANCE ==========\n")
print(vrp_metrics, digits = 4)

cat("\n========== ERROR DECOMPOSITION ==========\n")
print(error_decomp_summary[, .(Source, Pct_of_VRP_Var)], digits = 4)

cat("\n========== MINCER-ZARNOWITZ ==========\n")
cat(sprintf("VRP: α = %.4f, β = %.4f, R² = %.4f\n", mz_alpha, mz_beta, mz_vrp_summary$r.squared))
cat(sprintf("VIX: α = %.4f, β = %.4f, R² = %.4f\n", 
            coef(mz_vix)[1], coef(mz_vix)[2], summary(mz_vix)$r.squared))
cat(sprintf("RV:  α = %.4f, β = %.4f, R² = %.4f\n", 
            coef(mz_rv)[1], coef(mz_rv)[2], summary(mz_rv)$r.squared))

cat("\n========== BENCHMARK COMPARISON ==========\n")
print(benchmark_comparison, digits = 4)

cat("\n========== TRADING SIGNAL PERFORMANCE ==========\n")
print(signal_perf, digits = 4)

cat("\n========== BOOTSTRAP 95% CIs ==========\n")
cat(sprintf("RMSE: [%.4f, %.4f]\n", rmse_ci[1], rmse_ci[2]))
cat(sprintf("R²: [%.4f, %.4f]\n", r2_ci[1], r2_ci[2]))
cat(sprintf("Sign Accuracy: [%.4f, %.4f]\n", sign_acc_ci[1], sign_acc_ci[2]))

cat_progress(paste(rep("=", 70), collapse = ""))
cat_progress("Output files saved to:")
cat_progress("  results/models/vrp_comparison/")
cat_progress("  results/tables/vrp_comparison/")
cat_progress("  results/figures/vrp_comparison/")
cat_progress(paste(rep("=", 70), collapse = ""))

################################################################################
# END OF SCRIPT
################################################################################