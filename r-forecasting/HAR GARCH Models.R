################################################################################
# 01_HAR_GARCH_Models.R - CORRECTED VERSION
# 
# STATIONARITY DECISIONS:
# - HAR: Applied to VIX LEVELS (VIX is mean-reverting, bounded, stationary-like)
# - GARCH: Applied to VIX LOG RETURNS (requires stationarity)
#   -> AR(1) mean model provides return forecast
#   -> Convert return forecast to level forecast
#
# OUTPUTS:
# - HAR: Direct VIX level point forecast
# - AR(1)-GARCH(1,1): Return forecast -> Level forecast + Volatility forecast
# - Classification: Direction prediction (up/down) from both models
################################################################################

#------------------------------------------------------------------
# 0. SOURCE SETUP AND LOAD DATA
#------------------------------------------------------------------

source("Setup.R")

# Load additional packages 
garch_packages <- c("rugarch", "xts", "zoo", "parallel", "tseries", "sandwich", "lmtest")
install_and_load(garch_packages)

# Load data
merged_dt <- readRDS("data/merged_data.rds")
split_info <- readRDS("results/models/split_info.rds")
target_config <- readRDS("results/models/target_config.rds")

# Convert to xts
vix_xts <- xts(merged_dt$vix_close, order.by = merged_dt$date)
colnames(vix_xts) <- "vix_close"

# Compute log returns for GARCH
vix_returns_xts <- diff(log(vix_xts))
colnames(vix_returns_xts) <- "vix_return"

# Align dates (lose first observation due to differencing)
common_dates <- index(vix_returns_xts)
vix_levels <- vix_xts[common_dates]
vix_returns <- vix_returns_xts[common_dates]

cat_progress(sprintf("Data loaded: %d observations from %s to %s",
                     length(common_dates), min(common_dates), max(common_dates)))

#------------------------------------------------------------------
# 1. STATIONARITY TESTS
#------------------------------------------------------------------

cat_progress("Performing stationarity tests...")

# Remove NAs for testing
vix_levels_clean <- na.omit(as.numeric(vix_levels))
vix_returns_clean <- na.omit(as.numeric(vix_returns))

# ADF test on VIX levels
adf_levels <- adf.test(vix_levels_clean, alternative = "stationary")
cat(sprintf("\nADF Test - VIX Levels: statistic = %.3f, p-value = %.4f\n", 
            adf_levels$statistic, adf_levels$p.value))

# ADF test on VIX returns
adf_returns <- adf.test(vix_returns_clean, alternative = "stationary")
cat(sprintf("ADF Test - VIX Returns: statistic = %.3f, p-value = %.4f\n", 
            adf_returns$statistic, adf_returns$p.value))

# KPSS test (null = stationary)
kpss_levels <- kpss.test(vix_levels_clean, null = "Level")
kpss_returns <- kpss.test(vix_returns_clean, null = "Level")

cat(sprintf("KPSS Test - VIX Levels: statistic = %.3f, p-value = %.4f\n", 
            kpss_levels$statistic, kpss_levels$p.value))
cat(sprintf("KPSS Test - VIX Returns: statistic = %.3f, p-value = %.4f\n", 
            kpss_returns$statistic, kpss_returns$p.value))

stationarity_results <- data.frame(
  Series = c("VIX_Levels", "VIX_Returns"),
  ADF_Statistic = c(adf_levels$statistic, adf_returns$statistic),
  ADF_Pvalue = c(adf_levels$p.value, adf_returns$p.value),
  ADF_Stationary = c(adf_levels$p.value < 0.05, adf_returns$p.value < 0.05),
  KPSS_Statistic = c(kpss_levels$statistic, kpss_returns$statistic),
  KPSS_Pvalue = c(kpss_levels$p.value, kpss_returns$p.value),
  KPSS_Stationary = c(kpss_levels$p.value > 0.05, kpss_returns$p.value > 0.05)
)

cat("\n========== STATIONARITY TEST RESULTS ==========\n")
print(stationarity_results)

# Decision logging
cat("\nDecision: HAR will use VIX LEVELS (mean-reverting, ADF typically rejects unit root)\n")
cat("Decision: GARCH will use VIX RETURNS (definitely stationary)\n")

#------------------------------------------------------------------
# 2. FEATURE ENGINEERING FOR HAR MODEL (VIX LEVELS)
#------------------------------------------------------------------

cat_progress("Creating HAR features from VIX levels...")

har_features <- function(vix_series) {
  # Daily component (lagged by 1)
  vix_d <- as.numeric(lag(vix_series, 1))
  
  
  # Weekly average (past 5 days, lagged by 1)
  vix_w <- rollapply(vix_series, width = 5, FUN = mean, align = "right", fill = NA)
  vix_w <- as.numeric(lag(vix_w, 1))
  
  # Monthly average (past 22 days, lagged by 1)
  vix_m <- rollapply(vix_series, width = 22, FUN = mean, align = "right", fill = NA)
  vix_m <- as.numeric(lag(vix_m, 1))
  
  data.frame(
    vix_d = vix_d,
    vix_w = vix_w,
    vix_m = vix_m
  )
}

# Build HAR dataframe
har_df <- har_features(vix_levels)
har_df$date <- common_dates
har_df$vix_level <- as.numeric(vix_levels)
har_df$vix_return <- as.numeric(vix_returns)

# Targets (next day values)
har_df$target_level <- as.numeric(lead(vix_levels, 1))
har_df$target_return <- as.numeric(lead(vix_returns, 1))

# Direction target: 1 if VIX goes up, 0 if down
har_df$target_direction <- ifelse(har_df$target_level > har_df$vix_level, 1, 0)

# Remove NAs
har_df <- na.omit(har_df)

cat_progress(sprintf("HAR dataframe created: %d observations", nrow(har_df)))

#------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
#------------------------------------------------------------------

split_date <- split_info$split_date

train_idx <- har_df$date <= split_date
test_idx <- har_df$date > split_date
train_df <- har_df[train_idx, ]
test_df <- har_df[test_idx, ]

cat_progress(sprintf("Train set: %d obs (%s to %s)", 
                     nrow(train_df), min(train_df$date), max(train_df$date)))
cat_progress(sprintf("Test set:  %d obs (%s to %s)", 
                     nrow(test_df), min(test_df$date), max(test_df$date)))

#------------------------------------------------------------------
# 4. EVALUATION METRICS FUNCTIONS
#------------------------------------------------------------------

calc_regression_metrics <- function(actual, predicted) {
  valid_idx <- !is.na(actual) & !is.na(predicted)
  actual <- actual[valid_idx]
  predicted <- predicted[valid_idx]
  
  n <- length(actual)
  if (n == 0) return(NULL)
  
  errors <- actual - predicted
  
  rmse <- sqrt(mean(errors^2))
  mae <- mean(abs(errors))
  mape <- mean(abs(errors / actual)) * 100
  
  ss_res <- sum(errors^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  me <- mean(errors)  # Bias
  
  # Theil's U (relative to naive random walk)
  naive_forecast <- lag(actual)
  naive_errors <- actual[-1] - naive_forecast[-1]
  theil_u <- rmse / sqrt(mean(naive_errors^2, na.rm = TRUE))
  
  list(
    RMSE = rmse,
    MAE = mae,
    MAPE = mape,
    R2 = r2,
    ME = me,
    Theil_U = theil_u,
    N = n
  )
}

calc_classification_metrics <- function(actual, predicted_prob, threshold = 0.5) {
  valid_idx <- !is.na(actual) & !is.na(predicted_prob)
  actual <- actual[valid_idx]
  predicted_prob <- predicted_prob[valid_idx]
  
  n <- length(actual)
  if (n == 0) return(NULL)
  
  predicted_class <- ifelse(predicted_prob >= threshold, 1, 0)
  
  tp <- sum(predicted_class == 1 & actual == 1)
  tn <- sum(predicted_class == 0 & actual == 0)
  fp <- sum(predicted_class == 1 & actual == 0)
  fn <- sum(predicted_class == 0 & actual == 1)
  
  accuracy <- (tp + tn) / n
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  # AUC calculation
  auc <- tryCatch({
    ord <- order(predicted_prob, decreasing = TRUE)
    actual_sorted <- actual[ord]
    tpr <- cumsum(actual_sorted) / sum(actual_sorted)
    fpr <- cumsum(1 - actual_sorted) / sum(1 - actual_sorted)
    sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
  }, error = function(e) NA)
  
  list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    F1 = f1,
    AUC = auc,
    N = n
  )
}

#------------------------------------------------------------------
# 5. FIT HAR MODEL (VIX LEVELS -> VIX LEVELS)
#------------------------------------------------------------------

cat_progress("Fitting HAR model on VIX levels...")

har_formula <- target_level ~ vix_d + vix_w + vix_m
har_model <- lm(har_formula, data = train_df)

cat("\n========== HAR MODEL SUMMARY ==========\n")
print(summary(har_model))

# HAC standard errors (Newey-West)
har_nw_se <- tryCatch({
  coeftest(har_model, vcov = NeweyWest(har_model))
}, error = function(e) {
  cat("Newey-West SE computation failed, using OLS SE\n")
  summary(har_model)$coefficients
})
cat("\nHAR Coefficients with Newey-West SE:\n")
print(har_nw_se)

# Predictions
har_train_pred <- predict(har_model, newdata = train_df)
har_test_pred <- predict(har_model, newdata = test_df)

# Direction probability using prediction standard errors
har_test_pred_se <- predict(har_model, newdata = test_df, se.fit = TRUE)

# P(VIX goes up) = P(predicted > current)
# Using residual standard error for uncertainty
har_residual_se <- summary(har_model)$sigma
har_test_direction_prob <- pnorm(
  (har_test_pred - test_df$vix_level) / har_residual_se
)

# Store HAR results
har_results <- list(
  model = har_model,
  train_predictions = data.frame(
    date = train_df$date,
    actual_level = train_df$target_level,
    pred_level = har_train_pred,
    actual_direction = train_df$target_direction,
    pred_direction_prob = pnorm((har_train_pred - train_df$vix_level) / har_residual_se)
  ),
  test_predictions = data.frame(
    date = test_df$date,
    actual_level = test_df$target_level,
    pred_level = har_test_pred,
    actual_direction = test_df$target_direction,
    pred_direction_prob = har_test_direction_prob
  )
)

cat_progress("HAR model fitted successfully")

#------------------------------------------------------------------
# 6. FIT AR(1)-GARCH(1,1) MODEL (VIX RETURNS)
#------------------------------------------------------------------

cat_progress("Fitting AR(1)-GARCH(1,1) model on VIX returns...")

# Prepare return series
train_returns <- xts(train_df$vix_return, order.by = train_df$date)
all_returns <- xts(har_df$vix_return, order.by = har_df$date)

# AR(1)-GARCH(1,1) specification
# Mean model: r_t = mu + phi * r_{t-1} + epsilon_t
# Variance model: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = "std"  # Student-t for fat tails
)

# Fit on training data
garch_fit <- ugarchfit(
  spec = garch_spec,
  data = train_returns,
  solver = "hybrid"
)

cat("\n========== AR(1)-GARCH(1,1) MODEL SUMMARY ==========\n")
print(garch_fit)

# Extract fitted parameters
garch_params <- coef(garch_fit)
cat("\nFitted Parameters:\n")
print(garch_params)

# Persistence check
alpha <- garch_params["alpha1"]
beta <- garch_params["beta1"]
persistence <- alpha + beta
cat(sprintf("\nVolatility Persistence (alpha + beta): %.4f\n", persistence))
if (persistence >= 1) {
  warning("GARCH persistence >= 1, model may be mis-specified")
}

#------------------------------------------------------------------
# 7. ROLLING FORECAST FOR TEST PERIOD
#------------------------------------------------------------------

cat_progress("Generating rolling forecasts for test period...")

n_test <- nrow(test_df)

# Rolling forecast with monthly refitting
roll_forecast <- ugarchroll(
  spec = garch_spec,
  data = all_returns,
  n.ahead = 1,
  forecast.length = n_test,
  refit.every = 22,  # Refit monthly
  refit.window = "moving",
  window.size = nrow(train_df),
  calculate.VaR = TRUE,
  VaR.alpha = c(0.01, 0.05)
)

# Extract forecasts
roll_preds <- as.data.frame(roll_forecast)

cat("\nRolling forecast columns:", colnames(roll_preds), "\n")

# Mean forecast (from AR(1) component)
pred_returns <- roll_preds$Mu

# Volatility forecast (from GARCH component)
pred_sigma <- roll_preds$Sigma
pred_variance <- pred_sigma^2

# Convert return forecast to level forecast
# VIX_{t+1} = VIX_t * exp(r_{t+1})
test_current_level <- test_df$vix_level
garch_pred_levels <- test_current_level * exp(pred_returns)

# Direction probability: P(r_{t+1} > 0) using predicted mean and sigma
garch_direction_prob <- pnorm(pred_returns / pred_sigma)

# Debug output
cat("\n### GARCH Forecast Summary ###\n")
cat(sprintf("Predicted returns - Mean: %.6f, SD: %.6f\n", 
            mean(pred_returns), sd(pred_returns)))
cat(sprintf("Predicted sigma   - Mean: %.6f, SD: %.6f\n", 
            mean(pred_sigma), sd(pred_sigma)))
cat(sprintf("Predicted levels  - Mean: %.2f, SD: %.2f\n", 
            mean(garch_pred_levels), sd(garch_pred_levels)))

# Store GARCH results
garch_results <- list(
  GARCH = list(
    model = garch_fit,
    roll_forecast = roll_forecast,
    test_predictions = data.frame(
      date = test_df$date,
      actual_level = test_df$target_level,
      pred_level = garch_pred_levels,
      actual_return = test_df$target_return,
      pred_return = pred_returns,
      pred_sigma = pred_sigma,
      pred_variance = pred_variance,
      actual_direction = test_df$target_direction,
      pred_direction_prob = garch_direction_prob
    ),
    VaR_01 = roll_preds$`alpha(1%)`,
    VaR_05 = roll_preds$`alpha(5%)`
  )
)

cat_progress("AR(1)-GARCH(1,1) forecasts generated successfully")

#------------------------------------------------------------------
# 8. GARCH DIAGNOSTICS
#------------------------------------------------------------------

cat_progress("Computing GARCH diagnostics...")

# Standardised residuals
std_resid <- residuals(garch_fit, standardize = TRUE)

# Ljung-Box test on standardised residuals (should show no autocorrelation)
lb_resid <- Box.test(std_resid, lag = 10, type = "Ljung-Box")

# Ljung-Box test on squared standardised residuals (should show no ARCH effects)
lb_sq_resid <- Box.test(std_resid^2, lag = 10, type = "Ljung-Box")

# ARCH-LM test
arch_lm <- tryCatch({
  ArchTest <- function(x, lags = 10) {
    x <- as.numeric(x)
    n <- length(x)
    x2 <- x^2
    X <- embed(x2, lags + 1)
    fit <- lm(X[, 1] ~ X[, -1])
    chi2 <- n * summary(fit)$r.squared
    p_value <- 1 - pchisq(chi2, df = lags)
    list(statistic = chi2, p.value = p_value)
  }
  ArchTest(std_resid)
}, error = function(e) list(statistic = NA, p.value = NA))

# Sign bias test
sign_bias <- tryCatch({
  signbias(garch_fit)
}, error = function(e) NULL)

# Nyblom stability test
nyblom_test <- tryCatch({
  nyblom(garch_fit)
}, error = function(e) NULL)

garch_diagnostics <- list(
  GARCH = list(
    ljung_box_resid = lb_resid,
    ljung_box_sq_resid = lb_sq_resid,
    arch_lm = arch_lm,
    sign_bias = sign_bias,
    nyblom = nyblom_test,
    std_resid = as.numeric(std_resid)
  )
)

cat("\n========== GARCH DIAGNOSTICS ==========\n")
cat(sprintf("Ljung-Box (std resid):    Q = %.2f, p-value = %.4f %s\n", 
            lb_resid$statistic, lb_resid$p.value,
            ifelse(lb_resid$p.value > 0.05, "[PASS]", "[FAIL]")))
cat(sprintf("Ljung-Box (std resid^2):  Q = %.2f, p-value = %.4f %s\n", 
            lb_sq_resid$statistic, lb_sq_resid$p.value,
            ifelse(lb_sq_resid$p.value > 0.05, "[PASS]", "[FAIL]")))
cat(sprintf("ARCH-LM Test:             LM = %.2f, p-value = %.4f %s\n", 
            arch_lm$statistic, arch_lm$p.value,
            ifelse(arch_lm$p.value > 0.05, "[PASS]", "[FAIL]")))

if (!is.null(sign_bias)) {
  cat("\nSign Bias Test:\n")
  print(sign_bias)
}

#------------------------------------------------------------------
# 9. EVALUATE POINT FORECASTS
#------------------------------------------------------------------

cat_progress("Evaluating point forecasts...")

# HAR metrics
har_test_metrics <- calc_regression_metrics(
  har_results$test_predictions$actual_level,
  har_results$test_predictions$pred_level
)

# GARCH metrics
garch_test_metrics <- calc_regression_metrics(
  garch_results$GARCH$test_predictions$actual_level,
  garch_results$GARCH$test_predictions$pred_level
)

# Combine regression metrics
regression_metrics_test <- data.frame(
  Model = c("HAR", "AR(1)-GARCH(1,1)"),
  RMSE = c(har_test_metrics$RMSE, garch_test_metrics$RMSE),
  MAE = c(har_test_metrics$MAE, garch_test_metrics$MAE),
  MAPE = c(har_test_metrics$MAPE, garch_test_metrics$MAPE),
  R2 = c(har_test_metrics$R2, garch_test_metrics$R2),
  Theil_U = c(har_test_metrics$Theil_U, garch_test_metrics$Theil_U),
  N = c(har_test_metrics$N, garch_test_metrics$N)
)

cat("\n========== POINT FORECAST METRICS (Test Set) ==========\n")
print(regression_metrics_test, digits = 4)

#------------------------------------------------------------------
# 10. EVALUATE CLASSIFICATION (DIRECTION PREDICTION)
#------------------------------------------------------------------

cat_progress("Evaluating direction classification...")

# HAR classification metrics
har_class_metrics <- calc_classification_metrics(
  har_results$test_predictions$actual_direction,
  har_results$test_predictions$pred_direction_prob
)

# GARCH classification metrics
garch_class_metrics <- calc_classification_metrics(
  garch_results$GARCH$test_predictions$actual_direction,
  garch_results$GARCH$test_predictions$pred_direction_prob
)

# Combine classification metrics
classification_metrics_test <- data.frame(
  Model = c("HAR", "AR(1)-GARCH(1,1)"),
  Accuracy = c(har_class_metrics$Accuracy, garch_class_metrics$Accuracy),
  Precision = c(har_class_metrics$Precision, garch_class_metrics$Precision),
  Recall = c(har_class_metrics$Recall, garch_class_metrics$Recall),
  Specificity = c(har_class_metrics$Specificity, garch_class_metrics$Specificity),
  F1 = c(har_class_metrics$F1, garch_class_metrics$F1),
  AUC = c(har_class_metrics$AUC, garch_class_metrics$AUC),
  N = c(har_class_metrics$N, garch_class_metrics$N)
)

cat("\n========== CLASSIFICATION METRICS (Test Set) ==========\n")
print(classification_metrics_test, digits = 4)

#------------------------------------------------------------------
# 11. DIEBOLD-MARIANO TEST (POINT FORECASTS)
#------------------------------------------------------------------

cat_progress("Computing Diebold-Mariano test...")

har_errors <- har_results$test_predictions$actual_level - har_results$test_predictions$pred_level
garch_errors <- garch_results$GARCH$test_predictions$actual_level - garch_results$GARCH$test_predictions$pred_level

# Loss differential (squared errors)
d <- har_errors^2 - garch_errors^2

n <- length(d)
d_mean <- mean(d, na.rm = TRUE)

# Newey-West variance estimator
max_lag <- floor(n^(1/3))
gamma <- numeric(max_lag + 1)
d_centered <- d - d_mean
for (j in 0:max_lag) {
  gamma[j + 1] <- mean(d_centered[1:(n-j)] * d_centered[(j+1):n], na.rm = TRUE)
}
weights <- 1 - (1:max_lag) / (max_lag + 1)
var_d <- gamma[1] + 2 * sum(weights * gamma[-1])

dm_stat <- d_mean / sqrt(var_d / n)
dm_pvalue <- 2 * (1 - pnorm(abs(dm_stat)))

dm_results <- data.frame(
  Model1 = "HAR",
  Model2 = "AR(1)-GARCH(1,1)",
  Mean_Loss_Diff = d_mean,
  DM_Statistic = dm_stat,
  P_Value = dm_pvalue,
  Better_Model = ifelse(d_mean > 0, "GARCH", "HAR"),
  Significant_05 = dm_pvalue < 0.05
)

cat("\n========== DIEBOLD-MARIANO TEST ==========\n")
cat(sprintf("H0: Equal predictive accuracy (MSE loss)\n"))
cat(sprintf("Mean loss differential (HAR - GARCH): %.4f\n", d_mean))
cat(sprintf("DM statistic: %.3f\n", dm_stat))
cat(sprintf("P-value: %.4f\n", dm_pvalue))
cat(sprintf("Conclusion: %s\n", 
            ifelse(dm_pvalue < 0.05, 
                   paste(dm_results$Better_Model, "is significantly better"),
                   "No significant difference")))

#------------------------------------------------------------------
# 12. MINCER-ZARNOWITZ REGRESSION (FORECAST EFFICIENCY)
#------------------------------------------------------------------

cat_progress("Computing Mincer-Zarnowitz regressions...")

# HAR MZ regression: actual = alpha + beta * forecast
mz_har <- lm(actual_level ~ pred_level, data = har_results$test_predictions)
mz_har_summary <- summary(mz_har)

# GARCH MZ regression
mz_garch <- lm(actual_level ~ pred_level, data = garch_results$GARCH$test_predictions)
mz_garch_summary <- summary(mz_garch)

# Test H0: alpha = 0, beta = 1 (efficient forecast)
mz_test_har <- tryCatch({
  linearHypothesis(mz_har, c("(Intercept) = 0", "pred_level = 1"))
}, error = function(e) NULL)

mz_test_garch <- tryCatch({
  linearHypothesis(mz_garch, c("(Intercept) = 0", "pred_level = 1"))
}, error = function(e) NULL)

mz_results <- data.frame(
  Model = c("HAR", "AR(1)-GARCH(1,1)"),
  Alpha = c(coef(mz_har)[1], coef(mz_garch)[1]),
  Beta = c(coef(mz_har)[2], coef(mz_garch)[2]),
  R2 = c(mz_har_summary$r.squared, mz_garch_summary$r.squared),
  Joint_Test_Pval = c(
    ifelse(!is.null(mz_test_har), mz_test_har$`Pr(>F)`[2], NA),
    ifelse(!is.null(mz_test_garch), mz_test_garch$`Pr(>F)`[2], NA)
  )
)

cat("\n========== MINCER-ZARNOWITZ REGRESSION ==========\n")
cat("Test: actual = alpha + beta * forecast\n")
cat("Efficient forecast: alpha = 0, beta = 1\n\n")
print(mz_results, digits = 4)

#------------------------------------------------------------------
# 13. VaR BACKTESTING
#------------------------------------------------------------------

cat_progress("Performing VaR backtesting...")

VaR_05 <- garch_results$GARCH$VaR_05
actual_returns <- garch_results$GARCH$test_predictions$actual_return

if (!is.null(VaR_05) && length(VaR_05) == length(actual_returns)) {
  violations <- actual_returns < VaR_05
  violation_rate <- mean(violations, na.rm = TRUE)
  expected_rate <- 0.05
  
  n <- sum(!is.na(violations))
  n_violations <- sum(violations, na.rm = TRUE)
  
  # Kupiec test (unconditional coverage)
  if (n_violations > 0 && n_violations < n) {
    lr_uc <- -2 * (n_violations * log(expected_rate) + 
                     (n - n_violations) * log(1 - expected_rate) -
                     n_violations * log(n_violations / n) - 
                     (n - n_violations) * log(1 - n_violations / n))
    kupiec_pval <- 1 - pchisq(lr_uc, df = 1)
  } else {
    lr_uc <- NA
    kupiec_pval <- NA
  }
  
  var_backtest_results <- data.frame(
    Model = "AR(1)-GARCH(1,1)",
    Expected_Rate = expected_rate,
    Actual_Rate = violation_rate,
    N_Violations = n_violations,
    N_Total = n,
    Kupiec_LR = lr_uc,
    Kupiec_Pval = kupiec_pval,
    Pass_Kupiec = ifelse(!is.na(kupiec_pval), kupiec_pval > 0.05, NA)
  )
  
  cat("\n========== VaR BACKTEST (5% VaR) ==========\n")
  print(var_backtest_results, digits = 4)
} else {
  var_backtest_results <- data.frame()
  cat("\nVaR backtest skipped: VaR forecasts not available\n")
}

#------------------------------------------------------------------
# 14. COMBINE ALL TEST PREDICTIONS
#------------------------------------------------------------------

all_test_preds <- data.frame(
  date = test_df$date,
  actual = test_df$target_level,
  HAR = har_results$test_predictions$pred_level,
  GARCH = garch_results$GARCH$test_predictions$pred_level
)

all_direction_preds <- data.frame(
  date = test_df$date,
  actual_direction = test_df$target_direction,
  HAR_prob = har_results$test_predictions$pred_direction_prob,
  GARCH_prob = garch_results$GARCH$test_predictions$pred_direction_prob,
  HAR_pred = ifelse(har_results$test_predictions$pred_direction_prob >= 0.5, 1, 0),
  GARCH_pred = ifelse(garch_results$GARCH$test_predictions$pred_direction_prob >= 0.5, 1, 0)
)

# Volatility forecasts (from GARCH only)
all_variance_forecasts <- data.frame(
  date = test_df$date,
  GARCH_sigma = garch_results$GARCH$test_predictions$pred_sigma,
  GARCH_variance = garch_results$GARCH$test_predictions$pred_variance
)

#------------------------------------------------------------------
# 15. PLOTS
#------------------------------------------------------------------

cat_progress("Generating plots...")

# Plot 1: Point Forecast Comparison
pdf("results/figures/17_point_forecast_comparison.pdf", width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Time series plot
plot(all_test_preds$date, all_test_preds$actual, type = "l", col = "black", lwd = 1.5,
     xlab = "Date", ylab = "VIX Level", main = "Point Forecasts: HAR vs AR(1)-GARCH(1,1)")
lines(all_test_preds$date, all_test_preds$HAR, col = "blue", lwd = 1)
lines(all_test_preds$date, all_test_preds$GARCH, col = "red", lwd = 1)
legend("topright", c("Actual", "HAR", "AR(1)-GARCH(1,1)"), 
       col = c("black", "blue", "red"), lwd = c(1.5, 1, 1), cex = 0.8)

# HAR: Actual vs Predicted
plot(har_results$test_predictions$actual_level, har_results$test_predictions$pred_level,
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     main = "HAR: Actual vs Predicted", xlab = "Actual VIX", ylab = "Predicted VIX")
abline(0, 1, col = "red", lwd = 2)
abline(mz_har, col = "blue", lty = 2)

# GARCH: Actual vs Predicted
plot(garch_results$GARCH$test_predictions$actual_level, 
     garch_results$GARCH$test_predictions$pred_level,
     pch = 16, cex = 0.5, col = rgb(1, 0, 0, 0.3),
     main = "AR(1)-GARCH(1,1): Actual vs Predicted", xlab = "Actual VIX", ylab = "Predicted VIX")
abline(0, 1, col = "red", lwd = 2)
abline(mz_garch, col = "blue", lty = 2)

# RMSE comparison
barplot(regression_metrics_test$RMSE, names.arg = regression_metrics_test$Model,
        main = "RMSE Comparison", ylab = "RMSE", col = c("blue", "red"))

dev.off()

# Plot 2: Error Analysis
pdf("results/figures/18_error_analysis.pdf", width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# HAR error distribution
hist(har_errors, breaks = 50, main = "HAR: Prediction Error Distribution",
     xlab = "Error (Actual - Predicted)", col = "lightblue", freq = FALSE)
lines(density(har_errors), col = "blue", lwd = 2)
abline(v = 0, col = "red", lwd = 2)

# GARCH error distribution
hist(garch_errors, breaks = 50, main = "GARCH: Prediction Error Distribution",
     xlab = "Error (Actual - Predicted)", col = "lightcoral", freq = FALSE)
lines(density(garch_errors), col = "red", lwd = 2)
abline(v = 0, col = "blue", lwd = 2)

# Cumulative squared errors
cum_se_har <- cumsum(har_errors^2)
cum_se_garch <- cumsum(garch_errors^2)
plot(test_df$date, cum_se_har, type = "l", col = "blue", lwd = 1.5,
     xlab = "Date", ylab = "Cumulative Squared Error", 
     main = "Cumulative Squared Errors Over Time")
lines(test_df$date, cum_se_garch, col = "red", lwd = 1.5)
legend("topleft", c("HAR", "AR(1)-GARCH(1,1)"), col = c("blue", "red"), lwd = 1.5)

# Loss differential over time
plot(test_df$date, cumsum(d), type = "l", col = "purple", lwd = 1.5,
     xlab = "Date", ylab = "Cumulative Loss Differential",
     main = "Cumulative DM Loss Differential (HAR - GARCH)")
abline(h = 0, col = "gray", lty = 2)

dev.off()

# Plot 3: GARCH Volatility Forecast
pdf("results/figures/19_volatility_forecast.pdf", width = 14, height = 8)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Volatility forecast time series
plot(all_variance_forecasts$date, all_variance_forecasts$GARCH_sigma * 100, 
     type = "l", col = "blue", lwd = 1.5,
     xlab = "Date", ylab = "Conditional Volatility (%)", 
     main = "AR(1)-GARCH(1,1) Volatility Forecast")

# Volatility vs actual absolute returns
actual_abs_return <- abs(garch_results$GARCH$test_predictions$actual_return)
plot(all_variance_forecasts$GARCH_sigma, actual_abs_return,
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = "Predicted Sigma", ylab = "|Actual Return|",
     main = "Predicted Volatility vs Realised |Return|")
abline(0, 1, col = "red", lwd = 2)

# News Impact Curve
tryCatch({
  ni <- newsimpact(garch_fit)
  plot(ni$zx, ni$zy, type = "l", lwd = 2, col = "blue",
       main = "News Impact Curve",
       xlab = "z (standardised shock)", ylab = "Conditional Variance")
  abline(v = 0, col = "gray", lty = 2)
}, error = function(e) {
  plot.new()
  text(0.5, 0.5, "News Impact Curve failed")
})

# VaR violations
if (nrow(var_backtest_results) > 0) {
  plot(test_df$date, actual_returns, type = "l", col = "gray",
       xlab = "Date", ylab = "Return", main = "VaR(5%) Backtesting")
  lines(test_df$date, VaR_05, col = "red", lwd = 1.5)
  points(test_df$date[violations], actual_returns[violations], 
         col = "red", pch = 16, cex = 0.8)
  legend("bottomleft", c("Returns", "VaR(5%)", "Violations"), 
         col = c("gray", "red", "red"), lty = c(1, 1, NA), pch = c(NA, NA, 16))
}

dev.off()

# Plot 4: GARCH Diagnostics
pdf("results/figures/20_garch_diagnostics.pdf", width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

std_resid <- garch_diagnostics$GARCH$std_resid

# Standardised residuals
plot(std_resid, type = "l", main = "Standardised Residuals",
     xlab = "Observation", ylab = "Standardised Residual", col = "darkblue")
abline(h = c(-2, 0, 2), col = c("red", "gray", "red"), lty = c(2, 1, 2))

# ACF of standardised residuals
acf(std_resid, main = "ACF of Standardised Residuals", lag.max = 30)

# ACF of squared standardised residuals
acf(std_resid^2, main = "ACF of Squared Standardised Residuals", lag.max = 30)

# QQ plot
qqnorm(std_resid, main = "Q-Q Plot of Standardised Residuals")
qqline(std_resid, col = "red", lwd = 2)

dev.off()

# Plot 5: Classification Performance
pdf("results/figures/21_classification_performance.pdf", width = 14, height = 8)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# ROC-like: Direction probability histogram by actual direction
hist(har_results$test_predictions$pred_direction_prob[test_df$target_direction == 1],
     breaks = 20, col = rgb(0, 1, 0, 0.5), xlim = c(0, 1),
     main = "HAR: Direction Probability Distribution",
     xlab = "P(VIX Up)", freq = FALSE)
hist(har_results$test_predictions$pred_direction_prob[test_df$target_direction == 0],
     breaks = 20, col = rgb(1, 0, 0, 0.5), add = TRUE, freq = FALSE)
legend("topright", c("Actual Up", "Actual Down"), 
       fill = c(rgb(0, 1, 0, 0.5), rgb(1, 0, 0, 0.5)))

hist(garch_results$GARCH$test_predictions$pred_direction_prob[test_df$target_direction == 1],
     breaks = 20, col = rgb(0, 1, 0, 0.5), xlim = c(0, 1),
     main = "GARCH: Direction Probability Distribution",
     xlab = "P(VIX Up)", freq = FALSE)
hist(garch_results$GARCH$test_predictions$pred_direction_prob[test_df$target_direction == 0],
     breaks = 20, col = rgb(1, 0, 0, 0.5), add = TRUE, freq = FALSE)
legend("topright", c("Actual Up", "Actual Down"), 
       fill = c(rgb(0, 1, 0, 0.5), rgb(1, 0, 0, 0.5)))

dev.off()

#------------------------------------------------------------------
# 16. SAVE ALL RESULTS
#------------------------------------------------------------------

cat_progress("Saving results...")

# Models
saveRDS(har_results, "results/models/har_model_results.rds")
saveRDS(garch_results, "results/models/garch_model_results.rds")
saveRDS(garch_diagnostics, "results/models/garch_diagnostics.rds")

# Metrics tables
write.csv(regression_metrics_test, "results/tables/point_forecast_regression_metrics.csv", row.names = FALSE)
write.csv(classification_metrics_test, "results/tables/point_forecast_classification_metrics.csv", row.names = FALSE)
write.csv(dm_results, "results/tables/dm_point_forecast_test.csv", row.names = FALSE)
write.csv(mz_results, "results/tables/mincer_zarnowitz_results.csv", row.names = FALSE)
write.csv(stationarity_results, "results/tables/stationarity_tests.csv", row.names = FALSE)

if (nrow(var_backtest_results) > 0) {
  write.csv(var_backtest_results, "results/tables/var_backtest_results.csv", row.names = FALSE)
}

# Predictions
saveRDS(all_test_preds, "results/models/point_forecast_predictions.rds")
saveRDS(all_direction_preds, "results/models/direction_predictions.rds")
saveRDS(all_variance_forecasts, "results/models/variance_forecast_predictions.rds")

# Model configuration
model_config <- list(
  har_formula = "target_level ~ vix_d + vix_w + vix_m",
  har_input = "VIX_levels",
  garch_spec = list(
    variance_model = "sGARCH(1,1)",
    mean_model = "AR(1)",
    distribution = "Student-t"
  ),
  garch_input = "VIX_log_returns",
  split_date = split_date,
  train_n = nrow(train_df),
  test_n = nrow(test_df),
  refit_frequency = 22
)
saveRDS(model_config, "results/models/model_config.rds")

#------------------------------------------------------------------
# 17. FINAL SUMMARY
#------------------------------------------------------------------

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("                         FINAL SUMMARY                              \n")
cat(paste(rep("=", 70), collapse = ""), "\n")

cat("\n--- STATIONARITY ---\n")
print(stationarity_results[, c("Series", "ADF_Pvalue", "KPSS_Pvalue")])

cat("\n--- POINT FORECAST METRICS ---\n")
print(regression_metrics_test[, c("Model", "RMSE", "MAE", "MAPE", "R2", "Theil_U")], digits = 4)

cat("\n--- CLASSIFICATION METRICS ---\n")
print(classification_metrics_test[, c("Model", "Accuracy", "F1", "AUC")], digits = 4)

cat("\n--- DIEBOLD-MARIANO TEST ---\n")
cat(sprintf("DM statistic: %.3f, p-value: %.4f\n", dm_stat, dm_pvalue))
cat(sprintf("Better model: %s (%s)\n", 
            dm_results$Better_Model,
            ifelse(dm_pvalue < 0.05, "significant", "not significant")))

cat("\n--- MINCER-ZARNOWITZ (Forecast Efficiency) ---\n")
print(mz_results[, c("Model", "Alpha", "Beta", "R2")], digits = 4)

if (nrow(var_backtest_results) > 0) {
  cat("\n--- VaR BACKTEST ---\n")
  cat(sprintf("Expected: 5%%, Actual: %.2f%%, Kupiec p-value: %.4f\n",
              var_backtest_results$Actual_Rate * 100, var_backtest_results$Kupiec_Pval))
}

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat_progress("All results saved to results/models/ and results/tables/")
cat_progress("Plots saved to results/figures/")
cat(paste(rep("=", 70), collapse = ""), "\n")

cat("\nNext step: Run 02_Feature_Engineering.R\n")

################################################################################
# END OF SCRIPT
################################################################################