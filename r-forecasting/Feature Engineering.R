################################################################################
# 02_Feature_Engineering_v2.R 
# Improved version: AR(1) residual target + non-redundant features
################################################################################

#------------------------------------------------------------------
# 0. SETUP
#------------------------------------------------------------------

source("Setup.R")

merged_dt <- readRDS("data/merged_data.rds")
split_info <- readRDS("results/models/split_info.rds")
setDT(merged_dt)
setorder(merged_dt, date)

#------------------------------------------------------------------
# 1. BASE VARIABLES
#------------------------------------------------------------------

merged_dt[, vix := vix_close]
merged_dt[, vix_ret := c(NA, diff(log(vix_close)))]
merged_dt[, spx := close]
merged_dt[, spx_ret := c(NA, diff(log(close)))]

#------------------------------------------------------------------
# 2. VIX LAGS (Keep vix_lag_1 for AR(1) baseline and reconstruction)
#------------------------------------------------------------------

# VIX level lags - needed for target construction and classification
for (lag in c(1, 2, 5)) {
  merged_dt[, paste0("vix_lag_", lag) := shift(vix, n = lag, type = "lag")]
}

# VIX return lags (Classification + Regression momentum)
for (lag in c(1, 2, 5)) {
  merged_dt[, paste0("vix_ret_lag_", lag) := shift(vix_ret, n = lag, type = "lag")]
}

# Squared return lags (volatility clustering)
for (lag in c(1, 2, 5)) {
  merged_dt[, paste0("vix_ret_sq_lag_", lag) := shift(vix_ret^2, n = lag, type = "lag")]
}

#------------------------------------------------------------------
# 3. VIX ROLLING STATS (For ratios and relative features)
#------------------------------------------------------------------

# Moving averages - needed for ratio calculations
for (w in c(5, 22, 63, 252)) {
  merged_dt[, paste0("vix_ma_", w) := shift(frollmean(vix, n = w, align = "right"), 1)]
}

# Rolling min/max
merged_dt[, vix_min_22 := shift(frollapply(vix, n = 22, FUN = min, align = "right"), 1)]
merged_dt[, vix_max_22 := shift(frollapply(vix, n = 22, FUN = max, align = "right"), 1)]
merged_dt[, vix_range_22 := vix_max_22 - vix_min_22]

# Position within range (relative feature - KEEP)
merged_dt[, vix_range_pos := (vix_lag_1 - vix_min_22) / (vix_range_22 + 1e-8)]

# Rolling standard dev
for (w in c(5, 22, 63)) {
  merged_dt[, paste0("vix_sd_", w) := shift(frollapply(vix, n = w, FUN = sd, align = "right"), 1)]
}

#------------------------------------------------------------------
# 4. RELATIVE/MOMENTUM FEATURES (Key for beating AR(1))
#------------------------------------------------------------------

# Distance from moving averages (relative features - KEEP)
for (w in c(5, 22, 63, 252)) {
  ma_col <- paste0("vix_ma_", w)
  merged_dt[, paste0("vix_dist_ma_", w) := (vix_lag_1 - get(ma_col)) / (get(ma_col) + 1e-8)]
}

# MA ratios (trend indicators - KEEP)
merged_dt[, ma_ratio_5_22 := vix_ma_5 / (vix_ma_22 + 1e-8)]
merged_dt[, ma_ratio_22_63 := vix_ma_22 / (vix_ma_63 + 1e-8)]
merged_dt[, ma_ratio_63_252 := vix_ma_63 / (vix_ma_252 + 1e-8)]

# Rate of change (momentum - KEEP)
for (period in c(1, 5, 22)) {
  merged_dt[, paste0("vix_roc_", period) := (vix_lag_1 - shift(vix, period + 1)) / (shift(vix, period + 1) + 1e-8)]
}

# Coefficient of variation (relative volatility - KEEP)
merged_dt[, vix_cv_22 := vix_sd_22 / (vix_ma_22 + 1e-8)]
merged_dt[, vix_cv_63 := vix_sd_63 / (vix_ma_63 + 1e-8)]

#------------------------------------------------------------------
# 5. VIX PERCENTILE (Relative position - KEEP)
#------------------------------------------------------------------

calc_percentile_rank <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  current <- tail(x, 1)
  historical <- head(x[!is.na(x)], -1)
  if (length(historical) == 0) return(NA_real_)
  mean(historical <= current)
}

merged_dt[, vix_pct_rank_63 := shift(frollapply(vix, n = 64, FUN = calc_percentile_rank, align = "right"), 1)]
merged_dt[, vix_pct_rank_252 := shift(frollapply(vix, n = 253, FUN = calc_percentile_rank, align = "right"), 1)]

# Z-scores (standardised position - KEEP)
merged_dt[, vix_zscore_22 := (vix_lag_1 - vix_ma_22) / (vix_sd_22 + 1e-8)]
merged_dt[, vix_zscore_63 := (vix_lag_1 - vix_ma_63) / (vix_sd_63 + 1e-8)]

#------------------------------------------------------------------
# 5.5 REGIME CLASSIFICATION
#------------------------------------------------------------------

calc_regime <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  current <- tail(x, 1)
  historical <- head(x[!is.na(x)], -1)
  if (length(historical) < 10) return(NA_real_)
  q33 <- quantile(historical, 0.33)
  q67 <- quantile(historical, 0.67)
  if (current < q33) return(1)
  if (current <= q67) return(2)
  return(3)
}

merged_dt[, regime_252 := shift(frollapply(vix, n = 252, FUN = calc_regime, align = "right"), 1)]

#------------------------------------------------------------------
# 6. VOL OF VOL (Key feature - KEEP)
#------------------------------------------------------------------

merged_dt[, vix_ret_lagged := shift(vix_ret, 1)]

for (w in c(5, 22, 63)) {
  merged_dt[, paste0("vol_of_vol_", w) := shift(frollapply(vix_ret_lagged, n = w, FUN = sd, align = "right"), 1)]
}

# Vol of vol ratio (relative - KEEP)
merged_dt[, vol_of_vol_ratio := vol_of_vol_5 / (vol_of_vol_22 + 1e-8)]

merged_dt[, vix_ret_lagged := NULL]

#------------------------------------------------------------------
# 7. HURST EXPONENT (Mean-reversion indicator - KEEP)
#------------------------------------------------------------------

calc_hurst <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < 50) return(NA_real_)
  
  mean_x <- mean(x)
  cumdev <- cumsum(x - mean_x)
  R <- max(cumdev) - min(cumdev)
  S <- sd(x)
  
  if (S < 1e-10 || R < 1e-10) return(0.5)
  
  RS <- R / S
  n_seq <- 1:(n-1)
  E_RS <- sum(sqrt((n - n_seq) / n_seq)) * (gamma((n-1)/2) / (sqrt(pi) * gamma(n/2)))
  H_corrected <- log(RS / E_RS) / log(n) + 0.5
  max(0, min(1, H_corrected))
}

merged_dt[, vix_ret_lagged := shift(vix_ret, 1)]
merged_dt[, hurst_126 := shift(frollapply(vix_ret_lagged, n = 126, FUN = calc_hurst, align = "right"), 1)]
merged_dt[, vix_ret_lagged := NULL]

#------------------------------------------------------------------
# 7.5 JUMP DETECTION AND DAYS SINCE LAST JUMP (Ported from v1)
#------------------------------------------------------------------

# Use lagged returns throughout to avoid look-ahead
merged_dt[, vix_ret_lagged := shift(vix_ret, 1)]

# Rolling SD of lagged returns (shifted once more so window ends at t-1)
merged_dt[, return_sd_20 := shift(
  frollapply(vix_ret_lagged, n = 20, FUN = sd, align = "right"), 1
)]

# Jump = lagged return exceeded 3 * lagged rolling SD
# Both vix_ret_lagged and return_sd_20 are F_t-measurable
merged_dt[, is_jump := as.integer(abs(vix_ret_lagged) > 3 * return_sd_20)]

# Days since last jump
merged_dt[, jump_idx := cumsum(!is.na(is_jump) & is_jump == 1)]
merged_dt[, days_since_jump := sequence(rle(jump_idx)$lengths)]
merged_dt[jump_idx == 0, days_since_jump := NA]

# Clean up helper columns
merged_dt[, c("vix_ret_lagged", "return_sd_20", "is_jump", "jump_idx") := NULL]

#------------------------------------------------------------------
# 7.6 HALF-LIFE OF MEAN REVERSION (Ported from v1)
#------------------------------------------------------------------

# Rolling AR(1) on VIX levels to estimate OU mean-reversion speed
# half-life = -log(2) / log(beta) where beta is the AR(1) coefficient
calc_halflife <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < 20) return(NA_real_)
  
  y     <- x[-1]
  y_lag <- x[-n]
  
  tryCatch({
    beta <- sum((y_lag - mean(y_lag)) * (y - mean(y))) / sum((y_lag - mean(y_lag))^2)
    if (!is.na(beta) && beta > 0 && beta < 1) {
      hl <- -log(2) / log(beta)
      return(max(0.5, min(hl, 100)))  # Cap at reasonable bounds
    }
    NA_real_
  }, error = function(e) NA_real_)
}

# Compute on lagged VIX (shift input by 1 so window uses data up to t-1,
# then shift output by 1 so feature at t uses window ending at t-1)
merged_dt[, vix_lagged := shift(vix, 1)]
merged_dt[, halflife_mean_reversion := shift(
  frollapply(vix_lagged, n = 60, FUN = calc_halflife, align = "right"), 1
)]
merged_dt[, vix_lagged := NULL]

#------------------------------------------------------------------
# 8. FRACTIONAL DIFFERENCING (Stationary transformation - KEEP)
#------------------------------------------------------------------

frac_diff <- function(x, d, threshold = 1e-5, max_lag = 100) {
  n <- length(x)
  weights <- 1
  for (k in 1:max_lag) {
    w_k <- -weights[k] * (d - k + 1) / k
    if (abs(w_k) < threshold) break
    weights <- c(weights, w_k)
  }
  result <- stats::filter(x, weights, sides = 1)
  as.numeric(result)
}

merged_dt[, vix_fracdiff_04 := shift(frac_diff(vix, 0.4), 1)]

#------------------------------------------------------------------
# 9. ACF & SERIAL CORRELATION (Persistence indicators - KEEP)
#------------------------------------------------------------------

for (lag in c(1, 5, 22)) {
  merged_dt[, paste0("vix_acf_", lag) := shift(
    frollapply(vix_ret, 63, function(x) {
      x <- x[!is.na(x)]
      n <- length(x)
      if (n < lag + 10) return(NA_real_)
      x_demean <- x - mean(x)
      cor(x_demean[1:(n-lag)], x_demean[(lag+1):n])
    }, align = "right"), 1)]
}

# Ljung-Box Q statistic
ljung_box_stat <- function(x, h = 10) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < h + 5) return(NA_real_)
  acf_vals <- acf(x, lag.max = h, plot = FALSE)$acf[-1]
  n * (n + 2) * sum(acf_vals^2 / (n - 1:h))
}

merged_dt[, lb_stat_10 := shift(frollapply(vix_ret, 63, ljung_box_stat, align = "right"), 1)]

#------------------------------------------------------------------
# 10. VRP FEATURES (Cross-asset - KEEP)
#------------------------------------------------------------------

# SPX realised vol
merged_dt[, spx_ret_lagged := shift(spx_ret, 1)]
merged_dt[, spx_rv_22 := shift(sqrt(frollapply(spx_ret_lagged^2, 22, mean, align = "right") * 252), 1) * 100]
merged_dt[, spx_rv_63 := shift(sqrt(frollapply(spx_ret_lagged^2, 63, mean, align = "right") * 252), 1) * 100]
merged_dt[, spx_ret_lagged := NULL]

# VRP proxy
merged_dt[, vrp_proxy := vix_lag_1 - spx_rv_22]

# VRP z-score (relative - KEEP)
merged_dt[, vrp_ma_63 := shift(frollmean(vrp_proxy, 63, align = "right"), 1)]
merged_dt[, vrp_sd_63 := shift(frollapply(vrp_proxy, 63, sd, align = "right"), 1)]
merged_dt[, vrp_zscore := (vrp_proxy - vrp_ma_63) / (vrp_sd_63 + 1e-8)]

# VRP momentum
merged_dt[, vrp_momentum_5 := pmax(pmin(vrp_proxy - shift(vrp_proxy, 5), 20), -20)]
merged_dt[, vrp_momentum_22 := pmax(pmin(vrp_proxy - shift(vrp_proxy, 22), 30), -30)]

#------------------------------------------------------------------
# 11. CALENDAR FEATURES
#------------------------------------------------------------------

merged_dt[, dow := wday(date, week_start = 1)]
merged_dt[, dom := mday(date)]
merged_dt[, month := month(date)]

#------------------------------------------------------------------
# 12. S&P 500 FEATURES (Cross-asset signals - KEEP)
#------------------------------------------------------------------

# SPX return lags
for (lag in c(1, 2, 5)) {
  merged_dt[, paste0("spx_ret_lag_", lag) := shift(spx_ret, n = lag, type = "lag")]
}

# SPX ROC
for (period in c(5, 22)) {
  merged_dt[, paste0("spx_roc_", period) := (shift(spx, 1) - shift(spx, period + 1)) / (shift(spx, period + 1) + 1e-8)]
}

# SPX asymmetric sums
merged_dt[, spx_ret_lag1_temp := shift(spx_ret, 1)]
merged_dt[, spx_ret_pos_lag1 := pmax(spx_ret_lag1_temp, 0)]
merged_dt[, spx_ret_neg_lag1 := pmin(spx_ret_lag1_temp, 0)]

for (w in c(5, 22)) {
  merged_dt[, paste0("spx_pos_sum_", w) := shift(frollsum(spx_ret_pos_lag1, n = w, align = "right"), 1)]
  merged_dt[, paste0("spx_neg_sum_", w) := shift(frollsum(spx_ret_neg_lag1, n = w, align = "right"), 1)]
  merged_dt[, paste0("spx_asymmetry_", w) := pmin(abs(get(paste0("spx_neg_sum_", w))) / 
                                                    pmax(abs(get(paste0("spx_pos_sum_", w))), 0.005), 10)]
}

merged_dt[, c("spx_ret_lag1_temp", "spx_ret_pos_lag1", "spx_ret_neg_lag1") := NULL]

#------------------------------------------------------------------
# 13. VIX-SPX CROSS FEATURES
#------------------------------------------------------------------

# VIX/SPX ratio (relative - KEEP)
merged_dt[, vix_spx_ratio := vix_lag_1 / (spx / 100)]

# Rolling correlation
merged_dt[, vix_spx_cor_22 := {
  n <- .N
  cors <- rep(NA_real_, n)
  vix_r <- vix_ret
  spx_r <- spx_ret
  for (i in 23:n) {
    idx <- (i-21):i
    cors[i] <- cor(vix_r[idx], spx_r[idx], use = "complete.obs")
  }
  shift(cors, 1)
}]

merged_dt[, vix_spx_cor_63 := {
  n <- .N
  cors <- rep(NA_real_, n)
  vix_r <- vix_ret
  spx_r <- spx_ret
  for (i in 64:n) {
    idx <- (i-62):i
    cors[i] <- cor(vix_r[idx], spx_r[idx], use = "complete.obs")
  }
  shift(cors, 1)
}]

# Leverage effect (asymmetric response)
merged_dt[, spx_ret_lag1 := shift(spx_ret, 1)]
merged_dt[, vix_ret_lag1 := shift(vix_ret, 1)]
merged_dt[, leverage_asym := spx_ret_lag1 * as.integer(spx_ret_lag1 < 0) * vix_ret_lag1]
merged_dt[, c("spx_ret_lag1", "vix_ret_lag1") := NULL]

#------------------------------------------------------------------
# 14. EWMA FEATURES
#------------------------------------------------------------------

ewma_calc <- function(x, span) {
  lambda <- 2 / (span + 1)
  n <- length(x)
  ewma <- rep(NA_real_, n)
  ewma[1] <- x[1]
  for (i in 2:n) {
    if (!is.na(x[i]) && !is.na(ewma[i-1])) {
      ewma[i] <- lambda * x[i] + (1 - lambda) * ewma[i-1]
    } else if (!is.na(x[i])) {
      ewma[i] <- x[i]
    } else {
      ewma[i] <- ewma[i-1]
    }
  }
  ewma
}

merged_dt[, vix_ewma_22 := shift(ewma_calc(vix, 22), 1)]
merged_dt[, vix_dist_ewma_22 := (vix_lag_1 - vix_ewma_22) / (vix_ewma_22 + 1e-8)]

#------------------------------------------------------------------
# 15. TARGET VARIABLES
#------------------------------------------------------------------

# Original targets
merged_dt[, target_vix_level := shift(vix, n = 1, type = "lead")]
merged_dt[, target_vix_return := shift(vix_ret, n = 1, type = "lead")]
merged_dt[, target_direction := as.integer(target_vix_level > vix)]

# AR(1) RESIDUAL TARGET - Key change for regression improvement
# AR(1) forecast is simply: VIX_t+1 = VIX_t
merged_dt[, ar1_forecast := vix]  # Today's VIX as forecast for tomorrow
merged_dt[, target_ar1_residual := target_vix_level - ar1_forecast]

# Also create log-return based residual (alternative)
merged_dt[, target_log_residual := log(target_vix_level / vix)]

cat_progress("Created AR(1) residual targets")

#------------------------------------------------------------------
# 16. REMOVE WARMUP PERIOD
#------------------------------------------------------------------

warmup_days <- 252 + 22

key_features <- c("vix_ma_252", "hurst_126", "vix_pct_rank_252", "vix_spx_cor_63", "regime_252")

merged_dt[, n_na := rowSums(is.na(.SD)), .SDcols = key_features]
features_df <- merged_dt[n_na == 0]

cat_progress(sprintf("Removed %d warmup observations", nrow(merged_dt) - nrow(features_df)))
cat_progress(sprintf("Final dataset: %d observations from %s to %s",
                     nrow(features_df), min(features_df$date), max(features_df$date)))

features_df[, n_na := NULL]

#------------------------------------------------------------------
# 17. DEFINE FEATURE COLUMNS
#------------------------------------------------------------------

# REGRESSION FEATURES: Focus on momentum/relative features, NOT level features
# These should help predict the AR(1) residual (deviation from persistence)
regression_features <- c(
  # Return-based momentum (key for residual prediction)
  "vix_ret_lag_1", "vix_ret_lag_2", "vix_ret_lag_5",
  "vix_ret_sq_lag_1", "vix_ret_sq_lag_2", "vix_ret_sq_lag_5",
  
  # Rate of change (momentum signals)
  "vix_roc_1", "vix_roc_5", "vix_roc_22",
  
  # Relative position features (where is VIX vs history)
  "vix_dist_ma_5", "vix_dist_ma_22", "vix_dist_ma_63", "vix_dist_ma_252",
  "vix_zscore_22", "vix_zscore_63",
  "vix_pct_rank_63", "vix_pct_rank_252",
  "vix_range_pos",
  
  # MA ratios (trend indicators)
  "ma_ratio_5_22", "ma_ratio_22_63", "ma_ratio_63_252",
  
  # Volatility of volatility
  "vol_of_vol_5", "vol_of_vol_22", "vol_of_vol_63", "vol_of_vol_ratio",
  "vix_cv_22", "vix_cv_63",
  
  # Memory/persistence indicators
  "hurst_126", "vix_fracdiff_04",
  "vix_acf_1", "vix_acf_5", "vix_acf_22", "lb_stat_10",
  "halflife_mean_reversion", "days_since_jump",
  
  # VRP features (cross-asset)
  "vrp_proxy", "vrp_zscore", "vrp_momentum_5", "vrp_momentum_22",
  "spx_rv_22", "spx_rv_63",
  
  # SPX signals
  "spx_ret_lag_1", "spx_ret_lag_2", "spx_ret_lag_5",
  "spx_roc_5", "spx_roc_22",
  "spx_asymmetry_5", "spx_asymmetry_22",
  "spx_neg_sum_5", "spx_neg_sum_22",
  
  # Cross-asset
  "vix_spx_cor_22", "vix_spx_cor_63",
  "leverage_asym",
  
  # EWMA deviation
  "vix_dist_ewma_22",
  
  # Calendar (if genuinely useful)
  "dow"
)

# CLASSIFICATION FEATURES: Can include some level info for direction prediction
classification_features <- c(
  regression_features,
  "vix_lag_1", "vix_lag_2", "vix_lag_5",  # Levels help for direction
  "vix_min_22", "vix_max_22", "vix_range_22",
  "regime_252",
  "dom", "month"
)

# Combined feature set (union)
feature_cols <- unique(c(regression_features, classification_features))

# Verify all features exist
missing_features <- setdiff(feature_cols, names(features_df))
if (length(missing_features) > 0) {
  cat_progress(sprintf("Warning: Missing features: %s", paste(missing_features, collapse = ", ")))
  feature_cols <- intersect(feature_cols, names(features_df))
}

cat_progress(sprintf("Total features: %d", length(feature_cols)))
cat_progress(sprintf("Regression features: %d", length(regression_features)))
cat_progress(sprintf("Classification features: %d", length(classification_features)))

# Feature categories
feature_categories <- list(
  vix_returns = grep("^vix_ret_|^vix_roc_", feature_cols, value = TRUE),
  vix_relative = grep("^vix_dist_|^vix_zscore|^vix_pct_rank|^vix_range_pos", feature_cols, value = TRUE),
  vix_levels = grep("^vix_lag_|^vix_min|^vix_max", feature_cols, value = TRUE),
  ma_ratios = grep("^ma_ratio", feature_cols, value = TRUE),
  vol_of_vol = grep("^vol_of_vol|^vix_cv", feature_cols, value = TRUE),
  memory = grep("^hurst|^vix_fracdiff|^vix_acf|^lb_stat", feature_cols, value = TRUE),
  vrp = grep("^vrp_|^spx_rv", feature_cols, value = TRUE),
  spx = grep("^spx_ret|^spx_roc|^spx_asym|^spx_neg|^spx_pos", feature_cols, value = TRUE),
  cross = grep("^vix_spx_cor|^leverage", feature_cols, value = TRUE),
  calendar = grep("^dow$|^dom$|^month$", feature_cols, value = TRUE),
  regime = grep("^regime", feature_cols, value = TRUE)
)

cat("\nFeature counts by category:\n")
for (cat_name in names(feature_categories)) {
  cat(sprintf("  %s: %d\n", cat_name, length(feature_categories[[cat_name]])))
}

#------------------------------------------------------------------
# 18. TRAIN/TEST SPLIT
#------------------------------------------------------------------

split_date <- split_info$split_date
train_features <- features_df[date <= split_date]
test_features <- features_df[date > split_date]

cat_progress(sprintf("Train: %d obs (%s to %s)", 
                     nrow(train_features), min(train_features$date), max(train_features$date)))
cat_progress(sprintf("Test: %d obs (%s to %s)", 
                     nrow(test_features), min(test_features$date), max(test_features$date)))

#------------------------------------------------------------------
# 19. SAVE RESULTS
#------------------------------------------------------------------

saveRDS(features_df, "data/features_full.rds")
saveRDS(train_features, "data/features_train.rds")
saveRDS(test_features, "data/features_test.rds")
saveRDS(feature_cols, "data/feature_columns.rds")
saveRDS(regression_features, "data/regression_feature_columns.rds")
saveRDS(classification_features, "data/classification_feature_columns.rds")
saveRDS(feature_categories, "data/feature_categories.rds")

# Feature summary
feature_summary <- data.frame(
  Category = names(feature_categories),
  N_Features = sapply(feature_categories, length),
  Features = sapply(feature_categories, function(x) paste(head(x, 5), collapse = ", "))
)
write.csv(feature_summary, "results/tables/feature_summary.csv", row.names = FALSE)

# Feature statistics for residual target
residual_cors <- sapply(feature_cols, function(x) {
  cor(features_df[[x]], features_df$target_ar1_residual, use = "complete.obs")
})

feature_stats <- data.frame(
  Feature = feature_cols,
  Mean = sapply(feature_cols, function(x) mean(features_df[[x]], na.rm = TRUE)),
  SD = sapply(feature_cols, function(x) sd(features_df[[x]], na.rm = TRUE)),
  Cor_Residual = residual_cors[feature_cols]
)
feature_stats <- feature_stats[order(-abs(feature_stats$Cor_Residual)), ]
write.csv(feature_stats, "results/tables/feature_statistics.csv", row.names = FALSE)

cat_progress("Feature engineering complete (v2 with AR(1) residual)")

################################################################################
# END OF SCRIPT
################################################################################