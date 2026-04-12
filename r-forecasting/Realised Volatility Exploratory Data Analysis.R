#------------------------------------------------------------------
# RV_Comprehensive_Analysis.R
# PhD-Level Analysis of Realised Volatility for XGBoost Feature Engineering
# 
# Structure:
#   Part A: Daily RV Analysis (distributional, temporal, roughness, jumps)
#   Part B: Forward 22-day RV and VRP Analysis
#   Part C: Predictability Structure for Next-Day RV
#   Part D: XGBoost Target Construction and Feature Insights
#------------------------------------------------------------------

source("Setup.R")

# Additional packages for advanced analysis
additional_pkgs <- c(
  
  "fracdiff",      # Fractional differencing, long memory
  "longmemo",      # Long memory estimation (GPH, Whittle)
  "strucchange",   # Structural break tests
  "urca",          # Unit root and cointegration
  "fGarch",        # GARCH modelling
  "np",            # Nonparametric methods
  "entropy",       # Information theoretic measures
  "Hmisc",         # Correlation with p-values
  "corrplot",      # Correlation visualisation
  "e1071",         # SVM, additional stats
  "fitdistrplus",  # Distribution fitting
  "gamlss",        # Generalised additive models for location, scale, shape
  "boot"           # Bootstrap inference
)

for (pkg in additional_pkgs) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

#------------------------------------------------------------------
# 0. LOAD DATA AND CONFIGURATION
#------------------------------------------------------------------

cat_progress("Loading data with RV columns...")
merged_dt <- readRDS(file.path(dirs$data, "data_with_rv.rds"))

# Constants
ANNUAL_FACTOR <- 252
RV_WINDOW_FORWARD <- 22L  # Forward window for VIX comparison

# Ensure chronological ordering
setorder(merged_dt, date)

# Create analysis subset (complete cases for core variables)
core_vars <- c("date", "close", "log_return", "vix_close", 
               "rv_daily", "rv_weekly", "rv_monthly",
               "rv_cc", "rv_parkinson", "rv_gk", "rv_rs", "rv_yz",
               "vrp_cc", "vrp_parkinson")

dt <- merged_dt[complete.cases(merged_dt[, .SD, .SDcols = core_vars])]
cat_progress(sprintf("Analysis sample: %d observations (%s to %s)",
                     nrow(dt), min(dt$date), max(dt$date)))

#------------------------------------------------------------------
#
#                    PART A: DAILY REALISED VOLATILITY ANALYSIS
#
#------------------------------------------------------------------

cat_progress("="
)
cat_progress("PART A: DAILY REALISED VOLATILITY ANALYSIS")
cat_progress("=")

# Working with rv_daily = sqrt(r_t^2 * 252) * 100 (annualised daily vol proxy)
# Also compute log(RV) which is closer to Gaussian

dt[, log_rv_daily := log(rv_daily)]
dt[, log_rv_weekly := log(rv_weekly)]
dt[, log_rv_monthly := log(rv_monthly)]

#------------------------------------------------------------------
# A.1 DISTRIBUTIONAL ANALYSIS
#------------------------------------------------------------------

cat_progress("A.1: Distributional Analysis")

# -----------------------------------------------------------------------------
# A.1.1 Summary statistics with robust measures
# -----------------------------------------------------------------------------

compute_distribution_stats <- function(x, name) {
  x <- x[is.finite(x)]
  n <- length(x)
  
  # Standard moments
  mu <- mean(x)
  sigma <- sd(x)
  skew <- moments::skewness(x)
  kurt <- moments::kurtosis(x)  # Excess kurtosis + 3
  
  # Robust measures
  med <- median(x)
  mad <- mad(x, constant = 1.4826)  # Scaled MAD
  iqr <- IQR(x)
  
  # Tail measures
  q01 <- quantile(x, 0.01)
  q05 <- quantile(x, 0.05)
  q95 <- quantile(x, 0.95)
  q99 <- quantile(x, 0.99)
  
  # Tail index via Hill estimator (upper tail)
  k <- floor(0.1 * n)  # Top 10%
  x_sorted <- sort(x, decreasing = TRUE)
  hill_alpha <- k / sum(log(x_sorted[1:k] / x_sorted[k+1]))
  
  data.table(
    variable = name,
    n = n,
    mean = mu,
    sd = sigma,
    skewness = skew,
    kurtosis = kurt,
    median = med,
    mad = mad,
    iqr = iqr,
    q01 = q01,
    q05 = q05,
    q95 = q95,
    q99 = q99,
    hill_tail_index = hill_alpha
  )
}

dist_stats <- rbindlist(list(
  compute_distribution_stats(dt$rv_daily, "RV_daily"),
  compute_distribution_stats(dt$log_rv_daily, "log_RV_daily"),
  compute_distribution_stats(dt$rv_weekly, "RV_weekly"),
  compute_distribution_stats(dt$log_rv_weekly, "log_RV_weekly"),
  compute_distribution_stats(dt$rv_monthly, "RV_monthly"),
  compute_distribution_stats(dt$log_rv_monthly, "log_RV_monthly"),
  compute_distribution_stats(abs(dt$log_return) * 100, "abs_return_pct")
))

print(dist_stats)
write.csv(dist_stats, file.path(dirs$tables, "A1_rv_distribution_stats.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# A.1.2 Normality tests on log(RV) 
# -----------------------------------------------------------------------------

cat_progress("  Testing log-normality of RV...")

normality_tests <- function(x, name) {
  x <- x[is.finite(x)]
  
  # Jarque-Bera
  jb <- tseries::jarque.bera.test(x)
  
  # Shapiro-Wilk (on subsample if n > 5000)
  if (length(x) > 5000) {
    set.seed(42)
    x_sub <- sample(x, 5000)
    sw <- shapiro.test(x_sub)
  } else {
    sw <- shapiro.test(x)
  }
  
  # Anderson-Darling (via nortest if available)
  ad_stat <- NA
  ad_pval <- NA
  if (requireNamespace("nortest", quietly = TRUE)) {
    ad <- nortest::ad.test(x)
    ad_stat <- ad$statistic
    ad_pval <- ad$p.value
  }
  
  data.table(
    variable = name,
    JB_stat = jb$statistic,
    JB_pval = jb$p.value,
    SW_stat = sw$statistic,
    SW_pval = sw$p.value,
    AD_stat = ad_stat,
    AD_pval = ad_pval
  )
}

norm_tests <- rbindlist(list(
  normality_tests(dt$rv_daily, "RV_daily"),
  normality_tests(dt$log_rv_daily, "log_RV_daily"),
  normality_tests(dt$rv_weekly, "RV_weekly"),
  normality_tests(dt$log_rv_weekly, "log_RV_weekly")
))

print(norm_tests)

# -----------------------------------------------------------------------------
# A.1.3 Distribution fitting comparison
# -----------------------------------------------------------------------------

cat_progress("  Fitting parametric distributions to RV...")

# Fit distributions to RV_daily (positive support)
rv_fit_data <- dt$rv_daily[is.finite(dt$rv_daily) & dt$rv_daily > 0]

# Log-normal
fit_lnorm <- fitdistrplus::fitdist(rv_fit_data, "lnorm")

# Gamma
fit_gamma <- fitdistrplus::fitdist(rv_fit_data, "gamma")

# Weibull
fit_weibull <- fitdistrplus::fitdist(rv_fit_data, "weibull")

# Compare via AIC/BIC
dist_comparison <- data.table(
  distribution = c("Log-normal", "Gamma", "Weibull"),
  AIC = c(fit_lnorm$aic, fit_gamma$aic, fit_weibull$aic),
  BIC = c(fit_lnorm$bic, fit_gamma$bic, fit_weibull$bic),
  loglik = c(fit_lnorm$loglik, fit_gamma$loglik, fit_weibull$loglik)
)
dist_comparison[, delta_AIC := AIC - min(AIC)]

print(dist_comparison)
write.csv(dist_comparison, file.path(dirs$tables, "A1_distribution_fit_comparison.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# A.1.4 Distribution plots
# -----------------------------------------------------------------------------

pdf(file.path(dirs$figures, "30_rv_distributions.pdf"), width = 14, height = 10)

par(mfrow = c(2, 3))

# RV_daily histogram with fitted log-normal
hist(rv_fit_data, breaks = 100, freq = FALSE, 
     main = "RV Daily: Empirical vs Log-Normal",
     xlab = "RV (%)", col = "lightblue", border = "white")
x_seq <- seq(min(rv_fit_data), max(rv_fit_data), length.out = 200)
lines(x_seq, dlnorm(x_seq, fit_lnorm$estimate[1], fit_lnorm$estimate[2]), 
      col = "red", lwd = 2)
legend("topright", "Log-normal fit", col = "red", lwd = 2, bty = "n")

# log(RV) histogram with normal overlay
log_rv <- dt$log_rv_daily[is.finite(dt$log_rv_daily)]
hist(log_rv, breaks = 80, freq = FALSE,
     main = "log(RV Daily): Empirical vs Normal",
     xlab = "log(RV)", col = "lightgreen", border = "white")
x_seq <- seq(min(log_rv), max(log_rv), length.out = 200)
lines(x_seq, dnorm(x_seq, mean(log_rv), sd(log_rv)), col = "red", lwd = 2)

# Q-Q plot for log(RV)
qqnorm(log_rv, main = "Q-Q Plot: log(RV Daily)", pch = 16, cex = 0.3, col = rgb(0,0,0,0.3))
qqline(log_rv, col = "red", lwd = 2)

# RV_weekly
hist(dt$rv_weekly, breaks = 80, freq = FALSE,
     main = "RV Weekly Distribution",
     xlab = "RV (%)", col = "lightyellow", border = "white")

# log(RV_weekly)
hist(dt$log_rv_weekly[is.finite(dt$log_rv_weekly)], breaks = 80, freq = FALSE,
     main = "log(RV Weekly) Distribution",
     xlab = "log(RV)", col = "lightcoral", border = "white")

# Comparison of RV estimators density
plot(density(dt$rv_daily, na.rm = TRUE), main = "Density: Daily RV Estimators",
     xlab = "RV (%)", lwd = 2, col = "black", xlim = c(0, 80))
lines(density(sqrt(dt$rv_weekly^2), na.rm = TRUE), col = "blue", lwd = 2)
lines(density(sqrt(dt$rv_monthly^2), na.rm = TRUE), col = "red", lwd = 2)
legend("topright", c("Daily", "Weekly (5d)", "Monthly (22d)"),
       col = c("black", "blue", "red"), lwd = 2, bty = "n")

dev.off()

#------------------------------------------------------------------
# A.2 LONG MEMORY AND PERSISTENCE ANALYSIS
#------------------------------------------------------------------

cat_progress("A.2: Long Memory and Persistence Analysis")

# -----------------------------------------------------------------------------
# A.2.1 ACF Analysis with theoretical decay comparison
# -----------------------------------------------------------------------------

# Compute ACF for multiple horizons
max_lag <- 252  # 1 year

# Clean data for ACF computation (remove NA and non-finite values)
rv_daily_clean <- dt$rv_daily[is.finite(dt$rv_daily)]
log_rv_daily_clean <- dt$log_rv_daily[is.finite(dt$log_rv_daily)]
abs_ret_clean <- abs(dt$log_return[is.finite(dt$log_return)])
sq_ret_clean <- dt$log_return[is.finite(dt$log_return)]^2

acf_rv_daily <- acf(rv_daily_clean, lag.max = max_lag, plot = FALSE)
acf_log_rv <- acf(log_rv_daily_clean, lag.max = max_lag, plot = FALSE)
acf_abs_ret <- acf(abs_ret_clean, lag.max = max_lag, plot = FALSE)
acf_sq_ret <- acf(sq_ret_clean, lag.max = max_lag, plot = FALSE)

# Half-life computation (lag at which ACF drops below 0.5 * ACF(1))
compute_halflife <- function(acf_obj) {
  acf_vals <- acf_obj$acf[-1]  # Exclude lag 0
  target <- 0.5 * acf_vals[1]
  idx <- which(acf_vals < target)[1]
  if (is.na(idx)) idx <- length(acf_vals)
  return(idx)
}

halflife_rv <- compute_halflife(acf_rv_daily)
halflife_log_rv <- compute_halflife(acf_log_rv)
halflife_abs <- compute_halflife(acf_abs_ret)

cat_progress(sprintf("  ACF half-life: RV=%d, log(RV)=%d, |r|=%d days",
                     halflife_rv, halflife_log_rv, halflife_abs))

# -----------------------------------------------------------------------------
# A.2.2 Long Memory Estimation: GPH and Local Whittle
# -----------------------------------------------------------------------------

cat_progress("  Estimating fractional integration parameter d...")

# GPH (Geweke-Porter-Hudak) estimator
# d estimate for log(RV) - use the cleaned version from ACF section
n <- length(log_rv_daily_clean)

# GPH with different bandwidths
gph_bandwidths <- c(floor(n^0.5), floor(n^0.6), floor(n^0.7), floor(n^0.8))

gph_estimates <- data.table(
  bandwidth = gph_bandwidths,
  d_hat = sapply(gph_bandwidths, function(m) {
    tryCatch({
      fracdiff::fdGPH(log_rv_daily_clean, bandw.exp = log(m)/log(n))$d
    }, error = function(e) NA)
  })
)

# Local Whittle estimator (more efficient)
lw_estimate <- tryCatch({
  # Using longmemo package if available, otherwise manual implementation
  if (requireNamespace("longmemo", quietly = TRUE)) {
    longmemo::WhittleEst(log_rv_daily_clean)$coefficients[1]
  } else {
    fracdiff::fdGPH(log_rv_daily_clean, bandw.exp = 0.65)$d
  }
}, error = function(e) NA)

cat_progress(sprintf("  GPH d estimates: %s", 
                     paste(round(gph_estimates$d_hat, 3), collapse = ", ")))
cat_progress(sprintf("  Local Whittle d estimate: %.3f", lw_estimate))

# Interpretation
d_interpretation <- function(d) {
  if (is.na(d)) return("N/A")
  if (d < 0) return("Anti-persistent")
  if (d < 0.5) return("Stationary long memory")
  if (d < 1) return("Non-stationary long memory")
  return("Unit root or higher")
}

# -----------------------------------------------------------------------------
# A.2.3 R/S Analysis (Hurst exponent)
# -----------------------------------------------------------------------------

cat_progress("  Computing Hurst exponent via R/S analysis...")

compute_rs_hurst <- function(x, min_block = 10) {
  x <- x[is.finite(x)]
  n <- length(x)
  
  # Block sizes (powers of 2 up to n/4)
  block_sizes <- 2^(ceiling(log2(min_block)):floor(log2(n/4)))
  
  rs_values <- sapply(block_sizes, function(k) {
    n_blocks <- floor(n / k)
    rs_block <- numeric(n_blocks)
    
    for (i in 1:n_blocks) {
      block <- x[((i-1)*k + 1):(i*k)]
      block_mean <- mean(block)
      y <- cumsum(block - block_mean)
      R <- max(y) - min(y)
      S <- sd(block)
      rs_block[i] <- if (S > 0) R/S else NA
    }
    mean(rs_block, na.rm = TRUE)
  })
  
  # Regress log(R/S) on log(n)
  valid <- is.finite(rs_values) & rs_values > 0
  if (sum(valid) < 3) return(list(H = NA, se = NA))
  
  fit <- lm(log(rs_values[valid]) ~ log(block_sizes[valid]))
  
  list(
    H = coef(fit)[2],
    se = summary(fit)$coefficients[2, 2],
    r_squared = summary(fit)$r.squared,
    block_sizes = block_sizes[valid],
    rs_values = rs_values[valid]
  )
}

hurst_rv <- compute_rs_hurst(dt$rv_daily)
hurst_log_rv <- compute_rs_hurst(dt$log_rv_daily)
hurst_abs_ret <- compute_rs_hurst(abs(dt$log_return))

cat_progress(sprintf("  Hurst exponent: RV=%.3f (se=%.3f), log(RV)=%.3f, |r|=%.3f",
                     hurst_rv$H, hurst_rv$se, hurst_log_rv$H, hurst_abs_ret$H))

# -----------------------------------------------------------------------------
# A.2.4 Fractional differencing parameter via MLE
# -----------------------------------------------------------------------------

cat_progress("  Fitting ARFIMA(0,d,0) to log(RV)...")

arfima_fit <- tryCatch({
  fracdiff::fracdiff(log_rv_daily_clean, nar = 0, nma = 0)
}, error = function(e) NULL)

if (!is.null(arfima_fit)) {
  cat_progress(sprintf("  ARFIMA d = %.4f (se = %.4f)", 
                       arfima_fit$d, arfima_fit$stderror.dpq[1]))
}

# Compile long memory results
long_memory_results <- data.table(
  method = c("GPH (n^0.5)", "GPH (n^0.6)", "GPH (n^0.7)", "GPH (n^0.8)",
             "Local Whittle", "R/S Hurst", "ARFIMA MLE"),
  estimate = c(gph_estimates$d_hat, lw_estimate, 
               hurst_rv$H - 0.5,  # Convert H to d
               if(!is.null(arfima_fit)) arfima_fit$d else NA),
  interpretation = c(
    sapply(gph_estimates$d_hat, d_interpretation),
    d_interpretation(lw_estimate),
    d_interpretation(hurst_rv$H - 0.5),
    if(!is.null(arfima_fit)) d_interpretation(arfima_fit$d) else "N/A"
  )
)

print(long_memory_results)
write.csv(long_memory_results, file.path(dirs$tables, "A2_long_memory_estimates.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# A.2.5 Long memory visualisation
# -----------------------------------------------------------------------------

pdf(file.path(dirs$figures, "31_long_memory.pdf"), width = 14, height = 10)

par(mfrow = c(2, 2))

# ACF comparison - with defensive checks
ci_bound <- qnorm(0.975) / sqrt(length(log_rv_daily_clean))
acf_vals_plot <- acf_log_rv$acf[, 1, 1]
if (all(is.finite(acf_vals_plot))) {
  plot(0:max_lag, acf_vals_plot, type = "h", lwd = 2,
       main = "ACF of log(RV) - Evidence of Long Memory",
       xlab = "Lag (days)", ylab = "ACF", col = "darkblue")
  abline(h = c(-ci_bound, ci_bound), col = "red", lty = 2)
  abline(h = 0, col = "grey")
  
  # Theoretical decay comparison: short memory AR(1) vs long memory
  # For d = 0.4, ACF ~ k^(2d-1) = k^(-0.2)
  k <- 1:max_lag
  d_est <- mean(c(gph_estimates$d_hat, lw_estimate), na.rm = TRUE)
  if (!is.na(d_est) && d_est > 0 && d_est < 0.5 && is.finite(acf_vals_plot[2])) {
    theoretical_lm <- acf_vals_plot[2] * (k / 1)^(2*d_est - 1)
    lines(k, theoretical_lm, col = "red", lwd = 2, lty = 2)
    legend("topright", c("Empirical", sprintf("Long memory (d=%.2f)", d_est)),
           col = c("darkblue", "red"), lwd = 2, lty = c(1, 2), bty = "n")
  }
} else {
  plot.new()
  text(0.5, 0.5, "ACF computation failed - non-finite values", cex = 1.2)
}

# R/S plot
if (!is.na(hurst_rv$H) && length(hurst_rv$block_sizes) > 0) {
  plot(log(hurst_rv$block_sizes), log(hurst_rv$rs_values),
       pch = 16, cex = 1.5,
       main = sprintf("R/S Analysis: Hurst = %.3f", hurst_rv$H),
       xlab = "log(block size)", ylab = "log(R/S)")
  abline(lm(log(hurst_rv$rs_values) ~ log(hurst_rv$block_sizes)), col = "red", lwd = 2)
  # Reference lines
  abline(a = 0, b = 0.5, col = "grey", lty = 2)  # Random walk H=0.5
  legend("bottomright", c("Fitted", "H=0.5 (random)"), 
         col = c("red", "grey"), lwd = 2, lty = c(1, 2), bty = "n")
} else {
  plot.new()
  text(0.5, 0.5, "R/S analysis failed", cex = 1.2)
}

# ACF decay on log scale
acf_vals <- acf_log_rv$acf[-1, 1, 1]
positive_acf <- is.finite(acf_vals) & acf_vals > 0
if (sum(positive_acf) > 10) {
  plot(log(1:max_lag)[positive_acf], log(acf_vals[positive_acf]),
       pch = 16, cex = 0.8, col = rgb(0, 0, 0.5, 0.5),
       main = "Log-Log ACF Decay (Long Memory Test)",
       xlab = "log(lag)", ylab = "log(ACF)")
  # Fit decay rate using first 50 positive values
  n_fit <- min(50, sum(positive_acf))
  acf_fit_idx <- which(positive_acf)[1:n_fit]
  acf_fit <- lm(log(acf_vals[acf_fit_idx]) ~ log(acf_fit_idx))
  abline(acf_fit, col = "red", lwd = 2)
  legend("topright", sprintf("Slope = %.3f (2d-1 = %.3f -> d = %.3f)", 
                             coef(acf_fit)[2], coef(acf_fit)[2], (coef(acf_fit)[2] + 1)/2),
         bty = "n")
} else {
  plot.new()
  text(0.5, 0.5, "Insufficient positive ACF values for log-log plot", cex = 1.2)
}

# PACF
pacf_log_rv <- pacf(log_rv_daily_clean, lag.max = 60, plot = FALSE)
pacf_vals <- pacf_log_rv$acf[, 1, 1]
if (all(is.finite(pacf_vals[1:60]))) {
  plot(1:60, pacf_vals[1:60], type = "h", lwd = 2,
       main = "PACF of log(RV)",
       xlab = "Lag", ylab = "PACF", col = "darkgreen")
  abline(h = c(-ci_bound, ci_bound), col = "red", lty = 2)
  abline(h = 0, col = "grey")
} else {
  plot.new()
  text(0.5, 0.5, "PACF computation failed", cex = 1.2)
}

dev.off()

#------------------------------------------------------------------
# A.3 ROUGHNESS ANALYSIS (VOLATILITY OF VOLATILITY)
#------------------------------------------------------------------

cat_progress("A.3: Roughness Analysis")

# -----------------------------------------------------------------------------
# A.3.1 Power variation for Hurst estimation
# Gatheral, Jaisson, Rosenbaum (2018): Volatility is Rough
# H ≈ 0.1 for realised volatility
# -----------------------------------------------------------------------------

cat_progress("  Estimating roughness via power variation ratios...")

# For a process with Hurst exponent H:
# E[|X_{t+Δ} - X_t|^q] ~ Δ^(qH)
# Ratio of power variations at different lags gives H

compute_roughness <- function(x, q = 2) {
  x <- x[is.finite(x)]
  n <- length(x)
  
  # Lags to consider
  lags <- c(1, 2, 5, 10, 22)
  lags <- lags[lags < n/10]
  
  # Compute q-th power variation for each lag
  pv <- sapply(lags, function(lag) {
    diffs <- x[(lag+1):n] - x[1:(n-lag)]
    mean(abs(diffs)^q, na.rm = TRUE)
  })
  
  # Regress log(PV) on log(lag) to get qH
  fit <- lm(log(pv) ~ log(lags))
  H <- coef(fit)[2] / q
  se <- summary(fit)$coefficients[2, 2] / q
  
  list(H = H, se = se, r_sq = summary(fit)$r.squared,
       lags = lags, pv = pv)
}

# Roughness for log(RV)
rough_q1 <- compute_roughness(log_rv_daily_clean, q = 1)
rough_q2 <- compute_roughness(log_rv_daily_clean, q = 2)
rough_q3 <- compute_roughness(log_rv_daily_clean, q = 3)

# Roughness for RV in levels
rough_levels <- compute_roughness(rv_daily_clean, q = 2)

cat_progress(sprintf("  Roughness H (log RV): q=1: %.3f, q=2: %.3f, q=3: %.3f",
                     rough_q1$H, rough_q2$H, rough_q3$H))
cat_progress(sprintf("  Roughness H (RV levels): %.3f", rough_levels$H))

# Compare to Gatheral et al. finding of H ≈ 0.1
# Note: Daily data is noisy; intraday data gives cleaner estimates

roughness_results <- data.table(
  series = c("log(RV)", "log(RV)", "log(RV)", "RV_levels"),
  q = c(1, 2, 3, 2),
  H = c(rough_q1$H, rough_q2$H, rough_q3$H, rough_levels$H),
  se = c(rough_q1$se, rough_q2$se, rough_q3$se, rough_levels$se),
  r_squared = c(rough_q1$r_sq, rough_q2$r_sq, rough_q3$r_sq, rough_levels$r_sq)
)

print(roughness_results)
write.csv(roughness_results, file.path(dirs$tables, "A3_roughness_estimates.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# A.3.2 Variogram analysis
# -----------------------------------------------------------------------------

cat_progress("  Computing variogram for roughness confirmation...")

compute_variogram <- function(x, max_lag = 50) {
  x <- x[is.finite(x)]
  n <- length(x)
  
  lags <- 1:min(max_lag, floor(n/5))
  gamma <- sapply(lags, function(h) {
    diffs <- x[(h+1):n] - x[1:(n-h)]
    0.5 * mean(diffs^2, na.rm = TRUE)
  })
  
  list(lags = lags, gamma = gamma)
}

variogram_rv <- compute_variogram(log_rv_daily_clean, max_lag = 100)

# Fit variogram model: γ(h) ~ h^(2H)
valid_gamma <- is.finite(variogram_rv$gamma) & variogram_rv$gamma > 0
if (sum(valid_gamma) > 3) {
  vario_fit <- lm(log(variogram_rv$gamma[valid_gamma]) ~ log(variogram_rv$lags[valid_gamma]))
  H_variogram <- coef(vario_fit)[2] / 2
  cat_progress(sprintf("  Variogram H estimate: %.3f", H_variogram))
} else {
  vario_fit <- NULL
  H_variogram <- NA
  cat_progress("  Variogram estimation failed - insufficient valid points")
}

# Plot
pdf(file.path(dirs$figures, "32_roughness.pdf"), width = 12, height = 5)
par(mfrow = c(1, 3))

# Power variation regression
if (!is.na(rough_q2$H) && length(rough_q2$lags) > 2) {
  plot(log(rough_q2$lags), log(rough_q2$pv), pch = 16, cex = 2,
       main = sprintf("Power Variation (q=2): H = %.3f", rough_q2$H),
       xlab = "log(lag)", ylab = "log(PV)")
  abline(lm(log(rough_q2$pv) ~ log(rough_q2$lags)), col = "red", lwd = 2)
} else {
  plot.new()
  text(0.5, 0.5, "Power variation estimation failed", cex = 1.2)
}

# Variogram
valid_gamma <- is.finite(variogram_rv$gamma) & variogram_rv$gamma > 0
if (!is.na(H_variogram) && sum(valid_gamma) > 3) {
  plot(log(variogram_rv$lags[valid_gamma]), log(variogram_rv$gamma[valid_gamma]), 
       pch = 16, cex = 0.8,
       col = rgb(0, 0, 0.5, 0.5),
       main = sprintf("Variogram: H = %.3f", H_variogram),
       xlab = "log(lag)", ylab = "log(gamma(h))")
  abline(vario_fit, col = "red", lwd = 2)
  # Reference: standard Brownian motion H = 0.5
  abline(a = coef(vario_fit)[1], b = 1, col = "grey", lty = 2)
  legend("bottomright", c(sprintf("H = %.2f", H_variogram), "H = 0.5 (BM)"),
         col = c("red", "grey"), lwd = 2, lty = c(1, 2), bty = "n")
} else {
  plot.new()
  text(0.5, 0.5, "Variogram estimation failed", cex = 1.2)
}

# Comparison across q
qs <- c(0.5, 1, 1.5, 2, 2.5, 3)
Hs <- sapply(qs, function(q) compute_roughness(log_rv_daily_clean, q = q)$H)
valid_Hs <- is.finite(Hs)
if (sum(valid_Hs) > 2) {
  plot(qs[valid_Hs], Hs[valid_Hs], type = "b", pch = 16, cex = 1.5,
       main = "Roughness vs Power q",
       xlab = "Power q", ylab = "H estimate",
       ylim = range(Hs[valid_Hs], na.rm = TRUE) * c(0.8, 1.2))
  abline(h = mean(Hs, na.rm = TRUE), col = "red", lty = 2)
  abline(h = 0.1, col = "blue", lty = 2)  # Gatheral et al. reference
  legend("topright", c(sprintf("Mean H = %.3f", mean(Hs, na.rm = TRUE)), "H = 0.1 (rough vol)"),
         col = c("red", "blue"), lty = 2, bty = "n")
} else {
  plot.new()
  text(0.5, 0.5, "Insufficient roughness estimates across q", cex = 1.2)
}

dev.off()

#------------------------------------------------------------------
# A.4 JUMP AND DISCONTINUITY ANALYSIS
#------------------------------------------------------------------

cat_progress("A.4: Jump and Discontinuity Analysis")

# -----------------------------------------------------------------------------
# A.4.1 Bipower variation ratio (daily proxy)
# BPV/RV → 1 under continuous paths, < 1 with jumps
# Barndorff-Nielsen & Shephard (2004)
# -----------------------------------------------------------------------------

# For daily data, we use absolute return products as proxy
# μ_1^{-2} * |r_t| * |r_{t-1}| where μ_1 = sqrt(2/π)
mu_1 <- sqrt(2/pi)

dt[, abs_ret := abs(log_return)]
dt[, abs_ret_lag := shift(abs_ret, 1)]
dt[, bpv_daily := (1/mu_1^2) * abs_ret * abs_ret_lag]

# Rolling BPV and RV
dt[, rv_sq := log_return^2]
dt[, bpv_22d := frollmean(bpv_daily, n = 22, align = "right", na.rm = TRUE) * 252]
dt[, rv_22d := frollmean(rv_sq, n = 22, align = "right", na.rm = TRUE) * 252]

# Jump ratio: BPV/RV
dt[, jump_ratio := bpv_22d / rv_22d]
dt[, jump_ratio := pmin(jump_ratio, 1.5)]  # Cap outliers

# Under no jumps, ratio should be close to 1 (actually π/2 ≈ 1.57 for BPV construction)
# We use scaled version so ratio → 1

cat_progress(sprintf("  Mean jump ratio (BPV/RV): %.3f (theoretical no-jump: 1.0)",
                     mean(dt$jump_ratio, na.rm = TRUE)))

# -----------------------------------------------------------------------------
# A.4.2 Large return identification (jump proxy)
# -----------------------------------------------------------------------------

# Identify returns that exceed k * local volatility
local_vol <- dt[, frollapply(log_return, n = 22, FUN = sd, align = "right")]
dt[, local_vol := local_vol]
dt[, std_return := log_return / local_vol]

# Jump threshold: |z| > 4 (approx 3 per 10,000 under normality)
jump_threshold <- 4
dt[, jump_flag := abs(std_return) > jump_threshold]

n_jumps <- sum(dt$jump_flag, na.rm = TRUE)
jump_rate <- mean(dt$jump_flag, na.rm = TRUE) * 252  # Annualised
expected_rate <- 2 * pnorm(-jump_threshold) * 252

cat_progress(sprintf("  Jump detection (|z|>4): %d jumps, rate=%.2f/year (expected under normal: %.2f)",
                     n_jumps, jump_rate, expected_rate))

# -----------------------------------------------------------------------------
# A.4.3 Signed jump variation (up vs down jumps)
# -----------------------------------------------------------------------------

dt[, signed_jump := ifelse(jump_flag, sign(log_return), 0)]
up_jumps <- sum(dt$signed_jump == 1, na.rm = TRUE)
down_jumps <- sum(dt$signed_jump == -1, na.rm = TRUE)

cat_progress(sprintf("  Up jumps: %d, Down jumps: %d (asymmetry ratio: %.2f)",
                     up_jumps, down_jumps, down_jumps / max(up_jumps, 1)))

# -----------------------------------------------------------------------------
# A.4.4 Threshold RV decomposition
# Following Mancini (2009): truncate returns above threshold
# -----------------------------------------------------------------------------

# Threshold for continuous component
threshold_c <- 3 * local_vol  # 3 sigma

dt[, ret_truncated := ifelse(abs(log_return) > threshold_c, 0, log_return)]
dt[, rv_continuous := frollsum(ret_truncated^2, n = 22, align = "right", na.rm = TRUE)]
dt[, rv_continuous := sqrt(rv_continuous * 252 / 22) * 100]

dt[, rv_jump := sqrt(pmax(0, (rv_monthly/100)^2 - (rv_continuous/100)^2)) * 100]

# Jump summary
jump_summary <- dt[, .(
  mean_rv_total = mean(rv_monthly, na.rm = TRUE),
  mean_rv_continuous = mean(rv_continuous, na.rm = TRUE),
  mean_rv_jump = mean(rv_jump, na.rm = TRUE),
  pct_jump = mean(rv_jump / rv_monthly, na.rm = TRUE) * 100
)]

print(jump_summary)

# -----------------------------------------------------------------------------
# A.4.5 Jump analysis plots
# -----------------------------------------------------------------------------

pdf(file.path(dirs$figures, "33_jump_analysis.pdf"), width = 14, height = 10)

par(mfrow = c(2, 2))

# Jump ratio time series
plot(dt$date, dt$jump_ratio, type = "l", col = rgb(0, 0, 0.5, 0.5),
     main = "Bipower Variation Ratio (BPV/RV)",
     xlab = "Date", ylab = "Ratio", ylim = c(0.5, 1.5))
abline(h = 1, col = "red", lty = 2)
abline(h = mean(dt$jump_ratio, na.rm = TRUE), col = "blue", lty = 2)

# Jump flags on return series
plot(dt$date, dt$log_return, type = "l", col = "grey",
     main = sprintf("Returns with Detected Jumps (n=%d)", n_jumps),
     xlab = "Date", ylab = "Log Return")
points(dt$date[dt$jump_flag == TRUE], dt$log_return[dt$jump_flag == TRUE],
       col = ifelse(dt$log_return[dt$jump_flag == TRUE] > 0, "green", "red"),
       pch = 16, cex = 1)

# Standardised returns distribution vs normal
hist(dt$std_return[is.finite(dt$std_return)], breaks = 100, freq = FALSE,
     main = "Standardised Returns vs Normal",
     xlab = "z-score", col = "lightblue", xlim = c(-8, 8))
x_seq <- seq(-8, 8, 0.1)
lines(x_seq, dnorm(x_seq), col = "red", lwd = 2)
legend("topright", c("Empirical", "N(0,1)"), col = c("lightblue", "red"), 
       lwd = c(10, 2), bty = "n")

# RV decomposition: continuous vs jump
valid_idx <- !is.na(dt$rv_continuous) & !is.na(dt$rv_jump)
plot(dt$date[valid_idx], dt$rv_monthly[valid_idx], type = "l", col = "black",
     main = "RV Decomposition: Total vs Continuous",
     xlab = "Date", ylab = "RV (%)")
lines(dt$date[valid_idx], dt$rv_continuous[valid_idx], col = "blue")
legend("topright", c("Total RV", "Continuous"), col = c("black", "blue"), lwd = 1, bty = "n")

dev.off()

#------------------------------------------------------------------
# A.5 ASYMMETRY AND LEVERAGE EFFECT ANALYSIS
#------------------------------------------------------------------

cat_progress("A.5: Asymmetry and Leverage Effect Analysis")

# -----------------------------------------------------------------------------
# A.5.1 Contemporaneous asymmetry
# -----------------------------------------------------------------------------

# Correlation between returns and volatility changes
dt[, rv_change := rv_daily - shift(rv_daily, 1)]
dt[, rv_pct_change := (rv_daily / shift(rv_daily, 1) - 1) * 100]

# Overall
cor_contemp <- cor(dt$log_return, dt$rv_change, use = "complete.obs")

# Asymmetric: down days vs up days
down_days <- dt$log_return < 0
cor_down <- cor(dt$log_return[down_days], dt$rv_change[down_days], use = "complete.obs")
cor_up <- cor(dt$log_return[!down_days], dt$rv_change[!down_days], use = "complete.obs")

cat_progress(sprintf("  Contemporaneous corr(r, ΔRV): overall=%.3f, down=%.3f, up=%.3f",
                     cor_contemp, cor_down, cor_up))

# -----------------------------------------------------------------------------
# A.5.2 Lead-lag relationship (leverage vs volatility feedback)
# -----------------------------------------------------------------------------

# Leverage: past returns predict future volatility
# Volatility feedback: past volatility predicts future returns

max_lag <- 22
leverage_cors <- sapply(1:max_lag, function(k) {
  cor(dt$log_return, shift(dt$rv_daily, -k), use = "complete.obs")
})

feedback_cors <- sapply(1:max_lag, function(k) {
  cor(dt$rv_daily, shift(dt$log_return, -k), use = "complete.obs")
})

# Cross-correlation function
ccf_ret_rv <- ccf(dt$log_return, dt$rv_daily, lag.max = 30, plot = FALSE, na.action = na.pass)

# -----------------------------------------------------------------------------
# A.5.3 Signed volatility: positive vs negative returns
# -----------------------------------------------------------------------------

dt[, rv_pos := ifelse(log_return >= 0, log_return^2, 0)]
dt[, rv_neg := ifelse(log_return < 0, log_return^2, 0)]

# Rolling signed RV
dt[, rv_pos_22d := sqrt(frollsum(rv_pos, n = 22, align = "right", na.rm = TRUE) * 252 / 22) * 100]
dt[, rv_neg_22d := sqrt(frollsum(rv_neg, n = 22, align = "right", na.rm = TRUE) * 252 / 22) * 100]

# Ratio of negative to positive RV
dt[, neg_pos_ratio := rv_neg_22d / rv_pos_22d]

cat_progress(sprintf("  Mean signed RV: positive=%.2f%%, negative=%.2f%%, ratio=%.2f",
                     mean(dt$rv_pos_22d, na.rm = TRUE),
                     mean(dt$rv_neg_22d, na.rm = TRUE),
                     mean(dt$neg_pos_ratio, na.rm = TRUE)))

# -----------------------------------------------------------------------------
# A.5.4 News Impact Curve (nonparametric)
# -----------------------------------------------------------------------------

# Bin returns and compute mean subsequent RV change
dt[, ret_bin := cut(log_return, breaks = quantile(log_return, probs = seq(0, 1, 0.05), na.rm = TRUE),
                    include.lowest = TRUE, labels = FALSE)]

news_impact <- dt[, .(
  mean_return = mean(log_return, na.rm = TRUE),
  mean_rv_next = mean(shift(rv_daily, -1), na.rm = TRUE),
  mean_rv_change = mean(shift(rv_change, -1), na.rm = TRUE),
  n = .N
), by = ret_bin][order(ret_bin)]

# -----------------------------------------------------------------------------
# A.5.5 Asymmetry plots
# -----------------------------------------------------------------------------

pdf(file.path(dirs$figures, "34_asymmetry_leverage.pdf"), width = 14, height = 10)

par(mfrow = c(2, 2))

# Cross-correlation
plot(ccf_ret_rv$lag, ccf_ret_rv$acf, type = "h", lwd = 2,
     main = "Cross-Correlation: Returns vs RV",
     xlab = "Lag (negative = returns lead)", ylab = "CCF")
abline(h = 0, col = "grey")
abline(h = c(-1, 1) * qnorm(0.975) / sqrt(nrow(dt)), col = "red", lty = 2)

# Lead-lag correlations
plot(1:max_lag, leverage_cors, type = "b", pch = 16, col = "blue",
     main = "Leverage vs Feedback Effects",
     xlab = "Lag (days)", ylab = "Correlation", ylim = range(c(leverage_cors, feedback_cors)))
lines(1:max_lag, feedback_cors, type = "b", pch = 17, col = "red")
abline(h = 0, col = "grey")
legend("topright", c("Leverage: r_t → RV_{t+k}", "Feedback: RV_t → r_{t+k}"),
       col = c("blue", "red"), pch = c(16, 17), lty = 1, bty = "n")

# News impact curve
plot(news_impact$mean_return * 100, news_impact$mean_rv_next,
     type = "b", pch = 16, cex = 1.2,
     main = "News Impact Curve",
     xlab = "Return (%)", ylab = "Next-day RV (%)")
# Fit asymmetric quadratic
ni_fit <- lm(mean_rv_next ~ mean_return + I(mean_return^2) + I(mean_return * (mean_return < 0)),
             data = news_impact)
x_seq <- seq(min(news_impact$mean_return), max(news_impact$mean_return), length.out = 100)
y_pred <- predict(ni_fit, newdata = data.frame(mean_return = x_seq))
lines(x_seq * 100, y_pred, col = "red", lwd = 2)

# Signed RV over time
plot(dt$date, dt$rv_neg_22d, type = "l", col = "red",
     main = "Signed Realised Volatility (22-day)",
     xlab = "Date", ylab = "RV (%)", ylim = c(0, max(dt$rv_neg_22d, na.rm = TRUE) * 1.1))
lines(dt$date, dt$rv_pos_22d, col = "green")
legend("topright", c("RV (negative returns)", "RV (positive returns)"),
       col = c("red", "green"), lwd = 1, bty = "n")

dev.off()

#------------------------------------------------------------------
# A.6 REGIME AND STRUCTURAL BREAK ANALYSIS
#------------------------------------------------------------------

cat_progress("A.6: Regime and Structural Break Analysis")

# -----------------------------------------------------------------------------
# A.6.1 Bai-Perron structural break test
# -----------------------------------------------------------------------------

cat_progress("  Testing for structural breaks in RV...")

# Prepare data for strucchange - use cleaned log RV
rv_ts <- ts(log_rv_daily_clean, frequency = 252)

# Bai-Perron test (computationally intensive for large samples)
# Use subsample if necessary
if (length(rv_ts) > 5000) {
  cat_progress("  Using subsample for Bai-Perron test...")
  set.seed(42)
  bp_idx <- sort(sample(length(rv_ts), 5000))
  rv_ts_sub <- rv_ts[bp_idx]
} else {
  rv_ts_sub <- rv_ts
}

bp_test <- tryCatch({
  strucchange::breakpoints(rv_ts_sub ~ 1, breaks = 5)
}, error = function(e) {
  cat_progress(sprintf("  Bai-Perron test failed: %s", e$message))
  NULL
})

if (!is.null(bp_test) && !is.null(bp_test$breakpoints) && !any(is.na(bp_test$breakpoints))) {
  bp_summary <- summary(bp_test)
  cat_progress(sprintf("  Detected %d structural breaks", length(bp_test$breakpoints)))
  if (length(bp_test$breakpoints) > 0) {
    # Map back to dates - need to handle the subsample case
    finite_log_rv_idx <- which(is.finite(dt$log_rv_daily))
    if (length(rv_ts) > 5000) {
      break_dates <- dt$date[finite_log_rv_idx[bp_idx[bp_test$breakpoints]]]
    } else {
      break_dates <- dt$date[finite_log_rv_idx[bp_test$breakpoints]]
    }
    cat_progress(sprintf("  Break dates: %s", paste(break_dates, collapse = ", ")))
  }
} else {
  cat_progress("  No structural breaks detected or test failed")
}

# -----------------------------------------------------------------------------
# A.6.2 Two-state regime analysis (high/low volatility)
# -----------------------------------------------------------------------------

# Simple threshold-based regime identification
rv_threshold <- median(dt$rv_daily, na.rm = TRUE)
dt[, regime := ifelse(rv_daily > rv_threshold, "High", "Low")]

# Regime statistics
regime_stats <- dt[, .(
  mean_rv = mean(rv_daily, na.rm = TRUE),
  sd_rv = sd(rv_daily, na.rm = TRUE),
  mean_return = mean(log_return, na.rm = TRUE) * 252,  # Annualised
  sd_return = sd(log_return, na.rm = TRUE) * sqrt(252),
  sharpe = mean(log_return, na.rm = TRUE) / sd(log_return, na.rm = TRUE) * sqrt(252),
  n_days = .N,
  pct_days = .N / nrow(dt) * 100
), by = regime]

print(regime_stats)
write.csv(regime_stats, file.path(dirs$tables, "A6_regime_statistics.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# A.6.3 Regime persistence and transition
# -----------------------------------------------------------------------------

# Transition matrix
dt[, regime_lag := shift(regime, 1)]
transition_table <- dt[!is.na(regime) & !is.na(regime_lag), 
                       .N, by = .(from = regime_lag, to = regime)]
transition_table[, prob := N / sum(N), by = from]

cat_progress("  Regime transition probabilities:")
print(dcast(transition_table, from ~ to, value.var = "prob"))

# Regime duration
dt[, regime_change := regime != shift(regime, 1)]
dt[, regime_spell := cumsum(c(1, diff(as.numeric(regime_change != 0 | is.na(regime_change)))))]

regime_durations <- dt[, .(
  duration = .N,
  regime = regime[1]
), by = regime_spell][!is.na(regime)]

duration_summary <- regime_durations[, .(
  mean_duration = mean(duration),
  median_duration = median(duration),
  max_duration = max(duration)
), by = regime]

print(duration_summary)

# -----------------------------------------------------------------------------
# A.6.4 Markov regime switching (simple estimation)
# -----------------------------------------------------------------------------

cat_progress("  Estimating Markov switching model...")

# Hamilton filter for 2-state model
# Simplified: use threshold approach with smoothing

# Rolling probability of high regime
dt[, p_high := frollmean(as.numeric(regime == "High"), n = 63, align = "right", na.rm = TRUE)]

# -----------------------------------------------------------------------------
# A.6.5 Regime plots
# -----------------------------------------------------------------------------

pdf(file.path(dirs$figures, "35_regime_analysis.pdf"), width = 14, height = 10)

par(mfrow = c(2, 2))

# RV with regime shading
plot(dt$date, dt$rv_daily, type = "l", col = "darkblue",
     main = "RV with Regime Classification",
     xlab = "Date", ylab = "RV (%)")
abline(h = rv_threshold, col = "red", lty = 2)
legend("topright", sprintf("Threshold: %.1f%%", rv_threshold), col = "red", lty = 2, bty = "n")

# Regime probability
plot(dt$date, dt$p_high, type = "l", col = "purple",
     main = "Probability of High Volatility Regime (63d rolling)",
     xlab = "Date", ylab = "P(High)")
abline(h = 0.5, col = "grey", lty = 2)

# Regime-conditional return distribution
boxplot(log_return * 100 ~ regime, data = dt,
        main = "Return Distribution by Regime",
        xlab = "Regime", ylab = "Return (%)",
        col = c("lightgreen", "lightcoral"))

# Duration histogram
hist(regime_durations$duration[regime_durations$regime == "High"], breaks = 30,
     main = "High Volatility Regime Duration",
     xlab = "Duration (days)", col = "lightcoral",
     freq = TRUE)
abline(v = mean(regime_durations$duration[regime_durations$regime == "High"]), 
       col = "red", lwd = 2, lty = 2)

dev.off()

#------------------------------------------------------------------
#
#                  PART B: FORWARD 22-DAY RV AND VRP ANALYSIS
#
#------------------------------------------------------------------

cat_progress("=")
cat_progress("PART B: FORWARD 22-DAY RV AND VRP ANALYSIS")
cat_progress("=")

#------------------------------------------------------------------
# B.1 VIX-RV RELATIONSHIP ANALYSIS
#------------------------------------------------------------------

cat_progress("B.1: VIX-RV Relationship Analysis")

# -----------------------------------------------------------------------------
# B.1.1 Distributional comparison
# -----------------------------------------------------------------------------

# VIX is risk-neutral expectation, rv_cc is physical realisation
vix_rv_stats <- data.table(
  variable = c("VIX", "RV_22d_forward"),
  mean = c(mean(dt$vix_close, na.rm = TRUE), mean(dt$rv_cc, na.rm = TRUE)),
  sd = c(sd(dt$vix_close, na.rm = TRUE), sd(dt$rv_cc, na.rm = TRUE)),
  skew = c(moments::skewness(dt$vix_close, na.rm = TRUE), 
           moments::skewness(dt$rv_cc, na.rm = TRUE)),
  kurt = c(moments::kurtosis(dt$vix_close, na.rm = TRUE),
           moments::kurtosis(dt$rv_cc, na.rm = TRUE)),
  q05 = c(quantile(dt$vix_close, 0.05, na.rm = TRUE),
          quantile(dt$rv_cc, 0.05, na.rm = TRUE)),
  q50 = c(quantile(dt$vix_close, 0.50, na.rm = TRUE),
          quantile(dt$rv_cc, 0.50, na.rm = TRUE)),
  q95 = c(quantile(dt$vix_close, 0.95, na.rm = TRUE),
          quantile(dt$rv_cc, 0.95, na.rm = TRUE))
)

print(vix_rv_stats)

# -----------------------------------------------------------------------------
# B.1.2 Forecasting regression: VIX → RV
# -----------------------------------------------------------------------------

# Mincer-Zarnowitz regression: RV = α + β*VIX + ε
# Under unbiased forecast: α = 0, β = 1
# Under efficient forecast: R² should be high

mz_reg <- lm(rv_cc ~ vix_close, data = dt)
mz_summary <- summary(mz_reg)

# HAC standard errors (Newey-West)
mz_hac <- lmtest::coeftest(mz_reg, vcov = sandwich::NeweyWest(mz_reg, lag = 22))

cat_progress("  Mincer-Zarnowitz regression (VIX → RV_22d):")
cat_progress(sprintf("    α = %.3f (se = %.3f), t = %.2f",
                     coef(mz_reg)[1], mz_hac[1, 2], mz_hac[1, 3]))
cat_progress(sprintf("    β = %.3f (se = %.3f), t = %.2f",
                     coef(mz_reg)[2], mz_hac[2, 2], mz_hac[2, 3]))
cat_progress(sprintf("    R² = %.3f", mz_summary$r.squared))

# Test joint hypothesis α = 0, β = 1
mz_wald <- car::linearHypothesis(mz_reg, c("(Intercept) = 0", "vix_close = 1"),
                                 vcov = sandwich::NeweyWest(mz_reg, lag = 22))

cat_progress(sprintf("    Wald test (α=0, β=1): χ² = %.2f, p = %.4f",
                     mz_wald$Chisq[2], mz_wald$`Pr(>Chisq)`[2]))

# -----------------------------------------------------------------------------
# B.1.3 Conditional efficiency: VIX under/overestimates in different regimes
# -----------------------------------------------------------------------------

# Split by VIX level
dt[, vix_regime := cut(vix_close, breaks = quantile(vix_close, c(0, 0.25, 0.75, 1), na.rm = TRUE),
                       labels = c("Low", "Medium", "High"), include.lowest = TRUE)]

conditional_bias <- dt[, .(
  mean_vix = mean(vix_close, na.rm = TRUE),
  mean_rv = mean(rv_cc, na.rm = TRUE),
  bias = mean(vix_close - rv_cc, na.rm = TRUE),
  bias_pct = mean((vix_close - rv_cc) / rv_cc, na.rm = TRUE) * 100,
  n = .N
), by = vix_regime][order(vix_regime)]

print(conditional_bias)
write.csv(conditional_bias, file.path(dirs$tables, "B1_conditional_bias.csv"), row.names = FALSE)

#------------------------------------------------------------------
# B.2 VARIANCE RISK PREMIUM ANALYSIS
#------------------------------------------------------------------

cat_progress("B.2: Variance Risk Premium Analysis")

# -----------------------------------------------------------------------------
# B.2.1 VRP statistics and persistence
# -----------------------------------------------------------------------------

vrp_stats <- dt[, .(
  mean = mean(vrp_cc, na.rm = TRUE),
  sd = sd(vrp_cc, na.rm = TRUE),
  t_stat = mean(vrp_cc, na.rm = TRUE) / (sd(vrp_cc, na.rm = TRUE) / sqrt(sum(!is.na(vrp_cc)))),
  pct_positive = mean(vrp_cc > 0, na.rm = TRUE) * 100,
  q05 = quantile(vrp_cc, 0.05, na.rm = TRUE),
  q50 = quantile(vrp_cc, 0.50, na.rm = TRUE),
  q95 = quantile(vrp_cc, 0.95, na.rm = TRUE),
  ar1 = cor(vrp_cc, shift(vrp_cc, 1), use = "complete.obs")
)]

cat_progress(sprintf("  VRP mean: %.2f (t = %.2f), %% positive: %.1f%%",
                     vrp_stats$mean, vrp_stats$t_stat, vrp_stats$pct_positive))

# VRP ACF
acf_vrp <- acf(dt$vrp_cc, lag.max = 252, plot = FALSE, na.action = na.pass)

# -----------------------------------------------------------------------------
# B.2.2 VRP predictability for future returns
# -----------------------------------------------------------------------------

# VRP predicts future market returns (Bollerslev, Tauchen, Zhou 2009)
dt[, ret_future_1d := shift(log_return, -1)]
dt[, ret_future_5d := shift(frollsum(log_return, n = 5, align = "left"), -1)]
dt[, ret_future_22d := shift(frollsum(log_return, n = 22, align = "left"), -1)]

# Predictive regressions
pred_1d <- lm(ret_future_1d ~ vrp_cc, data = dt)
pred_5d <- lm(ret_future_5d ~ vrp_cc, data = dt)
pred_22d <- lm(ret_future_22d ~ vrp_cc, data = dt)

# HAC inference
pred_results <- data.table(
  horizon = c("1d", "5d", "22d"),
  beta = c(coef(pred_1d)[2], coef(pred_5d)[2], coef(pred_22d)[2]),
  se_hac = c(
    lmtest::coeftest(pred_1d, vcov = sandwich::NeweyWest(pred_1d))[2, 2],
    lmtest::coeftest(pred_5d, vcov = sandwich::NeweyWest(pred_5d, lag = 5))[2, 2],
    lmtest::coeftest(pred_22d, vcov = sandwich::NeweyWest(pred_22d, lag = 22))[2, 2]
  ),
  r_squared = c(summary(pred_1d)$r.squared, 
                summary(pred_5d)$r.squared,
                summary(pred_22d)$r.squared)
)
pred_results[, t_stat := beta / se_hac]

cat_progress("  VRP return predictability:")
print(pred_results)
write.csv(pred_results, file.path(dirs$tables, "B2_vrp_return_predictability.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# B.2.3 VRP decomposition by market conditions
# -----------------------------------------------------------------------------

# VRP under different market states
dt[, market_state := cut(frollmean(log_return, n = 63, align = "right", na.rm = TRUE) * 252,
                         breaks = c(-Inf, -0.05, 0.05, Inf),
                         labels = c("Bear", "Neutral", "Bull"))]

vrp_by_state <- dt[, .(
  mean_vrp = mean(vrp_cc, na.rm = TRUE),
  sd_vrp = sd(vrp_cc, na.rm = TRUE),
  mean_vix = mean(vix_close, na.rm = TRUE),
  mean_rv = mean(rv_cc, na.rm = TRUE),
  n = .N
), by = market_state][order(market_state)]

print(vrp_by_state)

# -----------------------------------------------------------------------------
# B.2.4 VRP plots
# -----------------------------------------------------------------------------

pdf(file.path(dirs$figures, "36_vrp_analysis.pdf"), width = 14, height = 10)

par(mfrow = c(2, 2))

# VRP time series
plot(dt$date, dt$vrp_cc, type = "l", col = rgb(0, 0, 0.5, 0.5),
     main = "Variance Risk Premium (VIX - RV)",
     xlab = "Date", ylab = "VRP (%)")
abline(h = 0, col = "red", lty = 2)
abline(h = mean(dt$vrp_cc, na.rm = TRUE), col = "blue", lty = 2)

# VRP ACF
plot(0:100, acf_vrp$acf[1:101], type = "h", lwd = 2,
     main = "ACF of VRP",
     xlab = "Lag", ylab = "ACF", col = "darkblue")
abline(h = qnorm(0.975) / sqrt(nrow(dt)), col = "red", lty = 2)
abline(h = -qnorm(0.975) / sqrt(nrow(dt)), col = "red", lty = 2)

# VIX vs RV scatter
plot(dt$vix_close, dt$rv_cc, pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.2),
     main = "VIX vs Realised Volatility (22-day forward)",
     xlab = "VIX", ylab = "RV (%)")
abline(0, 1, col = "red", lwd = 2)
abline(mz_reg, col = "blue", lwd = 2, lty = 2)
legend("topleft", c("45° line", "OLS fit"),
       col = c("red", "blue"), lwd = 2, lty = c(1, 2), bty = "n")

# VRP distribution
hist(dt$vrp_cc, breaks = 80, freq = FALSE,
     main = "VRP Distribution",
     xlab = "VRP (%)", col = "lightblue")
abline(v = 0, col = "red", lwd = 2)
abline(v = mean(dt$vrp_cc, na.rm = TRUE), col = "blue", lwd = 2, lty = 2)

dev.off()

#------------------------------------------------------------------
# B.3 RV ESTIMATOR COMPARISON
#------------------------------------------------------------------

cat_progress("B.3: RV Estimator Comparison")

# -----------------------------------------------------------------------------
# B.3.1 Efficiency comparison
# -----------------------------------------------------------------------------

# Under GBM, relative efficiency (variance reduction):
# Parkinson: 5.2x
# Garman-Klass: 8.4x
# Rogers-Satchell: drift-independent
# Yang-Zhang: handles overnight gaps

rv_estimators <- c("rv_cc", "rv_parkinson", "rv_gk", "rv_rs", "rv_yz")

# Correlation with forward RV (which is best predictor?)
rv_correlations <- cor(dt[, .SD, .SDcols = c(rv_estimators, "vix_close")], 
                       use = "pairwise.complete.obs")

# Noise reduction: compare standard deviation
rv_comparison <- data.table(
  estimator = rv_estimators,
  mean = sapply(rv_estimators, function(x) mean(dt[[x]], na.rm = TRUE)),
  sd = sapply(rv_estimators, function(x) sd(dt[[x]], na.rm = TRUE)),
  cv = sapply(rv_estimators, function(x) sd(dt[[x]], na.rm = TRUE) / mean(dt[[x]], na.rm = TRUE)),
  cor_with_vix = sapply(rv_estimators, function(x) cor(dt[[x]], dt$vix_close, use = "complete.obs"))
)

# Relative efficiency (using close-to-close as baseline)
rv_comparison[, relative_efficiency := sd[1]^2 / sd^2]

print(rv_comparison)
write.csv(rv_comparison, file.path(dirs$tables, "B3_rv_estimator_comparison.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# B.3.2 Bias analysis under market microstructure
# -----------------------------------------------------------------------------

# Compare estimators during high vs low volume days
dt[, volume_quintile := cut(volume, breaks = quantile(volume, probs = seq(0, 1, 0.2), na.rm = TRUE),
                            labels = 1:5, include.lowest = TRUE)]

estimator_by_volume <- dt[, lapply(.SD, mean, na.rm = TRUE), 
                          by = volume_quintile, 
                          .SDcols = rv_estimators][order(volume_quintile)]

cat_progress("  RV estimators by volume quintile:")
print(estimator_by_volume)

#------------------------------------------------------------------
#
#            PART C: PREDICTABILITY STRUCTURE FOR NEXT-DAY RV
#
#------------------------------------------------------------------

cat_progress("=")
cat_progress("PART C: PREDICTABILITY STRUCTURE FOR NEXT-DAY RV")
cat_progress("=")

#------------------------------------------------------------------
# C.1 HAR-RV MODEL ANALYSIS
#------------------------------------------------------------------

cat_progress("C.1: HAR-RV Model Analysis")

# Corsi (2009) HAR-RV: RV_{t+1} = α + β_d * RV_t + β_w * RV_t^{(w)} + β_m * RV_t^{(m)} + ε

# Next-day RV target
dt[, rv_next := shift(rv_daily, -1)]

# HAR-RV regression
har_rv <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly, data = dt)
har_summary <- summary(har_rv)

# HAC standard errors
har_hac <- lmtest::coeftest(har_rv, vcov = sandwich::NeweyWest(har_rv))

cat_progress("  HAR-RV model coefficients:")
cat_progress(sprintf("    β_d = %.4f (se = %.4f, t = %.2f)",
                     coef(har_rv)[2], har_hac[2, 2], har_hac[2, 3]))
cat_progress(sprintf("    β_w = %.4f (se = %.4f, t = %.2f)",
                     coef(har_rv)[3], har_hac[3, 2], har_hac[3, 3]))
cat_progress(sprintf("    β_m = %.4f (se = %.4f, t = %.2f)",
                     coef(har_rv)[4], har_hac[4, 2], har_hac[4, 3]))
cat_progress(sprintf("    R² = %.4f, Adj-R² = %.4f",
                     har_summary$r.squared, har_summary$adj.r.squared))

# Store fitted values for comparison
dt[, har_fitted := predict(har_rv, newdata = dt)]

# -----------------------------------------------------------------------------
# C.1.1 HAR-RV-J: Adding jump component
# -----------------------------------------------------------------------------

# Jump component from earlier analysis
dt[, rv_jump_component := rv_daily - rv_continuous / sqrt(22)]
dt[, rv_jump_component := pmax(rv_jump_component, 0)]

har_rv_j <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly + rv_jump_component, data = dt)

cat_progress(sprintf("  HAR-RV-J R²: %.4f (improvement: %.4f)",
                     summary(har_rv_j)$r.squared,
                     summary(har_rv_j)$r.squared - har_summary$r.squared))

# -----------------------------------------------------------------------------
# C.1.2 HAR-RV with leverage
# -----------------------------------------------------------------------------

dt[, rv_neg_daily := sqrt(rv_neg) * sqrt(252) * 100]
dt[, rv_neg_daily := ifelse(is.finite(rv_neg_daily), rv_neg_daily, 0)]

har_rv_lev <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly + rv_neg_daily, data = dt)

cat_progress(sprintf("  HAR-RV-Leverage R²: %.4f (improvement: %.4f)",
                     summary(har_rv_lev)$r.squared,
                     summary(har_rv_lev)$r.squared - har_summary$r.squared))

#------------------------------------------------------------------
# C.2 CROSS-CORRELATION AND LAG STRUCTURE
#------------------------------------------------------------------

cat_progress("C.2: Cross-Correlation and Lag Structure Analysis")

# -----------------------------------------------------------------------------
# C.2.1 Optimal lag selection for various predictors
# -----------------------------------------------------------------------------

# Compute correlations at multiple lags
predictors <- c("rv_daily", "vix_close", "vrp_cc", "abs_ret", "volume")
max_lag_test <- 30

lag_correlations <- data.table()

for (pred in predictors) {
  if (pred %in% names(dt)) {
    cors <- sapply(1:max_lag_test, function(k) {
      cor(dt$rv_next, shift(dt[[pred]], k), use = "complete.obs")
    })
    lag_correlations <- rbind(lag_correlations,
                              data.table(predictor = pred, lag = 1:max_lag_test, correlation = cors))
  }
}

# Find optimal lag for each predictor
optimal_lags <- lag_correlations[, .(
  optimal_lag = lag[which.max(abs(correlation))],
  max_correlation = max(abs(correlation))
), by = predictor]

print(optimal_lags)

# -----------------------------------------------------------------------------
# C.2.2 Granger causality tests
# -----------------------------------------------------------------------------

cat_progress("  Granger causality tests...")

# VIX → RV
gc_vix_rv <- tryCatch({
  lmtest::grangertest(rv_daily ~ vix_close, order = 5, data = dt)
}, error = function(e) NULL)

# Returns → RV
gc_ret_rv <- tryCatch({
  lmtest::grangertest(rv_daily ~ log_return, order = 5, data = dt)
}, error = function(e) NULL)

if (!is.null(gc_vix_rv)) {
  cat_progress(sprintf("  Granger: VIX → RV, F = %.2f, p = %.4f",
                       gc_vix_rv$F[2], gc_vix_rv$`Pr(>F)`[2]))
}

#------------------------------------------------------------------
# C.3 NONLINEAR RELATIONSHIPS
#------------------------------------------------------------------

cat_progress("C.3: Nonlinear Relationship Analysis")

# -----------------------------------------------------------------------------
# C.3.1 State-dependent predictability
# -----------------------------------------------------------------------------

# Does predictability differ in high vs low volatility states?
dt[, rv_state := ifelse(rv_daily > median(rv_daily, na.rm = TRUE), "High", "Low")]

har_high <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly, 
               data = dt[rv_state == "High"])
har_low <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly, 
              data = dt[rv_state == "Low"])

cat_progress(sprintf("  HAR R² by state: High = %.4f, Low = %.4f",
                     summary(har_high)$r.squared, summary(har_low)$r.squared))

# -----------------------------------------------------------------------------
# C.3.2 Mutual information (nonlinear dependence)
# -----------------------------------------------------------------------------

cat_progress("  Computing mutual information...")

# Discretise variables for MI computation
n_bins <- 20

mi_results <- data.table(
  predictor = c("rv_daily", "vix_close", "log_return", "vrp_cc"),
  linear_cor = numeric(4),
  mutual_info = numeric(4)
)

for (i in 1:nrow(mi_results)) {
  pred <- mi_results$predictor[i]
  if (pred %in% names(dt)) {
    valid <- complete.cases(dt[, c("rv_next", pred), with = FALSE])
    x <- dt[[pred]][valid]
    y <- dt$rv_next[valid]
    
    mi_results$linear_cor[i] <- cor(x, y)
    
    # Discretise for MI
    x_disc <- cut(x, breaks = n_bins, labels = FALSE)
    y_disc <- cut(y, breaks = n_bins, labels = FALSE)
    
    # Joint and marginal entropy
    joint <- table(x_disc, y_disc)
    p_joint <- joint / sum(joint)
    p_x <- rowSums(p_joint)
    p_y <- colSums(p_joint)
    
    H_xy <- -sum(p_joint[p_joint > 0] * log(p_joint[p_joint > 0]))
    H_x <- -sum(p_x[p_x > 0] * log(p_x[p_x > 0]))
    H_y <- -sum(p_y[p_y > 0] * log(p_y[p_y > 0]))
    
    mi_results$mutual_info[i] <- H_x + H_y - H_xy
  }
}

# Normalised MI
mi_results[, normalised_mi := mutual_info / max(mutual_info)]

print(mi_results)
write.csv(mi_results, file.path(dirs$tables, "C3_mutual_information.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# C.3.3 Partial dependence exploration
# -----------------------------------------------------------------------------

# Fit GAM to capture nonlinearities
gam_rv <- mgcv::gam(rv_next ~ s(rv_daily) + s(rv_weekly) + s(rv_monthly) + s(vix_close),
                    data = dt[complete.cases(dt[, c("rv_next", "rv_daily", "rv_weekly", 
                                                    "rv_monthly", "vix_close"), with = FALSE])])

cat_progress(sprintf("  GAM R²: %.4f (vs linear HAR: %.4f)",
                     summary(gam_rv)$r.sq, har_summary$r.squared))

# Effective degrees of freedom for each smooth
edf <- summary(gam_rv)$edf
cat_progress(sprintf("  GAM effective df: daily=%.1f, weekly=%.1f, monthly=%.1f, vix=%.1f",
                     edf[1], edf[2], edf[3], edf[4]))

#------------------------------------------------------------------
# C.4 FEATURE IMPORTANCE PREVIEW
#------------------------------------------------------------------

cat_progress("C.4: Feature Importance Preview (Linear Analysis)")

# -----------------------------------------------------------------------------
# C.4.1 Correlation matrix for potential features
# -----------------------------------------------------------------------------

# Construct potential feature set
feature_cols <- c(
  # HAR components
  "rv_daily", "rv_weekly", "rv_monthly",
  # VIX-based
  "vix_close", "vrp_cc",
  # Return-based
  "log_return", "abs_ret",
  # Signed RV
  "rv_neg_daily",
  # Regime
  "p_high"
)

feature_cols_available <- feature_cols[feature_cols %in% names(dt)]

# Correlation with target
target_correlations <- sapply(feature_cols_available, function(x) {
  cor(dt$rv_next, dt[[x]], use = "complete.obs")
})

feature_importance_linear <- data.table(
  feature = names(target_correlations),
  correlation = target_correlations,
  abs_correlation = abs(target_correlations)
)[order(-abs_correlation)]

print(feature_importance_linear)
write.csv(feature_importance_linear, file.path(dirs$tables, "C4_feature_importance_linear.csv"), 
          row.names = FALSE)

# -----------------------------------------------------------------------------
# C.4.2 Feature correlation matrix
# -----------------------------------------------------------------------------

feature_cor_matrix <- cor(dt[, .SD, .SDcols = feature_cols_available], use = "pairwise.complete.obs")

pdf(file.path(dirs$figures, "37_feature_correlations.pdf"), width = 10, height = 10)
corrplot::corrplot(feature_cor_matrix, method = "color", type = "upper",
                   addCoef.col = "black", number.cex = 0.7,
                   tl.col = "black", tl.srt = 45,
                   title = "Feature Correlation Matrix", mar = c(0, 0, 2, 0))
dev.off()

# -----------------------------------------------------------------------------
# C.4.3 Incremental R² analysis
# -----------------------------------------------------------------------------

cat_progress("  Incremental R² analysis...")

# Start with best single predictor and add sequentially
models_incremental <- list()
models_incremental[[1]] <- lm(rv_next ~ rv_daily, data = dt)
models_incremental[[2]] <- lm(rv_next ~ rv_daily + rv_weekly, data = dt)
models_incremental[[3]] <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly, data = dt)
models_incremental[[4]] <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly + vix_close, data = dt)
models_incremental[[5]] <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly + vix_close + vrp_cc, data = dt)
models_incremental[[6]] <- lm(rv_next ~ rv_daily + rv_weekly + rv_monthly + vix_close + vrp_cc + rv_neg_daily, data = dt)

incremental_r2 <- data.table(
  model = c("RV_daily", "+ RV_weekly", "+ RV_monthly", "+ VIX", "+ VRP", "+ RV_neg"),
  r_squared = sapply(models_incremental, function(m) summary(m)$r.squared),
  adj_r_squared = sapply(models_incremental, function(m) summary(m)$adj.r.squared)
)
incremental_r2[, delta_r2 := c(r_squared[1], diff(r_squared))]

print(incremental_r2)
write.csv(incremental_r2, file.path(dirs$tables, "C4_incremental_r2.csv"), row.names = FALSE)

#------------------------------------------------------------------
#
#            PART D: XGBOOST TARGET CONSTRUCTION AND PREPARATION
#
#------------------------------------------------------------------

cat_progress("=")
cat_progress("PART D: XGBOOST TARGET CONSTRUCTION AND PREPARATION")
cat_progress("=")

#------------------------------------------------------------------
# D.1 TARGET VARIABLE CONSTRUCTION
#------------------------------------------------------------------

cat_progress("D.1: Target Variable Construction")

# -----------------------------------------------------------------------------
# D.1.1 Regression target: next-day RV
# -----------------------------------------------------------------------------

# Already computed: dt$rv_next = shift(rv_daily, -1)
# Also consider log transform for better distributional properties
dt[, log_rv_next := log(rv_next)]

# Target summary
target_reg_summary <- dt[, .(
  target = c("rv_next", "log_rv_next"),
  mean = c(mean(rv_next, na.rm = TRUE), mean(log_rv_next, na.rm = TRUE)),
  sd = c(sd(rv_next, na.rm = TRUE), sd(log_rv_next, na.rm = TRUE)),
  skew = c(moments::skewness(rv_next, na.rm = TRUE), 
           moments::skewness(log_rv_next, na.rm = TRUE)),
  kurt = c(moments::kurtosis(rv_next, na.rm = TRUE),
           moments::kurtosis(log_rv_next, na.rm = TRUE))
)]

cat_progress("  Regression targets:")
print(target_reg_summary)

# -----------------------------------------------------------------------------
# D.1.2 Classification target: RV direction and magnitude
# -----------------------------------------------------------------------------

# Direction: up (1) vs down (0)
dt[, rv_direction := as.integer(rv_next > rv_daily)]

# Magnitude-based classes
dt[, rv_change_pct := (rv_next - rv_daily) / rv_daily * 100]

# Multiple threshold classifications
thresholds <- c(0, 2.5, 5, 10)  # Percentage change thresholds

for (thresh in thresholds) {
  col_name <- sprintf("rv_up_%s", gsub("\\.", "p", as.character(thresh)))
  dt[, (col_name) := as.integer(rv_change_pct > thresh)]
}

# Class balance analysis
class_balance <- data.table(
  threshold = thresholds,
  n_positive = sapply(thresholds, function(t) {
    sum(dt$rv_change_pct > t, na.rm = TRUE)
  }),
  n_negative = sapply(thresholds, function(t) {
    sum(dt$rv_change_pct <= t, na.rm = TRUE)
  })
)
class_balance[, pct_positive := n_positive / (n_positive + n_negative) * 100]
class_balance[, imbalance_ratio := n_negative / n_positive]

cat_progress("  Classification class balance:")
print(class_balance)
write.csv(class_balance, file.path(dirs$tables, "D1_class_balance.csv"), row.names = FALSE)

# Recommended: use threshold = 0 (direction) with class weights
cat_progress(sprintf("  Recommended scale_pos_weight for XGBoost: %.3f",
                     class_balance$imbalance_ratio[1]))

#------------------------------------------------------------------
# D.2 FEATURE ENGINEERING SUMMARY
#------------------------------------------------------------------

cat_progress("D.2: Feature Engineering Summary")

# Compile recommended features based on analysis
recommended_features <- data.table(
  feature = c(
    # Core HAR
    "rv_daily", "rv_weekly", "rv_monthly",
    # Log transforms
    "log_rv_daily", "log_rv_weekly", "log_rv_monthly",
    # VIX-based
    "vix_close", "vrp_cc",
    # Signed/asymmetric
    "rv_neg_daily", "rv_pos_22d", "rv_neg_22d",
    # Regime
    "p_high",
    # Lags
    "rv_daily_lag2", "rv_daily_lag5",
    # Jump proxy
    "jump_ratio",
    # Returns
    "log_return", "abs_ret"
  ),
  category = c(
    rep("HAR_core", 3),
    rep("HAR_log", 3),
    rep("VIX_based", 2),
    rep("Asymmetric", 3),
    "Regime",
    rep("Lags", 2),
    "Jump",
    rep("Returns", 2)
  ),
  rationale = c(
    "Core HAR component - daily persistence", 
    "Core HAR component - weekly aggregation",
    "Core HAR component - monthly aggregation",
    "Log transform - closer to Gaussian",
    "Log transform - closer to Gaussian",
    "Log transform - closer to Gaussian",
    "Risk-neutral volatility expectation",
    "Variance risk premium - return predictability",
    "Negative return contribution - leverage",
    "Signed RV - positive contribution",
    "Signed RV - negative contribution",
    "Regime probability - state dependence",
    "Additional lag - capture dynamics",
    "Additional lag - capture dynamics",
    "Jump detection ratio",
    "Return level - contemporaneous",
    "Absolute return - vol proxy"
  )
)

print(recommended_features)
write.csv(recommended_features, file.path(dirs$tables, "D2_recommended_features.csv"), row.names = FALSE)

# Create additional lag features
dt[, rv_daily_lag2 := shift(rv_daily, 2)]
dt[, rv_daily_lag5 := shift(rv_daily, 5)]
dt[, rv_daily_lag10 := shift(rv_daily, 10)]
dt[, vix_lag1 := shift(vix_close, 1)]
dt[, vrp_lag1 := shift(vrp_cc, 1)]

#------------------------------------------------------------------
# D.3 TRAIN/TEST SPLIT RECOMMENDATIONS
#------------------------------------------------------------------

cat_progress("D.3: Train/Test Split Analysis")

# Load existing split info if available
split_info <- tryCatch({
  readRDS(file.path(dirs$models, "split_info.rds"))
}, error = function(e) NULL)

if (!is.null(split_info)) {
  cat_progress(sprintf("  Existing split: Train %s to %s, Test %s to %s",
                       split_info$train_start, split_info$train_end,
                       split_info$test_start, split_info$test_end))
}

# Temporal cross-validation recommendation
n_obs <- nrow(dt)
cv_recommendation <- data.table(
  scheme = c("Single split (80/20)", "Expanding window", "Rolling window (5yr)"),
  description = c(
    sprintf("Train: first %d obs, Test: last %d obs", floor(0.8*n_obs), ceiling(0.2*n_obs)),
    "Sequential expansion with minimum 5yr training",
    "Fixed 5-year window, 1-year test, rolled forward"
  ),
  n_folds = c(1, floor((n_obs - 252*5) / 252), floor((n_obs - 252*6) / 252))
)

print(cv_recommendation)

#------------------------------------------------------------------
# D.4 SAVE ANALYSIS DATA
#------------------------------------------------------------------

cat_progress("D.4: Saving Analysis Data")

# Select final columns for XGBoost preparation
xgb_cols <- c(
  # Identifiers
  "date",
  # Targets
  "rv_next", "log_rv_next", "rv_direction", "rv_change_pct",
  # HAR features
  "rv_daily", "rv_weekly", "rv_monthly",
  "log_rv_daily", "log_rv_weekly", "log_rv_monthly",
  # VIX features
  "vix_close", "vrp_cc", "vix_lag1", "vrp_lag1",
  # Asymmetric features
  "rv_neg_daily", "rv_pos_22d", "rv_neg_22d",
  # Regime
  "p_high", "regime",
  # Lags
  "rv_daily_lag2", "rv_daily_lag5", "rv_daily_lag10",
  # Jump
  "jump_ratio", "jump_flag",
  # Returns
  "log_return", "abs_ret",
  # Forward RV (for reference)
  "rv_cc", "rv_parkinson", "rv_gk"
)

# Keep only columns that exist
xgb_cols_available <- xgb_cols[xgb_cols %in% names(dt)]

xgb_data <- dt[, .SD, .SDcols = xgb_cols_available]

# Save
saveRDS(xgb_data, file.path(dirs$data, "rv_analysis_for_xgboost.rds"))
cat_progress(sprintf("  Saved: %s (%d rows, %d columns)",
                     file.path(dirs$data, "rv_analysis_for_xgboost.rds"),
                     nrow(xgb_data), ncol(xgb_data)))

# Also save full analysis dataset
saveRDS(dt, file.path(dirs$data, "rv_full_analysis.rds"))

#------------------------------------------------------------------
# D.5 COMPREHENSIVE SUMMARY STATISTICS
#------------------------------------------------------------------

cat_progress("D.5: Generating Comprehensive Summary")

# Final summary table
final_summary <- list(
  # Sample info
  sample = list(
    n_observations = nrow(dt),
    start_date = min(dt$date),
    end_date = max(dt$date),
    years = as.numeric(difftime(max(dt$date), min(dt$date), units = "days")) / 365.25
  ),
  
  # RV characteristics
  rv_characteristics = list(
    long_memory_d = mean(c(gph_estimates$d_hat, lw_estimate), na.rm = TRUE),
    hurst_exponent = hurst_rv$H,
    roughness_H = mean(c(rough_q1$H, rough_q2$H), na.rm = TRUE),
    mean_rv_daily = mean(dt$rv_daily, na.rm = TRUE),
    persistence_halflife = halflife_rv
  ),
  
  # VRP characteristics
  vrp_characteristics = list(
    mean_vrp = vrp_stats$mean,
    pct_positive = vrp_stats$pct_positive,
    vrp_persistence = vrp_stats$ar1
  ),
  
  # Predictability
  predictability = list(
    har_r_squared = har_summary$r.squared,
    gam_r_squared = summary(gam_rv)$r.sq,
    best_single_predictor = feature_importance_linear$feature[1],
    best_correlation = feature_importance_linear$correlation[1]
  ),
  
  # Classification
  classification = list(
    class_balance = class_balance$pct_positive[1],
    scale_pos_weight = class_balance$imbalance_ratio[1]
  )
)

# Save summary
saveRDS(final_summary, file.path(dirs$models, "rv_analysis_summary.rds"))

# Print key findings
cat_progress("\n")
cat_progress("="
)
cat_progress("KEY FINDINGS SUMMARY")
cat_progress("=")
cat_progress(sprintf("Sample: %d observations (%.1f years)", 
                     final_summary$sample$n_observations, final_summary$sample$years))
cat_progress(sprintf("Long memory d: %.3f (stationary long memory confirmed)", 
                     final_summary$rv_characteristics$long_memory_d))
cat_progress(sprintf("Hurst exponent: %.3f (H > 0.5 indicates persistence)", 
                     final_summary$rv_characteristics$hurst_exponent))
cat_progress(sprintf("Roughness H: %.3f (H < 0.5 indicates rough volatility)", 
                     final_summary$rv_characteristics$roughness_H))
cat_progress(sprintf("VRP mean: %.2f%% (positive %% = %.1f%%)", 
                     final_summary$vrp_characteristics$mean_vrp,
                     final_summary$vrp_characteristics$pct_positive))
cat_progress(sprintf("HAR-RV R²: %.4f", final_summary$predictability$har_r_squared))
cat_progress(sprintf("GAM R² (nonlinear): %.4f", final_summary$predictability$gam_r_squared))
cat_progress(sprintf("Best predictor: %s (cor = %.3f)", 
                     final_summary$predictability$best_single_predictor,
                     final_summary$predictability$best_correlation))
cat_progress(sprintf("Classification balance: %.1f%% positive (scale_pos_weight = %.2f)",
                     final_summary$classification$class_balance,
                     final_summary$classification$scale_pos_weight))

#------------------------------------------------------------------
# CLEANUP
#------------------------------------------------------------------

gc()

cat_progress("\nRV Comprehensive Analysis Complete.")
cat_progress("Output files:")
cat_progress(sprintf("  Data: %s", file.path(dirs$data, "rv_analysis_for_xgboost.rds")))
cat_progress(sprintf("  Data: %s", file.path(dirs$data, "rv_full_analysis.rds")))
cat_progress(sprintf("  Summary: %s", file.path(dirs$models, "rv_analysis_summary.rds")))
cat_progress("  Tables: A1-A6, B1-B3, C3-C4, D1-D2 in results/tables/")
cat_progress("  Figures: A1-A6, B2, C4 in results/figures/")

#------------------------------------------------------------------
# END OF SCRIPT
#------------------------------------------------------------------