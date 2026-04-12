#------------------------------------------------------------------
# VRP_Diagnostics.R
# Comprehensive Variance Risk Premium Analysis
# VRP defined as: VIX^2 - RV^2 (variance terms) and VIX - RV (volatility terms)
#------------------------------------------------------------------

source("Setup.R")

cat_progress("Loading data with RV...")
dt <- readRDS(file.path(dirs$data, "data_with_rv.rds"))

#------------------------------------------------------------------
# 1. VRP CONSTRUCTION AND DEFINITIONS
#------------------------------------------------------------------

cat_progress("Constructing VRP measures...")

# Primary VRP definitions (variance space, scaled for interpretability)
# VIX is quoted in annualised vol %, so VIX^2/100 gives annualised variance in %
dt[, `:=`(
  # Variance space: (VIX^2 - RV^2) / 100
  vrp_var_cc = (vix_close^2 - rv_cc^2) / 100,
  vrp_var_pk = (vix_close^2 - rv_parkinson^2) / 100,
  vrp_var_gk = (vix_close^2 - rv_gk^2) / 100,
  vrp_var_rs = (vix_close^2 - rv_rs^2) / 100,
  vrp_var_yz = (vix_close^2 - rv_yz^2) / 100,
  
  # Log variance ratio: log(VIX^2 / RV^2) - stationarity inducing
  vrp_log_cc = log(vix_close^2 / rv_cc^2),
  vrp_log_pk = log(vix_close^2 / rv_parkinson^2),
  
  # Squared VIX and RV for decomposition
  vix_var = vix_close^2 / 100,
  rv_var_cc = rv_cc^2 / 100
)]

# Remove observations with missing VRP
dt_vrp <- dt[!is.na(vrp_var_cc) & !is.na(vrp_cc)]
n_obs <- nrow(dt_vrp)
cat_progress(sprintf("VRP sample: %d observations (%s to %s)", 
                     n_obs, min(dt_vrp$date), max(dt_vrp$date)))

#------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
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
    Q05 = quantile(x, 0.05),
    Q25 = quantile(x, 0.25),
    Median = median(x),
    Q75 = quantile(x, 0.75),
    Q95 = quantile(x, 0.95),
    Max = max(x),
    Pct_Positive = 100 * mean(x > 0),
    IQR = IQR(x)
  )
}

# VRP in volatility terms
vrp_vol_stats <- rbindlist(list(
  compute_moments(dt_vrp$vrp_cc, "VRP (CC)"),
  compute_moments(dt_vrp$vrp_parkinson, "VRP (Parkinson)"),
  compute_moments(dt_vrp$vrp_gk, "VRP (GK)"),
  compute_moments(dt_vrp$vrp_rs, "VRP (RS)"),
  compute_moments(dt_vrp$vrp_yz, "VRP (YZ)")
))

# VRP in variance terms
vrp_var_stats <- rbindlist(list(
  compute_moments(dt_vrp$vrp_var_cc, "VRP_Var (CC)"),
  compute_moments(dt_vrp$vrp_var_pk, "VRP_Var (Parkinson)"),
  compute_moments(dt_vrp$vrp_var_gk, "VRP_Var (GK)"),
  compute_moments(dt_vrp$vrp_var_rs, "VRP_Var (RS)"),
  compute_moments(dt_vrp$vrp_var_yz, "VRP_Var (YZ)")
))

cat("\n=== VRP Descriptive Statistics (Volatility Terms) ===\n")
print(vrp_vol_stats[, .(Variable, N, Mean, SD, Skewness, Kurtosis, Pct_Positive)])

cat("\n=== VRP Descriptive Statistics (Variance Terms) ===\n")
print(vrp_var_stats[, .(Variable, N, Mean, SD, Skewness, Kurtosis, Pct_Positive)])

#------------------------------------------------------------------
# 3. STATISTICAL INFERENCE: IS MEAN VRP SIGNIFICANTLY POSITIVE?
#------------------------------------------------------------------

cat_progress("Testing significance of mean VRP with HAC standard errors...")

# Newey-West HAC inference (robust to serial correlation)
# Bandwidth selection: Andrews (1991) automatic or fixed at T^(1/3)
nw_bandwidth <- floor(n_obs^(1/3))

vrp_inference <- function(vrp, name, bandwidth) {
  vrp <- vrp[!is.na(vrp)]
  n <- length(vrp)
  
  # OLS regression: vrp_t = mu + epsilon_t
  fit <- lm(vrp ~ 1)
  
  # HAC covariance (Newey-West)
  nw_vcov <- sandwich::NeweyWest(fit, lag = bandwidth, prewhite = FALSE)
  
  # t-statistic with HAC SE
  mu_hat <- coef(fit)
  se_hac <- sqrt(diag(nw_vcov))
  t_hac <- mu_hat / se_hac
  p_hac <- 2 * pt(-abs(t_hac), df = n - 1)
  
  # Standard OLS SE for comparison
  se_ols <- summary(fit)$coefficients[1, 2]
  t_ols <- mu_hat / se_ols
  
  data.table(
    Variable = name,
    Mean = mu_hat,
    SE_OLS = se_ols,
    SE_HAC = se_hac,
    t_OLS = t_ols,
    t_HAC = t_hac,
    p_HAC = p_hac,
    CI_Lower = mu_hat - 1.96 * se_hac,
    CI_Upper = mu_hat + 1.96 * se_hac
  )
}

vrp_test_results <- rbindlist(list(
  vrp_inference(dt_vrp$vrp_cc, "VRP_vol (CC)", nw_bandwidth),
  vrp_inference(dt_vrp$vrp_var_cc, "VRP_var (CC)", nw_bandwidth),
  vrp_inference(dt_vrp$vrp_log_cc, "VRP_log (CC)", nw_bandwidth)
))

cat("\n=== Mean VRP Significance Tests (Newey-West HAC) ===\n")
cat(sprintf("Bandwidth: %d lags\n\n", nw_bandwidth))
print(vrp_test_results)

#------------------------------------------------------------------
# 4. DISTRIBUTIONAL TESTS
#------------------------------------------------------------------

cat_progress("Testing distributional properties...")

vrp_cc <- dt_vrp$vrp_cc[!is.na(dt_vrp$vrp_cc)]
vrp_var <- dt_vrp$vrp_var_cc[!is.na(dt_vrp$vrp_var_cc)]

# Jarque-Bera test for normality
jb_vol <- tseries::jarque.bera.test(vrp_cc)
jb_var <- tseries::jarque.bera.test(vrp_var)

# Shapiro-Wilk (on subsample due to n limit)
sw_vol <- shapiro.test(sample(vrp_cc, min(5000, length(vrp_cc))))
sw_var <- shapiro.test(sample(vrp_var, min(5000, length(vrp_var))))

cat("\n=== Normality Tests ===\n")
cat(sprintf("VRP (vol): JB stat = %.2f, p = %.4e\n", jb_vol$statistic, jb_vol$p.value))
cat(sprintf("VRP (var): JB stat = %.2f, p = %.4e\n", jb_var$statistic, jb_var$p.value))
cat(sprintf("VRP (vol): SW stat = %.4f, p = %.4e (subsample)\n", sw_vol$statistic, sw_vol$p.value))

# Heavy tails: Hill estimator for tail index
# Right tail
compute_hill <- function(x, k = NULL) {
  x <- x[x > 0]
  x <- sort(x, decreasing = TRUE)
  n <- length(x)
  if (is.null(k)) k <- floor(sqrt(n))
  k <- min(k, n - 1)
  
  hill <- (1/k) * sum(log(x[1:k]) - log(x[k + 1]))
  alpha <- 1 / hill
  se <- alpha / sqrt(k)
  
  list(alpha = alpha, se = se, k = k)
}

hill_right <- compute_hill(vrp_var[vrp_var > 0])
hill_left <- compute_hill(-vrp_var[vrp_var < 0])

cat("\n=== Tail Index (Hill Estimator) ===\n")
cat(sprintf("Right tail: alpha = %.2f (SE = %.2f), k = %d\n", 
            hill_right$alpha, hill_right$se, hill_right$k))
cat(sprintf("Left tail:  alpha = %.2f (SE = %.2f), k = %d\n", 
            hill_left$alpha, hill_left$se, hill_left$k))

#------------------------------------------------------------------
# 5. STATIONARITY AND UNIT ROOT TESTS
#------------------------------------------------------------------

cat_progress("Testing stationarity...")

# ADF test with automatic lag selection (AIC)
adf_vrp_vol <- tseries::adf.test(vrp_cc, alternative = "stationary")
adf_vrp_var <- tseries::adf.test(vrp_var, alternative = "stationary")

# KPSS test (null: stationarity)
kpss_vrp_vol <- tseries::kpss.test(vrp_cc, null = "Level")
kpss_vrp_var <- tseries::kpss.test(vrp_var, null = "Level")

# Phillips-Perron test
pp_vrp_vol <- tseries::pp.test(vrp_cc, alternative = "stationary")
pp_vrp_var <- tseries::pp.test(vrp_var, alternative = "stationary")

cat("\n=== Unit Root / Stationarity Tests ===\n")
cat("H0 for ADF/PP: Unit root; H0 for KPSS: Stationarity\n\n")

stationarity_results <- data.table(
  Series = c("VRP (vol)", "VRP (var)"),
  ADF_stat = c(adf_vrp_vol$statistic, adf_vrp_var$statistic),
  ADF_p = c(adf_vrp_vol$p.value, adf_vrp_var$p.value),
  PP_stat = c(pp_vrp_vol$statistic, pp_vrp_var$statistic),
  PP_p = c(pp_vrp_vol$p.value, pp_vrp_var$p.value),
  KPSS_stat = c(kpss_vrp_vol$statistic, kpss_vrp_var$statistic),
  KPSS_p = c(kpss_vrp_vol$p.value, kpss_vrp_var$p.value)
)
print(stationarity_results)

#------------------------------------------------------------------
# 6. AUTOCORRELATION STRUCTURE
#------------------------------------------------------------------

cat_progress("Analysing autocorrelation structure...")

max_lag <- 60

# ACF and PACF
acf_vrp <- acf(vrp_cc, lag.max = max_lag, plot = FALSE)
pacf_vrp <- pacf(vrp_cc, lag.max = max_lag, plot = FALSE)
acf_vrp_var <- acf(vrp_var, lag.max = max_lag, plot = FALSE)

# Ljung-Box test at various lags
lb_results <- data.table(
  Lag = c(5, 10, 22, 44, 66),
  Q_stat = sapply(c(5, 10, 22, 44, 66), function(k) {
    Box.test(vrp_cc, lag = k, type = "Ljung-Box")$statistic
  }),
  p_value = sapply(c(5, 10, 22, 44, 66), function(k) {
    Box.test(vrp_cc, lag = k, type = "Ljung-Box")$p.value
  })
)

cat("\n=== Ljung-Box Tests for Serial Correlation ===\n")
print(lb_results)

# Half-life of autocorrelation (AR(1) approximation)
ar1_fit <- ar(vrp_cc, order.max = 1, aic = FALSE, method = "ols")
phi1 <- ar1_fit$ar[1]
half_life <- -log(2) / log(abs(phi1))

cat(sprintf("\nAR(1) coefficient: %.4f\n", phi1))
cat(sprintf("Implied half-life: %.1f days\n", half_life))

# ARCH effects test
arch_test <- FinTS::ArchTest(vrp_cc, lags = 10)
cat(sprintf("\nARCH-LM test (10 lags): stat = %.2f, p = %.4e\n", 
            arch_test$statistic, arch_test$p.value))

#------------------------------------------------------------------
# 7. PERSISTENCE ANALYSIS: FRACTIONAL INTEGRATION
#------------------------------------------------------------------

cat_progress("Estimating persistence (fractional integration)...")

# GPH estimator for d (log-periodogram regression)
estimate_d_gph <- function(x, m = NULL) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (is.null(m)) m <- floor(n^0.65)  # Geweke-Porter-Hudak bandwidth
  
  # Periodogram
  spec <- spectrum(x, plot = FALSE, taper = 0)
  freq <- spec$freq[1:m]
  pgram <- spec$spec[1:m]
  
  # Log-periodogram regression
  y <- log(pgram)
  X <- log(4 * sin(pi * freq)^2)
  
  fit <- lm(y ~ X)
  d_hat <- -coef(fit)[2] / 2
  se_d <- summary(fit)$coefficients[2, 2] / 2
  
  list(d = d_hat, se = se_d, m = m)
}

d_est <- estimate_d_gph(vrp_cc)
cat(sprintf("\nFractional integration parameter (GPH): d = %.3f (SE = %.3f)\n", 
            d_est$d, d_est$se))
cat(sprintf("Bandwidth m = %d\n", d_est$m))

if (d_est$d < 0.5) {
  cat("Interpretation: VRP is I(0) - stationary with possible long memory\n")
} else if (d_est$d < 1) {
  cat("Interpretation: VRP is I(d) with 0.5 < d < 1 - mean reverting but persistent\n")
}

#------------------------------------------------------------------
# 8. REGIME ANALYSIS
#------------------------------------------------------------------

cat_progress("Analysing VRP regimes...")

# Define regimes based on VRP level
dt_vrp[, vrp_regime := cut(vrp_cc, 
                           breaks = quantile(vrp_cc, c(0, 0.1, 0.25, 0.75, 0.9, 1), na.rm = TRUE),
                           labels = c("Very Low", "Low", "Normal", "High", "Very High"),
                           include.lowest = TRUE)]

# Also define by sign
dt_vrp[, vrp_sign := fifelse(vrp_cc > 0, "Positive", "Negative")]

# Summary by regime
regime_stats <- dt_vrp[, .(
  N = .N,
  Pct = 100 * .N / n_obs,
  Mean_VRP_vol = mean(vrp_cc, na.rm = TRUE),
  Mean_VRP_var = mean(vrp_var_cc, na.rm = TRUE),
  Mean_VIX = mean(vix_close, na.rm = TRUE),
  Mean_RV = mean(rv_cc, na.rm = TRUE),
  Mean_Return = mean(log_return, na.rm = TRUE) * 252  # Annualised
), by = vrp_regime]

cat("\n=== VRP Regime Analysis ===\n")
print(regime_stats[order(vrp_regime)])

# Sign analysis
sign_stats <- dt_vrp[, .(
  N = .N,
  Pct = 100 * .N / n_obs,
  Mean_VRP = mean(vrp_cc, na.rm = TRUE),
  Mean_FwdReturn = mean(shift(log_return, -22, type = "lead"), na.rm = TRUE) * 12  # 22-day forward, annualised
), by = vrp_sign]

cat("\n=== VRP Sign Analysis ===\n")
print(sign_stats)

# Transition matrix (Markov chain)
dt_vrp[, vrp_regime_lag := shift(vrp_regime, 1, type = "lag")]
transition_matrix <- dt_vrp[!is.na(vrp_regime) & !is.na(vrp_regime_lag), 
                            .N, by = .(vrp_regime_lag, vrp_regime)]
transition_matrix <- dcast(transition_matrix, vrp_regime_lag ~ vrp_regime, value.var = "N", fill = 0)

# Normalise to probabilities
trans_probs <- as.matrix(transition_matrix[, -1])
rownames(trans_probs) <- transition_matrix$vrp_regime_lag
trans_probs <- trans_probs / rowSums(trans_probs)

cat("\n=== Regime Transition Probabilities ===\n")
print(round(trans_probs, 3))

#------------------------------------------------------------------
# 9. ASYMMETRY ANALYSIS
#------------------------------------------------------------------

cat_progress("Testing VRP asymmetry...")

# VRP conditional on market returns
dt_vrp[, return_sign := fifelse(log_return >= 0, "Up", "Down")]
dt_vrp[, return_quintile := cut(log_return, 
                                breaks = quantile(log_return, seq(0, 1, 0.2), na.rm = TRUE),
                                labels = c("Q1 (Worst)", "Q2", "Q3", "Q4", "Q5 (Best)"),
                                include.lowest = TRUE)]

asymm_by_return <- dt_vrp[!is.na(return_quintile), .(
  Mean_VRP = mean(vrp_cc, na.rm = TRUE),
  SD_VRP = sd(vrp_cc, na.rm = TRUE),
  Mean_VIX = mean(vix_close, na.rm = TRUE),
  Mean_RV = mean(rv_cc, na.rm = TRUE),
  N = .N
), by = return_quintile]

cat("\n=== VRP by Return Quintile ===\n")
print(asymm_by_return[order(return_quintile)])

# Test asymmetry formally: VRP_t = a + b*I(r<0)*|r| + c*I(r>=0)*r + e
dt_vrp[, `:=`(
  neg_return = pmin(log_return, 0),
  pos_return = pmax(log_return, 0),
  abs_return = abs(log_return)
)]

asymm_reg <- lm(vrp_cc ~ neg_return + pos_return, data = dt_vrp)
asymm_vcov <- sandwich::NeweyWest(asymm_reg, lag = nw_bandwidth)
asymm_test <- lmtest::coeftest(asymm_reg, vcov = asymm_vcov)

cat("\n=== VRP Asymmetry Regression (HAC SE) ===\n")
cat("VRP_t = a + b*r^- + c*r^+ + e\n\n")
print(asymm_test)

# Wald test: H0: b = c (symmetric response)
wald_asymm <- car::linearHypothesis(asymm_reg, "neg_return = pos_return", vcov = asymm_vcov)
cat(sprintf("\nWald test for asymmetry (H0: b = c): F = %.2f, p = %.4f\n",
            wald_asymm$F[2], wald_asymm$`Pr(>F)`[2]))

#------------------------------------------------------------------
# 10. PREDICTIVE REGRESSIONS
#------------------------------------------------------------------

cat_progress("Running predictive regressions...")

# Create forward returns at various horizons
horizons <- c(1, 5, 22, 66)  # 1d, 1w, 1m, 3m
for (h in horizons) {
  dt_vrp[, paste0("fwd_ret_", h) := shift(log_return, -h, type = "lead")]
  # Cumulative returns
  if (h > 1) {
    dt_vrp[, paste0("fwd_cumret_", h) := frollsum(shift(log_return, -1, type = "lead"), n = h, align = "left")]
  }
}

# Forward RV
dt_vrp[, fwd_rv_22 := shift(rv_cc, -22, type = "lead")]

# Standardise VRP for comparability
dt_vrp[, vrp_std := (vrp_cc - mean(vrp_cc, na.rm = TRUE)) / sd(vrp_cc, na.rm = TRUE)]
dt_vrp[, vrp_var_std := (vrp_var_cc - mean(vrp_var_cc, na.rm = TRUE)) / sd(vrp_var_cc, na.rm = TRUE)]

# Predictive regression function with HAC and Hodrick (1992) correction
run_predictive_reg <- function(data, y_var, x_var, horizon, nw_lag = NULL) {
  if (is.null(nw_lag)) nw_lag <- horizon + 5
  
  formula <- as.formula(paste(y_var, "~", x_var))
  fit <- lm(formula, data = data)
  
  # Newey-West with appropriate lag
  hac_vcov <- sandwich::NeweyWest(fit, lag = nw_lag, prewhite = FALSE)
  coef_test <- lmtest::coeftest(fit, vcov = hac_vcov)
  
  # R-squared
  r2 <- summary(fit)$r.squared
  adj_r2 <- summary(fit)$adj.r.squared
  
  list(
    coef = coef(fit)[2],
    se_hac = sqrt(diag(hac_vcov))[2],
    t_hac = coef_test[2, 3],
    p_hac = coef_test[2, 4],
    r2 = r2,
    n = nobs(fit)
  )
}

# VRP predicting returns
cat("\n=== VRP Predicting Future Returns (HAC SE) ===\n")
pred_ret_results <- lapply(c(1, 5, 22, 66), function(h) {
  y_var <- ifelse(h == 1, "fwd_ret_1", paste0("fwd_cumret_", h))
  res <- run_predictive_reg(dt_vrp, y_var, "vrp_std", h)
  data.table(Horizon = h, Beta = res$coef, SE = res$se_hac, 
             t = res$t_hac, p = res$p_hac, R2 = res$r2 * 100)
})
pred_ret_dt <- rbindlist(pred_ret_results)
print(pred_ret_dt)

# VRP predicting future RV
cat("\n=== VRP Predicting Future RV ===\n")
pred_rv <- run_predictive_reg(dt_vrp, "fwd_rv_22", "vrp_std", 22)
cat(sprintf("Beta = %.4f, t = %.2f, p = %.4f, R2 = %.2f%%\n",
            pred_rv$coef, pred_rv$t_hac, pred_rv$p_hac, pred_rv$r2 * 100))

# VRP predicting VIX changes
dt_vrp[, fwd_vix_change := shift(vix_close, -22, type = "lead") - vix_close]
pred_vix <- run_predictive_reg(dt_vrp, "fwd_vix_change", "vrp_std", 22)
cat(sprintf("VRP -> VIX change: Beta = %.4f, t = %.2f, p = %.4f\n",
            pred_vix$coef, pred_vix$t_hac, pred_vix$p_hac))

#------------------------------------------------------------------
# 11. SUBSAMPLE STABILITY
#------------------------------------------------------------------

cat_progress("Testing subsample stability...")

# Define subsamples
subsamples <- list(
  "1990-1999" = dt_vrp[date >= "1990-01-01" & date < "2000-01-01"],
  "2000-2007" = dt_vrp[date >= "2000-01-01" & date < "2008-01-01"],
  "2008-2009 (GFC)" = dt_vrp[date >= "2008-01-01" & date < "2010-01-01"],
  "2010-2019" = dt_vrp[date >= "2010-01-01" & date < "2020-01-01"],
  "2020-Present" = dt_vrp[date >= "2020-01-01"]
)

subsample_stats <- rbindlist(lapply(names(subsamples), function(name) {
  d <- subsamples[[name]]
  if (nrow(d) < 50) return(NULL)
  data.table(
    Period = name,
    N = nrow(d),
    Mean_VRP_vol = mean(d$vrp_cc, na.rm = TRUE),
    SD_VRP_vol = sd(d$vrp_cc, na.rm = TRUE),
    Mean_VRP_var = mean(d$vrp_var_cc, na.rm = TRUE),
    Pct_Positive = 100 * mean(d$vrp_cc > 0, na.rm = TRUE),
    Mean_VIX = mean(d$vix_close, na.rm = TRUE),
    Corr_VIX_RV = cor(d$vix_close, d$rv_cc, use = "complete.obs")
  )
}))

cat("\n=== Subsample Statistics ===\n")
print(subsample_stats)

# Structural break test (Chow test approximation via rolling window)
roll_mean_vrp <- frollmean(dt_vrp$vrp_cc, n = 252, align = "right")
roll_sd_vrp <- frollapply(dt_vrp$vrp_cc, n = 252, FUN = sd, align = "right")

#------------------------------------------------------------------
# 12. CORRELATION WITH OTHER VARIABLES
#------------------------------------------------------------------

cat_progress("Computing correlations with market variables...")

# Lagged correlations
lag_range <- -22:22

compute_xcorr <- function(x, y, lags) {
  sapply(lags, function(k) {
    if (k >= 0) {
      cor(x, shift(y, k, type = "lag"), use = "complete.obs")
    } else {
      cor(x, shift(y, -k, type = "lead"), use = "complete.obs")
    }
  })
}

xcorr_ret <- compute_xcorr(dt_vrp$vrp_cc, dt_vrp$log_return, lag_range)
xcorr_vix <- compute_xcorr(dt_vrp$vrp_cc, dt_vrp$vix_close, lag_range)
xcorr_rv <- compute_xcorr(dt_vrp$vrp_cc, dt_vrp$rv_cc, lag_range)

xcorr_dt <- data.table(
  lag = lag_range,
  xcorr_return = xcorr_ret,
  xcorr_vix = xcorr_vix,
  xcorr_rv = xcorr_rv
)

# Contemporaneous correlations
cat("\n=== Contemporaneous Correlations ===\n")
cor_vars <- c("vrp_cc", "vrp_var_cc", "vix_close", "rv_cc", "log_return", "abs_return")
cor_matrix <- cor(dt_vrp[, .SD, .SDcols = cor_vars], use = "pairwise.complete.obs")
print(round(cor_matrix, 3))

#------------------------------------------------------------------
# 13. VRP DECOMPOSITION
#------------------------------------------------------------------

cat_progress("Decomposing VRP variance...")

# VRP = VIX - RV, so Var(VRP) = Var(VIX) + Var(RV) - 2*Cov(VIX, RV)
var_vix <- var(dt_vrp$vix_close, na.rm = TRUE)
var_rv <- var(dt_vrp$rv_cc, na.rm = TRUE)
cov_vix_rv <- cov(dt_vrp$vix_close, dt_vrp$rv_cc, use = "complete.obs")
var_vrp <- var(dt_vrp$vrp_cc, na.rm = TRUE)

cat("\n=== VRP Variance Decomposition ===\n")
cat(sprintf("Var(VIX) = %.2f\n", var_vix))
cat(sprintf("Var(RV)  = %.2f\n", var_rv))
cat(sprintf("Cov(VIX,RV) = %.2f\n", cov_vix_rv))
cat(sprintf("Cor(VIX,RV) = %.3f\n", cov_vix_rv / sqrt(var_vix * var_rv)))
cat(sprintf("\nVar(VRP) = Var(VIX) + Var(RV) - 2*Cov = %.2f\n", var_vrp))
cat(sprintf("Check: %.2f + %.2f - 2*%.2f = %.2f\n", 
            var_vix, var_rv, cov_vix_rv, var_vix + var_rv - 2*cov_vix_rv))

# Contribution percentages (for intuition, not exact for variance space VRP)
cat(sprintf("\nVariance contributions (approx):\n"))
cat(sprintf("  From VIX variance: %.1f%%\n", 100 * var_vix / (var_vix + var_rv)))
cat(sprintf("  From RV variance:  %.1f%%\n", 100 * var_rv / (var_vix + var_rv)))
cat(sprintf("  Correlation effect: reduces by %.1f%%\n", 
            100 * 2 * cov_vix_rv / (var_vix + var_rv)))

#------------------------------------------------------------------
# 14. ROLLING STATISTICS
#------------------------------------------------------------------

cat_progress("Computing rolling statistics...")

roll_windows <- c(63, 126, 252)  # 3m, 6m, 1y

for (w in roll_windows) {
  dt_vrp[, paste0("vrp_roll_mean_", w) := frollmean(vrp_cc, n = w, align = "right")]
  dt_vrp[, paste0("vrp_roll_sd_", w) := frollapply(vrp_cc, n = w, FUN = sd, align = "right")]
  dt_vrp[, paste0("vrp_roll_sharpe_", w) := get(paste0("vrp_roll_mean_", w)) / 
           get(paste0("vrp_roll_sd_", w)) * sqrt(252)]
}

# Z-score (how many SDs from rolling mean)
dt_vrp[, vrp_zscore := (vrp_cc - vrp_roll_mean_252) / vrp_roll_sd_252]

#------------------------------------------------------------------
# 15. DIAGNOSTIC PLOTS
#------------------------------------------------------------------

cat_progress("Creating diagnostic plots...")

theme_diag <- theme_minimal() +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "bottom"
  )

# Plot 1: VRP time series with regimes
p1 <- ggplot(dt_vrp, aes(x = date, y = vrp_cc)) +
  geom_line(linewidth = 0.3, colour = "grey30") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  geom_hline(yintercept = mean(dt_vrp$vrp_cc, na.rm = TRUE), 
             linetype = "dashed", colour = "blue") +
  labs(title = "Variance Risk Premium (VIX - RV)", 
       x = NULL, y = "VRP (vol points)") +
  theme_diag

# Plot 2: VRP distribution
p2 <- ggplot(dt_vrp, aes(x = vrp_cc)) +
  geom_histogram(aes(y = after_stat(density)), bins = 100, 
                 fill = "steelblue", alpha = 0.7) +
  geom_density(colour = "darkred", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = mean(dt_vrp$vrp_cc, na.rm = TRUE), 
             colour = "blue", linetype = "dashed") +
  labs(title = "VRP Distribution", x = "VRP (vol points)", y = "Density") +
  theme_diag

# Plot 3: ACF plot
acf_df <- data.table(
  lag = 1:max_lag,
  acf = acf_vrp$acf[2:(max_lag + 1)]
)
ci <- 1.96 / sqrt(n_obs)

p3 <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_hline(yintercept = c(-ci, ci), linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0) +
  geom_segment(aes(xend = lag, yend = 0), colour = "steelblue") +
  geom_point(colour = "steelblue", size = 1.5) +
  labs(title = "Autocorrelation Function of VRP", x = "Lag (days)", y = "ACF") +
  theme_diag

# Plot 4: PACF plot
pacf_df <- data.table(
  lag = 1:max_lag,
  pacf = pacf_vrp$acf[1:max_lag]
)

p4 <- ggplot(pacf_df, aes(x = lag, y = pacf)) +
  geom_hline(yintercept = c(-ci, ci), linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0) +
  geom_segment(aes(xend = lag, yend = 0), colour = "darkred") +
  geom_point(colour = "darkred", size = 1.5) +
  labs(title = "Partial ACF of VRP", x = "Lag (days)", y = "PACF") +
  theme_diag

# Plot 5: VRP vs VIX scatter
p5 <- ggplot(dt_vrp, aes(x = vix_close, y = vrp_cc)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "loess", colour = "red", linewidth = 0.8, se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "VRP vs VIX Level", x = "VIX", y = "VRP") +
  theme_diag

# Plot 6: VRP variance space time series
p6 <- ggplot(dt_vrp, aes(x = date, y = vrp_var_cc)) +
  geom_line(linewidth = 0.3, colour = "grey30") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  labs(title = expression(paste("VRP in Variance Space: ", VIX^2 - RV^2)), 
       x = NULL, y = "VRP (variance points)") +
  theme_diag

# Plot 7: Rolling mean VRP
roll_plot_data <- melt(
  dt_vrp[, .(date, `3M` = vrp_roll_mean_63, `6M` = vrp_roll_mean_126, `1Y` = vrp_roll_mean_252)],
  id.vars = "date", variable.name = "Window", value.name = "Mean_VRP"
)

p7 <- ggplot(roll_plot_data[!is.na(Mean_VRP)], aes(x = date, y = Mean_VRP, colour = Window)) +
  geom_line(linewidth = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_colour_viridis_d(option = "C", end = 0.8) +
  labs(title = "Rolling Mean VRP", x = NULL, y = "Mean VRP (vol points)") +
  theme_diag

# Plot 8: Cross-correlation with returns
p8 <- ggplot(xcorr_dt, aes(x = lag, y = xcorr_return)) +
  geom_hline(yintercept = c(-ci, ci), linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0) +
  geom_segment(aes(xend = lag, yend = 0), colour = "steelblue") +
  geom_point(colour = "steelblue", size = 1.5) +
  labs(title = "Cross-correlation: VRP vs Returns", 
       subtitle = "Negative lag = VRP leads returns",
       x = "Lag (days)", y = "Cross-correlation") +
  theme_diag

# Plot 9: VRP by VIX regime
dt_vrp[, vix_regime := cut(vix_close, 
                           breaks = c(0, 15, 20, 25, 35, 100),
                           labels = c("<15", "15-20", "20-25", "25-35", ">35"))]

p9 <- ggplot(dt_vrp[!is.na(vix_regime)], aes(x = vix_regime, y = vrp_cc, fill = vix_regime)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  scale_fill_viridis_d(option = "D") +
  labs(title = "VRP Distribution by VIX Regime", x = "VIX Level", y = "VRP") +
  theme_diag +
  theme(legend.position = "none")

# Plot 10: QQ plot
p10 <- ggplot(dt_vrp, aes(sample = vrp_cc)) +
  stat_qq(alpha = 0.3, size = 0.5) +
  stat_qq_line(colour = "red") +
  labs(title = "Q-Q Plot: VRP vs Normal", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_diag

# Plot 11: VRP Z-score
p11 <- ggplot(dt_vrp[!is.na(vrp_zscore)], aes(x = date, y = vrp_zscore)) +
  geom_line(linewidth = 0.3, colour = "grey30") +
  geom_hline(yintercept = c(-2, 0, 2), linetype = c("dashed", "solid", "dashed"), 
             colour = c("blue", "black", "red")) +
  labs(title = "VRP Z-Score (Rolling 1Y)", x = NULL, y = "Z-Score") +
  theme_diag

# Plot 12: Cumulative VRP (drift indicator)
dt_vrp[, cum_vrp := cumsum(fifelse(is.na(vrp_cc), 0, vrp_cc))]
p12 <- ggplot(dt_vrp, aes(x = date, y = cum_vrp)) +
  geom_line(colour = "steelblue", linewidth = 0.5) +
  labs(title = "Cumulative VRP (Drift Diagnostic)", x = NULL, y = "Cumulative VRP") +
  theme_diag

# Combine plots
combined_1 <- gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
combined_2 <- gridExtra::grid.arrange(p5, p6, p7, p8, ncol = 2, nrow = 2)
combined_3 <- gridExtra::grid.arrange(p9, p10, p11, p12, ncol = 2, nrow = 2)

ggsave(file.path(dirs$figures, "38_vrp_diagnostics_1.pdf"), combined_1, width = 12, height = 10)
ggsave(file.path(dirs$figures, "39_vrp_diagnostics_2.pdf"), combined_2, width = 12, height = 10)
ggsave(file.path(dirs$figures, "40_vrp_diagnostics_3.pdf"), combined_3, width = 12, height = 10)

#------------------------------------------------------------------
# 16. SUMMARY TABLE EXPORT
#------------------------------------------------------------------

cat_progress("Exporting summary tables...")

# Main summary
write.csv(vrp_vol_stats, file.path(dirs$tables, "03_vrp_vol_stats.csv"), row.names = FALSE)
write.csv(vrp_var_stats, file.path(dirs$tables, "03_vrp_var_stats.csv"), row.names = FALSE)
write.csv(vrp_test_results, file.path(dirs$tables, "03_vrp_inference.csv"), row.names = FALSE)
write.csv(stationarity_results, file.path(dirs$tables, "03_vrp_stationarity.csv"), row.names = FALSE)
write.csv(subsample_stats, file.path(dirs$tables, "03_vrp_subsamples.csv"), row.names = FALSE)
write.csv(pred_ret_dt, file.path(dirs$tables, "03_vrp_predictive.csv"), row.names = FALSE)
write.csv(round(cor_matrix, 4), file.path(dirs$tables, "03_vrp_correlations.csv"))

#------------------------------------------------------------------
# 17. SAVE UPDATED DATA
#------------------------------------------------------------------

cat_progress("Saving data with VRP diagnostics...")

saveRDS(dt_vrp, file.path(dirs$data, "data_with_vrp_diagnostics.rds"))

#------------------------------------------------------------------
# 18. FINAL SUMMARY
#------------------------------------------------------------------

cat("\n")
cat("================================================================================\n")
cat("                     VRP DIAGNOSTIC SUMMARY                                     \n")
cat("================================================================================\n\n")

cat(sprintf("Sample: %s to %s (%d observations)\n\n", 
            min(dt_vrp$date), max(dt_vrp$date), n_obs))

cat("--- Key Statistics ---\n")
cat(sprintf("Mean VRP (vol):  %.2f (t-HAC = %.2f, p = %.4f)\n", 
            vrp_test_results[Variable == "VRP_vol (CC)"]$Mean,
            vrp_test_results[Variable == "VRP_vol (CC)"]$t_HAC,
            vrp_test_results[Variable == "VRP_vol (CC)"]$p_HAC))
cat(sprintf("Mean VRP (var):  %.2f\n", mean(dt_vrp$vrp_var_cc, na.rm = TRUE)))
cat(sprintf("Pct Positive:    %.1f%%\n", 100 * mean(dt_vrp$vrp_cc > 0, na.rm = TRUE)))
cat(sprintf("AR(1) coef:      %.3f (half-life = %.1f days)\n\n", phi1, half_life))

cat("--- Stationarity ---\n")
cat(sprintf("ADF p-value:     %.4f (stationary if < 0.05)\n", adf_vrp_vol$p.value))
cat(sprintf("KPSS p-value:    %.4f (stationary if > 0.05)\n", kpss_vrp_vol$p.value))
cat(sprintf("Frac. diff d:    %.3f\n\n", d_est$d))

cat("--- Predictive Power (22-day horizon) ---\n")
cat(sprintf("VRP -> Returns:  R2 = %.2f%%, t = %.2f\n", 
            pred_ret_dt[Horizon == 22]$R2, pred_ret_dt[Horizon == 22]$t))
cat(sprintf("VRP -> RV:       R2 = %.2f%%, t = %.2f\n\n", 
            pred_rv$r2 * 100, pred_rv$t_hac))

cat("--- Files Saved ---\n")
cat("  Figures: 38_vrp_diagnostics_1.pdf, 39_vrp_diagnostics_2.pdf, 40_vrp_diagnostics_3.pdf\n")
cat("  Tables:  03_vrp_*.csv\n")
cat("  Data:    data_with_vrp_diagnostics.rds\n")

cat("\n================================================================================\n")

#------------------------------------------------------------------
# CLEANUP
#------------------------------------------------------------------

rm(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
   combined_1, combined_2, combined_3, roll_plot_data, xcorr_dt, acf_df, pacf_df)
gc()

cat_progress("VRP diagnostics complete.")

#------------------------------------------------------------------
# END OF SCRIPT
#------------------------------------------------------------------#------------------------------------------------------------------
# VRP_Diagnostics.R
# Comprehensive Variance Risk Premium Analysis
# VRP defined as: VIX^2 - RV^2 (variance terms) and VIX - RV (volatility terms)
#------------------------------------------------------------------

source("Setup.R")

cat_progress("Loading data with RV...")
dt <- readRDS(file.path(dirs$data, "data_with_rv.rds"))

#------------------------------------------------------------------
# 1. VRP CONSTRUCTION AND DEFINITIONS
#------------------------------------------------------------------

cat_progress("Constructing VRP measures...")

# Primary VRP definitions (variance space, scaled for interpretability)
# VIX is quoted in annualised vol %, so VIX^2/100 gives annualised variance in %
dt[, `:=`(
  # Variance space: (VIX^2 - RV^2) / 100
  vrp_var_cc = (vix_close^2 - rv_cc^2) / 100,
  vrp_var_pk = (vix_close^2 - rv_parkinson^2) / 100,
  vrp_var_gk = (vix_close^2 - rv_gk^2) / 100,
  vrp_var_rs = (vix_close^2 - rv_rs^2) / 100,
  vrp_var_yz = (vix_close^2 - rv_yz^2) / 100,
  
  # Log variance ratio: log(VIX^2 / RV^2) - stationarity inducing
  vrp_log_cc = log(vix_close^2 / rv_cc^2),
  vrp_log_pk = log(vix_close^2 / rv_parkinson^2),
  
  # Squared VIX and RV for decomposition
  vix_var = vix_close^2 / 100,
  rv_var_cc = rv_cc^2 / 100
)]

# Remove observations with missing VRP
dt_vrp <- dt[!is.na(vrp_var_cc) & !is.na(vrp_cc)]
n_obs <- nrow(dt_vrp)
cat_progress(sprintf("VRP sample: %d observations (%s to %s)", 
                     n_obs, min(dt_vrp$date), max(dt_vrp$date)))

#------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
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
    Q05 = quantile(x, 0.05),
    Q25 = quantile(x, 0.25),
    Median = median(x),
    Q75 = quantile(x, 0.75),
    Q95 = quantile(x, 0.95),
    Max = max(x),
    Pct_Positive = 100 * mean(x > 0),
    IQR = IQR(x)
  )
}

# VRP in volatility terms
vrp_vol_stats <- rbindlist(list(
  compute_moments(dt_vrp$vrp_cc, "VRP (CC)"),
  compute_moments(dt_vrp$vrp_parkinson, "VRP (Parkinson)"),
  compute_moments(dt_vrp$vrp_gk, "VRP (GK)"),
  compute_moments(dt_vrp$vrp_rs, "VRP (RS)"),
  compute_moments(dt_vrp$vrp_yz, "VRP (YZ)")
))

# VRP in variance terms
vrp_var_stats <- rbindlist(list(
  compute_moments(dt_vrp$vrp_var_cc, "VRP_Var (CC)"),
  compute_moments(dt_vrp$vrp_var_pk, "VRP_Var (Parkinson)"),
  compute_moments(dt_vrp$vrp_var_gk, "VRP_Var (GK)"),
  compute_moments(dt_vrp$vrp_var_rs, "VRP_Var (RS)"),
  compute_moments(dt_vrp$vrp_var_yz, "VRP_Var (YZ)")
))

cat("\n=== VRP Descriptive Statistics (Volatility Terms) ===\n")
print(vrp_vol_stats[, .(Variable, N, Mean, SD, Skewness, Kurtosis, Pct_Positive)])

cat("\n=== VRP Descriptive Statistics (Variance Terms) ===\n")
print(vrp_var_stats[, .(Variable, N, Mean, SD, Skewness, Kurtosis, Pct_Positive)])

#------------------------------------------------------------------
# 3. STATISTICAL INFERENCE: IS MEAN VRP SIGNIFICANTLY POSITIVE?
#------------------------------------------------------------------

cat_progress("Testing significance of mean VRP with HAC standard errors...")

# Newey-West HAC inference (robust to serial correlation)
# Bandwidth selection: Andrews (1991) automatic or fixed at T^(1/3)
nw_bandwidth <- floor(n_obs^(1/3))

vrp_inference <- function(vrp, name, bandwidth) {
  vrp <- vrp[!is.na(vrp)]
  n <- length(vrp)
  
  # OLS regression: vrp_t = mu + epsilon_t
  fit <- lm(vrp ~ 1)
  
  # HAC covariance (Newey-West)
  nw_vcov <- sandwich::NeweyWest(fit, lag = bandwidth, prewhite = FALSE)
  
  # t-statistic with HAC SE
  mu_hat <- coef(fit)
  se_hac <- sqrt(diag(nw_vcov))
  t_hac <- mu_hat / se_hac
  p_hac <- 2 * pt(-abs(t_hac), df = n - 1)
  
  # Standard OLS SE for comparison
  se_ols <- summary(fit)$coefficients[1, 2]
  t_ols <- mu_hat / se_ols
  
  data.table(
    Variable = name,
    Mean = mu_hat,
    SE_OLS = se_ols,
    SE_HAC = se_hac,
    t_OLS = t_ols,
    t_HAC = t_hac,
    p_HAC = p_hac,
    CI_Lower = mu_hat - 1.96 * se_hac,
    CI_Upper = mu_hat + 1.96 * se_hac
  )
}

vrp_test_results <- rbindlist(list(
  vrp_inference(dt_vrp$vrp_cc, "VRP_vol (CC)", nw_bandwidth),
  vrp_inference(dt_vrp$vrp_var_cc, "VRP_var (CC)", nw_bandwidth),
  vrp_inference(dt_vrp$vrp_log_cc, "VRP_log (CC)", nw_bandwidth)
))

cat("\n=== Mean VRP Significance Tests (Newey-West HAC) ===\n")
cat(sprintf("Bandwidth: %d lags\n\n", nw_bandwidth))
print(vrp_test_results)

#------------------------------------------------------------------
# 4. DISTRIBUTIONAL TESTS
#------------------------------------------------------------------

cat_progress("Testing distributional properties...")

vrp_cc <- dt_vrp$vrp_cc[!is.na(dt_vrp$vrp_cc)]
vrp_var <- dt_vrp$vrp_var_cc[!is.na(dt_vrp$vrp_var_cc)]

# Jarque-Bera test for normality
jb_vol <- tseries::jarque.bera.test(vrp_cc)
jb_var <- tseries::jarque.bera.test(vrp_var)

# Shapiro-Wilk (on subsample due to n limit)
sw_vol <- shapiro.test(sample(vrp_cc, min(5000, length(vrp_cc))))
sw_var <- shapiro.test(sample(vrp_var, min(5000, length(vrp_var))))

cat("\n=== Normality Tests ===\n")
cat(sprintf("VRP (vol): JB stat = %.2f, p = %.4e\n", jb_vol$statistic, jb_vol$p.value))
cat(sprintf("VRP (var): JB stat = %.2f, p = %.4e\n", jb_var$statistic, jb_var$p.value))
cat(sprintf("VRP (vol): SW stat = %.4f, p = %.4e (subsample)\n", sw_vol$statistic, sw_vol$p.value))

# Heavy tails: Hill estimator for tail index
# Right tail
compute_hill <- function(x, k = NULL) {
  x <- x[x > 0]
  x <- sort(x, decreasing = TRUE)
  n <- length(x)
  if (is.null(k)) k <- floor(sqrt(n))
  k <- min(k, n - 1)
  
  hill <- (1/k) * sum(log(x[1:k]) - log(x[k + 1]))
  alpha <- 1 / hill
  se <- alpha / sqrt(k)
  
  list(alpha = alpha, se = se, k = k)
}

hill_right <- compute_hill(vrp_var[vrp_var > 0])
hill_left <- compute_hill(-vrp_var[vrp_var < 0])

cat("\n=== Tail Index (Hill Estimator) ===\n")
cat(sprintf("Right tail: alpha = %.2f (SE = %.2f), k = %d\n", 
            hill_right$alpha, hill_right$se, hill_right$k))
cat(sprintf("Left tail:  alpha = %.2f (SE = %.2f), k = %d\n", 
            hill_left$alpha, hill_left$se, hill_left$k))

#------------------------------------------------------------------
# 5. STATIONARITY AND UNIT ROOT TESTS
#------------------------------------------------------------------

cat_progress("Testing stationarity...")

# ADF test with automatic lag selection (AIC)
adf_vrp_vol <- tseries::adf.test(vrp_cc, alternative = "stationary")
adf_vrp_var <- tseries::adf.test(vrp_var, alternative = "stationary")

# KPSS test (null: stationarity)
kpss_vrp_vol <- tseries::kpss.test(vrp_cc, null = "Level")
kpss_vrp_var <- tseries::kpss.test(vrp_var, null = "Level")

# Phillips-Perron test
pp_vrp_vol <- tseries::pp.test(vrp_cc, alternative = "stationary")
pp_vrp_var <- tseries::pp.test(vrp_var, alternative = "stationary")

cat("\n=== Unit Root / Stationarity Tests ===\n")
cat("H0 for ADF/PP: Unit root; H0 for KPSS: Stationarity\n\n")

stationarity_results <- data.table(
  Series = c("VRP (vol)", "VRP (var)"),
  ADF_stat = c(adf_vrp_vol$statistic, adf_vrp_var$statistic),
  ADF_p = c(adf_vrp_vol$p.value, adf_vrp_var$p.value),
  PP_stat = c(pp_vrp_vol$statistic, pp_vrp_var$statistic),
  PP_p = c(pp_vrp_vol$p.value, pp_vrp_var$p.value),
  KPSS_stat = c(kpss_vrp_vol$statistic, kpss_vrp_var$statistic),
  KPSS_p = c(kpss_vrp_vol$p.value, kpss_vrp_var$p.value)
)
print(stationarity_results)

#------------------------------------------------------------------
# 6. AUTOCORRELATION STRUCTURE
#------------------------------------------------------------------

cat_progress("Analysing autocorrelation structure...")

max_lag <- 60

# ACF and PACF
acf_vrp <- acf(vrp_cc, lag.max = max_lag, plot = FALSE)
pacf_vrp <- pacf(vrp_cc, lag.max = max_lag, plot = FALSE)
acf_vrp_var <- acf(vrp_var, lag.max = max_lag, plot = FALSE)

# Ljung-Box test at various lags
lb_results <- data.table(
  Lag = c(5, 10, 22, 44, 66),
  Q_stat = sapply(c(5, 10, 22, 44, 66), function(k) {
    Box.test(vrp_cc, lag = k, type = "Ljung-Box")$statistic
  }),
  p_value = sapply(c(5, 10, 22, 44, 66), function(k) {
    Box.test(vrp_cc, lag = k, type = "Ljung-Box")$p.value
  })
)

cat("\n=== Ljung-Box Tests for Serial Correlation ===\n")
print(lb_results)

# Half-life of autocorrelation (AR(1) approximation)
ar1_fit <- ar(vrp_cc, order.max = 1, aic = FALSE, method = "ols")
phi1 <- ar1_fit$ar[1]
half_life <- -log(2) / log(abs(phi1))

cat(sprintf("\nAR(1) coefficient: %.4f\n", phi1))
cat(sprintf("Implied half-life: %.1f days\n", half_life))

# ARCH effects test
arch_test <- FinTS::ArchTest(vrp_cc, lags = 10)
cat(sprintf("\nARCH-LM test (10 lags): stat = %.2f, p = %.4e\n", 
            arch_test$statistic, arch_test$p.value))

#------------------------------------------------------------------
# 7. PERSISTENCE ANALYSIS: FRACTIONAL INTEGRATION
#------------------------------------------------------------------

cat_progress("Estimating persistence (fractional integration)...")

# GPH estimator for d (log-periodogram regression)
estimate_d_gph <- function(x, m = NULL) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (is.null(m)) m <- floor(n^0.65)  # Geweke-Porter-Hudak bandwidth
  
  # Periodogram
  spec <- spectrum(x, plot = FALSE, taper = 0)
  freq <- spec$freq[1:m]
  pgram <- spec$spec[1:m]
  
  # Log-periodogram regression
  y <- log(pgram)
  X <- log(4 * sin(pi * freq)^2)
  
  fit <- lm(y ~ X)
  d_hat <- -coef(fit)[2] / 2
  se_d <- summary(fit)$coefficients[2, 2] / 2
  
  list(d = d_hat, se = se_d, m = m)
}

d_est <- estimate_d_gph(vrp_cc)
cat(sprintf("\nFractional integration parameter (GPH): d = %.3f (SE = %.3f)\n", 
            d_est$d, d_est$se))
cat(sprintf("Bandwidth m = %d\n", d_est$m))

if (d_est$d < 0.5) {
  cat("Interpretation: VRP is I(0) - stationary with possible long memory\n")
} else if (d_est$d < 1) {
  cat("Interpretation: VRP is I(d) with 0.5 < d < 1 - mean reverting but persistent\n")
}

#------------------------------------------------------------------
# 8. REGIME ANALYSIS
#------------------------------------------------------------------

cat_progress("Analysing VRP regimes...")

# Define regimes based on VRP level
dt_vrp[, vrp_regime := cut(vrp_cc, 
                           breaks = quantile(vrp_cc, c(0, 0.1, 0.25, 0.75, 0.9, 1), na.rm = TRUE),
                           labels = c("Very Low", "Low", "Normal", "High", "Very High"),
                           include.lowest = TRUE)]

# Also define by sign
dt_vrp[, vrp_sign := fifelse(vrp_cc > 0, "Positive", "Negative")]

# Summary by regime
regime_stats <- dt_vrp[, .(
  N = .N,
  Pct = 100 * .N / n_obs,
  Mean_VRP_vol = mean(vrp_cc, na.rm = TRUE),
  Mean_VRP_var = mean(vrp_var_cc, na.rm = TRUE),
  Mean_VIX = mean(vix_close, na.rm = TRUE),
  Mean_RV = mean(rv_cc, na.rm = TRUE),
  Mean_Return = mean(log_return, na.rm = TRUE) * 252  # Annualised
), by = vrp_regime]

cat("\n=== VRP Regime Analysis ===\n")
print(regime_stats[order(vrp_regime)])

# Sign analysis
sign_stats <- dt_vrp[, .(
  N = .N,
  Pct = 100 * .N / n_obs,
  Mean_VRP = mean(vrp_cc, na.rm = TRUE),
  Mean_FwdReturn = mean(shift(log_return, -22, type = "lead"), na.rm = TRUE) * 12  # 22-day forward, annualised
), by = vrp_sign]

cat("\n=== VRP Sign Analysis ===\n")
print(sign_stats)

# Transition matrix (Markov chain)
dt_vrp[, vrp_regime_lag := shift(vrp_regime, 1, type = "lag")]
transition_matrix <- dt_vrp[!is.na(vrp_regime) & !is.na(vrp_regime_lag), 
                            .N, by = .(vrp_regime_lag, vrp_regime)]
transition_matrix <- dcast(transition_matrix, vrp_regime_lag ~ vrp_regime, value.var = "N", fill = 0)

# Normalise to probabilities
trans_probs <- as.matrix(transition_matrix[, -1])
rownames(trans_probs) <- transition_matrix$vrp_regime_lag
trans_probs <- trans_probs / rowSums(trans_probs)

cat("\n=== Regime Transition Probabilities ===\n")
print(round(trans_probs, 3))

#------------------------------------------------------------------
# 9. ASYMMETRY ANALYSIS
#------------------------------------------------------------------

cat_progress("Testing VRP asymmetry...")

# VRP conditional on market returns
dt_vrp[, return_sign := fifelse(log_return >= 0, "Up", "Down")]
dt_vrp[, return_quintile := cut(log_return, 
                                breaks = quantile(log_return, seq(0, 1, 0.2), na.rm = TRUE),
                                labels = c("Q1 (Worst)", "Q2", "Q3", "Q4", "Q5 (Best)"),
                                include.lowest = TRUE)]

asymm_by_return <- dt_vrp[!is.na(return_quintile), .(
  Mean_VRP = mean(vrp_cc, na.rm = TRUE),
  SD_VRP = sd(vrp_cc, na.rm = TRUE),
  Mean_VIX = mean(vix_close, na.rm = TRUE),
  Mean_RV = mean(rv_cc, na.rm = TRUE),
  N = .N
), by = return_quintile]

cat("\n=== VRP by Return Quintile ===\n")
print(asymm_by_return[order(return_quintile)])

# Test asymmetry formally: VRP_t = a + b*I(r<0)*|r| + c*I(r>=0)*r + e
dt_vrp[, `:=`(
  neg_return = pmin(log_return, 0),
  pos_return = pmax(log_return, 0),
  abs_return = abs(log_return)
)]

asymm_reg <- lm(vrp_cc ~ neg_return + pos_return, data = dt_vrp)
asymm_vcov <- sandwich::NeweyWest(asymm_reg, lag = nw_bandwidth)
asymm_test <- lmtest::coeftest(asymm_reg, vcov = asymm_vcov)

cat("\n=== VRP Asymmetry Regression (HAC SE) ===\n")
cat("VRP_t = a + b*r^- + c*r^+ + e\n\n")
print(asymm_test)

# Wald test: H0: b = c (symmetric response)
wald_asymm <- car::linearHypothesis(asymm_reg, "neg_return = pos_return", vcov = asymm_vcov)
cat(sprintf("\nWald test for asymmetry (H0: b = c): F = %.2f, p = %.4f\n",
            wald_asymm$F[2], wald_asymm$`Pr(>F)`[2]))

#------------------------------------------------------------------
# 10. PREDICTIVE REGRESSIONS
#------------------------------------------------------------------

cat_progress("Running predictive regressions...")

# Create forward returns at various horizons
horizons <- c(1, 5, 22, 66)  # 1d, 1w, 1m, 3m
for (h in horizons) {
  dt_vrp[, paste0("fwd_ret_", h) := shift(log_return, -h, type = "lead")]
  # Cumulative returns
  if (h > 1) {
    dt_vrp[, paste0("fwd_cumret_", h) := frollsum(shift(log_return, -1, type = "lead"), n = h, align = "left")]
  }
}

# Forward RV
dt_vrp[, fwd_rv_22 := shift(rv_cc, -22, type = "lead")]

# Standardise VRP for comparability
dt_vrp[, vrp_std := (vrp_cc - mean(vrp_cc, na.rm = TRUE)) / sd(vrp_cc, na.rm = TRUE)]
dt_vrp[, vrp_var_std := (vrp_var_cc - mean(vrp_var_cc, na.rm = TRUE)) / sd(vrp_var_cc, na.rm = TRUE)]

# Predictive regression function with HAC and Hodrick (1992) correction
run_predictive_reg <- function(data, y_var, x_var, horizon, nw_lag = NULL) {
  if (is.null(nw_lag)) nw_lag <- horizon + 5
  
  formula <- as.formula(paste(y_var, "~", x_var))
  fit <- lm(formula, data = data)
  
  # Newey-West with appropriate lag
  hac_vcov <- sandwich::NeweyWest(fit, lag = nw_lag, prewhite = FALSE)
  coef_test <- lmtest::coeftest(fit, vcov = hac_vcov)
  
  # R-squared
  r2 <- summary(fit)$r.squared
  adj_r2 <- summary(fit)$adj.r.squared
  
  list(
    coef = coef(fit)[2],
    se_hac = sqrt(diag(hac_vcov))[2],
    t_hac = coef_test[2, 3],
    p_hac = coef_test[2, 4],
    r2 = r2,
    n = nobs(fit)
  )
}

# VRP predicting returns
cat("\n=== VRP Predicting Future Returns (HAC SE) ===\n")
pred_ret_results <- lapply(c(1, 5, 22, 66), function(h) {
  y_var <- ifelse(h == 1, "fwd_ret_1", paste0("fwd_cumret_", h))
  res <- run_predictive_reg(dt_vrp, y_var, "vrp_std", h)
  data.table(Horizon = h, Beta = res$coef, SE = res$se_hac, 
             t = res$t_hac, p = res$p_hac, R2 = res$r2 * 100)
})
pred_ret_dt <- rbindlist(pred_ret_results)
print(pred_ret_dt)

# VRP predicting future RV
cat("\n=== VRP Predicting Future RV ===\n")
pred_rv <- run_predictive_reg(dt_vrp, "fwd_rv_22", "vrp_std", 22)
cat(sprintf("Beta = %.4f, t = %.2f, p = %.4f, R2 = %.2f%%\n",
            pred_rv$coef, pred_rv$t_hac, pred_rv$p_hac, pred_rv$r2 * 100))

# VRP predicting VIX changes
dt_vrp[, fwd_vix_change := shift(vix_close, -22, type = "lead") - vix_close]
pred_vix <- run_predictive_reg(dt_vrp, "fwd_vix_change", "vrp_std", 22)
cat(sprintf("VRP -> VIX change: Beta = %.4f, t = %.2f, p = %.4f\n",
            pred_vix$coef, pred_vix$t_hac, pred_vix$p_hac))

#------------------------------------------------------------------
# 11. SUBSAMPLE STABILITY
#------------------------------------------------------------------

cat_progress("Testing subsample stability...")

# Define subsamples
subsamples <- list(
  "1990-1999" = dt_vrp[date >= "1990-01-01" & date < "2000-01-01"],
  "2000-2007" = dt_vrp[date >= "2000-01-01" & date < "2008-01-01"],
  "2008-2009 (GFC)" = dt_vrp[date >= "2008-01-01" & date < "2010-01-01"],
  "2010-2019" = dt_vrp[date >= "2010-01-01" & date < "2020-01-01"],
  "2020-Present" = dt_vrp[date >= "2020-01-01"]
)

subsample_stats <- rbindlist(lapply(names(subsamples), function(name) {
  d <- subsamples[[name]]
  if (nrow(d) < 50) return(NULL)
  data.table(
    Period = name,
    N = nrow(d),
    Mean_VRP_vol = mean(d$vrp_cc, na.rm = TRUE),
    SD_VRP_vol = sd(d$vrp_cc, na.rm = TRUE),
    Mean_VRP_var = mean(d$vrp_var_cc, na.rm = TRUE),
    Pct_Positive = 100 * mean(d$vrp_cc > 0, na.rm = TRUE),
    Mean_VIX = mean(d$vix_close, na.rm = TRUE),
    Corr_VIX_RV = cor(d$vix_close, d$rv_cc, use = "complete.obs")
  )
}))

cat("\n=== Subsample Statistics ===\n")
print(subsample_stats)

# Structural break test (Chow test approximation via rolling window)
roll_mean_vrp <- frollmean(dt_vrp$vrp_cc, n = 252, align = "right")
roll_sd_vrp <- frollapply(dt_vrp$vrp_cc, n = 252, FUN = sd, align = "right")

#------------------------------------------------------------------
# 12. CORRELATION WITH OTHER VARIABLES
#------------------------------------------------------------------

cat_progress("Computing correlations with market variables...")

# Lagged correlations
lag_range <- -22:22

compute_xcorr <- function(x, y, lags) {
  sapply(lags, function(k) {
    if (k >= 0) {
      cor(x, shift(y, k, type = "lag"), use = "complete.obs")
    } else {
      cor(x, shift(y, -k, type = "lead"), use = "complete.obs")
    }
  })
}

xcorr_ret <- compute_xcorr(dt_vrp$vrp_cc, dt_vrp$log_return, lag_range)
xcorr_vix <- compute_xcorr(dt_vrp$vrp_cc, dt_vrp$vix_close, lag_range)
xcorr_rv <- compute_xcorr(dt_vrp$vrp_cc, dt_vrp$rv_cc, lag_range)

xcorr_dt <- data.table(
  lag = lag_range,
  xcorr_return = xcorr_ret,
  xcorr_vix = xcorr_vix,
  xcorr_rv = xcorr_rv
)

# Contemporaneous correlations
cat("\n=== Contemporaneous Correlations ===\n")
cor_vars <- c("vrp_cc", "vrp_var_cc", "vix_close", "rv_cc", "log_return", "abs_return")
cor_matrix <- cor(dt_vrp[, .SD, .SDcols = cor_vars], use = "pairwise.complete.obs")
print(round(cor_matrix, 3))

#------------------------------------------------------------------
# 13. VRP DECOMPOSITION
#------------------------------------------------------------------

cat_progress("Decomposing VRP variance...")

# VRP = VIX - RV, so Var(VRP) = Var(VIX) + Var(RV) - 2*Cov(VIX, RV)
var_vix <- var(dt_vrp$vix_close, na.rm = TRUE)
var_rv <- var(dt_vrp$rv_cc, na.rm = TRUE)
cov_vix_rv <- cov(dt_vrp$vix_close, dt_vrp$rv_cc, use = "complete.obs")
var_vrp <- var(dt_vrp$vrp_cc, na.rm = TRUE)

cat("\n=== VRP Variance Decomposition ===\n")
cat(sprintf("Var(VIX) = %.2f\n", var_vix))
cat(sprintf("Var(RV)  = %.2f\n", var_rv))
cat(sprintf("Cov(VIX,RV) = %.2f\n", cov_vix_rv))
cat(sprintf("Cor(VIX,RV) = %.3f\n", cov_vix_rv / sqrt(var_vix * var_rv)))
cat(sprintf("\nVar(VRP) = Var(VIX) + Var(RV) - 2*Cov = %.2f\n", var_vrp))
cat(sprintf("Check: %.2f + %.2f - 2*%.2f = %.2f\n", 
            var_vix, var_rv, cov_vix_rv, var_vix + var_rv - 2*cov_vix_rv))

# Contribution percentages (for intuition, not exact for variance space VRP)
cat(sprintf("\nVariance contributions (approx):\n"))
cat(sprintf("  From VIX variance: %.1f%%\n", 100 * var_vix / (var_vix + var_rv)))
cat(sprintf("  From RV variance:  %.1f%%\n", 100 * var_rv / (var_vix + var_rv)))
cat(sprintf("  Correlation effect: reduces by %.1f%%\n", 
            100 * 2 * cov_vix_rv / (var_vix + var_rv)))

#------------------------------------------------------------------
# 14. ROLLING STATISTICS
#------------------------------------------------------------------

cat_progress("Computing rolling statistics...")

roll_windows <- c(63, 126, 252)  # 3m, 6m, 1y

for (w in roll_windows) {
  dt_vrp[, paste0("vrp_roll_mean_", w) := frollmean(vrp_cc, n = w, align = "right")]
  dt_vrp[, paste0("vrp_roll_sd_", w) := frollapply(vrp_cc, n = w, FUN = sd, align = "right")]
  dt_vrp[, paste0("vrp_roll_sharpe_", w) := get(paste0("vrp_roll_mean_", w)) / 
           get(paste0("vrp_roll_sd_", w)) * sqrt(252)]
}

# Z-score (how many SDs from rolling mean)
dt_vrp[, vrp_zscore := (vrp_cc - vrp_roll_mean_252) / vrp_roll_sd_252]

#------------------------------------------------------------------
# 15. DIAGNOSTIC PLOTS
#------------------------------------------------------------------

cat_progress("Creating diagnostic plots...")

theme_diag <- theme_minimal() +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "bottom"
  )

# Plot 1: VRP time series with regimes
p1 <- ggplot(dt_vrp, aes(x = date, y = vrp_cc)) +
  geom_line(linewidth = 0.3, colour = "grey30") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  geom_hline(yintercept = mean(dt_vrp$vrp_cc, na.rm = TRUE), 
             linetype = "dashed", colour = "blue") +
  labs(title = "Variance Risk Premium (VIX - RV)", 
       x = NULL, y = "VRP (vol points)") +
  theme_diag

# Plot 2: VRP distribution
p2 <- ggplot(dt_vrp, aes(x = vrp_cc)) +
  geom_histogram(aes(y = after_stat(density)), bins = 100, 
                 fill = "steelblue", alpha = 0.7) +
  geom_density(colour = "darkred", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = mean(dt_vrp$vrp_cc, na.rm = TRUE), 
             colour = "blue", linetype = "dashed") +
  labs(title = "VRP Distribution", x = "VRP (vol points)", y = "Density") +
  theme_diag

# Plot 3: ACF plot
acf_df <- data.table(
  lag = 1:max_lag,
  acf = acf_vrp$acf[2:(max_lag + 1)]
)
ci <- 1.96 / sqrt(n_obs)

p3 <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_hline(yintercept = c(-ci, ci), linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0) +
  geom_segment(aes(xend = lag, yend = 0), colour = "steelblue") +
  geom_point(colour = "steelblue", size = 1.5) +
  labs(title = "Autocorrelation Function of VRP", x = "Lag (days)", y = "ACF") +
  theme_diag

# Plot 4: PACF plot
pacf_df <- data.table(
  lag = 1:max_lag,
  pacf = pacf_vrp$acf[1:max_lag]
)

p4 <- ggplot(pacf_df, aes(x = lag, y = pacf)) +
  geom_hline(yintercept = c(-ci, ci), linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0) +
  geom_segment(aes(xend = lag, yend = 0), colour = "darkred") +
  geom_point(colour = "darkred", size = 1.5) +
  labs(title = "Partial ACF of VRP", x = "Lag (days)", y = "PACF") +
  theme_diag

# Plot 5: VRP vs VIX scatter
p5 <- ggplot(dt_vrp, aes(x = vix_close, y = vrp_cc)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "loess", colour = "red", linewidth = 0.8, se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "VRP vs VIX Level", x = "VIX", y = "VRP") +
  theme_diag

# Plot 6: VRP variance space time series
p6 <- ggplot(dt_vrp, aes(x = date, y = vrp_var_cc)) +
  geom_line(linewidth = 0.3, colour = "grey30") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  labs(title = expression(paste("VRP in Variance Space: ", VIX^2 - RV^2)), 
       x = NULL, y = "VRP (variance points)") +
  theme_diag

# Plot 7: Rolling mean VRP
roll_plot_data <- melt(
  dt_vrp[, .(date, `3M` = vrp_roll_mean_63, `6M` = vrp_roll_mean_126, `1Y` = vrp_roll_mean_252)],
  id.vars = "date", variable.name = "Window", value.name = "Mean_VRP"
)

p7 <- ggplot(roll_plot_data[!is.na(Mean_VRP)], aes(x = date, y = Mean_VRP, colour = Window)) +
  geom_line(linewidth = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_colour_viridis_d(option = "C", end = 0.8) +
  labs(title = "Rolling Mean VRP", x = NULL, y = "Mean VRP (vol points)") +
  theme_diag

# Plot 8: Cross-correlation with returns
p8 <- ggplot(xcorr_dt, aes(x = lag, y = xcorr_return)) +
  geom_hline(yintercept = c(-ci, ci), linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0) +
  geom_segment(aes(xend = lag, yend = 0), colour = "steelblue") +
  geom_point(colour = "steelblue", size = 1.5) +
  labs(title = "Cross-correlation: VRP vs Returns", 
       subtitle = "Negative lag = VRP leads returns",
       x = "Lag (days)", y = "Cross-correlation") +
  theme_diag

# Plot 9: VRP by VIX regime
dt_vrp[, vix_regime := cut(vix_close, 
                           breaks = c(0, 15, 20, 25, 35, 100),
                           labels = c("<15", "15-20", "20-25", "25-35", ">35"))]

p9 <- ggplot(dt_vrp[!is.na(vix_regime)], aes(x = vix_regime, y = vrp_cc, fill = vix_regime)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  scale_fill_viridis_d(option = "D") +
  labs(title = "VRP Distribution by VIX Regime", x = "VIX Level", y = "VRP") +
  theme_diag +
  theme(legend.position = "none")

# Plot 10: QQ plot
p10 <- ggplot(dt_vrp, aes(sample = vrp_cc)) +
  stat_qq(alpha = 0.3, size = 0.5) +
  stat_qq_line(colour = "red") +
  labs(title = "Q-Q Plot: VRP vs Normal", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_diag

# Plot 11: VRP Z-score
p11 <- ggplot(dt_vrp[!is.na(vrp_zscore)], aes(x = date, y = vrp_zscore)) +
  geom_line(linewidth = 0.3, colour = "grey30") +
  geom_hline(yintercept = c(-2, 0, 2), linetype = c("dashed", "solid", "dashed"), 
             colour = c("blue", "black", "red")) +
  labs(title = "VRP Z-Score (Rolling 1Y)", x = NULL, y = "Z-Score") +
  theme_diag

# Plot 12: Cumulative VRP (drift indicator)
dt_vrp[, cum_vrp := cumsum(fifelse(is.na(vrp_cc), 0, vrp_cc))]
p12 <- ggplot(dt_vrp, aes(x = date, y = cum_vrp)) +
  geom_line(colour = "steelblue", linewidth = 0.5) +
  labs(title = "Cumulative VRP (Drift Diagnostic)", x = NULL, y = "Cumulative VRP") +
  theme_diag

# Combine plots
combined_1 <- gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
combined_2 <- gridExtra::grid.arrange(p5, p6, p7, p8, ncol = 2, nrow = 2)
combined_3 <- gridExtra::grid.arrange(p9, p10, p11, p12, ncol = 2, nrow = 2)

ggsave(file.path(dirs$figures, "41_vrp_diagnostics_1.pdf"), combined_1, width = 12, height = 10)
ggsave(file.path(dirs$figures, "42_vrp_diagnostics_2.pdf"), combined_2, width = 12, height = 10)
ggsave(file.path(dirs$figures, "43_vrp_diagnostics_3.pdf"), combined_3, width = 12, height = 10)

#------------------------------------------------------------------
# 16. SUMMARY TABLE EXPORT
#------------------------------------------------------------------

cat_progress("Exporting summary tables...")

# Main summary
write.csv(vrp_vol_stats, file.path(dirs$tables, "03_vrp_vol_stats.csv"), row.names = FALSE)
write.csv(vrp_var_stats, file.path(dirs$tables, "03_vrp_var_stats.csv"), row.names = FALSE)
write.csv(vrp_test_results, file.path(dirs$tables, "03_vrp_inference.csv"), row.names = FALSE)
write.csv(stationarity_results, file.path(dirs$tables, "03_vrp_stationarity.csv"), row.names = FALSE)
write.csv(subsample_stats, file.path(dirs$tables, "03_vrp_subsamples.csv"), row.names = FALSE)
write.csv(pred_ret_dt, file.path(dirs$tables, "03_vrp_predictive.csv"), row.names = FALSE)
write.csv(round(cor_matrix, 4), file.path(dirs$tables, "03_vrp_correlations.csv"))

#------------------------------------------------------------------
# 17. SAVE UPDATED DATA
#------------------------------------------------------------------

cat_progress("Saving data with VRP diagnostics...")

saveRDS(dt_vrp, file.path(dirs$data, "data_with_vrp_diagnostics.rds"))

#------------------------------------------------------------------
# 18. FINAL SUMMARY
#------------------------------------------------------------------

cat("\n")
cat("================================================================================\n")
cat("                     VRP DIAGNOSTIC SUMMARY                                     \n")
cat("================================================================================\n\n")

cat(sprintf("Sample: %s to %s (%d observations)\n\n", 
            min(dt_vrp$date), max(dt_vrp$date), n_obs))

cat("--- Key Statistics ---\n")
cat(sprintf("Mean VRP (vol):  %.2f (t-HAC = %.2f, p = %.4f)\n", 
            vrp_test_results[Variable == "VRP_vol (CC)"]$Mean,
            vrp_test_results[Variable == "VRP_vol (CC)"]$t_HAC,
            vrp_test_results[Variable == "VRP_vol (CC)"]$p_HAC))
cat(sprintf("Mean VRP (var):  %.2f\n", mean(dt_vrp$vrp_var_cc, na.rm = TRUE)))
cat(sprintf("Pct Positive:    %.1f%%\n", 100 * mean(dt_vrp$vrp_cc > 0, na.rm = TRUE)))
cat(sprintf("AR(1) coef:      %.3f (half-life = %.1f days)\n\n", phi1, half_life))

cat("--- Stationarity ---\n")
cat(sprintf("ADF p-value:     %.4f (stationary if < 0.05)\n", adf_vrp_vol$p.value))
cat(sprintf("KPSS p-value:    %.4f (stationary if > 0.05)\n", kpss_vrp_vol$p.value))
cat(sprintf("Frac. diff d:    %.3f\n\n", d_est$d))

cat("--- Predictive Power (22-day horizon) ---\n")
cat(sprintf("VRP -> Returns:  R2 = %.2f%%, t = %.2f\n", 
            pred_ret_dt[Horizon == 22]$R2, pred_ret_dt[Horizon == 22]$t))
cat(sprintf("VRP -> RV:       R2 = %.2f%%, t = %.2f\n\n", 
            pred_rv$r2 * 100, pred_rv$t_hac))

cat("--- Files Saved ---\n")
cat("  Figures: 41_vrp_diagnostics_1.pdf, 42_vrp_diagnostics_2.pdf, 43_vrp_diagnostics_3.pdf\n")
cat("  Tables:  03_vrp_*.csv\n")
cat("  Data:    data_with_vrp_diagnostics.rds\n")

cat("\n================================================================================\n")

#------------------------------------------------------------------
# CLEANUP 
#------------------------------------------------------------------

rm(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
   combined_1, combined_2, combined_3, roll_plot_data, xcorr_dt, acf_df, pacf_df)
gc()

cat_progress("VRP diagnostics complete.")

#------------------------------------------------------------------
# END OF SCRIPT
#------------------------------------------------------------------