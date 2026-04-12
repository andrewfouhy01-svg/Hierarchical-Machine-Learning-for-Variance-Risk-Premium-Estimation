#------------------------------------------------------------------
# Realised_Volatility.R
#------------------------------------------------------------------

source("Setup.R")

merged_dt <- readRDS(file.path(dirs$data, "merged_data.rds"))

#------------------------------------------------------------------
# 1. CONFIGURATION
#------------------------------------------------------------------

RV_WINDOW <- 22L
ANNUAL_FACTOR <- 252

#------------------------------------------------------------------
# 2. DAILY VARIANCE ESTIMATORS
#------------------------------------------------------------------

# All estimators return daily variance (not volatility)

# 2.1 Close-to-close
merged_dt[, var_cc := log_return^2]

# 2.2 Parkinson (1980) - uses high/low range
# Var = (1 / 4*log(2)) * (log(H/L))^2
# More efficient than close-to-close by factor of ~5 under GBM
merged_dt[, var_parkinson := (1 / (4 * log(2))) * (log(high / low))^2]

# 2.3 Garman-Klass (1980) - uses OHLC
# Var = 0.5*(log(H/L))^2 - (2*log(2)-1)*(log(C/O))^2
# Efficiency ~8x close-to-close under GBM
merged_dt[, var_gk := 0.5 * (log(high / low))^2 - 
            (2 * log(2) - 1) * (log(close / open))^2]

# 2.4 Rogers-Satchell (1991) - drift-independent
# Var = log(H/C)*log(H/O) + log(L/C)*log(L/O)
merged_dt[, var_rs := log(high / close) * log(high / open) + 
            log(low / close) * log(low / open)]

# 2.5 Yang-Zhang (2000) - most efficient, handles overnight jumps
# Requires overnight and open-to-close components
merged_dt[, `:=`(
  log_oc = log(open / shift(close, 1L)),  # overnight return
  log_co = log(close / open)               # open-to-close return
)]

# Yang-Zhang parameters
k_yz <- 0.34 / (1.34 + (RV_WINDOW + 1) / (RV_WINDOW - 1))

#------------------------------------------------------------------
# 3. FORWARD ROLLING REALISED VOLATILITY
#------------------------------------------------------------------

# Function to compute forward RV using frollsum (vectorised)
# RV_t uses returns from t+1 to t+window (forward-looking to match VIX)
compute_forward_rv <- function(daily_var, window, annual_factor) {
  n <- length(daily_var)
  
  # Shift variance backward by 1 to align: we want var[t+1:t+window] for RV_t
  # Then use frollsum with align="left" 
  # frollsum with align="left" at position i sums [i, i+window-1]
  # We want sum of [i+1, i+window], so shift result back by 1
  
  rv_var <- frollsum(daily_var, n = window, align = "left", na.rm = FALSE)
  
  # Shift back: RV at t should use variance from t+1 onwards
  rv_var <- shift(rv_var, n = 1L, type = "lead")
  
  # Annualise and convert to volatility
  rv <- sqrt(rv_var * annual_factor / window) * 100  # percentage terms like VIX
  
  return(rv)
}

# Compute RV for each estimator
merged_dt[, `:=`(
  rv_cc = compute_forward_rv(var_cc, RV_WINDOW, ANNUAL_FACTOR),
  rv_parkinson = compute_forward_rv(var_parkinson, RV_WINDOW, ANNUAL_FACTOR),
  rv_gk = compute_forward_rv(var_gk, RV_WINDOW, ANNUAL_FACTOR),
  rv_rs = compute_forward_rv(var_rs, RV_WINDOW, ANNUAL_FACTOR)
)]

# Yang-Zhang requires separate handling (combines overnight, open-close, RS)
# YZ = k*var_overnight + (1-k)*var_rs (simplified robust version)
merged_dt[, var_overnight := log_oc^2]
merged_dt[, var_yz := k_yz * var_overnight + (1 - k_yz) * var_rs]
merged_dt[, rv_yz := compute_forward_rv(var_yz, RV_WINDOW, ANNUAL_FACTOR)]

#------------------------------------------------------------------
# 4. VARIANCE RISK PREMIUM
#------------------------------------------------------------------

cat_progress("Computing variance risk premium...")

# VRP_t = VIX_t - RV_t (in volatility terms)
# Positive VRP indicates risk-neutral > physical (typical)
merged_dt[, `:=`(
  vrp_cc = vix_close - rv_cc,
  vrp_parkinson = vix_close - rv_parkinson,
  vrp_gk = vix_close - rv_gk,
  vrp_rs = vix_close - rv_rs,
  vrp_yz = vix_close - rv_yz
)]

# Also compute in variance terms (VIX^2 - RV^2) / 100 for interpretability
merged_dt[, `:=`(
  vrp_var_cc = (vix_close^2 - rv_cc^2) / 100,
  vrp_var_parkinson = (vix_close^2 - rv_parkinson^2) / 100
)]

#------------------------------------------------------------------
# 5. BACKWARD-LOOKING RV (for HAR-RV model features)
#------------------------------------------------------------------

cat_progress("Computing backward-looking RV for HAR features...")

# HAR-RV uses lagged RV at daily, weekly, monthly horizons
compute_backward_rv <- function(daily_var, window, annual_factor) {
  rv_var <- frollsum(daily_var, n = window, align = "right", na.rm = FALSE)
  rv <- sqrt(rv_var * annual_factor / window) * 100
  return(rv)
}

# Standard HAR-RV components (using close-to-close for consistency with literature)
merged_dt[, `:=`(
  rv_daily = sqrt(var_cc * ANNUAL_FACTOR) * 100,  # 1-day
  rv_weekly = compute_backward_rv(var_cc, 5L, ANNUAL_FACTOR),   # 5-day
  rv_monthly = compute_backward_rv(var_cc, 22L, ANNUAL_FACTOR)  # 22-day
)]

#------------------------------------------------------------------
# 6. SUMMARY STATISTICS
#------------------------------------------------------------------

cat_progress("Computing RV summary statistics...")

rv_cols <- c("rv_cc", "rv_parkinson", "rv_gk", "rv_rs", "rv_yz", "vix_close")

rv_summary <- merged_dt[, lapply(.SD, function(x) {
  list(
    n = sum(!is.na(x)),
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    q25 = quantile(x, 0.25, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    q75 = quantile(x, 0.75, na.rm = TRUE),
    max = max(x, na.rm = TRUE),
    skew = moments::skewness(x, na.rm = TRUE),
    kurt = moments::kurtosis(x, na.rm = TRUE)
  )
}), .SDcols = rv_cols]

rv_summary_dt <- data.table(
  statistic = c("n", "mean", "sd", "min", "q25", "median", "q75", "max", "skew", "kurt"),
  sapply(rv_cols, function(col) unlist(rv_summary[[col]]))
)

print(rv_summary_dt)

# Correlation matrix between RV estimators and VIX
cat_progress("RV-VIX correlations:")
cor_matrix <- cor(merged_dt[, .SD, .SDcols = rv_cols], use = "pairwise.complete.obs")
print(round(cor_matrix, 3))

# Save correlation matrix
write.csv(round(cor_matrix, 4), file.path(dirs$tables, "02_rv_correlations.csv"))

#------------------------------------------------------------------
# 7. VRP SUMMARY
#------------------------------------------------------------------

cat_progress("VRP summary statistics...")

vrp_cols <- c("vrp_cc", "vrp_parkinson", "vrp_gk", "vrp_rs", "vrp_yz")

vrp_summary <- merged_dt[, lapply(.SD, function(x) {
  c(
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    pct_positive = 100 * mean(x > 0, na.rm = TRUE),
    t_stat = mean(x, na.rm = TRUE) / (sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))))
  )
}), .SDcols = vrp_cols]

print(t(vrp_summary))

# Save VRP summary
write.csv(t(vrp_summary), file.path(dirs$tables, "02_vrp_summary.csv"))

#------------------------------------------------------------------
# 8. DIAGNOSTIC PLOTS
#------------------------------------------------------------------

cat_progress("Creating diagnostic plots...")

# Common theme for publication
theme_pub <- theme_minimal() +
  theme(
    plot.title = element_text(size = 11, face = "bold"),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.grid.minor = element_blank()
  )

# --------------------------------------------------------------------------
# Plot 1: VIX vs RV_CC only (full sample)
# --------------------------------------------------------------------------



plot_vix_rv <- melt(
  
  merged_dt[, .(date, VIX = vix_close, `Realised Volatility` = rv_cc)],
  id.vars = "date",
  variable.name = "series",
  value.name = "volatility"
)

p_vix_rv_full <- ggplot(plot_vix_rv[!is.na(volatility)], 
                        aes(x = date, y = volatility, colour = series)) +
  
  geom_line(linewidth = 0.3, alpha = 0.85) +
  scale_colour_manual(values = c("VIX" = "#D62728", "Realised Volatility" = "#1F77B4")) +
  labs(title = "VIX vs 22-Day Forward Realised Volatility", 
       x = NULL, y = "Volatility (% annualised)") +
  theme_pub

ggsave(file.path(dirs$figures, "02a_vix_vs_rv_full.pdf"), 
       p_vix_rv_full, width = 10, height = 5)

# --------------------------------------------------------------------------
# Plot 2: VIX vs RV_CC (recent period: 2015+)
# --------------------------------------------------------------------------
p_vix_rv_recent <- ggplot(plot_vix_rv[plot_vix_rv$date >= as.Date("2015-01-01") & !is.na(plot_vix_rv$volatility), ], 
                          aes(x = date, y = volatility, colour = series)) +
  geom_line(linewidth = 0.4, alpha = 0.85) +
  scale_colour_manual(values = c("VIX" = "#D62728", "Realised Volatility" = "#1F77B4")) +
  labs(title = "VIX vs Realised Volatility (2015-Present)", 
       x = NULL, y = "Volatility (% annualised)") +
  theme_pub
ggsave(file.path(dirs$figures, "02b_vix_vs_rv_recent.pdf"), 
       p_vix_rv_recent, width = 10, height = 5)
# --------------------------------------------------------------------------
# Plot 3: VRP time series with regime shading
# --------------------------------------------------------------------------
vrp_mean <- mean(merged_dt$vrp_cc, na.rm = TRUE)
vrp_sd <- sd(merged_dt$vrp_cc, na.rm = TRUE)

p_vrp <- ggplot(merged_dt[!is.na(vrp_cc)], aes(x = date, y = vrp_cc)) +
  geom_hline(yintercept = 0, linetype = "solid", colour = "grey50", linewidth = 0.5) +
  geom_ribbon(aes(ymin = pmin(vrp_cc, 0), ymax = 0), fill = "#D62728", alpha = 0.3) +
  geom_ribbon(aes(ymin = 0, ymax = pmax(vrp_cc, 0)), fill = "#2CA02C", alpha = 0.3) +
  geom_line(colour = "grey20", linewidth = 0.3) +
  geom_hline(yintercept = vrp_mean, linetype = "dashed", colour = "#1F77B4", linewidth = 0.6) +
  annotate("text", x = min(merged_dt$date, na.rm = TRUE) + 500, y = vrp_mean + 1.5,
           label = sprintf("Mean = %.1f%%", vrp_mean), colour = "#1F77B4", size = 3) +
  labs(title = "Variance Risk Premium (VIX - RV)", 
       subtitle = "Green: VIX > RV (risk premium collected), Red: VIX < RV (premium paid)",
       x = NULL, y = "VRP (%)") +
  theme_pub

ggsave(file.path(dirs$figures, "02c_vrp_timeseries.pdf"), 
       p_vrp, width = 10, height = 5)

# --------------------------------------------------------------------------
# Plot 4: Scatter VIX vs RV with density
# --------------------------------------------------------------------------
p_scatter <- ggplot(merged_dt[!is.na(rv_cc) & !is.na(vix_close)], 
                    aes(x = rv_cc, y = vix_close)) +
  geom_point(alpha = 0.15, size = 0.8, colour = "grey30") +
  geom_abline(slope = 1, intercept = 0, colour = "#D62728", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "lm", colour = "#1F77B4", se = TRUE, linewidth = 0.8, alpha = 0.2) +
  labs(title = "VIX vs Realised Volatility", 
       subtitle = "Dashed red: 45° line (VIX = RV), Blue: OLS fit",
       x = "Realised Volatility (%)", y = "VIX (%)") +
  coord_fixed(xlim = c(0, 80), ylim = c(0, 80)) +
  theme_pub +
  theme(legend.position = "none")

ggsave(file.path(dirs$figures, "02d_vix_rv_scatter.pdf"), 
       p_scatter, width = 7, height = 7)

# --------------------------------------------------------------------------
# Plot 5: VRP distribution
# --------------------------------------------------------------------------
p_vrp_hist <- ggplot(merged_dt[!is.na(vrp_cc)], aes(x = vrp_cc)) +
  geom_histogram(aes(y = after_stat(density)), bins = 80, 
                 fill = "grey70", colour = "grey40", linewidth = 0.2) +
  geom_density(colour = "#1F77B4", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "#D62728", linewidth = 0.6) +
  geom_vline(xintercept = vrp_mean, linetype = "dashed", colour = "#2CA02C", linewidth = 0.6) +
  labs(title = "Distribution of Variance Risk Premium",
       subtitle = sprintf("Mean = %.2f%%, Median = %.2f%%, P(VRP > 0) = %.1f%%",
                          vrp_mean, median(merged_dt$vrp_cc, na.rm = TRUE),
                          100 * mean(merged_dt$vrp_cc > 0, na.rm = TRUE)),
       x = "VRP (%)", y = "Density") +
  theme_pub

ggsave(file.path(dirs$figures, "02e_vrp_distribution.pdf"), 
       p_vrp_hist, width = 8, height = 5)

# --------------------------------------------------------------------------
# Plot 6: RV estimator comparison (all estimators)
# --------------------------------------------------------------------------
rv_compare <- melt(
  merged_dt[, .(date, `Close-Close` = rv_cc, Parkinson = rv_parkinson, 
                `Garman-Klass` = rv_gk, `Yang-Zhang` = rv_yz)],
  id.vars = "date",
  variable.name = "Estimator",
  value.name = "rv"
)

p_rv_compare <- ggplot(subset(rv_compare, date >= as.Date("2018-01-01") & !is.na(rv)), 
                       aes(x = date, y = rv, colour = Estimator)) +
  geom_line(linewidth = 0.4, alpha = 0.8) +
  scale_colour_viridis_d(option = "D", end = 0.85) +
  labs(title = "Comparison of RV Estimators (2018-Present)", 
       x = NULL, y = "Realised Volatility (%)") +
  theme_pub

# --------------------------------------------------------------------------
# Plot 7: Combined overview (2x2)
# --------------------------------------------------------------------------
combined_plot <- gridExtra::grid.arrange(
  p_vix_rv_recent, p_vrp, p_scatter, p_vrp_hist, 
  ncol = 2, nrow = 2
)

ggsave(file.path(dirs$figures, "02_realised_volatility_overview.pdf"), 
       combined_plot, width = 14, height = 10)

# --------------------------------------------------------------------------
# Plot 8: ACF of RV and VRP (persistence diagnostics)
# --------------------------------------------------------------------------
# Compute ACF manually for ggplot compatibility
acf_rv <- acf(merged_dt$rv_cc[!is.na(merged_dt$rv_cc)], lag.max = 60, plot = FALSE)
acf_vrp <- acf(merged_dt$vrp_cc[!is.na(merged_dt$vrp_cc)], lag.max = 60, plot = FALSE)

acf_dt <- data.table(
  lag = rep(0:60, 2),
  acf = c(acf_rv$acf, acf_vrp$acf),
  series = rep(c("Realised Volatility", "VRP"), each = 61)
)

ci_bound <- 1.96 / sqrt(sum(!is.na(merged_dt$rv_cc)))

p_acf <- ggplot(acf_dt[lag > 0], aes(x = lag, y = acf, fill = series)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_hline(yintercept = c(-ci_bound, ci_bound), linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0, colour = "black") +
  scale_fill_manual(values = c("Realised Volatility" = "#1F77B4", "VRP" = "#D62728")) +
  labs(title = "Autocorrelation: RV vs VRP",
       subtitle = "RV exhibits strong persistence; VRP is less persistent (trading signal)",
       x = "Lag (days)", y = "ACF") +
  theme_pub

ggsave(file.path(dirs$figures, "02g_acf_rv_vrp.pdf"), 
       p_acf, width = 10, height = 5)

# --------------------------------------------------------------------------
# Plot 9: Rolling correlation VIX-RV (stability check)
# --------------------------------------------------------------------------
# frollapply doesn't support multivariate input; compute manually
roll_window <- 252L
n_obs <- nrow(merged_dt)

# Vectorised rolling correlation using rolling means and sds
merged_dt[, `:=`(
  roll_mean_vix = frollmean(vix_close, n = roll_window, align = "right"),
  roll_mean_rv = frollmean(rv_cc, n = roll_window, align = "right"),
  roll_sd_vix = frollapply(vix_close, n = roll_window, FUN = sd, align = "right"),
  roll_sd_rv = frollapply(rv_cc, n = roll_window, FUN = sd, align = "right"),
  roll_cov = frollapply(vix_close * rv_cc, n = roll_window, FUN = mean, align = "right") -
    frollmean(vix_close, n = roll_window, align = "right") * 
    frollmean(rv_cc, n = roll_window, align = "right")
)]

merged_dt[, roll_corr := roll_cov / (roll_sd_vix * roll_sd_rv)]

p_roll_cor <- ggplot(merged_dt[!is.na(roll_corr)], 
                     aes(x = date, y = roll_corr)) +
  geom_line(colour = "#1F77B4", linewidth = 0.4) +
  geom_hline(yintercept = cor(merged_dt$vix_close, merged_dt$rv_cc, use = "complete.obs"),
             linetype = "dashed", colour = "#D62728") +
  labs(title = "Rolling 1-Year Correlation: VIX vs RV",
       x = NULL, y = "Correlation") +
  ylim(0.5, 1) +
  theme_pub

ggsave(file.path(dirs$figures, "02h_rolling_correlation.pdf"), 
       p_roll_cor, width = 10, height = 4)

# Remove temporary columns
merged_dt[, c("roll_mean_vix", "roll_mean_rv", "roll_sd_vix", "roll_sd_rv", 
              "roll_cov", "roll_cor_vix_rv") := NULL]

# --------------------------------------------------------------------------
# Plot 10: Crisis period comparison (GFC, COVID, etc.)
# --------------------------------------------------------------------------
crisis_periods <- data.table(
  name = c("GFC", "Euro Crisis", "China/Oil", "COVID", "2022 Selloff"),
  start = as.Date(c("2008-09-01", "2011-07-01", "2015-08-01", "2020-02-01", "2022-01-01")),
  end = as.Date(c("2009-03-31", "2011-12-31", "2016-02-29", "2020-04-30", "2022-10-31"))
)

crisis_data <- lapply(1:nrow(crisis_periods), function(i) {
  merged_dt[date >= crisis_periods$start[i] & date <= crisis_periods$end[i],
            .(date, vix_close, rv_cc, vrp_cc, crisis = crisis_periods$name[i])]
})
crisis_data <- rbindlist(crisis_data)

p_crisis <- ggplot(crisis_data, aes(x = date)) +
  geom_line(aes(y = vix_close, colour = "VIX"), linewidth = 0.5) +
  geom_line(aes(y = rv_cc, colour = "RV"), linewidth = 0.5) +
  scale_colour_manual(values = c("VIX" = "#D62728", "RV" = "#1F77B4")) +
  facet_wrap(~crisis, scales = "free_x", ncol = 3) +
  labs(title = "VIX vs RV During Market Stress Episodes",
       x = NULL, y = "Volatility (%)") +
  theme_pub +
  theme(strip.text = element_text(face = "bold"))

ggsave(file.path(dirs$figures, "02i_crisis_comparison.pdf"), 
       p_crisis, width = 12, height = 6)

#------------------------------------------------------------------
# 9. SAVE DATA TO EXCEL (SEPARATE SHEETS PER RV ESTIMATOR)
#------------------------------------------------------------------
cat_progress("Saving data with RV columns...")

# Select columns to keep
cols_to_keep <- c(
  # Identifiers
  "date",
  # SPX data
  "open", "high", "low", "close", "volume", "adj_close",
  "log_return", "simple_return",
  # VIX data
  "vix_open", "vix_high", "vix_low", "vix_close", "vix_return", "vix_change",
  # Forward RV (for VRP)
  "rv_cc", "rv_parkinson", "rv_gk", "rv_rs", "rv_yz",
  # VRP
  "vrp_cc", "vrp_parkinson", "vrp_gk", "vrp_rs", "vrp_yz",
  # HAR-RV features (backward-looking)
  "rv_daily", "rv_weekly", "rv_monthly"
)

final_dt <- merged_dt[, .SD, .SDcols = cols_to_keep]

# Save
saveRDS(final_dt, file.path(dirs$data, "data_with_rv.rds"))
write.csv(rv_summary_dt, file.path(dirs$tables, "02_rv_summary.csv"), row.names = FALSE)

cat_progress(sprintf("Data saved: %s", file.path(dirs$data, "data_with_rv.rds")))
cat_progress(sprintf("RV summary: %s", file.path(dirs$tables, "02_rv_summary.csv")))
cat_progress("Saving RV data to Excel with separate sheets...")

# Load openxlsx (add to Setup.R if not already there)
if (!requireNamespace("openxlsx", quietly = TRUE)) {
  install.packages("openxlsx")
}
library(openxlsx)

# Create workbook
wb <- createWorkbook()

# Common columns for all sheets
common_cols <- c("date", "open", "high", "low", "close", "volume", "adj_close",
                 "log_return", "simple_return", 
                 "vix_open", "vix_high", "vix_low", "vix_close", "vix_return", "vix_change")

# --------------------------------------------------------------------------
# Sheet 1: Close-to-Close
# --------------------------------------------------------------------------
addWorksheet(wb, "RV_CloseClose")
dt_cc <- merged_dt[, c(common_cols, "var_cc", "rv_cc", "vrp_cc", 
                       "rv_daily", "rv_weekly", "rv_monthly"), with = FALSE]
writeData(wb, "RV_CloseClose", dt_cc)

# --------------------------------------------------------------------------
# Sheet 2: Parkinson
# --------------------------------------------------------------------------
addWorksheet(wb, "RV_Parkinson")
dt_parkinson <- merged_dt[, c(common_cols, "var_parkinson", "rv_parkinson", "vrp_parkinson"), with = FALSE]
writeData(wb, "RV_Parkinson", dt_parkinson)

# --------------------------------------------------------------------------
# Sheet 3: Garman-Klass
# --------------------------------------------------------------------------
addWorksheet(wb, "RV_GarmanKlass")
dt_gk <- merged_dt[, c(common_cols, "var_gk", "rv_gk", "vrp_gk"), with = FALSE]
writeData(wb, "RV_GarmanKlass", dt_gk)

# --------------------------------------------------------------------------
# Sheet 4: Rogers-Satchell
# --------------------------------------------------------------------------
addWorksheet(wb, "RV_RogersSatchell")
dt_rs <- merged_dt[, c(common_cols, "var_rs", "rv_rs", "vrp_rs"), with = FALSE]
writeData(wb, "RV_RogersSatchell", dt_rs)

# --------------------------------------------------------------------------
# Sheet 5: Yang-Zhang
# --------------------------------------------------------------------------
addWorksheet(wb, "RV_YangZhang")
dt_yz <- merged_dt[, c(common_cols, "log_oc", "log_co", "var_overnight", "var_yz", 
                       "rv_yz", "vrp_yz"), with = FALSE]
writeData(wb, "RV_YangZhang", dt_yz)

# --------------------------------------------------------------------------
# Sheet 6: Summary statistics
# --------------------------------------------------------------------------
addWorksheet(wb, "Summary")
writeData(wb, "Summary", rv_summary_dt)

# --------------------------------------------------------------------------
# Sheet 7: Correlation matrix
# --------------------------------------------------------------------------
addWorksheet(wb, "Correlations")
cor_df <- as.data.frame(round(cor_matrix, 4))
cor_df <- cbind(Estimator = rownames(cor_df), cor_df)
writeData(wb, "Correlations", cor_df)

# --------------------------------------------------------------------------
# Sheet 8: VRP summary
# --------------------------------------------------------------------------
addWorksheet(wb, "VRP_Summary")
vrp_summary_df <- as.data.frame(t(vrp_summary))
vrp_summary_df <- cbind(Estimator = rownames(vrp_summary_df), vrp_summary_df)
writeData(wb, "VRP_Summary", vrp_summary_df)

# --------------------------------------------------------------------------
# Sheet 9: All estimators combined (wide format)
# --------------------------------------------------------------------------
addWorksheet(wb, "All_Estimators")
dt_all <- merged_dt[, .(
  date,
  vix_close,
  rv_cc, rv_parkinson, rv_gk, rv_rs, rv_yz,
  vrp_cc, vrp_parkinson, vrp_gk, vrp_rs, vrp_yz
)]
writeData(wb, "All_Estimators", dt_all)

# Save workbook
excel_path <- file.path(dirs$tables, "02_realised_volatility_all.xlsx")
saveWorkbook(wb, excel_path, overwrite = TRUE)

cat_progress(sprintf("Excel workbook saved: %s", excel_path))
cat_progress("  Sheets: RV_CloseClose, RV_Parkinson, RV_GarmanKlass, RV_RogersSatchell, RV_YangZhang")
cat_progress("  Sheets: Summary, Correlations, VRP_Summary, All_Estimators")

# Cleanup temporary data.tables
rm(dt_cc, dt_parkinson, dt_gk, dt_rs, dt_yz, dt_all, cor_df, vrp_summary_df, wb)
#------------------------------------------------------------------
# 10. CLEANUP
#------------------------------------------------------------------

rm(plot_vix_rv, rv_compare, acf_dt, acf_rv, acf_vrp, ci_bound,
   crisis_periods, crisis_data, p_crisis,
   p_vix_rv_full, p_vix_rv_recent, p_vrp, p_scatter, p_vrp_hist, p_rv_compare,
   p_acf, p_roll_cor, combined_plot, vrp_mean, vrp_sd)
gc()

cat_progress("Realised volatility computation complete.")
cat_progress(sprintf("  Forward RV window: %d trading days", RV_WINDOW))
cat_progress(sprintf("  Annualisation: %d days", ANNUAL_FACTOR))
cat_progress(sprintf("  Estimators: CC, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang"))
cat_progress("Figures saved:")
cat_progress("  - 02a_vix_vs_rv_full.pdf")
cat_progress("  - 02b_vix_vs_rv_recent.pdf")
cat_progress("  - 02c_vrp_timeseries.pdf")
cat_progress("  - 02d_vix_rv_scatter.pdf")
cat_progress("  - 02e_vrp_distribution.pdf")
cat_progress("  - 02f_rv_estimator_comparison.pdf")
cat_progress("  - 02g_acf_rv_vrp.pdf")
cat_progress("  - 02h_rolling_correlation.pdf")
cat_progress("  - 02i_crisis_comparison.pdf")
cat_progress("  - 02_realised_volatility_overview.pdf")

#------------------------------------------------------------------
# END OF SCRIPT
#------------------------------------------------------------------