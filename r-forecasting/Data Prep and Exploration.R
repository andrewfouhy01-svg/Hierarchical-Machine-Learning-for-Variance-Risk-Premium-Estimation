################################################################################
# 00_data_prep_and_exploration.R
################################################################################

#------------------------------------------------------------------
# 0. SOURCE SETUP AND LOAD DATA
#------------------------------------------------------------------

source("Setup.R")

# Load strucchange for Bai-Perron test (not in core_packages)
if (!require("strucchange", quietly = TRUE)) {
  install.packages("strucchange", dependencies = TRUE)
  library(strucchange)
}

# Load merged data from Data_Download.R
merged_dt <- readRDS("data/merged_data.rds")

# Convert to xts objects for compatibility with downstream analysis
vix_xts <- xts(merged_dt$vix_close, order.by = merged_dt$date)
colnames(vix_xts) <- "vix_close"

spx_xts <- xts(merged_dt$close, order.by = merged_dt$date)
colnames(spx_xts) <- "spx_close"

# Compute log returns
vix_returns_xts <- diff(log(vix_xts))
colnames(vix_returns_xts) <- "vix_return"
vix_returns_xts <- na.omit(vix_returns_xts)

spx_returns_xts <- diff(log(spx_xts))
colnames(spx_returns_xts) <- "spx_return"
spx_returns_xts <- na.omit(spx_returns_xts)

#------------------------------------------------------------------
# 1. Basic Data Exploration
#------------------------------------------------------------------

cat("\n================== VIX DATA ==================\n")
cat("\nVIX Levels (vix_xts):\n")
cat("  Start date:", as.character(start(vix_xts)), "\n")
cat("  End date:", as.character(end(vix_xts)), "\n")
cat("  VIX min:", min(vix_xts, na.rm = TRUE), "\n")
cat("  VIX max:", max(vix_xts, na.rm = TRUE), "\n")
cat("  VIX mean:", mean(vix_xts, na.rm = TRUE), "\n")
cat("  VIX median:", median(vix_xts, na.rm = TRUE), "\n")

cat("\n================== S&P 500 DATA ==================\n")
cat("\nS&P 500 Levels (spx_xts):\n")
cat("  Start date:", as.character(start(spx_xts)), "\n")
cat("  End date:", as.character(end(spx_xts)), "\n")
cat("  SPX min:", min(spx_xts, na.rm = TRUE), "\n")
cat("  SPX max:", max(spx_xts, na.rm = TRUE), "\n")
cat("  SPX mean:", mean(spx_xts, na.rm = TRUE), "\n")
cat("  SPX median:", median(spx_xts, na.rm = TRUE), "\n")

#------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT 
#------------------------------------------------------------------

total_years <- as.numeric(difftime(end(vix_xts), start(vix_xts), units = "days")) / 365.25
total_years
cat("\nTotal years available:", round(total_years, 2), "\n")

train_prop <- 21.5/ total_years

# Calculate split date
split_index <- floor(nrow(vix_xts) * train_prop)
split_date <- index(vix_xts)[split_index]
cat("  Split date:", as.character(split_date), "\n")
cat("  Training observations:", split_index, "\n")
cat("  Testing observations:", nrow(vix_xts) - split_index, "\n")

# VIX splits
vix_train <- vix_xts[paste0("/", split_date)]
vix_test <- vix_xts[paste0(split_date, "/")]
vix_returns_train <- vix_returns_xts[paste0("/", split_date)]
vix_returns_test <- vix_returns_xts[paste0(split_date, "/")]

# SPX splits
spx_train <- spx_xts[paste0("/", split_date)]
spx_test <- spx_xts[paste0(split_date, "/")]
spx_returns_train <- spx_returns_xts[paste0("/", split_date)]
spx_returns_test <- spx_returns_xts[paste0(split_date, "/")]

#Save split info for data preservation
split_info <- list(
  split_date = split_date,
  train_start = start(vix_train),
  train_end = end(vix_train),
  test_start = start(vix_test),
  test_end = end(vix_test),
  train_n = nrow(vix_train),
  test_n = nrow(vix_test)
)
saveRDS(split_info, "results/models/split_info.rds")

#------------------------------------------------------------------
# 3. TARGET DEFINITION
#------------------------------------------------------------------

vix_train_df <- data.frame(
  date = index(vix_train),
  vix = as.numeric(vix_train),
  vix_next = as.numeric(stats::lag(vix_train, -1))
)
vix_train_df$change <- vix_train_df$vix_next - vix_train_df$vix
vix_train_df$pct_change <- (vix_train_df$vix_next - vix_train_df$vix) / vix_train_df$vix * 100

thresholds <- c(0, 0.1, 0.5)
class_balance <- data.frame(
  Threshold = character(),
  Up_Count = numeric(),
  Down_Count = numeric(),
  Up_Pct = numeric(),
  Down_Pct = numeric(),
  stringsAsFactors = FALSE
)

for (thresh in thresholds) {
  if (thresh == 0) {
    up_count <- sum(vix_train_df$vix_next > vix_train_df$vix, na.rm = TRUE)
    down_count <- sum(vix_train_df$vix_next <= vix_train_df$vix, na.rm = TRUE)
    label <- "0% (raw)"
  } else {
    up_count <- sum(vix_train_df$pct_change > thresh, na.rm = TRUE)
    down_count <- sum(vix_train_df$pct_change <= -thresh, na.rm = TRUE)
    label <- paste0(thresh, "%")
  }
  
  total <- up_count + down_count
  class_balance <- rbind(class_balance, data.frame(
    Threshold = label,
    Up_Count = up_count,
    Down_Count = down_count,
    Up_Pct = round(up_count / total * 100, 2),
    Down_Pct = round(down_count / total * 100, 2)
  ))
}

print(class_balance)

#Calculate scale_pos_weight for use in XGBoost model 
n_down <- class_balance$Down_Count[1]
n_up <- class_balance$Up_Count[1]
scale_pos_weight <- n_down / n_up

#Save it 
target_config <- list(
  classification_threshold = 0,
  class_balance = class_balance,
  scale_pos_weight = scale_pos_weight,
  regression_target = "next_day_level"
)
saveRDS(target_config, "results/models/target_config.rds")

#------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
#------------------------------------------------------------------

desc_stats <- function(x, name) {
  cat("\n", name, ":\n", sep = "")
  cat("  N:", length(na.omit(x)), "\n")
  cat("  Mean:", mean(x, na.rm = TRUE), "\n")
  cat("  Median:", median(x, na.rm = TRUE), "\n")
  cat("  SD:", sd(x, na.rm = TRUE), "\n")
  cat("  Min:", min(x, na.rm = TRUE), "\n")
  cat("  Max:", max(x, na.rm = TRUE), "\n")
  cat("  Skewness:", moments::skewness(x, na.rm = TRUE), "\n")
  cat("  Kurtosis:", moments::kurtosis(x, na.rm = TRUE), "\n")
  cat("  Q25:", quantile(x, 0.25, na.rm = TRUE), "\n")
  cat("  Q75:", quantile(x, 0.75, na.rm = TRUE), "\n")
}

cat("\n================== VIX DESCRIPTIVE STATS ==================\n")
desc_stats(vix_train, "VIX Levels (Training)")
desc_stats(vix_returns_train, "VIX Returns (Training)")

cat("\n================== S&P 500 DESCRIPTIVE STATS ==================\n")
desc_stats(spx_train, "S&P 500 Levels (Training)")
desc_stats(spx_returns_train, "S&P 500 Returns (Training)")

#------------------------------------------------------------------
# 5. STATISTICAL TESTS
#------------------------------------------------------------------

#Set up results df
test_results <- data.frame(
  Series = character(),
  Test = character(),
  Statistic = character(),
  P_Value = character(),
  Interpretation = character(),
  stringsAsFactors = FALSE
)

#Helper function to add results
add_test <- function(series, name, stat, pval, interp) {
  test_results <<- rbind(test_results, data.frame(
    Series = series,
    Test = name,
    Statistic = stat,
    P_Value = pval,
    Interpretation = interp,
    stringsAsFactors = FALSE
  ))
}

#------------------------------------------------------------------
# 5.1 Stationarity Tests - VIX
#------------------------------------------------------------------


getAnywhere(adf.test)
adf_vix_levels <- tseries::adf.test(as.numeric(vix_train), k = 50)
adf_vix_returns <- tseries::adf.test(as.numeric(vix_returns_train), k = 50)

cat("    VIX Levels: statistic =", adf_vix_levels$statistic, 
    ", p-value =", adf_vix_levels$p.value, "\n")
cat("    VIX Returns: statistic =", adf_vix_returns$statistic, 
    ", p-value =", adf_vix_returns$p.value, "\n")

add_test("VIX", "ADF (Levels)", as.character(adf_vix_levels$statistic),
         as.character(adf_vix_levels$p.value),
         ifelse(adf_vix_levels$p.value < 0.05, "Stationary", "Non-stationary"))

add_test("VIX", "ADF (Returns)", as.character(adf_vix_returns$statistic),
         as.character(adf_vix_returns$p.value),
         ifelse(adf_vix_returns$p.value < 0.05, "Stationary", "Non-stationary"))

#KPSS Test - VIX
kpss_vix_levels <- tseries::kpss.test(as.numeric(vix_train), null = "Trend")
kpss_vix_returns <- tseries::kpss.test(as.numeric(vix_returns_train), null = "Level")

cat("    KPSS VIX Levels: statistic =", kpss_vix_levels$statistic, 
    ", p-value =", kpss_vix_levels$p.value, "\n")
cat("    KPSS VIX Returns: statistic =", kpss_vix_returns$statistic, 
    ", p-value =", kpss_vix_returns$p.value, "\n")

add_test("VIX", "KPSS (Levels)", as.character(kpss_vix_levels$statistic),
         as.character(kpss_vix_levels$p.value),
         ifelse(kpss_vix_levels$p.value >= 0.05, "Stationary", "Non-stationary"))

add_test("VIX", "KPSS (Returns)", as.character(kpss_vix_returns$statistic),
         as.character(kpss_vix_returns$p.value),
         ifelse(kpss_vix_returns$p.value >= 0.05, "Stationary", "Non-stationary"))

#------------------------------------------------------------------
# 5.1 Stationarity Tests - S&P 500
#------------------------------------------------------------------


adf_spx_levels <- tseries::adf.test(as.numeric(spx_train), k = 50)
adf_spx_returns <- tseries::adf.test(as.numeric(spx_returns_train), k = 50)

cat("    SPX Levels: statistic =", adf_spx_levels$statistic, 
    ", p-value =", adf_spx_levels$p.value, "\n")
cat("    SPX Returns: statistic =", adf_spx_returns$statistic, 
    ", p-value =", adf_spx_returns$p.value, "\n")

add_test("SPX", "ADF (Levels)", as.character(adf_spx_levels$statistic),
         as.character(adf_spx_levels$p.value),
         ifelse(adf_spx_levels$p.value < 0.05, "Stationary", "Non-stationary"))

add_test("SPX", "ADF (Returns)", as.character(adf_spx_returns$statistic),
         as.character(adf_spx_returns$p.value),
         ifelse(adf_spx_returns$p.value < 0.05, "Stationary", "Non-stationary"))

#KPSS Test - SPX
kpss_spx_levels <- tseries::kpss.test(as.numeric(spx_train), null = "Trend")
kpss_spx_returns <- tseries::kpss.test(as.numeric(spx_returns_train), null = "Level")

cat("    KPSS SPX Levels: statistic =", kpss_spx_levels$statistic, 
    ", p-value =", kpss_spx_levels$p.value, "\n")
cat("    KPSS SPX Returns: statistic =", kpss_spx_returns$statistic, 
    ", p-value =", kpss_spx_returns$p.value, "\n")

add_test("SPX", "KPSS (Levels)", as.character(kpss_spx_levels$statistic),
         as.character(kpss_spx_levels$p.value),
         ifelse(kpss_spx_levels$p.value >= 0.05, "Stationary", "Non-stationary"))

add_test("SPX", "KPSS (Returns)", as.character(kpss_spx_returns$statistic),
         as.character(kpss_spx_returns$p.value),
         ifelse(kpss_spx_returns$p.value >= 0.05, "Stationary", "Non-stationary"))

#------------------------------------------------------------------
# 5.2 Autocorrelation Tests - VIX
#------------------------------------------------------------------

lb_vix_levels <- Box.test(as.numeric(vix_train), lag = 20, type = "Ljung-Box")
lb_vix_returns <- Box.test(as.numeric(vix_returns_train), lag = 20, type = "Ljung-Box")
lb_vix_squared <- Box.test(as.numeric(vix_returns_train)^2, lag = 20, type = "Ljung-Box")

cat("    VIX Levels: statistic =", round(lb_vix_levels$statistic, 4), 
    ", p-value =", format(lb_vix_levels$p.value, scientific = TRUE), "\n")
cat("    VIX Returns: statistic =", round(lb_vix_returns$statistic, 4), 
    ", p-value =", format(lb_vix_returns$p.value, scientific = TRUE), "\n")
cat("    VIX Squared Returns: statistic =", round(lb_vix_squared$statistic, 4), 
    ", p-value =", format(lb_vix_squared$p.value, scientific = TRUE), "\n")

add_test("VIX", "Ljung-Box (Levels)", as.character(round(lb_vix_levels$statistic, 4)),
         format(lb_vix_levels$p.value, scientific = TRUE),
         ifelse(lb_vix_levels$p.value < 0.05, "Autocorrelation present", "No autocorrelation"))

add_test("VIX", "Ljung-Box (Returns)", as.character(round(lb_vix_returns$statistic, 4)),
         format(lb_vix_returns$p.value, scientific = TRUE),
         ifelse(lb_vix_returns$p.value < 0.05, "Autocorrelation present", "No autocorrelation"))

add_test("VIX", "Ljung-Box (Squared Returns)", as.character(round(lb_vix_squared$statistic, 4)),
         format(lb_vix_squared$p.value, scientific = TRUE),
         ifelse(lb_vix_squared$p.value < 0.05, "Volatility clustering", "No volatility clustering"))

#------------------------------------------------------------------
# 5.2 Autocorrelation Tests - S&P 500
#------------------------------------------------------------------

lb_spx_levels <- Box.test(as.numeric(spx_train), lag = 20, type = "Ljung-Box")
lb_spx_returns <- Box.test(as.numeric(spx_returns_train), lag = 20, type = "Ljung-Box")
lb_spx_squared <- Box.test(as.numeric(spx_returns_train)^2, lag = 20, type = "Ljung-Box")

cat("    SPX Levels: statistic =", round(lb_spx_levels$statistic, 4), 
    ", p-value =", format(lb_spx_levels$p.value, scientific = TRUE), "\n")
cat("    SPX Returns: statistic =", round(lb_spx_returns$statistic, 4), 
    ", p-value =", format(lb_spx_returns$p.value, scientific = TRUE), "\n")
cat("    SPX Squared Returns: statistic =", round(lb_spx_squared$statistic, 4), 
    ", p-value =", format(lb_spx_squared$p.value, scientific = TRUE), "\n")

add_test("SPX", "Ljung-Box (Levels)", as.character(round(lb_spx_levels$statistic, 4)),
         format(lb_spx_levels$p.value, scientific = TRUE),
         ifelse(lb_spx_levels$p.value < 0.05, "Autocorrelation present", "No autocorrelation"))

add_test("SPX", "Ljung-Box (Returns)", as.character(round(lb_spx_returns$statistic, 4)),
         format(lb_spx_returns$p.value, scientific = TRUE),
         ifelse(lb_spx_returns$p.value < 0.05, "Autocorrelation present", "No autocorrelation"))

add_test("SPX", "Ljung-Box (Squared Returns)", as.character(round(lb_spx_squared$statistic, 4)),
         format(lb_spx_squared$p.value, scientific = TRUE),
         ifelse(lb_spx_squared$p.value < 0.05, "Volatility clustering", "No volatility clustering"))

#------------------------------------------------------------------
# 5.3 Heteroskedasticity Test - VIX
#------------------------------------------------------------------

tryCatch({
  arch_vix <- FinTS::ArchTest(as.numeric(vix_returns_train), lags = 12)
  cat("    VIX Chi-squared =", round(arch_vix$statistic, 4), 
      ", p-value =", format(arch_vix$p.value, scientific = TRUE), "\n")
  
  add_test("VIX", "ARCH-LM", as.character(round(arch_vix$statistic, 4)),
           format(arch_vix$p.value, scientific = TRUE),
           ifelse(arch_vix$p.value < 0.05, "ARCH effects present", "No ARCH effects"))
}, error = function(e) {
  cat("    Error running ARCH-LM test (VIX):", e$message, "\n")
})

#------------------------------------------------------------------
# 5.3 Heteroskedasticity Test - S&P 500
#------------------------------------------------------------------

tryCatch({
  arch_spx <- FinTS::ArchTest(as.numeric(spx_returns_train), lags = 12)
  cat("    SPX Chi-squared =", round(arch_spx$statistic, 4), 
      ", p-value =", format(arch_spx$p.value, scientific = TRUE), "\n")
  
  add_test("SPX", "ARCH-LM", as.character(round(arch_spx$statistic, 4)),
           format(arch_spx$p.value, scientific = TRUE),
           ifelse(arch_spx$p.value < 0.05, "ARCH effects present", "No ARCH effects"))
}, error = function(e) {
  cat("    Error running ARCH-LM test (SPX):", e$message, "\n")
})

#------------------------------------------------------------------
# 5.4 Normality Tests - VIX
#------------------------------------------------------------------

jb_vix_levels <- jarque.bera.test(as.numeric(na.omit(vix_train)))
jb_vix_returns <- jarque.bera.test(as.numeric(na.omit(vix_returns_train)))

cat("    VIX Levels: statistic =", round(jb_vix_levels$statistic, 4), 
    ", p-value =", format(jb_vix_levels$p.value, scientific = TRUE), "\n")
cat("    VIX Returns: statistic =", round(jb_vix_returns$statistic, 4), 
    ", p-value =", format(jb_vix_returns$p.value, scientific = TRUE), "\n")

add_test("VIX", "Jarque-Bera (Levels)", as.character(round(jb_vix_levels$statistic, 4)),
         format(jb_vix_levels$p.value, scientific = TRUE),
         ifelse(jb_vix_levels$p.value >= 0.05, "Normal", "Non-normal"))

add_test("VIX", "Jarque-Bera (Returns)", as.character(round(jb_vix_returns$statistic, 4)),
         format(jb_vix_returns$p.value, scientific = TRUE),
         ifelse(jb_vix_returns$p.value >= 0.05, "Normal", "Non-normal"))

#------------------------------------------------------------------
# 5.4 Normality Tests - S&P 500
#------------------------------------------------------------------

jb_spx_levels <- jarque.bera.test(as.numeric(na.omit(spx_train)))
jb_spx_returns <- jarque.bera.test(as.numeric(na.omit(spx_returns_train)))

cat("    SPX Levels: statistic =", round(jb_spx_levels$statistic, 4), 
    ", p-value =", format(jb_spx_levels$p.value, scientific = TRUE), "\n")
cat("    SPX Returns: statistic =", round(jb_spx_returns$statistic, 4), 
    ", p-value =", format(jb_spx_returns$p.value, scientific = TRUE), "\n")

add_test("SPX", "Jarque-Bera (Levels)", as.character(round(jb_spx_levels$statistic, 4)),
         format(jb_spx_levels$p.value, scientific = TRUE),
         ifelse(jb_spx_levels$p.value >= 0.05, "Normal", "Non-normal"))

add_test("SPX", "Jarque-Bera (Returns)", as.character(round(jb_spx_returns$statistic, 4)),
         format(jb_spx_returns$p.value, scientific = TRUE),
         ifelse(jb_spx_returns$p.value >= 0.05, "Normal", "Non-normal"))

#------------------------------------------------------------------
# 5.5 Structural Breaks - VIX
#------------------------------------------------------------------

tryCatch({
  bp_vix <- strucchange::breakpoints(as.numeric(vix_train) ~ 1, h = 0.25)
  n_breaks_vix <- length(bp_vix$breakpoints)
  
  if (n_breaks_vix > 0 && !is.na(bp_vix$breakpoints[1])) {
    break_dates_vix <- index(vix_train)[bp_vix$breakpoints]
    cat("    VIX Number of breaks detected:", n_breaks_vix, "\n")
    cat("    VIX Break dates:\n")
    print(break_dates_vix)
    
    add_test("VIX", "Bai-Perron", paste(n_breaks_vix, "breaks"), "N/A",
             paste("Breaks at:", paste(as.character(break_dates_vix), collapse = ", ")))
  } else {
    add_test("VIX", "Bai-Perron", "0 breaks", "N/A", "No breaks detected")
  }
}, error = function(e) {
  cat("    Error running Bai-Perron test (VIX):", e$message, "\n")
})

#------------------------------------------------------------------
# 5.5 Structural Breaks - S&P 500
#------------------------------------------------------------------

tryCatch({
  bp_spx <- strucchange::breakpoints(as.numeric(spx_train) ~ 1, h = 0.3)
  n_breaks_spx <- length(bp_spx$breakpoints)
  
  if (n_breaks_spx > 0 && !is.na(bp_spx$breakpoints[1])) {
    break_dates_spx <- index(spx_train)[bp_spx$breakpoints]
    cat("    SPX Number of breaks detected:", n_breaks_spx, "\n")
    cat("    SPX Break dates:\n")
    print(break_dates_spx)
    
    add_test("SPX", "Bai-Perron", paste(n_breaks_spx, "breaks"), "N/A",
             paste("Breaks at:", paste(as.character(break_dates_spx), collapse = ", ")))
  } else {
    add_test("SPX", "Bai-Perron", "0 breaks", "N/A", "No breaks detected")
  }
}, error = function(e) {
  cat("    Error running Bai-Perron test (SPX):", e$message, "\n")
})

#------------------------------------------------------------------
# 5.6 Hurst Exponent - VIX
#------------------------------------------------------------------

tryCatch({
  hurst_vix <- hurstexp(as.numeric(na.omit(vix_returns_train)), display = FALSE)
  cat("    VIX Hurst exponent (R/S):", round(hurst_vix$Hs, 4), "\n")
  cat("    Interpretation: H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending\n")
  
  interpretation_vix <- ifelse(hurst_vix$Hs < 0.5, "Mean-reverting",
                               ifelse(hurst_vix$Hs > 0.5, "Trending", "Random walk"))
  
  add_test("VIX", "Hurst Exponent", as.character(round(hurst_vix$Hs, 4)), "N/A", interpretation_vix)
}, error = function(e) {
  cat("    Error calculating Hurst exponent (VIX):", e$message, "\n")
})

#------------------------------------------------------------------
# 5.6 Hurst Exponent - S&P 500
#------------------------------------------------------------------

tryCatch({
  hurst_spx <- hurstexp(as.numeric(na.omit(spx_returns_train)), display = FALSE)
  cat("    SPX Hurst exponent (R/S):", round(hurst_spx$Hs, 4), "\n")
  cat("    Interpretation: H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending\n")
  
  interpretation_spx <- ifelse(hurst_spx$Hs < 0.5, "Mean-reverting",
                               ifelse(hurst_spx$Hs > 0.5, "Trending", "Random walk"))
  
  add_test("SPX", "Hurst Exponent", as.character(round(hurst_spx$Hs, 4)), "N/A", interpretation_spx)
}, error = function(e) {
  cat("    Error calculating Hurst exponent (SPX):", e$message, "\n")
})

#------------------------------------------------------------------
# 5.7 VIX-SPX Correlation Analysis
#------------------------------------------------------------------


# Contemporaneous correlation
cor_levels <- cor(as.numeric(vix_train), as.numeric(spx_train), use = "complete.obs")
cor_returns <- cor(as.numeric(vix_returns_train), as.numeric(spx_returns_train), use = "complete.obs")

cat("    Correlation (Levels):", round(cor_levels, 4), "\n")
cat("    Correlation (Returns):", round(cor_returns, 4), "\n")

add_test("VIX-SPX", "Correlation (Levels)", as.character(round(cor_levels, 4)), "N/A",
         ifelse(cor_levels < 0, "Negative correlation", "Positive correlation"))
add_test("VIX-SPX", "Correlation (Returns)", as.character(round(cor_returns, 4)), "N/A",
         ifelse(cor_returns < 0, "Negative correlation (leverage effect)", "Positive correlation"))

# Asymmetric correlation (down vs up days)
spx_down <- as.numeric(spx_returns_train) < 0
cor_down <- cor(as.numeric(vix_returns_train)[spx_down], 
                as.numeric(spx_returns_train)[spx_down], use = "complete.obs")
cor_up <- cor(as.numeric(vix_returns_train)[!spx_down], 
              as.numeric(spx_returns_train)[!spx_down], use = "complete.obs")

cat("    Correlation (SPX down days):", round(cor_down, 4), "\n")
cat("    Correlation (SPX up days):", round(cor_up, 4), "\n")

add_test("VIX-SPX", "Correlation (Down days)", as.character(round(cor_down, 4)), "N/A",
         "Asymmetric leverage")
add_test("VIX-SPX", "Correlation (Up days)", as.character(round(cor_up, 4)), "N/A",
         "Asymmetric leverage")

#Save test results
write.csv(test_results, "results/tables/statistical_tests.csv", row.names = FALSE)
cat("\nStatistical tests saved to results/tables/statistical_tests.csv\n")

#------------------------------------------------------------------
# 6. Plotting
#------------------------------------------------------------------

#------------------------------------------------------------------
# 6.1 VIX Time series plots
#------------------------------------------------------------------

pdf("results/figures/02_vix_timeseries_plots.pdf", width = 12, height = 8)

par(mfrow = c(4, 1), mar = c(4, 4, 2, 1))
plot(vix_train, main = "VIX Levels - Training Set", 
     ylab = "VIX", col = "blue", lwd = 0.5)
abline(h = mean(vix_train, na.rm = TRUE), col = "red", lty = 2)

plot(vix_returns_train, main = "VIX Returns - Training Set", 
     ylab = "Returns", col = "darkgreen", lwd = 0.5)
abline(h = 0, col = "red", lty = 2)

plot(vix_xts, main = "VIX Levels (Full Sample)", 
     ylab = "VIX", col = "blue", lwd = 0.5)
abline(h = mean(vix_xts, na.rm = TRUE), col = "red", lty = 2)

plot(vix_returns_xts, main = "VIX Returns (Full Sample)", 
     ylab = "Returns", col = "darkgreen", lwd = 0.5)
abline(h = 0, col = "red", lty = 2)

dev.off()

#------------------------------------------------------------------
# 6.2 S&P 500 Time series plots
#------------------------------------------------------------------

pdf("results/figures/03_spx_timeseries_plots.pdf", width = 12, height = 8)

par(mfrow = c(4, 1), mar = c(4, 4, 2, 1))
plot(spx_train, main = "S&P 500 Levels - Training Set", 
     ylab = "SPX", col = "darkblue", lwd = 0.5)
abline(h = mean(spx_train, na.rm = TRUE), col = "red", lty = 2)

plot(spx_returns_train, main = "S&P 500 Returns - Training Set", 
     ylab = "Returns", col = "darkgreen", lwd = 0.5)
abline(h = 0, col = "red", lty = 2)

plot(spx_xts, main = "S&P 500 Levels (Full Sample)", 
     ylab = "SPX", col = "darkblue", lwd = 0.5)
abline(h = mean(spx_xts, na.rm = TRUE), col = "red", lty = 2)

plot(spx_returns_xts, main = "S&P 500 Returns (Full Sample)", 
     ylab = "Returns", col = "darkgreen", lwd = 0.5)
abline(h = 0, col = "red", lty = 2)

dev.off()

#------------------------------------------------------------------
# 6.3 VIX Distribution plots
#------------------------------------------------------------------

pdf("results/figures/04_vix_distributions.pdf", width = 12, height = 8)

par(mfrow = c(2, 2))

hist(as.numeric(vix_xts), breaks = 50, 
     main = "VIX Levels Distribution", xlab = "VIX", col = "lightblue")

hist(as.numeric(vix_returns_xts), breaks = 50, 
     main = "VIX Returns Distribution", xlab = "Returns", col = "lightgreen")

qqnorm(as.numeric(vix_xts), main = "Q-Q Plot: VIX Levels")
qqline(as.numeric(vix_xts), col = "red")

qqnorm(as.numeric(vix_returns_xts), main = "Q-Q Plot: VIX Returns")
qqline(as.numeric(vix_returns_xts), col = "red")

dev.off()

#------------------------------------------------------------------
# 6.4 S&P 500 Distribution plots
#------------------------------------------------------------------

pdf("results/figures/05_spx_distributions.pdf", width = 12, height = 8)

par(mfrow = c(2, 2))

hist(as.numeric(spx_xts), breaks = 50, 
     main = "S&P 500 Levels Distribution", xlab = "SPX", col = "lightblue")

hist(as.numeric(spx_returns_xts), breaks = 50, 
     main = "S&P 500 Returns Distribution", xlab = "Returns", col = "lightgreen")

qqnorm(as.numeric(spx_xts), main = "Q-Q Plot: S&P 500 Levels")
qqline(as.numeric(spx_xts), col = "red")

qqnorm(as.numeric(spx_returns_xts), main = "Q-Q Plot: S&P 500 Returns")
qqline(as.numeric(spx_returns_xts), col = "red")

dev.off()

#------------------------------------------------------------------
# 6.5 VIX ACF/PACF plots
#------------------------------------------------------------------

pdf("results/figures/06_vix_acf_pacf.pdf", width = 12, height = 10)

par(mfrow = c(3, 2))

acf(as.numeric(na.omit(vix_xts)), lag.max = 60, main = "ACF: VIX Levels")
pacf(as.numeric(na.omit(vix_xts)), lag.max = 60, main = "PACF: VIX Levels")

acf(as.numeric(na.omit(vix_returns_xts)), lag.max = 60, main = "ACF: VIX Returns")
pacf(as.numeric(na.omit(vix_returns_xts)), lag.max = 60, main = "PACF: VIX Returns")

acf(as.numeric(na.omit(vix_returns_xts))^2, lag.max = 60, main = "ACF: Squared VIX Returns")
pacf(as.numeric(na.omit(vix_returns_xts))^2, lag.max = 60, main = "PACF: Squared VIX Returns")

dev.off()

#------------------------------------------------------------------
# 6.6 S&P 500 ACF/PACF plots
#------------------------------------------------------------------

pdf("results/figures/07_spx_acf_pacf.pdf", width = 12, height = 10)

par(mfrow = c(3, 2))

acf(as.numeric(na.omit(spx_xts)), lag.max = 60, main = "ACF: S&P 500 Levels")
pacf(as.numeric(na.omit(spx_xts)), lag.max = 60, main = "PACF: S&P 500 Levels")

acf(as.numeric(na.omit(spx_returns_xts)), lag.max = 60, main = "ACF: S&P 500 Returns")
pacf(as.numeric(na.omit(spx_returns_xts)), lag.max = 60, main = "PACF: S&P 500 Returns")

acf(as.numeric(na.omit(spx_returns_xts))^2, lag.max = 60, main = "ACF: Squared S&P 500 Returns")
pacf(as.numeric(na.omit(spx_returns_xts))^2, lag.max = 60, main = "PACF: Squared S&P 500 Returns")

dev.off()

#------------------------------------------------------------------
# 6.7 VIX-SPX Joint plots
#------------------------------------------------------------------

pdf("results/figures/08_vix_spx_joint.pdf", width = 12, height = 10)

par(mfrow = c(2, 2))

# Scatter: Returns
plot(as.numeric(spx_returns_train), as.numeric(vix_returns_train),
     pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.3),
     xlab = "S&P 500 Returns", ylab = "VIX Returns",
     main = "VIX vs S&P 500 Returns (Training)")
abline(lm(as.numeric(vix_returns_train) ~ as.numeric(spx_returns_train)), col = "red", lwd = 2)

# Scatter: Levels
plot(as.numeric(spx_train), as.numeric(vix_train),
     pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.3),
     xlab = "S&P 500 Level", ylab = "VIX Level",
     main = "VIX vs S&P 500 Levels (Training)")

# Rolling correlation
roll_cor <- rollapply(merge(vix_returns_xts, spx_returns_xts), width = 63,
                      FUN = function(x) cor(x[,1], x[,2], use = "complete.obs"),
                      by.column = FALSE, align = "right")
plot(roll_cor, main = "63-day Rolling Correlation (VIX vs SPX Returns)",
     ylab = "Correlation", col = "purple", lwd = 0.5)
abline(h = 0, col = "grey", lty = 2)
abline(h = mean(roll_cor, na.rm = TRUE), col = "red", lty = 2)

# Cross-correlation
ccf_result <- ccf(as.numeric(na.omit(spx_returns_xts)), 
                  as.numeric(na.omit(vix_returns_xts)), 
                  lag.max = 20, plot = TRUE,
                  main = "Cross-correlation: SPX Returns vs VIX Returns")

dev.off()


#------------------------------------------------------------------
# 6.8 VIX REGIME ANALYSIS
#------------------------------------------------------------------

pdf("results/figures/09_vix_regime_analysis.pdf", width = 14, height = 12)

par(mfrow = c(3, 2), mar = c(4, 4, 3, 1))

# Define regimes
vix_numeric <- as.numeric(vix_xts)
regime_breaks <- c(0, 15, 20, 25, 30, Inf)
regime_labels <- c("Very Low (<15)", "Low (15-20)", "Medium (20-25)", 
                   "High (25-30)", "Very High (>30)")
vix_regime <- cut(vix_numeric, breaks = regime_breaks, labels = regime_labels)

# Plot 1: VIX with regime bands
plot(index(vix_xts), vix_numeric, type = "l", col = "black", lwd = 0.5,
     xlab = "Date", ylab = "VIX", main = "VIX Levels with Regime Bands")
abline(h = c(15, 20, 25, 30), col = c("green", "blue", "orange", "red"), lty = 2, lwd = 1.5)
legend("topright", c("15 (Low)", "20 (Medium)", "25 (High)", "30 (Very High)"),
       col = c("green", "blue", "orange", "red"), lty = 2, cex = 0.7)

# Plot 2: Regime frequency
regime_freq <- table(vix_regime) / length(na.omit(vix_regime)) * 100
barplot(regime_freq, col = c("darkgreen", "green", "yellow", "orange", "red"),
        main = "Time Spent in Each Regime (%)", ylab = "Percentage",
        las = 2, cex.names = 0.8)

# Plot 3: Regime transition matrix heatmap
regime_numeric <- as.numeric(vix_regime)
transitions <- table(head(regime_numeric, -1), tail(regime_numeric, -1))
trans_prob <- prop.table(transitions, margin = 1) * 100

image(1:nrow(trans_prob), 1:ncol(trans_prob), as.matrix(trans_prob),
      col = heat.colors(20, rev = TRUE), axes = FALSE,
      xlab = "From Regime", ylab = "To Regime",
      main = "Regime Transition Probabilities (%)")
axis(1, at = 1:5, labels = 1:5)
axis(2, at = 1:5, labels = 1:5)
for (i in 1:nrow(trans_prob)) {
  for (j in 1:ncol(trans_prob)) {
    text(i, j, sprintf("%.1f", trans_prob[i, j]), cex = 0.7)
  }
}

# Plot 4: Regime duration distribution
regime_runs <- rle(regime_numeric)
duration_by_regime <- split(regime_runs$lengths, regime_runs$values)

boxplot(duration_by_regime, names = 1:5,
        main = "Regime Duration Distribution (Days)",
        xlab = "Regime", ylab = "Duration (Days)",
        col = c("darkgreen", "green", "yellow", "orange", "red"))

# Plot 5: Returns distribution by regime
vix_returns_numeric <- as.numeric(vix_returns_xts)
vix_regime_returns <- cut(head(vix_numeric, -1), breaks = regime_breaks, labels = regime_labels)

boxplot(vix_returns_numeric ~ vix_regime_returns,
        main = "VIX Returns by Regime",
        xlab = "Regime", ylab = "Daily Return",
        col = c("darkgreen", "green", "yellow", "orange", "red"),
        las = 2, cex.axis = 0.7)
abline(h = 0, col = "black", lty = 2)

# Plot 6: Volatility of volatility by regime
rolling_vol <- rollapply(vix_returns_xts, width = 22, FUN = sd, align = "right", fill = NA)
rolling_vol_numeric <- as.numeric(rolling_vol)
vol_regime <- cut(as.numeric(vix_xts)[1:length(rolling_vol_numeric)], 
                  breaks = regime_breaks, labels = regime_labels)

boxplot(rolling_vol_numeric ~ vol_regime,
        main = "Volatility of VIX (22-day) by Regime",
        xlab = "Regime", ylab = "Rolling Std Dev",
        col = c("darkgreen", "green", "yellow", "orange", "red"),
        las = 2, cex.axis = 0.7)

dev.off()

#------------------------------------------------------------------
# 6.9 MEAN REVERSION ANALYSIS
#------------------------------------------------------------------

pdf("results/figures/10_mean_reversion_analysis.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Plot 1: VIX deviation from long-term mean
lt_mean <- mean(vix_numeric, na.rm = TRUE)
deviation <- vix_numeric - lt_mean

plot(index(vix_xts), deviation, type = "l", col = "blue", lwd = 0.5,
     xlab = "Date", ylab = "Deviation from Mean",
     main = sprintf("VIX Deviation from Long-term Mean (%.1f)", lt_mean))
abline(h = 0, col = "red", lwd = 2)
abline(h = c(-sd(deviation, na.rm = TRUE), sd(deviation, na.rm = TRUE)) * 2, 
       col = "orange", lty = 2)

# Plot 2: Autocorrelation decay (evidence of mean reversion)
acf_values <- acf(vix_numeric, lag.max = 252, plot = FALSE)$acf
half_life_idx <- which(acf_values < 0.5)[1]

plot(0:252, acf_values, type = "l", col = "blue", lwd = 2,
     xlab = "Lag (Days)", ylab = "Autocorrelation",
     main = sprintf("VIX Level Autocorrelation (Half-life ≈ %d days)", half_life_idx))
abline(h = 0.5, col = "red", lty = 2)
abline(v = half_life_idx, col = "red", lty = 2)
text(half_life_idx + 20, 0.6, sprintf("Half-life: %d days", half_life_idx), col = "red")

# Plot 3: Mean reversion speed by starting level
# Bin starting VIX and track average path forward
horizon <- 22  # 1 month ahead
n <- length(vix_numeric)

starting_bins <- cut(vix_numeric[1:(n-horizon)], 
                     breaks = c(0, 12, 15, 20, 25, 30, 40, 100),
                     labels = c("<12", "12-15", "15-20", "20-25", "25-30", "30-40", ">40"))

future_change <- vix_numeric[(horizon+1):n] - vix_numeric[1:(n-horizon)]

avg_change <- tapply(future_change, starting_bins, mean, na.rm = TRUE)
se_change <- tapply(future_change, starting_bins, function(x) sd(x, na.rm = TRUE) / sqrt(length(x)))

bp <- barplot(avg_change, col = ifelse(avg_change < 0, "darkgreen", "darkred"),
              main = "Average 22-day VIX Change by Starting Level",
              ylab = "Average Change", xlab = "Starting VIX Level",
              ylim = c(min(avg_change - 2*se_change), max(avg_change + 2*se_change)))
arrows(bp, avg_change - 1.96*se_change, bp, avg_change + 1.96*se_change,
       angle = 90, code = 3, length = 0.05)
abline(h = 0, col = "black", lty = 2)

# Plot 4: Scatter of current VIX vs future change
plot(vix_numeric[1:(n-horizon)], future_change,
     pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.2),
     xlab = "Current VIX", ylab = "22-day Forward Change",
     main = "Mean Reversion: Current Level vs Future Change")
abline(h = 0, col = "grey", lty = 2)
abline(lm(future_change ~ vix_numeric[1:(n-horizon)]), col = "red", lwd = 2)

# Add regression stats
mr_reg <- lm(future_change ~ vix_numeric[1:(n-horizon)])
mr_coef <- coef(mr_reg)[2]
mr_r2 <- summary(mr_reg)$r.squared
text(60, max(future_change) * 0.8, 
     sprintf("β = %.3f\nR² = %.3f", mr_coef, mr_r2), col = "red")

dev.off()

#------------------------------------------------------------------
# 6.10 LEVERAGE EFFECT / ASYMMETRY ANALYSIS
#------------------------------------------------------------------

pdf("results/figures/11_leverage_asymmetry.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Align returns
merged <- merge(vix_returns_xts, spx_returns_xts, join = "inner")
vix_ret <- as.numeric(merged[, 1])
spx_ret <- as.numeric(merged[, 2])
common_idx <- index(merged)

# Plot 1: Asymmetric scatter with separate regression lines
plot(spx_ret, vix_ret, pch = 16, cex = 0.3,
     col = ifelse(spx_ret < 0, rgb(1, 0, 0, 0.3), rgb(0, 0, 1, 0.3)),
     xlab = "S&P 500 Return", ylab = "VIX Return",
     main = "Leverage Effect: VIX vs SPX Returns")

# Separate regressions
down_idx <- spx_ret < 0
up_idx <- spx_ret >= 0

abline(lm(vix_ret[down_idx] ~ spx_ret[down_idx]), col = "red", lwd = 2)
abline(lm(vix_ret[up_idx] ~ spx_ret[up_idx]), col = "blue", lwd = 2)
abline(h = 0, v = 0, col = "grey", lty = 2)

beta_down <- coef(lm(vix_ret[down_idx] ~ spx_ret[down_idx]))[2]
beta_up <- coef(lm(vix_ret[up_idx] ~ spx_ret[up_idx]))[2]
legend("topright", 
       c(sprintf("Down days: β = %.2f", beta_down),
         sprintf("Up days: β = %.2f", beta_up)),
       col = c("red", "blue"), lwd = 2, cex = 0.8)

# Plot 2: News Impact Curve (binned)
spx_bins <- cut(spx_ret, breaks = quantile(spx_ret, probs = seq(0, 1, 0.05), na.rm = TRUE),
                include.lowest = TRUE)
avg_vix_response <- tapply(vix_ret, spx_bins, mean, na.rm = TRUE)
bin_midpoints <- tapply(spx_ret, spx_bins, mean, na.rm = TRUE)

plot(bin_midpoints, avg_vix_response, type = "b", pch = 19, col = "purple",
     xlab = "S&P 500 Return (binned)", ylab = "Average VIX Return",
     main = "News Impact Curve")
abline(h = 0, v = 0, col = "grey", lty = 2)

# Add smoothed line
lo <- loess(avg_vix_response ~ bin_midpoints)
lines(sort(bin_midpoints), predict(lo, sort(bin_midpoints)), col = "red", lwd = 2)

# Plot 3: Asymmetry over time (rolling betas)
window <- 252
n_obs <- length(spx_ret)
rolling_beta_down <- rolling_beta_up <- rep(NA, n_obs)

for (i in window:n_obs) {
  idx <- (i - window + 1):i
  down_sub <- spx_ret[idx] < 0
  up_sub <- spx_ret[idx] >= 0
  
  if (sum(down_sub) > 20) {
    rolling_beta_down[i] <- coef(lm(vix_ret[idx][down_sub] ~ spx_ret[idx][down_sub]))[2]
  }
  if (sum(up_sub) > 20) {
    rolling_beta_up[i] <- coef(lm(vix_ret[idx][up_sub] ~ spx_ret[idx][up_sub]))[2]
  }
}

length(common_idx)
length(rolling_beta_down)


plot(common_idx, rolling_beta_down, type = "l", col = "red", lwd = 1,
     xlab = "Date", ylab = "Beta",
     main = "Rolling 252-day Asymmetric Betas",
     ylim = range(c(rolling_beta_down, rolling_beta_up), na.rm = TRUE))
lines(common_idx, rolling_beta_up, col = "blue", lwd = 1)
legend("topright", c("Down days", "Up days"), col = c("red", "blue"), lwd = 2)
abline(h = 0, col = "grey", lty = 2)

# Plot 4: Asymmetry ratio over time
asymmetry_ratio <- abs(rolling_beta_down) / abs(rolling_beta_up)
plot(common_idx, asymmetry_ratio, type = "l", col = "purple", lwd = 1,
     xlab = "Date", ylab = "Asymmetry Ratio",
     main = "Leverage Asymmetry Ratio (|β_down| / |β_up|)")
abline(h = 1, col = "red", lty = 2)
abline(h = mean(asymmetry_ratio, na.rm = TRUE), col = "blue", lty = 2)
text(common_idx[100], mean(asymmetry_ratio, na.rm = TRUE) + 0.2, 
     sprintf("Mean: %.2f", mean(asymmetry_ratio, na.rm = TRUE)), col = "blue")

dev.off()

#------------------------------------------------------------------
# 6.11 TAIL BEHAVIOUR / EXTREME EVENTS
#------------------------------------------------------------------

pdf("results/figures/12_tail_analysis.pdf", width = 14, height = 12)

par(mfrow = c(3, 2), mar = c(4, 4, 3, 1))

# Plot 1: Empirical vs Normal density (VIX returns)
hist(vix_ret, breaks = 100, freq = FALSE, col = "lightblue",
     main = "VIX Returns: Empirical vs Normal",
     xlab = "Return", xlim = c(-0.3, 0.3))
curve(dnorm(x, mean = mean(vix_ret), sd = sd(vix_ret)), 
      add = TRUE, col = "red", lwd = 2)
legend("topright", c("Empirical", "Normal"), 
       fill = c("lightblue", NA), border = c("black", NA),
       lty = c(NA, 1), lwd = c(NA, 2), col = c(NA, "red"))

# Plot 2: Log density (better for tails)
hist(vix_ret, breaks = 100, freq = FALSE, col = "lightblue",
     main = "VIX Returns: Log Density Comparison",
     xlab = "Return", xlim = c(-0.3, 0.3), ylim = c(0.01, 100), log = "y")
curve(dnorm(x, mean = mean(vix_ret), sd = sd(vix_ret)), 
      add = TRUE, col = "red", lwd = 2)

# Plot 3: Exceedance probability plot
sorted_vix <- sort(abs(vix_ret), decreasing = TRUE)
n <- length(sorted_vix)
empirical_prob <- (1:n) / n

plot(sorted_vix, empirical_prob, log = "xy", type = "l", col = "blue", lwd = 2,
     xlab = "|VIX Return|", ylab = "P(|Return| > x)",
     main = "Tail Exceedance Probability")

# Add normal reference
normal_exceed <- 2 * (1 - pnorm(sorted_vix / sd(vix_ret)))
lines(sorted_vix, normal_exceed, col = "red", lwd = 2, lty = 2)
legend("topright", c("Empirical", "Normal"), col = c("blue", "red"), lwd = 2, lty = c(1, 2))

# Plot 4: Extreme VIX spikes timeline
spike_threshold <- quantile(vix_numeric, 0.95, na.rm = TRUE)
spike_dates <- index(vix_xts)[vix_numeric > spike_threshold]

plot(index(vix_xts), vix_numeric, type = "l", col = "grey", lwd = 0.5,
     xlab = "Date", ylab = "VIX", main = sprintf("VIX Spikes (> %.1f, 95th percentile)", spike_threshold))
points(spike_dates, vix_numeric[vix_numeric > spike_threshold], col = "red", pch = 16, cex = 0.5)
abline(h = spike_threshold, col = "red", lty = 2)

# Plot 5: Time between spikes
if (length(spike_dates) > 1) {
  time_between <- as.numeric(diff(spike_dates))
  hist(time_between, breaks = 30, col = "lightcoral",
       main = "Time Between VIX Spikes (Days)",
       xlab = "Days")
  abline(v = median(time_between), col = "blue", lwd = 2, lty = 2)
  text(median(time_between) + 20, par("usr")[4] * 0.9, 
       sprintf("Median: %d days", median(time_between)), col = "blue")
}

# Plot 6: Spike magnitude vs duration
# Find spike clusters
spike_idx <- which(vix_numeric > spike_threshold)
spike_runs <- rle(diff(spike_idx) == 1)

# Calculate spike characteristics
spike_magnitudes <- sapply(split(vix_numeric[vix_numeric > spike_threshold], 
                                 cumsum(c(1, diff(spike_idx) > 5))), max)
spike_durations <- sapply(split(vix_numeric[vix_numeric > spike_threshold], 
                                cumsum(c(1, diff(spike_idx) > 5))), length)

if (length(spike_magnitudes) > 5) {
  plot(spike_durations, spike_magnitudes, pch = 19, col = "darkred",
       xlab = "Spike Duration (Days)", ylab = "Peak VIX",
       main = "Spike Magnitude vs Duration")
  abline(lm(spike_magnitudes ~ spike_durations), col = "blue", lwd = 2)
}

dev.off()

#------------------------------------------------------------------
# 6.12 SEASONALITY ANALYSIS
#------------------------------------------------------------------

pdf("results/figures/13_seasonality.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Create date components
dates <- index(vix_xts)
dow <- weekdays(dates)
month <- months(dates)
year <- format(dates, "%Y")

# Plot 1: Day of week effect
dow_order <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
dow_factor <- factor(dow, levels = dow_order)

vix_by_dow <- tapply(vix_numeric, dow_factor, mean, na.rm = TRUE)
vix_se_dow <- tapply(vix_numeric, dow_factor, function(x) sd(x, na.rm = TRUE) / sqrt(length(x)))

bp <- barplot(vix_by_dow, col = "steelblue",
              main = "Average VIX by Day of Week",
              ylab = "Average VIX", ylim = c(0, max(vix_by_dow) * 1.1))
arrows(bp, vix_by_dow - 1.96*vix_se_dow, bp, vix_by_dow + 1.96*vix_se_dow,
       angle = 90, code = 3, length = 0.05)
abline(h = mean(vix_numeric, na.rm = TRUE), col = "red", lty = 2)

# Plot 2: Monthly seasonality
month_order <- c("January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December")
month_factor <- factor(month, levels = month_order)

vix_by_month <- tapply(vix_numeric, month_factor, mean, na.rm = TRUE)
vix_se_month <- tapply(vix_numeric, month_factor, function(x) sd(x, na.rm = TRUE) / sqrt(length(x)))

bp <- barplot(vix_by_month, col = "coral",
              main = "Average VIX by Month",
              ylab = "Average VIX", las = 2, cex.names = 0.7)
arrows(bp, vix_by_month - 1.96*vix_se_month, bp, vix_by_month + 1.96*vix_se_month,
       angle = 90, code = 3, length = 0.05)
abline(h = mean(vix_numeric, na.rm = TRUE), col = "red", lty = 2)

# Plot 3: Day of week returns
vix_ret_dow <- tapply(vix_ret, dow_factor[1:length(vix_ret)], mean, na.rm = TRUE) * 100
vix_ret_se_dow <- tapply(vix_ret, dow_factor[1:length(vix_ret)], 
                         function(x) sd(x, na.rm = TRUE) / sqrt(length(x))) * 100

bp <- barplot(vix_ret_dow, col = ifelse(vix_ret_dow > 0, "darkgreen", "darkred"),
              main = "Average VIX Return by Day of Week (%)",
              ylab = "Average Return (%)")
arrows(bp, vix_ret_dow - 1.96*vix_ret_se_dow, bp, vix_ret_dow + 1.96*vix_ret_se_dow,
       angle = 90, code = 3, length = 0.05)
abline(h = 0, col = "black", lty = 2)

# Plot 4: Year-over-year comparison
vix_by_year <- tapply(vix_numeric, year, mean, na.rm = TRUE)

barplot(vix_by_year, col = viridis::viridis(length(vix_by_year)),
        main = "Average VIX by Year",
        ylab = "Average VIX", las = 2, cex.names = 0.7)
abline(h = mean(vix_numeric, na.rm = TRUE), col = "red", lty = 2)

dev.off()

#------------------------------------------------------------------
# 6.13 VOLATILITY OF VOLATILITY (VoV)
#------------------------------------------------------------------

pdf("results/figures/14_volatility_of_volatility.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Calculate rolling volatility of VIX
vov_22 <- rollapply(vix_returns_xts, width = 22, FUN = sd, align = "right", fill = NA) * sqrt(252)
vov_63 <- rollapply(vix_returns_xts, width = 63, FUN = sd, align = "right", fill = NA) * sqrt(252)

# Plot 1: VoV time series
plot(index(vov_22), as.numeric(vov_22), type = "l", col = "blue", lwd = 0.8,
     xlab = "Date", ylab = "Annualised Volatility",
     main = "Volatility of VIX (Annualised)")
lines(index(vov_63), as.numeric(vov_63), col = "red", lwd = 1)
legend("topright", c("22-day", "63-day"), col = c("blue", "red"), lwd = c(0.8, 1))

# Plot 2: VoV vs VIX level
vov_numeric <- as.numeric(vov_22)
vix_aligned <- vix_numeric[1:length(vov_numeric)]

plot(vix_aligned, vov_numeric, pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.2),
     xlab = "VIX Level", ylab = "22-day Realised Vol of VIX",
     main = "Volatility of Volatility vs VIX Level")
abline(lm(vov_numeric ~ vix_aligned), col = "red", lwd = 2)

vov_reg <- lm(vov_numeric ~ vix_aligned)
text(60, max(vov_numeric, na.rm = TRUE) * 0.9, 
     sprintf("β = %.3f, R² = %.3f", coef(vov_reg)[2], summary(vov_reg)$r.squared), col = "red")

# Plot 3: VoV distribution
hist(vov_numeric, breaks = 50, col = "lightblue",
     main = "Distribution of Volatility of VIX",
     xlab = "22-day Annualised Vol")
abline(v = mean(vov_numeric, na.rm = TRUE), col = "red", lwd = 2)
abline(v = median(vov_numeric, na.rm = TRUE), col = "blue", lwd = 2, lty = 2)
legend("topright", c("Mean", "Median"), col = c("red", "blue"), lwd = 2, lty = c(1, 2))

# Plot 4: VoV regime persistence
vov_high <- vov_numeric > quantile(vov_numeric, 0.75, na.rm = TRUE)
vov_runs <- rle(vov_high)
high_vov_durations <- vov_runs$lengths[vov_runs$values == TRUE]

if (length(high_vov_durations) > 5) {
  hist(high_vov_durations, breaks = 20, col = "coral",
       main = "Duration of High VoV Regimes (Days)",
       xlab = "Duration")
  abline(v = mean(high_vov_durations), col = "blue", lwd = 2, lty = 2)
}

dev.off()

#------------------------------------------------------------------
# 6.14 PREDICTABILITY ANALYSIS
#------------------------------------------------------------------

pdf("results/figures/15_predictability_analysis.pdf", width = 14, height = 12)

par(mfrow = c(3, 2), mar = c(4, 4, 3, 1))

# Plot 1: Lagged VIX autocorrelation (levels vs returns)
acf_levels <- acf(vix_numeric, lag.max = 60, plot = FALSE)$acf
acf_returns <- acf(vix_ret, lag.max = 60, plot = FALSE)$acf

plot(0:60, acf_levels, type = "h", col = "blue", lwd = 2,
     xlab = "Lag (Days)", ylab = "Autocorrelation",
     main = "Autocorrelation: Levels vs Returns", ylim = c(-0.1, 1))
lines(0:60 + 0.3, acf_returns, type = "h", col = "red", lwd = 2)
abline(h = 0, col = "grey")
abline(h = c(-1.96, 1.96) / sqrt(length(vix_ret)), col = "grey", lty = 2)
legend("topright", c("Levels", "Returns"), col = c("blue", "red"), lwd = 2)


spx_numeric <- as.numeric(spx_xts[common_idx])




# Plot 2: Partial autocorrelation of squared returns (ARCH effects)
pacf_obj <- pacf(vix_ret^2, lag.max = 30, plot = FALSE)
barplot(as.vector(pacf_obj$acf), names.arg = 1:30, col = "purple",
        main = "PACF of Squared VIX Returns",
        xlab = "Lag", ylab = "Partial Autocorrelation")


# Plot 3: Cross-predictability (lagged SPX -> VIX)
ccf_vals <- ccf(spx_ret, vix_ret, lag.max = 20, plot = FALSE)
plot(ccf_vals$lag, ccf_vals$acf, type = "h", col = "darkgreen", lwd = 2,
     xlab = "Lag (SPX leads at negative lags)", ylab = "Cross-correlation",
     main = "Cross-correlation: SPX Returns → VIX Returns")
abline(h = 0, col = "grey")
abline(h = c(-1.96, 1.96) / sqrt(length(vix_ret)), col = "red", lty = 2)

# Plot 4: VIX as predictor of future SPX returns
horizons <- c(1, 5, 22, 63)
pred_power <- numeric(length(horizons))

for (i in seq_along(horizons)) {
  h <- horizons[i]
  future_spx <- (spx_numeric[(h+1):length(spx_numeric)] - spx_numeric[1:(length(spx_numeric)-h)]) / 
    spx_numeric[1:(length(spx_numeric)-h)]
  current_vix <- vix_numeric[1:(length(vix_numeric)-h)]
  
  # Align lengths
  min_len <- min(length(future_spx), length(current_vix))
  pred_reg <- lm(future_spx[1:min_len] ~ current_vix[1:min_len])
  pred_power[i] <- summary(pred_reg)$r.squared
}

spx_numeric <- as.numeric(spx_xts)
barplot(pred_power * 100, names.arg = paste0(horizons, "d"),
        col = "steelblue",
        main = "VIX Predictive Power for Future SPX Returns",
        ylab = "R² (%)", xlab = "Horizon")

# Plot 5: Scatter - High VIX predicts positive returns?
h <- 22
future_spx_22 <- (spx_numeric[(h+1):length(spx_numeric)] - spx_numeric[1:(length(spx_numeric)-h)]) / 
  spx_numeric[1:(length(spx_numeric)-h)] * 100
current_vix_22 <- vix_numeric[1:(length(vix_numeric)-h)]
min_len <- min(length(future_spx_22), length(current_vix_22))

plot(current_vix_22[1:min_len], future_spx_22[1:min_len],
     pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.2),
     xlab = "Current VIX", ylab = "22-day Forward SPX Return (%)",
     main = "VIX Level vs Future SPX Returns")
abline(h = 0, col = "grey", lty = 2)
abline(lm(future_spx_22[1:min_len] ~ current_vix_22[1:min_len]), col = "red", lwd = 2)

# Bin and show average
vix_bins <- cut(current_vix_22[1:min_len], breaks = c(0, 15, 20, 25, 30, 40, 100))
avg_ret_by_vix <- tapply(future_spx_22[1:min_len], vix_bins, mean, na.rm = TRUE)

# Plot 6: Average future returns by VIX level
barplot(avg_ret_by_vix, col = ifelse(avg_ret_by_vix > 0, "darkgreen", "darkred"),
        main = "Average 22-day SPX Return by VIX Level",
        ylab = "Average Return (%)", las = 2, cex.names = 0.8)
abline(h = 0, col = "black", lty = 2)

dev.off()

#------------------------------------------------------------------
# 6.15 CORRELATION STRUCTURE OVER TIME
#------------------------------------------------------------------

pdf("results/figures/16_correlation_dynamics.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Plot 1: Rolling correlation with confidence bands
roll_cor <- rollapply(merge(vix_returns_xts, spx_returns_xts), width = 63,
                      FUN = function(x) cor(x[,1], x[,2], use = "complete.obs"),
                      by.column = FALSE, align = "right")

plot(index(roll_cor), as.numeric(roll_cor), type = "l", col = "purple", lwd = 1,
     xlab = "Date", ylab = "Correlation",
     main = "63-day Rolling Correlation (VIX vs SPX Returns)")
abline(h = mean(roll_cor, na.rm = TRUE), col = "red", lwd = 2, lty = 2)
abline(h = 0, col = "grey", lty = 2)

# Add regime shading based on VIX level
vix_high_periods <- vix_numeric > 25

# Plot 2: Correlation by VIX regime
cor_by_regime <- tapply(1:length(vix_ret), vix_regime[1:length(vix_ret)], function(idx) {
  cor(vix_ret[idx], spx_ret[idx], use = "complete.obs")
})

barplot(cor_by_regime, col = c("darkgreen", "green", "yellow", "orange", "red"),
        main = "VIX-SPX Correlation by VIX Regime",
        ylab = "Correlation", las = 2, cex.names = 0.7)
abline(h = mean(cor_by_regime, na.rm = TRUE), col = "blue", lty = 2)

# Plot 3: Correlation vs VIX level scatter
roll_cor_numeric <- as.numeric(roll_cor)
vix_for_cor <- vix_numeric[(63):length(vix_numeric)]
min_len <- min(length(roll_cor_numeric), length(vix_for_cor))

plot(vix_for_cor[1:min_len], roll_cor_numeric[1:min_len],
     pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.3),
     xlab = "VIX Level", ylab = "63-day Rolling Correlation",
     main = "Correlation vs VIX Level")
abline(lm(roll_cor_numeric[1:min_len] ~ vix_for_cor[1:min_len]), col = "red", lwd = 2)

# Plot 4: Correlation stability (rolling std of correlation)
roll_cor_vol <- rollapply(roll_cor, width = 252, FUN = sd, align = "right", fill = NA)

plot(index(roll_cor_vol), as.numeric(roll_cor_vol), type = "l", col = "darkblue", lwd = 1,
     xlab = "Date", ylab = "Std Dev of Correlation",
     main = "Correlation Stability (252-day Rolling Std Dev)")
abline(h = mean(roll_cor_vol, na.rm = TRUE), col = "red", lwd = 2, lty = 2)

dev.off()

#------------------------------------------------------------------
# 6.16 SUMMARY STATISTICS TABLE (LaTeX-ready)
#------------------------------------------------------------------

# Create comprehensive summary table
summary_stats <- data.frame(
  Statistic = c("N", "Mean", "Median", "Std Dev", "Min", "Max", 
                "Skewness", "Kurtosis", "Q1", "Q3", "IQR",
                "ADF p-value", "KPSS p-value", "JB p-value"),
  VIX_Levels = c(
    length(na.omit(vix_numeric)),
    mean(vix_numeric, na.rm = TRUE),
    median(vix_numeric, na.rm = TRUE),
    sd(vix_numeric, na.rm = TRUE),
    min(vix_numeric, na.rm = TRUE),
    max(vix_numeric, na.rm = TRUE),
    moments::skewness(vix_numeric, na.rm = TRUE),
    moments::kurtosis(vix_numeric, na.rm = TRUE),
    quantile(vix_numeric, 0.25, na.rm = TRUE),
    quantile(vix_numeric, 0.75, na.rm = TRUE),
    IQR(vix_numeric, na.rm = TRUE),
    adf_vix_levels$p.value,
    kpss_vix_levels$p.value,
    jb_vix_levels$p.value
  ),
  VIX_Returns = c(
    length(na.omit(vix_ret)),
    mean(vix_ret, na.rm = TRUE),
    median(vix_ret, na.rm = TRUE),
    sd(vix_ret, na.rm = TRUE),
    min(vix_ret, na.rm = TRUE),
    max(vix_ret, na.rm = TRUE),
    moments::skewness(vix_ret, na.rm = TRUE),
    moments::kurtosis(vix_ret, na.rm = TRUE),
    quantile(vix_ret, 0.25, na.rm = TRUE),
    quantile(vix_ret, 0.75, na.rm = TRUE),
    IQR(vix_ret, na.rm = TRUE),
    adf_vix_returns$p.value,
    kpss_vix_returns$p.value,
    jb_vix_returns$p.value
  ),
  SPX_Levels = c(
    length(na.omit(spx_numeric)),
    mean(spx_numeric, na.rm = TRUE),
    median(spx_numeric, na.rm = TRUE),
    sd(spx_numeric, na.rm = TRUE),
    min(spx_numeric, na.rm = TRUE),
    max(spx_numeric, na.rm = TRUE),
    moments::skewness(spx_numeric, na.rm = TRUE),
    moments::kurtosis(spx_numeric, na.rm = TRUE),
    quantile(spx_numeric, 0.25, na.rm = TRUE),
    quantile(spx_numeric, 0.75, na.rm = TRUE),
    IQR(spx_numeric, na.rm = TRUE),
    adf_spx_levels$p.value,
    kpss_spx_levels$p.value,
    jb_spx_levels$p.value
  ),
  SPX_Returns = c(
    length(na.omit(spx_ret)),
    mean(spx_ret, na.rm = TRUE),
    median(spx_ret, na.rm = TRUE),
    sd(spx_ret, na.rm = TRUE),
    min(spx_ret, na.rm = TRUE),
    max(spx_ret, na.rm = TRUE),
    moments::skewness(spx_ret, na.rm = TRUE),
    moments::kurtosis(spx_ret, na.rm = TRUE),
    quantile(spx_ret, 0.25, na.rm = TRUE),
    quantile(spx_ret, 0.75, na.rm = TRUE),
    IQR(spx_ret, na.rm = TRUE),
    adf_spx_returns$p.value,
    kpss_spx_returns$p.value,
    jb_spx_returns$p.value
  )
)

write.csv(summary_stats, "results/tables/comprehensive_summary_stats.csv", row.names = FALSE)

# Generate LaTeX table
if (requireNamespace("xtable", quietly = TRUE)) {
  latex_table <- xtable::xtable(summary_stats, 
                                caption = "Descriptive Statistics and Statistical Tests",
                                label = "tab:desc_stats",
                                digits = 4)
  print(latex_table, file = "results/tables/summary_stats.tex",
        include.rownames = FALSE)
}

cat("\nNext step: Run HAR GARCH Models.R\n")