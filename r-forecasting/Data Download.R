################################################################################
# 1. SOURCE SETUP
################################################################################

source("Setup.R")

#------------------------------------------------------------------
# 2. DOWNLOAD S&P 500 DATA
#------------------------------------------------------------------

spx_data <- tryCatch({
  getSymbols(
    Symbols   = config$tickers$spx,
    src       = "yahoo",
    from      = config$start_date,
    to        = config$end_date,
    auto.assign = FALSE
  )
}, error = function(e) {
  cat_progress(sprintf("ERROR downloading SPX: %s", e$message))
  NULL
})

if (is.null(spx_data)) {
  stop("Failed to download S&P 500 data.")
}

# Convert to data.table
spx_dt <- data.table(
  date   = index(spx_data),
  open   = as.numeric(Op(spx_data)),
  high   = as.numeric(Hi(spx_data)),
  low    = as.numeric(Lo(spx_data)),
  close  = as.numeric(Cl(spx_data)),
  volume = as.numeric(Vo(spx_data)),
  adj_close = as.numeric(Ad(spx_data))
)

cat_progress(sprintf("SPX data: %d observations from %s to %s",
                     nrow(spx_dt), min(spx_dt$date), max(spx_dt$date)))

#------------------------------------------------------------------
# 3. DOWNLOAD VIX DATA
#------------------------------------------------------------------

vix_data <- tryCatch({
  getSymbols(
    Symbols   = config$tickers$vix,
    src       = "yahoo",
    from      = config$start_date,
    to        = config$end_date,
    auto.assign = FALSE
  )
}, error = function(e) {
  cat_progress(sprintf("ERROR downloading VIX: %s", e$message))
  NULL
})

if (is.null(vix_data)) {
  stop("Failed to download VIX data. Check internet connection.")
}

# Convert to data.table
vix_dt <- data.table(
  date      = index(vix_data),
  vix_open  = as.numeric(Op(vix_data)),
  vix_high  = as.numeric(Hi(vix_data)),
  vix_low   = as.numeric(Lo(vix_data)),
  vix_close = as.numeric(Cl(vix_data))
)

cat_progress(sprintf("VIX data: %d observations from %s to %s",
                     nrow(vix_dt), min(vix_dt$date), max(vix_dt$date)))

#------------------------------------------------------------------
# 4. MERGE DATASETS
#------------------------------------------------------------------



# Inner join on date to ensure alignment
merged_dt <- merge(spx_dt, vix_dt, by = "date", all = FALSE)

cat_progress(sprintf("Merged data: %d observations from %s to %s",
                     nrow(merged_dt), min(merged_dt$date), max(merged_dt$date)))

#------------------------------------------------------------------
# 5. BASIC DATA QUALITY CHECKS
#------------------------------------------------------------------

n_duplicates <- sum(duplicated(merged_dt$date))
if (n_duplicates > 0) {
  cat_progress(sprintf("WARNING: Found %d duplicate dates", n_duplicates))
  merged_dt <- unique(merged_dt, by = "date")
}


missing_summary <- sapply(merged_dt, function(x) sum(is.na(x)))
if (any(missing_summary > 0)) {
  cat_progress("Missing values by column:")
  print(missing_summary[missing_summary > 0])
}

merged_dt <- merged_dt[order(date)]
date_gaps <- diff(merged_dt$date)
large_gaps <- which(date_gaps > 5)

if (length(large_gaps) > 0) {
  cat_progress(sprintf("Note: %d date gaps > 5 days (weekends/holidays expected)",
                       length(large_gaps)))
}

#------------------------------------------------------------------
# 6. COMPUTE BASIC RETURNS
#------------------------------------------------------------------


merged_dt[, `:=`(
  
  # Log returns
  log_return = c(NA, diff(log(close))),
  
  # Simple returns
  simple_return = c(NA, diff(close) / head(close, -1)),
  
  # VIX changes
  vix_return = c(NA, diff(log(vix_close))),
  vix_change = c(NA, diff(vix_close))
)]

#------------------------------------------------------------------
# 7. SUMMARY STATISTICS
#------------------------------------------------------------------

summary_stats <- data.table(
  variable = c("SPX Close", "SPX Log Return", "VIX Close", "VIX Change"),
  n = c(
    sum(!is.na(merged_dt$close)),
    sum(!is.na(merged_dt$log_return)),
    sum(!is.na(merged_dt$vix_close)),
    sum(!is.na(merged_dt$vix_change))
  ),
  mean = c(
    mean(merged_dt$close, na.rm = TRUE),
    mean(merged_dt$log_return, na.rm = TRUE),
    mean(merged_dt$vix_close, na.rm = TRUE),
    mean(merged_dt$vix_change, na.rm = TRUE)
  ),
  sd = c(
    sd(merged_dt$close, na.rm = TRUE),
    sd(merged_dt$log_return, na.rm = TRUE),
    sd(merged_dt$vix_close, na.rm = TRUE),
    sd(merged_dt$vix_change, na.rm = TRUE)
  ),
  min = c(
    min(merged_dt$close, na.rm = TRUE),
    min(merged_dt$log_return, na.rm = TRUE),
    min(merged_dt$vix_close, na.rm = TRUE),
    min(merged_dt$vix_change, na.rm = TRUE)
  ),
  max = c(
    max(merged_dt$close, na.rm = TRUE),
    max(merged_dt$log_return, na.rm = TRUE),
    max(merged_dt$vix_close, na.rm = TRUE),
    max(merged_dt$vix_change, na.rm = TRUE)
  ),
  skewness = c(
    moments::skewness(merged_dt$close, na.rm = TRUE),
    moments::skewness(merged_dt$log_return, na.rm = TRUE),
    moments::skewness(merged_dt$vix_close, na.rm = TRUE),
    moments::skewness(merged_dt$vix_change, na.rm = TRUE)
  ),
  kurtosis = c(
    moments::kurtosis(merged_dt$close, na.rm = TRUE),
    moments::kurtosis(merged_dt$log_return, na.rm = TRUE),
    moments::kurtosis(merged_dt$vix_close, na.rm = TRUE),
    moments::kurtosis(merged_dt$vix_change, na.rm = TRUE)
  )
)

print(summary_stats)

#------------------------------------------------------------------
# 8. SAVE DATA
#------------------------------------------------------------------

saveRDS(spx_dt, file.path(dirs$data, "spx_raw.rds"))
saveRDS(vix_dt, file.path(dirs$data, "vix_raw.rds"))
saveRDS(merged_dt, file.path(dirs$data, "merged_data.rds"))

write.csv(summary_stats, file.path(dirs$tables, "01_download_summary.csv"), 
          row.names = FALSE)

cat_progress("Data saved successfully:")
cat_progress(sprintf("  - SPX raw: %s", file.path(dirs$data, "spx_raw.rds")))
cat_progress(sprintf("  - VIX raw: %s", file.path(dirs$data, "vix_raw.rds")))
cat_progress(sprintf("  - Merged: %s", file.path(dirs$data, "merged_data.rds")))

#------------------------------------------------------------------
# 9. BASIC PLOTS
#------------------------------------------------------------------

# Plot SPX price series
p1 <- ggplot(merged_dt, aes(x = date, y = close)) +
  geom_line(colour = "darkblue", linewidth = 0.3) +
  labs(
    title = "S&P 500 Index (Daily Close)",
    x = "Date",
    y = "Price"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 8)
  )

# Plot VIX series
p2 <- ggplot(merged_dt, aes(x = date, y = vix_close)) +
  geom_line(colour = "darkred", linewidth = 0.3) +
  labs(
    title = "CBOE VIX Index (Daily Close)",
    x = "Date",
    y = "VIX Level"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 8)
  )

# Plot SPX returns
p3 <- ggplot(merged_dt[!is.na(log_return)], aes(x = date, y = log_return)) +
  geom_line(colour = "grey30", linewidth = 0.2) +
  labs(
    title = "S&P 500 Log Returns",
    x = "Date",
    y = "Log Return"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 8)
  )

# Combined plot
combined_plot <- gridExtra::grid.arrange(p1, p2, p3, ncol = 1)

ggsave(
  file.path(dirs$figures, "01_data_overview.pdf"),
  combined_plot,
  width = 10,
  height = 10
)

#------------------------------------------------------------------
# 10. CLEANUP
#------------------------------------------------------------------

rm(spx_data, vix_data, p1, p2, p3, combined_plot)
gc()

################################################################################
# END OF SCRIPT
################################################################################




library(readr)


data <- readRDS("data/data_with_rv.rds")
write_csv(data, "data/data_with_rv.csv")