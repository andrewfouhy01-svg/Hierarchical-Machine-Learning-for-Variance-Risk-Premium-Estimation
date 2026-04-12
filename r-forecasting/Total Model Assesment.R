################################################################################
# 05_Model_Comparison.R - Updated with Proper Separation of Comparisons
################################################################################
#
# STRUCTURE:
# 1. POINT FORECASTING: XGBoost vs HAR vs GARCH(1,1)
#    - GARCH variants use same AR(1) mean equation, so comparing them is meaningless
#    - Only GARCH(1,1) as representative traditional benchmark
#
# 2. VOLATILITY FORECASTING: GARCH vs EGARCH vs GJR-GARCH vs TGARCH
#    - This is where GARCH variants genuinely differ
#    - Use QLIKE loss, MCS, Mincer-Zarnowitz
#
# 3. DIRECTION FORECASTING: XGBoost vs HAR vs GARCH(1,1)
#    - Classification metrics
#
################################################################################

#------------------------------------------------------------------
# 0. SETUP
#------------------------------------------------------------------

source("Setup.R")

#------------------------------------------------------------------
# 1. LOAD MODEL RESULTS
#------------------------------------------------------------------

har_results <- readRDS("results/models/har_model_results.rds")
garch_results <- readRDS("results/models/garch_model_results.rds")

xgb_class_results <- readRDS("results/models/xgb_classification_full_results.rds")
xgb_reg_results <- readRDS("results/models/xgb_regression_mse_full_results.rds")

xgb_class_preds <- readRDS("results/models/xgb_classification_predictions.rds")
xgb_reg_preds <- readRDS("results/models/xgb_regression_mse_predictions.rds")

# Volatility forecasts (if available from updated HAR_GARCH script)
variance_forecasts <- tryCatch({
  readRDS("results/models/variance_forecast_predictions.rds")
}, error = function(e) NULL)

volatility_metrics <- tryCatch({
  read.csv("results/tables/volatility_forecast_metrics.csv")
}, error = function(e) NULL)

test_features <- readRDS("data/features_test.rds")
split_info <- readRDS("results/models/split_info.rds")

#------------------------------------------------------------------
# 2. HELPER FUNCTIONS
#------------------------------------------------------------------

# Classification metrics
calc_classification_metrics <- function(actual, pred_prob, pred_class = NULL, 
                                        threshold = 0.5) {
  if (is.null(pred_class)) {
    pred_class <- as.integer(pred_prob >= threshold)
  }
  
  tp <- sum(pred_class == 1 & actual == 1)
  tn <- sum(pred_class == 0 & actual == 0)
  fp <- sum(pred_class == 1 & actual == 0)
  fn <- sum(pred_class == 0 & actual == 1)
  
  n <- length(actual)
  
  accuracy <- (tp + tn) / n
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  f1 <- ifelse((precision + recall) > 0, 
               2 * precision * recall / (precision + recall), 0)
  
  balanced_acc <- (recall + specificity) / 2
  
  mcc_num <- (tp * tn) - (fp * fn)
  mcc_den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- ifelse(mcc_den > 0, mcc_num / mcc_den, 0)
  
  if (length(unique(actual)) > 1 && length(unique(pred_prob)) > 1) {
    roc_obj <- pROC::roc(actual, pred_prob, quiet = TRUE)
    auc <- as.numeric(pROC::auc(roc_obj))
  } else {
    auc <- NA
  }
  
  eps <- 1e-15
  pred_prob_clipped <- pmax(pmin(pred_prob, 1 - eps), eps)
  log_loss <- -mean(actual * log(pred_prob_clipped) + 
                      (1 - actual) * log(1 - pred_prob_clipped))
  
  brier <- mean((pred_prob - actual)^2)
  
  list(
    Accuracy = accuracy,
    Balanced_Accuracy = balanced_acc,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    F1 = f1,
    MCC = mcc,
    AUC = auc,
    Log_Loss = log_loss,
    Brier = brier,
    TP = tp, TN = tn, FP = fp, FN = fn
  )
}

# Regression metrics
calc_regression_metrics <- function(actual, predicted) {
  errors <- actual - predicted
  n <- length(actual)
  
  rmse <- sqrt(mean(errors^2))
  mae <- mean(abs(errors))
  me <- mean(errors)
  mape <- mean(abs(errors / actual)) * 100
  smape <- mean(2 * abs(errors) / (abs(actual) + abs(predicted))) * 100
  
  ss_res <- sum(errors^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  naive_errors <- diff(actual)
  theil_u <- rmse / sqrt(mean(naive_errors^2, na.rm = TRUE))
  
  actual_dir <- sign(diff(actual))
  pred_dir <- sign(diff(predicted))
  dir_accuracy <- mean(actual_dir == pred_dir, na.rm = TRUE)
  
  list(
    RMSE = rmse,
    MAE = mae,
    ME = me,
    MAPE = mape,
    SMAPE = smape,
    R2 = r2,
    Theil_U = theil_u,
    Dir_Accuracy = dir_accuracy
  )
}

# Diebold-Mariano test with HAC variance
dm_test_hac <- function(e1, e2, h = 1, power = 2) {
  d <- abs(e1)^power - abs(e2)^power
  n <- length(d)
  d_mean <- mean(d)
  
  # Newey-West bandwidth
  max_lag <- floor(n^(1/3))
  gamma <- numeric(max_lag + 1)
  for (j in 0:max_lag) {
    gamma[j + 1] <- mean((d[1:(n-j)] - d_mean) * (d[(j+1):n] - d_mean))
  }
  
  weights <- 1 - (1:max_lag) / (max_lag + 1)
  var_d <- gamma[1] + 2 * sum(weights * gamma[-1])
  
  dm_stat <- d_mean / sqrt(var_d / n)
  p_value <- 2 * (1 - pnorm(abs(dm_stat)))
  
  list(
    statistic = dm_stat,
    p_value = p_value,
    mean_diff = d_mean,
    model1_better = d_mean < 0
  )
}

# McNemar's test
mcnemar_test <- function(pred1, pred2, actual) {
  correct1 <- pred1 == actual
  correct2 <- pred2 == actual
  
  b <- sum(correct1 & !correct2)
  c <- sum(!correct1 & correct2)
  
  if ((b + c) > 0) {
    chi2 <- (abs(b - c) - 1)^2 / (b + c)
    p_value <- 1 - pchisq(chi2, df = 1)
  } else {
    chi2 <- 0
    p_value <- 1
  }
  
  list(
    statistic = chi2,
    p_value = p_value,
    b = b,
    c = c,
    model1_better = b > c
  )
}

#------------------------------------------------------------------
# 3. ALIGN PREDICTIONS
#------------------------------------------------------------------

# XGBoost predictions
xgb_class_dates <- xgb_class_preds$date
xgb_reg_dates <- xgb_reg_preds$date

# Time series predictions
har_test_preds <- har_results$test_predictions
ts_dates <- har_test_preds$date

# Find common dates
common_dates <- Reduce(intersect, list(
  as.character(xgb_class_dates),
  as.character(xgb_reg_dates),
  as.character(ts_dates)
))
common_dates <- as.Date(common_dates)

cat_progress(sprintf("Common test dates: %d observations from %s to %s",
                     length(common_dates), min(common_dates), max(common_dates)))

#------------------------------------------------------------------
# 4. POINT FORECASTING: XGBoost vs HAR vs GARCH(1,1)
#------------------------------------------------------------------

cat_progress("=" %>% rep(70) %>% paste(collapse = ""))
cat_progress("SECTION 1: POINT FORECASTING (XGBoost vs HAR vs GARCH(1,1))")
cat_progress("=" %>% rep(70) %>% paste(collapse = ""))

# Align regression predictions
reg_preds_aligned <- data.frame(
  date = common_dates,
  actual_level = xgb_reg_preds$actual[match(common_dates, xgb_reg_preds$date)]
)

# XGBoost
reg_preds_aligned$XGBoost <- xgb_reg_preds$predicted[match(common_dates, xgb_reg_preds$date)]

# HAR
har_idx <- match(common_dates, har_test_preds$date)
reg_preds_aligned$HAR <- har_test_preds$pred_level[har_idx]

# GARCH(1,1) only - other variants are meaningless for point forecasts
if (!is.null(garch_results[["GARCH"]])) {
  garch_preds <- garch_results[["GARCH"]]$test_predictions
  garch_idx <- match(common_dates, garch_preds$date)
  reg_preds_aligned$GARCH <- garch_preds$pred_level[garch_idx]
}

# Compute regression metrics
point_models <- c("XGBoost", "HAR")
if ("GARCH" %in% colnames(reg_preds_aligned)) {
  point_models <- c(point_models, "GARCH")
}

reg_metrics_df <- data.frame()
for (model_name in point_models) {
  metrics <- calc_regression_metrics(
    reg_preds_aligned$actual_level,
    reg_preds_aligned[[model_name]]
  )
  
  reg_metrics_df <- rbind(reg_metrics_df, data.frame(
    Model = model_name,
    RMSE = metrics$RMSE,
    MAE = metrics$MAE,
    MAPE = metrics$MAPE,
    R2 = metrics$R2,
    Theil_U = metrics$Theil_U,
    Dir_Accuracy = metrics$Dir_Accuracy
  ))
}

# Sort by RMSE
reg_metrics_df <- reg_metrics_df[order(reg_metrics_df$RMSE), ]

cat("\n========== POINT FORECAST METRICS ==========\n")
print(reg_metrics_df, digits = 4)

# Diebold-Mariano tests for point forecasts
dm_point_results <- data.frame()
best_model <- reg_metrics_df$Model[1]

for (model_name in point_models) {
  if (model_name != best_model) {
    e1 <- reg_preds_aligned$actual_level - reg_preds_aligned[[best_model]]
    e2 <- reg_preds_aligned$actual_level - reg_preds_aligned[[model_name]]
    
    dm_mse <- dm_test_hac(e1, e2, power = 2)
    dm_mae <- dm_test_hac(e1, e2, power = 1)
    
    dm_point_results <- rbind(dm_point_results, data.frame(
      Model1 = best_model,
      Model2 = model_name,
      DM_MSE_Stat = dm_mse$statistic,
      DM_MSE_P = dm_mse$p_value,
      MSE_Sig_05 = dm_mse$p_value < 0.05,
      DM_MAE_Stat = dm_mae$statistic,
      DM_MAE_P = dm_mae$p_value,
      MAE_Sig_05 = dm_mae$p_value < 0.05
    ))
  }
}

cat("\n========== DIEBOLD-MARIANO TESTS (Point Forecasts) ==========\n")
print(dm_point_results, digits = 4)

#------------------------------------------------------------------
# 5. DIRECTION FORECASTING: XGBoost vs HAR vs GARCH(1,1)
#------------------------------------------------------------------

cat_progress("=" %>% rep(70) %>% paste(collapse = ""))
cat_progress("SECTION 2: DIRECTION FORECASTING (XGBoost vs HAR vs GARCH(1,1))")
cat_progress("=" %>% rep(70) %>% paste(collapse = ""))

# Align classification predictions
class_preds_aligned <- data.frame(
  date = common_dates,
  actual_direction = xgb_class_preds$actual[match(common_dates, xgb_class_preds$date)]
)

# XGBoost
class_preds_aligned$XGBoost_prob <- xgb_class_preds$pred_prob[match(common_dates, xgb_class_preds$date)]
class_preds_aligned$XGBoost_class <- xgb_class_preds$pred_class[match(common_dates, xgb_class_preds$date)]

# HAR
class_preds_aligned$HAR_prob <- har_test_preds$pred_direction_prob[har_idx]
class_preds_aligned$HAR_class <- as.integer(har_test_preds$pred_direction_prob[har_idx] >= 0.5)

# GARCH(1,1) only
if (!is.null(garch_results[["GARCH"]])) {
  garch_preds <- garch_results[["GARCH"]]$test_predictions
  garch_idx <- match(common_dates, garch_preds$date)
  class_preds_aligned$GARCH_prob <- garch_preds$pred_direction_prob[garch_idx]
  class_preds_aligned$GARCH_class <- as.integer(garch_preds$pred_direction_prob[garch_idx] >= 0.5)
}

# Compute classification metrics
class_models <- c("XGBoost", "HAR")
if ("GARCH_prob" %in% colnames(class_preds_aligned)) {
  class_models <- c(class_models, "GARCH")
}

class_metrics_df <- data.frame()
for (model_name in class_models) {
  prob_col <- paste0(model_name, "_prob")
  class_col <- paste0(model_name, "_class")
  
  if (prob_col %in% colnames(class_preds_aligned)) {
    metrics <- calc_classification_metrics(
      class_preds_aligned$actual_direction,
      class_preds_aligned[[prob_col]],
      class_preds_aligned[[class_col]]
    )
    
    class_metrics_df <- rbind(class_metrics_df, data.frame(
      Model = model_name,
      Accuracy = metrics$Accuracy,
      Balanced_Acc = metrics$Balanced_Accuracy,
      Precision = metrics$Precision,
      Recall = metrics$Recall,
      F1 = metrics$F1,
      AUC = metrics$AUC,
      MCC = metrics$MCC,
      Brier = metrics$Brier
    ))
  }
}

# Sort by Accuracy
class_metrics_df <- class_metrics_df[order(-class_metrics_df$Accuracy), ]

cat("\n========== DIRECTION FORECAST METRICS ==========\n")
print(class_metrics_df, digits = 4)

# McNemar tests
mcnemar_results <- data.frame()
best_class_model <- class_metrics_df$Model[1]

for (model_name in class_models) {
  if (model_name != best_class_model) {
    class_col1 <- paste0(best_class_model, "_class")
    class_col2 <- paste0(model_name, "_class")
    
    if (class_col1 %in% colnames(class_preds_aligned) && class_col2 %in% colnames(class_preds_aligned)) {
      mc <- mcnemar_test(
        class_preds_aligned[[class_col1]],
        class_preds_aligned[[class_col2]],
        class_preds_aligned$actual_direction
      )
      
      mcnemar_results <- rbind(mcnemar_results, data.frame(
        Model1 = best_class_model,
        Model2 = model_name,
        Chi2_Stat = mc$statistic,
        P_Value = mc$p_value,
        Significant_05 = mc$p_value < 0.05,
        Model1_Better = mc$model1_better
      ))
    }
  }
}

cat("\n========== McNEMAR TESTS (Direction Forecasts) ==========\n")
print(mcnemar_results, digits = 4)

#------------------------------------------------------------------
# 6. VOLATILITY FORECASTING: GARCH VARIANTS (Meaningful Comparison)
#------------------------------------------------------------------

cat_progress("=" %>% rep(70) %>% paste(collapse = ""))
cat_progress("SECTION 3: VOLATILITY FORECASTING (GARCH Variants - QLIKE/MCS)")
cat_progress("=" %>% rep(70) %>% paste(collapse = ""))
cat_progress("Note: This is where GARCH variants genuinely differ!")

if (!is.null(volatility_metrics)) {
  cat("\n========== VOLATILITY FORECAST METRICS (QLIKE) ==========\n")
  print(volatility_metrics, digits = 4)
  
  cat("\nInterpretation:\n")
  cat("- QLIKE: Lower is better (robust to proxy noise per Patton 2011)\n")
  cat("- MZ_R2: Mincer-Zarnowitz R² for variance forecasts\n")
  cat("- MZ_Beta close to 1 indicates efficient forecasts\n")
} else {
  cat("\nVolatility metrics not available. Run updated HAR_GARCH_Models.R first.\n")
  
  # Construct basic QLIKE comparison if variance forecasts available in garch_results
  cat("\nComputing QLIKE from available GARCH results...\n")
  
  vol_metrics <- data.frame()
  
  for (model_name in names(garch_results)) {
    if (!is.null(garch_results[[model_name]]$test_predictions)) {
      preds <- garch_results[[model_name]]$test_predictions
      
      if ("pred_variance" %in% colnames(preds) && "actual_variance" %in% colnames(preds)) {
        # QLIKE
        valid_idx <- !is.na(preds$pred_variance) & !is.na(preds$actual_variance) &
          preds$pred_variance > 0
        
        qlike <- mean(log(preds$pred_variance[valid_idx]) + 
                        preds$actual_variance[valid_idx] / preds$pred_variance[valid_idx])
        
        vol_metrics <- rbind(vol_metrics, data.frame(
          Model = model_name,
          QLIKE = qlike
        ))
      }
    }
  }
  
  if (nrow(vol_metrics) > 0) {
    vol_metrics <- vol_metrics[order(vol_metrics$QLIKE), ]
    cat("\n========== QLIKE COMPARISON (GARCH Variants) ==========\n")
    print(vol_metrics, digits = 4)
  }
}

#------------------------------------------------------------------
# 7. REGIME-DEPENDENT PERFORMANCE
#------------------------------------------------------------------

cat_progress("=" %>% rep(70) %>% paste(collapse = ""))
cat_progress("SECTION 4: REGIME-DEPENDENT PERFORMANCE")
cat_progress("=" %>% rep(70) %>% paste(collapse = ""))

# Add current VIX to aligned predictions
reg_preds_aligned$current_vix <- xgb_reg_preds$current_vix[match(common_dates, xgb_reg_preds$date)]

# Define regimes
reg_preds_aligned$regime <- cut(
  reg_preds_aligned$current_vix,
  breaks = c(0, 15, 20, 30, Inf),
  labels = c("Low (<15)", "Medium (15-20)", "High (20-30)", "Very High (>30)")
)

# Regime performance for point forecasts
regime_reg_df <- data.frame()

for (model_name in point_models) {
  for (regime in levels(reg_preds_aligned$regime)) {
    idx <- reg_preds_aligned$regime == regime
    if (sum(idx) >= 20) {
      metrics <- calc_regression_metrics(
        reg_preds_aligned$actual_level[idx],
        reg_preds_aligned[[model_name]][idx]
      )
      
      regime_reg_df <- rbind(regime_reg_df, data.frame(
        Model = model_name,
        Regime = regime,
        N = sum(idx),
        RMSE = metrics$RMSE,
        MAE = metrics$MAE,
        R2 = metrics$R2
      ))
    }
  }
}

cat("\n========== REGIME PERFORMANCE (Point Forecasts) ==========\n")
print(regime_reg_df, digits = 4)

# Regime performance for direction
class_preds_aligned$regime <- reg_preds_aligned$regime

regime_class_df <- data.frame()

for (model_name in class_models) {
  prob_col <- paste0(model_name, "_prob")
  class_col <- paste0(model_name, "_class")
  
  if (prob_col %in% colnames(class_preds_aligned)) {
    for (regime in levels(class_preds_aligned$regime)) {
      idx <- class_preds_aligned$regime == regime
      if (sum(idx) >= 20) {
        metrics <- calc_classification_metrics(
          class_preds_aligned$actual_direction[idx],
          class_preds_aligned[[prob_col]][idx],
          class_preds_aligned[[class_col]][idx]
        )
        
        regime_class_df <- rbind(regime_class_df, data.frame(
          Model = model_name,
          Regime = regime,
          N = sum(idx),
          Accuracy = metrics$Accuracy,
          F1 = metrics$F1,
          AUC = metrics$AUC
        ))
      }
    }
  }
}

cat("\n========== REGIME PERFORMANCE (Direction Forecasts) ==========\n")
print(regime_class_df, digits = 4)

#------------------------------------------------------------------
# 8. ROLLING PERFORMANCE ANALYSIS
#------------------------------------------------------------------

cat_progress("Computing rolling performance metrics...")

window_size <- 63  # ~3 months

# Rolling RMSE
rolling_rmse <- data.frame(date = reg_preds_aligned$date)
for (model_name in point_models) {
  errors <- reg_preds_aligned$actual_level - reg_preds_aligned[[model_name]]
  rolling_rmse[[model_name]] <- zoo::rollapply(
    errors^2, width = window_size, FUN = function(x) sqrt(mean(x)),
    align = "right", fill = NA
  )
}

# Rolling Accuracy
rolling_accuracy <- data.frame(date = class_preds_aligned$date)
for (model_name in class_models) {
  class_col <- paste0(model_name, "_class")
  if (class_col %in% colnames(class_preds_aligned)) {
    correct <- class_preds_aligned[[class_col]] == class_preds_aligned$actual_direction
    rolling_accuracy[[model_name]] <- zoo::rollapply(
      correct, width = window_size, FUN = mean, align = "right", fill = NA
    )
  }
}

#------------------------------------------------------------------
# 9. PLOTS
#------------------------------------------------------------------

# Colour scheme
model_colors <- c(
  "XGBoost" = "#E41A1C",
  "HAR" = "#377EB8",
  "GARCH" = "#4DAF4A",
  "EGARCH" = "#984EA3",
  "GJR_GARCH" = "#FF7F00",
  "TGARCH" = "#A65628"
)

pdf("results/figures/26_point_forecast_comparison.pdf", width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Time series of predictions (last 250 days)
plot_idx <- max(1, nrow(reg_preds_aligned) - 250):nrow(reg_preds_aligned)

plot(reg_preds_aligned$date[plot_idx], reg_preds_aligned$actual_level[plot_idx],
     type = "l", lwd = 2, col = "black",
     xlab = "Date", ylab = "VIX Level",
     main = "Point Forecasts: XGBoost vs HAR vs GARCH(1,1)",
     ylim = range(c(reg_preds_aligned$actual_level[plot_idx],
                    reg_preds_aligned$XGBoost[plot_idx],
                    reg_preds_aligned$HAR[plot_idx]), na.rm = TRUE))

lines(reg_preds_aligned$date[plot_idx], reg_preds_aligned$XGBoost[plot_idx],
      col = model_colors["XGBoost"], lwd = 1.2)
lines(reg_preds_aligned$date[plot_idx], reg_preds_aligned$HAR[plot_idx],
      col = model_colors["HAR"], lwd = 1.2)
if ("GARCH" %in% colnames(reg_preds_aligned)) {
  lines(reg_preds_aligned$date[plot_idx], reg_preds_aligned$GARCH[plot_idx],
        col = model_colors["GARCH"], lwd = 1.2)
}

legend("topright", c("Actual", "XGBoost", "HAR", "GARCH(1,1)"),
       col = c("black", model_colors[c("XGBoost", "HAR", "GARCH")]),
       lwd = c(2, 1.2, 1.2, 1.2), cex = 0.8)

# RMSE comparison bar plot
barplot(reg_metrics_df$RMSE, names.arg = reg_metrics_df$Model,
        main = "RMSE Comparison (Lower is Better)",
        ylab = "RMSE", col = model_colors[reg_metrics_df$Model])

# R² comparison
barplot(reg_metrics_df$R2, names.arg = reg_metrics_df$Model,
        main = "R² Comparison (Higher is Better)",
        ylab = "R²", col = model_colors[reg_metrics_df$Model], ylim = c(0, 1))

# Error distribution boxplot
errors_df <- data.frame(
  XGBoost = reg_preds_aligned$actual_level - reg_preds_aligned$XGBoost,
  HAR = reg_preds_aligned$actual_level - reg_preds_aligned$HAR
)
if ("GARCH" %in% colnames(reg_preds_aligned)) {
  errors_df$GARCH <- reg_preds_aligned$actual_level - reg_preds_aligned$GARCH
}

boxplot(errors_df, col = model_colors[colnames(errors_df)],
        main = "Forecast Error Distribution",
        ylab = "Error (Actual - Predicted)")
abline(h = 0, col = "red", lty = 2)

dev.off()

pdf("results/figures/27_direction_forecast_comparison.pdf", width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# Accuracy comparison
barplot(class_metrics_df$Accuracy * 100, names.arg = class_metrics_df$Model,
        main = "Directional Accuracy (%)",
        ylab = "Accuracy (%)", col = model_colors[class_metrics_df$Model],
        ylim = c(0, 100))
abline(h = 50, col = "red", lty = 2)
text(0.5, 52, "Random: 50%", col = "red", cex = 0.8)

# AUC comparison
barplot(class_metrics_df$AUC, names.arg = class_metrics_df$Model,
        main = "AUC-ROC (Higher is Better)",
        ylab = "AUC", col = model_colors[class_metrics_df$Model],
        ylim = c(0, 1))
abline(h = 0.5, col = "red", lty = 2)

# F1 comparison
barplot(class_metrics_df$F1, names.arg = class_metrics_df$Model,
        main = "F1 Score",
        ylab = "F1", col = model_colors[class_metrics_df$Model],
        ylim = c(0, 1))

# ROC curves
if (requireNamespace("pROC", quietly = TRUE)) {
  plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1),
       xlab = "False Positive Rate", ylab = "True Positive Rate",
       main = "ROC Curves")
  abline(0, 1, col = "grey", lty = 2)
  
  for (i in seq_along(class_models)) {
    model_name <- class_models[i]
    prob_col <- paste0(model_name, "_prob")
    
    if (prob_col %in% colnames(class_preds_aligned)) {
      roc_obj <- pROC::roc(class_preds_aligned$actual_direction,
                           class_preds_aligned[[prob_col]], quiet = TRUE)
      lines(1 - roc_obj$specificities, roc_obj$sensitivities,
            col = model_colors[model_name], lwd = 2)
    }
  }
  legend("bottomright", class_models, col = model_colors[class_models], lwd = 2)
}

dev.off()

pdf("results/figures/28_rolling_performance.pdf", width = 14, height = 8)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Rolling RMSE
valid_idx <- !is.na(rolling_rmse$XGBoost)
plot(rolling_rmse$date[valid_idx], rolling_rmse$XGBoost[valid_idx],
     type = "l", col = model_colors["XGBoost"], lwd = 1.5,
     xlab = "Date", ylab = "Rolling RMSE (63-day)",
     main = "Rolling RMSE Over Time",
     ylim = range(c(rolling_rmse$XGBoost, rolling_rmse$HAR), na.rm = TRUE))
lines(rolling_rmse$date[valid_idx], rolling_rmse$HAR[valid_idx],
      col = model_colors["HAR"], lwd = 1.5)
if ("GARCH" %in% colnames(rolling_rmse)) {
  lines(rolling_rmse$date[valid_idx], rolling_rmse$GARCH[valid_idx],
        col = model_colors["GARCH"], lwd = 1.5)
}
legend("topright", point_models, col = model_colors[point_models], lwd = 1.5, cex = 0.8)

# Rolling Accuracy
valid_idx <- !is.na(rolling_accuracy$XGBoost)
plot(rolling_accuracy$date[valid_idx], rolling_accuracy$XGBoost[valid_idx] * 100,
     type = "l", col = model_colors["XGBoost"], lwd = 1.5,
     xlab = "Date", ylab = "Rolling Accuracy (%)",
     main = "Rolling Directional Accuracy Over Time",
     ylim = c(30, 80))
lines(rolling_accuracy$date[valid_idx], rolling_accuracy$HAR[valid_idx] * 100,
      col = model_colors["HAR"], lwd = 1.5)
if ("GARCH" %in% colnames(rolling_accuracy)) {
  lines(rolling_accuracy$date[valid_idx], rolling_accuracy$GARCH[valid_idx] * 100,
        col = model_colors["GARCH"], lwd = 1.5)
}
abline(h = 50, col = "red", lty = 2)
legend("bottomright", class_models, col = model_colors[class_models], lwd = 1.5, cex = 0.8)

dev.off()

pdf("results/figures/29_regime_performance.pdf", width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(5, 4, 3, 1))

# Regime RMSE heatmap-style bar plot
if (nrow(regime_reg_df) > 0) {
  regime_rmse_mat <- reshape2::dcast(regime_reg_df, Model ~ Regime, value.var = "RMSE")
  barplot(t(as.matrix(regime_rmse_mat[, -1])), beside = TRUE,
          names.arg = regime_rmse_mat$Model,
          col = viridis::viridis(4),
          main = "RMSE by Regime (Point Forecasts)",
          ylab = "RMSE", legend = TRUE,
          args.legend = list(x = "topright", cex = 0.7, title = "Regime"))
}

# Regime Accuracy
if (nrow(regime_class_df) > 0) {
  regime_acc_mat <- reshape2::dcast(regime_class_df, Model ~ Regime, value.var = "Accuracy")
  barplot(t(as.matrix(regime_acc_mat[, -1])) * 100, beside = TRUE,
          names.arg = regime_acc_mat$Model,
          col = viridis::viridis(4),
          main = "Accuracy by Regime (Direction Forecasts)",
          ylab = "Accuracy (%)", legend = TRUE,
          args.legend = list(x = "topright", cex = 0.7, title = "Regime"))
  abline(h = 50, col = "red", lty = 2)
}

# Statistical significance (DM tests)
if (nrow(dm_point_results) > 0) {
  barplot(-log10(dm_point_results$DM_MSE_P),
          names.arg = dm_point_results$Model2,
          main = sprintf("DM Test: %s vs Others (Point)", best_model),
          ylab = "-log10(p-value)",
          col = ifelse(dm_point_results$MSE_Sig_05, "darkgreen", "grey"))
  abline(h = -log10(0.05), col = "red", lty = 2)
  text(0.5, -log10(0.05) + 0.2, "p = 0.05", col = "red", cex = 0.8)
}

# McNemar significance
if (nrow(mcnemar_results) > 0) {
  barplot(-log10(mcnemar_results$P_Value),
          names.arg = mcnemar_results$Model2,
          main = sprintf("McNemar Test: %s vs Others (Direction)", best_class_model),
          ylab = "-log10(p-value)",
          col = ifelse(mcnemar_results$Significant_05, "darkgreen", "grey"))
  abline(h = -log10(0.05), col = "red", lty = 2)
}

dev.off()

#------------------------------------------------------------------
# 10. SUMMARY TABLES
#------------------------------------------------------------------

# Rankings
reg_metrics_df$Rank_RMSE <- rank(reg_metrics_df$RMSE)
reg_metrics_df$Rank_MAE <- rank(reg_metrics_df$MAE)
reg_metrics_df$Rank_R2 <- rank(-reg_metrics_df$R2)
reg_metrics_df$Avg_Rank <- (reg_metrics_df$Rank_RMSE + reg_metrics_df$Rank_MAE + reg_metrics_df$Rank_R2) / 3
reg_summary <- reg_metrics_df[order(reg_metrics_df$Avg_Rank), ]

class_metrics_df$Rank_Acc <- rank(-class_metrics_df$Accuracy)
class_metrics_df$Rank_AUC <- rank(-class_metrics_df$AUC)
class_metrics_df$Rank_F1 <- rank(-class_metrics_df$F1)
class_metrics_df$Avg_Rank <- (class_metrics_df$Rank_Acc + class_metrics_df$Rank_AUC + class_metrics_df$Rank_F1) / 3
class_summary <- class_metrics_df[order(class_metrics_df$Avg_Rank), ]

#------------------------------------------------------------------
# 11. SAVE RESULTS
#------------------------------------------------------------------

# Point forecasting results
write.csv(reg_metrics_df, "results/tables/comparison_point_forecast_metrics.csv", row.names = FALSE)
write.csv(dm_point_results, "results/tables/comparison_dm_point_tests.csv", row.names = FALSE)

# Direction forecasting results
write.csv(class_metrics_df, "results/tables/comparison_direction_forecast_metrics.csv", row.names = FALSE)
write.csv(mcnemar_results, "results/tables/comparison_mcnemar_tests.csv", row.names = FALSE)

# Regime performance
write.csv(regime_reg_df, "results/tables/comparison_regime_point.csv", row.names = FALSE)
write.csv(regime_class_df, "results/tables/comparison_regime_direction.csv", row.names = FALSE)

# Summary tables
write.csv(reg_summary, "results/tables/comparison_point_summary.csv", row.names = FALSE)
write.csv(class_summary, "results/tables/comparison_direction_summary.csv", row.names = FALSE)

# Aligned predictions
saveRDS(reg_preds_aligned, "results/models/comparison_point_predictions_aligned.rds")
saveRDS(class_preds_aligned, "results/models/comparison_direction_predictions_aligned.rds")
saveRDS(rolling_rmse, "results/models/comparison_rolling_rmse.rds")
saveRDS(rolling_accuracy, "results/models/comparison_rolling_accuracy.rds")

# Full comparison results
comparison_results <- list(
  point_forecasting = list(
    models_compared = c("XGBoost", "HAR", "GARCH(1,1)"),
    note = "GARCH variants use same AR(1) mean equation - only GARCH(1,1) included as benchmark",
    metrics = reg_metrics_df,
    summary = reg_summary,
    dm_tests = dm_point_results,
    regime_performance = regime_reg_df,
    rolling_rmse = rolling_rmse,
    best_model = best_model
  ),
  direction_forecasting = list(
    models_compared = c("XGBoost", "HAR", "GARCH(1,1)"),
    metrics = class_metrics_df,
    summary = class_summary,
    mcnemar_tests = mcnemar_results,
    regime_performance = regime_class_df,
    rolling_accuracy = rolling_accuracy,
    best_model = best_class_model
  ),
  volatility_forecasting = list(
    models_compared = c("GARCH", "EGARCH", "GJR_GARCH", "TGARCH"),
    note = "This is where GARCH variants genuinely differ - use QLIKE/MCS for comparison",
    metrics = volatility_metrics,
    see_file = "results/tables/volatility_forecast_metrics.csv"
  ),
  aligned_predictions = list(
    point = reg_preds_aligned,
    direction = class_preds_aligned
  ),
  common_dates = common_dates,
  n_observations = length(common_dates)
)

saveRDS(comparison_results, "results/models/model_comparison_full_results.rds")

#------------------------------------------------------------------
# 12. FINAL SUMMARY
#------------------------------------------------------------------

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("FINAL COMPARISON SUMMARY\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\n[1] POINT FORECASTING: XGBoost vs HAR vs GARCH(1,1)\n")
cat("    (Other GARCH variants omitted - same AR(1) mean equation)\n\n")
print(reg_summary[, c("Model", "RMSE", "MAE", "R2", "Avg_Rank")], digits = 4)

cat("\n[2] DIRECTION FORECASTING: XGBoost vs HAR vs GARCH(1,1)\n\n")
print(class_summary[, c("Model", "Accuracy", "AUC", "F1", "Avg_Rank")], digits = 4)

cat("\n[3] VOLATILITY FORECASTING: GARCH Variants (QLIKE/MCS)\n")
cat("    See: results/tables/volatility_forecast_metrics.csv\n")
if (!is.null(volatility_metrics)) {
  print(volatility_metrics[, c("Model", "QLIKE", "MZ_R2")], digits = 4)
}

cat("\n[4] STATISTICAL SIGNIFICANCE\n")
cat(sprintf("    Best point model: %s\n", best_model))
if (nrow(dm_point_results) > 0) {
  sig_dm <- sum(dm_point_results$MSE_Sig_05)
  cat(sprintf("    DM tests significant at 5%%: %d/%d\n", sig_dm, nrow(dm_point_results)))
}

cat(sprintf("    Best direction model: %s\n", best_class_model))
if (nrow(mcnemar_results) > 0) {
  sig_mc <- sum(mcnemar_results$Significant_05)
  cat(sprintf("    McNemar tests significant at 5%%: %d/%d\n", sig_mc, nrow(mcnemar_results)))
}

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat_progress("All results saved to results/models/ and results/tables/")
cat_progress("Plots saved to results/figures/")
cat(paste(rep("=", 80), collapse = ""), "\n")

################################################################################
# END OF SCRIPT
################################################################################