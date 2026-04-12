################################################################################
# RV_XGBoost_Regression.R
# XGBoost Regression for 22-Day Forward Realised Volatility Prediction
#
# OBJECTIVE:
# ----------
# Predict E^P[RV_{t+1,t+23} | F_t] directly (continuous target) for comparison
# with VIX_{t+1} to estimate the Variance Risk Premium:
#   VRP_{t+1} = VIX_{t+1} - E[RV_{t+1,t+23} | F_t]
#
# METHODOLOGY:
# ------------
# 1. XGBoost with reg:squarederror objective
# 2. Bayesian hyperparameter optimisation with purged CV
# 3. Mincer-Zarnowitz regression to assess forecast quality
# 4. Comparison against VIX as benchmark forecast
# 5. VRP estimation and trading signal generation
#
################################################################################

#------------------------------------------------------------------
# 0. SETUP
#------------------------------------------------------------------

source("Setup.R")

cat_progress(paste(rep("=", 70), collapse = ""))
cat_progress("XGBoost Regression: 22-Day Forward RV Prediction")
cat_progress(paste(rep("=", 70), collapse = ""))

# Create output directories
rv_reg_dirs <- c("results/models/rv_regression", 
                 "results/tables/rv_regression",
                 "results/figures/rv_regression")
for (d in rv_reg_dirs) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

#------------------------------------------------------------------
# 1. LOAD FEATURES
#------------------------------------------------------------------

cat_progress("Loading RV feature data...")

train_features <- readRDS("data/rv_forward/features_rv_train.rds")
test_features <- readRDS("data/rv_forward/features_rv_test.rds")
feature_cols <- readRDS("data/rv_forward/feature_columns_rv.rds")
feature_categories <- readRDS("data/rv_forward/feature_categories_rv.rds")
target_config <- readRDS("results/models/rv_forward/target_config_rv.rds")

cat_progress(sprintf("Training set: %d observations", nrow(train_features)))
cat_progress(sprintf("Test set: %d observations", nrow(test_features)))
cat_progress(sprintf("Features: %d", length(feature_cols)))

#------------------------------------------------------------------
# 2. LEAKAGE PREVENTION
#------------------------------------------------------------------

cat_progress("Performing leakage checks...")

# Forbidden patterns for regression target
forbidden_patterns <- c(
  "^target_",           # Any target variable
  "^rv_cc$",            # Contemporaneous RV
  "^rv_cc_next$",       # Target itself
  "rv_change",          # Derived from target
  "^vix_close$"         # Current VIX (we use lagged only)
)

leakage_check <- sapply(feature_cols, function(f) {
  any(sapply(forbidden_patterns, function(p) grepl(p, f)))
})

if (any(leakage_check)) {
  cat_progress("LEAKAGE DETECTED - removing features:")
  print(feature_cols[leakage_check])
  feature_cols <- feature_cols[!leakage_check]
  cat_progress(sprintf("Removed %d leakage features", sum(leakage_check)))
}

# Verify temporal ordering
stopifnot(all(diff(train_features$date) >= 0))
stopifnot(all(diff(test_features$date) >= 0))
stopifnot(max(train_features$date) < min(test_features$date))

cat_progress("Data integrity checks passed.")

#------------------------------------------------------------------
# 3. FEATURE PREPROCESSING
#------------------------------------------------------------------

cat_progress("Preprocessing features...")

# 3.1 Remove features with NA values
X_train_raw <- as.matrix(train_features[, ..feature_cols])

na_counts <- colSums(is.na(X_train_raw))
if (any(na_counts > 0)) {
  cat_progress(sprintf("Removing %d features with NA values", sum(na_counts > 0)))
  feature_cols <- feature_cols[na_counts == 0]
  X_train_raw <- as.matrix(train_features[, ..feature_cols])
}

# 3.2 Remove highly correlated features (keep higher target correlation)
cor_threshold <- 0.9999

cor_matrix <- cor(X_train_raw, use = "pairwise.complete.obs")
cor_matrix[is.na(cor_matrix)] <- 0

high_cor_pairs <- which(abs(cor_matrix) > cor_threshold & upper.tri(cor_matrix), arr.ind = TRUE)

if (nrow(high_cor_pairs) > 0) {
  # Correlation with target (rv_cc_next)
  target_cors <- abs(cor(X_train_raw, train_features$rv_cc_next, use = "complete.obs"))
  
  features_to_remove <- c()
  for (i in 1:nrow(high_cor_pairs)) {
    f1 <- feature_cols[high_cor_pairs[i, 1]]
    f2 <- feature_cols[high_cor_pairs[i, 2]]
    
    # Remove feature with lower target correlation
    if (target_cors[high_cor_pairs[i, 1]] < target_cors[high_cor_pairs[i, 2]]) {
      features_to_remove <- c(features_to_remove, f1)
    } else {
      features_to_remove <- c(features_to_remove, f2)
    }
  }
  
  features_to_remove <- unique(features_to_remove)
  feature_cols <- setdiff(feature_cols, features_to_remove)
  
  cat_progress(sprintf("Removed %d highly correlated features (r > %.4f)",
                       length(features_to_remove), cor_threshold))
}

# 3.3 Prepare final matrices
# Remove rows with NA target
train_features <- train_features[!is.na(rv_cc_next)]
test_features <- test_features[!is.na(rv_cc_next)]

X_train <- as.matrix(train_features[, ..feature_cols])
X_test <- as.matrix(test_features[, ..feature_cols])

# Target: rv_cc_next (RV_{t+1, t+23})
y_train <- train_features$rv_cc_next
y_test <- test_features$rv_cc_next

# Store dates and auxiliary variables for analysis
train_dates <- train_features$date
test_dates <- test_features$date

# VIX at t+1 for comparison (shift vix_close forward by 1)
# We need vix_close at t+1, but we have vix_close at t
# For test set evaluation, we need actual VIX_{t+1}
# This should be aligned with the target date
test_vix <- test_features$vix_close  # VIX at prediction time t

cat_progress(sprintf("Final feature count: %d", length(feature_cols)))
cat_progress(sprintf("Training observations: %d", length(y_train)))
cat_progress(sprintf("Test observations: %d", length(y_test)))

# Summary statistics for target
cat_progress(sprintf("Training target: mean=%.2f, sd=%.2f, range=[%.2f, %.2f]",
                     mean(y_train), sd(y_train), min(y_train), max(y_train)))
cat_progress(sprintf("Test target: mean=%.2f, sd=%.2f, range=[%.2f, %.2f]",
                     mean(y_test), sd(y_test), min(y_test), max(y_test)))

#------------------------------------------------------------------
# 4. PURGED K-FOLD CV FOR TIME SERIES
#------------------------------------------------------------------

cat_progress("Setting up purged cross-validation...")

create_purged_kfold <- function(n_obs, n_folds = 5, purge_days = 5, embargo_days = 22) {
  # embargo_days = 22 to ensure no overlap with 22-day RV window
  
  fold_size <- floor(n_obs / n_folds)
  folds <- list()
  
  for (k in 1:n_folds) {
    val_start <- (k - 1) * fold_size + 1
    val_end <- min(k * fold_size, n_obs)
    val_idx <- val_start:val_end
    
    # Purge: remove observations just before validation (information leakage)
    purge_start <- max(1, val_start - purge_days)
    # Embargo: remove observations just after validation (target overlap)
    embargo_end <- min(n_obs, val_end + embargo_days)
    
    excluded <- purge_start:embargo_end
    train_idx <- setdiff(1:n_obs, excluded)
    
    folds[[k]] <- list(
      train = train_idx,
      val = val_idx,
      purge_removed = length(purge_start:(val_start - 1)),
      embargo_removed = length((val_end + 1):embargo_end)
    )
  }
  
  return(folds)
}

create_cpcv_folds <- function(n_obs, n_groups = 6, n_test_groups = 2, 
                              purge_days = 5, embargo_days = 22) {
  # Combinatorial Purged Cross-Validation
  
  group_size <- floor(n_obs / n_groups)
  groups <- lapply(1:n_groups, function(g) {
    start_idx <- (g - 1) * group_size + 1
    end_idx <- min(g * group_size, n_obs)
    start_idx:end_idx
  })
  
  test_combos <- combn(1:n_groups, n_test_groups, simplify = FALSE)
  
  folds <- list()
  for (i in seq_along(test_combos)) {
    test_groups <- test_combos[[i]]
    train_groups <- setdiff(1:n_groups, test_groups)
    
    val_idx <- unlist(groups[test_groups])
    
    train_idx <- c()
    for (tg in train_groups) {
      group_idx <- groups[[tg]]
      
      keep <- rep(TRUE, length(group_idx))
      
      for (test_g in test_groups) {
        test_start <- min(groups[[test_g]])
        test_end <- max(groups[[test_g]])
        
        purge_mask <- group_idx >= (test_start - purge_days) & group_idx < test_start
        embargo_mask <- group_idx > test_end & group_idx <= (test_end + embargo_days)
        
        keep <- keep & !purge_mask & !embargo_mask
      }
      
      train_idx <- c(train_idx, group_idx[keep])
    }
    
    folds[[i]] <- list(
      train = sort(train_idx),
      val = sort(val_idx)
    )
  }
  
  return(folds)
}

n_train <- nrow(X_train)
purge_days <- 5
embargo_days <- 22  # Critical: RV window is 22 days

purged_folds <- create_purged_kfold(n_train, n_folds = 5, 
                                    purge_days = purge_days, 
                                    embargo_days = embargo_days)

cpcv_folds <- create_cpcv_folds(n_train, n_groups = 6, n_test_groups = 2,
                                purge_days = purge_days, 
                                embargo_days = embargo_days)

# Verify no overlap
for (i in seq_along(purged_folds)) {
  overlap <- intersect(purged_folds[[i]]$train, purged_folds[[i]]$val)
  if (length(overlap) > 0) {
    stop(sprintf("Fold %d has train/val overlap!", i))
  }
}

cat_progress(sprintf("Created %d purged folds and %d CPCV folds", 
                     length(purged_folds), length(cpcv_folds)))

#------------------------------------------------------------------
# 5. BAYESIAN HYPERPARAMETER OPTIMISATION
#------------------------------------------------------------------

cat_progress("Starting Bayesian hyperparameter optimisation...")

# 5.1 Search bounds (adjusted for regression)
bounds_reg <- list(
  max_depth = c(2L, 8L),
  min_child_weight = c(20L, 100L),
  subsample = c(0.5, 0.95),
  colsample_bytree = c(0.4, 0.95),
  colsample_bynode = c(0.5, 1.0),
  eta = c(0.005, 0.125),
  gamma = c(0, 0.5),
  lambda = c(1, 20),
  alpha = c(0, 1)
  # No max_delta_step for regression
)

# Use first 3 folds for inner CV in Bayesian opt
inner_folds <- purged_folds[1:3]

# Scoring function: minimise RMSE (return negative for maximisation)
scoring_function_reg <- function(max_depth, min_child_weight, subsample, 
                                 colsample_bytree, colsample_bynode, eta, 
                                 gamma, lambda, alpha) {
  
  max_depth <- round(max_depth)
  min_child_weight <- round(min_child_weight)
  
  params <- list(
    booster = "gbtree",
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    colsample_bynode = colsample_bynode,
    eta = eta,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    nthread = 2
  )
  
  cv_rmses <- numeric(length(inner_folds))
  
  for (i in seq_along(inner_folds)) {
    fold <- inner_folds[[i]]
    
    X_tr <- X_train[fold$train, , drop = FALSE]
    y_tr <- y_train[fold$train]
    X_val <- X_train[fold$val, , drop = FALSE]
    y_val <- y_train[fold$val]
    
    dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
    dval <- xgb.DMatrix(data = X_val, label = y_val)
    
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 10,
      watchlist = list(val = dval),
      early_stopping_rounds = 4,
      verbose = 0
    )
    
    pred <- predict(model, dval)
    cv_rmses[i] <- sqrt(mean((pred - y_val)^2))
  }
  
  # Return negative RMSE (bayesOpt maximises)
  list(Score = -mean(cv_rmses, na.rm = TRUE))
}

# Export for parallel processing
clusterExport(cl, c("X_train", "y_train", "inner_folds"))
clusterEvalQ(cl, {
  library(xgboost)
})

set.seed(42)

bayes_opt_result <- bayesOpt(
  FUN = scoring_function_reg,
  bounds = bounds_reg,
  initPoints = 98,
  iters.n = 42,
  iters.k = 7,
  acq = "ei",
  eps = 0.225,
  gsPoints = 15500,
  acqThresh = 0.5,
  parallel = TRUE,
  verbose = 2,
  plotProgress = TRUE
)

# Save optimisation progress plot
pdf("results/figures/rv_regression/01_bayesian_opt_progress.pdf", width = 12, height = 8)
plot(bayes_opt_result)
dev.off()

# Extract best parameters
best_params <- getBestPars(bayes_opt_result)
best_params$max_depth <- round(best_params$max_depth)
best_params$min_child_weight <- round(best_params$min_child_weight)

cat_progress("Best hyperparameters found:")
print(best_params)

#------------------------------------------------------------------
# 6. OUTER CROSS-VALIDATION WITH OPTIMAL HYPERPARAMETERS
#------------------------------------------------------------------

cat_progress("Running outer CV with optimal hyperparameters...")

final_params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = best_params$max_depth,
  min_child_weight = best_params$min_child_weight,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  colsample_bynode = best_params$colsample_bynode,
  eta = best_params$eta,
  gamma = best_params$gamma,
  lambda = best_params$lambda,
  alpha = best_params$alpha
)

cv_results <- list()
cv_predictions <- data.frame()
cv_metrics <- data.frame()

for (i in seq_along(cpcv_folds)) {
  fold <- cpcv_folds[[i]]
  
  X_tr <- X_train[fold$train, , drop = FALSE]
  y_tr <- y_train[fold$train]
  X_val <- X_train[fold$val, , drop = FALSE]
  y_val <- y_train[fold$val]
  val_dates <- train_dates[fold$val]
  
  dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
  dval <- xgb.DMatrix(data = X_val, label = y_val)
  
  model <- xgb.train(
    params = final_params,
    data = dtrain,
    nrounds = 1000,
    watchlist = list(train = dtrain, val = dval),
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  best_iter <- tryCatch({
    bi <- model$best_iteration
    if (is.null(bi) || length(bi) == 0) {
      as.integer(xgb.attr(model, "best_iteration"))
    } else {
      bi
    }
  }, error = function(e) NA_integer_)
  if (is.null(best_iter) || length(best_iter) == 0 || is.na(best_iter)) {
    best_iter <- 1000L
  }
  
  # Predictions
  pred <- predict(model, dval)
  
  # Store predictions
  fold_preds <- data.frame(
    fold = i,
    date = val_dates,
    actual = y_val,
    predicted = pred
  )
  cv_predictions <- rbind(cv_predictions, fold_preds)
  
  # Calculate regression metrics
  residuals <- y_val - pred
  mse <- mean(residuals^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(residuals))
  mape <- mean(abs(residuals / (y_val + 1e-10))) * 100
  r2 <- 1 - sum(residuals^2) / sum((y_val - mean(y_val))^2)
  
  # Directional accuracy (for comparison with classification)
  # Predict direction: is RV increasing?
  actual_direction <- as.integer(y_val > train_features$rv_cc[fold$val])
  pred_direction <- as.integer(pred > train_features$rv_cc[fold$val])
  direction_acc <- mean(actual_direction == pred_direction, na.rm = TRUE)
  
  fold_metrics <- data.frame(
    fold = i,
    n_train = length(y_tr),
    n_val = length(y_val),
    rmse = rmse,
    mae = mae,
    mape = mape,
    r2 = r2,
    direction_accuracy = direction_acc,
    best_iteration = best_iter
  )
  cv_metrics <- rbind(cv_metrics, fold_metrics)
  
  cv_results[[i]] <- list(
    model = model,
    predictions = fold_preds,
    metrics = fold_metrics
  )
  
  cat_progress(sprintf("Fold %d: RMSE=%.4f, MAE=%.4f, R²=%.4f, Dir.Acc=%.4f",
                       i, rmse, mae, r2, direction_acc))
}

# CV Summary
cv_summary <- data.frame(
  Metric = c("RMSE", "MAE", "MAPE", "R²", "Direction Accuracy"),
  Mean = c(mean(cv_metrics$rmse),
           mean(cv_metrics$mae),
           mean(cv_metrics$mape),
           mean(cv_metrics$r2),
           mean(cv_metrics$direction_accuracy)),
  SD = c(sd(cv_metrics$rmse),
         sd(cv_metrics$mae),
         sd(cv_metrics$mape),
         sd(cv_metrics$r2),
         sd(cv_metrics$direction_accuracy))
)

cat_progress("Cross-validation summary:")
print(cv_summary, digits = 4)

#------------------------------------------------------------------
# 7. FINAL MODEL TRAINING ON FULL TRAINING SET
#------------------------------------------------------------------

cat_progress("Training final model on full training set...")

dtrain_full <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

optimal_nrounds <- round(mean(cv_metrics$best_iteration) * 1.1)

final_model <- xgb.train(
  params = final_params,
  data = dtrain_full,
  nrounds = optimal_nrounds,
  watchlist = list(train = dtrain_full, test = dtest),
  verbose = 2
)

cat_progress(sprintf("Final model trained with %d rounds", optimal_nrounds))

#------------------------------------------------------------------
# 8. TEST SET EVALUATION
#------------------------------------------------------------------

cat_progress("Evaluating on test set...")

test_pred <- predict(final_model, dtest)

# 8.1 Basic regression metrics
test_residuals <- y_test - test_pred
test_mse <- mean(test_residuals^2)
test_rmse <- sqrt(test_mse)
test_mae <- mean(abs(test_residuals))
test_mape <- mean(abs(test_residuals / (y_test + 1e-10))) * 100
test_r2 <- 1 - sum(test_residuals^2) / sum((y_test - mean(y_test))^2)
test_adj_r2 <- 1 - (1 - test_r2) * (length(y_test) - 1) / (length(y_test) - length(feature_cols) - 1)

# 8.2 Directional accuracy
actual_rv_current <- test_features$rv_cc
test_actual_direction <- as.integer(y_test > actual_rv_current)
test_pred_direction <- as.integer(test_pred > actual_rv_current)
test_direction_acc <- mean(test_actual_direction == test_pred_direction, na.rm = TRUE)

# 8.3 QLIKE loss (heteroskedasticity-robust, standard for volatility forecasting)
# QLIKE = mean(actual/pred - log(actual/pred) - 1)
qlike <- mean(y_test / (test_pred + 1e-10) - log(y_test / (test_pred + 1e-10)) - 1)

# 8.4 Compile test metrics
test_metrics <- data.frame(
  Metric = c("RMSE", "MAE", "MAPE (%)", "R²", "Adjusted R²", 
             "Direction Accuracy", "QLIKE"),
  Value = c(test_rmse, test_mae, test_mape, test_r2, test_adj_r2,
            test_direction_acc, qlike)
)

cat_progress("Test Set Metrics:")
print(test_metrics, digits = 4)

#------------------------------------------------------------------
# 9. VIX BENCHMARK COMPARISON (MINCER-ZARNOWITZ)
#------------------------------------------------------------------

cat_progress("Comparing XGBoost forecast to VIX benchmark...")

# VIX as forecast of future RV
# Note: VIX_t is the 30-day implied vol, we compare to RV_{t+1,t+23}
# We use VIX at time t as the forecast

# Get VIX aligned with test set (lagged VIX as forecast)
vix_forecast <- test_features$vix_lag_1  # VIX one day before prediction

# 9.1 VIX forecast metrics
vix_residuals <- y_test - vix_forecast
vix_rmse <- sqrt(mean(vix_residuals^2, na.rm = TRUE))
vix_mae <- mean(abs(vix_residuals), na.rm = TRUE)
vix_mape <- mean(abs(vix_residuals / (y_test + 1e-10)), na.rm = TRUE) * 100
vix_r2 <- 1 - sum(vix_residuals^2, na.rm = TRUE) / sum((y_test - mean(y_test, na.rm = TRUE))^2)
vix_qlike <- mean(y_test / (vix_forecast + 1e-10) - log(y_test / (vix_forecast + 1e-10)) - 1, na.rm = TRUE)

# 9.2 Mincer-Zarnowitz regression: RV_actual = alpha + beta * forecast + epsilon
# Efficient forecast: alpha = 0, beta = 1

# For XGBoost
mz_xgb <- lm(y_test ~ test_pred)
mz_xgb_summary <- summary(mz_xgb)
mz_xgb_alpha <- coef(mz_xgb)[1]
mz_xgb_beta <- coef(mz_xgb)[2]

# Test for unbiasedness: H0: alpha = 0, beta = 1
# Wald test
library(car)
mz_xgb_wald <- linearHypothesis(mz_xgb, c("(Intercept) = 0", "test_pred = 1"))

# For VIX
valid_vix_idx <- !is.na(vix_forecast)
mz_vix <- lm(y_test[valid_vix_idx] ~ vix_forecast[valid_vix_idx])
mz_vix_summary <- summary(mz_vix)
mz_vix_alpha <- coef(mz_vix)[1]
mz_vix_beta <- coef(mz_vix)[2]

# 9.3 Comparison table
benchmark_comparison <- data.frame(
  Model = c("XGBoost", "VIX"),
  RMSE = c(test_rmse, vix_rmse),
  MAE = c(test_mae, vix_mae),
  MAPE = c(test_mape, vix_mape),
  R2 = c(test_r2, vix_r2),
  QLIKE = c(qlike, vix_qlike),
  MZ_Alpha = c(mz_xgb_alpha, mz_vix_alpha),
  MZ_Beta = c(mz_xgb_beta, mz_vix_beta),
  MZ_R2 = c(mz_xgb_summary$r.squared, mz_vix_summary$r.squared)
)

cat_progress("Forecast Comparison (XGBoost vs VIX):")
print(benchmark_comparison, digits = 4)

# RMSE improvement
rmse_improvement <- (vix_rmse - test_rmse) / vix_rmse * 100
cat_progress(sprintf("RMSE improvement over VIX: %.2f%%", rmse_improvement))

#------------------------------------------------------------------
# 10. VARIANCE RISK PREMIUM ESTIMATION
#------------------------------------------------------------------

cat_progress("Computing Variance Risk Premium estimates...")

# VRP = VIX - E[RV]
# Using XGBoost forecast
vrp_xgb <- test_features$vix_close - test_pred

# Using VIX as forecast (VRP = 0 by construction if using VIX itself)
# Instead, compare VIX to actual RV (ex-post VRP)
vrp_actual <- test_features$vix_close - y_test

# VRP statistics
vrp_summary <- data.frame(
  Measure = c("XGBoost VRP", "Actual VRP (ex-post)"),
  Mean = c(mean(vrp_xgb, na.rm = TRUE), mean(vrp_actual, na.rm = TRUE)),
  SD = c(sd(vrp_xgb, na.rm = TRUE), sd(vrp_actual, na.rm = TRUE)),
  Pct_Positive = c(mean(vrp_xgb > 0, na.rm = TRUE) * 100, 
                   mean(vrp_actual > 0, na.rm = TRUE) * 100),
  T_Stat = c(mean(vrp_xgb, na.rm = TRUE) / (sd(vrp_xgb, na.rm = TRUE) / sqrt(sum(!is.na(vrp_xgb)))),
             mean(vrp_actual, na.rm = TRUE) / (sd(vrp_actual, na.rm = TRUE) / sqrt(sum(!is.na(vrp_actual)))))
)

cat_progress("VRP Summary:")
print(vrp_summary, digits = 4)

# Correlation between XGBoost VRP estimate and actual VRP
vrp_correlation <- cor(vrp_xgb, vrp_actual, use = "complete.obs")
cat_progress(sprintf("Correlation(XGBoost VRP, Actual VRP): %.4f", vrp_correlation))

#------------------------------------------------------------------
# 11. FEATURE IMPORTANCE AND SHAP
#------------------------------------------------------------------

cat_progress("Computing feature importance...")

# 11.1 Gain-based importance
importance_gain <- xgb.importance(model = final_model, feature_names = feature_cols)
importance_gain <- importance_gain[order(-Gain)]

cat_progress("Top 20 features by Gain:")
print(head(importance_gain, 20))

# 11.2 SHAP values
shap_sample_size <- min(10000, nrow(X_test))
set.seed(42)
shap_idx <- sample(1:nrow(X_test), shap_sample_size)

shap_values <- predict(final_model, X_test[shap_idx, ], predcontrib = TRUE)
shap_matrix <- shap_values[, -ncol(shap_values)]  # Remove BIAS column
colnames(shap_matrix) <- feature_cols

shap_importance <- data.frame(
  Feature = feature_cols,
  Mean_Abs_SHAP = colMeans(abs(shap_matrix))
)
shap_importance <- shap_importance[order(-shap_importance$Mean_Abs_SHAP), ]

cat_progress("Top 20 features by Mean |SHAP|:")
print(head(shap_importance, 20))

#------------------------------------------------------------------
# 12. STATISTICAL TESTS
#------------------------------------------------------------------

cat_progress("Performing statistical tests...")

# 12.1 Diebold-Mariano test: XGBoost vs VIX
# H0: equal predictive accuracy
dm_test <- function(e1, e2, h = 1) {
  # e1, e2 are forecast errors
  d <- e1^2 - e2^2
  n <- length(d)
  d_bar <- mean(d, na.rm = TRUE)
  
  # HAC variance (Newey-West)
  gamma_0 <- var(d, na.rm = TRUE)
  
  # Long-run variance approximation
  var_d <- gamma_0 / n
  
  dm_stat <- d_bar / sqrt(var_d)
  p_value <- 2 * pnorm(-abs(dm_stat))
  
  list(statistic = dm_stat, p_value = p_value)
}

dm_result <- dm_test(test_residuals, vix_residuals[valid_vix_idx])
cat_progress(sprintf("Diebold-Mariano test: stat=%.4f, p-value=%.4f",
                     dm_result$statistic, dm_result$p_value))

# 12.2 Bootstrap confidence interval for RMSE
set.seed(42)
n_bootstrap <- 10000
bootstrap_rmses <- numeric(n_bootstrap)

for (b in 1:n_bootstrap) {
  boot_idx <- sample(1:length(y_test), replace = TRUE)
  y_boot <- y_test[boot_idx]
  pred_boot <- test_pred[boot_idx]
  bootstrap_rmses[b] <- sqrt(mean((y_boot - pred_boot)^2))
}

rmse_ci <- quantile(bootstrap_rmses, c(0.025, 0.975))
cat_progress(sprintf("RMSE 95%% CI: [%.4f, %.4f]", rmse_ci[1], rmse_ci[2]))

# 12.3 Bootstrap CI for R²
bootstrap_r2s <- numeric(n_bootstrap)

for (b in 1:n_bootstrap) {
  boot_idx <- sample(1:length(y_test), replace = TRUE)
  y_boot <- y_test[boot_idx]
  pred_boot <- test_pred[boot_idx]
  ss_res <- sum((y_boot - pred_boot)^2)
  ss_tot <- sum((y_boot - mean(y_boot))^2)
  bootstrap_r2s[b] <- 1 - ss_res / ss_tot
}

r2_ci <- quantile(bootstrap_r2s, c(0.025, 0.975))
cat_progress(sprintf("R² 95%% CI: [%.4f, %.4f]", r2_ci[1], r2_ci[2]))

#------------------------------------------------------------------
# 13. REGIME-CONDITIONAL PERFORMANCE
#------------------------------------------------------------------

cat_progress("Analysing regime-conditional performance...")

# Create regime indicator based on VIX level
test_results <- data.frame(
  date = test_dates,
  actual = y_test,
  predicted = test_pred,
  vix = test_features$vix_close,
  vrp_xgb = vrp_xgb,
  vrp_actual = vrp_actual
)

# VIX regimes: Low (<15), Medium (15-25), High (>25)
test_results$vix_regime <- cut(test_results$vix,
                               breaks = c(0, 15, 25, Inf),
                               labels = c("Low", "Medium", "High"))

regime_performance <- data.frame()

for (r in c("Low", "Medium", "High")) {
  regime_mask <- test_results$vix_regime == r
  if (sum(regime_mask, na.rm = TRUE) > 10) {
    y_r <- test_results$actual[regime_mask]
    pred_r <- test_results$predicted[regime_mask]
    
    rmse_r <- sqrt(mean((y_r - pred_r)^2))
    mae_r <- mean(abs(y_r - pred_r))
    r2_r <- 1 - sum((y_r - pred_r)^2) / sum((y_r - mean(y_r))^2)
    
    regime_performance <- rbind(regime_performance, data.frame(
      regime = r,
      n = sum(regime_mask, na.rm = TRUE),
      rmse = rmse_r,
      mae = mae_r,
      r2 = r2_r
    ))
  }
}

cat_progress("Performance by VIX Regime:")
print(regime_performance, digits = 4)

#------------------------------------------------------------------
# 14. TEMPORAL STABILITY ANALYSIS
#------------------------------------------------------------------

cat_progress("Analysing temporal stability...")

window_size <- 63  # ~3 months rolling window
rolling_metrics <- data.frame()

for (i in window_size:nrow(test_results)) {
  window_idx <- (i - window_size + 1):i
  y_w <- test_results$actual[window_idx]
  pred_w <- test_results$predicted[window_idx]
  
  rmse_w <- sqrt(mean((y_w - pred_w)^2))
  r2_w <- 1 - sum((y_w - pred_w)^2) / sum((y_w - mean(y_w))^2)
  
  rolling_metrics <- rbind(rolling_metrics, data.frame(
    date = test_results$date[i],
    rmse = rmse_w,
    r2 = r2_w
  ))
}

mean_rolling_rmse <- mean(rolling_metrics$rmse)
sd_rolling_rmse <- sd(rolling_metrics$rmse)
degradation_threshold <- mean_rolling_rmse + 2 * sd_rolling_rmse

degradation_periods <- rolling_metrics$date[rolling_metrics$rmse > degradation_threshold]

cat_progress(sprintf("Mean rolling RMSE: %.4f (SD: %.4f)", mean_rolling_rmse, sd_rolling_rmse))
cat_progress(sprintf("Degradation periods (RMSE > %.4f): %d", 
                     degradation_threshold, length(degradation_periods)))

#------------------------------------------------------------------
# 15. VISUALISATIONS
#------------------------------------------------------------------

cat_progress("Generating plots...")

# 15.1 Actual vs Predicted scatter
pdf("results/figures/rv_regression/02_actual_vs_predicted.pdf", width = 12, height = 6)

par(mfrow = c(1, 2))

# XGBoost
plot(test_pred, y_test, pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = "XGBoost Predicted RV", ylab = "Actual RV",
     main = sprintf("XGBoost: R² = %.4f", test_r2))
abline(0, 1, col = "red", lwd = 2)
abline(mz_xgb, col = "blue", lty = 2)
legend("topleft", c("Perfect forecast", sprintf("MZ: α=%.2f, β=%.2f", mz_xgb_alpha, mz_xgb_beta)),
       col = c("red", "blue"), lty = c(1, 2), cex = 0.8)

# VIX
plot(vix_forecast[valid_vix_idx], y_test[valid_vix_idx], pch = 16, cex = 0.5, col = rgb(1, 0, 0, 0.3),
     xlab = "VIX Forecast", ylab = "Actual RV",
     main = sprintf("VIX Benchmark: R² = %.4f", vix_r2))
abline(0, 1, col = "red", lwd = 2)
abline(mz_vix, col = "blue", lty = 2)
legend("topleft", c("Perfect forecast", sprintf("MZ: α=%.2f, β=%.2f", mz_vix_alpha, mz_vix_beta)),
       col = c("red", "blue"), lty = c(1, 2), cex = 0.8)

dev.off()

# 15.2 Time series of predictions
pdf("results/figures/rv_regression/03_predictions_timeseries.pdf", width = 14, height = 8)

par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))

# Top: Actual vs Predicted RV
plot(test_dates, y_test, type = "l", col = "black", lwd = 1,
     xlab = "", ylab = "Volatility (%)",
     main = "Actual RV vs XGBoost Forecast")
lines(test_dates, test_pred, col = "blue", lwd = 1)
lines(test_dates, vix_forecast, col = "red", lwd = 1, lty = 2)
legend("topright", c("Actual RV", "XGBoost Forecast", "VIX"),
       col = c("black", "blue", "red"), lwd = c(1, 1, 1), lty = c(1, 1, 2))

# Bottom: Forecast errors
plot(test_dates, test_residuals, type = "l", col = "blue", lwd = 1,
     xlab = "Date", ylab = "Forecast Error",
     main = "Forecast Errors: XGBoost vs VIX")
lines(test_dates, vix_residuals, col = "red", lwd = 1)
abline(h = 0, col = "grey", lty = 2)
legend("topright", c("XGBoost Error", "VIX Error"),
       col = c("blue", "red"), lwd = 1)

dev.off()

# 15.3 VRP analysis
pdf("results/figures/rv_regression/04_vrp_analysis.pdf", width = 12, height = 10)

par(mfrow = c(2, 2))

# VRP time series
plot(test_dates, vrp_xgb, type = "l", col = "blue", lwd = 1,
     xlab = "Date", ylab = "VRP (VIX - E[RV])",
     main = "Variance Risk Premium (XGBoost-based)")
abline(h = 0, col = "red", lty = 2)
abline(h = mean(vrp_xgb, na.rm = TRUE), col = "green", lty = 2)

# VRP distribution
hist(vrp_xgb, breaks = 50, col = "lightblue",
     main = "VRP Distribution", xlab = "VRP")
abline(v = 0, col = "red", lwd = 2)
abline(v = mean(vrp_xgb, na.rm = TRUE), col = "green", lwd = 2, lty = 2)

# XGBoost VRP vs Actual VRP scatter
plot(vrp_actual, vrp_xgb, pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = "Actual VRP (ex-post)", ylab = "XGBoost VRP Estimate",
     main = sprintf("VRP: Predicted vs Actual (r = %.3f)", vrp_correlation))
abline(0, 1, col = "red", lwd = 2)

# VRP by regime
boxplot(vrp_xgb ~ test_results$vix_regime,
        main = "VRP by VIX Regime", ylab = "VRP",
        col = c("green", "yellow", "red"))
abline(h = 0, col = "black", lty = 2)

dev.off()

# 15.4 Feature importance
pdf("results/figures/rv_regression/05_feature_importance.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

# Top 20 by Gain
top_20_gain <- head(importance_gain, 20)
barplot(rev(top_20_gain$Gain), names.arg = rev(top_20_gain$Feature),
        horiz = TRUE, las = 1, cex.names = 0.6,
        main = "Top 20 Features by Gain", xlab = "Gain",
        col = "steelblue")

# Top 20 by SHAP
top_20_shap <- head(shap_importance, 20)
barplot(rev(top_20_shap$Mean_Abs_SHAP), names.arg = rev(top_20_shap$Feature),
        horiz = TRUE, las = 1, cex.names = 0.6,
        main = "Top 20 Features by Mean |SHAP|", xlab = "Mean |SHAP|",
        col = "coral")

# Category importance
category_importance <- data.frame()
for (cat_name in names(feature_categories)) {
  cat_features <- intersect(feature_categories[[cat_name]], importance_gain$Feature)
  if (length(cat_features) > 0) {
    cat_gain <- sum(importance_gain$Gain[importance_gain$Feature %in% cat_features])
    category_importance <- rbind(category_importance, data.frame(
      category = cat_name,
      total_gain = cat_gain,
      n_features = length(cat_features)
    ))
  }
}
category_importance <- category_importance[order(-category_importance$total_gain), ]

barplot(category_importance$total_gain, names.arg = category_importance$category,
        las = 2, cex.names = 0.7, main = "Feature Category Importance",
        ylab = "Total Gain", col = rainbow(nrow(category_importance)))

# SHAP dependence for top feature
top_feature <- shap_importance$Feature[1]
top_feature_idx <- which(feature_cols == top_feature)
plot(X_test[shap_idx, top_feature_idx], shap_matrix[, top_feature_idx],
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = top_feature, ylab = "SHAP Value",
     main = sprintf("SHAP Dependence: %s", top_feature))
abline(h = 0, col = "red", lty = 2)
lines(lowess(X_test[shap_idx, top_feature_idx], shap_matrix[, top_feature_idx]), 
      col = "red", lwd = 2)

dev.off()

# 15.5 Temporal stability
pdf("results/figures/rv_regression/06_temporal_stability.pdf", width = 14, height = 8)

par(mfrow = c(2, 2))

# Rolling RMSE
plot(rolling_metrics$date, rolling_metrics$rmse, type = "l",
     col = "blue", lwd = 1.5,
     main = "Rolling 63-day RMSE", xlab = "Date", ylab = "RMSE")
abline(h = mean_rolling_rmse, col = "green", lty = 2)
abline(h = degradation_threshold, col = "red", lty = 2)

# Rolling R²
plot(rolling_metrics$date, rolling_metrics$r2, type = "l",
     col = "blue", lwd = 1.5,
     main = "Rolling 63-day R²", xlab = "Date", ylab = "R²")
abline(h = mean(rolling_metrics$r2), col = "green", lty = 2)

# Regime performance
barplot(regime_performance$rmse, names.arg = regime_performance$regime,
        main = "RMSE by VIX Regime", ylab = "RMSE",
        col = c("green", "yellow", "red"))

# Residual ACF
acf(test_residuals, lag.max = 30, main = "Forecast Error ACF")

dev.off()

# 15.6 Residual diagnostics
pdf("results/figures/rv_regression/07_residual_diagnostics.pdf", width = 12, height = 10)

par(mfrow = c(2, 2))

# Residuals vs fitted
plot(test_pred, test_residuals, pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, col = "red", lty = 2)
lines(lowess(test_pred, test_residuals), col = "red", lwd = 2)

# Q-Q plot
qqnorm(test_residuals, pch = 16, cex = 0.5)
qqline(test_residuals, col = "red")

# Residual histogram
hist(test_residuals, breaks = 50, col = "lightblue", probability = TRUE,
     main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean = mean(test_residuals), sd = sd(test_residuals)),
      add = TRUE, col = "red", lwd = 2)

# Scale-location
plot(test_pred, sqrt(abs(test_residuals)), pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = "Fitted Values", ylab = "√|Residuals|",
     main = "Scale-Location")
lines(lowess(test_pred, sqrt(abs(test_residuals))), col = "red", lwd = 2)

dev.off()

#------------------------------------------------------------------
# 16. SAVE RESULTS
#------------------------------------------------------------------

cat_progress("Saving results...")

# Save model
xgb.save(final_model, "results/models/rv_regression/xgb_rv_regression_model.xgb")
saveRDS(final_params, "results/models/rv_regression/xgb_rv_regression_params.rds")

# Save predictions
test_predictions <- data.frame(
  date = test_dates,
  actual_rv = y_test,
  predicted_rv = test_pred,
  vix_forecast = vix_forecast,
  vix_current = test_features$vix_close,
  vrp_xgb = vrp_xgb,
  vrp_actual = vrp_actual,
  forecast_error = test_residuals,
  vix_error = vix_residuals
)
saveRDS(test_predictions, "results/models/rv_regression/xgb_rv_regression_predictions.rds")
writexl::write_xlsx(test_predictions, "results/tables/rv_regression/xgb_rv_regression_predictions.xlsx")

# Save CV results
saveRDS(cv_results, "results/models/rv_regression/xgb_rv_regression_cv_results.rds")
saveRDS(cv_metrics, "results/models/rv_regression/xgb_rv_regression_cv_metrics.rds")
saveRDS(cv_predictions, "results/models/rv_regression/xgb_rv_regression_cv_predictions.rds")

# Save feature importance
saveRDS(importance_gain, "results/models/rv_regression/xgb_rv_regression_importance_gain.rds")
saveRDS(shap_importance, "results/models/rv_regression/xgb_rv_regression_shap_importance.rds")
saveRDS(shap_matrix, "results/models/rv_regression/xgb_rv_regression_shap_matrix.rds")

# Save metrics and summaries
write.csv(test_metrics, "results/tables/rv_regression/xgb_rv_regression_test_metrics.csv", row.names = FALSE)
write.csv(cv_summary, "results/tables/rv_regression/xgb_rv_regression_cv_summary.csv", row.names = FALSE)
write.csv(benchmark_comparison, "results/tables/rv_regression/xgb_rv_vix_comparison.csv", row.names = FALSE)
write.csv(vrp_summary, "results/tables/rv_regression/xgb_vrp_summary.csv", row.names = FALSE)
write.csv(regime_performance, "results/tables/rv_regression/xgb_rv_regime_performance.csv", row.names = FALSE)
write.csv(as.data.frame(importance_gain), "results/tables/rv_regression/xgb_rv_feature_importance.csv", row.names = FALSE)

# Save Bayesian optimisation results
saveRDS(bayes_opt_result, "results/models/rv_regression/xgb_rv_bayes_opt.rds")

# Compile full results object
xgb_rv_results <- list(
  model_path = "results/models/rv_regression/xgb_rv_regression_model.xgb",
  params = final_params,
  hyperparameter_search = bayes_opt_result,
  cv_summary = cv_summary,
  cv_metrics = cv_metrics,
  test_metrics = test_metrics,
  test_predictions = test_predictions,
  benchmark_comparison = benchmark_comparison,
  mincer_zarnowitz = list(
    xgb_alpha = mz_xgb_alpha,
    xgb_beta = mz_xgb_beta,
    xgb_r2 = mz_xgb_summary$r.squared,
    xgb_wald_test = mz_xgb_wald,
    vix_alpha = mz_vix_alpha,
    vix_beta = mz_vix_beta,
    vix_r2 = mz_vix_summary$r.squared
  ),
  vrp_analysis = list(
    vrp_summary = vrp_summary,
    vrp_correlation = vrp_correlation
  ),
  statistical_tests = list(
    diebold_mariano = dm_result,
    rmse_bootstrap_ci = rmse_ci,
    r2_bootstrap_ci = r2_ci
  ),
  feature_importance = list(
    gain = importance_gain,
    shap = shap_importance
  ),
  regime_performance = regime_performance,
  temporal_stability = list(
    rolling_metrics = rolling_metrics,
    degradation_periods = degradation_periods
  ),
  feature_cols = feature_cols,
  n_features = length(feature_cols),
  purge_days = purge_days,
  embargo_days = embargo_days
)

saveRDS(xgb_rv_results, "results/models/rv_regression/xgb_rv_regression_full_results.rds")


################################################################################
# END OF SCRIPT
################################################################################