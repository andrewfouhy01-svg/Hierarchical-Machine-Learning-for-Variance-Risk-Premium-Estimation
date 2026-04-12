################################################################################
# 04_XGBoost_Regression_AR1_Residuals.R 
# XGBoost model predicting AR(1) residuals for VIX forecasting
################################################################################

#------------------------------------------------------------------
# 0. SETUP
#------------------------------------------------------------------

source("Setup.R")

#------------------------------------------------------------------
# 1. LOAD FEATURE ENGINEERED DATA
#------------------------------------------------------------------

train_features <- readRDS("data/features_train.rds")
test_features <- readRDS("data/features_test.rds")
feature_cols <- readRDS("data/regression_feature_columns.rds")
feature_categories <- readRDS("data/feature_categories.rds")


#------------------------------------------------------------------
# 1.5 REMOVE ROWS WITH NA 
#------------------------------------------------------------------

train_na_count <- sum(is.na(train_features$target_vix_level))
test_na_count <- sum(is.na(test_features$target_vix_level))
train_features <- train_features[!is.na(target_vix_level)]
test_features <- test_features[!is.na(target_vix_level)]


#------------------------------------------------------------------
# 2. LEAKAGE CHECKS
#------------------------------------------------------------------

forbidden_patterns <- c("target_", "^vix$", "^spx$", "vix_close", "close")

leakage_check <- sapply(feature_cols, function(f) {
  any(sapply(forbidden_patterns, function(p) grepl(p, f)))
})

if (any(leakage_check)) {
  cat_progress("leakage features detected:")
  print(feature_cols[leakage_check])
  feature_cols <- feature_cols[!leakage_check]
  cat_progress(sprintf("Removed %d leakage features", sum(leakage_check)))
}


stopifnot(all(diff(train_features$date) >= 0))
stopifnot(all(diff(test_features$date) >= 0))
stopifnot(max(train_features$date) < min(test_features$date))


#------------------------------------------------------------------
# 3. AR(1) MODEL FITTING AND RESIDUAL COMPUTATION
#------------------------------------------------------------------

cat_progress("Fitting AR(1) model on training data...")

# Extract target and lagged VIX
y_train_raw <- train_features$target_vix_level
y_test_raw <- test_features$target_vix_level
train_current_vix <- train_features$vix_lag_1
test_current_vix <- test_features$vix_lag_1

# Fit AR(1) on training data: y_t = c + phi * y_{t-1}
# Using OLS estimation
ar1_fit <- lm(y_train_raw ~ train_current_vix)
ar1_intercept <- coef(ar1_fit)[1]
ar1_phi <- coef(ar1_fit)[2]
ar1_sigma <- sd(residuals(ar1_fit))

cat_progress(sprintf("AR(1) Parameters: intercept=%.4f, phi=%.4f, sigma=%.4f",
                     ar1_intercept, ar1_phi, ar1_sigma))

# Generate AR(1) predictions
ar1_pred_train <- ar1_intercept + ar1_phi * train_current_vix
ar1_pred_test <- ar1_intercept + ar1_phi * test_current_vix

# Compute AR(1) residuals (these are what XGBoost will predict)
ar1_resid_train <- y_train_raw - ar1_pred_train
ar1_resid_test <- y_test_raw - ar1_pred_test

cat_progress(sprintf("AR(1) residuals - Train: mean=%.4f, sd=%.4f", 
                     mean(ar1_resid_train), sd(ar1_resid_train)))
cat_progress(sprintf("AR(1) residuals - Test: mean=%.4f, sd=%.4f", 
                     mean(ar1_resid_test), sd(ar1_resid_test)))


#------------------------------------------------------------------
# 4. FEATURE PREPROCESSING
#------------------------------------------------------------------

# 4.1 Remove highly correlated features
cor_threshold <- 0.9999

X_train_raw <- as.matrix(train_features[, ..feature_cols])

na_counts <- colSums(is.na(X_train_raw))
if (any(na_counts > 0)) {
  cat_progress(sprintf("Removing %d features with NA values", sum(na_counts > 0)))
  feature_cols <- feature_cols[na_counts == 0]
  X_train_raw <- as.matrix(train_features[, ..feature_cols])
}

cor_matrix <- cor(X_train_raw, use = "pairwise.complete.obs")
cor_matrix[is.na(cor_matrix)] <- 0

high_cor_pairs <- which(abs(cor_matrix) > cor_threshold & upper.tri(cor_matrix), arr.ind = TRUE)

if (nrow(high_cor_pairs) > 0) {
  # For AR(1) residuals, use correlation with residuals for feature selection
  target_cors <- abs(cor(X_train_raw, ar1_resid_train, use = "complete.obs"))
  
  features_to_remove <- c()
  for (i in 1:nrow(high_cor_pairs)) {
    f1 <- feature_cols[high_cor_pairs[i, 1]]
    f2 <- feature_cols[high_cor_pairs[i, 2]]
    
    if (target_cors[high_cor_pairs[i, 1]] < target_cors[high_cor_pairs[i, 2]]) {
      features_to_remove <- c(features_to_remove, f1)
    } else {
      features_to_remove <- c(features_to_remove, f2)
    }
  }
  
  features_to_remove <- unique(features_to_remove)
  feature_cols <- setdiff(feature_cols, features_to_remove)
  cat_progress(sprintf("Removed %d highly correlated features", length(features_to_remove)))
}


X_train <- as.matrix(train_features[, ..feature_cols])
X_test <- as.matrix(test_features[, ..feature_cols])

# Target is now AR(1) residuals
y_train <- ar1_resid_train
y_test <- ar1_resid_test

stopifnot(!any(is.na(y_train)))
stopifnot(!any(is.na(y_test)))

train_dates <- train_features$date
test_dates <- test_features$date

cat_progress(sprintf("Final feature set: %d features", length(feature_cols)))


#------------------------------------------------------------------
# 5. TIME-SERIES CROSS-VALIDATION (EXPANDING WINDOW)
#------------------------------------------------------------------

#' Create expanding window CV folds for time series
#' 
#' @param n Total number of observations
#' @param n_folds Number of CV folds
#' @param initial_window Size of initial training window (proportion)
#' @param horizon Validation horizon (proportion of data)
#' @return List of train/validation indices for each fold
create_ts_cv_folds <- function(n, n_folds = 5, initial_window = 0.5, horizon = 0.05) {
  
  folds <- list()
  
  init_size <- floor(n * initial_window)
  val_size <- floor(n * horizon)
  step_size <- floor((n - init_size - val_size) / n_folds)
  
  for (fold in 1:n_folds) {
    train_end <- init_size + (fold - 1) * step_size
    val_start <- train_end + 1
    val_end <- min(val_start + val_size - 1, n)
    
    if (val_end <= val_start) break
    
    folds[[fold]] <- list(
      train = 1:train_end,
      val = val_start:val_end
    )
  }
  
  return(folds)
}

n_train <- nrow(X_train)
purge_days <- 0
embargo_days <- 0

# Inner folds for Bayesian optimisation
purged_folds <- create_ts_cv_folds(
  n = n_train,
  n_folds = 8,
  initial_window = 0.5,
  horizon = 0.1
)

# Outer folds for final CV evaluation
cpcv_folds <- create_ts_cv_folds(
  n = n_train,
  n_folds = 8,
  initial_window = 0.5,
  horizon = 0.1
)

for (i in seq_along(purged_folds)) {
  cat("  Fold", i, ": Train [1:", length(purged_folds[[i]]$train), 
      "], Val [", min(purged_folds[[i]]$val), ":", max(purged_folds[[i]]$val), "]\n", sep = "")
}

#------------------------------------------------------------------
# 6. METRICS (ADAPTED FOR RESIDUAL PREDICTION)
#------------------------------------------------------------------

# Metrics for residual prediction
calc_residual_metrics <- function(actual_resid, predicted_resid, 
                                  train_mean = NULL) {
  valid_idx <- !is.na(actual_resid) & !is.na(predicted_resid) & 
    is.finite(actual_resid) & is.finite(predicted_resid)
  actual_resid <- actual_resid[valid_idx]
  predicted_resid <- predicted_resid[valid_idx]
  
  n <- length(actual_resid)
  if (n == 0) return(NULL)
  
  errors <- actual_resid - predicted_resid
  
  rmse <- sqrt(mean(errors^2))
  mae <- mean(abs(errors))
  me <- mean(errors)
  
  # R² for residuals
  baseline_mean <- if (!is.null(train_mean)) train_mean else mean(actual_resid)
  ss_res <- sum(errors^2)
  ss_tot <- sum((actual_resid - baseline_mean)^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  # Additional metrics
  max_ae <- max(abs(errors))
  medae <- median(abs(errors))
  
  list(
    N = n,
    RMSE = rmse,
    MAE = mae,
    MedAE = medae,
    ME = me,
    R2 = r2,
    Max_AE = max_ae
  )
}

# Full metrics for final VIX predictions (AR(1) + XGBoost residual)
calc_regression_metrics <- function(actual, predicted, current_vix = NULL, 
                                    train_mean = NULL, train_log_mean = NULL) {
  valid_idx <- !is.na(actual) & !is.na(predicted) & 
    is.finite(actual) & is.finite(predicted) &
    actual > 0 & predicted > 0
  actual <- actual[valid_idx]
  predicted <- predicted[valid_idx]
  if (!is.null(current_vix)) {
    current_vix <- current_vix[valid_idx]
  }
  
  n <- length(actual)
  if (n == 0) return(NULL)
  
  errors <- actual - predicted
  
  rmse <- sqrt(mean(errors^2))
  mae <- mean(abs(errors))
  mape <- mean(abs(errors / actual)) * 100
  me <- mean(errors)
  
  baseline_mean <- if (!is.null(train_mean)) train_mean else mean(actual)
  ss_res <- sum(errors^2)
  ss_tot <- sum((actual - baseline_mean)^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  qlike <- mean(log(predicted) + actual / predicted)
  
  log_errors <- log(predicted) - log(actual)
  mse_log <- mean(log_errors^2)
  
  baseline_log_mean <- if (!is.null(train_log_mean)) train_log_mean else mean(log(actual))
  ss_res_log <- sum(log_errors^2)
  ss_tot_log <- sum((log(actual) - baseline_log_mean)^2)
  r2_log <- 1 - (ss_res_log / ss_tot_log)
  
  hmse <- mean((errors / actual)^2)
  
  if (!is.null(current_vix)) {
    naive_errors <- actual - current_vix
    naive_rmse <- sqrt(mean(naive_errors^2))
    theil_u <- rmse / naive_rmse
    
    actual_direction <- actual > current_vix
    predicted_direction <- predicted > current_vix
    dir_accuracy <- mean(actual_direction == predicted_direction)
  } else {
    theil_u <- NA
    dir_accuracy <- NA
  }
  
  smape <- mean(2 * abs(errors) / (abs(actual) + abs(predicted))) * 100
  rmspe <- sqrt(mean((errors / actual)^2)) * 100
  max_ae <- max(abs(errors))
  medae <- median(abs(errors))
  
  list(
    N = n,
    RMSE = rmse,
    MAE = mae,
    MedAE = medae,
    MAPE = mape,
    sMAPE = smape,
    RMSPE = rmspe,
    ME = me,
    R2 = r2,
    R2_LOG = r2_log,
    QLIKE = qlike,
    MSE_LOG = mse_log,
    HMSE = hmse,
    Theil_U = theil_u,
    Dir_Accuracy = dir_accuracy,
    Max_AE = max_ae
  )
}


#------------------------------------------------------------------
# 7. BAYESIAN HYPERPARAMETER OPTIMISATION WITH NESTED CV
#------------------------------------------------------------------

bounds <- list(
  max_depth = c(2L, 10L),
  min_child_weight = c(1, 100),
  subsample = c(0.2, 1.0),
  colsample_bytree = c(0.2, 1.0),
  colsample_bynode = c(0.2, 1.0),
  eta = c(0.0025, 0.8),
  gamma = c(0, 15),
  lambda = c(1, 35),
  alpha = c(0, 4),
  max_delta_step = c(0, 3)
)

inner_folds <- purged_folds[1:8]

# Scoring function for AR(1) residuals
scoring_function <- function(max_depth, min_child_weight, subsample,
                             colsample_bytree, colsample_bynode, eta, 
                             gamma, lambda, alpha, max_delta_step) {
  
  max_depth <- as.integer(round(max_depth))
  min_child_weight <- as.integer(round(min_child_weight))
  
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
    max_delta_step = max_delta_step,
    nthread = 2
  )
  
  cv_losses <- rep(NA_real_, length(inner_folds))
  cv_iters <- rep(NA_integer_, length(inner_folds))
  
  for (i in seq_along(inner_folds)) {
    fold <- inner_folds[[i]]
    
    X_tr <- X_train[fold$train, , drop = FALSE]
    y_tr <- y_train[fold$train]  # AR(1) residuals
    X_val <- X_train[fold$val, , drop = FALSE]
    y_val <- y_train[fold$val]   # AR(1) residuals
    
    if (any(is.na(y_tr)) || any(is.na(y_val)) || 
        length(y_tr) < 100 || length(y_val) < 20) next
    
    dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
    dval <- xgb.DMatrix(data = X_val, label = y_val)
    
    early_stop <- min(1000L, max(50L, as.integer(20 / eta)))
    
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 4000,
      maximize = FALSE,
      watchlist = list(val = dval),
      early_stopping_rounds = early_stop,
      verbose = 2
    )
    
    pred_resid <- predict(model, dval)
    
    # For residual prediction, use MSE as the loss
    mse <- mean((y_val - pred_resid)^2)
    cv_losses[i] <- mse
    cv_iters[i] <- model$best_iteration
  }
  
  valid_losses <- cv_losses[!is.na(cv_losses)]
  if (length(valid_losses) == 0) return(list(Score = -Inf))
  
  return(list(
    Score = -mean(valid_losses)  # Negative because bayesOpt maximises
  ))
}


set.seed(42)

if (!exists("cl") || !inherits(cl, "cluster")) {
  cat_progress("Creating parallel cluster...")
  cl <- makeCluster(detectCores() - 1)
}

clusterExport(cl, c(
  "X_train", "y_train", "inner_folds",
  "scoring_function"
))
clusterEvalQ(cl, {
  library(xgboost)
})

output_fig_dir <- "results/figures"

bayes_opt_result <- bayesOpt(
  FUN = scoring_function,
  bounds = bounds,
  initPoints = 98,
  iters.n = 63,
  iters.k = 7,
  acq = "ei",
  eps = 0.25,
  gsPoints = 14700,
  acqThresh = 0.3,
  parallel = TRUE,
  verbose = 2,
  plotProgress = TRUE
)

pdf(file.path(output_fig_dir, "bayesian_optimisation_regression_ar1_progress.pdf"), width = 12, height = 8)
plot(bayes_opt_result)
dev.off()

best_params <- getBestPars(bayes_opt_result)
best_params$max_depth <- round(best_params$max_depth)
best_params$min_child_weight <- round(best_params$min_child_weight)

stopCluster(cl)
rm(cl)


#------------------------------------------------------------------
# 8. OUTER CROSS-VALIDATION WITH OPTIMAL HYPERPARAMETERS
#------------------------------------------------------------------

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
  alpha = best_params$alpha,
  max_delta_step = best_params$max_delta_step
)

cv_results <- list()
cv_predictions <- data.frame()
cv_metrics <- data.frame()
cv_metrics_level <- data.frame()  # Metrics for final VIX level predictions

for (i in seq_along(cpcv_folds)) {
  fold <- cpcv_folds[[i]]
  
  X_tr <- X_train[fold$train, , drop = FALSE]
  y_tr <- y_train[fold$train]  # AR(1) residuals
  X_val <- X_train[fold$val, , drop = FALSE]
  y_val <- y_train[fold$val]   # AR(1) residuals
  val_dates <- train_dates[fold$val]
  val_current_vix <- train_current_vix[fold$val]
  val_actual_vix <- y_train_raw[fold$val]  # Actual VIX levels
  val_ar1_pred <- ar1_pred_train[fold$val]  # AR(1) predictions
  
  if (any(is.na(y_tr)) || any(is.na(y_val))) {
    cat_progress(sprintf("Fold %d: Skipping due to NA values", i))
    next
  }
  
  dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
  dval <- xgb.DMatrix(data = X_val, label = y_val)
  
  early_stop <- min(1000L, max(50L, as.integer(20 / final_params$eta)))
  
  model <- xgb.train(
    params = final_params,
    data = dtrain,
    nrounds = 4000,
    maximize = FALSE,
    watchlist = list(train = dtrain, val = dval),
    early_stopping_rounds = early_stop,
    verbose = 2  
  )
  
  best_iter <- model$best_iteration
  if (is.null(best_iter) || is.na(best_iter)) {
    warning(sprintf("Fold %d: early stopping failed, using 5000 rounds", i))
    best_iter <- 5000L
  }
  
  # Predict AR(1) residuals
  pred_resid <- predict(model, dval)
  
  # Final VIX prediction = AR(1) prediction + XGBoost residual prediction
  pred_vix <- val_ar1_pred + pred_resid
  
  fold_preds <- data.frame(
    fold = i,
    date = val_dates,
    actual_resid = y_val,
    predicted_resid = pred_resid,
    actual_vix = val_actual_vix,
    predicted_vix = pred_vix,
    ar1_pred = val_ar1_pred,
    current_vix = val_current_vix,
    error_resid = y_val - pred_resid,
    error_vix = val_actual_vix - pred_vix
  )
  cv_predictions <- rbind(cv_predictions, fold_preds)
  
  # Metrics for residual prediction
  resid_metrics <- calc_residual_metrics(
    y_val, pred_resid,
    train_mean = mean(y_tr)
  )
  
  # Metrics for final VIX level prediction
  level_metrics <- calc_regression_metrics(
    val_actual_vix, pred_vix, val_current_vix,
    train_mean = mean(y_train_raw[fold$train]),
    train_log_mean = mean(log(y_train_raw[fold$train]))
  )
  
  fold_metrics <- data.frame(
    fold = i,
    n_train = length(y_tr),
    n_val = length(y_val),
    RMSE_resid = resid_metrics$RMSE,
    MAE_resid = resid_metrics$MAE,
    R2_resid = resid_metrics$R2,
    RMSE_level = level_metrics$RMSE,
    MAE_level = level_metrics$MAE,
    MAPE_level = level_metrics$MAPE,
    R2_level = level_metrics$R2,
    R2_LOG_level = level_metrics$R2_LOG,
    QLIKE_level = level_metrics$QLIKE,
    MSE_LOG_level = level_metrics$MSE_LOG,
    Theil_U = level_metrics$Theil_U,
    Dir_Accuracy = level_metrics$Dir_Accuracy,
    best_iteration = best_iter
  )
  cv_metrics <- rbind(cv_metrics, fold_metrics)
  
  cv_results[[i]] <- list(
    model = model,
    predictions = fold_preds,
    metrics = fold_metrics
  )
  
  cat_progress(sprintf("Fold %d: RMSE_resid=%.4f, R2_resid=%.4f, RMSE_level=%.4f, Dir=%.2f%%, iter=%d",
                       i, resid_metrics$RMSE, resid_metrics$R2, 
                       level_metrics$RMSE, level_metrics$Dir_Accuracy * 100, best_iter))
}

cv_summary <- data.frame(
  Metric = c("RMSE_resid", "MAE_resid", "R2_resid", 
             "RMSE_level", "MAE_level", "MAPE_level", "R2_level", "R2_LOG_level", 
             "QLIKE_level", "MSE_LOG_level", "Theil_U", "Dir_Accuracy"),
  Mean = c(mean(cv_metrics$RMSE_resid, na.rm = TRUE),
           mean(cv_metrics$MAE_resid, na.rm = TRUE),
           mean(cv_metrics$R2_resid, na.rm = TRUE),
           mean(cv_metrics$RMSE_level, na.rm = TRUE),
           mean(cv_metrics$MAE_level, na.rm = TRUE),
           mean(cv_metrics$MAPE_level, na.rm = TRUE),
           mean(cv_metrics$R2_level, na.rm = TRUE),
           mean(cv_metrics$R2_LOG_level, na.rm = TRUE),
           mean(cv_metrics$QLIKE_level, na.rm = TRUE),
           mean(cv_metrics$MSE_LOG_level, na.rm = TRUE),
           mean(cv_metrics$Theil_U, na.rm = TRUE),
           mean(cv_metrics$Dir_Accuracy, na.rm = TRUE)),
  SD = c(sd(cv_metrics$RMSE_resid, na.rm = TRUE),
         sd(cv_metrics$MAE_resid, na.rm = TRUE),
         sd(cv_metrics$R2_resid, na.rm = TRUE),
         sd(cv_metrics$RMSE_level, na.rm = TRUE),
         sd(cv_metrics$MAE_level, na.rm = TRUE),
         sd(cv_metrics$MAPE_level, na.rm = TRUE),
         sd(cv_metrics$R2_level, na.rm = TRUE),
         sd(cv_metrics$R2_LOG_level, na.rm = TRUE),
         sd(cv_metrics$QLIKE_level, na.rm = TRUE),
         sd(cv_metrics$MSE_LOG_level, na.rm = TRUE),
         sd(cv_metrics$Theil_U, na.rm = TRUE),
         sd(cv_metrics$Dir_Accuracy, na.rm = TRUE))
)
print(cv_summary, digits = 4)


#------------------------------------------------------------------
# 9. FINAL MODEL TRAINING (Early stopping then retrain)
#------------------------------------------------------------------

# Step 1: Train on 80% with early stopping against 20% validation
train_val_split <- floor(nrow(X_train) * 0.8)

dtrain_es <- xgb.DMatrix(
  data = X_train[1:train_val_split, , drop = FALSE],
  label = y_train[1:train_val_split]
)
dval_es <- xgb.DMatrix(
  data = X_train[(train_val_split + 1):nrow(X_train), , drop = FALSE],
  label = y_train[(train_val_split + 1):nrow(X_train)]
)

early_stop <- min(1000L, max(50L, as.integer(20 / final_params$eta)))

es_model <- xgb.train(
  params = final_params,
  data = dtrain_es,
  nrounds = 5000,
  watchlist = list(train = dtrain_es, val = dval_es),
  early_stopping_rounds = early_stop,
  verbose = 2
)

optimal_nrounds <- es_model$best_iteration
if (is.null(optimal_nrounds) || is.na(optimal_nrounds)) {
  warning("Early stopping failed, falling back to median CV iterations")
  optimal_nrounds <- as.integer(median(cv_metrics$best_iteration, na.rm = TRUE))
}

cat_progress(sprintf("Optimal nrounds from early stopping: %d", optimal_nrounds))

# Step 2: Retrain on full training data for exactly optimal_nrounds
dtrain_full <- xgb.DMatrix(data = X_train, label = y_train)

final_model <- xgb.train(
  params = final_params,
  data = dtrain_full,
  nrounds = optimal_nrounds,
  verbose = 2
)

xgb.save(final_model, "results/models/xgb_ar1_resid_final.model")

# Feature importance
importance <- xgb.importance(model = final_model)
pdf(file.path(output_fig_dir, "feature_importance_ar1.pdf"), width = 10, height = 8)
xgb.plot.importance(importance[1:min(30, nrow(importance)), ])
dev.off()

cat_progress("Final model training complete")


#------------------------------------------------------------------
# 10. TEST SET EVALUATION
#------------------------------------------------------------------

dtest <- xgb.DMatrix(data = X_test, label = y_test)

# Predict AR(1) residuals
test_pred_resid <- predict(final_model, dtest)

# Final VIX prediction = AR(1) prediction + XGBoost residual prediction
test_pred_vix <- ar1_pred_test + test_pred_resid

# Metrics for residual prediction
test_resid_metrics <- calc_residual_metrics(y_test, test_pred_resid)

# Metrics for final VIX level prediction
test_metrics_list <- calc_regression_metrics(y_test_raw, test_pred_vix, test_current_vix)

cat_progress("Test Set Residual Metrics:")
test_resid_metrics_df <- data.frame(
  Metric = names(test_resid_metrics),
  Value = unlist(test_resid_metrics)
)
print(test_resid_metrics_df, digits = 4)

cat_progress("Test Set VIX Level Metrics:")
test_metrics <- data.frame(
  Metric = names(test_metrics_list),
  Value = unlist(test_metrics_list)
)
print(test_metrics, digits = 4)

# Errors
test_errors_resid <- y_test - test_pred_resid
test_errors_level <- y_test_raw - test_pred_vix
test_pct_errors <- (y_test_raw - test_pred_vix) / y_test_raw * 100


#------------------------------------------------------------------
# 11. NAIVE BASELINE COMPARISONS
#------------------------------------------------------------------

# Random Walk (naive)
naive_rw_pred <- test_current_vix
naive_rw_metrics <- calc_regression_metrics(y_test_raw, naive_rw_pred, test_current_vix)

# Historical Mean
naive_mean_pred <- rep(mean(y_train_raw), length(y_test_raw))
naive_mean_metrics <- calc_regression_metrics(y_test_raw, naive_mean_pred, test_current_vix)

# AR(1) baseline (without XGBoost enhancement)
naive_ar1_metrics <- calc_regression_metrics(y_test_raw, ar1_pred_test, test_current_vix)

# EWMA
ewma_pred <- if ("vix_ewma_10" %in% names(test_features)) test_features$vix_ewma_10 else NULL
if (!is.null(ewma_pred) && !all(is.na(ewma_pred))) {
  naive_ewma_metrics <- calc_regression_metrics(y_test_raw, ewma_pred, test_current_vix)
} else {
  naive_ewma_metrics <- NULL
}

baseline_comparison <- data.frame(
  Model = c("XGBoost_AR1_Resid", "AR(1) Only", "Random Walk", "Historical Mean", "EWMA(10)"),
  RMSE = c(test_metrics_list$RMSE, naive_ar1_metrics$RMSE, naive_rw_metrics$RMSE, 
           naive_mean_metrics$RMSE, ifelse(!is.null(naive_ewma_metrics), naive_ewma_metrics$RMSE, NA)),
  MAE = c(test_metrics_list$MAE, naive_ar1_metrics$MAE, naive_rw_metrics$MAE,
          naive_mean_metrics$MAE, ifelse(!is.null(naive_ewma_metrics), naive_ewma_metrics$MAE, NA)),
  R2 = c(test_metrics_list$R2, naive_ar1_metrics$R2, naive_rw_metrics$R2,
         naive_mean_metrics$R2, ifelse(!is.null(naive_ewma_metrics), naive_ewma_metrics$R2, NA)),
  QLIKE = c(test_metrics_list$QLIKE, naive_ar1_metrics$QLIKE, naive_rw_metrics$QLIKE,
            naive_mean_metrics$QLIKE, ifelse(!is.null(naive_ewma_metrics), naive_ewma_metrics$QLIKE, NA)),
  Dir_Acc = c(test_metrics_list$Dir_Accuracy, naive_ar1_metrics$Dir_Accuracy, 
              naive_rw_metrics$Dir_Accuracy, naive_mean_metrics$Dir_Accuracy,
              ifelse(!is.null(naive_ewma_metrics), naive_ewma_metrics$Dir_Accuracy, NA))
)

cat_progress("Baseline Comparison:")
print(baseline_comparison, digits = 4)


#------------------------------------------------------------------
# 12. DIEBOLD-MARIANO TESTS
#------------------------------------------------------------------

dm_test <- function(e1, e2, h = 1, power = 2) {
  d <- abs(e1)^power - abs(e2)^power
  n <- length(d)
  
  d_mean <- mean(d)
  
  gamma_0 <- var(d)
  gamma_sum <- 0
  for (k in 1:min(h-1, n-1)) {
    gamma_k <- cov(d[1:(n-k)], d[(k+1):n])
    gamma_sum <- gamma_sum + 2 * gamma_k
  }
  
  var_d <- (gamma_0 + gamma_sum) / n
  if (var_d <= 0) var_d <- gamma_0 / n
  
  dm_stat <- d_mean / sqrt(var_d)
  p_value <- 2 * pnorm(-abs(dm_stat))
  
  list(
    dm_stat = dm_stat,
    p_value = p_value,
    mean_diff = d_mean
  )
}

# DM tests comparing XGBoost+AR(1) vs baselines
dm_vs_rw <- dm_test(test_errors_level, y_test_raw - naive_rw_pred)
dm_vs_ar1 <- dm_test(test_errors_level, y_test_raw - ar1_pred_test)
dm_vs_mean <- dm_test(test_errors_level, y_test_raw - naive_mean_pred)

cat_progress("Diebold-Mariano Tests (XGBoost+AR(1) vs Baselines):")
cat(sprintf("  vs Random Walk: DM=%.4f, p=%.4f\n", dm_vs_rw$dm_stat, dm_vs_rw$p_value))
cat(sprintf("  vs AR(1) Only:  DM=%.4f, p=%.4f\n", dm_vs_ar1$dm_stat, dm_vs_ar1$p_value))
cat(sprintf("  vs Hist Mean:   DM=%.4f, p=%.4f\n", dm_vs_mean$dm_stat, dm_vs_mean$p_value))


#------------------------------------------------------------------
# 13. RESIDUAL DIAGNOSTICS
#------------------------------------------------------------------

cat_progress("Residual Diagnostics...")

# Standardised errors (for both residual and level predictions)
std_errors_resid <- test_errors_resid / sd(test_errors_resid)
std_errors_level <- test_errors_level / sd(test_errors_level)

# Shapiro-Wilk test for normality
shapiro_test_resid <- tryCatch({
  shapiro.test(sample(std_errors_resid, min(5000, length(std_errors_resid))))
}, error = function(e) list(statistic = NA, p.value = NA))

shapiro_test_level <- tryCatch({
  shapiro.test(sample(std_errors_level, min(5000, length(std_errors_level))))
}, error = function(e) list(statistic = NA, p.value = NA))

# Ljung-Box test for autocorrelation
lb_test_resid <- Box.test(std_errors_resid, lag = 10, type = "Ljung-Box")
lb_test_resid_sq <- Box.test(std_errors_resid^2, lag = 10, type = "Ljung-Box")

lb_test_level <- Box.test(std_errors_level, lag = 10, type = "Ljung-Box")
lb_test_level_sq <- Box.test(std_errors_level^2, lag = 10, type = "Ljung-Box")

cat_progress("Residual Diagnostics - AR(1) Residual Prediction:")
cat(sprintf("  Shapiro-Wilk: W=%.4f, p=%.4f\n", shapiro_test_resid$statistic, shapiro_test_resid$p.value))
cat(sprintf("  Ljung-Box(10): Q=%.4f, p=%.4f\n", lb_test_resid$statistic, lb_test_resid$p.value))
cat(sprintf("  Ljung-Box(10) Squared: Q=%.4f, p=%.4f\n", lb_test_resid_sq$statistic, lb_test_resid_sq$p.value))

cat_progress("Residual Diagnostics - VIX Level Prediction:")
cat(sprintf("  Shapiro-Wilk: W=%.4f, p=%.4f\n", shapiro_test_level$statistic, shapiro_test_level$p.value))
cat(sprintf("  Ljung-Box(10): Q=%.4f, p=%.4f\n", lb_test_level$statistic, lb_test_level$p.value))
cat(sprintf("  Ljung-Box(10) Squared: Q=%.4f, p=%.4f\n", lb_test_level_sq$statistic, lb_test_level_sq$p.value))


#------------------------------------------------------------------
# 14. PREDICTION INTERVALS VIA QUANTILE BOOTSTRAP
#------------------------------------------------------------------

cat_progress("Computing prediction intervals...")

n_boot <- 10000
boot_errors <- matrix(NA, n_boot, length(test_errors_level))

set.seed(42)
for (b in 1:n_boot) {
  boot_idx <- sample(1:length(cv_predictions$error_vix), length(test_errors_level), replace = TRUE)
  boot_errors[b, ] <- cv_predictions$error_vix[boot_idx]
}

test_pi_95_lower <- test_pred_vix + apply(boot_errors, 2, quantile, 0.025)
test_pi_95_upper <- test_pred_vix + apply(boot_errors, 2, quantile, 0.975)
test_pi_90_lower <- test_pred_vix + apply(boot_errors, 2, quantile, 0.05)
test_pi_90_upper <- test_pred_vix + apply(boot_errors, 2, quantile, 0.95)

coverage_95 <- mean(y_test_raw >= test_pi_95_lower & y_test_raw <= test_pi_95_upper)
coverage_90 <- mean(y_test_raw >= test_pi_90_lower & y_test_raw <= test_pi_90_upper)

mean_width_95 <- mean(test_pi_95_upper - test_pi_95_lower)
mean_width_90 <- mean(test_pi_90_upper - test_pi_90_lower)

cat_progress(sprintf("Prediction Intervals - 95%%: Coverage=%.2f%%, Width=%.4f", 
                     coverage_95 * 100, mean_width_95))
cat_progress(sprintf("Prediction Intervals - 90%%: Coverage=%.2f%%, Width=%.4f", 
                     coverage_90 * 100, mean_width_90))


#------------------------------------------------------------------
# 15. FEATURE IMPORTANCE & SHAP ANALYSIS
#------------------------------------------------------------------

cat_progress("Computing SHAP values...")

# 15.1 Gain-based importance
importance_gain <- xgb.importance(feature_names = feature_cols, model = final_model)
importance_gain <- importance_gain[order(-Gain)]

# 15.2 SHAP values computation
shap_sample_size <- min(5000, nrow(X_test))
set.seed(42)
shap_sample_idx <- sample(1:nrow(X_test), shap_sample_size)

shap_contrib <- predict(final_model, X_test[shap_sample_idx, ], predcontrib = TRUE)
shap_matrix <- shap_contrib[, -ncol(shap_contrib)]  # Remove BIAS column
colnames(shap_matrix) <- feature_cols
shap_bias <- shap_contrib[, ncol(shap_contrib)]

# Feature matrix for sampled observations
X_shap <- X_test[shap_sample_idx, ]

# 15.3 SHAP importance (mean |SHAP|)
shap_importance <- data.frame(
  Feature = feature_cols,
  mean_abs_shap = colMeans(abs(shap_matrix)),
  mean_shap = colMeans(shap_matrix),
  sd_shap = apply(shap_matrix, 2, sd)
)
shap_importance <- shap_importance[order(-shap_importance$mean_abs_shap), ]
rownames(shap_importance) <- NULL

# 15.4 Compare Gain vs SHAP importance
importance_comparison <- merge(
  importance_gain[, .(Feature, Gain)],
  shap_importance[, c("Feature", "mean_abs_shap")],
  by = "Feature"
)
importance_comparison$Gain_Rank <- rank(-importance_comparison$Gain)
importance_comparison$SHAP_Rank <- rank(-importance_comparison$mean_abs_shap)
importance_comparison$Rank_Diff <- abs(importance_comparison$Gain_Rank - importance_comparison$SHAP_Rank)
importance_comparison <- importance_comparison[order(importance_comparison$SHAP_Rank), ]

cat_progress("Top 15 features by SHAP importance:")
print(head(shap_importance, 15), digits = 4)

# 15.5 SHAP Summary Plot (Beeswarm)
top_n_features <- 20
top_features <- head(shap_importance$Feature, top_n_features)
top_feature_idx <- match(top_features, feature_cols)

pdf("results/figures/shap_summary_regression_ar1.pdf", width = 12, height = 10)

par(mar = c(5, 12, 4, 2))
plot(NULL, xlim = range(shap_matrix[, top_feature_idx]), 
     ylim = c(0.5, top_n_features + 0.5),
     xlab = "SHAP Value (impact on AR(1) residual prediction)", ylab = "", yaxt = "n",
     main = "SHAP Summary Plot - AR(1) Residual Regression")
axis(2, at = 1:top_n_features, labels = rev(top_features), las = 1, cex.axis = 0.8)
abline(v = 0, lty = 2, col = "grey50")

for (i in 1:top_n_features) {
  feat_idx <- top_feature_idx[i]
  shap_vals <- shap_matrix[, feat_idx]
  feat_vals <- X_shap[, feat_idx]
  
  # Normalise feature values for colour mapping
  feat_norm <- (feat_vals - min(feat_vals, na.rm = TRUE)) / 
    (max(feat_vals, na.rm = TRUE) - min(feat_vals, na.rm = TRUE) + 1e-10)
  
  # Colour gradient: blue (low) to red (high)
  colours <- rgb(feat_norm, 0.2, 1 - feat_norm, 0.6)
  
  # Jitter y-position
  y_pos <- top_n_features - i + 1 + runif(length(shap_vals), -0.3, 0.3)
  
  points(shap_vals, y_pos, pch = 16, cex = 0.4, col = colours)
}

legend("topright", legend = c("High", "Low"), 
       pch = 16, col = c("red", "blue"), title = "Feature Value",
       bty = "n", cex = 0.9)

dev.off()

# 15.6 SHAP Dependence Plots for Top Features
pdf("results/figures/shap_dependence_regression_ar1.pdf", width = 14, height = 12)
par(mfrow = c(3, 3), mar = c(4, 4, 3, 1))

top_9_features <- head(shap_importance$Feature, 9)

for (feat in top_9_features) {
  feat_idx <- which(feature_cols == feat)
  shap_vals <- shap_matrix[, feat_idx]
  feat_vals <- X_shap[, feat_idx]
  
  # Find best interaction feature (highest correlation with SHAP residuals)
  lm_fit <- lm(shap_vals ~ feat_vals)
  residuals_lm <- residuals(lm_fit)
  
  interact_cors <- sapply(setdiff(top_features, feat), function(f) {
    f_idx <- which(feature_cols == f)
    abs(cor(residuals_lm, X_shap[, f_idx], use = "complete.obs"))
  })
  
  if (length(interact_cors) > 0) {
    best_interact <- names(which.max(interact_cors))
    interact_idx <- which(feature_cols == best_interact)
    interact_vals <- X_shap[, interact_idx]
    interact_norm <- (interact_vals - min(interact_vals, na.rm = TRUE)) / 
      (max(interact_vals, na.rm = TRUE) - min(interact_vals, na.rm = TRUE) + 1e-10)
    colours <- rgb(interact_norm, 0.2, 1 - interact_norm, 0.7)
  } else {
    colours <- rgb(0.2, 0.4, 0.8, 0.5)
    best_interact <- "N/A"
  }
  
  plot(feat_vals, shap_vals, pch = 16, cex = 0.5, col = colours,
       xlab = feat, ylab = "SHAP Value",
       main = sprintf("%s\n(colour: %s)", feat, best_interact))
  
  # Add LOESS smoother
  lo <- loess(shap_vals ~ feat_vals, span = 0.3)
  ord <- order(feat_vals)
  lines(feat_vals[ord], predict(lo)[ord], col = "black", lwd = 2)
  abline(h = 0, lty = 2, col = "grey50")
}

dev.off()

# 15.7 SHAP Interaction Analysis (top feature pairs)
cat_progress("Computing SHAP interaction values for top features...")

top_10_idx <- match(head(shap_importance$Feature, 10), feature_cols)
shap_top <- shap_matrix[, top_10_idx]

shap_cor_matrix <- cor(shap_top)
colnames(shap_cor_matrix) <- rownames(shap_cor_matrix) <- head(shap_importance$Feature, 10)

# Find strongest interactions
shap_cor_matrix[lower.tri(shap_cor_matrix, diag = TRUE)] <- NA
interaction_pairs <- which(!is.na(shap_cor_matrix) & abs(shap_cor_matrix) > 0.3, arr.ind = TRUE)

if (nrow(interaction_pairs) > 0) {
  interactions_df <- data.frame(
    Feature1 = rownames(shap_cor_matrix)[interaction_pairs[, 1]],
    Feature2 = colnames(shap_cor_matrix)[interaction_pairs[, 2]],
    SHAP_Correlation = shap_cor_matrix[interaction_pairs]
  )
  interactions_df <- interactions_df[order(-abs(interactions_df$SHAP_Correlation)), ]
  cat_progress("Strong SHAP interactions (|r| > 0.3):")
  print(interactions_df)
}

# 15.8 SHAP Waterfall Plot for Example Predictions
pdf("results/figures/shap_waterfall_regression_ar1.pdf", width = 12, height = 8)
par(mfrow = c(2, 2), mar = c(4, 10, 3, 2))

# Select interesting predictions
pred_errors <- abs(y_test[shap_sample_idx] - test_pred_resid[shap_sample_idx])
example_idx <- c(
  which.min(pred_errors),  # Best prediction
  which.max(pred_errors),  # Worst prediction
  which.min(abs(pred_errors - median(pred_errors))),  # Median error
  which.max(test_current_vix[shap_sample_idx])  # Highest VIX
)
example_labels <- c("Best Prediction", "Worst Prediction", "Median Error", "High VIX Day")

for (e in 1:4) {
  idx <- example_idx[e]
  shap_vals <- shap_matrix[idx, ]
  
  # Sort by absolute SHAP value, keep top 10
  top_contrib <- order(abs(shap_vals), decreasing = TRUE)[1:10]
  shap_sorted <- shap_vals[top_contrib]
  feat_names <- feature_cols[top_contrib]
  
  # Waterfall data
  base_val <- shap_bias[idx]
  cumsum_shap <- cumsum(c(base_val, shap_sorted))
  
  # Plot
  colours <- ifelse(shap_sorted > 0, "firebrick", "steelblue")
  
  barplot(shap_sorted, names.arg = feat_names, horiz = TRUE, las = 1,
          col = colours, border = NA, cex.names = 0.7,
          main = sprintf("%s\nActual Resid: %.2f, Pred Resid: %.2f", 
                         example_labels[e],
                         y_test[shap_sample_idx[idx]], 
                         test_pred_resid[shap_sample_idx[idx]]),
          xlab = "SHAP Value")
  abline(v = 0, lty = 1, col = "grey30")
}

dev.off()

# 15.9 SHAP by VIX Regime
vix_regimes_shap <- cut(test_current_vix[shap_sample_idx],
                        breaks = c(0, 15, 20, 25, 30, Inf),
                        labels = c("Very Low", "Low", "Medium", "High", "Very High"))

regime_shap_importance <- data.frame()
for (regime in levels(vix_regimes_shap)) {
  regime_idx <- which(vix_regimes_shap == regime)
  if (length(regime_idx) < 50) next
  
  regime_mean_shap <- colMeans(abs(shap_matrix[regime_idx, ]))
  
  for (feat in head(shap_importance$Feature, 10)) {
    feat_idx <- which(feature_cols == feat)
    regime_shap_importance <- rbind(regime_shap_importance, data.frame(
      Regime = regime,
      Feature = feat,
      Mean_Abs_SHAP = regime_mean_shap[feat_idx]
    ))
  }
}

pdf("results/figures/shap_regime_regression_ar1.pdf", width = 12, height = 8)

# Reshape for grouped barplot
regime_wide <- reshape(regime_shap_importance, 
                       idvar = "Feature", timevar = "Regime",
                       direction = "wide")

regime_matrix <- as.matrix(regime_wide[, -1])
rownames(regime_matrix) <- regime_wide$Feature

barplot(t(regime_matrix), beside = TRUE, las = 2, cex.names = 0.8,
        col = c("green3", "yellow3", "orange", "red", "darkred"),
        main = "SHAP Importance by VIX Regime (AR(1) Residual Model)",
        ylab = "Mean |SHAP|")
legend("topright", legend = levels(vix_regimes_shap), 
       fill = c("green3", "yellow3", "orange", "red", "darkred"),
       title = "VIX Regime", cex = 0.8)

dev.off()

# 15.10 Save SHAP results
saveRDS(shap_matrix, "results/models/xgb_regression_ar1_shap_matrix.rds")
saveRDS(shap_importance, "results/models/xgb_regression_ar1_shap_importance.rds")
saveRDS(importance_comparison, "results/models/xgb_regression_ar1_importance_comparison.rds")
write.csv(shap_importance, "results/tables/xgb_regression_ar1_shap_importance.csv", row.names = FALSE)
write.csv(importance_comparison, "results/tables/xgb_regression_ar1_importance_comparison.csv", row.names = FALSE)

cat_progress("SHAP analysis complete")


#------------------------------------------------------------------
# 16. BOOTSTRAP CONFIDENCE INTERVALS FOR METRICS
#------------------------------------------------------------------

cat_progress("Computing bootstrap confidence intervals...")

boot_metrics <- function(data, indices) {
  d <- data[indices, ]
  actual <- d$actual
  predicted <- d$predicted
  
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2 <- 1 - ss_res / ss_tot
  
  # QLIKE (only for positive values)
  valid <- actual > 0 & predicted > 0
  if (sum(valid) > 10) {
    qlike <- mean(log(predicted[valid]) + actual[valid] / predicted[valid])
  } else {
    qlike <- NA
  }
  
  c(RMSE = rmse, MAE = mae, R2 = r2, QLIKE = qlike)
}

boot_data <- data.frame(actual = y_test_raw, predicted = test_pred_vix)
set.seed(42)
boot_results <- boot::boot(boot_data, boot_metrics, R = 1000)

rmse_ci <- boot::boot.ci(boot_results, type = "perc", index = 1)$percent[4:5]
mae_ci <- boot::boot.ci(boot_results, type = "perc", index = 2)$percent[4:5]
r2_ci <- boot::boot.ci(boot_results, type = "perc", index = 3)$percent[4:5]
qlike_ci <- boot::boot.ci(boot_results, type = "perc", index = 4)$percent[4:5]

cat_progress("Bootstrap 95% CIs:")
cat(sprintf("  RMSE:  [%.4f, %.4f]\n", rmse_ci[1], rmse_ci[2]))
cat(sprintf("  MAE:   [%.4f, %.4f]\n", mae_ci[1], mae_ci[2]))
cat(sprintf("  R2:    [%.4f, %.4f]\n", r2_ci[1], r2_ci[2]))
cat(sprintf("  QLIKE: [%.4f, %.4f]\n", qlike_ci[1], qlike_ci[2]))


#------------------------------------------------------------------
# 17. REGIME-DEPENDENT PERFORMANCE
#------------------------------------------------------------------

cat_progress("Computing regime-dependent performance...")

vix_regimes <- cut(test_current_vix,
                   breaks = c(0, 15, 20, 25, 30, Inf),
                   labels = c("Very Low (<15)", "Low (15-20)", 
                              "Medium (20-25)", "High (25-30)", "Very High (>30)"))

regime_performance <- data.frame()
for (regime in levels(vix_regimes)) {
  idx <- vix_regimes == regime
  if (sum(idx) < 10) next
  
  metrics <- calc_regression_metrics(y_test_raw[idx], test_pred_vix[idx], test_current_vix[idx])
  
  regime_performance <- rbind(regime_performance, data.frame(
    Regime = regime,
    N = sum(idx),
    RMSE = metrics$RMSE,
    MAE = metrics$MAE,
    QLIKE = metrics$QLIKE,
    R2 = metrics$R2,
    Dir_Acc = metrics$Dir_Accuracy
  ))
}

cat_progress("Regime-Dependent Performance:")
print(regime_performance, digits = 4)


#------------------------------------------------------------------
# 18. TEMPORAL STABILITY ANALYSIS
#------------------------------------------------------------------

cat_progress("Computing temporal stability metrics...")

window_size <- 63  # ~3 months
n_windows <- floor(length(test_dates) / window_size)

rolling_metrics <- data.frame()
for (w in 1:n_windows) {
  start_idx <- (w - 1) * window_size + 1
  end_idx <- min(w * window_size, length(test_dates))
  
  idx <- start_idx:end_idx
  metrics <- calc_regression_metrics(y_test_raw[idx], test_pred_vix[idx], test_current_vix[idx])
  
  rolling_metrics <- rbind(rolling_metrics, data.frame(
    window = w,
    start_date = test_dates[start_idx],
    end_date = test_dates[end_idx],
    RMSE = metrics$RMSE,
    MAE = metrics$MAE,
    QLIKE = metrics$QLIKE,
    R2 = metrics$R2
  ))
}

mean_rolling_rmse <- mean(rolling_metrics$RMSE)
sd_rolling_rmse <- sd(rolling_metrics$RMSE)
degradation_threshold <- mean_rolling_rmse + 2 * sd_rolling_rmse
degradation_periods <- sum(rolling_metrics$RMSE > degradation_threshold)

# CUSUM test for structural breaks
cusum <- cumsum(test_errors_level - mean(test_errors_level))
max_cusum <- max(abs(cusum))
cusum_critical <- 1.36 * sqrt(length(test_errors_level)) * sd(test_errors_level)

cat_progress(sprintf("Temporal Stability: %d degradation periods detected", degradation_periods))
cat_progress(sprintf("CUSUM: max=%.4f, critical=%.4f, stable=%s", 
                     max_cusum, cusum_critical, ifelse(max_cusum < cusum_critical, "YES", "NO")))


#------------------------------------------------------------------
# 19. ADDITIONAL DIAGNOSTIC PLOTS
#------------------------------------------------------------------

cat_progress("Generating diagnostic plots...")

# 19.1 Actual vs Predicted Plot
pdf("results/figures/actual_vs_predicted_ar1.pdf", width = 10, height = 8)
par(mfrow = c(1, 2))

# Residual prediction
plot(y_test, test_pred_resid, pch = 16, cex = 0.5, col = rgb(0.2, 0.4, 0.8, 0.4),
     xlab = "Actual AR(1) Residual", ylab = "Predicted AR(1) Residual",
     main = "AR(1) Residual Prediction")
abline(0, 1, col = "red", lwd = 2)
abline(lm(test_pred_resid ~ y_test), col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Perfect", "Fitted"), col = c("red", "blue"), lty = c(1, 2), lwd = 2)

# VIX level prediction
plot(y_test_raw, test_pred_vix, pch = 16, cex = 0.5, col = rgb(0.2, 0.4, 0.8, 0.4),
     xlab = "Actual VIX", ylab = "Predicted VIX",
     main = "VIX Level Prediction (AR(1) + XGBoost)")
abline(0, 1, col = "red", lwd = 2)
abline(lm(test_pred_vix ~ y_test_raw), col = "blue", lwd = 2, lty = 2)
legend("topleft", legend = c("Perfect", "Fitted"), col = c("red", "blue"), lty = c(1, 2), lwd = 2)

dev.off()

# 19.2 Time Series Plot
pdf("results/figures/time_series_ar1.pdf", width = 14, height = 10)
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))

# Full time series
plot(test_dates, y_test_raw, type = "l", col = "black", lwd = 1,
     xlab = "Date", ylab = "VIX",
     main = "VIX: Actual vs Predicted (AR(1) + XGBoost)")
lines(test_dates, test_pred_vix, col = "red", lwd = 1)
lines(test_dates, ar1_pred_test, col = "blue", lwd = 1, lty = 2)
legend("topright", legend = c("Actual", "XGBoost+AR(1)", "AR(1) Only"), 
       col = c("black", "red", "blue"), lty = c(1, 1, 2), lwd = 2)

# Prediction intervals
plot(test_dates, y_test_raw, type = "l", col = "black", lwd = 1,
     xlab = "Date", ylab = "VIX",
     main = "VIX Prediction with 95% Confidence Intervals")
polygon(c(test_dates, rev(test_dates)), 
        c(test_pi_95_lower, rev(test_pi_95_upper)),
        col = rgb(0.8, 0.2, 0.2, 0.2), border = NA)
lines(test_dates, test_pred_vix, col = "red", lwd = 1)
lines(test_dates, y_test_raw, col = "black", lwd = 1)
legend("topright", legend = c("Actual", "Predicted", "95% PI"), 
       col = c("black", "red", rgb(0.8, 0.2, 0.2, 0.4)), lty = c(1, 1, NA), 
       pch = c(NA, NA, 15), lwd = 2)

dev.off()

# 19.3 Error Distribution Plot
pdf("results/figures/error_distribution_ar1.pdf", width = 12, height = 8)
par(mfrow = c(2, 2))

# Residual errors histogram
hist(test_errors_resid, breaks = 50, probability = TRUE, col = "lightblue",
     main = "AR(1) Residual Prediction Errors", xlab = "Error")
curve(dnorm(x, mean = mean(test_errors_resid), sd = sd(test_errors_resid)), 
      add = TRUE, col = "red", lwd = 2)

# Level errors histogram
hist(test_errors_level, breaks = 50, probability = TRUE, col = "lightblue",
     main = "VIX Level Prediction Errors", xlab = "Error")
curve(dnorm(x, mean = mean(test_errors_level), sd = sd(test_errors_level)), 
      add = TRUE, col = "red", lwd = 2)

# Q-Q plot residuals
qqnorm(std_errors_resid, main = "Q-Q Plot: AR(1) Residual Errors")
qqline(std_errors_resid, col = "red", lwd = 2)

# Q-Q plot levels
qqnorm(std_errors_level, main = "Q-Q Plot: VIX Level Errors")
qqline(std_errors_level, col = "red", lwd = 2)

dev.off()

# 19.4 ACF Plots
pdf("results/figures/acf_residuals_ar1.pdf", width = 12, height = 8)
par(mfrow = c(2, 2))

acf(test_errors_resid, lag.max = 30, main = "ACF: AR(1) Residual Errors")
pacf(test_errors_resid, lag.max = 30, main = "PACF: AR(1) Residual Errors")
acf(test_errors_level, lag.max = 30, main = "ACF: VIX Level Errors")
pacf(test_errors_level, lag.max = 30, main = "PACF: VIX Level Errors")

dev.off()

# 19.5 Rolling RMSE Plot
pdf("results/figures/rolling_rmse_ar1.pdf", width = 12, height = 6)

plot(rolling_metrics$start_date, rolling_metrics$RMSE, type = "b", pch = 16,
     col = "steelblue", lwd = 2,
     xlab = "Window Start Date", ylab = "RMSE",
     main = "Rolling RMSE (63-day windows)")
abline(h = mean_rolling_rmse, col = "red", lwd = 2, lty = 2)
abline(h = degradation_threshold, col = "orange", lwd = 2, lty = 3)
legend("topright", legend = c("Rolling RMSE", "Mean", "2SD Threshold"),
       col = c("steelblue", "red", "orange"), lty = c(1, 2, 3), lwd = 2, pch = c(16, NA, NA))

dev.off()

# 19.6 CUSUM Plot
pdf("results/figures/cusum_ar1.pdf", width = 12, height = 6)

plot(test_dates, cusum, type = "l", col = "steelblue", lwd = 2,
     xlab = "Date", ylab = "CUSUM",
     main = "CUSUM Test for Structural Stability")
abline(h = c(-cusum_critical, cusum_critical), col = "red", lwd = 2, lty = 2)
abline(h = 0, col = "grey50", lty = 3)
legend("topleft", legend = c("CUSUM", "Critical Bounds"),
       col = c("steelblue", "red"), lty = c(1, 2), lwd = 2)

dev.off()

# 19.7 Regime Performance Bar Plot
pdf("results/figures/regime_performance_ar1.pdf", width = 12, height = 8)
par(mfrow = c(2, 2), mar = c(5, 4, 3, 1))

barplot(regime_performance$RMSE, names.arg = regime_performance$Regime,
        col = c("green3", "yellow3", "orange", "red", "darkred")[1:nrow(regime_performance)],
        main = "RMSE by VIX Regime", ylab = "RMSE", las = 2)

barplot(regime_performance$MAE, names.arg = regime_performance$Regime,
        col = c("green3", "yellow3", "orange", "red", "darkred")[1:nrow(regime_performance)],
        main = "MAE by VIX Regime", ylab = "MAE", las = 2)

barplot(regime_performance$Dir_Acc * 100, names.arg = regime_performance$Regime,
        col = c("green3", "yellow3", "orange", "red", "darkred")[1:nrow(regime_performance)],
        main = "Directional Accuracy by VIX Regime", ylab = "Dir Acc (%)", las = 2)

barplot(regime_performance$N, names.arg = regime_performance$Regime,
        col = c("green3", "yellow3", "orange", "red", "darkred")[1:nrow(regime_performance)],
        main = "Sample Size by VIX Regime", ylab = "N", las = 2)

dev.off()


#------------------------------------------------------------------
# 20. SAVE ALL RESULTS
#------------------------------------------------------------------

cat_progress("Saving results...")

xgb.save(final_model, "results/models/xgb_regression_ar1_model.xgb")
saveRDS(final_params, "results/models/xgb_regression_ar1_params.rds")

# AR(1) model parameters
ar1_params <- list(
  intercept = ar1_intercept,
  phi = ar1_phi,
  sigma = ar1_sigma
)
saveRDS(ar1_params, "results/models/ar1_params.rds")

test_predictions <- data.frame(
  date = test_dates,
  actual_vix = y_test_raw,
  predicted_vix = test_pred_vix,
  ar1_pred = ar1_pred_test,
  actual_resid = y_test,
  predicted_resid = test_pred_resid,
  error_level = test_errors_level,
  error_resid = test_errors_resid,
  current_vix = test_current_vix,
  pi_95_lower = test_pi_95_lower,
  pi_95_upper = test_pi_95_upper,
  pi_90_lower = test_pi_90_lower,
  pi_90_upper = test_pi_90_upper
)
saveRDS(test_predictions, "results/models/xgb_regression_ar1_predictions.rds")

tryCatch({
  writexl::write_xlsx(test_predictions, "results/tables/xgb_regression_ar1_predictions.xlsx")
  cat_progress("Excel file saved successfully.")
}, error = function(e) {
  cat_progress("Could not write Excel file. Saving as CSV instead.")
  write.csv(test_predictions, "results/tables/xgb_regression_ar1_predictions.csv", row.names = FALSE)
})

saveRDS(cv_results, "results/models/xgb_regression_ar1_cv_results.rds")
saveRDS(cv_metrics, "results/models/xgb_regression_ar1_cv_metrics.rds")
saveRDS(cv_predictions, "results/models/xgb_regression_ar1_cv_predictions.rds")
saveRDS(importance_gain, "results/models/xgb_regression_ar1_importance_gain.rds")
saveRDS(shap_importance, "results/models/xgb_regression_ar1_shap_importance.rds")
saveRDS(shap_matrix, "results/models/xgb_regression_ar1_shap_matrix.rds")
write.csv(test_metrics, "results/tables/xgb_regression_ar1_test_metrics.csv", row.names = FALSE)
write.csv(cv_summary, "results/tables/xgb_regression_ar1_cv_summary.csv", row.names = FALSE)
write.csv(baseline_comparison, "results/tables/xgb_regression_ar1_baseline_comparison.csv", row.names = FALSE)
write.csv(regime_performance, "results/tables/xgb_regression_ar1_regime_performance.csv", row.names = FALSE)
write.csv(as.data.frame(importance_gain), "results/tables/xgb_regression_ar1_feature_importance.csv", row.names = FALSE)
write.csv(rolling_metrics, "results/tables/xgb_regression_ar1_rolling_metrics.csv", row.names = FALSE)
saveRDS(bayes_opt_result, "results/models/xgb_regression_ar1_bayes_opt.rds")

xgb_regression_ar1_results <- list(
  model_path = "results/models/xgb_regression_ar1_model.xgb",
  params = final_params,
  ar1_params = ar1_params,
  loss_function = "MSE (on AR(1) residuals)",
  hyperparameter_search = bayes_opt_result,
  cv_summary = cv_summary,
  cv_metrics = cv_metrics,
  test_metrics_resid = test_resid_metrics,
  test_metrics_level = test_metrics_list,
  test_predictions = test_predictions,
  baseline_comparison = baseline_comparison,
  statistical_tests = list(
    dm_vs_rw = dm_vs_rw,
    dm_vs_ar1 = dm_vs_ar1,
    dm_vs_mean = dm_vs_mean,
    shapiro_resid = shapiro_test_resid,
    shapiro_level = shapiro_test_level,
    ljung_box_resid = lb_test_resid,
    ljung_box_resid_sq = lb_test_resid_sq,
    ljung_box_level = lb_test_level,
    ljung_box_level_sq = lb_test_level_sq
  ),
  prediction_intervals = list(
    coverage_90 = coverage_90,
    coverage_95 = coverage_95,
    width_90 = mean_width_90,
    width_95 = mean_width_95
  ),
  bootstrap_ci = list(
    rmse = rmse_ci,
    mae = mae_ci,
    r2 = r2_ci,
    qlike = qlike_ci
  ),
  feature_importance = list(
    gain = importance_gain,
    shap = shap_importance
  ),
  regime_performance = regime_performance,
  temporal_stability = list(
    rolling_metrics = rolling_metrics,
    degradation_periods = degradation_periods,
    cusum_max = max_cusum,
    cusum_critical = cusum_critical
  ),
  feature_cols = feature_cols,
  n_features = length(feature_cols),
  purge_days = purge_days,
  embargo_days = embargo_days
)

saveRDS(xgb_regression_ar1_results, "results/models/xgb_regression_ar1_full_results.rds")

cat_progress("All results saved successfully!")


################################################################################
# END OF SCRIPT
################################################################################