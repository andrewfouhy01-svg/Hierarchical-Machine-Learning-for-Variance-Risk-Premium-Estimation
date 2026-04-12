################################################################################
# 03_XGBoost_Classification.R
################################################################################

#------------------------------------------------------------------
# 0. SETUP
#------------------------------------------------------------------

source("Setup.R")


#------------------------------------------------------------------
# 1. LOAD FEATURES
#------------------------------------------------------------------

train_features <- readRDS("data/features_train.rds")
test_features <- readRDS("data/features_test.rds")
feature_cols <- readRDS("data/feature_columns.rds")
feature_categories <- readRDS("data/feature_categories.rds")
target_config <- readRDS("results/models/target_config.rds")


#------------------------------------------------------------------
# 2. LEAKAGE PREVENTION 
#------------------------------------------------------------------

forbidden_patterns <- c("target_", "^vix$", "^spx$", "vix_close", "close")

leakage_check <- sapply(feature_cols, function(f) {
  
  any(sapply(forbidden_patterns, function(p) grepl(p, f)))
})

if (any(leakage_check)) {
  cat_progress("leakage features")
  print(feature_cols[leakage_check])
  feature_cols <- feature_cols[!leakage_check]
  cat_progress(sprintf("Removed %d leakage features", sum(leakage_check)))
}

stopifnot(all(diff(train_features$date) >= 0))
stopifnot(all(diff(test_features$date) >= 0))
stopifnot(max(train_features$date) < min(test_features$date))

cat_progress("Data leakage checks passed.")

#------------------------------------------------------------------
# 3. FEATURE PREPROCESSING
#------------------------------------------------------------------

# 3.1 Remove highly corr
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
  target_cors <- abs(cor(X_train_raw, train_features$target_direction, use = "complete.obs"))
  
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
  
  cat_progress(sprintf("Removed %d highly correlated features (r > %.2f)",
                       length(features_to_remove), cor_threshold))
}


train_features <- train_features[!is.na(target_direction)]
test_features <- test_features[!is.na(target_direction)]


X_train_class <- as.matrix(train_features[, ..feature_cols])
X_test_class <- as.matrix(test_features[, ..feature_cols])
y_train_class <- train_features$target_direction
y_test_class <- test_features$target_direction


train_dates <- train_features$date
test_dates <- test_features$date

class_balance <- table(y_train_class)
cat_progress(sprintf("Training class balance: Down=%d (%.1f%%), Up=%d (%.1f%%)",
                     class_balance[1], 100 * class_balance[1] / sum(class_balance),
                     class_balance[2], 100 * class_balance[2] / sum(class_balance)))

scale_pos_weight_class <- class_balance[1] / class_balance[2]
cat_progress(sprintf("scale_pos_weight: %.4f", scale_pos_weight_class))

#------------------------------------------------------------------
# 4. TIME-SERIES CROSS-VALIDATION (EXPANDING WINDOW)
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

n_train_class <- nrow(X_train_class)
purge_days <- 0
embargo_days <- 0

# Inner folds for Bayesian optimisation (fewer folds, smaller windows)
purged_folds_class <- create_ts_cv_folds(
  n = n_train_class,
  n_folds = 5,
  initial_window = 0.5,
  horizon = 0.05
)

# Outer folds for final CV evaluation (more folds for robustness)
cpcv_folds_class <- create_ts_cv_folds(
  n = n_train_class,
  n_folds = 10,
  initial_window = 0.5,
  horizon = 0.05
)

for (i in seq_along(purged_folds_class)) {
  cat("  Fold", i, ": Train [1:", length(purged_folds_class[[i]]$train), 
      "], Val [", min(purged_folds_class[[i]]$val), ":", max(purged_folds_class[[i]]$val), "]\n", sep = "")
}

#------------------------------------------------------------------
# 5. BAYESIAN HYPERPARAMETER OPTIMISATION WITH NESTED CV
#------------------------------------------------------------------

# 5.1 Search bounds
bounds_class <- list(
  max_depth = c(2L,5L),
  min_child_weight = c(20L, 100L),
  subsample = c(0.5, 0.95),
  colsample_bytree = c(0.4, 0.95),
  colsample_bynode = c(0.5, 1.0),
  eta = c(0.0025, 0.01),
  gamma = c(0, 0.5),
  lambda = c(1, 20),
  alpha = c(0, 1),
  max_delta_step = c(0, 3)
)

inner_folds_class <- purged_folds_class[1:3]

scoring_function_class <- function(max_depth, min_child_weight, subsample, 
                                   colsample_bytree, colsample_bynode, eta, 
                                   gamma, lambda, alpha, max_delta_step) {
  
  max_depth <- as.integer(round(max_depth))
  min_child_weight <- as.integer(round(min_child_weight))
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",  
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
    scale_pos_weight = scale_pos_weight_class,
    nthread = 2
  )
  
  cv_aucs <- rep(NA_real_, length(inner_folds_class))
  cv_iters <- rep(NA_integer_, length(inner_folds_class))
  
  for (i in seq_along(inner_folds_class)) {
    fold <- inner_folds_class[[i]]
    
    X_tr <- X_train_class[fold$train, , drop = FALSE]
    y_tr <- y_train_class[fold$train]
    X_val <- X_train_class[fold$val, , drop = FALSE]
    y_val <- y_train_class[fold$val]
    
    # Add NA and size checks
    if (any(is.na(y_tr)) || any(is.na(y_val)) ||
        length(y_tr) < 100 || length(y_val) < 20) next
    
    # Check class balance in validation
    if (length(unique(y_val)) < 2) next
    
    dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
    dval <- xgb.DMatrix(data = X_val, label = y_val)
    
    # Scale early stopping with eta
    early_stop <- min( 1000L, max(50L, as.integer(20 / eta)))
    
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 400,
      maximize = TRUE,  # AUC should be maximised
      watchlist = list(val = dval),
      early_stopping_rounds = early_stop,
      verbose = 2
    )
    
    pred_prob <- predict(model, dval)
    cv_aucs[i] <- as.numeric(pROC::auc(pROC::roc(y_val, pred_prob, quiet = TRUE)))
    cv_iters[i] <- model$best_iteration
  }
  
  valid_aucs <- cv_aucs[!is.na(cv_aucs)]
  if (length(valid_aucs) == 0) return(list(Score = -Inf))
  
  list(Score = mean(valid_aucs))
}


set.seed(42)

clusterExport(cl, c("scale_pos_weight_class", "X_train_class", "y_train_class", "inner_folds_class"))
clusterEvalQ(cl, {
  library(xgboost)
  library(pROC)
})

output_fig_dir <- "results/figures"
bayes_opt_result_class <- bayesOpt(
  FUN = scoring_function_class,
  bounds = bounds_class,
  initPoints = 28,  
  iters.n = 14,
  iters.k = 7,
  acq = "ei", 
  eps = 0.25,   
  gsPoints = 1000,
  acqThresh = 0.6,
  parallel = TRUE,
  verbose = 2,
  plotProgress = TRUE
)

pdf(file.path(output_fig_dir, "bayesian_optimisation_classification_progress.pdf"), width = 12, height = 8)
plot(bayes_opt_result_class)
dev.off()


best_params_class <- getBestPars(bayes_opt_result_class)
best_params_class$max_depth <- round(best_params_class$max_depth)
best_params_class$min_child_weight <- round(best_params_class$min_child_weight)


#------------------------------------------------------------------
# 6. OUTER CROSS-VALIDATION WITH OPTIMAL HYPERPARAMETERS
#------------------------------------------------------------------

# Final XGBoost parameters
final_params_class <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_params_class$max_depth,
  min_child_weight = best_params_class$min_child_weight,
  subsample = best_params_class$subsample,
  colsample_bytree = best_params_class$colsample_bytree,
  colsample_bynode = best_params_class$colsample_bynode,
  eta = best_params_class$eta,
  gamma = best_params_class$gamma,
  lambda = best_params_class$lambda,
  alpha = best_params_class$alpha,
  max_delta_step = best_params_class$max_delta_step,
  scale_pos_weight = scale_pos_weight_class
)

cv_results_class <- list()
cv_predictions_class <- data.frame()
cv_metrics_class <- data.frame()

for (i in seq_along(cpcv_folds_class)) {
  fold <- cpcv_folds_class[[i]]
  
  X_tr <- X_train_class[fold$train, , drop = FALSE]
  y_tr <- y_train_class[fold$train]
  X_val <- X_train_class[fold$val, , drop = FALSE]
  y_val <- y_train_class[fold$val]
  val_dates <- train_dates[fold$val]
  
  if (any(is.na(y_tr)) || any(is.na(y_val))) {
    cat_progress(sprintf("Fold %d: Skipping due to NA values", i))
    next
  }
  
  dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
  dval <- xgb.DMatrix(data = X_val, label = y_val)
  
  # Scale early stopping with eta
  early_stop <- min( 1000L, max(50L, as.integer(20 / final_params_class$eta)))
  
  model <- xgb.train(
    params = final_params_class,
    data = dtrain,
    nrounds = 4000, 
    maximize = TRUE,
    watchlist = list(val = dval),  
    early_stopping_rounds = early_stop,
    verbose = 2
  )
  
  best_iter <- model$best_iteration
  if (is.null(best_iter) || is.na(best_iter)) {
    warning(sprintf("Fold %d: early stopping failed", i))
    best_iter <- 4000L
  }
  
  # Predictions
  pred_prob <- predict(model, dval)
  pred_class <- as.integer(pred_prob >= 0.5)
  
  # Store predictions
  fold_preds <- data.frame(
    fold = i,
    date = val_dates,
    actual = y_val,
    pred_prob = pred_prob,
    pred_class = pred_class
  )
  cv_predictions_class <- rbind(cv_predictions_class, fold_preds)
  
  # Calculate metrics
  if (length(unique(y_val)) > 1) {
    roc_obj <- pROC::roc(y_val, pred_prob, quiet = TRUE)
    auc <- as.numeric(pROC::auc(roc_obj))
  } else {
    auc <- NA_real_
  }
  
  accuracy <- mean(pred_class == y_val)
  
  tp <- sum(pred_class == 1 & y_val == 1)
  tn <- sum(pred_class == 0 & y_val == 0)
  fp <- sum(pred_class == 1 & y_val == 0)
  fn <- sum(pred_class == 0 & y_val == 1)
  
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  log_loss <- -mean(y_val * log(pmax(pred_prob, 1e-15)) + 
                      (1 - y_val) * log(pmax(1 - pred_prob, 1e-15)))
  brier <- mean((pred_prob - y_val)^2)
  
  fold_metrics <- data.frame(
    fold = i,
    n_train = length(y_tr),
    n_val = length(y_val),
    auc = auc,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    log_loss = log_loss,
    brier = brier,
    best_iteration = best_iter
  )
  cv_metrics_class <- rbind(cv_metrics_class, fold_metrics)
  
  cv_results_class[[i]] <- list(
    model = model,
    predictions = fold_preds,
    metrics = fold_metrics
  )
  
  cat_progress(sprintf("Fold %d: AUC=%.4f, Acc=%.4f, F1=%.4f", 
                       i, auc, accuracy, f1))
}


cv_summary_class <- data.frame(
  Metric = c("AUC", "Accuracy", "Precision", "Recall", "F1", "Log Loss", "Brier"),
  Mean = c(mean(cv_metrics_class$auc, na.rm = TRUE),
           mean(cv_metrics_class$accuracy),
           mean(cv_metrics_class$precision),
           mean(cv_metrics_class$recall),
           mean(cv_metrics_class$f1),
           mean(cv_metrics_class$log_loss),
           mean(cv_metrics_class$brier)),
  SD = c(sd(cv_metrics_class$auc, na.rm = TRUE),
         sd(cv_metrics_class$accuracy),
         sd(cv_metrics_class$precision),
         sd(cv_metrics_class$recall),
         sd(cv_metrics_class$f1),
         sd(cv_metrics_class$log_loss),
         sd(cv_metrics_class$brier))
)
print(cv_summary_class, digits = 4)
#------------------------------------------------------------------
# 7. FINAL MODEL TRAINING ON FULL TRAINING SET
#------------------------------------------------------------------


dtrain_full_class <- xgb.DMatrix(data = X_train_class, label = y_train_class)

# Use median, no arbitrary multiplier
optimal_nrounds_class <- as.integer(median(cv_metrics_class$best_iteration, na.rm = TRUE))
cat_progress(sprintf("Training final model with nrounds = %d", optimal_nrounds_class))

final_model_class <- xgb.train(
  params = final_params_class,
  data = dtrain_full_class,
  nrounds = optimal_nrounds_class,
  verbose = 2  
)

# Save model
xgb.save(final_model_class, "results/models/xgb_vix_class_final.model")

#------------------------------------------------------------------
# 8. TEST SET EVAL
#------------------------------------------------------------------


dtest_class <- xgb.DMatrix(data = X_test_class, label = y_test_class)

test_pred_prob_class <- predict(final_model_class, dtest_class)
test_pred_class_class <- as.integer(test_pred_prob_class >= 0.5)



test_roc_class <- pROC::roc(y_test_class, test_pred_prob_class, quiet = TRUE)
test_auc_class <- as.numeric(pROC::auc(test_roc_class))

test_accuracy_class <- mean(test_pred_class_class == y_test_class)

tp <- sum(test_pred_class_class == 1 & y_test_class == 1)
tn <- sum(test_pred_class_class == 0 & y_test_class == 0)
fp <- sum(test_pred_class_class == 1 & y_test_class == 0)
fn <- sum(test_pred_class_class == 0 & y_test_class == 1)

test_precision_class <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
test_recall_class <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
test_specificity_class <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
test_f1_class <- ifelse((test_precision_class + test_recall_class) > 0, 
                        2 * test_precision_class * test_recall_class / (test_precision_class + test_recall_class), 0)

test_log_loss_class <- -mean(y_test_class * log(pmax(test_pred_prob_class, 1e-15)) + 
                               (1 - y_test_class) * log(pmax(1 - test_pred_prob_class, 1e-15)))
test_brier_class <- mean((test_pred_prob_class - y_test_class)^2)

# Matthews Correlation Coefficient
mcc_num <- (tp * tn) - (fp * fn)
mcc_den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
test_mcc_class <- ifelse(mcc_den > 0, mcc_num / mcc_den, 0)

# Cohen's Kappa
po <- test_accuracy_class
pe <- ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (length(y_test_class)^2)
test_kappa_class <- (po - pe) / (1 - pe)

test_metrics_class <- data.frame(
  Metric = c("AUC-ROC", "Accuracy", "Precision", "Recall", "Specificity",
             "F1 Score", "Log Loss", "Brier Score", "MCC", "Cohen's Kappa"),
  Value = c(test_auc_class, test_accuracy_class, test_precision_class, test_recall_class, test_specificity_class,
            test_f1_class, test_log_loss_class, test_brier_class, test_mcc_class, test_kappa_class)
)
print(test_metrics_class, digits = 4)


cat_progress("\nConfusion Matrix:")
conf_matrix_class <- matrix(c(tn, fp, fn, tp), nrow = 2, 
                            dimnames = list(Predicted = c("Down", "Up"),
                                            Actual = c("Down", "Up")))
print(conf_matrix_class)

#------------------------------------------------------------------
# 9. THRESHOLD OPTIMISATION
#------------------------------------------------------------------

thresholds <- seq(0.3, 0.7, by = 0.01)
threshold_results_class <- data.frame()

# Use CV predictions for threshold selection
cv_probs <- cv_predictions_class$pred_prob
cv_actuals <- cv_predictions_class$actual

for (thresh in thresholds) {
  pred_class_t <- as.integer(cv_probs >= thresh)
  
  tp_t <- sum(pred_class_t == 1 & cv_actuals == 1)
  tn_t <- sum(pred_class_t == 0 & cv_actuals == 0)
  fp_t <- sum(pred_class_t == 1 & cv_actuals == 0)
  fn_t <- sum(pred_class_t == 0 & cv_actuals == 1)
  
  sens_t <- tp_t / (tp_t + fn_t)
  spec_t <- tn_t / (tn_t + fp_t)
  youden_j <- sens_t + spec_t - 1
  
  prec_t <- ifelse((tp_t + fp_t) > 0, tp_t / (tp_t + fp_t), 0)
  f1_t <- ifelse((prec_t + sens_t) > 0, 2 * prec_t * sens_t / (prec_t + sens_t), 0)
  
  threshold_results_class <- rbind(threshold_results_class, data.frame(
    threshold = thresh,
    sensitivity = sens_t,
    specificity = spec_t,
    youden_j = youden_j,
    precision = prec_t,
    f1 = f1_t,
    accuracy = (tp_t + tn_t) / length(cv_actuals)
  ))
}

optimal_youden_class <- threshold_results_class$threshold[which.max(threshold_results_class$youden_j)]
optimal_f1_class <- threshold_results_class$threshold[which.max(threshold_results_class$f1)]

cat_progress(sprintf("Optimal threshold (Youden): %.2f", optimal_youden_class))
cat_progress(sprintf("Optimal threshold (F1): %.2f", optimal_f1_class))

# NOW apply to test set
test_pred_optimal <- as.integer(test_pred_prob_class >= optimal_youden_class)


#------------------------------------------------------------------
# 10. CALIBRATION
#------------------------------------------------------------------

n_bins <- 20
bin_edges <- seq(0, 1, length.out = n_bins + 1)
calibration_df_class <- data.frame()

for (i in 1:n_bins) {
  bin_mask <- test_pred_prob_class >= bin_edges[i] & test_pred_prob_class < bin_edges[i + 1]
  if (sum(bin_mask) > 0) {
    mean_pred <- mean(test_pred_prob_class[bin_mask])
    mean_actual <- mean(y_test_class[bin_mask])
    n_obs <- sum(bin_mask)
    
    calibration_df_class <- rbind(calibration_df_class, data.frame(
      bin = i,
      bin_mid = (bin_edges[i] + bin_edges[i + 1]) / 2,
      mean_predicted = mean_pred,
      mean_actual = mean_actual,
      n = n_obs
    ))
  }
}


ece_class <- sum(calibration_df_class$n * abs(calibration_df_class$mean_predicted - calibration_df_class$mean_actual)) / 
  sum(calibration_df_class$n)
mce_class <- max(abs(calibration_df_class$mean_predicted - calibration_df_class$mean_actual))


#------------------------------------------------------------------
# 11. REGIME-CONDITIONAL PERFORMANCE
#------------------------------------------------------------------

test_results_class <- data.frame(
  date = test_dates,
  actual = y_test_class,
  pred_prob = test_pred_prob_class,
  pred_class = test_pred_class_class,
  regime = test_features$regime_252,
  vix_level = test_features$vix
)
names(test_features)
regime_performance_class <- data.frame()

for (r in 1:3) {
  regime_mask <- test_results_class$regime == r
  if (sum(regime_mask) > 10) {
    y_r <- test_results_class$actual[regime_mask]
    pred_r <- test_results_class$pred_prob[regime_mask]
    class_r <- test_results_class$pred_class[regime_mask]
    
    if (length(unique(y_r)) > 1) {
      auc_r <- as.numeric(pROC::auc(pROC::roc(y_r, pred_r, quiet = TRUE)))
    } else {
      auc_r <- NA
    }
    
    acc_r <- mean(class_r == y_r)
    
    regime_performance_class <- rbind(regime_performance_class, data.frame(
      regime = c("Low", "Medium", "High")[r],
      n = sum(regime_mask),
      accuracy = acc_r,
      auc = auc_r
    ))
  }
}


#------------------------------------------------------------------
# 12. FEATURE IMPORTANCE & SHAP ANALYSIS
#------------------------------------------------------------------

# 12.1 Gain-based importance
importance_gain_class <- xgb.importance(model = final_model_class, feature_names = feature_cols)
importance_gain_class <- importance_gain_class[order(-Gain)]

# 12.2 SHAP values computation
shap_sample_size <- min(5000, nrow(X_test_class))
set.seed(42)
shap_idx_class <- sample(1:nrow(X_test_class), shap_sample_size)

# SHAP values (on log-odds scale for classification)
shap_values_class <- predict(final_model_class, X_test_class[shap_idx_class, ], predcontrib = TRUE)
shap_matrix_class <- shap_values_class[, -ncol(shap_values_class)]  # Remove BIAS column
colnames(shap_matrix_class) <- feature_cols
shap_bias_class <- shap_values_class[, ncol(shap_values_class)]

# Feature matrix for sampled observations
X_shap_class <- X_test_class[shap_idx_class, ]
y_shap_class <- y_test_class[shap_idx_class]
pred_prob_shap <- test_pred_prob_class[shap_idx_class]

# 12.3 SHAP importance (mean |SHAP|)
shap_importance_class <- data.frame(
  Feature = feature_cols,
  Mean_Abs_SHAP = colMeans(abs(shap_matrix_class)),
  Mean_SHAP = colMeans(shap_matrix_class),
  SD_SHAP = apply(shap_matrix_class, 2, sd)
)
shap_importance_class <- shap_importance_class[order(-shap_importance_class$Mean_Abs_SHAP), ]
rownames(shap_importance_class) <- NULL

# 12.4 Compare Gain vs SHAP importance
importance_comparison_class <- merge(
  importance_gain_class[, .(Feature, Gain)],
  shap_importance_class[, c("Feature", "Mean_Abs_SHAP")],
  by = "Feature"
)
importance_comparison_class$Gain_Rank <- rank(-importance_comparison_class$Gain)
importance_comparison_class$SHAP_Rank <- rank(-importance_comparison_class$Mean_Abs_SHAP)
importance_comparison_class$Rank_Diff <- abs(importance_comparison_class$Gain_Rank - importance_comparison_class$SHAP_Rank)
importance_comparison_class <- importance_comparison_class[order(importance_comparison_class$SHAP_Rank), ]

cat_progress("Top 15 features by SHAP importance (Classification):")
print(head(shap_importance_class, 15), digits = 4)

# 12.5 SHAP Summary Plot (Beeswarm)
top_n_features <- min(20, length(feature_cols))
top_features_class <- head(shap_importance_class$Feature, top_n_features)
top_feature_idx_class <- match(top_features_class, feature_cols)

pdf("results/figures/shap_summary_classification.pdf", width = 12, height = 10)

par(mar = c(5, 12, 4, 2))
plot(NULL, xlim = range(shap_matrix_class[, top_feature_idx_class]), 
     ylim = c(0.5, top_n_features + 0.5),
     xlab = "SHAP Value (impact on log-odds)", ylab = "", yaxt = "n",
     main = "SHAP Summary Plot - Classification\n(Positive = Push toward VIX Up)")
axis(2, at = 1:top_n_features, labels = rev(top_features_class), las = 1, cex.axis = 0.8)
abline(v = 0, lty = 2, col = "grey50")

for (i in 1:top_n_features) {
  feat_idx <- top_feature_idx_class[i]
  shap_vals <- shap_matrix_class[, feat_idx]
  feat_vals <- X_shap_class[, feat_idx]
  
  # Normalise feature values for colour mapping
  feat_norm <- (feat_vals - min(feat_vals, na.rm = TRUE)) / 
    (max(feat_vals, na.rm = TRUE) - min(feat_vals, na.rm = TRUE) + 1e-10)
  
  # Colour gradient: blue (low) to red (high)
  colours <- rgb(feat_norm, 0.2, 1 - feat_norm, 0.6)
  
  # Jitter y-position
  y_pos <- top_n_features - i + 1 + runif(length(shap_vals), -0.3, 0.3)
  
  points(shap_vals, y_pos, pch = 16, cex = 0.5, col = colours)
}

legend("topright", legend = c("High", "Low"), 
       pch = 16, col = c("red", "blue"), title = "Feature Value",
       bty = "n", cex = 0.9)

dev.off()

# 12.6 SHAP Dependence Plots for Top Features
pdf("results/figures/shap_dependence_classification.pdf", width = 14, height = 12)
par(mfrow = c(3, 3), mar = c(4, 4, 3, 1))

top_9_features_class <- head(shap_importance_class$Feature, min(9, length(feature_cols)))

for (feat in top_9_features_class) {
  feat_idx <- which(feature_cols == feat)
  shap_vals <- shap_matrix_class[, feat_idx]
  feat_vals <- X_shap_class[, feat_idx]
  
  # Colour by actual outcome
  colours <- ifelse(y_shap_class == 1, 
                    rgb(0.8, 0.2, 0.2, 0.5),  # Red for VIX Up
                    rgb(0.2, 0.2, 0.8, 0.5))  # Blue for VIX Down
  
  plot(feat_vals, shap_vals, pch = 16, cex = 0.5, col = colours,
       xlab = feat, ylab = "SHAP Value (log-odds)",
       main = feat)
  
  # Add LOESS smoother
  if (length(unique(feat_vals)) > 10) {
    lo <- loess(shap_vals ~ feat_vals, span = 0.3)
    ord <- order(feat_vals)
    lines(feat_vals[ord], predict(lo)[ord], col = "black", lwd = 2)
  }
  abline(h = 0, lty = 2, col = "grey50")
}

dev.off()

# 12.7 SHAP Force Plot - Correct vs Incorrect Predictions
pdf("results/figures/shap_waterfall_classification.pdf", width = 14, height = 10)
par(mfrow = c(2, 2), mar = c(4, 10, 3, 2))

# Find examples: true positive, true negative, false positive, false negative
pred_class_shap <- as.integer(pred_prob_shap >= 0.5)
tp_idx <- which(pred_class_shap == 1 & y_shap_class == 1)
tn_idx <- which(pred_class_shap == 0 & y_shap_class == 0)
fp_idx <- which(pred_class_shap == 1 & y_shap_class == 0)
fn_idx <- which(pred_class_shap == 0 & y_shap_class == 1)

# Pick highest confidence example from each category
example_cases <- list(
  "True Positive (Confident)" = if(length(tp_idx) > 0) tp_idx[which.max(pred_prob_shap[tp_idx])] else NA,
  "True Negative (Confident)" = if(length(tn_idx) > 0) tn_idx[which.min(pred_prob_shap[tn_idx])] else NA,
  "False Positive" = if(length(fp_idx) > 0) fp_idx[which.max(pred_prob_shap[fp_idx])] else NA,
  "False Negative" = if(length(fn_idx) > 0) fn_idx[which.min(pred_prob_shap[fn_idx])] else NA
)

for (case_name in names(example_cases)) {
  idx <- example_cases[[case_name]]
  if (is.na(idx)) {
    plot.new()
    text(0.5, 0.5, paste(case_name, "- No examples"), cex = 1.2)
    next
  }
  
  shap_vals <- shap_matrix_class[idx, ]
  
  # Sort by absolute SHAP value, keep top 10
  top_contrib <- order(abs(shap_vals), decreasing = TRUE)[1:min(10, length(shap_vals))]
  shap_sorted <- shap_vals[top_contrib]
  feat_names <- feature_cols[top_contrib]
  
  # Plot
  colours <- ifelse(shap_sorted > 0, "firebrick", "steelblue")
  
  barplot(shap_sorted, names.arg = feat_names, horiz = TRUE, las = 1,
          col = colours, border = NA, cex.names = 0.7,
          main = sprintf("%s\nP(Up)=%.2f, Actual=%s", 
                         case_name, pred_prob_shap[idx],
                         ifelse(y_shap_class[idx] == 1, "Up", "Down")),
          xlab = "SHAP Value (log-odds)")
  abline(v = 0, lty = 1, col = "grey30")
}

dev.off()

# 12.8 SHAP Analysis by Prediction Confidence
confidence <- abs(pred_prob_shap - 0.5)
conf_tertiles <- cut(confidence, 
                     breaks = quantile(confidence, c(0, 1/3, 2/3, 1)),
                     labels = c("Low", "Medium", "High"),
                     include.lowest = TRUE)

confidence_shap <- data.frame()
for (conf_level in levels(conf_tertiles)) {
  conf_idx <- which(conf_tertiles == conf_level)
  if (length(conf_idx) < 30) next
  
  conf_mean_shap <- colMeans(abs(shap_matrix_class[conf_idx, ]))
  
  for (feat in head(shap_importance_class$Feature, 10)) {
    feat_idx <- which(feature_cols == feat)
    confidence_shap <- rbind(confidence_shap, data.frame(
      Confidence = conf_level,
      Feature = feat,
      Mean_Abs_SHAP = conf_mean_shap[feat_idx]
    ))
  }
}

pdf("results/figures/shap_confidence_classification.pdf", width = 12, height = 6)
par(mfrow = c(1, 2))

# Barplot by confidence level
if (nrow(confidence_shap) > 0) {
  conf_wide <- reshape(confidence_shap, 
                       idvar = "Feature", timevar = "Confidence",
                       direction = "wide")
  conf_matrix <- as.matrix(conf_wide[, -1])
  rownames(conf_matrix) <- conf_wide$Feature
  
  barplot(t(conf_matrix), beside = TRUE, las = 2, cex.names = 0.7,
          col = c("lightblue", "steelblue", "darkblue"),
          main = "SHAP Importance by Prediction Confidence",
          ylab = "Mean |SHAP|")
  legend("topright", legend = c("Low", "Medium", "High"), 
         fill = c("lightblue", "steelblue", "darkblue"),
         title = "Confidence", cex = 0.8)
}

# SHAP values distribution for correct vs incorrect
correct_mask <- (pred_class_shap == y_shap_class)
total_shap_magnitude <- rowSums(abs(shap_matrix_class))

boxplot(total_shap_magnitude ~ correct_mask, 
        names = c("Incorrect", "Correct"),
        col = c("salmon", "lightgreen"),
        main = "Total SHAP Magnitude by Prediction Outcome",
        ylab = "Sum of |SHAP| values")

dev.off()

# 12.9 Day-of-Week SHAP Analysis (since DOW is top feature)
if ("dow" %in% feature_cols) {
  dow_idx <- which(feature_cols == "dow")
  dow_values <- X_shap_class[, dow_idx]
  dow_shap <- shap_matrix_class[, dow_idx]
  
  dow_analysis <- data.frame(
    dow = dow_values,
    shap = dow_shap,
    actual = y_shap_class
  )
  
  dow_summary <- aggregate(cbind(shap, actual) ~ dow, data = dow_analysis, 
                           FUN = function(x) c(mean = mean(x), sd = sd(x), n = length(x)))
  
  pdf("results/figures/shap_dow_classification.pdf", width = 10, height = 6)
  par(mfrow = c(1, 2))
  
  # SHAP by day of week
  dow_labels <- c("Mon", "Tue", "Wed", "Thu", "Fri")
  barplot(dow_summary$shap[, "mean"], names.arg = dow_labels,
          col = ifelse(dow_summary$shap[, "mean"] > 0, "firebrick", "steelblue"),
          main = "Mean SHAP Value by Day of Week",
          ylab = "Mean SHAP (log-odds)",
          xlab = "Day of Week")
  abline(h = 0, lty = 2)
  
  # Actual up probability by day
  up_prob_dow <- aggregate(actual ~ dow, data = dow_analysis, FUN = mean)
  barplot(up_prob_dow$actual, names.arg = dow_labels,
          col = "grey60",
          main = "Actual P(VIX Up) by Day of Week",
          ylab = "Proportion VIX Up",
          xlab = "Day of Week",
          ylim = c(0, 1))
  abline(h = 0.5, lty = 2, col = "red")
  
  dev.off()
  
  cat_progress("Day-of-week analysis complete")
}

# 12.10 SHAP Interaction Matrix (approximate)
cat_progress("Computing SHAP interaction approximations...")

top_features_interact <- head(shap_importance_class$Feature, min(10, length(feature_cols)))
top_idx_interact <- match(top_features_interact, feature_cols)
shap_top_class <- shap_matrix_class[, top_idx_interact]

# Correlation of SHAP values as proxy for interaction
shap_cor_matrix_class <- cor(shap_top_class)
colnames(shap_cor_matrix_class) <- rownames(shap_cor_matrix_class) <- top_features_interact

pdf("results/figures/shap_interaction_classification.pdf", width = 10, height = 8)

# Heatmap of SHAP correlations
image(1:nrow(shap_cor_matrix_class), 1:ncol(shap_cor_matrix_class), 
      shap_cor_matrix_class,
      col = colorRampPalette(c("blue", "white", "red"))(50),
      xlab = "", ylab = "", axes = FALSE,
      main = "SHAP Value Correlations (Interaction Proxy)")
axis(1, at = 1:nrow(shap_cor_matrix_class), labels = top_features_interact, las = 2, cex.axis = 0.7)
axis(2, at = 1:ncol(shap_cor_matrix_class), labels = top_features_interact, las = 1, cex.axis = 0.7)

# Add correlation values
for (i in 1:nrow(shap_cor_matrix_class)) {
  for (j in 1:ncol(shap_cor_matrix_class)) {
    if (i != j) {
      text(i, j, sprintf("%.2f", shap_cor_matrix_class[i, j]), cex = 0.6)
    }
  }
}

dev.off()

# 12.11 Save SHAP results
saveRDS(shap_matrix_class, "results/models/xgb_classification_shap_matrix.rds")
saveRDS(shap_importance_class, "results/models/xgb_classification_shap_importance.rds")
saveRDS(importance_comparison_class, "results/models/xgb_classification_importance_comparison.rds")
write.csv(shap_importance_class, "results/tables/xgb_classification_shap_importance.csv", row.names = FALSE)
write.csv(importance_comparison_class, "results/tables/xgb_classification_importance_comparison.csv", row.names = FALSE)

cat_progress("SHAP analysis complete (Classification)")

#------------------------------------------------------------------
# 13. STAT TESTS
#------------------------------------------------------------------

binom_test_class <- binom.test(sum(test_pred_class_class == y_test_class), length(y_test_class), p = 0.5)


majority_class <- as.integer(mean(y_train_class) > 0.5)
naive_accuracy_class <- mean(y_test_class == majority_class)

# McNemar's test
naive_pred <- rep(majority_class, length(y_test_class))
xgb_correct <- test_pred_class_class == y_test_class
naive_correct <- naive_pred == y_test_class

contingency <- matrix(c(
  sum(xgb_correct & naive_correct),   
  sum(xgb_correct & !naive_correct),  
  sum(!xgb_correct & naive_correct),  
  sum(!xgb_correct & !naive_correct)  
), nrow = 2)

mcnemar_result_class <- mcnemar.test(contingency)

set.seed(42)
n_bootstrap <- 10000
bootstrap_aucs_class <- numeric(n_bootstrap)

for (b in 1:n_bootstrap) {
  boot_idx <- sample(1:length(y_test_class), replace = TRUE)
  y_boot <- y_test_class[boot_idx]
  pred_boot <- test_pred_prob_class[boot_idx]
  
  if (length(unique(y_boot)) > 1) {
    bootstrap_aucs_class[b] <- as.numeric(pROC::auc(pROC::roc(y_boot, pred_boot, quiet = TRUE)))
  } else {
    bootstrap_aucs_class[b] <- NA
  }
}

bootstrap_aucs_class <- bootstrap_aucs_class[!is.na(bootstrap_aucs_class)]
auc_ci_class <- quantile(bootstrap_aucs_class, c(0.025, 0.975))


#------------------------------------------------------------------
# 14. TEMPORAL STABILITY 
#------------------------------------------------------------------


window_size <- 63  # ~3 months
rolling_accuracy_class <- data.frame()

for (i in window_size:nrow(test_results_class)) {
  window_idx <- (i - window_size + 1):i
  acc_w <- mean(test_results_class$pred_class[window_idx] == test_results_class$actual[window_idx])
  
  rolling_accuracy_class <- rbind(rolling_accuracy_class, data.frame(
    date = test_results_class$date[i],
    accuracy = acc_w
  ))
}

mean_rolling_acc_class <- mean(rolling_accuracy_class$accuracy)
sd_rolling_acc_class <- sd(rolling_accuracy_class$accuracy)
degradation_threshold_class <- mean_rolling_acc_class - 2 * sd_rolling_acc_class

degradation_periods_class <- rolling_accuracy_class$date[rolling_accuracy_class$accuracy < degradation_threshold_class]


#------------------------------------------------------------------
# 15. PLOTS
#------------------------------------------------------------------

pdf("results/figures/22_xgb_classification_roc.pdf", width = 12, height = 6)

par(mfrow = c(1, 2))

plot(test_roc_class, main = sprintf("ROC Curve (AUC = %.4f)", test_auc_class),
     col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "grey")

pr_curve_class <- PRROC::pr.curve(scores.class0 = test_pred_prob_class, weights.class0 = y_test_class, curve = TRUE)
plot(pr_curve_class$curve[, 1], pr_curve_class$curve[, 2], type = "l", col = "blue", lwd = 2,
     xlab = "Recall", ylab = "Precision",
     main = sprintf("Precision-Recall Curve (AUC = %.4f)", pr_curve_class$auc.integral))
baseline_pr <- sum(y_test_class) / length(y_test_class)
abline(h = baseline_pr, lty = 2, col = "grey")

dev.off()

pdf("results/figures/23_xgb_calibration.pdf", width = 10, height = 8)

par(mfrow = c(2, 2))

plot(calibration_df_class$mean_predicted, calibration_df_class$mean_actual,
     pch = 19, cex = sqrt(calibration_df_class$n) / 10,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Mean Predicted Probability", ylab = "Mean Actual Frequency",
     main = sprintf("Calibration Plot (ECE = %.4f)", ece_class))
abline(0, 1, col = "red", lty = 2)

hist(test_pred_prob_class, breaks = 50, main = "Prediction Distribution",
     xlab = "Predicted Probability", col = "lightblue")
abline(v = 0.5, col = "red", lty = 2)

plot(threshold_results_class$threshold, threshold_results_class$youden_j, type = "l",
     col = "blue", lwd = 2, xlab = "Threshold", ylab = "Score",
     main = "Threshold Optimisation", ylim = c(0, max(threshold_results_class$youden_j) * 1.2))
lines(threshold_results_class$threshold, threshold_results_class$f1, col = "red", lwd = 2)
abline(v = optimal_youden_class, col = "blue", lty = 2)
abline(v = optimal_f1_class, col = "red", lty = 2)
legend("topright", c("Youden's J", "F1"), col = c("blue", "red"), lwd = 2)

conf_matrix_norm_class <- conf_matrix_class / rowSums(conf_matrix_class)
image(1:2, 1:2, t(conf_matrix_norm_class)[, 2:1], col = heat.colors(20),
      xlab = "Predicted", ylab = "Actual", main = "Normalised Confusion Matrix",
      axes = FALSE)
axis(1, at = 1:2, labels = c("Down", "Up"))
axis(2, at = 1:2, labels = c("Up", "Down"))
for (i in 1:2) {
  for (j in 1:2) {
    text(i, 3 - j, sprintf("%d\n(%.1f%%)", conf_matrix_class[i, j], 
                           100 * conf_matrix_norm_class[i, j]), cex = 0.9)
  }
}

dev.off()

pdf("results/figures/24_xgb_feature_importance.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

top_20_gain_class <- head(importance_gain_class, 20)
barplot(rev(top_20_gain_class$Gain), names.arg = rev(top_20_gain_class$Feature),
        horiz = TRUE, las = 1, cex.names = 0.6,
        main = "Top 20 Features by Gain", xlab = "Gain",
        col = "steelblue")

top_20_shap_class <- head(shap_importance_class, 20)
barplot(rev(top_20_shap_class$Mean_Abs_SHAP), names.arg = rev(top_20_shap_class$Feature),
        horiz = TRUE, las = 1, cex.names = 0.6,
        main = "Top 20 Features by Mean |SHAP|", xlab = "Mean |SHAP|",
        col = "coral")

category_importance_class <- data.frame()
for (cat_name in names(feature_categories)) {
  cat_features <- intersect(feature_categories[[cat_name]], importance_gain_class$Feature)
  if (length(cat_features) > 0) {
    cat_gain <- sum(importance_gain_class$Gain[importance_gain_class$Feature %in% cat_features])
    category_importance_class <- rbind(category_importance_class, data.frame(
      category = cat_name,
      total_gain = cat_gain,
      n_features = length(cat_features)
    ))
  }
}
category_importance_class <- category_importance_class[order(-category_importance_class$total_gain), ]

barplot(category_importance_class$total_gain, names.arg = category_importance_class$category,
        las = 2, cex.names = 0.7, main = "Feature Category Importance",
        ylab = "Total Gain", col = rainbow(nrow(category_importance_class)))

top_feature_class <- shap_importance_class$Feature[1]
top_feature_idx_class <- which(feature_cols == top_feature_class)
plot(X_test_class[shap_idx_class, top_feature_idx_class], shap_matrix_class[, top_feature_idx_class],
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3),
     xlab = top_feature_class, ylab = "SHAP Value",
     main = sprintf("SHAP Dependence: %s", top_feature_class))
abline(h = 0, col = "red", lty = 2)
lines(lowess(X_test_class[shap_idx_class, top_feature_idx_class], shap_matrix_class[, top_feature_idx_class]), 
      col = "red", lwd = 2)

dev.off()

pdf("results/figures/25_xgb_temporal_stability.pdf", width = 14, height = 10)

par(mfrow = c(2, 2))

plot(rolling_accuracy_class$date, rolling_accuracy_class$accuracy, type = "l",
     col = "blue", lwd = 1.5, ylim = c(0.3, 0.8),
     main = "Rolling 63-day Accuracy", xlab = "Date", ylab = "Accuracy")
abline(h = 0.5, col = "red", lty = 2)
abline(h = mean_rolling_acc_class, col = "green", lty = 2)
abline(h = degradation_threshold_class, col = "orange", lty = 2)

barplot(regime_performance_class$accuracy, names.arg = regime_performance_class$regime,
        main = "Accuracy by VIX Regime", ylab = "Accuracy",
        col = c("green", "yellow", "red"), ylim = c(0, 1))
abline(h = 0.5, col = "red", lty = 2)

cum_correct_class <- cumsum(test_pred_class_class == y_test_class)
cum_n_class <- 1:length(y_test_class)
cum_accuracy_class <- cum_correct_class / cum_n_class

plot(test_dates, cum_accuracy_class, type = "l", col = "blue", lwd = 1.5,
     main = "Cumulative Accuracy Over Time", xlab = "Date", ylab = "Cumulative Accuracy")
abline(h = 0.5, col = "red", lty = 2)

correct_mask_class <- test_pred_class_class == y_test_class
correct_probs_class <- abs(test_pred_prob_class - 0.5)
correct_probs_correct_class <- correct_probs_class[correct_mask_class]
correct_probs_wrong_class <- correct_probs_class[!correct_mask_class]

boxplot(list(Correct = correct_probs_correct_class, Wrong = correct_probs_wrong_class),
        main = "Prediction Confidence by Outcome",
        ylab = "Distance from 0.5", col = c("green", "red"))

dev.off()

#------------------------------------------------------------------
# 16. SAVE RES
#------------------------------------------------------------------

xgb.save(final_model_class, "results/models/xgb_classification_model.xgb")
saveRDS(final_params_class, "results/models/xgb_classification_params.rds")

test_predictions_class <- data.frame(
  date = test_dates,
  actual = y_test_class,
  pred_prob = test_pred_prob_class,
  pred_class = test_pred_class_class,
  pred_class_optimal = as.integer(test_pred_prob_class >= optimal_youden_class)
)
saveRDS(test_predictions_class, "results/models/xgb_classification_predictions.rds")
writexl::write_xlsx(test_predictions_class, "results/tables/xgb_classification_predictions.xlsx")

saveRDS(cv_results_class, "results/models/xgb_classification_cv_results.rds")
saveRDS(cv_metrics_class, "results/models/xgb_classification_cv_metrics.rds")
saveRDS(cv_predictions_class, "results/models/xgb_classification_cv_predictions.rds")
saveRDS(importance_gain_class, "results/models/xgb_classification_importance_gain.rds")
saveRDS(shap_importance_class, "results/models/xgb_classification_shap_importance.rds")
saveRDS(shap_matrix_class, "results/models/xgb_classification_shap_matrix.rds")
write.csv(test_metrics_class, "results/tables/xgb_classification_test_metrics.csv", row.names = FALSE)
write.csv(cv_summary_class, "results/tables/xgb_classification_cv_summary.csv", row.names = FALSE)
write.csv(threshold_results_class, "results/tables/xgb_classification_thresholds.csv", row.names = FALSE)
write.csv(calibration_df_class, "results/tables/xgb_classification_calibration.csv", row.names = FALSE)
write.csv(regime_performance_class, "results/tables/xgb_classification_regime_performance.csv", row.names = FALSE)
write.csv(as.data.frame(importance_gain_class), "results/tables/xgb_classification_feature_importance.csv", row.names = FALSE)
saveRDS(bayes_opt_result_class, "results/models/xgb_classification_bayes_opt.rds")


xgb_classification_results <- list(
  model_path = "results/models/xgb_classification_model.xgb",
  params = final_params_class,
  hyperparameter_search = bayes_opt_result_class,
  cv_summary = cv_summary_class,
  cv_metrics = cv_metrics_class,
  test_metrics = test_metrics_class,
  test_predictions = test_predictions_class,
  threshold_analysis = list(
    optimal_youden = optimal_youden_class,
    optimal_f1 = optimal_f1_class,
    threshold_results = threshold_results_class
  ),
  calibration = list(
    ece = ece_class,
    mce = mce_class,
    calibration_df = calibration_df_class
  ),
  statistical_tests = list(
    binom_test = binom_test_class,
    mcnemar_vs_naive = mcnemar_result_class,
    auc_bootstrap_ci = auc_ci_class
  ),
  feature_importance = list(
    gain = importance_gain_class,
    shap = shap_importance_class
  ),
  regime_performance = regime_performance_class,
  temporal_stability = list(
    rolling_accuracy = rolling_accuracy_class,
    degradation_periods = degradation_periods_class
  ),
  feature_cols = feature_cols,
  n_features = length(feature_cols),
  purge_days = purge_days,
  embargo_days = embargo_days
)

saveRDS(xgb_classification_results, "results/models/xgb_classification_full_results.rds")


################################################################################
# END OF SCRIPT
################################################################################