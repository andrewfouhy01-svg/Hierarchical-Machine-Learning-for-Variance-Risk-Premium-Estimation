#------------------------------------------------------------------
# RV_Forward_XGBoost_Classification.R
# XGBoost Classification for Predicting 22-Day Forward Realised Volatility
#
# OBJECTIVE:
# ----------
# Predict direction of RV_{t+1, t+23} for direct comparison with VIX_{t+1}
# as a forecast of future realised volatility.
#
# MODEL ARCHITECTURE:
# -------------------
# - XGBoost binary classifier
# - Target: I{RV_{t+1,t+23} > RV_{t,t+22}} (RV direction)
# - Features: Historical RV (lagged 22+ days), SPX returns, range estimators
# - No VIX features used (enables fair comparison with VIX as benchmark)
#
# EVALUATION:
# -----------
# 1. Standard classification metrics (AUC, accuracy, F1, etc.)
# 2. Comparison with VIX-implied direction
# 3. Economic significance for vol trading strategies
#
#------------------------------------------------------------------

#------------------------------------------------------------------
# 0. SOURCE SETUP AND CONFIGURATION
#------------------------------------------------------------------

source("Setup.R")

# Additional packages
rv_packages <- c("xgboost", "caret", "pROC", "PRROC")
install_and_load(rv_packages)

if (!require("shapviz", quietly = TRUE)) {
  install.packages("shapviz")
  library(shapviz)
}

cat_progress(paste(rep("=", 70), collapse = ""))
cat_progress("XGBoost RV Forward Classification: PhD-Level Implementation")
cat_progress("Target: 22-Day Forward Realised Volatility Direction")
cat_progress(paste(rep("=", 70), collapse = ""))

# ============================================================================
# CONFIGURATION: Centralised parameters for reproducibility
# ============================================================================
CONFIG <- list(
  # Cross-validation
  n_folds = 15L,
  n_cpcv_groups = 6L,
  n_cpcv_test_groups = 2L,
  purge_days = 22L,        
  embargo_days = 22L,      
  
  # Bayesian optimisation
  bo_init_points = 98L,
  bo_iterations = 49L,
  bo_parallel_k = 7L,
  bo_gs_points = 14500L,
  
  # Training
  nrounds_max = 2000L,
  early_stopping = 600L,
  
  # Preprocessing
  cor_threshold = 0.9999,
  na_threshold_pct = 0.20,
  
  # Analysis
  shap_sample_size = 1000L,
  n_bootstrap = 1000L,
  rolling_window = 63L,
  
  # Numerical stability
  eps = 1e-10,
  
  # Random seed
  seed = 42L
)

set.seed(CONFIG$seed)

#------------------------------------------------------------------
# 1. LOAD RV FEATURE DATA
#------------------------------------------------------------------

cat_progress("Loading RV feature data...")

train_rv <- readRDS("data/rv_forward/features_rv_train.rds")
test_rv <- readRDS("data/rv_forward/features_rv_test.rds")
feature_cols_rv <- readRDS("data/rv_forward/feature_columns_rv.rds")
feature_categories_rv <- readRDS("data/rv_forward/feature_categories_rv.rds")
target_config_rv <- readRDS("results/models/rv_forward/target_config_rv.rds")

setDT(train_rv)
setDT(test_rv)

cat_progress(sprintf("Training set: %d observations, %d features",
                     nrow(train_rv), length(feature_cols_rv)))
cat_progress(sprintf("Test set: %d observations", nrow(test_rv)))
cat_progress(sprintf("Date range: Train %s to %s, Test %s to %s",
                     min(train_rv$date), max(train_rv$date),
                     min(test_rv$date), max(test_rv$date)))

#------------------------------------------------------------------
# 2. RIGOROUS DATA LEAKAGE PREVENTION
#------------------------------------------------------------------

cat_progress("Performing rigorous data leakage checks...")

# 2.1 Forbidden pattern matching
forbidden_patterns <- list(
  "Target variables" = "^target_|_next$",
  "Contemporaneous RV" = "^rv_(cc|parkinson|gk|rs|yz|daily|weekly|monthly)$",
  "Contemporaneous VRP" = "^vrp_",
  "Future actuals" = "actual$|_actual$",
  "Error terms" = "error|residual"
)

leakage_detected <- FALSE
for (pattern_name in names(forbidden_patterns)) {
  matches <- grep(forbidden_patterns[[pattern_name]], feature_cols_rv, value = TRUE)
  if (length(matches) > 0) {
    cat_progress(sprintf("LEAKAGE - %s: %s", pattern_name, paste(matches, collapse = ", ")))
    feature_cols_rv <- setdiff(feature_cols_rv, matches)
    leakage_detected <- TRUE
  }
}

if (leakage_detected) {
  cat_progress(sprintf("Removed leaky features. Remaining: %d", length(feature_cols_rv)))
}

# 2.2 Temporal ordering verification
stopifnot("Training data not temporally ordered" = all(diff(train_rv$date) >= 0))
stopifnot("Test data not temporally ordered" = all(diff(test_rv$date) >= 0))
stopifnot("Train/test overlap detected" = max(train_rv$date) < min(test_rv$date))

# 2.3 Feature-target correlation check
X_check <- as.matrix(train_rv[, ..feature_cols_rv])
y_check <- train_rv$target_rv_direction

cor_with_target <- sapply(seq_len(ncol(X_check)), function(j) {
  if (sd(X_check[, j], na.rm = TRUE) > 0) {
    cor(X_check[, j], y_check, use = "complete.obs")
  } else {
    NA_real_
  }
})
names(cor_with_target) <- feature_cols_rv

# Flag extreme correlations (likely leakage)
extreme_cor_features <- names(cor_with_target[abs(cor_with_target) > 0.5 & !is.na(cor_with_target)])
if (length(extreme_cor_features) > 0) {
  cat_progress("WARNING: Features with |cor| > 0.5 with target (potential leakage):")
  for (f in extreme_cor_features) {
    cat_progress(sprintf("  %s: r = %.4f", f, cor_with_target[f]))
  }
  remove_extreme <- names(cor_with_target[abs(cor_with_target) > 0.7 & !is.na(cor_with_target)])
  if (length(remove_extreme) > 0) {
    cat_progress(sprintf("Removing %d features with |cor| > 0.7", length(remove_extreme)))
    feature_cols_rv <- setdiff(feature_cols_rv, remove_extreme)
  }
}

cat_progress("Data leakage checks passed.")

#------------------------------------------------------------------
# 3. FEATURE PREPROCESSING
#------------------------------------------------------------------

cat_progress("Preprocessing features...")

# 3.1 Handle NA values
X_train_raw <- as.matrix(train_rv[, ..feature_cols_rv])
na_counts <- colSums(is.na(X_train_raw))
na_pct <- na_counts / nrow(X_train_raw)

high_na_features <- names(na_pct[na_pct > CONFIG$na_threshold_pct])
if (length(high_na_features) > 0) {
  cat_progress(sprintf("Removing %d features with >%.0f%% NA values",
                       length(high_na_features), CONFIG$na_threshold_pct * 100))
  feature_cols_rv <- setdiff(feature_cols_rv, high_na_features)
  X_train_raw <- as.matrix(train_rv[, ..feature_cols_rv])
}

# 3.2 Remove zero-variance features
feature_vars <- apply(X_train_raw, 2, var, na.rm = TRUE)
zero_var_features <- names(feature_vars[feature_vars < CONFIG$eps | is.na(feature_vars)])
if (length(zero_var_features) > 0) {
  cat_progress(sprintf("Removing %d zero-variance features", length(zero_var_features)))
  feature_cols_rv <- setdiff(feature_cols_rv, zero_var_features)
  X_train_raw <- as.matrix(train_rv[, ..feature_cols_rv])
}

# 3.3 Correlation-based filtering
cat_progress("Filtering highly correlated features...")

cor_matrix <- cor(X_train_raw, use = "pairwise.complete.obs")
cor_matrix[is.na(cor_matrix)] <- 0

high_cor_pairs <- which(abs(cor_matrix) > CONFIG$cor_threshold & upper.tri(cor_matrix), arr.ind = TRUE)

if (nrow(high_cor_pairs) > 0) {
  target_cors <- abs(cor_with_target[feature_cols_rv])
  
  features_to_remove <- character(0)
  for (i in seq_len(nrow(high_cor_pairs))) {
    f1_idx <- high_cor_pairs[i, 1]
    f2_idx <- high_cor_pairs[i, 2]
    f1 <- feature_cols_rv[f1_idx]
    f2 <- feature_cols_rv[f2_idx]
    
    if (f1 %in% features_to_remove || f2 %in% features_to_remove) next
    
    cor1 <- ifelse(is.na(target_cors[f1]), 0, target_cors[f1])
    cor2 <- ifelse(is.na(target_cors[f2]), 0, target_cors[f2])
    
    if (cor1 < cor2) {
      features_to_remove <- c(features_to_remove, f1)
    } else {
      features_to_remove <- c(features_to_remove, f2)
    }
  }
  
  features_to_remove <- unique(features_to_remove)
  feature_cols_rv <- setdiff(feature_cols_rv, features_to_remove)
  
  cat_progress(sprintf("Removed %d highly correlated features (r > %.2f)",
                       length(features_to_remove), CONFIG$cor_threshold))
}

cat_progress(sprintf("Features after preprocessing: %d", length(feature_cols_rv)))

# 3.4 Prepare final matrices
X_train_rv <- as.matrix(train_rv[, ..feature_cols_rv])
X_test_rv <- as.matrix(test_rv[, ..feature_cols_rv])
y_train_rv <- train_rv$target_rv_direction
y_test_rv <- test_rv$target_rv_direction

train_dates_rv <- train_rv$date
test_dates_rv <- test_rv$date

# Store VIX for comparison
vix_test <- test_rv$vix_close
rv_cc_test <- test_rv$rv_cc
rv_cc_next_test <- test_rv$rv_cc_next

# 3.5 Class balance analysis
class_balance_rv <- table(y_train_rv)
class_pct_0 <- 100 * class_balance_rv["0"] / sum(class_balance_rv)
class_pct_1 <- 100 * class_balance_rv["1"] / sum(class_balance_rv)

cat_progress(sprintf("Training class balance: RV_Down=%d (%.1f%%), RV_Up=%d (%.1f%%)",
                     class_balance_rv["0"], class_pct_0,
                     class_balance_rv["1"], class_pct_1))

# Compute scale_pos_weight
scale_pos_weight_rv <- as.numeric(class_balance_rv["0"] / class_balance_rv["1"])
cat_progress(sprintf("scale_pos_weight: %.4f", scale_pos_weight_rv))

#------------------------------------------------------------------
# 4. PURGED CROSS-VALIDATION WITH EMBARGO
#------------------------------------------------------------------

cat_progress("Implementing Purged K-Fold with Embargo...")

create_purged_kfold <- function(n_obs, n_folds, purge_days, embargo_days) {
  fold_size <- floor(n_obs / n_folds)
  folds <- vector("list", n_folds)
  
  for (k in seq_len(n_folds)) {
    val_start <- (k - 1L) * fold_size + 1L
    val_end <- min(k * fold_size, n_obs)
    val_idx <- val_start:val_end
    
    purge_start <- max(1L, val_start - purge_days)
    embargo_end <- min(n_obs, val_end + embargo_days)
    
    excluded <- purge_start:embargo_end
    train_idx <- setdiff(seq_len(n_obs), excluded)
    
    folds[[k]] <- list(
      train = train_idx,
      val = val_idx,
      purge_removed = length(setdiff(purge_start:(val_start - 1L), integer(0))),
      embargo_removed = length(setdiff((val_end + 1L):embargo_end, integer(0)))
    )
  }
  
  return(folds)
}

create_cpcv_folds <- function(n_obs, n_groups, n_test_groups, purge_days, embargo_days) {
  group_size <- floor(n_obs / n_groups)
  groups <- lapply(seq_len(n_groups), function(g) {
    start_idx <- (g - 1L) * group_size + 1L
    end_idx <- min(g * group_size, n_obs)
    start_idx:end_idx
  })
  
  test_combos <- combn(seq_len(n_groups), n_test_groups, simplify = FALSE)
  
  folds <- vector("list", length(test_combos))
  for (i in seq_along(test_combos)) {
    test_groups <- test_combos[[i]]
    train_groups <- setdiff(seq_len(n_groups), test_groups)
    
    val_idx <- unlist(groups[test_groups])
    
    train_idx <- integer(0)
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

n_train_rv <- nrow(X_train_rv)

purged_folds_rv <- create_purged_kfold(
  n_train_rv, 
  n_folds = CONFIG$n_folds,
  purge_days = CONFIG$purge_days,
  embargo_days = CONFIG$embargo_days
)

cpcv_folds_rv <- create_cpcv_folds(
  n_train_rv, 
  n_groups = CONFIG$n_cpcv_groups, 
  n_test_groups = CONFIG$n_cpcv_test_groups,
  purge_days = CONFIG$purge_days,
  embargo_days = CONFIG$embargo_days
)

cat_progress(sprintf("Created %d purged k-fold splits (purge=%d, embargo=%d)",
                     length(purged_folds_rv), CONFIG$purge_days, CONFIG$embargo_days))
cat_progress(sprintf("Created %d CPCV splits", length(cpcv_folds_rv)))

# Verify no overlap
for (i in seq_along(purged_folds_rv)) {
  overlap <- intersect(purged_folds_rv[[i]]$train, purged_folds_rv[[i]]$val)
  if (length(overlap) > 0) {
    stop(sprintf("Fold %d has train/val overlap!", i))
  }
}
cat_progress("Verified: No train/validation overlap in any fold")

#------------------------------------------------------------------
# 5. BAYESIAN HYPERPARAMETER OPTIMISATION
#------------------------------------------------------------------

cat_progress("Starting Bayesian Hyperparameter Optimisation...")
cat_progress("Philosophy: Aggressive regularisation for noisy RV direction signal")

# Hyperparameter bounds
bounds_rv <- list(
  max_depth = c(2L, 6L),
  min_child_weight = c(20, 120),
  subsample = c(0.5, 0.8),
  colsample_bytree = c(0.3, 0.7),
  colsample_bynode = c(0.5, 1.0),
  eta = c(0.005, 0.05),
  gamma = c(0, 0.3),
  lambda = c(3, 25),
  alpha = c(0, 0.5),
  max_delta_step = c(0, 2)
)

# Inner folds for hyperparameter tuning
inner_folds_rv <- purged_folds_rv[1:3]

# Scoring function
scoring_function_rv <- function(max_depth, min_child_weight, subsample,
                                colsample_bytree, colsample_bynode, eta,
                                gamma, lambda, alpha, max_delta_step) {
  
  max_depth <- round(max_depth)
  min_child_weight <- round(min_child_weight)
  
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
    scale_pos_weight = scale_pos_weight_rv,
    nthread = 2L
  )
  
  cv_scores <- numeric(length(inner_folds_rv))
  
  for (i in seq_along(inner_folds_rv)) {
    fold <- inner_folds_rv[[i]]
    
    X_tr <- X_train_rv[fold$train, , drop = FALSE]
    y_tr <- y_train_rv[fold$train]
    X_val <- X_train_rv[fold$val, , drop = FALSE]
    y_val <- y_train_rv[fold$val]
    
    dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
    dval <- xgb.DMatrix(data = X_val, label = y_val)
    
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = CONFIG$nrounds_max,
      watchlist = list(val = dval),
      early_stopping_rounds = CONFIG$early_stopping,
      verbose = 2
    )
    
    pred_prob <- predict(model, X_val)
    cv_scores[i] <- -mean(y_val * log(pmax(pred_prob, 1e-15)) +
                            (1 - y_val) * log(pmax(1 - pred_prob, 1e-15)))
  }
  
  # Return negative logloss (we maximise in BO)
  list(Score = -mean(cv_scores), Pred = 0)
}

# Run Bayesian optimisation
cat_progress("Running Bayesian Optimisation (this may take a while)...")

clusterExport(cl, c(
  "X_train_rv",
  "y_train_rv",
  "inner_folds_rv",
  "scale_pos_weight_rv",
  "CONFIG",
  "scoring_function_rv"
))
clusterEvalQ(cl, {
  library(xgboost)
  library(pROC)
})


bayes_opt_result_rv <- ParBayesianOptimization::bayesOpt(
  FUN = scoring_function_rv,
  bounds = bounds_rv,
  initPoints = CONFIG$bo_init_points,
  iters.n = CONFIG$bo_iterations,
  iters.k = CONFIG$bo_parallel_k,
  gsPoints = CONFIG$bo_gs_points,
  parallel = TRUE,
  verbose = 2
)

best_params_rv <- ParBayesianOptimization::getBestPars(bayes_opt_result_rv)
cat_progress(sprintf(
  "Best CV logloss: %.4f",
  -max(bayes_opt_result_rv$scoreSummary$Score)
))


# Build final parameter list
final_params_rv <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = round(best_params_rv$max_depth),
  min_child_weight = round(best_params_rv$min_child_weight),
  subsample = best_params_rv$subsample,
  colsample_bytree = best_params_rv$colsample_bytree,
  colsample_bynode = best_params_rv$colsample_bynode,
  eta = best_params_rv$eta,
  gamma = best_params_rv$gamma,
  lambda = best_params_rv$lambda,
  alpha = best_params_rv$alpha,
  max_delta_step = best_params_rv$max_delta_step,
  scale_pos_weight = scale_pos_weight_rv,
  nthread = parallel::detectCores(logical = FALSE)
)

cat_progress("Optimised hyperparameters:")
for (p in names(best_params_rv)) {
  cat_progress(sprintf("  %s: %.4f", p, best_params_rv[[p]]))
}

#------------------------------------------------------------------
# 6. FULL CROSS-VALIDATION WITH OPTIMAL PARAMETERS
#------------------------------------------------------------------

cat_progress("Running full cross-validation with optimal parameters...")

cv_results_rv <- vector("list", length(cpcv_folds_rv))
cv_predictions_rv <- data.table()

for (fold_idx in seq_along(cpcv_folds_rv)) {
  fold <- cpcv_folds_rv[[fold_idx]]
  
  X_tr <- X_train_rv[fold$train, , drop = FALSE]
  y_tr <- y_train_rv[fold$train]
  X_val <- X_train_rv[fold$val, , drop = FALSE]
  y_val <- y_train_rv[fold$val]
  
  dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
  dval <- xgb.DMatrix(data = X_val, label = y_val)
  
  model <- xgb.train(
    params = final_params_rv,
    data = dtrain,
    nrounds = CONFIG$nrounds_max,
    watchlist = list(train = dtrain, val = dval),
    early_stopping_rounds = CONFIG$early_stopping,
    verbose = 0
  )
  
  pred_prob <- predict(model, X_val)
  pred_class <- as.integer(pred_prob >= 0.5)
  
  # Metrics
  tp <- sum(pred_class == 1 & y_val == 1)
  tn <- sum(pred_class == 0 & y_val == 0)
  fp <- sum(pred_class == 1 & y_val == 0)
  fn <- sum(pred_class == 0 & y_val == 1)
  
  accuracy <- (tp + tn) / length(y_val)
  
  roc_obj <- tryCatch({
    pROC::roc(y_val, pred_prob, quiet = TRUE)
  }, error = function(e) NULL)
  
  auc <- ifelse(!is.null(roc_obj), as.numeric(pROC::auc(roc_obj)), NA)
  
  logloss <- -mean(y_val * log(pmax(pred_prob, 1e-15)) +
                     (1 - y_val) * log(pmax(1 - pred_prob, 1e-15)))
  
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  cv_results_rv[[fold_idx]] <- list(
    fold = fold_idx,
    n_train = length(y_tr),
    n_val = length(y_val),
    best_iteration = model$best_iteration,
    accuracy = accuracy,
    auc = auc,
    logloss = logloss,
    precision = precision,
    recall = recall,
    specificity = specificity,
    f1 = f1
  )
  
  # Store predictions
  fold_preds <- data.table(
    fold = fold_idx,
    idx = fold$val,
    date = train_dates_rv[fold$val],
    actual = y_val,
    pred_prob = pred_prob,
    pred_class = pred_class
  )
  cv_predictions_rv <- rbind(cv_predictions_rv, fold_preds)
  
  cat_progress(sprintf("  Fold %d/%d: AUC=%.4f, Acc=%.4f, LogLoss=%.4f",
                       fold_idx, length(cpcv_folds_rv), auc, accuracy, logloss))
}

# Summarise CV results
cv_metrics_rv <- rbindlist(lapply(cv_results_rv, as.data.table))

cv_summary_rv <- data.frame(
  Metric = c("AUC", "Accuracy", "LogLoss", "Precision", "Recall", "Specificity", "F1"),
  Mean = c(mean(cv_metrics_rv$auc, na.rm = TRUE),
           mean(cv_metrics_rv$accuracy),
           mean(cv_metrics_rv$logloss),
           mean(cv_metrics_rv$precision),
           mean(cv_metrics_rv$recall),
           mean(cv_metrics_rv$specificity),
           mean(cv_metrics_rv$f1)),
  SD = c(sd(cv_metrics_rv$auc, na.rm = TRUE),
         sd(cv_metrics_rv$accuracy),
         sd(cv_metrics_rv$logloss),
         sd(cv_metrics_rv$precision),
         sd(cv_metrics_rv$recall),
         sd(cv_metrics_rv$specificity),
         sd(cv_metrics_rv$f1))
)

cat_progress("\n========== CV SUMMARY ==========")
print(cv_summary_rv, digits = 4)

#------------------------------------------------------------------
# 7. TRAIN FINAL MODEL ON FULL TRAINING SET
#------------------------------------------------------------------

cat_progress("Training final model on full training set...")

dtrain_full <- xgb.DMatrix(data = X_train_rv, label = y_train_rv)
dtest <- xgb.DMatrix(data = X_test_rv, label = y_test_rv)

final_model_rv <- xgb.train(
  params = final_params_rv,
  data = dtrain_full,
  nrounds = CONFIG$nrounds_max,
  watchlist = list(train = dtrain_full, test = dtest),
  early_stopping_rounds = CONFIG$early_stopping,
  verbose = 1
)

cat_progress(sprintf("Final model trained: %d rounds (best: %d)",
                     final_model_rv$niter, final_model_rv$best_iteration))

# Save model
xgb.save(final_model_rv, "results/models/rv_forward/xgb_rv_classification_model.xgb")

#------------------------------------------------------------------
# 8. TEST SET EVALUATION
#------------------------------------------------------------------

cat_progress("Evaluating on test set...")

test_pred_prob_rv <- predict(final_model_rv, X_test_rv)
test_pred_class_rv <- as.integer(test_pred_prob_rv >= 0.5)

# ROC analysis
roc_obj_rv <- pROC::roc(y_test_rv, test_pred_prob_rv, quiet = TRUE)
test_auc_rv <- as.numeric(pROC::auc(roc_obj_rv))

# Confusion matrix components
tp <- sum(test_pred_class_rv == 1 & y_test_rv == 1)
tn <- sum(test_pred_class_rv == 0 & y_test_rv == 0)
fp <- sum(test_pred_class_rv == 1 & y_test_rv == 0)
fn <- sum(test_pred_class_rv == 0 & y_test_rv == 1)

test_accuracy_rv <- (tp + tn) / length(y_test_rv)
test_precision_rv <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
test_recall_rv <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
test_specificity_rv <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
test_f1_rv <- ifelse((test_precision_rv + test_recall_rv) > 0,
                     2 * test_precision_rv * test_recall_rv / 
                       (test_precision_rv + test_recall_rv), 0)

test_log_loss_rv <- -mean(y_test_rv * log(pmax(test_pred_prob_rv, 1e-15)) +
                            (1 - y_test_rv) * log(pmax(1 - test_pred_prob_rv, 1e-15)))
test_brier_rv <- mean((test_pred_prob_rv - y_test_rv)^2)

# MCC
mcc_num <- (tp * tn) - (fp * fn)
mcc_den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
test_mcc_rv <- ifelse(mcc_den > 0, mcc_num / mcc_den, 0)

# Cohen's Kappa
po <- test_accuracy_rv
pe <- ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (length(y_test_rv)^2)
test_kappa_rv <- ifelse(abs(1 - pe) > CONFIG$eps, (po - pe) / (1 - pe), 0)

# PR-AUC
pr_obj <- PRROC::pr.curve(scores.class0 = test_pred_prob_rv[y_test_rv == 1],
                          scores.class1 = test_pred_prob_rv[y_test_rv == 0],
                          curve = TRUE)
test_pr_auc_rv <- pr_obj$auc.integral

cat_progress("\n========== TEST SET RESULTS ==========")
test_metrics_rv <- data.frame(
  Metric = c("AUC-ROC", "AUC-PR", "Accuracy", "Precision", "Recall", "Specificity",
             "F1 Score", "Log Loss", "Brier Score", "MCC", "Cohen's Kappa"),
  Value = c(test_auc_rv, test_pr_auc_rv, test_accuracy_rv, test_precision_rv, 
            test_recall_rv, test_specificity_rv, test_f1_rv, test_log_loss_rv, 
            test_brier_rv, test_mcc_rv, test_kappa_rv)
)
print(test_metrics_rv, digits = 4)

# Confusion matrix
cat_progress("\nConfusion Matrix:")
conf_matrix_rv <- matrix(c(tn, fp, fn, tp), nrow = 2,
                         dimnames = list(Predicted = c("RV_Down", "RV_Up"),
                                         Actual = c("RV_Down", "RV_Up")))
print(conf_matrix_rv)

# Baseline comparison
baseline_accuracy <- max(mean(y_test_rv), 1 - mean(y_test_rv))
cat_progress(sprintf("\nBaseline (majority class): %.4f", baseline_accuracy))
cat_progress(sprintf("Improvement over baseline: %.4f (%.1f%%)", 
                     test_accuracy_rv - baseline_accuracy,
                     100 * (test_accuracy_rv - baseline_accuracy) / baseline_accuracy))

#------------------------------------------------------------------
# 9. VIX COMPARISON (KEY ANALYSIS)
#------------------------------------------------------------------

cat_progress("Comparing model predictions with VIX-implied direction...")

# VIX-implied RV direction: VIX suggests RV will be at that level
# If VIX_{t} > RV_{t,t+22}, VIX implies RV will increase
# This is a heuristic: compare lagged VIX to lagged RV

# Simple VIX benchmark: predict RV up if VIX > recent RV
# Use test_rv which has vix_close and rv_cc
vix_implied_up <- as.integer(test_rv$vix_close > test_rv$rv_cc)

# Compare VIX-implied vs actual
vix_accuracy <- mean(vix_implied_up == y_test_rv, na.rm = TRUE)
vix_tp <- sum(vix_implied_up == 1 & y_test_rv == 1, na.rm = TRUE)
vix_tn <- sum(vix_implied_up == 0 & y_test_rv == 0, na.rm = TRUE)
vix_fp <- sum(vix_implied_up == 1 & y_test_rv == 0, na.rm = TRUE)
vix_fn <- sum(vix_implied_up == 0 & y_test_rv == 1, na.rm = TRUE)

vix_precision <- ifelse((vix_tp + vix_fp) > 0, vix_tp / (vix_tp + vix_fp), 0)
vix_recall <- ifelse((vix_tp + vix_fn) > 0, vix_tp / (vix_tp + vix_fn), 0)

cat_progress("\n========== VIX BENCHMARK COMPARISON ==========")
cat_progress(sprintf("VIX-implied accuracy: %.4f", vix_accuracy))
cat_progress(sprintf("XGBoost accuracy: %.4f", test_accuracy_rv))
cat_progress(sprintf("Improvement over VIX: %.4f (%.1f%%)", 
                     test_accuracy_rv - vix_accuracy,
                     100 * (test_accuracy_rv - vix_accuracy) / (vix_accuracy + CONFIG$eps)))

# Statistical test: McNemar's test for difference
contingency_vix_model <- matrix(c(
  sum(test_pred_class_rv == y_test_rv & vix_implied_up == y_test_rv),
  sum(test_pred_class_rv == y_test_rv & vix_implied_up != y_test_rv),
  sum(test_pred_class_rv != y_test_rv & vix_implied_up == y_test_rv),
  sum(test_pred_class_rv != y_test_rv & vix_implied_up != y_test_rv)
), nrow = 2)

mcnemar_vix <- tryCatch({
  mcnemar.test(contingency_vix_model)
}, error = function(e) list(p.value = NA))

cat_progress(sprintf("McNemar's test (Model vs VIX): p = %.4f", mcnemar_vix$p.value))

#------------------------------------------------------------------
# 10. THRESHOLD OPTIMISATION
#------------------------------------------------------------------

cat_progress("Optimising classification threshold...")

thresholds <- seq(0.30, 0.70, by = 0.01)
threshold_results_rv <- data.frame()

for (thresh in thresholds) {
  pred_class_t <- as.integer(test_pred_prob_rv >= thresh)
  
  tp_t <- sum(pred_class_t == 1 & y_test_rv == 1)
  tn_t <- sum(pred_class_t == 0 & y_test_rv == 0)
  fp_t <- sum(pred_class_t == 1 & y_test_rv == 0)
  fn_t <- sum(pred_class_t == 0 & y_test_rv == 1)
  
  sens_t <- ifelse((tp_t + fn_t) > 0, tp_t / (tp_t + fn_t), 0)
  spec_t <- ifelse((tn_t + fp_t) > 0, tn_t / (tn_t + fp_t), 0)
  youden_j <- sens_t + spec_t - 1
  
  prec_t <- ifelse((tp_t + fp_t) > 0, tp_t / (tp_t + fp_t), 0)
  f1_t <- ifelse((prec_t + sens_t) > 0, 2 * prec_t * sens_t / (prec_t + sens_t), 0)
  
  acc_t <- (tp_t + tn_t) / length(y_test_rv)
  
  mcc_num_t <- (tp_t * tn_t) - (fp_t * fn_t)
  mcc_den_t <- sqrt((tp_t + fp_t) * (tp_t + fn_t) * (tn_t + fp_t) * (tn_t + fn_t))
  mcc_t <- ifelse(mcc_den_t > 0, mcc_num_t / mcc_den_t, 0)
  
  threshold_results_rv <- rbind(threshold_results_rv, data.frame(
    threshold = thresh,
    sensitivity = sens_t,
    specificity = spec_t,
    youden_j = youden_j,
    precision = prec_t,
    f1 = f1_t,
    accuracy = acc_t,
    mcc = mcc_t
  ))
}

optimal_youden_rv <- threshold_results_rv$threshold[which.max(threshold_results_rv$youden_j)]
optimal_f1_rv <- threshold_results_rv$threshold[which.max(threshold_results_rv$f1)]
optimal_mcc_rv <- threshold_results_rv$threshold[which.max(threshold_results_rv$mcc)]

cat_progress(sprintf("Optimal threshold (Youden's J): %.2f (J=%.4f)", 
                     optimal_youden_rv, max(threshold_results_rv$youden_j)))
cat_progress(sprintf("Optimal threshold (F1): %.2f (F1=%.4f)", 
                     optimal_f1_rv, max(threshold_results_rv$f1)))
cat_progress(sprintf("Optimal threshold (MCC): %.2f (MCC=%.4f)", 
                     optimal_mcc_rv, max(threshold_results_rv$mcc)))

#------------------------------------------------------------------
# 11. CALIBRATION ANALYSIS
#------------------------------------------------------------------

cat_progress("Performing calibration analysis...")

n_bins <- 10
bin_edges <- seq(0, 1, length.out = n_bins + 1)
calibration_df_rv <- data.frame()

for (i in seq_len(n_bins)) {
  bin_mask <- test_pred_prob_rv >= bin_edges[i] & test_pred_prob_rv < bin_edges[i + 1]
  if (sum(bin_mask) > 0) {
    mean_pred <- mean(test_pred_prob_rv[bin_mask])
    mean_actual <- mean(y_test_rv[bin_mask])
    n_obs <- sum(bin_mask)
    se_actual <- sqrt(mean_actual * (1 - mean_actual) / n_obs)
    
    calibration_df_rv <- rbind(calibration_df_rv, data.frame(
      bin = i,
      bin_lower = bin_edges[i],
      bin_upper = bin_edges[i + 1],
      mean_predicted = mean_pred,
      mean_actual = mean_actual,
      se_actual = se_actual,
      n = n_obs
    ))
  }
}

ece_rv <- sum(calibration_df_rv$n * abs(calibration_df_rv$mean_predicted - 
                                          calibration_df_rv$mean_actual)) / sum(calibration_df_rv$n)
mce_rv <- max(abs(calibration_df_rv$mean_predicted - calibration_df_rv$mean_actual))

if (nrow(calibration_df_rv) >= 3) {
  cal_fit <- lm(mean_actual ~ mean_predicted, data = calibration_df_rv, weights = n)
  cal_slope <- coef(cal_fit)[2]
  cal_intercept <- coef(cal_fit)[1]
} else {
  cal_slope <- NA
  cal_intercept <- NA
}

cat_progress(sprintf("Expected Calibration Error (ECE): %.4f", ece_rv))
cat_progress(sprintf("Maximum Calibration Error (MCE): %.4f", mce_rv))
cat_progress(sprintf("Calibration slope: %.4f (ideal: 1.0)", cal_slope))

#------------------------------------------------------------------
# 12. FEATURE IMPORTANCE AND SHAP ANALYSIS
#------------------------------------------------------------------

cat_progress("Calculating feature importance and SHAP values...")

# XGBoost native importance
importance_gain_rv <- xgb.importance(model = final_model_rv, feature_names = feature_cols_rv)
importance_gain_rv <- importance_gain_rv[order(-Gain)]

cat_progress("\nTop 20 Features by Gain:")
print(head(importance_gain_rv, 20))

# SHAP values
cat_progress("Computing SHAP values...")

shap_sample_size <- min(CONFIG$shap_sample_size, nrow(X_test_rv))
set.seed(CONFIG$seed)
shap_idx_rv <- sample(seq_len(nrow(X_test_rv)), shap_sample_size)

shap_values_rv <- predict(final_model_rv, X_test_rv[shap_idx_rv, ], predcontrib = TRUE)
shap_matrix_rv <- shap_values_rv[, -ncol(shap_values_rv)]
colnames(shap_matrix_rv) <- feature_cols_rv

shap_importance_rv <- data.frame(
  Feature = feature_cols_rv,
  Mean_Abs_SHAP = colMeans(abs(shap_matrix_rv))
)
shap_importance_rv <- shap_importance_rv[order(-shap_importance_rv$Mean_Abs_SHAP), ]

cat_progress("\nTop 20 Features by Mean |SHAP|:")
print(head(shap_importance_rv, 20))

# Feature category importance
total_gain <- sum(importance_gain_rv$Gain)

rv_lagged_features <- feature_cols_rv[grepl("^rv_", feature_cols_rv)]
ret_features <- feature_cols_rv[grepl("^ret_", feature_cols_rv)]
range_features <- feature_cols_rv[grepl("parkinson|gk_vol|range", feature_cols_rv)]

rv_lagged_gain <- sum(importance_gain_rv[Feature %in% rv_lagged_features]$Gain)
ret_gain <- sum(importance_gain_rv[Feature %in% ret_features]$Gain)
range_gain <- sum(importance_gain_rv[Feature %in% range_features]$Gain)

cat_progress(sprintf("\nRV lagged features: %.1f%% of total gain", 100 * rv_lagged_gain / total_gain))
cat_progress(sprintf("Return features: %.1f%% of total gain", 100 * ret_gain / total_gain))
cat_progress(sprintf("Range features: %.1f%% of total gain", 100 * range_gain / total_gain))

#------------------------------------------------------------------
# 13. STATISTICAL SIGNIFICANCE TESTS
#------------------------------------------------------------------

cat_progress("Performing statistical significance tests...")

# Binomial test vs random (50%)
binom_test_rv <- binom.test(sum(test_pred_class_rv == y_test_rv), 
                            length(y_test_rv), p = 0.5)

cat_progress(sprintf("\nBinomial test vs random (50%%):"))
cat_progress(sprintf("  Accuracy: %.4f", test_accuracy_rv))
cat_progress(sprintf("  p-value: %.2e", binom_test_rv$p.value))
cat_progress(sprintf("  95%% CI: [%.4f, %.4f]", binom_test_rv$conf.int[1], binom_test_rv$conf.int[2]))

# Binomial test vs baseline
baseline_pct <- max(mean(y_test_rv), 1 - mean(y_test_rv))
binom_test_baseline <- binom.test(sum(test_pred_class_rv == y_test_rv), 
                                  length(y_test_rv), p = baseline_pct,
                                  alternative = "greater")

cat_progress(sprintf("\nBinomial test vs baseline (%.1f%%):", 100 * baseline_pct))
cat_progress(sprintf("  p-value: %.4f", binom_test_baseline$p.value))

# Bootstrap CI for AUC
boot_auc <- replicate(CONFIG$n_bootstrap, {
  idx <- sample(seq_len(length(y_test_rv)), replace = TRUE)
  tryCatch({
    as.numeric(pROC::auc(pROC::roc(y_test_rv[idx], test_pred_prob_rv[idx], quiet = TRUE)))
  }, error = function(e) NA)
})
boot_auc <- boot_auc[!is.na(boot_auc)]
auc_ci_rv <- quantile(boot_auc, c(0.025, 0.975))
auc_se_rv <- sd(boot_auc)

cat_progress(sprintf("\nAUC Bootstrap 95%% CI: [%.4f, %.4f]", auc_ci_rv[1], auc_ci_rv[2]))
cat_progress(sprintf("AUC SE: %.4f", auc_se_rv))

# DeLong test
delong_test <- tryCatch({
  pROC::roc.test(roc_obj_rv, pROC::roc(y_test_rv, rep(0.5, length(y_test_rv)), quiet = TRUE))
}, error = function(e) list(p.value = NA))

cat_progress(sprintf("DeLong test (AUC > 0.5): p = %.4f", delong_test$p.value))

#------------------------------------------------------------------
# 14. ROLLING PERFORMANCE ANALYSIS
#------------------------------------------------------------------

cat_progress("Analysing rolling performance stability...")

test_predictions_df <- data.table(
  date = test_dates_rv,
  actual = y_test_rv,
  pred_prob = test_pred_prob_rv,
  pred_class = test_pred_class_rv,
  correct = as.integer(test_pred_class_rv == y_test_rv)
)

test_predictions_df[, rolling_accuracy := frollmean(correct, n = CONFIG$rolling_window, 
                                                    align = "right")]
test_predictions_df[, rolling_auc := rollapply(
  cbind(actual, pred_prob),
  width = CONFIG$rolling_window,
  FUN = function(x) {
    tryCatch(
      as.numeric(pROC::auc(pROC::roc(x[,1], x[,2], quiet = TRUE))),
      error = function(e) NA_real_
    )
  },
  by.column = FALSE,
  align = "right",
  fill = NA
)]


rolling_metrics_rv <- test_predictions_df[!is.na(rolling_accuracy)]

mean_rolling_acc_rv <- mean(rolling_metrics_rv$rolling_accuracy, na.rm = TRUE)
sd_rolling_acc_rv <- sd(rolling_metrics_rv$rolling_accuracy, na.rm = TRUE)
mean_rolling_auc_rv <- mean(rolling_metrics_rv$rolling_auc, na.rm = TRUE)

cat_progress(sprintf("Rolling accuracy (window=%d): mean=%.4f, sd=%.4f",
                     CONFIG$rolling_window, mean_rolling_acc_rv, sd_rolling_acc_rv))
cat_progress(sprintf("Rolling AUC: mean=%.4f", mean_rolling_auc_rv))

# Detect degradation periods
degradation_threshold_rv <- mean_rolling_acc_rv - 2 * sd_rolling_acc_rv
degradation_periods_rv <- which(rolling_metrics_rv$rolling_accuracy < degradation_threshold_rv)
cat_progress(sprintf("Degradation periods (acc < %.4f): %d observations",
                     degradation_threshold_rv, length(degradation_periods_rv)))

#------------------------------------------------------------------
# 15. VISUALISATIONS
#------------------------------------------------------------------

cat_progress("Creating visualisations...")

# 15.1 ROC Curve
pdf("results/figures/rv_forward/1_roc_curve.pdf", width = 8, height = 6)
plot(roc_obj_rv, main = sprintf("ROC Curve: RV Direction (AUC = %.4f)", test_auc_rv),
     col = "darkblue", lwd = 2)
abline(a = 0, b = 1, col = "grey", lty = 2)
dev.off()

# 15.2 Calibration Plot
pdf("results/figures/rv_forward/02_calibration.pdf", width = 8, height = 6)
plot(calibration_df_rv$mean_predicted, calibration_df_rv$mean_actual,
     pch = 16, cex = sqrt(calibration_df_rv$n / 50),
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted Probability", ylab = "Observed Frequency",
     main = sprintf("Calibration Plot (ECE = %.4f)", ece_rv))
abline(0, 1, col = "red", lwd = 2)
if (!is.na(cal_slope)) {
  abline(cal_intercept, cal_slope, col = "blue", lty = 2)
}
legend("topleft", c("Perfect", sprintf("Fitted (slope=%.2f)", cal_slope)),
       col = c("red", "blue"), lwd = 2, lty = c(1, 2), bty = "n")
dev.off()

# 15.3 Feature Importance
pdf("results/figures/rv_forward/03_feature_importance.pdf", width = 10, height = 8)
par(mar = c(5, 12, 4, 2))
top_20 <- head(importance_gain_rv, 20)
barplot(rev(top_20$Gain), names.arg = rev(top_20$Feature),
        horiz = TRUE, las = 1, col = "steelblue",
        main = "Top 20 Features by Gain", xlab = "Gain")
dev.off()

# 15.4 Rolling Performance
pdf("results/figures/rv_forward/04_rolling_performance.pdf", width = 12, height = 6)
par(mfrow = c(1, 2))
plot(rolling_metrics_rv$date, rolling_metrics_rv$rolling_accuracy, type = "l",
     main = sprintf("Rolling Accuracy (%d-day)", CONFIG$rolling_window),
     xlab = "Date", ylab = "Accuracy", col = "darkblue")
abline(h = mean_rolling_acc_rv, col = "red", lty = 2)
abline(h = baseline_accuracy, col = "grey", lty = 2)
abline(h = degradation_threshold_rv, col = "orange", lty = 2)

plot(rolling_metrics_rv$date, rolling_metrics_rv$rolling_auc, type = "l",
     main = sprintf("Rolling AUC (%d-day)", CONFIG$rolling_window),
     xlab = "Date", ylab = "AUC", col = "darkgreen")
abline(h = 0.5, col = "grey", lty = 2)
abline(h = mean_rolling_auc_rv, col = "red", lty = 2)
dev.off()

# 15.5 Model vs VIX Comparison
pdf("results/figures/rv_forward/05_model_vs_vix.pdf", width = 10, height = 6)
comparison_df <- data.frame(
  Method = c("XGBoost", "VIX Benchmark", "Baseline"),
  Accuracy = c(test_accuracy_rv, vix_accuracy, baseline_accuracy)
)
barplot(comparison_df$Accuracy, names.arg = comparison_df$Method,
        col = c("steelblue", "coral", "grey"),
        main = "Model vs VIX Benchmark",
        ylab = "Accuracy", ylim = c(0, 1))
abline(h = 0.5, lty = 2, col = "black")
dev.off()

# 15.6 Threshold Analysis
pdf("results/figures/rv_forward/06_threshold_analysis.pdf", width = 10, height = 6)
par(mfrow = c(1, 2))
plot(threshold_results_rv$threshold, threshold_results_rv$youden_j, type = "l",
     main = "Youden's J vs Threshold", xlab = "Threshold", ylab = "Youden's J",
     col = "darkblue", lwd = 2)
abline(v = optimal_youden_rv, col = "red", lty = 2)

plot(threshold_results_rv$threshold, threshold_results_rv$f1, type = "l",
     main = "F1 Score vs Threshold", xlab = "Threshold", ylab = "F1",
     col = "darkgreen", lwd = 2)
abline(v = optimal_f1_rv, col = "red", lty = 2)
dev.off()

#------------------------------------------------------------------
# 16. SAVE ALL RESULTS
#------------------------------------------------------------------

cat_progress("Saving all results...")

# Model
xgb.save(final_model_rv, "results/models/rv_forward/xgb_rv_classification_model.xgb")

# Predictions
test_predictions_rv <- data.table(
  date = test_dates_rv,
  actual = y_test_rv,
  rv_cc = rv_cc_test,
  rv_cc_next = rv_cc_next_test,
  vix_close = vix_test,
  pred_prob = test_pred_prob_rv,
  pred_class = test_pred_class_rv,
  pred_class_youden = as.integer(test_pred_prob_rv >= optimal_youden_rv),
  pred_class_f1 = as.integer(test_pred_prob_rv >= optimal_f1_rv),
  vix_implied = vix_implied_up
)
saveRDS(test_predictions_rv, "results/models/rv_forward/xgb_rv_classification_predictions.rds")

# Simple predictions format
predictions_simple <- data.table(
  date = test_dates_rv,
  actual = y_test_rv,
  pred_prob = test_pred_prob_rv,
  pred_class = test_pred_class_rv,
  pred_class_optimal = as.integer(test_pred_prob_rv >= optimal_youden_rv)
)
write.csv(predictions_simple, "results/models/rv_forward/xgb_rv_predictions_simple.csv", row.names = FALSE)

# CV Results
saveRDS(cv_results_rv, "results/models/rv_forward/xgb_rv_classification_cv_results.rds")
saveRDS(cv_metrics_rv, "results/models/rv_forward/xgb_rv_classification_cv_metrics.rds")
saveRDS(cv_predictions_rv, "results/models/rv_forward/xgb_rv_classification_cv_predictions.rds")

# Feature Importance
saveRDS(importance_gain_rv, "results/models/rv_forward/xgb_rv_classification_importance.rds")
saveRDS(shap_importance_rv, "results/models/rv_forward/xgb_rv_classification_shap_importance.rds")
saveRDS(shap_matrix_rv, "results/models/rv_forward/xgb_rv_classification_shap_matrix.rds")

# Tables
write.csv(test_metrics_rv, "results/tables/rv_forward/xgb_rv_test_metrics.csv", row.names = FALSE)
write.csv(cv_summary_rv, "results/tables/rv_forward/xgb_rv_cv_summary.csv", row.names = FALSE)
write.csv(threshold_results_rv, "results/tables/rv_forward/xgb_rv_thresholds.csv", row.names = FALSE)
write.csv(calibration_df_rv, "results/tables/rv_forward/xgb_rv_calibration.csv", row.names = FALSE)
write.csv(as.data.frame(importance_gain_rv), "results/tables/rv_forward/xgb_rv_feature_importance.csv", row.names = FALSE)
write.csv(rolling_metrics_rv, "results/tables/rv_forward/xgb_rv_rolling_metrics.csv", row.names = FALSE)

# Bayesian Optimisation
if (!is.null(bayes_opt_result_rv)) {
  saveRDS(bayes_opt_result_rv, "results/models/rv_forward/xgb_rv_bayes_opt.rds")
}

# Comprehensive Results Object
xgb_rv_results <- list(
  # Model info
  model_path = "results/models/rv_forward/xgb_rv_classification_model.xgb",
  params = final_params_rv,
  best_params_raw = best_params_rv,
  config = CONFIG,
  
  # Architecture
  architecture = list(
    type = "Direct RV classification",
    target = "I{RV_{t+1,t+23} > RV_{t,t+22}}",
    rv_estimator = "close-to-close",
    features = "Historical RV (lagged) + SPX returns + Range estimators"
  ),
  
  # Data
  data_summary = list(
    n_train = nrow(X_train_rv),
    n_test = nrow(X_test_rv),
    n_features = length(feature_cols_rv),
    class_balance_train = as.list(class_balance_rv),
    scale_pos_weight = scale_pos_weight_rv
  ),
  
  # CV
  cv = list(
    n_folds = length(cpcv_folds_rv),
    purge_days = CONFIG$purge_days,
    embargo_days = CONFIG$embargo_days,
    summary = cv_summary_rv,
    metrics = cv_metrics_rv
  ),
  
  # Test performance
  test = list(
    metrics = test_metrics_rv,
    predictions = test_predictions_rv,
    confusion_matrix = conf_matrix_rv
  ),
  
  # VIX comparison
  vix_comparison = list(
    xgb_accuracy = test_accuracy_rv,
    vix_accuracy = vix_accuracy,
    improvement = test_accuracy_rv - vix_accuracy,
    mcnemar_p = mcnemar_vix$p.value
  ),
  
  # Threshold optimisation
  thresholds = list(
    optimal_youden = optimal_youden_rv,
    optimal_f1 = optimal_f1_rv,
    optimal_mcc = optimal_mcc_rv,
    results = threshold_results_rv
  ),
  
  # Calibration
  calibration = list(
    ece = ece_rv,
    mce = mce_rv,
    slope = cal_slope,
    intercept = cal_intercept,
    data = calibration_df_rv
  ),
  
  # Statistical tests
  statistical_tests = list(
    binom_vs_random = list(p_value = binom_test_rv$p.value, ci = binom_test_rv$conf.int),
    binom_vs_baseline = list(p_value = binom_test_baseline$p.value),
    auc_bootstrap_ci = auc_ci_rv,
    auc_se = auc_se_rv
  ),
  
  # Feature importance
  feature_importance = list(
    gain = importance_gain_rv,
    shap = shap_importance_rv,
    rv_lagged_pct = 100 * rv_lagged_gain / total_gain,
    ret_pct = 100 * ret_gain / total_gain,
    range_pct = 100 * range_gain / total_gain
  ),
  
  # Temporal stability
  temporal = list(
    rolling_metrics = rolling_metrics_rv,
    mean_rolling_acc = mean_rolling_acc_rv,
    sd_rolling_acc = sd_rolling_acc_rv,
    mean_rolling_auc = mean_rolling_auc_rv,
    degradation_threshold = degradation_threshold_rv,
    n_degradation_periods = length(degradation_periods_rv)
  ),
  
  # Feature list
  feature_cols = feature_cols_rv
)

saveRDS(xgb_rv_results, "results/models/rv_forward/xgb_rv_classification_full_results.rds")


#------------------------------------------------------------------
# END OF SCRIPT
#------------------------------------------------------------------