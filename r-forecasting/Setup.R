################################################################################
# 00_setup.R
################################################################################

set.seed(42)


options(repos = c(CRAN = "https://cran.rstudio.com/"))

#Setup Directory
project_dirs <- c(
  "data",
  "scripts", 
  "results",
  "results/figures",
  "results/tables",
  "results/models",
  "docs"
)

for (dir in project_dirs) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
    cat("Created directory:", dir, "\n")
  }
}
setwd("C:\\Users\\Administrator\\Desktop\\080326")
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing package:", pkg, "\n")
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    } else {
      library(pkg, character.only = TRUE)
    }
  }
}

core_packages <- c(
  
  # Data manipulation
  "data.table",
  "dplyr",
  "tidyr",
  "lubridate",
  
  # Data download
  "quantmod",
  "zoo",
  
  # Modelling - XGBoost
  "xgboost",
  "lightgbm",
  
  # Bayesian optimisation
  "remotes",
  "ParBayesianOptimization",
  
  # SHAP and interpretability
  "SHAPforxgboost",
  "shapviz",
  "shapr",
  
  # Econometrics and HAC inference
  "sandwich",
  "lmtest",
  "car",
  
  # Model Confidence Set
  "MCS",
  
  # Quantile regression for CQR
  "quantreg",
  
  # Parallel processing
  "parallel",
  "doParallel",
  "foreach",
  
  # Plotting
  "ggplot2",
  "gridExtra",
  "scales",
  "viridis",
  
  # Statistical tests
  "tseries",
  "FinTS",
  "moments",
  "pracma",
  "forecast",
  
  # Misc
  "Matrix",
  "glmnet",
  "mgcv",
  "writexl",
  "caret",
  "pROC",
  "PRROC",
  "remotes",
  "calibrate",
  "Metrics",
  "roll",
  "reshape2",
  "viridis",
  "scales",
  "knitr",
  "xtable"
)

install_and_load(core_packages)
#remotes::install_github("AnotherSamWilson/ParBayesianOptimization", 
                        #type = "binary", 
                       # dependencies = TRUE)
#Records versions for clashes  
pkg_versions <- data.frame(
  Package = core_packages,
  Version = sapply(core_packages, function(x) {
    as.character(packageVersion(x))
  })
)

print(pkg_versions)

write.csv(pkg_versions, "docs/package_versions.csv", row.names = FALSE)

# Set G Options for across pages
options(scipen = 999)
options(digits = 4)    

#------------------------------------------------------------------
#HELPER FUNCTIONS
#------------------------------------------------------------------

# Configuration list for tickers and date range
config <- list(
  tickers = list(
    spx = "^GSPC",
    vix = "^VIX"
  ),
  start_date = as.Date("1990-01-01"),
  end_date = Sys.Date()
)

# Directory paths list
dirs <- list(
  data = "data",
  figures = "results/figures",
  tables = "results/tables",
  models = "results/models"
)

# Progress logging function
cat_progress <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg))
}

#------------------------------------------------------------------

#Parallel processing to speed it up

n_cores <- (parallel::detectCores(logical = TRUE) - 1)
n_cores
cl <- makeCluster(n_cores)
registerDoParallel(cl)

getDoParWorkers()


saveRDS(cl, "results/models/parallel_cluster.rds")

cat("\nRun Data_Download.R\n")