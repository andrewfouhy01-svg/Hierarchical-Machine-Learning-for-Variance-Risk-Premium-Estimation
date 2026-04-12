Hierarchical Machine Learning for Variance Risk Premium Estimation
Hierarchical XGBoost pipeline forecasting VIX and realised volatility to estimate variance risk premium and systematically trade SPX bull put spreads.
Key Results
MetricValueTotal Return53.0%Sharpe Ratio1.24Hit Rate94.1%Number of Trades68
Forecasting Performance
ModelRMSEDirectional AccuracyVIX (XGBoost)1.88656.7%Realised Volatility7.49-
Project Overview
The variance risk premium (VRP) is the difference between implied volatility (VIX) and subsequent realised volatility. This project builds a two-stage hierarchical model to forecast VRP and exploit it through systematic options trading.
Pipeline Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE 1: FORECASTING                     │
├────────────────────────────┬────────────────────────────────────┤
│     VIX Forecasting        │     Realised Volatility            │
│     AR(1) + XGBoost        │     XGBoost (22-day ahead)         │
│     150+ engineered        │     OHLC estimators                │
│     features               │     (Parkinson, GK, RS, YZ)        │
└────────────────────────────┴────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 2: VRP ESTIMATION                     │
│            VRP = E[VIX] - E[RV]                                 │
│            Hierarchical combination of Stage 1 outputs          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 3: TRADING                            │
│            SPX Bull Put Spreads                                 │
│            Signal: VRP > threshold                              │
└─────────────────────────────────────────────────────────────────┘
Repository Structure
├── r-forecasting/           # R code for VIX and RV forecasting
├── python-backtester/       # Python backtesting engine
├── thesis/                  # LaTeX dissertation source
└── presentation/            # Beamer slides
r-forecasting/
Core forecasting pipeline built in R with XGBoost, featuring purged expanding-window cross-validation and Bayesian hyperparameter optimisation.
FileDescriptionSetup.RPackage loading and environment configurationData Download.RFetches VIX, SPX, and options dataData Prep and Exploration.RInitial cleaning and EDACBOE VIX Construction.RReplicates CBOE VIX methodologyRealised Volatility Construction.RBuilds RV using five OHLC estimatorsRealised Volatility Exploratory Data Analysis.RRV distributional analysisFeature Engineering.R150+ features for VIX forecastingRV Feature Engineering.RFeature set for RV modelXGBOOST Regression.RVIX point forecast modelXGBOOST Classification.RVIX direction classificationRV XGBOOST Regression.R22-day RV forecast modelRV XGBOOST Classification.RRV direction classificationHAR GARCH Models.RBenchmark models (HAR-RV, GARCH family)Combination Prediction for VRP from VIX and RV Predictions.RHierarchical VRP estimationVolatility Risk Premium Exploratory Analysis.RVRP characteristics and regime analysisTotal Model Assessment.RFinal model evaluation and SHAP analysis
python-backtester/
Event-driven backtesting framework for SPX options strategies.
python-backtester/
├── main.py              # Entry point and orchestration
├── requirements.txt     # Dependencies
├── config/              # Strategy parameters and settings
├── data/                # Price and signal data
├── signals/             # VRP signals from R pipeline
├── strategy/            # Bull put spread logic
├── analytics/           # Performance metrics (Sharpe, drawdown, etc.)
├── charts/              # Visualisation outputs
└── results/             # Trade logs and summary statistics
Methodology
VIX Forecasting

Base model: AR(1) captures autoregressive structure
Residual model: XGBoost learns non-linear patterns in AR(1) residuals
Features: Lagged VIX, term structure slope, volume, momentum indicators, calendar effects
Validation: Purged expanding-window CV to prevent lookahead bias
Hyperparameters: Bayesian optimisation via ParBayesianOptimization

Realised Volatility

Target: 22-day ahead close-to-close realised volatility
Estimators: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
Structural breaks: Bai-Perron testing for regime shifts

Trading Strategy

Instrument: SPX bull put spreads (short put + long put at lower strike)
Entry signal: Positive VRP forecast above calibrated threshold
Position sizing: Fixed notional per trade
Exit: Expiration or stop-loss

Requirements
R

R >= 4.0
xgboost, ParBayesianOptimization, tidyverse, quantmod, rugarch

Python
bashpip install -r python-backtester/requirements.txt
Usage

Run the R scripts in order (Setup.R first, then Data Download.R, etc.)
Export VRP signals to python-backtester/signals/
Run the backtester:

bashcd python-backtester
python main.py


Author
Andrew Fouhy
BSc Financial Mathematics and Actuarial Science, University College Cork
andrewfouhy01@gmail.com
