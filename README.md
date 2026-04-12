# Hierarchical Machine Learning for Variance Risk Premium Estimation

Hierarchical XGBoost pipeline forecasting VIX and realised volatility to estimate variance risk premium and systematically trade SPX bull put spreads.

## Key Results

| Metric | Value |
|--------|-------|
| Total Return | 53.0% |
| Sharpe Ratio | 1.24 |
| Hit Rate | 94.1% |
| Number of Trades | 68 |

### Forecasting Performance

| Model | RMSE | Directional Accuracy |
|-------|------|---------------------|
| VIX (XGBoost) | 1.886 | 56.7% |
| Realised Volatility | 7.49 | - |

## Project Overview

The variance risk premium (VRP) is the difference between implied volatility (VIX) and subsequent realised volatility. This project builds a two-stage hierarchical model to forecast VRP and exploit it through systematic options trading.

### Pipeline Architecture

```
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
```

## Repository Structure

```
├── r-forecasting/           # R code for VIX and RV forecasting
├── python-backtester/       # Python backtesting engine
├── thesis/                  # LaTeX dissertation source
└── presentation/            # Beamer slides
```

### r-forecasting/

Core forecasting pipeline built in R with XGBoost, featuring purged expanding-window cross-validation and Bayesian hyperparameter optimisation.

| File | Description |
|------|-------------|
| `Setup.R` | Package loading and environment configuration |
| `Data Download.R` | Fetches VIX, SPX, and options data |
| `Data Prep and Exploration.R` | Initial cleaning and EDA |
| `CBOE VIX Construction.R` | Replicates CBOE VIX methodology |
| `Realised Volatility Construction.R` | Builds RV using five OHLC estimators |
| `Realised Volatility Exploratory Data Analysis.R` | RV distributional analysis |
| `Feature Engineering.R` | 150+ features for VIX forecasting |
| `RV Feature Engineering.R` | Feature set for RV model |
| `XGBOOST Regression.R` | VIX point forecast model |
| `XGBOOST Classification.R` | VIX direction classification |
| `RV XGBOOST Regression.R` | 22-day RV forecast model |
| `RV XGBOOST Classification.R` | RV direction classification |
| `HAR GARCH Models.R` | Benchmark models (HAR-RV, GARCH family) |
| `Combination Prediction for VRP from VIX and RV Predictions.R` | Hierarchical VRP estimation |
| `Volatility Risk Premium Exploratory Analysis.R` | VRP characteristics and regime analysis |
| `Total Model Assessment.R` | Final model evaluation and SHAP analysis |

### python-backtester/

Event-driven backtesting framework for SPX options strategies.

```
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
```

## Methodology

### VIX Forecasting

- **Base model**: AR(1) captures autoregressive structure
- **Residual model**: XGBoost learns non-linear patterns in AR(1) residuals
- **Features**: Lagged VIX, term structure slope, volume, momentum indicators, calendar effects
- **Validation**: Purged expanding-window CV to prevent lookahead bias
- **Hyperparameters**: Bayesian optimisation via ParBayesianOptimization

### Realised Volatility

- **Target**: 22-day ahead close-to-close realised volatility
- **Estimators**: Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
- **Structural breaks**: Bai-Perron testing for regime shifts

### Trading Strategy

- **Instrument**: SPX bull put spreads (short put + long put at lower strike)
- **Entry signal**: Positive VRP forecast above calibrated threshold
- **Position sizing**: Fixed notional per trade
- **Exit**: Expiration or stop-loss

## Requirements

### R
- R >= 4.0
- xgboost, ParBayesianOptimization, tidyverse, quantmod, rugarch

### Python
```bash
pip install -r python-backtester/requirements.txt
```

## Usage

1. Run the R scripts in order (Setup.R first, then Data Download.R, etc.)
2. Export VRP signals to `python-backtester/signals/`
3. Run the backtester:
```bash
cd python-backtester
python main.py
```


## Author

Andrew Fouhy  
BSc Financial Mathematics and Actuarial Science, University College Cork  
andrewfouhy01@gmail.com
