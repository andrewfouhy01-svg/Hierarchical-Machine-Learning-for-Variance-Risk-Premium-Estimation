"""
VRP Signal-Gated SPX Options Backtesting System
================================================
Main entry point. Run from the project root:

    python main.py

Ensure all data files are in place per config/default.yaml paths.
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.models import AppConfig
from data.manager import DataManager
from strategy.engine import BacktestEngine
from analytics.metrics import compute_metrics, PerformanceMetrics
from analytics.tearsheet import generate_tearsheet
from analytics.signal_diagnostics import signal_diagnostics, print_diagnostics


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_metrics(m: PerformanceMetrics) -> None:
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  Total Return:        {m.total_return:>10.2%}")
    print(f"  CAGR:                {m.cagr:>10.2%}")
    print(f"  Annualised Vol:      {m.annualised_vol:>10.2%}")
    print(f"  Sharpe Ratio:        {m.sharpe:>10.2f}")
    print(f"  Sortino Ratio:       {m.sortino:>10.2f}")
    print(f"  Calmar Ratio:        {m.calmar:>10.2f}")
    print(f"  Omega Ratio:         {m.omega:>10.2f}")
    print(f"  Max Drawdown:        {m.max_drawdown:>10.2%}")
    print(f"  Max DD Duration:     {m.max_drawdown_duration_days:>10d} days")
    print(f"  VaR (95%):           {m.var_95:>10.2%}")
    print(f"  CVaR (95%):          {m.cvar_95:>10.2%}")
    print(f"  VaR (99%):           {m.var_99:>10.2%}")
    print(f"  CVaR (99%):          {m.cvar_99:>10.2%}")
    print(f"  ---")
    print(f"  Total Trades:        {m.total_trades:>10d}")
    print(f"  Winning Trades:      {m.winning_trades:>10d}")
    print(f"  Losing Trades:       {m.losing_trades:>10d}")
    print(f"  Hit Rate:            {m.hit_rate:>10.1%}")
    print(f"  Avg Win:             ${m.avg_win:>10,.0f}")
    print(f"  Avg Loss:            ${m.avg_loss:>10,.0f}")
    print(f"  Profit Factor:       {m.profit_factor:>10.2f}")
    print(f"  Premium Capture:     {m.premium_capture_ratio:>10.2%}")
    print(f"  ---")
    print(f"  Avg Monthly Return:  {m.avg_monthly_return:>10.2%}")
    print(f"  Best Month:          {m.best_month:>10.2%}")
    print(f"  Worst Month:         {m.worst_month:>10.2%}")
    print(f"  % Positive Months:   {m.pct_positive_months:>10.1%}")
    print("=" * 60)


def main():
    setup_logging("INFO")
    logger = logging.getLogger("main")

    # Load config
    config_path = Path(__file__).parent / "config" / "default.yaml"
    logger.info(f"Loading config from {config_path}")
    config = AppConfig.from_yaml(str(config_path))

    # Validate data files exist
    base_path = Path(__file__).parent
    missing = []
    for field_name in ["market_data_path", "option_data_path", "rv_forecast_path",
                       "vix_regression_path", "rv_classification_path", "vix_classification_path"]:
        fpath = base_path / getattr(config.data, field_name)
        if not fpath.exists():
            missing.append(str(fpath))

    if missing:
        logger.error("Missing data files:")
        for m in missing:
            logger.error(f"  {m}")
        logger.error("Please ensure all data files are in place.")
        sys.exit(1)

    # Initialise data manager
    logger.info("Initialising DataManager...")
    dm = DataManager(config.data, base_path=str(base_path))

    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(config, dm)
    portfolio = engine.run()

    # Compute metrics
    logger.info("Computing performance metrics...")
    metrics = compute_metrics(portfolio, config.backtest.risk_free_rate_annual)
    print_metrics(metrics)

    # Signal diagnostics
    logger.info("Computing signal diagnostics...")
    diag = signal_diagnostics(portfolio)
    print_diagnostics(diag)

    # Generate tearsheet 
    tearsheet_path = str(base_path / "tearsheet.html")
    logger.info(f"Generating tearsheet at {tearsheet_path}...")
    generate_tearsheet(portfolio, metrics, tearsheet_path)

    # Save trade log to CSV
    trade_log_path = str(base_path / "trade_log.csv")
    import pandas as pd
    trade_df = pd.DataFrame(portfolio.trade_log)
    trade_df.to_csv(trade_log_path, index=False)
    logger.info(f"Trade log saved to {trade_log_path}")

    # Export decision logs
    from analytics.decision_log import export_decision_log_csv, export_monthly_summary_csv
    export_decision_log_csv(portfolio, str(base_path / "decision_log.csv"))
    export_monthly_summary_csv(portfolio, str(base_path / "monthly_decision_log.csv"))

    logger.info("Done.")


if __name__ == "__main__":
    main()