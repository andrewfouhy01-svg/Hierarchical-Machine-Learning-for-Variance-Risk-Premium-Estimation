"""File-specific loaders handling CSV and XLSX with various date formats."""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_market_data(path: str) -> pd.DataFrame:
    """Load the market data CSV (converted from RDS). YYYY-MM-DD dates."""
    logger.info(f"Loading market data: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_rv_forecasts(path: str) -> pd.DataFrame:
    """Load RV forward regression (XLSX or CSV, YYYY-MM-DD dates)."""
    logger.info(f"Loading RV forecasts: {path}")
    if path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = pd.read_excel(path, engine="openpyxl")
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_vix_regression(path: str) -> pd.DataFrame:
    """Load VIX regression XLSX (YYYY-MM-DD dates)."""
    logger.info(f"Loading VIX regression: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_rv_classification(path: str) -> pd.DataFrame:
    """Load RV classification (CSV or XLSX, YYYY-MM-DD dates)."""
    logger.info(f"Loading RV classification: {path}")
    if path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = pd.read_excel(path, engine="openpyxl")
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_vix_classification(path: str) -> pd.DataFrame:
    """Load VIX classification XLSX (DD/MM/YYYY dates)."""
    logger.info(f"Loading VIX classification: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df


def load_option_chain(path: str) -> pd.DataFrame:
    """Load the SPX option chain XLSX (M/DD/YYYY dates). Cache as parquet."""
    parquet_path = Path(path).with_suffix(".parquet")
    if parquet_path.exists():
        logger.info(f"Loading cached option chain from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        return df

    logger.info(f"Loading option chain XLSX (this may take a while): {path}")
    df = pd.read_excel(path, engine="openpyxl")

    # Parse dates - M/DD/YYYY format
    for col in ["Date", "Expiration"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)

    # Cache as parquet for subsequent runs
    try:
        df.to_parquet(parquet_path)
        logger.info(f"Cached option chain to: {parquet_path}")
    except Exception as e:
        logger.warning(f"Could not cache to parquet: {e}")

    return df
