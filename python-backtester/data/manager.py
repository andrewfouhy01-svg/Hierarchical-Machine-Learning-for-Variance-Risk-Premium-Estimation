"""DataManager: unified data access layer with internal caching."""

import logging
from datetime import date
from typing import Optional

import pandas as pd

from config.models import DataConfig
from data.loaders import (
    load_market_data,
    load_rv_forecasts,
    load_vix_regression,
    load_rv_classification,
    load_vix_classification,
    load_option_chain,
)

logger = logging.getLogger(__name__)


class DataManager:
    """Centralised data access. Loads all sources once and caches in memory."""

    def __init__(self, config: DataConfig, base_path: str = "."):
        self._config = config
        self._base = base_path
        self._market_data: Optional[pd.DataFrame] = None
        self._option_data: Optional[pd.DataFrame] = None
        self._rv_forecasts: Optional[pd.DataFrame] = None
        self._vix_regression: Optional[pd.DataFrame] = None
        self._rv_classification: Optional[pd.DataFrame] = None
        self._vix_classification: Optional[pd.DataFrame] = None

    def _path(self, rel: str) -> str:
        return f"{self._base}/{rel}"

    # --- Lazy loaders ---

    @property
    def market_data(self) -> pd.DataFrame:
        if self._market_data is None:
            self._market_data = load_market_data(self._path(self._config.market_data_path))
            self._market_data = self._market_data.sort_values("date").reset_index(drop=True)
        return self._market_data

    @property
    def option_data(self) -> pd.DataFrame:
        if self._option_data is None:
            self._option_data = load_option_chain(self._path(self._config.option_data_path))
        return self._option_data

    @property
    def rv_forecasts(self) -> pd.DataFrame:
        if self._rv_forecasts is None:
            self._rv_forecasts = load_rv_forecasts(self._path(self._config.rv_forecast_path))
            self._rv_forecasts = self._rv_forecasts.sort_values("date").reset_index(drop=True)
        return self._rv_forecasts

    @property
    def vix_regression(self) -> pd.DataFrame:
        if self._vix_regression is None:
            self._vix_regression = load_vix_regression(self._path(self._config.vix_regression_path))
            self._vix_regression = self._vix_regression.sort_values("date").reset_index(drop=True)
        return self._vix_regression

    @property
    def rv_classification(self) -> pd.DataFrame:
        if self._rv_classification is None:
            self._rv_classification = load_rv_classification(
                self._path(self._config.rv_classification_path)
            )
            self._rv_classification = self._rv_classification.sort_values("date").reset_index(drop=True)
        return self._rv_classification

    @property
    def vix_classification(self) -> pd.DataFrame:
        if self._vix_classification is None:
            self._vix_classification = load_vix_classification(
                self._path(self._config.vix_classification_path)
            )
            self._vix_classification = self._vix_classification.sort_values("date").reset_index(drop=True)
        return self._vix_classification

    # --- Query methods ---

    def get_market_data(self, start: date, end: date) -> pd.DataFrame:
        df = self.market_data
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()

    def get_spot(self, dt: date) -> Optional[float]:
        df = self.market_data
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None
        return float(row.iloc[0]["close"])

    def get_vix(self, dt: date) -> Optional[float]:
        df = self.market_data
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None
        return float(row.iloc[0]["vix_close"])

    def get_option_chain(self, dt: date, dte_range: tuple[int, int]) -> pd.DataFrame:
        """Return option chain rows for a given observation date within DTE range."""
        df = self.option_data
        # Filter by observation date
        mask_date = df["Date"].dt.date == dt
        sub = df.loc[mask_date].copy()
        if sub.empty:
            return sub
        # Filter by DTE range
        min_dte, max_dte = dte_range
        mask_dte = (sub["Days"] >= min_dte) & (sub["Days"] <= max_dte)
        return sub.loc[mask_dte].copy()

    def get_option_chain_dates(self) -> list[date]:
        """Return sorted unique observation dates in the option data."""
        dates = self.option_data["Date"].dt.date.unique()
        return sorted(dates)

    def get_rv_forecast_row(self, dt: date) -> Optional[pd.Series]:
        df = self.rv_forecasts
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None
        return row.iloc[0]

    def get_vix_regression_row(self, dt: date) -> Optional[pd.Series]:
        df = self.vix_regression
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None
        return row.iloc[0]

    def get_rv_classification_row(self, dt: date) -> Optional[pd.Series]:
        df = self.rv_classification
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None
        return row.iloc[0]

    def get_vix_classification_row(self, dt: date) -> Optional[pd.Series]:
        df = self.vix_classification
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None
        return row.iloc[0]

    def get_rv_forecasts_series(self, start: date, end: date) -> pd.DataFrame:
        df = self.rv_forecasts
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()

    def get_vix_regression_series(self, start: date, end: date) -> pd.DataFrame:
        df = self.vix_regression
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()
