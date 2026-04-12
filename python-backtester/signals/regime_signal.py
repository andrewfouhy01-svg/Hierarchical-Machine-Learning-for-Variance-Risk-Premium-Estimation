"""Regime Signal (Signal 2): Forecast Uncertainty based on VIX prediction interval width."""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config.models import RegimeSignalConfig
from data.manager import DataManager
from signals.base import BaseSignal, SignalOutput

logger = logging.getLogger(__name__)


class ForecastUncertaintyRegime(BaseSignal):
    """
    Regime filter using VIX regression prediction interval width.

    PI_width = pi_95_upper - pi_95_lower
    Z-score over 252-day rolling window.

    Wide intervals => uncertain/dangerous => unfavourable.

    z_PI_width mapping:
        < +0.5   -> STABLE    (multiplier 1.0)
        +0.5..+1.0 -> ELEVATED (multiplier 0.75)
        +1.0..+1.5 -> STRESSED (multiplier 0.50)
        > +1.5   -> CRISIS    (multiplier 0.0, no trade)
    """

    def __init__(self, config: RegimeSignalConfig, data_manager: DataManager):
        self._config = config
        self._dm = data_manager
        self._precomputed: Optional[pd.DataFrame] = None

    def precompute(self, start: date, end: date) -> pd.DataFrame:
        df = self._dm.vix_regression.copy()

        # Compute PI width
        if "pi_95_upper" in df.columns and "pi_95_lower" in df.columns:
            df["pi_width"] = df["pi_95_upper"] - df["pi_95_lower"]
        else:
            # Fallback: use 90% interval if 95% not available
            logger.warning("95% PI columns not found, trying 90%")
            if "pi_90_upper" in df.columns and "pi_90_lower" in df.columns:
                df["pi_width"] = df["pi_90_upper"] - df["pi_90_lower"]
            else:
                logger.error("No prediction interval columns found. Setting pi_width=0.")
                df["pi_width"] = 0.0

        window = self._config.zscore_window

        # Rolling z-score of PI width
        rolling_mean = df["pi_width"].rolling(window=window, min_periods=1).mean()
        rolling_std = df["pi_width"].rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, np.nan)
        df["pi_zscore"] = (df["pi_width"] - rolling_mean) / rolling_std
        df["pi_zscore"] = df["pi_zscore"].fillna(0.0)

        # Map to regime
        t = self._config.thresholds
        df["regime_label"] = df["pi_zscore"].apply(lambda z: self._classify(z, t))
        df["regime_multiplier"] = df["pi_zscore"].apply(lambda z: self._multiplier(z, t))

        self._precomputed = df
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()

    def _classify(self, z: float, t) -> str:
        if z < t.stable:
            return "STABLE"
        elif z < t.elevated:
            return "ELEVATED"
        elif z < t.stressed:
            return "STRESSED"
        else:
            return "CRISIS"

    def _multiplier(self, z: float, t) -> float:
        if z < t.stable:
            return 1.0
        elif z < t.elevated:
            return 0.75
        elif z < t.stressed:
            return 0.50
        else:
            return 0.0

    def compute(self, dt: date) -> Optional[SignalOutput]:
        if self._precomputed is None:
            raise RuntimeError("Call precompute() before compute()")

        df = self._precomputed
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None

        row = row.iloc[0]
        return SignalOutput(
            date=dt,
            raw_value=float(row["pi_width"]) if not pd.isna(row["pi_width"]) else 0.0,
            z_score=float(row["pi_zscore"]),
            regime=row["regime_label"],
            size_scalar=float(row["regime_multiplier"]),
            confidence=1.0,
            metadata={"pi_width": float(row["pi_width"]) if not pd.isna(row["pi_width"]) else 0.0},
        )