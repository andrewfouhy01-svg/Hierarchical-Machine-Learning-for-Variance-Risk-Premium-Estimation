"""VRP Level Signal (Signal 1): consumes vrp_xgb from RV forward regression."""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config.models import VRPSignalConfig
from data.manager import DataManager
from signals.base import BaseSignal, SignalOutput

logger = logging.getLogger(__name__)


class VRPSignal(BaseSignal):
    """
    Primary entry gate based on VRP z-score.

    VRP = VIX_current - predicted_RV (forward-looking, from XGBoost).
    Z-score computed over a rolling 252-day window.

    Regime mapping:
        z > +1.5  -> EXTREMELY_RICH  (scalar 1.0, call overlay eligible)
        +0.5..+1.5 -> RICH           (scalar 1.0)
        0..+0.5   -> NORMAL          (scalar 0.67)
        -0.5..0   -> LEAN            (scalar 0.33)
        < -0.5    -> INVERTED        (scalar 0.0, no trade)

    Hard floor: vrp_raw must be > 0.
    """

    def __init__(self, config: VRPSignalConfig, data_manager: DataManager):
        self._config = config
        self._dm = data_manager
        self._precomputed: Optional[pd.DataFrame] = None

    def precompute(self, start: date, end: date) -> pd.DataFrame:
        """Precompute the full VRP signal series."""
        df = self._dm.rv_forecasts.copy()
        col = self._config.source_column  # 'vrp_xgb'

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in RV forecast data. Available: {list(df.columns)}")

        window = self._config.zscore_window

        # Rolling z-score with expanding window for warm-up
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        df["vrp_zscore"] = (df[col] - rolling_mean) / rolling_std
        df["vrp_zscore"] = df["vrp_zscore"].fillna(0.0)

        # Flag warm-up period
        df["is_warmup"] = df.index < window

        # Map to regime and scalar
        thresholds = self._config.thresholds
        df["vrp_regime"] = df["vrp_zscore"].apply(
            lambda z: self._classify_regime(z, thresholds)
        )
        df["vrp_scalar"] = df.apply(
            lambda row: self._compute_scalar(row[col], row["vrp_zscore"], thresholds),
            axis=1,
        )

        self._precomputed = df
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()

    def _classify_regime(self, z: float, t) -> str:
        if z > t.extremely_rich:
            return "EXTREMELY_RICH"
        elif z > t.rich:
            return "RICH"
        elif z > t.normal:
            return "NORMAL"
        elif z > t.lean:
            return "LEAN"
        else:
            return "INVERTED"

    def _compute_scalar(self, vrp_raw: float, z: float, t) -> float:
        # Hard floor: VRP must be positive
        if pd.isna(vrp_raw) or vrp_raw <= self._config.min_vrp_raw:
            return 0.0
        if z > t.extremely_rich:
            return 1.0
        elif z > t.rich:
            return 1.0
        elif z > t.normal:
            return 0.67
        elif z > t.lean:
            return 0.33
        else:
            return 0.0

    def compute(self, dt: date) -> Optional[SignalOutput]:
        """Compute VRP signal for a single date."""
        if self._precomputed is None:
            raise RuntimeError("Call precompute() before compute()")

        df = self._precomputed
        row = df.loc[df["date"].dt.date == dt]
        if row.empty:
            return None

        row = row.iloc[0]
        col = self._config.source_column
        vrp_raw = float(row[col]) if not pd.isna(row[col]) else 0.0

        return SignalOutput(
            date=dt,
            raw_value=vrp_raw,
            z_score=float(row["vrp_zscore"]),
            regime=row["vrp_regime"],
            size_scalar=float(row["vrp_scalar"]),
            confidence=1.0 if not row["is_warmup"] else 0.5,
            metadata={
                "vrp_raw": vrp_raw,
                "is_warmup": bool(row["is_warmup"]),
            },
        )