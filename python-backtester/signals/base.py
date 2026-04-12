"""Base signal interface and shared dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SignalOutput:
    date: date
    raw_value: float
    z_score: float
    regime: str          # e.g. 'EXTREMELY_RICH', 'RICH', 'NORMAL', 'LEAN', 'INVERTED'
    size_scalar: float   # 0.0 to 1.0
    confidence: float    # signal quality metric
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TradeDecision:
    date: date
    action: str          # 'ENTER', 'FLAT'
    size_scalar: float
    signals: dict        # {'vrp': SignalOutput, 'regime': SignalOutput}
    reason_codes: list = field(default_factory=list)


class BaseSignal(ABC):
    """Abstract base for all signal generators."""

    @abstractmethod
    def compute(self, dt: date) -> Optional[SignalOutput]:
        """Compute signal for a single date."""
        ...

    @abstractmethod
    def precompute(self, start: date, end: date) -> pd.DataFrame:
        """Precompute signal series for efficiency."""
        ...
