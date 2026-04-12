"""Pydantic configuration models for the VRP backtesting system."""

from pydantic import BaseModel, field_validator
from datetime import date
from typing import Optional


class BacktestConfig(BaseModel):
    start_date: date
    end_date: date
    initial_capital: float
    risk_free_rate_annual: float


class StrategyConfig(BaseModel):
    target_dte: int
    min_dte_entry: int
    max_dte_entry: int
    short_put_delta: float
    long_put_delta: float
    short_call_delta: float
    long_vol_enabled: bool = False
    long_vol_risk_pct_nav: float = 0.015
    long_vol_buy_delta: float = 0.30
    long_vol_sell_delta: float = 0.10
    long_vol_max_margin: float = 0.10
    max_risk_pct_nav: float
    max_margin_utilisation: float
    circuit_breaker_monthly_pct: float
    circuit_breaker_cooloff_days: int
    hold_to_expiry: bool = True


class VRPThresholds(BaseModel):
    extremely_rich: float
    rich: float
    normal: float
    lean: float


class VRPSignalConfig(BaseModel):
    source_column: str
    zscore_window: int
    min_vrp_raw: float
    thresholds: VRPThresholds


class RegimeThresholds(BaseModel):
    stable: float
    elevated: float
    stressed: float


class RegimeSignalConfig(BaseModel):
    type: str
    zscore_window: int
    thresholds: RegimeThresholds


class SupplementaryConfig(BaseModel):
    vix_class_prob_threshold: float
    rv_class_prob_threshold: float
    supplementary_reduction: float


class SignalsConfig(BaseModel):
    vrp: VRPSignalConfig
    regime: RegimeSignalConfig
    supplementary: SupplementaryConfig


class ExecutionConfig(BaseModel):
    spread_fraction: float


class DataConfig(BaseModel):
    market_data_path: str
    option_data_path: str
    rv_forecast_path: str
    vix_regression_path: str
    rv_classification_path: str
    vix_classification_path: str


class AppConfig(BaseModel):
    backtest: BacktestConfig
    strategy: StrategyConfig
    signals: SignalsConfig
    execution: ExecutionConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
