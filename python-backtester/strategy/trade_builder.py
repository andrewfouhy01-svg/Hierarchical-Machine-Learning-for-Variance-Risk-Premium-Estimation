"""Trade builder: delta-targeted strike selection and spread construction."""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config.models import StrategyConfig, ExecutionConfig

logger = logging.getLogger(__name__)


@dataclass
class SpreadTrade:
    """Represents a single put spread (or iron condor) trade at entry."""
    entry_date: date
    expiry: date
    # Put spread
    short_put_strike: float
    long_put_strike: float
    short_put_delta: float
    long_put_delta: float
    short_put_mid: float
    long_put_mid: float
    # Credit and risk
    entry_credit_per_spread: float  # net credit received per spread
    spread_width: float             # short_strike - long_strike (always positive for bull put)
    max_loss_per_spread: float      # spread_width * 100 - credit
    # Sizing
    contracts: int
    total_credit: float
    total_max_loss: float
    # Context
    entry_spot: float
    entry_vix: float
    dte: int
    size_scalar: float
    # Optional call overlay
    short_call_strike: Optional[float] = None
    short_call_delta: Optional[float] = None
    short_call_mid: Optional[float] = None
    call_credit_per_spread: Optional[float] = None
    # Expiry data for P&L
    spot_at_expiry: Optional[float] = None
    # Direction flag
    is_long_vol: bool = False


def select_strike_by_delta(
    chain: pd.DataFrame,
    target_delta: float,
    option_type: str = "P",
) -> Optional[pd.Series]:
    """
    Select the option row closest to the target absolute delta.

    For puts, delta is negative; we match on |delta|.
    For calls, delta is positive.
    """
    sub = chain[chain["CallPut"] == option_type].copy()
    if sub.empty:
        return None

    sub["abs_delta"] = sub["Delta"].abs()

    # Filter out options with zero bid (illiquid)
    sub = sub[sub["BestBid"] > 0]
    if sub.empty:
        return None

    # Find closest to target delta
    sub["delta_diff"] = (sub["abs_delta"] - target_delta).abs()
    best_idx = sub["delta_diff"].idxmin()
    return sub.loc[best_idx]


def _find_long_put_for_budget(
    puts: pd.DataFrame,
    short_strike: float,
    max_width: float,
    min_width: float = 15.0,
) -> Optional[pd.Series]:
    """Find a long put strike within max_width of short strike, preferring widest available."""
    target_long = short_strike - max_width
    candidates = puts[
        (puts["Strike"] >= target_long) &
        (puts["Strike"] <= short_strike - min_width) &  # at least min_width apart
        (puts["BestBid"] > 0)
    ].copy()

    if candidates.empty:
        return None

    # Pick the lowest strike that fits (widest spread within budget)
    candidates = candidates.sort_values("Strike", ascending=True)
    return candidates.iloc[0]


def build_put_spread(
    chain: pd.DataFrame,
    strategy_config: StrategyConfig,
    execution_config: ExecutionConfig,
    nav: float,
    size_scalar: float,
    entry_date: date,
    entry_spot: float,
    entry_vix: float,
    call_overlay: bool = False,
) -> Optional[SpreadTrade]:
    """
    Build a delta-targeted bull put spread from the option chain.

    Dynamically adjusts spread width to ensure at least 1 contract fits
    within the risk budget.
    """
    # Select puts only
    puts = chain[chain["CallPut"] == "P"].copy()
    if puts.empty:
        logger.warning(f"No put options available for {entry_date}")
        return None

    # Get expiry from the chain
    expiry_dates = puts["Expiration"].unique()
    if len(expiry_dates) == 0:
        return None
    expiry = pd.Timestamp(expiry_dates[0]).date()
    dte = int(puts["Days"].iloc[0])

    # Short put: nearest to 16-delta
    short_put = select_strike_by_delta(puts, strategy_config.short_put_delta, "P")
    if short_put is None:
        logger.warning(f"Could not find short put strike for {entry_date}")
        return None

    short_strike = float(short_put["Strike"])

    # Compute risk budget first to determine max affordable spread width
    risk_budget = nav * strategy_config.max_risk_pct_nav * size_scalar

    # Max spread width: min of config cap AND what we can afford for 1 contract
    config_max_width = getattr(strategy_config, 'max_spread_width', 150.0)
    affordable_width = risk_budget / 100  # $risk_budget buys 1 contract of this width
    effective_max_width = min(config_max_width, affordable_width)

    # Minimum viable spread width (in strike points)
    min_width = 15.0

    if effective_max_width < min_width:
        logger.warning(
            f"Risk budget too small for any spread on {entry_date}: "
            f"budget=${risk_budget:.0f}, min_width={min_width}"
        )
        return None

    # Try delta-targeted long put first
    long_put = select_strike_by_delta(puts, strategy_config.long_put_delta, "P")

    # Check if natural spread fits, otherwise narrow it
    if long_put is not None and long_put["Strike"] < short_strike:
        natural_width = short_strike - float(long_put["Strike"])
        if natural_width > effective_max_width:
            # Natural spread too wide - find a closer long put
            long_put = _find_long_put_for_budget(
                puts, short_strike, effective_max_width, min_width
            )
    elif long_put is not None and long_put["Strike"] >= short_strike:
        # Invalid - long >= short, try budget-based selection
        long_put = _find_long_put_for_budget(
            puts, short_strike, effective_max_width, min_width
        )

    if long_put is None:
        long_put = _find_long_put_for_budget(
            puts, short_strike, effective_max_width, min_width
        )

    if long_put is None:
        logger.warning(f"Could not find suitable long put for {entry_date}")
        return None

    if float(long_put["Strike"]) >= short_strike:
        logger.warning(f"Long put >= short put for {entry_date}. Skipping.")
        return None

    # Compute mid prices
    short_mid = float(short_put["MidPrice"]) if "MidPrice" in short_put.index else (
        (float(short_put["BestBid"]) + float(short_put["BestOffer"])) / 2
    )
    long_mid = float(long_put["MidPrice"]) if "MidPrice" in long_put.index else (
        (float(long_put["BestBid"]) + float(long_put["BestOffer"])) / 2
    )

    # Apply slippage
    short_half_spread = (float(short_put["BestOffer"]) - float(short_put["BestBid"])) / 2
    sell_price = short_mid - execution_config.spread_fraction * short_half_spread

    long_half_spread = (float(long_put["BestOffer"]) - float(long_put["BestBid"])) / 2
    buy_price = long_mid + execution_config.spread_fraction * long_half_spread

    credit_per_spread = sell_price - buy_price
    if credit_per_spread <= 0:
        logger.warning(f"Negative credit ({credit_per_spread:.2f}) for {entry_date}. Skipping.")
        return None

    spread_width = short_strike - float(long_put["Strike"])
    max_loss_per_spread = spread_width * 100 - credit_per_spread * 100
    risk_per_spread = spread_width * 100

    # Position sizing
    contracts = int(np.floor(risk_budget / risk_per_spread)) if risk_per_spread > 0 else 0

    if contracts <= 0:
        logger.warning(
            f"Zero contracts for {entry_date}: risk_budget=${risk_budget:.0f}, "
            f"risk_per_spread=${risk_per_spread:.0f}, scalar={size_scalar:.3f}"
        )
        return None

    # Check margin utilisation
    total_margin = contracts * risk_per_spread
    if total_margin > nav * strategy_config.max_margin_utilisation:
        contracts = int(np.floor(nav * strategy_config.max_margin_utilisation / risk_per_spread))

    if contracts <= 0:
        return None

    total_credit = credit_per_spread * contracts * 100
    total_max_loss = max_loss_per_spread * contracts

    # Spot at expiry (for later P&L calc)
    spot_at_expiry = float(short_put["PriceExp"]) if "PriceExp" in short_put.index else None

    trade = SpreadTrade(
        entry_date=entry_date,
        expiry=expiry,
        short_put_strike=short_strike,
        long_put_strike=float(long_put["Strike"]),
        short_put_delta=float(short_put["Delta"]),
        long_put_delta=float(long_put["Delta"]),
        short_put_mid=short_mid,
        long_put_mid=long_mid,
        entry_credit_per_spread=credit_per_spread,
        spread_width=spread_width,
        max_loss_per_spread=max_loss_per_spread,
        contracts=contracts,
        total_credit=total_credit,
        total_max_loss=total_max_loss,
        entry_spot=entry_spot,
        entry_vix=entry_vix,
        dte=dte,
        size_scalar=size_scalar,
        spot_at_expiry=spot_at_expiry,
    )

    # Optional call overlay
    if call_overlay:
        calls = chain[chain["CallPut"] == "C"].copy()
        short_call = select_strike_by_delta(calls, strategy_config.short_call_delta, "C")
        if short_call is not None:
            call_mid = float(short_call["MidPrice"]) if "MidPrice" in short_call.index else (
                (float(short_call["BestBid"]) + float(short_call["BestOffer"])) / 2
            )
            call_half_spread = (float(short_call["BestOffer"]) - float(short_call["BestBid"])) / 2
            call_sell_price = call_mid - execution_config.spread_fraction * call_half_spread
            trade.short_call_strike = float(short_call["Strike"])
            trade.short_call_delta = float(short_call["Delta"])
            trade.short_call_mid = call_mid
            trade.call_credit_per_spread = call_sell_price

    return trade


def build_long_put_spread(
    chain: pd.DataFrame,
    strategy_config: StrategyConfig,
    execution_config: ExecutionConfig,
    nav: float,
    entry_date: date,
    entry_spot: float,
    entry_vix: float,
) -> Optional[SpreadTrade]:
    """
    Build a bear put spread (debit spread) for long volatility exposure.

    Buy a higher-delta put (closer to ATM), sell a lower-delta put (further OTM).
    Profits when spot drops. Used when the composite signal says FLAT.

    SpreadTrade convention is preserved:
      - short_put_strike = the put we SELL (lower strike)
      - long_put_strike  = the put we BUY  (higher strike)
      - total_credit < 0 (it's a debit)
    Settlement maths in portfolio.py handles this correctly without changes.
    """
    puts = chain[chain["CallPut"] == "P"].copy()
    if puts.empty:
        logger.warning(f"[LONG VOL] No put options for {entry_date}")
        return None

    expiry_dates = puts["Expiration"].unique()
    if len(expiry_dates) == 0:
        return None
    expiry = pd.Timestamp(expiry_dates[0]).date()
    dte = int(puts["Days"].iloc[0])

    # Delta targets for long vol (use config with getattr defaults)
    buy_delta = getattr(strategy_config, "long_vol_buy_delta", 0.30)
    sell_delta = getattr(strategy_config, "long_vol_sell_delta", 0.10)

    # Buy the higher-delta put (closer to ATM) - this is our LONG put
    buy_put = select_strike_by_delta(puts, buy_delta, "P")
    if buy_put is None:
        logger.warning(f"[LONG VOL] No buy-side put for {entry_date}")
        return None

    # Sell the lower-delta put (further OTM) - this is our SHORT put
    sell_put = select_strike_by_delta(puts, sell_delta, "P")
    if sell_put is None:
        logger.warning(f"[LONG VOL] No sell-side put for {entry_date}")
        return None

    buy_strike = float(buy_put["Strike"])
    sell_strike = float(sell_put["Strike"])

    # Validate: buy strike must be above sell strike for a bear put spread
    if buy_strike <= sell_strike:
        logger.warning(
            f"[LONG VOL] Invalid strikes on {entry_date}: "
            f"buy={buy_strike}, sell={sell_strike}"
        )
        return None

    # Mid prices
    buy_mid = float(buy_put["MidPrice"]) if "MidPrice" in buy_put.index else (
        (float(buy_put["BestBid"]) + float(buy_put["BestOffer"])) / 2
    )
    sell_mid = float(sell_put["MidPrice"]) if "MidPrice" in sell_put.index else (
        (float(sell_put["BestBid"]) + float(sell_put["BestOffer"])) / 2
    )

    # Apply slippage: we cross the spread on both legs (adverse fills)
    buy_half = (float(buy_put["BestOffer"]) - float(buy_put["BestBid"])) / 2
    buy_price = buy_mid + execution_config.spread_fraction * buy_half  # pay up

    sell_half = (float(sell_put["BestOffer"]) - float(sell_put["BestBid"])) / 2
    sell_price = sell_mid - execution_config.spread_fraction * sell_half  # receive less

    debit_per_spread = buy_price - sell_price  # positive number
    if debit_per_spread <= 0:
        logger.warning(f"[LONG VOL] Zero/negative debit on {entry_date}. Skipping.")
        return None

    spread_width = buy_strike - sell_strike  # positive
    risk_per_spread = debit_per_spread * 100  # max loss = debit paid

    # Sizing: configurable % of NAV for long vol
    long_vol_risk_pct = getattr(strategy_config, "long_vol_risk_pct_nav", 0.015)
    risk_budget = nav * long_vol_risk_pct

    contracts = int(np.floor(risk_budget / risk_per_spread)) if risk_per_spread > 0 else 0
    if contracts <= 0:
        logger.warning(
            f"[LONG VOL] Zero contracts on {entry_date}: "
            f"budget=${risk_budget:.0f}, risk_per=${risk_per_spread:.0f}"
        )
        return None

    # Cap margin utilisation
    total_margin = contracts * risk_per_spread
    max_margin = nav * getattr(strategy_config, "long_vol_max_margin", 0.10)
    if total_margin > max_margin:
        contracts = int(np.floor(max_margin / risk_per_spread))
    if contracts <= 0:
        return None

    # Credits are negative for debit spreads
    total_credit = -debit_per_spread * contracts * 100
    max_profit_per_spread = (spread_width - debit_per_spread) * 100
    total_max_loss = debit_per_spread * contracts * 100

    # Spot at expiry
    spot_at_expiry = float(buy_put["PriceExp"]) if "PriceExp" in buy_put.index else None

    trade = SpreadTrade(
        entry_date=entry_date,
        expiry=expiry,
        # Convention: short = sold, long = bought
        short_put_strike=sell_strike,
        long_put_strike=buy_strike,
        short_put_delta=float(sell_put["Delta"]),
        long_put_delta=float(buy_put["Delta"]),
        short_put_mid=sell_mid,
        long_put_mid=buy_mid,
        entry_credit_per_spread=-debit_per_spread,
        spread_width=spread_width,
        max_loss_per_spread=debit_per_spread * 100,
        contracts=contracts,
        total_credit=total_credit,
        total_max_loss=total_max_loss,
        entry_spot=entry_spot,
        entry_vix=entry_vix,
        dte=dte,
        size_scalar=0.0,  # signal was FLAT
        spot_at_expiry=spot_at_expiry,
        is_long_vol=True,
    )

    return trade