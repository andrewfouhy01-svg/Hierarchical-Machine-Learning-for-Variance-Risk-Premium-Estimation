"""Main backtest engine: iterates over option chain dates, evaluates signals, enters trades."""

import logging
from datetime import date

import pandas as pd

from config.models import AppConfig
from data.manager import DataManager
from signals.composite import CompositeSignal
from strategy.portfolio import Portfolio
from strategy.trade_builder import build_put_spread, build_long_put_spread

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Simplified backtest engine for hold-to-expiry strategy.

    Workflow per option chain date:
      1. Settle any expired positions
      2. Evaluate composite signal
      3. If trade permitted and no existing position for this cycle, enter
      4. Record snapshot
    """

    def __init__(self, config: AppConfig, data_manager: DataManager):
        self.config = config
        self.dm = data_manager
        self.signal = CompositeSignal(config.signals, data_manager)
        self.portfolio = Portfolio(config.backtest.initial_capital)

    def run(self) -> Portfolio:
        """Execute the full backtest."""
        start = self.config.backtest.start_date
        end = self.config.backtest.end_date

        logger.info(f"Starting backtest: {start} to {end}")
        logger.info(f"Initial capital: ${self.config.backtest.initial_capital:,.0f}")

        # Precompute signals
        self.signal.precompute(start, end)

        # Get all option chain observation dates (for entry evaluation)
        all_chain_dates = set(self.dm.get_option_chain_dates())
        chain_dates_in_range = {d for d in all_chain_dates if start <= d <= end}
        logger.info(f"Option chain dates in range: {len(chain_dates_in_range)}")

        # Get all trading days from market data for daily snapshots
        market_df = self.dm.get_market_data(start, end)
        trading_days = sorted(market_df["date"].dt.date.unique())
        logger.info(f"Trading days in range: {len(trading_days)}")

        if not trading_days:
            logger.error("No trading days found in backtest range!")
            return self.portfolio

        trades_entered = 0
        trades_entered_long_vol = 0
        trades_skipped_signal = 0
        trades_skipped_no_chain = 0
        trades_skipped_existing = 0
        trades_skipped_circuit = 0
        trades_skipped_build_fail = 0

        dte_range = (
            self.config.strategy.min_dte_entry,
            self.config.strategy.max_dte_entry,
        )

        for day in trading_days:
            # 1. Settle expired positions
            settled = self.portfolio.settle_expired(day)
            for closed in settled:
                self.portfolio.update_monthly_pnl(
                    closed.pnl,
                    day,
                    self.config.strategy.circuit_breaker_monthly_pct,
                    self.config.strategy.circuit_breaker_cooloff_days,
                )

            # 2. Only evaluate entries on option chain dates
            if day not in chain_dates_in_range:
                self.portfolio.record_snapshot(day)
                continue

            # 3. Check circuit breaker (blocks ALL trading, including long vol)
            if self.portfolio.check_circuit_breaker(
                day,
                self.config.strategy.circuit_breaker_monthly_pct,
                self.config.strategy.circuit_breaker_cooloff_days,
            ):
                trades_skipped_circuit += 1
                self.portfolio.trade_log.append({
                    "action": "SKIP",
                    "date": day,
                    "reason_codes": ["CIRCUIT_BREAKER_ACTIVE"],
                })
                self.portfolio.record_snapshot(day)
                continue

            # 4. Get option chain (need data for any trade direction)
            chain = self.dm.get_option_chain(day, dte_range)
            if chain.empty:
                trades_skipped_no_chain += 1
                self.portfolio.trade_log.append({
                    "action": "SKIP",
                    "date": day,
                    "reason_codes": ["NO_OPTION_CHAIN"],
                })
                self.portfolio.record_snapshot(day)
                continue

            expiry_dates = chain["Expiration"].unique()
            if len(expiry_dates) == 0:
                trades_skipped_no_chain += 1
                self.portfolio.trade_log.append({
                    "action": "SKIP",
                    "date": day,
                    "reason_codes": ["NO_EXPIRY_IN_CHAIN"],
                })
                self.portfolio.record_snapshot(day)
                continue

            expiry = pd.Timestamp(expiry_dates[0]).date()

            # ============================================================
            # SHORT VOL PATH — identical to original, sets rejection flag
            # ============================================================
            short_vol_rejected = False
            reject_reason_codes = []
            decision = None
            entry_spot = None
            entry_vix = None
            vrp_sig = None
            reg_sig = None

            # Check for existing SHORT VOL position in this expiry cycle
            if self.portfolio.has_position_for_expiry(expiry, is_long_vol=False):
                trades_skipped_existing += 1
                self.portfolio.trade_log.append({
                    "action": "SKIP",
                    "date": day,
                    "reason_codes": [f"EXISTING_SHORT_VOL_FOR_{expiry}"],
                })
                short_vol_rejected = True
                reject_reason_codes = [f"EXISTING_SHORT_VOL_FOR_{expiry}"]

            # Only evaluate signal if no existing position (same as original)
            if not short_vol_rejected:
                decision = self.signal.evaluate(day)

                entry_spot = self.dm.get_spot(day)
                entry_vix = self.dm.get_vix(day)
                if entry_spot is None:
                    entry_spot = float(chain["PriceDate"].iloc[0]) if "PriceDate" in chain.columns else 0.0
                if entry_vix is None:
                    entry_vix = 0.0

                vrp_sig = decision.signals.get("vrp")
                reg_sig = decision.signals.get("regime")

                if decision.action != "ENTER":
                    trades_skipped_signal += 1
                    self.portfolio.trade_log.append({
                        "action": "SKIP",
                        "date": day,
                        "reason_codes": decision.reason_codes,
                        "vrp_regime": vrp_sig.regime if vrp_sig else None,
                        "vrp_raw": vrp_sig.raw_value if vrp_sig else None,
                        "vrp_zscore": vrp_sig.z_score if vrp_sig else None,
                        "vrp_scalar": vrp_sig.size_scalar if vrp_sig else None,
                        "vrp_confidence": vrp_sig.confidence if vrp_sig else None,
                        "regime": reg_sig.regime if reg_sig else None,
                        "regime_pi_width": reg_sig.raw_value if reg_sig else None,
                        "regime_zscore": reg_sig.z_score if reg_sig else None,
                        "regime_multiplier": reg_sig.size_scalar if reg_sig else None,
                        "entry_spot": entry_spot,
                        "entry_vix": entry_vix,
                    })
                    short_vol_rejected = True
                    reject_reason_codes = decision.reason_codes

                else:
                    # Signal says ENTER — attempt to build short vol trade
                    call_overlay = "CALL_OVERLAY_ELIGIBLE" in decision.reason_codes

                    trade = build_put_spread(
                        chain=chain,
                        strategy_config=self.config.strategy,
                        execution_config=self.config.execution,
                        nav=self.portfolio.nav,
                        size_scalar=decision.size_scalar,
                        entry_date=day,
                        entry_spot=entry_spot,
                        entry_vix=entry_vix,
                        call_overlay=call_overlay,
                    )

                    if trade is None:
                        trades_skipped_build_fail += 1
                        self.portfolio.trade_log.append({
                            "action": "SKIP",
                            "date": day,
                            "reason_codes": ["BUILD_FAIL"] + decision.reason_codes,
                            "vrp_regime": vrp_sig.regime if vrp_sig else None,
                            "vrp_raw": vrp_sig.raw_value if vrp_sig else None,
                            "vrp_zscore": vrp_sig.z_score if vrp_sig else None,
                            "vrp_scalar": vrp_sig.size_scalar if vrp_sig else None,
                            "regime": reg_sig.regime if reg_sig else None,
                            "regime_pi_width": reg_sig.raw_value if reg_sig else None,
                            "regime_zscore": reg_sig.z_score if reg_sig else None,
                            "regime_multiplier": reg_sig.size_scalar if reg_sig else None,
                            "composite_scalar": decision.size_scalar,
                            "entry_spot": entry_spot,
                            "entry_vix": entry_vix,
                        })
                        short_vol_rejected = True
                        reject_reason_codes = ["BUILD_FAIL"] + decision.reason_codes

                    else:
                        # Short vol trade entered successfully
                        self.portfolio.enter_trade(trade)
                        trades_entered += 1

                        self.portfolio.trade_log[-1].update({
                            "reason_codes": decision.reason_codes,
                            "vrp_regime": vrp_sig.regime if vrp_sig else None,
                            "vrp_raw": vrp_sig.raw_value if vrp_sig else None,
                            "vrp_zscore": vrp_sig.z_score if vrp_sig else None,
                            "vrp_scalar": vrp_sig.size_scalar if vrp_sig else None,
                            "vrp_confidence": vrp_sig.confidence if vrp_sig else None,
                            "regime": reg_sig.regime if reg_sig else None,
                            "regime_pi_width": reg_sig.raw_value if reg_sig else None,
                            "regime_zscore": reg_sig.z_score if reg_sig else None,
                            "regime_multiplier": reg_sig.size_scalar if reg_sig else None,
                            "composite_scalar": decision.size_scalar,
                        })

            # ============================================================
            # LONG VOL PATH — only reached if short vol was rejected
            # ============================================================
            long_vol_enabled = getattr(self.config.strategy, "long_vol_enabled", False)

            if short_vol_rejected and long_vol_enabled:
                # Need spot/VIX if we skipped signal evaluation (existing position case)
                if entry_spot is None:
                    entry_spot = self.dm.get_spot(day)
                    if entry_spot is None:
                        entry_spot = float(chain["PriceDate"].iloc[0]) if "PriceDate" in chain.columns else 0.0
                if entry_vix is None:
                    entry_vix = self.dm.get_vix(day)
                    if entry_vix is None:
                        entry_vix = 0.0

                if not self.portfolio.has_position_for_expiry(expiry, is_long_vol=True):
                    long_vol_trade = build_long_put_spread(
                        chain=chain,
                        strategy_config=self.config.strategy,
                        execution_config=self.config.execution,
                        nav=self.portfolio.nav,
                        entry_date=day,
                        entry_spot=entry_spot,
                        entry_vix=entry_vix,
                    )

                    if long_vol_trade is not None:
                        self.portfolio.enter_trade(long_vol_trade)
                        trades_entered_long_vol += 1

                        self.portfolio.trade_log[-1].update({
                            "reason_codes": ["LONG_VOL"] + reject_reason_codes,
                            "vrp_regime": vrp_sig.regime if vrp_sig else None,
                            "vrp_raw": vrp_sig.raw_value if vrp_sig else None,
                            "vrp_zscore": vrp_sig.z_score if vrp_sig else None,
                            "vrp_scalar": vrp_sig.size_scalar if vrp_sig else None,
                            "vrp_confidence": vrp_sig.confidence if vrp_sig else None,
                            "regime": reg_sig.regime if reg_sig else None,
                            "regime_pi_width": reg_sig.raw_value if reg_sig else None,
                            "regime_zscore": reg_sig.z_score if reg_sig else None,
                            "regime_multiplier": reg_sig.size_scalar if reg_sig else None,
                            "composite_scalar": 0.0,
                            "is_long_vol": True,
                        })

            self.portfolio.record_snapshot(day)

        # Final settlement of any remaining positions
        if self.portfolio.open_positions:
            logger.info(f"Settling {len(self.portfolio.open_positions)} remaining positions at end")
            # Use end date for final settlement
            final_settled = self.portfolio.settle_expired(end)
            # For positions that still haven't expired, force-settle at last known spot
            for trade in list(self.portfolio.open_positions):
                spot = self.dm.get_spot(end)
                if spot is None:
                    spot = trade.entry_spot
                trade.spot_at_expiry = spot
                # Manually settle
                from strategy.portfolio import ClosedPosition
                settlement = self.portfolio._compute_expiry_pnl(trade, spot)
                trade_pnl = trade.total_credit + settlement
                self.portfolio.cash += settlement
                closed = ClosedPosition(
                    trade=trade,
                    exit_date=end,
                    spot_at_expiry=spot,
                    pnl=trade_pnl,
                    pnl_pct_nav=trade_pnl / self.portfolio.nav if self.portfolio.nav > 0 else 0.0,
                    exit_reason="BACKTEST_END",
                )
                self.portfolio.closed_positions.append(closed)
            self.portfolio.open_positions = []

        self.portfolio.record_snapshot(end)

        # Summary
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Trades entered (short vol): {trades_entered}")
        logger.info(f"Trades entered (long vol):  {trades_entered_long_vol}")
        logger.info(f"Skipped (signal):         {trades_skipped_signal}")
        logger.info(f"Skipped (no chain):       {trades_skipped_no_chain}")
        logger.info(f"Skipped (existing pos):   {trades_skipped_existing}")
        logger.info(f"Skipped (circuit breaker): {trades_skipped_circuit}")
        logger.info(f"Skipped (build fail):     {trades_skipped_build_fail}")
        logger.info(f"Final NAV:                ${self.portfolio.nav:,.0f}")
        logger.info(f"Total return:             {(self.portfolio.nav / self.config.backtest.initial_capital - 1):.2%}")
        logger.info("=" * 60)

        return self.portfolio