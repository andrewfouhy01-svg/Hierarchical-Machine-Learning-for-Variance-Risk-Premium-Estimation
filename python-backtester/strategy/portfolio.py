"""Portfolio: tracks positions, cash, P&L, and daily snapshots."""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from strategy.trade_builder import SpreadTrade

logger = logging.getLogger(__name__)


@dataclass
class ClosedPosition:
    """A position that has been settled at expiry."""
    trade: SpreadTrade
    exit_date: date
    spot_at_expiry: float
    pnl: float
    pnl_pct_nav: float
    exit_reason: str  # 'EXPIRY'


@dataclass
class PortfolioSnapshot:
    """Daily portfolio state."""
    date: date
    nav: float
    cash: float
    unrealised_pnl: float
    num_open_positions: int


class Portfolio:
    """Manages open positions, cash, and records history."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions: list[SpreadTrade] = []
        self.closed_positions: list[ClosedPosition] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self.trade_log: list[dict] = []
        self._monthly_pnl: float = 0.0
        self._current_month: Optional[int] = None
        self._cooloff_until: Optional[date] = None

    @property
    def nav(self) -> float:
        """Net asset value = cash (credits received are already in cash)."""
        return self.cash

    def enter_trade(self, trade: SpreadTrade) -> None:
        """Record a new trade entry. Credit is received (or debit paid) into cash."""
        self.open_positions.append(trade)
        self.cash += trade.total_credit  # positive for credit, negative for debit
        self.trade_log.append({
            "action": "ENTER",
            "date": trade.entry_date,
            "expiry": trade.expiry,
            "short_strike": trade.short_put_strike,
            "long_strike": trade.long_put_strike,
            "contracts": trade.contracts,
            "credit_received": trade.total_credit,
            "spread_width": trade.spread_width,
            "size_scalar": trade.size_scalar,
            "entry_spot": trade.entry_spot,
            "entry_vix": trade.entry_vix,
            "dte": trade.dte,
            "call_overlay": trade.short_call_strike is not None,
            "is_long_vol": trade.is_long_vol,
        })
        direction = "LONG VOL" if trade.is_long_vol else "SHORT VOL"
        credit_label = "debit" if trade.total_credit < 0 else "credit"
        logger.info(
            f"ENTER [{direction}] {trade.entry_date}: {trade.contracts}x "
            f"{trade.short_put_strike}/{trade.long_put_strike} put spread, "
            f"{credit_label}=${abs(trade.total_credit):.0f}, DTE={trade.dte}"
        )

    def settle_expired(self, current_date: date) -> list[ClosedPosition]:
        """Settle all positions that have expired on or before current_date."""
        still_open = []
        settled = []

        for trade in self.open_positions:
            if trade.expiry <= current_date:
                # Determine spot at expiry
                spot_exp = trade.spot_at_expiry
                if spot_exp is None:
                    logger.warning(
                        f"No expiry spot for trade {trade.entry_date} -> {trade.expiry}. "
                        f"Using entry spot as fallback."
                    )
                    spot_exp = trade.entry_spot

                settlement = self._compute_expiry_pnl(trade, spot_exp)
                trade_pnl = trade.total_credit + settlement  # total P&L = credit received + settlement
                self.cash += settlement  # only settlement hits cash (credit already added at entry)
                pnl_pct = trade_pnl / self.nav if self.nav > 0 else 0.0

                closed = ClosedPosition(
                    trade=trade,
                    exit_date=trade.expiry,
                    spot_at_expiry=spot_exp,
                    pnl=trade_pnl,
                    pnl_pct_nav=pnl_pct,
                    exit_reason="EXPIRY",
                )
                self.closed_positions.append(closed)
                settled.append(closed)

                self.trade_log.append({
                    "action": "EXIT",
                    "date": trade.expiry,
                    "entry_date": trade.entry_date,
                    "short_strike": trade.short_put_strike,
                    "long_strike": trade.long_put_strike,
                    "contracts": trade.contracts,
                    "spot_at_expiry": spot_exp,
                    "credit_received": trade.total_credit,
                    "settlement": settlement,
                    "pnl": trade_pnl,
                    "pnl_pct_nav": pnl_pct,
                    "exit_reason": "EXPIRY",
                })

                logger.info(
                    f"EXIT {trade.expiry}: {trade.contracts}x "
                    f"{trade.short_put_strike}/{trade.long_put_strike}, "
                    f"S_exp={spot_exp:.1f}, P&L=${trade_pnl:.0f} ({pnl_pct:.2%})"
                )
            else:
                still_open.append(trade)

        self.open_positions = still_open
        return settled

    def _compute_expiry_pnl(self, trade: SpreadTrade, spot_exp: float) -> float:
        """
        Compute P&L at expiry for a bull put spread.

        At expiry:
        - Short put payoff (from our perspective as seller): -max(K_short - S, 0) * 100
        - Long put payoff (we own it): +max(K_long - S, 0) * 100

        Credit was already added to cash at entry, so here we compute
        the settlement cost only.
        """
        short_payoff = -max(trade.short_put_strike - spot_exp, 0) * 100 * trade.contracts
        long_payoff = max(trade.long_put_strike - spot_exp, 0) * 100 * trade.contracts

        settlement = short_payoff + long_payoff  # net cash flow at settlement

        # Call overlay settlement if present
        call_settlement = 0.0
        if trade.short_call_strike is not None:
            call_settlement = -max(spot_exp - trade.short_call_strike, 0) * 100 * trade.contracts

        return settlement + call_settlement

    def record_snapshot(self, dt: date) -> None:
        """Record a daily portfolio snapshot."""
        self.snapshots.append(PortfolioSnapshot(
            date=dt,
            nav=self.nav,
            cash=self.cash,
            unrealised_pnl=0.0,  # hold-to-expiry: no unrealised tracking
            num_open_positions=len(self.open_positions),
        ))

    def check_circuit_breaker(
        self, dt: date, monthly_loss_limit_pct: float, cooloff_days: int
    ) -> bool:
        """Check if circuit breaker is active. Returns True if trading is blocked."""
        # Reset monthly P&L tracker on new month
        if self._current_month != dt.month:
            self._current_month = dt.month
            self._monthly_pnl = 0.0

        # Check cooloff
        if self._cooloff_until is not None and dt < self._cooloff_until:
            return True

        return False

    def update_monthly_pnl(self, pnl: float, dt: date, monthly_loss_limit_pct: float, cooloff_days: int) -> None:
        """Update monthly P&L and trigger circuit breaker if needed."""
        from datetime import timedelta
        self._monthly_pnl += pnl
        if self._monthly_pnl < -(self.nav * monthly_loss_limit_pct):
            logger.warning(
                f"CIRCUIT BREAKER triggered on {dt}: monthly P&L = ${self._monthly_pnl:.0f}"
            )
            self._cooloff_until = dt + timedelta(days=cooloff_days)

    def has_position_for_expiry(self, expiry: date, is_long_vol: bool = False) -> bool:
        """Check if there's already an open position of the same direction for this expiry."""
        return any(
            t.expiry == expiry and t.is_long_vol == is_long_vol
            for t in self.open_positions
        )