"""Performance metrics: Sharpe, Sortino, Calmar, max drawdown, VaR, etc."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    # Returns
    total_return: float
    cagr: float
    annualised_vol: float
    # Risk-adjusted
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    # Tail risk
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    premium_capture_ratio: float
    # Summary
    avg_monthly_return: float
    best_month: float
    worst_month: float
    pct_positive_months: float


def compute_metrics(
    portfolio,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute full performance metrics from a completed portfolio.

    Notes on risk_free_rate:
        Default is 0.0 because this backtest does NOT accrue interest on
        uninvested capital.  Subtracting a positive rf would double-count
        opportunity cost.  If you add cash accrual to the NAV, set this
        to the appropriate annualised rate (e.g. 0.04).
    """
    snapshots = portfolio.snapshots
    closed = portfolio.closed_positions

    if len(snapshots) < 2:
        return _empty_metrics()

    # ── Build NAV series ──────────────────────────────────────────────
    nav_series = pd.DataFrame([
        {"date": s.date, "nav": s.nav} for s in snapshots
    ])
    nav_series["date"] = pd.to_datetime(nav_series["date"])
    nav_series = nav_series.drop_duplicates(subset="date", keep="last")
    nav_series = nav_series.sort_values("date").reset_index(drop=True)

    initial_nav = portfolio.initial_capital
    final_nav = nav_series["nav"].iloc[-1]
    total_return = final_nav / initial_nav - 1

    # Time span
    days = (nav_series["date"].iloc[-1] - nav_series["date"].iloc[0]).days
    years = days / 365.25 if days > 0 else 1.0

    cagr = (final_nav / initial_nav) ** (1 / years) - 1 if years > 0 else 0.0

    # ── Monthly calendar returns (proper basis for Sharpe) ────────────
    # Resample NAV to month-end, forward-filling between snapshots
    nav_ts = nav_series.set_index("date")["nav"]
    nav_daily = nav_ts.resample("D").ffill()
    nav_monthly = nav_daily.resample("ME").last()
    monthly_returns = nav_monthly.pct_change().dropna()

    if len(monthly_returns) < 2:
        ann_vol = 0.0
        sharpe = 0.0
        sortino = 0.0
    else:
        monthly_std = monthly_returns.std()
        ann_vol = monthly_std * np.sqrt(12)

        # Sharpe = CAGR / annualised_vol  (no rf subtracted)
        sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

        # Sortino: use downside deviation only
        downside = monthly_returns[monthly_returns < 0]
        if len(downside) > 1:
            downside_std = downside.std() * np.sqrt(12)
            sortino = cagr / downside_std if downside_std > 0 else 0.0
        else:
            sortino = float("inf") if cagr > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────
    navs = nav_series["nav"].values
    dates = nav_series["date"].values
    running_max = np.maximum.accumulate(navs)
    drawdowns = (navs - running_max) / running_max
    max_dd = drawdowns.min()
    dd_duration = _max_drawdown_duration(navs, dates)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("inf")

    # ── VaR / CVaR (from monthly returns) ─────────────────────────────
    if len(monthly_returns) > 5:
        var_95 = float(np.percentile(monthly_returns, 5))
        var_99 = float(np.percentile(monthly_returns, 1))
        cvar_95 = float(monthly_returns[monthly_returns <= var_95].mean())
        cvar_99 = float(monthly_returns[monthly_returns <= var_99].mean())
    else:
        var_95 = var_99 = cvar_95 = cvar_99 = 0.0

    # ── Omega ratio ───────────────────────────────────────────────────
    threshold = risk_free_rate / 12  # monthly rf
    gains = monthly_returns[monthly_returns > threshold] - threshold
    losses = threshold - monthly_returns[monthly_returns <= threshold]
    omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else float("inf")

    # ── Trade statistics ──────────────────────────────────────────────
    pnls = [c.pnl for c in closed]
    total_trades = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses_list = [p for p in pnls if p <= 0]

    hit_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses_list) if losses_list else 0.0

    gross_wins = sum(wins)
    gross_losses = abs(sum(losses_list))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Premium capture ratio: (total credit + total P&L) / total credit
    total_credit = sum(c.trade.total_credit for c in closed)
    total_pnl = sum(pnls)
    premium_capture = (total_pnl + total_credit) / total_credit if total_credit > 0 else 0.0

    # ── Monthly summary ───────────────────────────────────────────────
    avg_monthly = float(monthly_returns.mean()) if len(monthly_returns) > 0 else 0.0
    best_month = float(monthly_returns.max()) if len(monthly_returns) > 0 else 0.0
    worst_month = float(monthly_returns.min()) if len(monthly_returns) > 0 else 0.0
    pct_pos_months = float((monthly_returns > 0).mean()) if len(monthly_returns) > 0 else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualised_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        omega=omega,
        max_drawdown=max_dd,
        max_drawdown_duration_days=dd_duration,
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses_list),
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        premium_capture_ratio=premium_capture,
        avg_monthly_return=avg_monthly,
        best_month=best_month,
        worst_month=worst_month,
        pct_positive_months=pct_pos_months,
    )


def _max_drawdown_duration(navs, dates) -> int:
    """Compute max drawdown duration in days."""
    peak = navs[0]
    max_dur = 0
    peak_date = dates[0]
    for i in range(1, len(navs)):
        if navs[i] >= peak:
            peak = navs[i]
            peak_date = dates[i]
        else:
            dur = (pd.Timestamp(dates[i]) - pd.Timestamp(peak_date)).days
            max_dur = max(max_dur, dur)
    return max_dur


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0, cagr=0, annualised_vol=0,
        sharpe=0, sortino=0, calmar=0, omega=0,
        max_drawdown=0, max_drawdown_duration_days=0,
        var_95=0, cvar_95=0, var_99=0, cvar_99=0,
        total_trades=0, winning_trades=0, losing_trades=0,
        hit_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
        premium_capture_ratio=0,
        avg_monthly_return=0, best_month=0, worst_month=0,
        pct_positive_months=0,
    )
