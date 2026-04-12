"""Signal diagnostics: conditional performance, signal attribution, avoided-loss analysis."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from strategy.portfolio import Portfolio

logger = logging.getLogger(__name__)


def signal_diagnostics(portfolio: Portfolio) -> dict:
    """Compute signal diagnostic statistics."""
    closed = portfolio.closed_positions
    trade_log = portfolio.trade_log

    if not closed:
        return {"error": "No closed positions to analyse."}

    results = {}

    # 1. Performance by VRP regime
    regime_data = []
    for c in closed:
        vrp_sig = c.trade.__dict__.get("_vrp_signal", None)
        # We can infer from trade log
        entry_log = [
            t for t in trade_log
            if t.get("action") == "ENTER" and t.get("date") == c.trade.entry_date
        ]
        regime_data.append({
            "pnl": c.pnl,
            "credit": c.trade.total_credit,
            "size_scalar": c.trade.size_scalar,
        })

    # 2. Trades entered vs skipped
    entered = [t for t in trade_log if t.get("action") == "ENTER"]
    skipped = [t for t in trade_log if t.get("action") == "SKIP"]

    results["trades_entered"] = len(entered)
    results["trades_skipped"] = len(skipped)

    # 3. Skip reason breakdown
    skip_reasons = {}
    for s in skipped:
        for code in s.get("reason_codes", []):
            key = code.split("(")[0].strip()
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
    results["skip_reason_breakdown"] = skip_reasons

    # 4. Size scalar distribution
    scalars = [c.trade.size_scalar for c in closed]
    results["size_scalar_stats"] = {
        "mean": np.mean(scalars) if scalars else 0,
        "median": np.median(scalars) if scalars else 0,
        "min": np.min(scalars) if scalars else 0,
        "max": np.max(scalars) if scalars else 0,
    }

    # 5. P&L summary
    pnls = [c.pnl for c in closed]
    results["pnl_stats"] = {
        "total": sum(pnls),
        "mean": np.mean(pnls),
        "median": np.median(pnls),
        "std": np.std(pnls),
        "skew": float(pd.Series(pnls).skew()),
        "kurtosis": float(pd.Series(pnls).kurtosis()),
    }

    # 6. Conditional P&L by size scalar buckets
    scalar_buckets = {"low (0-0.33)": [], "mid (0.33-0.67)": [], "high (0.67-1.0)": []}
    for c in closed:
        s = c.trade.size_scalar
        if s <= 0.33:
            scalar_buckets["low (0-0.33)"].append(c.pnl)
        elif s <= 0.67:
            scalar_buckets["mid (0.33-0.67)"].append(c.pnl)
        else:
            scalar_buckets["high (0.67-1.0)"].append(c.pnl)

    results["pnl_by_scalar_bucket"] = {
        k: {
            "count": len(v),
            "total_pnl": sum(v),
            "avg_pnl": np.mean(v) if v else 0,
            "hit_rate": sum(1 for p in v if p > 0) / len(v) if v else 0,
        }
        for k, v in scalar_buckets.items()
    }

    return results


def print_diagnostics(diag: dict) -> None:
    """Pretty-print signal diagnostics."""
    print("\n" + "=" * 60)
    print("SIGNAL DIAGNOSTICS")
    print("=" * 60)

    print(f"\nTrades entered: {diag.get('trades_entered', 0)}")
    print(f"Trades skipped: {diag.get('trades_skipped', 0)}")

    print("\nSkip reason breakdown:")
    for reason, count in diag.get("skip_reason_breakdown", {}).items():
        print(f"  {reason}: {count}")

    print("\nSize scalar distribution:")
    for k, v in diag.get("size_scalar_stats", {}).items():
        print(f"  {k}: {v:.3f}")

    print("\nP&L statistics:")
    for k, v in diag.get("pnl_stats", {}).items():
        print(f"  {k}: ${v:,.0f}" if k != "skew" and k != "kurtosis" else f"  {k}: {v:.3f}")

    print("\nP&L by size scalar bucket:")
    for bucket, stats in diag.get("pnl_by_scalar_bucket", {}).items():
        print(f"  {bucket}:")
        for k, v in stats.items():
            if k == "hit_rate":
                print(f"    {k}: {v:.1%}")
            elif k == "count":
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: ${v:,.0f}")
