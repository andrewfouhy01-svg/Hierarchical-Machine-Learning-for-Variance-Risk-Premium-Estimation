"""Monthly decision log: detailed CSV explaining each decision date."""

import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

from strategy.portfolio import Portfolio

logger = logging.getLogger(__name__)


def build_decision_log_df(portfolio: Portfolio) -> pd.DataFrame:
    """
    Build a detailed DataFrame from the trade log with one row per decision date.

    Columns cover signal state, sizing rationale, strike selection, and outcome.
    """
    rows = []
    trade_log = portfolio.trade_log

    # Build a lookup for closed position P&L by entry_date
    closed_lookup = {}
    for c in portfolio.closed_positions:
        closed_lookup[c.trade.entry_date] = c

    for entry in trade_log:
        action = entry.get("action")
        dt = entry.get("date")
        reason_codes = entry.get("reason_codes", [])

        row = {
            "date": dt,
            "action": action,
            "reason_codes": " | ".join(str(r) for r in reason_codes),
            # Signal 1: VRP
            "vrp_regime": entry.get("vrp_regime"),
            "vrp_raw": entry.get("vrp_raw"),
            "vrp_zscore": entry.get("vrp_zscore"),
            "vrp_scalar": entry.get("vrp_scalar"),
            "vrp_confidence": entry.get("vrp_confidence"),
            # Signal 2: Regime
            "regime_label": entry.get("regime"),
            "regime_pi_width": entry.get("regime_pi_width"),
            "regime_zscore": entry.get("regime_zscore"),
            "regime_multiplier": entry.get("regime_multiplier"),
            # Market context
            "spot": entry.get("entry_spot"),
            "vix": entry.get("entry_vix") if entry.get("entry_vix") else entry.get("entry_vix"),
        }

        if action == "ENTER":
            row.update({
                "composite_scalar": entry.get("composite_scalar", entry.get("size_scalar")),
                "expiry": entry.get("expiry"),
                "short_strike": entry.get("short_strike"),
                "long_strike": entry.get("long_strike"),
                "spread_width": entry.get("spread_width"),
                "contracts": entry.get("contracts"),
                "credit_received": entry.get("credit_received"),
                "dte": entry.get("dte"),
                "call_overlay": entry.get("call_overlay"),
            })

            # Attach outcome if available
            closed = closed_lookup.get(dt)
            if closed is not None:
                row["pnl"] = closed.pnl
                row["pnl_pct_nav"] = closed.pnl_pct_nav
                row["spot_at_expiry"] = closed.spot_at_expiry
                row["exit_reason"] = closed.exit_reason
        elif action == "EXIT":
            # EXIT entries are separate; include for completeness but
            # the main narrative is on the ENTER/SKIP rows
            row.update({
                "expiry": entry.get("date"),
                "pnl": entry.get("pnl"),
                "pnl_pct_nav": entry.get("pnl_pct_nav"),
                "spot_at_expiry": entry.get("spot_at_expiry"),
                "exit_reason": entry.get("exit_reason"),
            })

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by date
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def build_monthly_summary_df(portfolio: Portfolio) -> pd.DataFrame:
    """
    Build a monthly summary with narrative explanation for each month.

    Groups all decisions by calendar month and produces a human-readable
    description of what happened and why.
    """
    detail_df = build_decision_log_df(portfolio)
    if detail_df.empty:
        return pd.DataFrame()

    # Filter to ENTER and signal SKIP only (exclude EXIT rows)
    decisions = detail_df[detail_df["action"].isin(["ENTER", "SKIP"])].copy()
    decisions["date"] = pd.to_datetime(decisions["date"])
    decisions["year_month"] = decisions["date"].dt.to_period("M")

    monthly_rows = []
    for period, group in decisions.groupby("year_month"):
        enters = group[group["action"] == "ENTER"]
        skips = group[group["action"] == "SKIP"]

        n_enters = len(enters)
        n_skips = len(skips)

        # Aggregate P&L for the month
        month_pnl = enters["pnl"].sum() if "pnl" in enters.columns else 0.0

        # Build narrative
        narrative_parts = []

        if n_enters > 0:
            for _, e in enters.iterrows():
                parts = [f"ENTERED on {e['date'].strftime('%Y-%m-%d')}:"]
                if pd.notna(e.get("vrp_regime")):
                    parts.append(
                        f"VRP regime={e['vrp_regime']} "
                        f"(raw={e.get('vrp_raw', 'N/A'):.2f}, "
                        f"z={e.get('vrp_zscore', 'N/A'):.2f}, "
                        f"scalar={e.get('vrp_scalar', 'N/A'):.2f})"
                        if pd.notna(e.get("vrp_raw")) else
                        f"VRP regime={e['vrp_regime']}"
                    )
                if pd.notna(e.get("regime_label")):
                    parts.append(
                        f"Regime={e['regime_label']} "
                        f"(PI_width={e.get('regime_pi_width', 'N/A'):.2f}, "
                        f"z={e.get('regime_zscore', 'N/A'):.2f}, "
                        f"mult={e.get('regime_multiplier', 'N/A'):.2f})"
                        if pd.notna(e.get("regime_pi_width")) else
                        f"Regime={e['regime_label']}"
                    )
                if pd.notna(e.get("composite_scalar")):
                    parts.append(f"Composite scalar={e['composite_scalar']:.3f}")
                if pd.notna(e.get("short_strike")):
                    parts.append(
                        f"Strikes: {e['short_strike']:.0f}/{e['long_strike']:.0f} "
                        f"(width={e.get('spread_width', 0):.0f}), "
                        f"{int(e.get('contracts', 0))}x, "
                        f"credit=${e.get('credit_received', 0):,.0f}, "
                        f"DTE={int(e.get('dte', 0))}"
                    )
                if pd.notna(e.get("spot")):
                    parts.append(f"SPX={e['spot']:.1f}, VIX={e.get('vix', 0):.1f}")
                if pd.notna(e.get("pnl")):
                    parts.append(f"Outcome: P&L=${e['pnl']:,.0f} ({e.get('pnl_pct_nav', 0):.2%})")
                if e.get("reason_codes"):
                    parts.append(f"Codes: {e['reason_codes']}")
                narrative_parts.append(" | ".join(parts))

        if n_skips > 0:
            # Group skips by reason
            skip_reasons = defaultdict(int)
            skip_details = []
            for _, s in skips.iterrows():
                codes = s.get("reason_codes", "")
                skip_reasons[codes] += 1
                # Only detail signal-level skips (not chain/existing)
                if pd.notna(s.get("vrp_regime")) or pd.notna(s.get("regime_label")):
                    detail = f"SKIPPED on {s['date'].strftime('%Y-%m-%d')}: "
                    if pd.notna(s.get("vrp_regime")):
                        detail += (
                            f"VRP regime={s['vrp_regime']} "
                            f"(raw={s.get('vrp_raw', 0):.2f}, z={s.get('vrp_zscore', 0):.2f}) "
                        )
                    if pd.notna(s.get("regime_label")):
                        detail += (
                            f"Regime={s['regime_label']} "
                            f"(PI_z={s.get('regime_zscore', 0):.2f}) "
                        )
                    detail += f"Reason: {codes}"
                    skip_details.append(detail)
                else:
                    skip_details.append(
                        f"SKIPPED on {s['date'].strftime('%Y-%m-%d')}: {codes}"
                    )

            for d in skip_details:
                narrative_parts.append(d)

        monthly_rows.append({
            "month": str(period),
            "trades_entered": n_enters,
            "trades_skipped": n_skips,
            "month_pnl": month_pnl if not pd.isna(month_pnl) else 0.0,
            "narrative": "\n".join(narrative_parts),
        })

    return pd.DataFrame(monthly_rows)


def export_decision_log_csv(portfolio: Portfolio, output_path: str = "decision_log.csv") -> str:
    """Export the full per-date decision log as CSV."""
    df = build_decision_log_df(portfolio)
    # Drop EXIT rows for clarity (outcome is already on the ENTER row)
    df = df[df["action"] != "EXIT"].copy()
    df.to_csv(output_path, index=False)
    logger.info(f"Decision log saved to {output_path} ({len(df)} rows)")
    return output_path


def export_monthly_summary_csv(portfolio: Portfolio, output_path: str = "monthly_decision_log.csv") -> str:
    """Export the monthly narrative summary as CSV."""
    df = build_monthly_summary_df(portfolio)
    df.to_csv(output_path, index=False)
    logger.info(f"Monthly decision log saved to {output_path} ({len(df)} rows)")
    return output_path
