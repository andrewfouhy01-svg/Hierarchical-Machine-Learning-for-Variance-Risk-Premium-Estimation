"""Generate interactive HTML tearsheet with Plotly."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.metrics import PerformanceMetrics
from strategy.portfolio import Portfolio
from analytics.decision_log import build_decision_log_df, build_monthly_summary_df

logger = logging.getLogger(__name__)


def generate_tearsheet(
    portfolio: Portfolio,
    metrics: PerformanceMetrics,
    output_path: str = "tearsheet.html",
) -> str:
    """Generate a self-contained HTML tearsheet."""

    # ── Build DataFrames ──────────────────────────────────────────────
    snap_df = pd.DataFrame([
        {"date": s.date, "nav": s.nav} for s in portfolio.snapshots
    ])
    snap_df["date"] = pd.to_datetime(snap_df["date"])
    snap_df = snap_df.drop_duplicates(subset="date", keep="last").sort_values("date")

    trade_df = pd.DataFrame([
        {
            "entry_date": c.trade.entry_date,
            "expiry": c.trade.expiry,
            "short_strike": c.trade.short_put_strike,
            "long_strike": c.trade.long_put_strike,
            "contracts": c.trade.contracts,
            "credit": c.trade.total_credit,
            "pnl": c.pnl,
            "pnl_pct": c.pnl_pct_nav,
            "spot_entry": c.trade.entry_spot,
            "spot_expiry": c.spot_at_expiry,
            "vix_entry": c.trade.entry_vix,
            "dte": c.trade.dte,
            "size_scalar": c.trade.size_scalar,
            "exit_reason": c.exit_reason,
        }
        for c in portfolio.closed_positions
    ])

    # ── Resample NAV to daily for smooth curves ───────────────────────
    nav_ts = snap_df.set_index("date")["nav"]
    nav_daily = nav_ts.resample("D").ffill().dropna()

    # Drawdown series
    cummax = nav_daily.cummax()
    dd_series = (nav_daily - cummax) / cummax

    # Monthly returns for heatmap
    nav_monthly = nav_daily.resample("ME").last()
    monthly_returns = nav_monthly.pct_change().dropna()

    # ── Figure 1: Equity Curve + Drawdown ─────────────────────────────
    fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=["Equity Curve", "Drawdown"],
        vertical_spacing=0.08,
    )
    fig1.add_trace(
        go.Scatter(
            x=nav_daily.index.strftime("%Y-%m-%d").tolist(),
            y=nav_daily.values.tolist(),
            name="NAV", line=dict(color="#2563eb", width=1.5),
        ),
        row=1, col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=dd_series.index.strftime("%Y-%m-%d").tolist(),
            y=dd_series.values.tolist(),
            fill="tozeroy", name="Drawdown",
            line=dict(color="#dc2626", width=1),
            fillcolor="rgba(220,38,38,0.3)",
        ),
        row=2, col=1,
    )
    fig1.update_layout(height=500, showlegend=True, template="plotly_white")
    fig1.update_yaxes(title_text="NAV ($)", row=1, col=1)
    fig1.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)

    # ── Figure 2: Trade P&L Bar Chart ─────────────────────────────────
    fig2 = go.Figure()
    if not trade_df.empty:
        trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
        colors = ["#16a34a" if p > 0 else "#dc2626" for p in trade_df["pnl"]]
        fig2.add_trace(go.Bar(
            x=trade_df["entry_date"].astype(str).tolist(),
            y=trade_df["pnl"].tolist(),
            marker_color=colors, name="Trade P&L",
        ))
    fig2.update_layout(
        height=350, template="plotly_white",
        xaxis_title="Entry Date", yaxis_title="P&L ($)",
    )

    # ── Figure 3: Monthly Returns Heatmap ─────────────────────────────
    if len(monthly_returns) > 0:
        mr_df = monthly_returns.to_frame("ret")
        mr_df["year"] = mr_df.index.year
        mr_df["month"] = mr_df.index.month
        pivot = mr_df.pivot_table(values="ret", index="year", columns="month", aggfunc="sum")
        pivot = pivot.reindex(columns=range(1, 13))

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig3 = go.Figure(data=go.Heatmap(
            z=(pivot.values * 100).tolist(),
            x=month_labels,
            y=[str(y) for y in pivot.index],
            colorscale=[[0, "#dc2626"], [0.5, "#ffffff"], [1, "#16a34a"]],
            zmid=0,
            text=np.where(np.isnan(pivot.values), "",
                          np.char.add(np.char.mod("%.2f", pivot.values * 100), "%")).tolist(),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Return %"),
        ))
        fig3.update_layout(
            height=max(250, len(pivot) * 28 + 100),
            template="plotly_white",
            yaxis=dict(autorange="reversed"),
        )
    else:
        fig3 = go.Figure()
        fig3.update_layout(height=200, template="plotly_white",
                           annotations=[dict(text="No monthly data", showarrow=False)])

    # ── Figure 4: Return Distribution ─────────────────────────────────
    fig4 = go.Figure()
    if not trade_df.empty:
        fig4.add_trace(go.Histogram(
            x=trade_df["pnl"].tolist(), nbinsx=30,
            marker_color="#2563eb", opacity=0.7,
            name="Trade P&L Distribution",
        ))
        fig4.add_vline(x=0, line_dash="dash", line_color="grey")
        fig4.add_vline(x=trade_df["pnl"].mean(), line_dash="dot",
                       line_color="#dc2626", annotation_text="Mean")
    fig4.update_layout(
        height=300, template="plotly_white",
        xaxis_title="P&L ($)", yaxis_title="Count",
    )

    # ── Figure 5: Size Scalar vs P&L ──────────────────────────────────
    fig5 = go.Figure()
    if not trade_df.empty:
        colors5 = ["#16a34a" if p > 0 else "#dc2626" for p in trade_df["pnl"]]
        fig5.add_trace(go.Scatter(
            x=trade_df["size_scalar"].tolist(),
            y=trade_df["pnl"].tolist(),
            mode="markers",
            marker=dict(color=colors5, size=8, opacity=0.7),
            name="Trades",
        ))
        fig5.add_hline(y=0, line_dash="dash", line_color="grey")
    fig5.update_layout(
        height=350, template="plotly_white",
        xaxis_title="Size Scalar", yaxis_title="P&L ($)",
    )

    # ── Figure 6: Rolling 12-month Sharpe ─────────────────────────────
    fig6 = go.Figure()
    if len(monthly_returns) >= 12:
        rolling_mean = monthly_returns.rolling(12).mean()
        rolling_std = monthly_returns.rolling(12).std()
        rolling_sharpe = (rolling_mean * 12) / (rolling_std * np.sqrt(12))
        rolling_sharpe = rolling_sharpe.dropna()
        fig6.add_trace(go.Scatter(
            x=rolling_sharpe.index.strftime("%Y-%m-%d").tolist(),
            y=rolling_sharpe.values.tolist(),
            name="Rolling 12m Sharpe", line=dict(color="#7c3aed", width=1.5),
        ))
        fig6.add_hline(y=0, line_dash="dash", line_color="grey")
    fig6.update_layout(
        height=300, template="plotly_white",
        yaxis_title="Sharpe Ratio",
    )

    # ── Figure 7: VIX at Entry vs P&L ────────────────────────────────
    fig7 = go.Figure()
    if not trade_df.empty and "vix_entry" in trade_df.columns:
        colors7 = ["#16a34a" if p > 0 else "#dc2626" for p in trade_df["pnl"]]
        fig7.add_trace(go.Scatter(
            x=trade_df["vix_entry"].tolist(),
            y=trade_df["pnl"].tolist(),
            mode="markers",
            marker=dict(color=colors7, size=8, opacity=0.7),
            name="Trades",
        ))
        fig7.add_hline(y=0, line_dash="dash", line_color="grey")
    fig7.update_layout(
        height=350, template="plotly_white",
        xaxis_title="VIX at Entry", yaxis_title="P&L ($)",
    )

    # ── Export individual chart PDFs ────────────────────────────────
    charts_dir = Path(output_path).parent / "charts"
    charts_dir.mkdir(exist_ok=True)
    chart_figures = [
        (fig1, "equity_curve_drawdown"),
        (fig2, "trade_pnl"),
        (fig3, "monthly_returns_heatmap"),
        (fig4, "pnl_distribution"),
        (fig5, "size_scalar_vs_pnl"),
        (fig6, "rolling_sharpe"),
        (fig7, "vix_entry_vs_pnl"),
    ]
    for fig, name in chart_figures:
        pdf_path = charts_dir / f"{name}.pdf"
        try:
            fig.write_image(str(pdf_path), format="pdf")
            logger.info(f"Chart PDF saved: {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not export {name}.pdf: {e}")

    # ── Assemble HTML ─────────────────────────────────────────────────
    fig1_html = fig1.to_html(full_html=False, include_plotlyjs=False)
    fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    fig3_html = fig3.to_html(full_html=False, include_plotlyjs=False)
    fig4_html = fig4.to_html(full_html=False, include_plotlyjs=False)
    fig5_html = fig5.to_html(full_html=False, include_plotlyjs=False)
    fig6_html = fig6.to_html(full_html=False, include_plotlyjs=False)
    fig7_html = fig7.to_html(full_html=False, include_plotlyjs=False)

    metrics_html = _metrics_table(metrics)
    trades_html = _trades_table(trade_df) if not trade_df.empty else "<p>No trades.</p>"

    # ── Monthly Decision Log ──────────────────────────────────────────
    monthly_df = build_monthly_summary_df(portfolio)
    monthly_log_html = _monthly_decision_table(monthly_df) if not monthly_df.empty else "<p>No decision data.</p>"

    # ── Per-Date Decision Log ─────────────────────────────────────────
    decision_df = build_decision_log_df(portfolio)
    decision_df = decision_df[decision_df["action"] != "EXIT"]  # exclude EXIT rows
    decision_log_html = _decision_log_table(decision_df) if not decision_df.empty else "<p>No decisions.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VRP Strategy Tearsheet</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px 40px; background: #fafafa; color: #1a1a1a; }}
    h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
    h2 {{ margin-top: 32px; color: #374151; }}
    .metrics-grid {{
        display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 10px; margin: 16px 0;
    }}
    .metric-card {{
        background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 12px; text-align: center;
    }}
    .metric-card .label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; }}
    .metric-card .value {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
    th {{ background: #f3f4f6; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f9fafb; }}
    .section {{ margin-bottom: 32px; }}
</style>
</head>
<body>

<h1>VRP Signal-Gated SPX Options Strategy &mdash; Tearsheet</h1>

<div class="section">
<h2>Performance Summary</h2>
{metrics_html}
</div>

<div class="section">
<h2>Equity Curve &amp; Drawdown</h2>
{fig1_html}
</div>

<div class="section">
<h2>Trade P&amp;L</h2>
{fig2_html}
</div>

<div class="section">
<h2>Monthly Returns Heatmap</h2>
{fig3_html}
</div>

<div class="section">
<h2>P&amp;L Distribution</h2>
{fig4_html}
</div>

<div class="section">
<h2>Size Scalar vs P&amp;L</h2>
{fig5_html}
</div>

<div class="section">
<h2>Rolling 12-Month Sharpe</h2>
{fig6_html}
</div>

<div class="section">
<h2>VIX at Entry vs P&amp;L</h2>
{fig7_html}
</div>

<div class="section">
<h2>Trade Log</h2>
{trades_html}
</div>

<div class="section">
<h2>Monthly Decision Log</h2>
<p style="color:#6b7280; font-size:12px;">Each month's signal state, trade rationale, and outcome. Explains why trades were entered or skipped.</p>
{monthly_log_html}
</div>

<div class="section">
<h2>Per-Date Signal Decisions</h2>
<p style="color:#6b7280; font-size:12px;">Every option chain date evaluated: signal values, regime classifications, sizing, and reason codes.</p>
{decision_log_html}
</div>

<p style="color:#9ca3af; font-size:11px; margin-top:40px;">
    Generated from {len(portfolio.closed_positions)} closed trades.
    All P&amp;L figures in USD. Sharpe computed from monthly calendar returns with rf=0%.
</p>
</body>
</html>"""

    Path(output_path).write_text(html)
    logger.info(f"Tearsheet saved to {output_path}")
    return output_path


def _metrics_table(m: PerformanceMetrics) -> str:
    cards = [
        ("Total Return", f"{m.total_return:.2%}"),
        ("CAGR", f"{m.cagr:.2%}"),
        ("Annualised Vol", f"{m.annualised_vol:.2%}"),
        ("Sharpe Ratio", f"{m.sharpe:.2f}"),
        ("Sortino Ratio", f"{m.sortino:.2f}"),
        ("Calmar Ratio", f"{m.calmar:.2f}"),
        ("Omega Ratio", f"{m.omega:.2f}"),
        ("Max Drawdown", f"{m.max_drawdown:.2%}"),
        ("Max DD Duration", f"{m.max_drawdown_duration_days} days"),
        ("VaR (95%)", f"{m.var_95:.2%}"),
        ("CVaR (95%)", f"{m.cvar_95:.2%}"),
        ("VaR (99%)", f"{m.var_99:.2%}"),
        ("CVaR (99%)", f"{m.cvar_99:.2%}"),
        ("Total Trades", f"{m.total_trades}"),
        ("Hit Rate", f"{m.hit_rate:.1%}"),
        ("Avg Win", f"${m.avg_win:,.0f}"),
        ("Avg Loss", f"${m.avg_loss:,.0f}"),
        ("Profit Factor", f"{m.profit_factor:.2f}"),
        ("Premium Capture", f"{m.premium_capture_ratio:.2%}"),
        ("Avg Monthly Return", f"{m.avg_monthly_return:.2%}"),
        ("Best Month", f"{m.best_month:.2%}"),
        ("Worst Month", f"{m.worst_month:.2%}"),
        ("% Positive Months", f"{m.pct_positive_months:.1%}"),
    ]
    html = '<div class="metrics-grid">'
    for label, value in cards:
        html += (f'<div class="metric-card">'
                 f'<div class="label">{label}</div>'
                 f'<div class="value">{value}</div>'
                 f'</div>')
    html += "</div>"
    return html


def _trades_table(df: pd.DataFrame) -> str:
    cols = ["entry_date", "expiry", "short_strike", "long_strike", "contracts",
            "credit", "pnl", "pnl_pct", "spot_entry", "spot_expiry", "vix_entry",
            "dte", "size_scalar"]
    available = [c for c in cols if c in df.columns]
    sub = df[available].head(200)

    html = '<div style="overflow-x:auto;"><table><thead><tr>'
    for c in available:
        html += f"<th>{c}</th>"
    html += "</tr></thead><tbody>"

    for _, row in sub.iterrows():
        pnl_val = row.get("pnl", 0)
        row_color = ""
        if pnl_val < 0:
            row_color = ' style="background:#fef2f2;"'
        html += f"<tr{row_color}>"
        for c in available:
            val = row[c]
            if isinstance(val, float):
                if c in ("pnl_pct", "size_scalar"):
                    html += f"<td>{val:.3f}</td>"
                elif c in ("credit", "pnl"):
                    html += f"<td>${val:,.0f}</td>"
                else:
                    html += f"<td>{val:,.1f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


def _monthly_decision_table(df: pd.DataFrame) -> str:
    """Render monthly decision summary as an HTML table with narrative cells."""
    html = '<div style="overflow-x:auto;"><table>'
    html += ("<thead><tr>"
             "<th style='min-width:80px;'>Month</th>"
             "<th>Entered</th>"
             "<th>Skipped</th>"
             "<th>Month P&amp;L</th>"
             "<th style='min-width:400px; text-align:left;'>Decision Narrative</th>"
             "</tr></thead><tbody>")

    for _, row in df.iterrows():
        pnl = row.get("month_pnl", 0)
        pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"
        narrative = str(row.get("narrative", "")).replace("\n", "<br>")
        html += (f"<tr>"
                 f"<td>{row['month']}</td>"
                 f"<td style='text-align:center;'>{int(row['trades_entered'])}</td>"
                 f"<td style='text-align:center;'>{int(row['trades_skipped'])}</td>"
                 f"<td style='color:{pnl_color};'>${pnl:,.0f}</td>"
                 f"<td style='text-align:left; font-size:11px; white-space:pre-wrap;'>{narrative}</td>"
                 f"</tr>")

    html += "</tbody></table></div>"
    return html


def _decision_log_table(df: pd.DataFrame) -> str:
    """Render per-date decision log as a scrollable HTML table."""
    cols = [
        ("date", "Date"),
        ("action", "Action"),
        ("vrp_regime", "VRP Regime"),
        ("vrp_raw", "VRP Raw"),
        ("vrp_zscore", "VRP Z"),
        ("vrp_scalar", "VRP Scalar"),
        ("regime_label", "Regime"),
        ("regime_pi_width", "PI Width"),
        ("regime_zscore", "PI Z"),
        ("regime_multiplier", "Regime Mult"),
        ("composite_scalar", "Comp Scalar"),
        ("spot", "SPX"),
        ("vix", "VIX"),
        ("short_strike", "Short K"),
        ("long_strike", "Long K"),
        ("spread_width", "Width"),
        ("contracts", "Contracts"),
        ("credit_received", "Credit"),
        ("dte", "DTE"),
        ("pnl", "P&L"),
        ("reason_codes", "Reason Codes"),
    ]
    available = [(c, label) for c, label in cols if c in df.columns]

    html = '<div style="overflow-x:auto; max-height:600px; overflow-y:auto;"><table>'
    html += "<thead><tr>"
    for _, label in available:
        html += f"<th style='position:sticky; top:0; background:#f3f4f6;'>{label}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        action = row.get("action", "")
        if action == "ENTER":
            row_style = ' style="background:#f0fdf4;"'
        elif action == "SKIP":
            row_style = ' style="background:#fef2f2;"'
        else:
            row_style = ""

        html += f"<tr{row_style}>"
        for col, _ in available:
            val = row.get(col)
            if pd.isna(val) if not isinstance(val, str) else False:
                html += "<td>—</td>"
            elif isinstance(val, float):
                if col in ("pnl", "credit_received"):
                    html += f"<td>${val:,.0f}</td>"
                elif col in ("vrp_zscore", "regime_zscore", "vrp_scalar",
                             "regime_multiplier", "composite_scalar"):
                    html += f"<td>{val:.3f}</td>"
                elif col in ("vrp_raw", "regime_pi_width"):
                    html += f"<td>{val:.2f}</td>"
                else:
                    html += f"<td>{val:,.1f}</td>"
            else:
                html += f"<td style='font-size:11px;'>{val}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html