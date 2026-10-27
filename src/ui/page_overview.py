"""Page 1 — Overview.

The page answers one question first ("what does the customer base look like,
and is it retaining?") and only then offers the supporting analysis. All values
come from the same pipeline as every other page; nothing is computed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import customer_metrics, visualizations as viz
from src.ui import components as ui
from src.ui.context import AppContext


def render(ctx: AppContext) -> None:
    summary = ctx.metrics.summary
    orders = ctx.tables["order_values"]
    customers = ctx.metrics.customers

    split = customer_metrics.monthly_new_vs_returning(orders)
    recent = split.iloc[-6:]

    # ------------------------------------------------------------------
    # 1. Four primary metrics
    # ------------------------------------------------------------------
    ui.kpi_strip([
        ui.metric("Eligible customers", ui.fmt_int(summary["n_customers"]),
                  "distinct customer_unique_id",
                  f"{summary['n_orders_per_customer']:.2f} orders per customer on average"),
        ui.metric("Repeat-purchase rate", ui.fmt_pct(summary["repeat_purchase_rate"]),
                  f"{ui.fmt_int(summary['n_repeat_customers'])} of "
                  f"{ui.fmt_int(summary['n_customers'])} customers ordered twice or more",
                  "the central constraint on any retention plan"),
        ui.metric("Eligible orders", ui.fmt_int(summary["n_orders"]),
                  "status delivered, valid purchase date",
                  f"of {ui.fmt_int(ctx.raw_meta['row_counts'].get('orders', 0))} rows in the "
                  f"source orders table"),
        ui.metric("Historical merchandise value", ui.fmt_money(summary["merchandise_value"]),
                  f"freight {ui.fmt_money(summary['freight_value'])} recorded separately",
                  "sum of item prices — revenue-like, not profit"),
    ])
    ui.note(
        f"Observation window <b>{summary['first_order_date']:%d %b %Y}</b> → "
        f"<b>{summary['as_of_date']:%d %b %Y}</b> ({summary['observation_days']:,} days). "
        f"Recency is measured against the last day on which an eligible order was placed, "
        f"not against today."
    )

    # ------------------------------------------------------------------
    # 2. One primary chart
    # ------------------------------------------------------------------
    ui.section_header(
        "Customers acquired versus customers retained",
        "Monthly active customers, split by whether that month contained their first eligible "
        "order. Bars are acquisition; the thin band above them is the repeat business.",
    )
    ui.chart(viz.new_vs_returning(split), height=330, title="")
    ui.note(
        f"Across the last six months of the window, returning customers averaged "
        f"<b>{recent['returning_customers'].mean():,.0f}</b> per month against "
        f"<b>{recent['new_customers'].mean():,.0f}</b> first-time buyers. Read the 2016 bars "
        f"with care: the marketplace was still filling its catalogue, so early months are thin."
    )

    # ------------------------------------------------------------------
    # 3. Key findings
    # ------------------------------------------------------------------
    interval_median = summary.get("median_days_to_second_order", float("nan"))
    last = split.iloc[-1]
    ui.section_header(
        "What the data says",
        "Four findings that survived scrutiny, each with the number it rests on. "
        "They are descriptive statements about this window, not forecasts.",
    )
    ui.findings([
        {
            "statement": "Repeat purchasing is the exception, not the norm",
            "why": f"Any retention programme starts from a base rate of "
                   f"{ui.fmt_pct(summary['repeat_purchase_rate'])}. That small a starting point "
                   f"caps the incremental lift a campaign can realistically produce, which is why "
                   f"the planner later models scenarios instead of promising outcomes.",
            "metric": ui.fmt_pct(summary["repeat_purchase_rate"]),
            "metric_label": "repeat rate",
        },
        {
            "statement": "When a customer does come back, they come back quickly",
            "why": f"{ui.fmt_pct(summary.get('share_gaps_within_30d'))} of observed repeat gaps "
                   f"fall inside 30 days and {ui.fmt_pct(summary.get('share_gaps_within_90d'))} "
                   f"inside 90 days. A second-order window opened early is worth more than one "
                   f"opened at month six.",
            "metric": f"{interval_median:.0f} d" if not np.isnan(interval_median) else "—",
            "metric_label": "median gap to 2nd order",
        },
        {
            "statement": "A minority of customers carries a large share of historical value",
            "why": f"The top 10% of customers account for "
                   f"{ui.fmt_pct(summary.get('top_10pct_share'))} of merchandise value and the "
                   f"top 20% for {ui.fmt_pct(summary.get('top_20pct_share'))} "
                   f"(Gini {summary['gini_customer_value']:.2f}). Concentration is real but not "
                   f"extreme, so a top-decile-only strategy still leaves most revenue untouched.",
            "metric": ui.fmt_pct(summary.get("top_10pct_share")),
            "metric_label": "value held by top 10%",
        },
        {
            "statement": "Growth inside this window is acquisition-led",
            "why": f"In {last['month']:%b %Y}, the final month of the window, "
                   f"{ui.fmt_int(last['new_customers'])} customers bought for the first time and "
                   f"{ui.fmt_int(last['returning_customers'])} were returning. Retention is an "
                   f"untapped lever here rather than an already-optimised one.",
            "metric": f"{recent['new_customers'].mean() / max(recent['returning_customers'].mean(), 1e-9):,.0f}×",
            "metric_label": "new buyers per returning",
        },
    ])
    ui.note(
        "Merchandise value excludes freight and there is no cost, margin or campaign data in "
        "the source, so no profitability claim is made anywhere in this analysis."
    )

    # ------------------------------------------------------------------
    # 4. Supporting analysis
    # ------------------------------------------------------------------
    ui.section_header(
        "Supporting detail",
        "Secondary views: where value sits, how ordering behaves month to month, and why the "
        "order population is smaller than the raw table.",
        eyebrow="Further reading",
    )

    ui.stat_list([
        ui.metric("Average order value", ui.fmt_money(summary["avg_order_value"], digits=2),
                  f"median {ui.fmt_money(summary['median_order_value'], digits=2)}"),
        ui.metric("Orders per customer", f"{summary['n_orders_per_customer']:.2f}"),
        ui.metric("Top 1% of value", ui.fmt_pct(summary.get("top_1pct_share"))),
        ui.metric("Gini of customer value", f"{summary['gini_customer_value']:.3f}",
                  "0 = equal, 1 = one customer holds everything"),
        ui.metric("Median days to 2nd order",
                  f"{interval_median:.0f}" if not np.isnan(interval_median) else "—"),
    ])

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1], gap="large")
    with col1:
        ui.chart(viz.concentration_curve(customers), height=310, title="")
    with col2:
        deciles = customer_metrics.customer_value_deciles(customers).head(5).copy()
        deciles["share_of_value"] = deciles["share_of_value"] * 100
        ui.note("Top five value deciles")
        ui.styled_table(deciles.rename(columns={
            "value_decile": "Decile", "n_customers": "Customers",
            "merchandise_value": "Value", "share_of_value": "Share",
            "mean_orders": "Orders / cust.",
        }), column_config={
            "Value": st.column_config.NumberColumn(format="%.0f"),
            "Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                     max_value=100),
            "Orders / cust.": st.column_config.NumberColumn(format="%.2f"),
        }, height=214)

    with st.expander("Monthly context — orders and average order value", expanded=False):
        monthly = customer_metrics.monthly_trend(orders)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            ui.chart(viz.trend_area(monthly, "n_orders", "Eligible orders per month", "Orders"),
                     height=250)
        with c2:
            ui.chart(viz.trend_area(monthly, "avg_order_value",
                                    "Average order value per month", "R$ per order"), height=250)
        ui.note(
            "AOV is shown as a level, not a target: it mixes category mix, freight policy and "
            "price changes, so a rise is not automatically good news."
        )

    with st.expander(f"Why {ui.fmt_int(summary['n_orders'])} orders, when the source has "
                     f"{ui.fmt_int(ctx.raw_meta['row_counts'].get('orders', 0))}?", expanded=False):
        status_counts = pd.DataFrame(
            list(ctx.raw_meta["status_counts"].items()),
            columns=["order_status", "orders"]).sort_values("orders", ascending=False)
        status_counts["share"] = status_counts["orders"] / status_counts["orders"].sum()
        status_counts["eligible"] = np.where(
            status_counts["order_status"].isin(settings.PRIMARY_ELIGIBLE_STATUSES), "Yes", "No")
        col1, col2 = st.columns([1, 1.2], gap="large")
        with col1:
            ui.styled_table(status_counts.rename(columns={
                "order_status": "Status", "orders": "Orders", "share": "Share",
                "eligible": "Eligible"}), column_config={
                "Share": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=1)})
        with col2:
            ui.callout(
                "A retention KPI should count purchases that actually happened. "
                "<code>canceled</code> and <code>unavailable</code> orders delivered no goods; "
                "in-flight statuses (<code>shipped</code>, <code>processing</code>, …) are only "
                "partially observed and cluster at the end of the window, which would make recent "
                "periods look artificially inactive. The rule is applied identically to every "
                "order in every month.", "plain", "Why only delivered orders")
            ui.note("Change <code>PRIMARY_ELIGIBLE_STATUSES</code> in "
                    "<code>config/settings.py</code> to run the sensitivity variant; every page "
                    "recomputes from the same rule.")

    with st.expander("Export summary metrics", expanded=False):
        export = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
        ui.download_csv(export, "executive_summary.csv")
        ui.note("Aggregates only — no customer-level records.")
