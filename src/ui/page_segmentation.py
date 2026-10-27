"""Page 3 — Customers (RFM segmentation).

The page leads with the two things a marketing manager actually uses: how the
base splits (chart) and what each segment looks like (table). The controls that
recompute the segmentation sit behind one settings control, and the segment
comparison charts live in an expander because the table already carries the
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import segmentation as sg
from src import visualizations as viz
from src.ui import components as ui
from src.ui.context import AppContext

SEGMENT_EXPLANATIONS = {
    "High-Value Active Repeat":
        "≥2 eligible orders, recent within the lapsing threshold, and historical value at or above "
        "the 80th percentile of the customer base.",
    "Active Repeat":
        "≥2 eligible orders and recent within the lapsing threshold, but below the high-value cut-off.",
    "Lapsing Repeat":
        "≥2 eligible orders, last order between the lapsing and inactive thresholds.",
    "Recent One-Time Buyer":
        "Exactly one eligible order, placed within the lapsing threshold.",
    "Dormant One-Time Buyer":
        "Exactly one eligible order, last order between the lapsing and inactive thresholds.",
    "Long Inactive":
        "No eligible order for longer than the inactive threshold, regardless of order count.",
    "New / Insufficient History":
        "First eligible order less than "
        f"{settings.INSUFFICIENT_HISTORY_DAYS} days before the analysis date and no second order — "
        "there has not been a fair chance to repeat.",
}

SETTING_KEYS = ("seg_lapsing", "seg_inactive", "seg_hvq")


def render(ctx: AppContext) -> None:
    ui.apply_pending_reset()          # must run before any widget is created
    thresholds = ctx.metrics.thresholds
    customers = ctx.metrics.customers

    # ---------------- settings (collapsed, affects every number below) ------
    c1, c2 = st.columns([4, 1.1], gap="small")
    with c1:
        ui.note("Rule-based RFM. Default thresholds are derived from observed repurchase gaps, "
                "not assumed.")
    with c2:
        with ui.filters("Settings", "custom" if any(
                k in st.session_state for k in SETTING_KEYS) else ""):
            st.session_state.setdefault("method", "rule_based")
            st.session_state.setdefault("seg_lapsing", int(round(thresholds["lapsing_days"])))
            st.session_state.setdefault("seg_inactive", int(round(thresholds["inactive_days"])))
            st.session_state.setdefault("seg_hvq", int(settings.HIGH_VALUE_QUANTILE * 100))
            st.radio(
                "Method", ["rule_based", "kmeans"], horizontal=True, key="method",
                format_func=lambda m: "Rule-based RFM" if m == "rule_based" else "K-means",
                help="K-means finds clusters by shape; the rules stay explainable in one sentence "
                     "per segment.")
            st.slider("Lapsing threshold (days since last order)", 30, 365, step=5,
                      key="seg_lapsing")
            st.slider("Inactive threshold (days since last order)", 60, 600, step=10,
                      key="seg_inactive")
            st.slider("High-value cut-off (value percentile)", 50, 99, step=1, key="seg_hvq")
            ui.note(f"Defaults: lapsing {thresholds['gap_p75_days']:.0f} d (p75 repeat gap, "
                    f"floor 90), inactive {thresholds['gap_p90_days']:.0f} d (p90, floor 180).")
            ui.reset_button({
                "method": "rule_based",
                "seg_lapsing": int(round(thresholds["lapsing_days"])),
                "seg_inactive": int(round(thresholds["inactive_days"])),
                "seg_hvq": int(settings.HIGH_VALUE_QUANTILE * 100),
            })

    method = st.session_state.get("method", "rule_based")
    lapsing = st.session_state.get("seg_lapsing", int(round(thresholds["lapsing_days"])))
    inactive = st.session_state.get("seg_inactive", int(round(thresholds["inactive_days"])))
    hvq = st.session_state.get("seg_hvq", int(settings.HIGH_VALUE_QUANTILE * 100))

    if inactive <= lapsing:
        ui.callout("The inactive threshold must be larger than the lapsing threshold. "
                   "Using the data-derived defaults instead.", "risk", "Invalid configuration")
        lapsing, inactive = int(thresholds["lapsing_days"]), int(thresholds["inactive_days"])

    if method == "kmeans":
        _render_kmeans(sg.run_segmentation(customers, method="kmeans"))
        return

    result = sg.run_segmentation(customers, method="rule_based", lapsing_days=lapsing,
                                 inactive_days=inactive, high_value_quantile=hvq / 100)
    data = result.data
    profile = result.profile

    # ---------------- primary metrics ----------------
    high_value = profile[profile["segment"] == "High-Value Active Repeat"]
    hv_share_value = float(high_value["share_value"].iloc[0]) if len(high_value) else np.nan
    biggest = profile.sort_values("n_customers", ascending=False).iloc[0]
    actionable = profile[profile["segment"].isin(
        ["High-Value Active Repeat", "Active Repeat", "Lapsing Repeat", "Recent One-Time Buyer"])]

    ui.kpi_strip([
        ui.metric("Actionable base",
                  ui.fmt_pct(actionable["n_customers"].sum() / profile["n_customers"].sum()),
                  f"{ui.fmt_int(actionable['n_customers'].sum())} customers",
                  "active repeaters, lapsing repeaters and recent one-time buyers"),
        ui.metric("Value held by high-value repeaters", ui.fmt_pct(hv_share_value),
                  f"{ui.fmt_int(high_value['n_customers'].iloc[0]) if len(high_value) else 0} "
                  f"customers", "share of historical merchandise value"),
        ui.metric("Largest segment", f"{biggest['segment']}",
                  f"{ui.fmt_pct(biggest['share_customers'])} of customers",
                  f"{ui.fmt_int(biggest['n_customers'])} customers"),
        ui.metric("Segments", f"{len(profile)}",
                  f"lapsing {lapsing} d · inactive {inactive} d · high value p{hvq}",
                  "mutually exclusive and exhaustive"),
    ])

    # ---------------- distribution ----------------
    ui.section_header(
        "Segments: customers versus value",
        "A segment that is large but holds little value is not where retention budget belongs.",
    )
    ui.chart(viz.segment_value_chart(profile), height=360, title="")

    # ---------------- profile table ----------------
    ui.section_header(
        "Segment profile",
        "Every number computed from the loaded dataset with the thresholds in Settings.",
    )
    table = profile[[
        "segment", "n_customers", "share_customers", "n_orders", "monetary_value", "share_value",
        "avg_recency_days", "avg_frequency", "avg_order_value", "share_repeat",
    ]].copy().rename(columns={
        "segment": "Segment", "n_customers": "Customers", "share_customers": "Share of customers",
        "n_orders": "Orders", "monetary_value": "Merchandise value",
        "share_value": "Share of value", "avg_recency_days": "Avg recency (days)",
        "avg_frequency": "Avg orders / customer", "avg_order_value": "Avg order value",
        "share_repeat": "Share with ≥2 orders",
    })
    table = ui.pct(table, ["Share of customers", "Share of value", "Share with ≥2 orders"])
    ui.styled_table(table, column_config={
        "Customers": st.column_config.NumberColumn(format="%d"),
        "Share of customers": st.column_config.NumberColumn(format="%.2f%%"),
        "Orders": st.column_config.NumberColumn(format="%d"),
        "Merchandise value": st.column_config.NumberColumn(format="R$ %.0f"),
        "Share of value": st.column_config.NumberColumn(format="%.2f%%"),
        "Avg recency (days)": st.column_config.NumberColumn(format="%.0f"),
        "Avg orders / customer": st.column_config.NumberColumn(format="%.2f"),
        "Avg order value": st.column_config.NumberColumn(format="R$ %.2f"),
        "Share with ≥2 orders": st.column_config.NumberColumn(format="%.1f%%"),
    }, height=270)
    ui.download_csv(table, "segment_profile.csv", "Download segment profile (CSV)")

    # ---------------- recency / value landscape ----------------
    ui.section_header(
        "Recency versus value",
        "Each dot is one customer. The vertical axis compresses at the top; the point is the "
        "overlap between segments, not the individual dots.",
    )
    ui.chart(viz.segment_scatter(data, lapsing_days=lapsing, inactive_days=inactive),
             height=380, title="")
    ui.note("A random sample of up to 6,000 customers is plotted for readability; the tables use "
            "the full population.")

    with st.expander("Segment comparison charts", expanded=False):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            ui.chart(viz.segment_metric_bars(
                profile, "avg_recency_days", "Average recency by segment",
                "Days since last order"), height=300)
            ui.chart(viz.segment_metric_bars(
                profile, "avg_merchandise_per_customer",
                "Historical merchandise value per customer", "R$ per customer"), height=300)
        with c2:
            ui.chart(viz.segment_metric_bars(
                profile, "avg_frequency", "Average orders per customer by segment", "Orders"),
                height=300)
            ui.chart(viz.segment_metric_bars(
                profile, "avg_order_value", "Average order value by segment", "R$ per order"),
                height=300)

    # ---------------- drill-down ----------------
    ui.section_header(
        "Customer drill-down",
        "Pseudonymous records for one segment. The label is a stable hash of "
        "<code>customer_unique_id</code>; raw identifiers are never displayed or exported.",
    )
    seg_choice = st.selectbox("Segment", profile["segment"].tolist())
    sub = data[data["segment"] == seg_choice].copy()
    if seg_choice in SEGMENT_EXPLANATIONS:
        ui.definition("Segment rule", SEGMENT_EXPLANATIONS[seg_choice])

    f1, f2 = st.columns([1.2, 1], gap="small")
    with f2:
        with ui.filters("Filters", "custom" if (st.session_state.get("seg_states")
                                                or st.session_state.get("seg_sort") not in
                                                (None, "monetary_value")) else ""):
            if "customer_state" in sub:
                st.session_state.setdefault("seg_states", [])
                st.multiselect("Customer state",
                               sorted(sub["customer_state"].dropna().unique()),
                               key="seg_states")
            st.session_state.setdefault("seg_sort", "monetary_value")
            st.selectbox(
                "Sort by", ["monetary_value", "recency_days", "frequency",
                            "days_to_second_order"],
                format_func=lambda c: {
                    "monetary_value": "Historical value (high → low)",
                    "recency_days": "Recency (oldest → newest)",
                    "frequency": "Order count",
                    "days_to_second_order": "Days to second order",
                }.get(c, c), key="seg_sort")
            ui.reset_button({"seg_states": [], "seg_sort": "monetary_value"})
    if st.session_state.get("seg_states"):
        sub = sub[sub["customer_state"].isin(st.session_state["seg_states"])]
    sort_by = st.session_state.get("seg_sort", "monetary_value")
    ascending = sort_by not in ("monetary_value", "frequency")
    sub = sub.sort_values(sort_by, ascending=ascending, na_position="last")

    view = pd.DataFrame({
        "Customer": sub["customer_unique_id"].map(ui.anonymize_id),
        "State": sub.get("customer_state"),
        "Orders": sub["frequency"].astype(int),
        "Historical value (R$)": sub["monetary_value"].round(2),
        "Recency (days)": sub["recency_days"].astype("Int64"),
        "Avg order value (R$)": sub["avg_order_value"].round(2),
        "Days to 2nd order": (sub["days_to_second_order"].round(0)
                              if "days_to_second_order" in sub else None),
        "Mean review score": (sub["mean_review_score"].round(2)
                              if "mean_review_score" in sub else None),
        "RFM cell": sub["rfm_cell"],
    }).dropna(axis=1, how="all")

    ui.styled_table(view.head(500), height=360)
    ui.note(f"Showing the first 500 of {ui.fmt_int(len(sub))} customers in this segment after "
            f"filters. The export contains up to 5,000 pseudonymous rows.")
    ui.download_csv(view.head(5000), f"segment_{seg_choice.replace(' ', '_').lower()}.csv",
                    "Download segment customers (anonymised CSV)")

    # ---------------- definitions ----------------
    with st.expander("Segment definitions and limitations", expanded=False):
        for name in profile["segment"]:
            if name in SEGMENT_EXPLANATIONS:
                ui.definition(name, SEGMENT_EXPLANATIONS[name])
        ui.callout(
            "<b>Monetary value is historical merchandise revenue</b>, not profit and not customer "
            "lifetime value. The dataset contains no costs, margins, campaign exposure or "
            "acquisition spend, so no profitability or CLV claim can be made from these segments. "
            "They are <i>proposed analytical categories</i> created for this case study — not "
            "Olist's official customer classifications, and Olist has not reviewed them.",
            "caution", "What these segments are not",
        )
        ui.callout(
            "Segments are a snapshot at the analysis date. They do not model movement between "
            "segments over time, and the thresholds are judgement calls anchored on observed "
            "repurchase gaps. A customer one day either side of a threshold can land in a different "
            "segment, so treat boundaries as soft.", "plain", "Snapshot, not a forecast",
        )

    with st.expander("RFM score grids (1 = weakest, 5 = strongest)", expanded=False):
        grid = pd.crosstab(data["r_score"], data["f_score"])
        st.markdown("**Customer count by recency score × frequency score**")
        st.dataframe(grid, width="stretch")
        ui.note(
            "Scores are quantile bins within this dataset: R is reversed so 5 = most recent. "
            "Because frequency is 1 for ~97% of customers, the frequency score is dominated by a "
            "single bin — which is why the rule-based segments use order counts directly instead "
            "of relying on the F score."
        )


def _render_kmeans(result) -> None:
    """K-means extension view."""
    profile = result.profile
    meta = result.metadata

    ui.kpi_strip([
        ui.metric("Clusters selected", str(meta.get("best_k", "—")),
                  "chosen by highest silhouette score"),
        ui.metric("Silhouette", f"{meta.get('best_silhouette', float('nan')):.3f}",
                  "on a 30k sample", "0.5+ is strong structure; below 0.25 suggests weak separation"),
        ui.metric("Customers clustered", ui.fmt_int(int(profile["n_customers"].sum())),
                  "every customer assigned to a cluster"),
    ])

    ui.section_header("How many clusters?", "Silhouette score by k, on the sample the model was "
                                            "fitted with.")
    ui.chart(viz.style(_silhouette_fig(meta["silhouette"]),
                       "Silhouette score by number of clusters", height=300,
                       x_title="k", y_title="Silhouette"), height=300)

    ui.section_header("Cluster profile", "Read each cluster before naming it.")
    table = profile.rename(columns={
        "cluster": "Cluster", "n_customers": "Customers", "mean_recency": "Mean recency (days)",
        "mean_frequency": "Mean orders", "mean_monetary": "Mean historical value",
        "total_monetary": "Total value", "share_customers": "Share of customers",
        "share_value": "Share of value",
    })
    table = ui.pct(table, ["Share of customers", "Share of value"])
    ui.styled_table(table, column_config={
        "Share of customers": st.column_config.NumberColumn(format="%.2f%%"),
        "Share of value": st.column_config.NumberColumn(format="%.2f%%"),
        "Mean historical value": st.column_config.NumberColumn(format="R$ %.2f"),
        "Total value": st.column_config.NumberColumn(format="R$ %.0f"),
    })

    ui.callout(
        "K-means clusters RFM <i>shape</i>; it does not know what is commercially useful. Read each "
        "cluster's profile before naming it, and treat small clusters with distinctive profiles as "
        "candidates for investigation rather than audiences in their own right. Because frequency "
        "is almost always 1 in this dataset, k-means mostly separates customers by recency and "
        "value.", "caution", "How to read this")

    ui.download_csv(table, "kmeans_clusters.csv", "Download cluster profile (CSV)")


def _silhouette_fig(df: pd.DataFrame):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=df["k"], y=df["silhouette"], mode="lines+markers",
                               line=dict(width=2, color=settings.PALETTE["primary"]),
                               marker=dict(size=6)))
    return fig
