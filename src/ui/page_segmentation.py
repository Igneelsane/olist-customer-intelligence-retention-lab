"""Page 3 — Customer Segmentation (RFM)."""

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


def render(ctx: AppContext) -> None:
    summary = ctx.metrics.summary
    thresholds = ctx.metrics.thresholds
    customers = ctx.metrics.customers

    ui.section_header(
        "Customer segmentation",
        "An interpretable RFM model: Recency, Frequency and historical Monetary value, turned into "
        "segments a marketing manager can act on — and argue with.",
        eyebrow="Page 3 of 6",
    )

    # ---------------- Configurable thresholds ----------------
    with st.expander("⚙︎ Segmentation settings (changes recompute every number on this page)",
                     expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            lapsing = st.slider("Lapsing threshold (days since last order)",
                                30, 365, int(round(thresholds["lapsing_days"])), 5)
        with c2:
            inactive = st.slider("Inactive threshold (days since last order)",
                                 60, 600, int(round(thresholds["inactive_days"])), 10)
        with c3:
            hvq = st.slider("High-value cut-off (value percentile)", 50, 99,
                            int(settings.HIGH_VALUE_QUANTILE * 100), 1)
        if inactive <= lapsing:
            ui.callout("The inactive threshold must be larger than the lapsing threshold. "
                       "Using the data-derived defaults instead.", "danger", "Invalid configuration")
            lapsing, inactive = int(thresholds["lapsing_days"]), int(thresholds["inactive_days"])
        method = st.radio("Method", ["rule_based", "kmeans"], horizontal=True,
                          format_func=lambda m: "Rule-based RFM (default)" if m == "rule_based"
                          else "K-means extension",
                          help="K-means finds clusters by shape; the rules stay explainable in one "
                               "sentence per segment.")
        ui.note(
            f"Defaults come from the observed data: lapsing = max(90, p75 repeat gap) = "
            f"{thresholds['gap_p75_days']:.0f} days, inactive = max(180, p90 repeat gap) = "
            f"{thresholds['gap_p90_days']:.0f} days."
        )

    if method == "kmeans":
        result = sg.run_segmentation(customers, method="kmeans")
        _render_kmeans(result)
        return

    result = sg.run_segmentation(customers, method="rule_based", lapsing_days=lapsing,
                                 inactive_days=inactive, high_value_quantile=hvq / 100)
    data = result.data
    profile = result.profile

    # ---------------- KPI row ----------------
    high_value = profile[profile["segment"] == "High-Value Active Repeat"]
    hv_share_value = float(high_value["share_value"].iloc[0]) if len(high_value) else np.nan
    biggest = profile.sort_values("n_customers", ascending=False).iloc[0]
    actionable = profile[profile["segment"].isin(
        ["High-Value Active Repeat", "Active Repeat", "Lapsing Repeat", "Recent One-Time Buyer"])]

    ui.kpi_row([
        ui.kpi_card("Segments", f"{len(profile)}", "mutually exclusive, exhaustive",
                    "Every customer is assigned exactly one segment.", "primary"),
        ui.kpi_card("Largest segment", biggest["segment"].replace(" ", "\u00a0"),
                    f"{ui.fmt_pct(biggest['share_customers'])} of customers",
                    "Largest segment by customer count.", "violet"),
        ui.kpi_card("Value held by high-value repeaters", ui.fmt_pct(hv_share_value),
                    f"{ui.fmt_int(high_value['n_customers'].iloc[0]) if len(high_value) else 0} "
                    f"customers",
                    "Share of historical merchandise value.", "teal"),
        ui.kpi_card("Priority-actionable base",
                    ui.fmt_pct(actionable["n_customers"].sum() / profile["n_customers"].sum()),
                    f"{ui.fmt_int(actionable['n_customers'].sum())} customers",
                    "Active repeaters, lapsing repeaters and recent one-time buyers combined.",
                    "amber"),
    ], per_row=4)

    # ---------------- Distribution ----------------
    ui.section_header(
        "Segment distribution: customers versus value",
        "A segment that is large but holds little value is not where retention budget belongs.",
        eyebrow="Distribution",
    )
    st.plotly_chart(viz.segment_value_chart(profile), width="stretch",
                    config={"displayModeBar": False})

    # ---------------- Profile table ----------------
    ui.section_header(
        "Segment profile",
        "Everything below is computed from the loaded dataset with the thresholds shown above.",
        eyebrow="Profile",
    )
    table = profile[[
        "segment", "n_customers", "share_customers", "n_orders", "monetary_value", "share_value",
        "avg_recency_days", "avg_frequency", "avg_order_value", "share_repeat",
    ]].copy()
    table = table.rename(columns={
        "segment": "Segment", "n_customers": "Customers", "share_customers": "Share of customers",
        "n_orders": "Orders", "monetary_value": "Merchandise value",
        "share_value": "Share of value", "avg_recency_days": "Avg recency (days)",
        "avg_frequency": "Avg orders / customer", "avg_order_value": "Avg order value",
        "share_repeat": "Share with ≥2 orders",
    })
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
    })
    ui.download_csv(table, "segment_profile.csv", "Download segment profile (CSV)")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.plotly_chart(
            viz.segment_metric_bars(profile, "avg_recency_days",
                                    "Average recency by segment", "Days since last order"),
            width="stretch", config={"displayModeBar": False})
    with col2:
        st.plotly_chart(
            viz.segment_metric_bars(profile, "avg_frequency",
                                    "Average orders per customer by segment", "Orders"),
            width="stretch", config={"displayModeBar": False})

    ui.section_header("Where the value sits", "Historical value per customer by segment, and the "
                                              "recency–value landscape.", eyebrow="Comparison")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.plotly_chart(
            viz.segment_metric_bars(profile, "avg_merchandise_per_customer",
                                    "Historical merchandise value per customer",
                                    "R$ per customer"),
            width="stretch", config={"displayModeBar": False})
    with col2:
        st.plotly_chart(
            viz.segment_metric_bars(profile, "avg_order_value", "Average order value by segment",
                                    "R$ per order"),
            width="stretch", config={"displayModeBar": False})

    st.plotly_chart(viz.segment_scatter(data), width="stretch",
                    config={"displayModeBar": False})
    ui.note("A random sample of up to 6,000 customers is plotted for readability; the tables use "
            "the full population.")

    # ---------------- Drill-down ----------------
    ui.section_header(
        "Customer drill-down",
        "Pseudonymous records for the selected segment. Raw identifiers are never displayed — the "
        "label is a stable hash of <code>customer_unique_id</code>.",
        eyebrow="Drill-down",
    )
    seg_choice = st.selectbox("Segment", profile["segment"].tolist())
    sub = data[data["segment"] == seg_choice].copy()

    if seg_choice in SEGMENT_EXPLANATIONS:
        ui.definition("Segment rule", SEGMENT_EXPLANATIONS[seg_choice])

    state_cols = [c for c in ("customer_state",) if c in sub.columns]
    f1, f2 = st.columns([1, 1])
    with f1:
        if state_cols:
            states = st.multiselect("Customer state", sorted(sub["customer_state"].dropna().unique()),
                                    default=[])
            if states:
                sub = sub[sub["customer_state"].isin(states)]
    with f2:
        sort_by = st.selectbox(
            "Sort by", ["monetary_value", "recency_days", "frequency", "days_to_second_order"],
            format_func=lambda c: {
                "monetary_value": "Historical value (high → low)",
                "recency_days": "Recency (oldest → newest)",
                "frequency": "Order count",
                "days_to_second_order": "Days to second order",
            }.get(c, c),
        )

    ascending = sort_by not in ("monetary_value", "frequency")
    sub = sub.sort_values(sort_by, ascending=ascending, na_position="last")

    view = pd.DataFrame({
        "Customer": sub["customer_unique_id"].map(ui.anonymize_id),
        "State": sub.get("customer_state"),
        "Orders": sub["frequency"].astype(int),
        "Historical value (R$)": sub["monetary_value"].round(2),
        "Recency (days)": sub["recency_days"].astype("Int64"),
        "Avg order value (R$)": sub["avg_order_value"].round(2),
        "Days to 2nd order": sub["days_to_second_order"].round(0) if "days_to_second_order" in sub else None,
        "Mean review score": sub["mean_review_score"].round(2) if "mean_review_score" in sub else None,
        "RFM cell": sub["rfm_cell"],
    })
    view = view.dropna(axis=1, how="all")

    ui.styled_table(view.head(500), height=380)
    ui.note(f"Showing the first 500 of {ui.fmt_int(len(sub))} customers in this segment "
            f"(after filters). Export includes up to 5,000 pseudonymous rows.")
    ui.download_csv(view.head(5000), f"segment_{seg_choice.replace(' ', '_').lower()}.csv",
                    "Download segment customers (anonymised CSV)")

    # ---------------- RFM scoring ----------------
    with st.expander("RFM score grids (1 = weakest, 5 = strongest)", expanded=False):
        grid = pd.crosstab(data["r_score"], data["f_score"])
        st.markdown("**Customer count by recency score × frequency score**")
        st.dataframe(grid, width="stretch")
        ui.note(
            "Scores are quantile bins within this dataset: R is reversed so 5 = most recent. "
            "Because frequency is 1 for ~97% of customers, the frequency score is dominated by a "
            "single bin — which is exactly why the rule-based segments use order counts directly "
            "rather than relying on the F score."
        )

    # ---------------- Definitions and limitations ----------------
    ui.section_header("Definitions and limitations", "", eyebrow="Method")
    for name in profile["segment"]:
        if name in SEGMENT_EXPLANATIONS:
            ui.definition(name, SEGMENT_EXPLANATIONS[name])

    ui.callout(
        "<b>Monetary value is historical merchandise revenue</b>, not profit and not customer "
        "lifetime value. The dataset contains no costs, margins, campaign exposure or acquisition "
        "spend, so no profitability or CLV claim can be made from these segments. "
        "The segments are <i>proposed analytical categories</i> created for this case study — they "
        "are not Olist's official customer classifications, and Olist has not reviewed or endorsed "
        "them.",
        "warning", "What these segments are not",
    )
    ui.callout(
        "Segments are a snapshot at the analysis date. They do not model movement between segments "
        "over time, and the thresholds are judgement calls anchored on observed repurchase gaps. "
        "A customer one day on either side of a threshold can land in a different segment, so "
        "operational use should treat boundaries as soft.",
        "definition", "Snapshot, not a forecast",
    )


def _render_kmeans(result) -> None:
    """K-means extension view."""
    profile = result.profile
    meta = result.metadata

    ui.kpi_row([
        ui.kpi_card("Clusters selected", str(meta.get("best_k", "—")),
                    "chosen by highest silhouette score", "Number of k-means clusters.", "primary"),
        ui.kpi_card("Silhouette score", f"{meta.get('best_silhouette', float('nan')):.3f}",
                    "on a 30k sample", "0.5+ is strong structure; below 0.25 suggests weak "
                                       "separation between clusters.", "teal"),
        ui.kpi_card("Customers clustered", ui.fmt_int(int(profile["n_customers"].sum())),
                    "all customers", "K-means assigns every customer to a cluster.", "violet"),
    ], per_row=3)

    st.plotly_chart(
        viz.style(_silhouette_fig(meta["silhouette"]), "Silhouette score by number of clusters",
                  x_title="k", y_title="Silhouette"),
        width="stretch", config={"displayModeBar": False})

    table = profile.rename(columns={
        "cluster": "Cluster", "n_customers": "Customers", "mean_recency": "Mean recency (days)",
        "mean_frequency": "Mean orders", "mean_monetary": "Mean historical value",
        "total_monetary": "Total value", "share_customers": "Share of customers",
        "share_value": "Share of value",
    })
    ui.styled_table(table, column_config={
        "Share of customers": st.column_config.NumberColumn(format="%.2f%%"),
        "Share of value": st.column_config.NumberColumn(format="%.2f%%"),
        "Mean historical value": st.column_config.NumberColumn(format="R$ %.2f"),
        "Total value": st.column_config.NumberColumn(format="R$ %.0f"),
    })

    ui.callout(
        "K-means clusters RFM <i>shape</i>; it does not know what is commercially useful. Read each "
        "cluster's profile before naming it, and treat small clusters with distinctive profiles as "
        "candidates for investigation rather than audiences in their own right. Because frequency is "
        "almost always 1 in this dataset, k-means mostly separates customers by recency and value.",
        "warning", "How to read this",
    )

    ui.download_csv(table, "kmeans_clusters.csv", "Download cluster profile (CSV)")


def _silhouette_fig(df: pd.DataFrame):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=df["k"], y=df["silhouette"], mode="lines+markers",
                               line=dict(width=2.4), marker=dict(size=7)))
    return fig
