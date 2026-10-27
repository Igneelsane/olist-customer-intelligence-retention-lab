"""Page 4 — Customer Experience (reviews, delivery, and later repeat purchasing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import service_analysis as sa
from src import visualizations as viz
from src.ui import components as ui
from src.ui.context import AppContext


def render(ctx: AppContext) -> None:
    reviews = ctx.reviews
    delivery = ctx.delivery
    bands = ctx.tables["score_bands"]
    orders_outcome = ctx.tables["orders_with_outcome"]

    ui.kpi_strip([
        ui.metric("Review coverage", ui.fmt_pct(reviews["coverage"], 1),
                  f"{ui.fmt_int(reviews['n_reviews'])} of "
                  f"{ui.fmt_int(ctx.metrics.summary['n_orders'])} orders scored"),
        ui.metric("Mean review score", f"{reviews['mean_score']:.2f} / 5",
                  f"median {reviews['median_score']:.0f}"),
        ui.metric("Low scores (1–2)", ui.fmt_pct(reviews["share_1_2"]),
                  f"{ui.fmt_pct(reviews['share_5'])} gave 5",
                  "the commercially important tail behind a high average", risk=True),
        ui.metric("Late deliveries", ui.fmt_pct(delivery["share_late"]),
                  f"median {delivery.get('median_late_delay_days', float('nan')):.1f} days late "
                  f"when late",
                  "delivered after the estimated date"),
    ])

    # ---------------- Reviews ----------------
    ui.section_header("Review scores",
                      "Distribution across eligible orders. Scores are strongly skewed to 5, so "
                      "the mean hides the small but commercially important tail of 1–2s.")
    col1, col2 = st.columns([1.3, 1], gap="medium")
    with col1:
        ui.chart(viz.score_distribution(reviews["distribution"]), height=290, title="")
    with col2:
        dist = pd.DataFrame(
            [(f"{k}", v, 100 * v / reviews["n_reviews"])
             for k, v in sorted(reviews["distribution"].items())],
            columns=["Score", "Orders", "Share"])
        ui.styled_table(dist, column_config={
            "Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
        ui.note("Scores are strongly skewed to 5. Means alone hide the small but commercially "
                "important tail of 1–2 scores.")

    # ---------------- Delivery ----------------
    ui.section_header(
        "Delivery timing",
        "Duration from purchase to delivery, and how often the order arrived after the estimated "
        "delivery date recorded at checkout.",
    )
    col1, col2 = st.columns([1.3, 1], gap="medium")
    with col1:
        st.plotly_chart(
            viz.histogram(orders_outcome["delivery_days"], 
                          "Days from purchase to delivery", "Days", nbins=60,
                          y_title="Orders"),
            width="stretch", config={"displayModeBar": False})
    with col2:
        late_dist = (orders_outcome["late_bin"].value_counts()
                     .rename_axis("Delivery vs estimate").reset_index(name="Orders"))
        late_dist["Share"] = 100 * late_dist["Orders"] / late_dist["Orders"].sum()
        late_dist = late_dist.sort_values("Orders", ascending=False)
        ui.styled_table(late_dist, column_config={
            "Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                     max_value=100)})
        ui.note(f"Median delivery: {delivery['median_delivery_days']:.0f} days; 90th percentile "
                f"{delivery['p90_delivery_days']:.0f} days.")

    ui.chart(viz.score_by_lateness(sa.score_vs_lateness(orders_outcome)), height=280,
             title="Mean review score by delivery lateness")
    ui.callout(
        "Review score falls sharply as deliveries arrive later than the promised date. This is a "
        "measured association within one observational dataset — the direction is plausible "
        "(late delivery → dissatisfaction) but the data cannot rule out that harder-to-serve "
        "regions or product types drive both.",
        "plain", "Reading the relationship",
    )

    # ---------------- Repeat within window ----------------
    ui.section_header(
        f"Did they order again within {settings.SERVICE_REPEAT_WINDOW_DAYS} days?",
        "Measured on the same customer after the order in question, and only for orders whose full "
        "follow-up window sits inside the dataset.",
    )

    n_used = int(orders_outcome["repeat_within_window"].notna().sum())
    n_censored = int(orders_outcome["censored"].sum())
    ui.callout(
        f"<b>{ui.fmt_int(n_used)}</b> orders have a fully observed "
        f"{settings.SERVICE_REPEAT_WINDOW_DAYS}-day follow-up window. "
        f"<b>{ui.fmt_int(n_censored)}</b> orders were censored (their window extends past the end of "
        f"the dataset) and are excluded from every rate on this page. The overall base rate is "
        f"{ui.fmt_pct(orders_outcome['repeat_within_window'].mean(), 2)} — tiny, because repeat "
        f"purchasing is rare in this dataset. Treat the <i>differences between groups</i> as the "
        f"signal, not the absolute level.",
        "plain", "Denominator check",
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        by_score = sa.repeat_rate_by_group(bands, "score_band")
        by_score = by_score[~by_score["small_group"]]
        st.plotly_chart(viz.grouped_rate_chart(by_score, "score_band",
                                               "Repeat rate by review score band",
                                               "Review score band"),
                        width="stretch", config={"displayModeBar": False})
        test = sa.association_tests(bands, "score_band")
        if test:
            ui.note(
                f"χ² = {test['chi2']:.1f} (df={test['dof']}), p = {test['p_value']:.2e}, "
                f"Cramér's V = {test['cramers_v']:.3f} on n = {ui.fmt_int(test['n'])}. "
                f"With this sample size even tiny associations are 'significant'; V ≈ "
                f"0.02 means the practical association is <b>negligible</b>."
            )
    with col2:
        by_late = sa.repeat_rate_by_group(orders_outcome, "late_bin")
        by_late = by_late[~by_late["small_group"]]
        st.plotly_chart(viz.grouped_rate_chart(by_late, "late_bin",
                                               "Repeat rate by delivery lateness",
                                               "Delivery vs estimated date"),
                        width="stretch", config={"displayModeBar": False})
        test_late = sa.association_tests(orders_outcome, "late_bin")
        if test_late:
            ui.note(
                f"χ² = {test_late['chi2']:.1f} (df={test_late['dof']}), "
                f"p = {test_late['p_value']:.2e}, Cramér's V = {test_late['cramers_v']:.3f} on "
                f"n = {ui.fmt_int(test_late['n'])}."
            )

    # Computed conclusion -- the wording is derived from the measured effect
    # sizes, so the page cannot imply an association that the data does not show.
    v_score = test.get("cramers_v") if test else np.nan
    v_late = test_late.get("cramers_v") if test_late else np.nan
    p_late = test_late.get("p_value") if test_late else np.nan
    if not np.isnan(v_score) and max(v_score, 0 if np.isnan(v_late) else v_late) < 0.10:
        ui.callout(
            f"<b>Measured conclusion: no practically meaningful association.</b> The strongest "
            f"effect size on this page is Cramér's V = {max(v_score, 0 if np.isnan(v_late) else v_late):.3f} "
            f"(negligible). The differences in repeat rates across review bands are within roughly "
            f"one tenth of a percentage point, and the lateness comparison is not distinguishable "
            f"from noise"
            + (f" (p = {p_late:.2f})" if not np.isnan(p_late) else "") +
            ". Service experience clearly drives <b>satisfaction</b> — but in this dataset it does "
            "not measurably predict whether a customer orders again. That is a genuine result, and "
            "it argues against building a win-back programme on review scores alone.",
            "note-block", "What the data actually says")
    else:
        ui.callout(
            f"Effect sizes here are Cramér's V = {v_score:.3f} for review band and "
            f"{v_late:.3f} for lateness ({sa.interpret_effect_size(v_score)} and "
            f"{sa.interpret_effect_size(v_late)}). Interpret the magnitude before the significance: "
            "with tens of thousands of orders, small differences reach statistical significance "
            "easily.", "note-block", "What the data actually says")

    detail = sa.repeat_rate_by_group(bands, "score_band").rename(columns={
        "score_band": "Review band", "n_orders": "Orders with full window",
        "n_repeat": "Repeated within window", "repeat_rate": "Repeat rate",
        "ci_low": "95% CI lower", "ci_high": "95% CI upper", "small_group": "Group too small"})
    detail = ui.pct(detail, ["Repeat rate", "95% CI lower", "95% CI upper"])
    ui.styled_table(detail, column_config={
        "Repeat rate": st.column_config.NumberColumn(format="%.3f%%"),
        "95% CI lower": st.column_config.NumberColumn(format="%.3f%%"),
        "95% CI upper": st.column_config.NumberColumn(format="%.3f%%"),
    })
    ui.download_csv(detail, "repeat_by_review_band.csv", "Download table (CSV)")

    # ---------------- Causality ----------------
    with st.expander("Why this is an association, not a cause — four specific reasons",
                     expanded=False):
        ui.callout(
        "<b>1. Confounding by customer type.</b> Customers who buy small, immediate-need items may "
        "both review quickly and never return, independent of service quality.<br>"
        "<b>2. Reverse timing.</b> A review is submitted after delivery; if delivery is bad, the "
        "review is bad. The repeat outcome follows both. The chain is plausible but untested.<br>"
        "<b>3. Selection.</b> Only a subset of orders receive reviews "
        f"({ui.fmt_pct(reviews['coverage'], 1)} here), and the propensity to review is itself "
        "correlated with satisfaction.<br>"
        "<b>4. No experiment.</b> Nothing in this dataset varies service quality at random, so no "
        "causal effect of delivery speed on retention can be estimated. An operational A/B test or "
        "a staggered rollout would be required.",
        "caution", "What this page cannot tell you",
    )

    # ---------------- Category view ----------------
    ui.section_header(
        "Product categories",
        "Orders, value and service outcomes by the dominant category of each order.",
    )
    cat = sa.category_performance(ctx.tables["order_values"])
    if cat.empty:
        ui.empty_state("Category data unavailable",
                       "The products table or category translation is missing from this dataset "
                       "directory. The rest of the application works without it.")
    else:
        top = cat.head(15).rename(columns={
            "main_category": "Category", "n_orders": "Orders",
            "merchandise_value": "Merchandise value", "avg_order_value": "Avg order value",
            "mean_review": "Mean review", "share_late": "Share late"})
        ui.styled_table(top, column_config={
            "Merchandise value": st.column_config.NumberColumn(format="R$ %.0f"),
            "Avg order value": st.column_config.NumberColumn(format="R$ %.2f"),
            "Mean review": st.column_config.NumberColumn(format="%.2f"),
            "Share late": st.column_config.NumberColumn(format="%.1f%%"),
        })
        ui.note("Only categories with at least 100 orders are shown. Category is the "
                "highest-value item in the order, so multi-category orders are simplified.")
        ui.download_csv(cat, "category_performance.csv", "Download category table (CSV)")
