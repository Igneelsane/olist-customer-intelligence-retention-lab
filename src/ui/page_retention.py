"""Page 2 — Retention.

Structure: the retention shape first (heatmap), then the curve detail, then the
time-to-second-purchase distribution, then the cohort table. Filters that only
affect the view live behind the Filters control so the page opens calm.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import cohort_analysis as ca
from src import visualizations as viz
from src.ui import components as ui
from src.ui.context import AppContext

FILTER_KEYS = ("ret_years", "ret_months", "ret_min_size")
HIGHLIGHT_KEY = "ret_highlight"


def render(ctx: AppContext) -> None:
    ui.apply_pending_reset()          # must run before any widget is created
    summary = ctx.metrics.summary
    cohort_full = ctx.cohort
    orders = ctx.tables["order_values"]
    kpis = ca.retention_kpis(cohort_full, orders)

    sizes = ca.cohort_sizes(cohort_full)
    years = sorted(pd.to_datetime(sizes["cohort_month"]).dt.year.unique())
    default_months = min(12, int(cohort_full["month_index"].max()))

    # ---------------- filters (collapsed) ----------------
    pre = {k: st.session_state.get(k) for k in FILTER_KEYS}
    trigger = ui.active_summary([
        (pre["ret_years"] not in (None, years), "year subset"),
        (pre["ret_months"] not in (None, default_months),
         f"M{pre['ret_months']}" if pre["ret_months"] else ""),
        (pre["ret_min_size"] not in (None, 100),
         f"min {pre['ret_min_size']} customers" if pre["ret_min_size"] else ""),
    ])

    c1, c2 = st.columns([4, 1.1], gap="small")
    with c2:
        with ui.filters("Filters", trigger):
            # Defaults are seeded into session state rather than passed as widget
            # defaults, so the same keys can be re-assigned by the reset control.
            st.session_state.setdefault("ret_years", list(years))
            st.session_state.setdefault("ret_months", default_months)
            st.session_state.setdefault("ret_min_size", 100)
            st.multiselect(
                "Cohort year", years, key="ret_years",
                help="Filters which first-purchase cohorts are shown. Cohort sizes always use "
                     "the full cohort.")
            st.slider(
                "Months since first purchase", 3, 23, key="ret_months",
                help="Higher values expose long-run retention but only for older, fully "
                     "observed cohorts.")
            st.number_input(
                "Minimum cohort size", min_value=1, step=50, key="ret_min_size",
                help="Cohorts smaller than this are hidden from the curves to avoid noise being "
                     "read as signal.")
            ui.reset_button({"ret_years": list(years), "ret_months": default_months,
                             "ret_min_size": 100})

    year_sel = st.session_state.get("ret_years", list(years))
    max_index = int(st.session_state.get("ret_months", default_months))
    min_size = int(st.session_state.get("ret_min_size", 100))

    cohort = (cohort_full[pd.to_datetime(cohort_full["cohort_month"]).dt.year.isin(year_sel)]
              if year_sel else cohort_full)
    if cohort.empty:
        ui.empty_state("No cohorts match the filter", "Widen the cohort year selection.")
        return
    scoped = ca.retention_kpis(cohort, orders)

    # ---------------- primary metrics ----------------
    ui.kpi_strip([
        ui.metric("Month-1 retention", ui.fmt_pct(scoped["month_1_retention_pooled"], 3),
                  f"pooled over {int(cohort['cohort_month'].nunique())} cohorts",
                  "share ordering in the month after their first purchase"),
        ui.metric("Month-2 retention", ui.fmt_pct(scoped["month_2_retention_pooled"], 3),
                  "specific-month activity, not cumulative"),
        ui.metric("Repeated within 12 months",
                  ui.fmt_pct(kpis["median_12m_cumulative_repeat_rate"]),
                  "median across mature cohorts",
                  "at least one order in months 1–12; cumulative, not a sum of monthly rates"),
        ui.metric("Mature cohorts", f"{kpis['n_cohorts_mature_12m']} of {kpis['n_cohorts']}",
                  f"{ui.fmt_int(kpis['n_customers_in_cohorts'])} customers in total",
                  "a cohort is mature when 12 months of data follow it"),
    ])

    ui.callout(
        "Retention here is <b>specific-month</b> activity: a customer counts in month 3 only if "
        "they placed an eligible order in that exact calendar month. Someone who bought once and "
        "never returned is not counted in later months, and someone who returned in month 5 is not "
        "counted in months 1–4. That is why these rates sit far below the "
        f"{ui.fmt_pct(summary['repeat_purchase_rate'])} ‘ever repeated’ figure.",
        "plain", "What the percentages mean",
    )

    # ---------------- heatmap ----------------
    ui.section_header(
        "Cohort retention",
        "Rows are first-purchase cohorts; columns are months since that first purchase. "
        "Blank cells lie beyond the end of the dataset — they are missing, not zero.",
    )
    cohorts_for_heatmap = cohort[cohort["cohort_size"] >= min_size]
    if cohorts_for_heatmap.empty:
        cohorts_for_heatmap = cohort
        ui.note("No cohort reaches the minimum size — showing all cohorts. Percentages for very "
                "small cohorts are not statistically meaningful.")
    matrix = ca.cohort_matrix(cohorts_for_heatmap, max_index=max_index)
    ui.chart(viz.cohort_heatmap(matrix, max_index=max_index), title="")
    ui.note("Month 0 is 100% by construction and is excluded from the colour scale, so the "
            "months that actually differ stay readable.")

    # ---------------- curves ----------------
    ui.section_header(
        "Retention curves",
        "Pooled rate by month, plus individual cohorts traced month by month. Curves stop where "
        "the observation window ends rather than being padded with zeros.",
    )
    cohort_options = [pd.Timestamp(c) for c in sorted(cohort["cohort_month"].unique())]
    mature = ca.cohort_summary(cohort, upto=12)
    default_highlights = (list(mature[mature["is_mature"]]["cohort_month"].sort_values().iloc[-3:])
                          if mature["is_mature"].any() else cohort_options[:3])
    h1, h2 = st.columns([4, 1.1], gap="small")
    with h2:
        with ui.filters("Highlight", "custom" if st.session_state.get(HIGHLIGHT_KEY) else ""):
            st.session_state.setdefault(HIGHLIGHT_KEY, default_highlights)
            st.multiselect(
                "Highlighted cohorts", cohort_options, key=HIGHLIGHT_KEY,
                format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"),
                help="Highlighted cohorts are drawn opaque and thicker; the rest stay faint.")

    highlight = st.session_state.get(HIGHLIGHT_KEY) or default_highlights
    curves = ca.retention_curves(cohort, min_size=min_size, cohorts=cohort_options)
    if curves.empty:
        ui.empty_state("No cohort meets the minimum size",
                       f"Every cohort is smaller than {min_size} customers. Lower the threshold "
                       f"in Filters.")
    else:
        pooled = ca.weighted_retention_by_month(cohort, upto=max_index)
        # Month 0 is 100% by construction; leaving it in would flatten months 1–12
        # against the top of the axis and hide the shape that matters.
        pooled_after_m0 = pooled[pooled["month_index"] >= 1]
        ui.note("Pooled rate for months 1–12 — size-weighted across cohorts, using only cells where "
                "that month is observable. Month 0 is 100% by construction and is left out of this "
                "chart for scale.")
        ui.chart(viz.pooled_retention(curves, pooled_after_m0), height=280, title="")
        ui.note("Individual cohorts. Highlighted cohorts are drawn in green; the rest are shown "
                "faint for context.")
        ui.chart(viz.retention_curves(curves, highlight=highlight), height=320, title="")
        ui.note(
            "The bar view weights each cohort by size and only uses cohorts where that month is "
            "observable — the fairest single summary of the retention shape. The line view shows "
            "individual cohorts so you can see how much they disagree with each other."
        )

    # ---------------- time to second purchase ----------------
    second = ctx.metrics.second_purchase
    ui.section_header(
        "Time to second order",
        "Measured only among customers who did place a second eligible order.",
    )
    col1, col2 = st.columns([1.35, 1], gap="large")
    with col1:
        if len(second):
            ui.chart(viz.time_to_second(second), height=280, title="")
        else:
            ui.empty_state("No repeat purchases in this population",
                           "Time-to-second-purchase cannot be computed without repeat customers.")
    with col2:
        if len(second):
            s = second["days_to_second_order"]
            ui.stat_list([
                ui.metric("Customers who repeated", ui.fmt_int(len(second))),
                ui.metric("Median", f"{s.median():.0f} days"),
                ui.metric("75th percentile", f"{s.quantile(0.75):.0f} days"),
                ui.metric("90th percentile", f"{s.quantile(0.90):.0f} days"),
                ui.metric("Within 30 days", ui.fmt_pct((s <= 30).mean())),
                ui.metric("Within 90 days", ui.fmt_pct((s <= 90).mean())),
            ])
            ui.note(
                "Right-censored: customers whose first order lands near the end of the dataset have "
                "less time to show a second purchase. This is a distribution <i>of repeaters</i>, "
                "not the probability that a given customer repeats."
            )

    # ---------------- cohort table ----------------
    with st.expander("Cohort table — size, maturity and 12-month repeat rate", expanded=False):
        detail = mature.merge(
            ca.cumulative_repeat_rate(cohort, orders, upto=12)[
                ["cohort_month", "n_cumulative_repeaters", "cumulative_repeat_rate"]],
            on="cohort_month", how="left",
        )
        detail["cohort_month"] = pd.to_datetime(detail["cohort_month"]).dt.strftime("%Y-%m")
        detail = detail.rename(columns={
            "cohort_month": "Cohort", "cohort_size": "Customers",
            "m1_active": "Active in month 1", "m1_retention": "Month-1 retention",
            "max_observable_index": "Observable months", "is_mature": "Mature (12m)",
            "n_cumulative_repeaters": "Repeated within 12m",
            "cumulative_repeat_rate": "12m repeat rate",
        })
        detail = ui.pct(detail, ["Month-1 retention", "12m repeat rate"])
        ui.styled_table(detail, column_config={
            "Month-1 retention": st.column_config.NumberColumn(format="%.3f%%"),
            "12m repeat rate": st.column_config.NumberColumn(format="%.2f%%"),
        }, height=320)
        ui.download_csv(detail, "cohort_summary.csv", "Download cohort table (CSV)")

    with st.expander("Cohort counts behind the heatmap", expanded=False):
        counts = ca.cohort_counts_matrix(cohorts_for_heatmap, max_index=max_index)
        counts.columns = [str(c) for c in counts.columns]
        display = counts.reset_index().rename(columns={"cohort_month": "Cohort"})
        display["Cohort"] = pd.to_datetime(display["Cohort"]).dt.strftime("%Y-%m")
        size_map = cohort.groupby("cohort_month")["cohort_size"].first()
        display.insert(1, "Cohort size", display["Cohort"].map(
            lambda m: int(size_map.get(pd.Timestamp(m + "-01"), 0))))
        ui.styled_table(display)
        ui.note("Counts are distinct customers, de-duplicated within each cohort-month cell.")

    ui.callout(
        "The earliest cohorts (Sep–Dec 2016) hold very few customers: the marketplace was in "
        "launch phase and order volume was minimal. Their retention percentages are "
        "arithmetically valid but statistically fragile — read them as context, not a benchmark.",
        "caution", "Small early cohorts",
    )
