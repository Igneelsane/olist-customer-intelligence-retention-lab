"""Page 2 — Retention Intelligence (cohort analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import cohort_analysis as ca
from src import visualizations as viz
from src.ui import components as ui
from src.ui.context import AppContext


def render(ctx: AppContext) -> None:
    summary = ctx.metrics.summary
    cohort_full = ctx.cohort

    kpis = ca.retention_kpis(cohort_full, ctx.tables["order_values"])

    ui.section_header(
        "Retention intelligence",
        "Monthly cohorts grouped by the calendar month of a customer's first eligible order. "
        "Every cell answers one question: of the people who first bought in month X, how many "
        "bought again in month X+n?",
        eyebrow="Page 2 of 6",
    )

    # ---------------- Filters ----------------
    sizes = ca.cohort_sizes(cohort_full)
    years = sorted(pd.to_datetime(sizes["cohort_month"]).dt.year.unique())
    with st.container():
        f1, f2, f3 = st.columns([1, 1.4, 1], gap="medium")
        with f1:
            year_sel = st.multiselect("Cohort year", years, default=years,
                                      help="Filters which first-purchase cohorts are shown. "
                                           "Cohort sizes always use the full cohort.")
        with f2:
            max_index = st.slider("Months since first purchase shown", 3, 23,
                                  min(12, int(cohort_full["month_index"].max())),
                                  help="Higher values expose long-run retention but only for "
                                       "older, fully observed cohorts.")
        with f3:
            min_size = st.number_input("Minimum cohort size", min_value=1, value=100, step=50,
                                       help="Cohorts smaller than this are hidden from curves "
                                            "to avoid noise being read as signal.")

    cohort = cohort_full[
        pd.to_datetime(cohort_full["cohort_month"]).dt.year.isin(year_sel)
    ] if year_sel else cohort_full

    if cohort.empty:
        ui.empty_state("No cohorts match the filter", "Widen the cohort year selection.")
        return

    cohort_years_kpi = ca.retention_kpis(cohort, ctx.tables["order_values"])

    # ---------------- KPI row ----------------
    ui.kpi_row([
        ui.kpi_card(
            "Month-1 retention (pooled)", ui.fmt_pct(cohort_years_kpi["month_1_retention_pooled"], 3),
            f"size-weighted across {int(cohort['cohort_month'].nunique())} cohorts",
            "Customers ordering in their 1st month after the cohort month ÷ all cohort members "
            "whose month 1 is observable.",
            "primary",
        ),
        ui.kpi_card(
            "Month-2 retention (pooled)", ui.fmt_pct(cohort_years_kpi["month_2_retention_pooled"], 3),
            "specific-month activity, not cumulative",
            "Same denominator logic, month index = 2.",
            "primary",
        ),
        ui.kpi_card(
            "Ever repeated within 12 months", ui.fmt_pct(kpis["median_12m_cumulative_repeat_rate"]),
            "median across mature cohorts",
            "At least one order in months 1–12, per cohort. This is cumulative; it is NOT the sum "
            "of the monthly rates.",
            "teal",
        ),
        ui.kpi_card(
            "Mature cohorts (12+ months)", f"{kpis['n_cohorts_mature_12m']} of {kpis['n_cohorts']}",
            f"{ui.fmt_int(kpis['n_customers_in_cohorts'])} customers total",
            "A cohort is mature when the dataset contains at least 12 months after it.",
            "amber",
        ),
    ], per_row=4)

    ui.callout(
        "Retention here is <b>specific-month</b> activity: a customer counts in month 3 only if they "
        "placed an eligible order in that exact calendar month. A customer who bought once and never "
        "returned is <i>not</i> counted in later months — and a customer who returned in month 5 is "
        "not counted in months 1–4. That is why these rates sit far below the "
        f"{ui.fmt_pct(summary['repeat_purchase_rate'])} 'ever repeated' figure.",
        "definition", "What the percentages mean",
    )

    # ---------------- Heatmap ----------------
    ui.section_header(
        "Cohort retention heatmap",
        "Rows are first-purchase cohorts, columns are months since that first purchase. "
        "Blank cells are months beyond the end of the dataset — they are missing, not zero.",
        eyebrow="Retention matrix",
    )
    # Cohorts below the minimum size are dropped from the heatmap as well: a
    # 1-customer cohort produces 100% cells that are visually misleading.
    cohorts_for_heatmap = cohort[cohort["cohort_size"] >= min_size]
    if cohorts_for_heatmap.empty:
        cohorts_for_heatmap = cohort
        ui.note("No cohort reaches the minimum size — showing all cohorts. Percentages for very "
                "small cohorts are not statistically meaningful.")
    matrix = ca.cohort_matrix(cohorts_for_heatmap, max_index=max_index)
    st.plotly_chart(viz.cohort_heatmap(matrix, max_index=max_index), width="stretch",
                    config={"displayModeBar": False})

    with st.expander("Show the underlying counts heatmap", expanded=False):
        counts = ca.cohort_counts_matrix(cohorts_for_heatmap, max_index=max_index)
        counts.columns = [str(c) for c in counts.columns]
        display = counts.reset_index().rename(columns={"cohort_month": "Cohort"})
        display["Cohort"] = pd.to_datetime(display["Cohort"]).dt.strftime("%Y-%m")
        size_map = cohort.groupby("cohort_month")["cohort_size"].first()
        display.insert(1, "Cohort size", display["Cohort"].map(
            lambda m: int(size_map.get(pd.Timestamp(m + "-01"), 0))))
        ui.styled_table(display)
        ui.note("Counts are distinct customers, de-duplicated within each cohort-month cell.")

    # ---------------- Curves ----------------
    ui.section_header(
        "Retention curves",
        "Selected cohorts traced month by month. Curves stop where the observation window ends, "
        "so older cohorts appear longer rather than being padded with zeros.",
        eyebrow="Curves",
    )
    cohort_options = [pd.Timestamp(c) for c in sorted(cohort["cohort_month"].unique())]
    mature = ca.cohort_summary(cohort, upto=12)
    default_highlights = list(
        mature[mature["is_mature"]]["cohort_month"].sort_values().iloc[-3:]
    ) if mature["is_mature"].any() else cohort_options[:3]

    highlight = st.multiselect(
        "Highlight cohorts", cohort_options, default=default_highlights,
        format_func=lambda d: pd.Timestamp(d).strftime("%B %Y"),
        help="Highlighted cohorts are drawn opaque and thicker.",
    )
    curves = ca.retention_curves(cohort, min_size=min_size, cohorts=cohort_options)
    if curves.empty:
        ui.empty_state("No cohort meets the minimum size",
                       f"Every cohort is smaller than {min_size} customers. Lower the threshold.")
    else:
        st.plotly_chart(viz.retention_curves(curves, highlight=highlight),
                        width="stretch", config={"displayModeBar": False})

        pooled = ca.weighted_retention_by_month(cohort, upto=max_index)
        st.plotly_chart(viz.pooled_retention(curves, pooled), width="stretch",
                        config={"displayModeBar": False})
        ui.note(
            "Pooled bars weight each cohort by its size and use only cohorts where that month is "
            "actually observable. This is the fairest single summary of the retention shape, but it "
            "still mixes cohorts acquired in different conditions."
        )

    # ---------------- Time to second purchase ----------------
    ui.section_header(
        "How long until a customer buys again?",
        "Measured only among customers who did place a second eligible order.",
        eyebrow="Intervals",
    )
    second = ctx.metrics.second_purchase
    col1, col2 = st.columns([1.4, 1], gap="medium")
    with col1:
        if len(second):
            st.plotly_chart(viz.time_to_second(second), width="stretch",
                            config={"displayModeBar": False})
        else:
            ui.empty_state("No repeat purchases in this population",
                           "Time-to-second-purchase cannot be computed without repeat customers.")
    with col2:
        if len(second):
            s = second["days_to_second_order"]
            stats = pd.DataFrame({
                "Statistic": ["Customers who repeated", "Median days", "Mean days",
                              "75th percentile", "90th percentile",
                              "Within 30 days", "Within 90 days"],
                "Value": [
                    ui.fmt_int(len(second)), f"{s.median():.0f}", f"{s.mean():.0f}",
                    f"{s.quantile(0.75):.0f}", f"{s.quantile(0.90):.0f}",
                    ui.fmt_pct((s <= 30).mean()), ui.fmt_pct((s <= 90).mean()),
                ],
            })
            ui.styled_table(stats)
            ui.definition(
                "Right-censoring warning",
                "Customers whose first order happened near the end of the dataset have less time to "
                "show a second purchase. Time-to-second-purchase is therefore computed only among "
                "observed repeaters — it is a distribution <i>of repeaters</i>, not the probability "
                "that a given customer repeats."
            )

    # ---------------- Cohort table ----------------
    ui.section_header("Cohort table", "Size, maturity and first repeat signal per cohort.",
                      eyebrow="Detail")
    detail = mature.merge(
        ca.cumulative_repeat_rate(cohort, ctx.tables["order_values"], upto=12)[
            ["cohort_month", "n_cumulative_repeaters", "cumulative_repeat_rate"]],
        on="cohort_month", how="left",
    )
    detail["cohort_month"] = pd.to_datetime(detail["cohort_month"]).dt.strftime("%Y-%m")
    detail = detail.rename(columns={
        "cohort_month": "Cohort", "cohort_size": "Customers", "m1_active": "Active in month 1",
        "m1_retention": "Month-1 retention", "max_observable_index": "Observable months",
        "is_mature": "Mature (12m)", "n_cumulative_repeaters": "Repeated within 12m",
        "cumulative_repeat_rate": "12m repeat rate",
    })
    ui.styled_table(detail, column_config={
        "Month-1 retention": st.column_config.NumberColumn(format="%.3f%%"),
        "12m repeat rate": st.column_config.NumberColumn(format="%.2f%%"),
    }, height=340)
    ui.download_csv(detail, "cohort_summary.csv", "Download cohort table (CSV)")

    ui.callout(
        "The earliest cohorts (Sep–Dec 2016) contain very few customers: the marketplace was in "
        "launch phase and order volume was minimal. Their retention percentages are therefore "
        "arithmetically valid but statistically fragile — read them as context, not as a benchmark.",
        "warning", "Small early cohorts",
    )
