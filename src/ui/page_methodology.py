"""Page 6 — Methodology and Data Quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import cohort_analysis as ca
from src import data_validation as dv
from src.ui import components as ui
from src.ui.context import AppContext


def render(ctx: AppContext) -> None:
    summary = ctx.metrics.summary
    thresholds = ctx.metrics.thresholds

    ui.section_header(
        "Methodology and data quality",
        "Every rule that produced the numbers elsewhere in this application, plus what the data "
        "cannot support.",
        eyebrow="Page 6 of 6",
    )

    tabs = st.tabs(["Source & scope", "Cleaning log", "Metric definitions",
                    "Data quality checks", "Raw data audit", "Limitations"])

    # ---------------- Source ----------------
    with tabs[0]:
        col1, col2 = st.columns([1.15, 1], gap="medium")
        with col1:
            ui.definition(
                "Dataset",
                "<b>Brazilian E-Commerce Public Dataset by Olist</b><br>"
                f"Source: <a href='{settings.DATASET_SOURCE_URL}' target='_blank'>"
                f"{settings.DATASET_SOURCE_URL}</a><br>"
                f"Licence: {settings.DATASET_LICENSE}<br>"
                "Publisher: Olist (Brazilian marketplace), released as a public research dataset."
            )
            ui.definition(
                "Scope of this analysis",
                f"Observation window <b>{summary['first_order_date']:%d %b %Y}</b> → "
                f"<b>{summary['as_of_date']:%d %b %Y}</b> "
                f"({summary['observation_days']:,} days). "
                "This is a <b>historical retrospective</b> case study, not real-time analytics. "
                "No data after the end of the dataset is represented anywhere in this application."
            )
            ui.callout(
                "This is an independent student case study built on public data. I am not an "
                "employee or consultant of Olist and have no access to its internal systems, "
                "campaign data or customer records. Nothing here describes Olist's actual "
                "operations, marketing campaigns or customer intelligence.",
                "danger", "Independence statement",
            )
        with col2:
            st.markdown("**Data files loaded**")
            files = pd.DataFrame([
                {"Table": name, "File": fname, "Loaded": "yes"}
                for name, fname in ctx.raw_meta["files"].items()
            ])
            ui.styled_table(files)
            st.markdown("**Row counts**")
            counts = pd.DataFrame([
                {"Table": k, "Rows": v} for k, v in ctx.raw_meta["row_counts"].items()
            ])
            ui.styled_table(counts)

        ui.section_header("Customer identity: the critical decision", "", eyebrow="Identity")
        ui.callout(
            "The raw <code>customers</code> table has one row per <code>customer_id</code>, and a "
            "<code>customer_id</code> is issued <b>per order</b>. A person who orders twice appears "
            "under two different <code>customer_id</code> values but shares one "
            "<code>customer_unique_id</code>. Every retention, cohort, recency and segmentation "
            "calculation in this application therefore uses <b><code>customer_unique_id</code></b>. "
            "Using <code>customer_id</code> would have made the repeat-purchase rate exactly zero by "
            "construction — a silent, total error.",
            "insight", "Why customer_unique_id",
        )
        n_ids = ctx.raw_meta["n_customer_ids"]
        n_unique = ctx.raw_meta["n_unique_ids"]
        ui.note(f"Raw table: {n_ids:,} customer_id values vs {n_unique:,} distinct "
                f"customer_unique_id values — {n_ids - n_unique:,} additional order-scoped keys.")

    # ---------------- Cleaning ----------------
    with tabs[1]:
        st.markdown("### Order eligibility")
        ui.definition(
            "Rule applied to every page",
            f"<code>order_status ∈ {list(settings.PRIMARY_ELIGIBLE_STATUSES)}</code> AND a parseable "
            f"<code>order_purchase_timestamp</code> at or after {settings.MIN_VALID_PURCHASE_DATE}."
        )
        ui.callout(
            "<b>Delivered only.</b> <code>canceled</code> and <code>unavailable</code> orders are not "
            "purchases. In-flight statuses (<code>shipped</code>, <code>processing</code>, "
            "<code>invoiced</code>, …) are excluded because they are only partially observed and "
            "concentrate at the end of the window, which would make the most recent period look "
            "artificially inactive and bias cohort maturity. The alternative "
            f"({list(settings.BROAD_ELIGIBLE_STATUSES)}) is available via "
            "<code>config/settings.py</code> for a sensitivity check.",
            "warning", "Status choice",
        )

        log = ctx.tables.get("cleaning_log")
        if log is not None:
            st.markdown("### Cleaning log")
            st.dataframe(log.to_frame(), width="stretch", hide_index=True)

        st.markdown("### Table relationships and join safety")
        st.markdown(
            """
| Child table | Cardinality per order | How it is handled |
|---|---|---|
| `order_items` | 1–21 rows | Aggregated **first** (count, merchandise sum, freight sum, distinct products/sellers), then joined one-to-one on `order_id`. |
| `order_payments` | 1–29 rows | Aggregated first (total payment value, payment count, max instalments, dominant type), then joined one-to-one. |
| `order_reviews` | 1–3 rows | Latest review by answer timestamp kept per order; all review scores summarised separately from order value. |
| `orders` | 1 row | Grain of every downstream order-level table. |
| `customers` | 1 row per order | Collapsed to `customer_unique_id` with modal geography. |
            """
        )
        ui.callout(
            "Joining items and payments directly to orders would multiply rows and inflate revenue. "
            "The pipeline aggregates each child table independently, joins to the order grain with "
            "<code>validate='one_to_one'</code>, and then verifies order count is unchanged and that "
            "item-level totals reconcile with an independent aggregation. Both checks run on every "
            "application start and are shown in the next tab.",
            "success", "Double-counting prevention",
        )

        st.markdown("### Monetary definitions kept separate")
        ui.styled_table(pd.DataFrame([
            {"Measure": "merchandise_value", "Definition": "Σ order_items.price",
             "Used as": "Primary KPI (all value figures)"},
            {"Measure": "freight_value", "Definition": "Σ order_items.freight_value",
             "Used as": "Reported separately, never added to merchandise value"},
            {"Measure": "order_total", "Definition": "merchandise + freight",
             "Used as": "Not used for cohort/KPI arithmetic"},
            {"Measure": "payment_value", "Definition": "Σ order_payments.payment_value",
             "Used as": "Independent reconciliation only"},
        ]))
        ui.note("Payment value can differ from merchandise + freight because of vouchers, "
                "instalments and rounding in the source system. Where the two disagree the "
                "reconciliation check flags it rather than forcing them to match.")

    # ---------------- Definitions ----------------
    with tabs[2]:
        defs = pd.DataFrame([
            {"Metric": "Eligible order",
             "Definition": f"order_status in {list(settings.PRIMARY_ELIGIBLE_STATUSES)} with a valid purchase timestamp",
             "Denominator": "—"},
            {"Metric": "Unique eligible customer",
             "Definition": "Distinct customer_unique_id with ≥1 eligible order",
             "Denominator": "—"},
            {"Metric": "Repeat-purchase rate",
             "Definition": "Customers with ≥2 eligible orders ÷ unique eligible customers",
             "Denominator": "All customers (ever-repeated measure)"},
            {"Metric": "Repeat customer",
             "Definition": "Customer with ≥2 distinct eligible orders at any point in the window",
             "Denominator": "—"},
            {"Metric": "Month-n retention",
             "Definition": "Cohort members with an eligible order in the month n after their cohort month ÷ cohort size",
             "Denominator": "Cohort size, for cohorts where month n is observable"},
            {"Metric": "Month-1 retention (pooled)",
             "Definition": "Σ active-in-month-1 ÷ Σ cohort sizes across observable cohorts",
             "Denominator": "Sum of cohort sizes with observable month 1"},
            {"Metric": "Cumulative repeat rate (12m)",
             "Definition": "Cohort members with ≥1 order in months 1–12 ÷ cohort size",
             "Denominator": "Cohorts with ≥12 observable months"},
            {"Metric": "Recency",
             "Definition": f"Days from last eligible order to {summary['as_of_date']:%d %b %Y}",
             "Denominator": "—"},
            {"Metric": "Frequency",
             "Definition": "Count of distinct eligible orders in the window",
             "Denominator": "—"},
            {"Metric": "Monetary value",
             "Definition": "Σ merchandise_value over eligible orders (revenue-like, not profit)",
             "Denominator": "—"},
            {"Metric": "Average order value",
             "Definition": "Σ merchandise_value ÷ eligible orders",
             "Denominator": "Eligible orders"},
            {"Metric": "Value concentration (top X%)",
             "Definition": "Value held by top X% of customers ÷ total value",
             "Denominator": "All customers"},
            {"Metric": "Gini of customer value",
             "Definition": "Gini coefficient over customer-level monetary values",
             "Denominator": "All customers"},
            {"Metric": "Days to second purchase",
             "Definition": "Days between first and second eligible order",
             "Denominator": "Customers with ≥2 orders only"},
            {"Metric": "Repeat within window (service page)",
             "Definition": f"Share of orders followed by another order from the same customer within {settings.SERVICE_REPEAT_WINDOW_DAYS} days",
             "Denominator": "Orders with a fully observed follow-up window"},
        ])
        ui.styled_table(defs, height=430)

        st.markdown("### Inactivity thresholds")
        ui.definition(
            "How the thresholds were derived",
            f"Observed repeat-purchase gaps in this dataset: 75th percentile "
            f"<b>{thresholds['gap_p75_days']:.0f} days</b>, 90th percentile "
            f"<b>{thresholds['gap_p90_days']:.0f} days</b>. "
            f"lapsing = max({settings.LAPSING_FLOOR_DAYS}, p75) = "
            f"<b>{thresholds['lapsing_days']:.0f} days</b>; "
            f"inactive = max({settings.INACTIVE_FLOOR_DAYS}, p90) = "
            f"<b>{thresholds['inactive_days']:.0f} days</b>. "
            f"Customers whose first order is more recent than "
            f"{settings.INSUFFICIENT_HISTORY_DAYS} days are labelled 'insufficient history' "
            "instead of being classified as inactive."
        )
        ui.callout(
            "These are behavioural reference points relative to the customer base's own repurchase "
            "rhythm — they are <b>not</b> churn definitions. An 'inactive' customer may be seasonal, "
            "single-occasion or simply gone; the dataset cannot distinguish these.",
            "warning", "Threshold caveat",
        )

    # ---------------- Quality checks ----------------
    with tabs[3]:
        rep = ctx.quality
        status = "all hard checks passed" if rep.passed else f"{len(rep.failures)} failure(s)"
        ui.callout(
            f"{len(rep.checks)} checks ran on this run · {status}. These are recomputed from the "
            "loaded data every time the app starts; nothing is hardcoded.",
            "success" if rep.passed else "danger", "Automated quality checks"
        )
        checks = pd.DataFrame([{
            "Check": c.name, "Result": "PASS" if c.passed else "FAIL",
            "Severity": c.severity, "Detail": c.detail} for c in rep.checks])
        ui.styled_table(checks, height=340)

        st.markdown("### Cohort validation")
        problems = ca.validate_cohort_table(ctx.cohort)
        if problems:
            for p in problems:
                ui.callout(p, "danger", "Cohort validation failure")
        else:
            ui.callout(
                "Cohort table validated: month-0 equals 100% for every cohort, no cell exceeds its "
                "cohort size, retention lies in [0,1], each cohort has a single size, no duplicate "
                "cohort-month rows, and unobservable future months are absent rather than zero.",
                "success", "Cohort validation passed")

        st.markdown("### Observation-window handling")
        maturity = ca.cohort_sizes(ctx.cohort)[["cohort_month", "cohort_size", "max_observable_index"]]
        maturity["cohort_month"] = pd.to_datetime(maturity["cohort_month"]).dt.strftime("%Y-%m")
        ui.styled_table(maturity.rename(columns={
            "cohort_month": "Cohort", "cohort_size": "Customers",
            "max_observable_index": "Observable months"}), height=300)
        ui.note("Later cohorts have fewer observable follow-up months. They are never padded with "
                "zeros, and they are excluded from denominators for months they cannot reach.")

    # ---------------- Raw audit ----------------
    with tabs[4]:
        st.markdown("### Raw table audit")
        for a in ctx.raw_meta["audit"]:
            with st.expander(f"{a['table']} — {a['rows']:,} rows × {a['columns']} columns",
                             expanded=False):
                cols = pd.DataFrame({
                    "Column": a["column_names"],
                    "Dtype": [a["dtypes"][c] for c in a["column_names"]],
                    "Missing": [a["missing_counts"][c] for c in a["column_names"]],
                    "Missing %": [a["missing_pct"][c] for c in a["column_names"]],
                })
                ui.styled_table(cols)
                if a["key_uniqueness"]:
                    k = a["key_uniqueness"]
                    ui.note(f"Key <code>{'+'.join(k['key'])}</code>: "
                            f"{'unique' if k['is_unique'] else '**not unique**'} — "
                            f"{k['duplicate_key_rows']:,} duplicate rows.")
                if a["date_ranges"]:
                    dr = pd.DataFrame([{"Column": c, "Min": v["min"], "Max": v["max"],
                                        "Nulls": v["nulls"]} for c, v in a["date_ranges"].items()])
                    ui.styled_table(dr)
                neg = {c: v for c, v in a["numeric_ranges"].items() if v["negative_count"] > 0}
                if neg:
                    ui.callout("Negative values found in: " + ", ".join(
                        f"<code>{c}</code> ({v['negative_count']:,} rows)" for c, v in neg.items()) +
                        ". These were investigated — see the limitations tab.", "warning",
                        "Negative values")
        ui.download_csv(pd.DataFrame([{
            "table": a["table"], "rows": a["rows"], "columns": a["columns"],
            "duplicate_key_rows": a["key_uniqueness"].get("duplicate_key_rows")
            if a["key_uniqueness"] else None,
        } for a in ctx.raw_meta["audit"]]), "raw_audit_summary.csv", "Download audit summary (CSV)")

    # ---------------- Limitations ----------------
    with tabs[5]:
        limits = [
            ("Historical, not real-time",
             "The dataset ends in 2018. Every figure describes that historical window. This is not a "
             "live feed and nothing here should be read as Olist's current performance."),
            ("No profit, cost or margin data",
             "All monetary figures are revenue-like merchandise value. No contribution margin, COGS, "
             "acquisition cost or campaign cost exists in the source, so profitability and true CLV "
             "cannot be computed."),
            ("No campaign or exposure data",
             "There is no record of emails, ads, discounts, cohorts of acquisition channel or contact "
             "history. No marketing intervention can be evaluated, and no uplift can be attributed."),
            ("Inactivity is not churn",
             "Recency is an observed state, not a permanent exit. The dataset has no cancellation, "
             "unsubscribe, competitor-purchase or off-platform data."),
            ("Associations are not causal",
             "Delivery timing and review scores correlate with later behaviour, but the data contains "
             "no randomisation and no counterfactual. Causal claims require experiments."),
            ("Right-censoring at both ends",
             "Recent customers have had less time to repeat; early cohorts are tiny (the marketplace "
             "was launching in late 2016). Cohort maturity is displayed everywhere it matters."),
            ("Identity approximations",
             "customer_unique_id is the best available person-level key, but shared devices, "
             "households and duplicate accounts mean it is not guaranteed to equal a unique human."),
            ("Data quality artefacts",
             "Missing delivery timestamps, orders with no item rows, and multi-row reviews exist in "
             "the source. Their handling is documented and counted in the cleaning log rather than "
             "silently fixed."),
            ("Segment thresholds are judgement calls",
             "The p80 value cut-off and day thresholds are defensible but arbitrary. Small threshold "
             "changes move customers between segments; boundaries should be treated as soft."),
            ("Test-data separation",
             "Unit tests use small synthetic fixtures (< 10 rows) purely to verify formulas. Those "
             "fixtures never mix with the real data and never appear in any reported finding."),
        ]
        for title, body in limits:
            ui.callout(body, "warning", title)

        ui.section_header("What would make this analysis decision-grade?", "", eyebrow="Next steps")
        st.markdown(
            """
- **Campaign exposure and contact history** — to attribute incremental behaviour to marketing.
- **Contribution margin per order or category** — to convert revenue into profitability.
- **Acquisition cost and channel** — to compute CAC payback and true CLV.
- **A hold-out control group** — the only way to estimate incremental uplift.
- **Longer observation window** — to observe maturity for recent cohorts and reduce censoring bias.
- **Product-level returns and cancellation reasons** — to separate service failure from demand loss.
            """
        )
