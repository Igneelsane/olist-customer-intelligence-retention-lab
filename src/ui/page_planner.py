"""Page 5 — Retention Action Planner (evidence-linked recommendations + simulator)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import recommendations as rec
from src import segmentation as sg
from src.ui import components as ui
from src.ui.context import AppContext


def render(ctx: AppContext) -> None:
    summary = ctx.metrics.summary
    thresholds = ctx.metrics.thresholds
    profile = ctx.seg.profile

    recs = rec.build_recommendations(
        profile, summary, thresholds, ctx.reviews, ctx.delivery
    )

    ui.section_header(
        "Retention action planner",
        "Measured segment characteristics mapped to proposed marketing actions. Every "
        "recommendation states its evidence, its mechanism, the KPI to watch, the experiment that "
        "would validate it, and what could go wrong.",
        eyebrow="Page 5 of 6",
    )

    ui.callout(
        "Everything on this page is a <b>hypothesis</b>. The dataset contains no campaigns, "
        "customer contacts, costs, margins or competitor activity, so no intervention has been "
        "tested and no uplift has been measured. What the data can do is show <i>where</i> a "
        "business should look — not prove that any action works.",
        "warning", "Read this first",
    )

    # ---------------- Recommendation selector ----------------
    labels = [f"{r.segment} — {r.headline}" for r in recs]
    choice = st.selectbox("Choose a recommendation", labels, index=0)
    selected = recs[labels.index(choice)]

    col_a, col_b = st.columns([2.2, 1], gap="medium")
    with col_a:
        st.markdown(f"### {selected.headline}")
        st.markdown(f"{ui.priority_pill(selected.priority)} &nbsp; "
                    f"<span class='pill low'>{selected.evidence_strength}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**Target segment:** {selected.target_segment}")

        st.markdown("**Evidence from the data**")
        for e in selected.evidence:
            st.markdown(f"- {e}")

        st.markdown("**Proposed action**")
        st.markdown(selected.action)

        st.markdown("**Expected mechanism**")
        st.markdown(selected.mechanism)

        st.markdown("**KPI to monitor**")
        st.markdown(selected.kpi)

        st.markdown("**Experiment design**")
        st.markdown(selected.experiment_design)

    with col_b:
        seg_row = profile[profile["segment"] == selected.segment]
        if len(seg_row):
            r0 = seg_row.iloc[0]
            st.markdown(ui.kpi_card(
                "Customers in segment", ui.fmt_int(r0["n_customers"]),
                f"{ui.fmt_pct(r0['share_customers'])} of the base",
                "Population available to this action.", "primary"), unsafe_allow_html=True)
            st.markdown(ui.kpi_card(
                "Historical value", ui.fmt_money(r0["monetary_value"]),
                f"{ui.fmt_pct(r0['share_value'])} of total value",
                "Merchandise value, not profit.", "violet"), unsafe_allow_html=True)
            st.markdown(ui.kpi_card(
                "Avg recency", ui.fmt_days(r0["avg_recency_days"]),
                f"avg {r0['avg_frequency']:.2f} orders / customer",
                "Days since last eligible order.", "amber"), unsafe_allow_html=True)

        st.markdown("**Costs and risks**")
        for c in selected.costs_and_risks:
            st.markdown(f"- {c}")

    ui.callout(selected.uncertainty, "danger", "Statement of uncertainty")

    # ---------------- All recommendations ----------------
    ui.section_header("All recommendations at a glance", "", eyebrow="Portfolio")
    overview = pd.DataFrame([{
        "Priority": r.priority,
        "Segment": r.segment,
        "Recommendation": r.headline,
        "Primary KPI": r.kpi.split(";")[0],
        "Evidence strength": r.evidence_strength,
    } for r in recs])
    ui.styled_table(overview, height=300)
    ui.download_csv(rec.recommendations_frame(recs), "recommendations.csv",
                    "Download recommendation catalogue (CSV)")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ---------------- Scenario simulator ----------------
    ui.section_header(
        "Scenario simulator (hypothetical)",
        "Unit economics of a proposed campaign. Only the segment size, average order value and "
        "observed repeat rate come from the data — everything else is an assumption you supply.",
        eyebrow="Simulator",
    )

    seg_names = profile["segment"].tolist()
    default_seg = "Lapsing Repeat" if "Lapsing Repeat" in seg_names else seg_names[0]
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        sim_seg_name = st.selectbox("Segment", seg_names, index=seg_names.index(default_seg))
        seg_row = profile[profile["segment"] == sim_seg_name].iloc[0]
        seg_size = int(seg_row["n_customers"])
        observed_aov = float(seg_row["avg_order_value"])
        st.markdown(f"Segment size: **{ui.fmt_int(seg_size)}** customers  \n"
                    f"Observed avg order value: **{ui.fmt_money(observed_aov, digits=2)}**")
        use_observed_aov = st.checkbox("Use observed average order value", value=True)
        aov = observed_aov if use_observed_aov else st.number_input(
            "Average order value (R$)", min_value=1.0, value=float(round(observed_aov, 2)),
            step=5.0)

    with col2:
        participation = st.slider("Intervention participation rate", 0.01, 1.0,
                                 float(settings.SCENARIO_DEFAULTS["participation_rate"]), 0.01,
                                 help="Share of the segment that receives and engages with the "
                                      "message.")
        incremental = st.slider("Assumed incremental repeat rate", 0.0, 0.30,
                                float(settings.SCENARIO_DEFAULTS["incremental_repeat_rate"]), 0.005,
                                format="%.3f",
                                help="Extra share of contacted customers who order because of the "
                                     "intervention. No dataset can supply this — it comes from a "
                                     "test.")
        margin = st.slider("Contribution margin", 0.0, 0.9,
                           float(settings.SCENARIO_DEFAULTS["contribution_margin"]), 0.01,
                           help="Gross margin share of revenue. Not available in this dataset; "
                                "typical ecommerce assumption only.")

    with col3:
        cost_contact = st.number_input("Cost per contacted customer (R$)", min_value=0.0,
                                       value=float(settings.SCENARIO_DEFAULTS["cost_per_contact"]),
                                       step=0.10)
        incentive = st.number_input("Incentive cost per redemption (R$)", min_value=0.0,
                                    value=float(settings.SCENARIO_DEFAULTS["incentive_cost_per_redemption"]),
                                    step=1.0)
        baseline = st.slider("Baseline repeat rate among contacted", 0.0, 0.30,
                             float(min(0.10, max(0.0, seg_row["share_repeat"]))), 0.005,
                             format="%.3f",
                             help="Defaults to the segment's observed share with ≥2 orders. The "
                                  "incentive is also redeemed by these customers, which is why "
                                  "'wasted incentive' is reported.")

    base = rec.simulate_scenario(
        segment_size=seg_size, participation_rate=participation,
        incremental_repeat_rate=incremental, average_order_value=aov,
        contribution_margin=margin, cost_per_contact=cost_contact,
        incentive_cost_per_redemption=incentive, baseline_repeat_rate=baseline,
    )
    band = rec.scenario_range(
        segment_size=seg_size, participation_rate=participation,
        incremental_repeat_rate=incremental, average_order_value=aov,
        contribution_margin=margin, cost_per_contact=cost_contact,
        incentive_cost_per_redemption=incentive, baseline_repeat_rate=baseline,
    )

    ui.kpi_row([
        ui.kpi_card("Contacted customers", ui.fmt_int(base["contacted"]),
                    f"of {ui.fmt_int(seg_size)} in segment",
                    "segment size × participation rate.", "primary"),
        ui.kpi_card("Incremental orders", f"{base['incremental_orders']:,.0f}",
                    f"plus {base['baseline_orders']:,.0f} baseline orders",
                    "contacted × assumed incremental repeat rate.", "teal"),
        ui.kpi_card("Incremental revenue", ui.fmt_money(base["incremental_revenue"]),
                    f"gross contribution {ui.fmt_money(base['gross_contribution'])}",
                    "incremental orders × AOV; contribution applies the assumed margin.", "violet"),
        ui.kpi_card("Net contribution", ui.fmt_money(base["net_contribution"]),
                    f"ROI {base['roi']:.2f}" if not np.isnan(base["roi"]) else "ROI n/a",
                    "gross contribution − incentive − contact cost.", 
                    "teal" if base["net_contribution"] > 0 else "rose"),
    ], per_row=4)

    col1, col2 = st.columns([1.2, 1], gap="medium")
    with col1:
        display = band[["scenario", "incremental_rate", "incremental_orders", "incremental_revenue",
                        "gross_contribution", "total_cost", "net_contribution", "roi"]].copy()
        display = display.rename(columns={
            "scenario": "Scenario", "incremental_rate": "Incremental rate",
            "incremental_orders": "Incremental orders", "incremental_revenue": "Incremental revenue",
            "gross_contribution": "Gross contribution", "total_cost": "Total cost",
            "net_contribution": "Net contribution", "roi": "ROI"})
        ui.styled_table(display, column_config={
            "Incremental rate": st.column_config.NumberColumn(format="%.3f"),
            "Incremental orders": st.column_config.NumberColumn(format="%.0f"),
            "Incremental revenue": st.column_config.NumberColumn(format="R$ %.0f"),
            "Gross contribution": st.column_config.NumberColumn(format="R$ %.0f"),
            "Total cost": st.column_config.NumberColumn(format="R$ %.0f"),
            "Net contribution": st.column_config.NumberColumn(format="R$ %.0f"),
            "ROI": st.column_config.NumberColumn(format="%.2f"),
        })
        ui.note("Low / Base / High scale only the assumed incremental repeat rate "
                "(×0.5 / ×1.0 / ×1.75). Costs are held constant.")

    with col2:
        be = rec.breakeven_incremental_rate(
            segment_size=seg_size, participation_rate=participation,
            average_order_value=aov, contribution_margin=margin,
            cost_per_contact=cost_contact, incentive_cost_per_redemption=incentive,
            baseline_repeat_rate=baseline)
        if np.isinf(be):
            be_text = "impossible — each incremental order loses money at this margin and incentive"
        else:
            be_text = f"<b>{be:.2%}</b> incremental repeat rate"
        ui.definition(
            "Break-even check",
            f"With these inputs the campaign breaks even at {be_text}. "
            f"You assumed {incremental:.2%}. "
            f"Wasted incentive (paid to customers who would have bought anyway): "
            f"{ui.fmt_money(base['wasted_incentive'])}. "
            f"Cost per incremental order: "
            f"{ui.fmt_money(base['cost_per_incremental_order'], digits=2)}."
        )
        ui.callout(
            "Break-even rates are often far above what a business has ever observed. If the required "
            "rate is implausible, the correct conclusion is <i>do not run the campaign on these "
            "terms</i> — not to change the assumption until the numbers look good.",
            "warning", "Use the break-even, not the ROI",
        )

    ui.callout(
        "<b>These are not results.</b> The simulator is arithmetic on assumptions. It cannot tell "
        "you whether a campaign would work; it only shows what would have to be true for the "
        "campaign to pay for itself. The only way to obtain the incremental rate is a randomised "
        "test with a held-out control group.",
        "danger", "Simulation ≠ forecast",
    )
