"""Page 5 — Action planner.

A five-step workflow, in the order a marketing manager would actually use it:
choose the segment → read the evidence → read the action → agree the KPI →
size the idea in a scenario model. Measured facts and supplied assumptions are
kept visually distinct throughout, because conflating them is the fastest way to
turn an analysis into a forecast it cannot support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import recommendations as rec
from src.ui import components as ui
from src.ui.context import AppContext


def render(ctx: AppContext) -> None:
    summary = ctx.metrics.summary
    thresholds = ctx.metrics.thresholds
    profile = ctx.seg.profile

    recs = rec.build_recommendations(profile, summary, thresholds, ctx.reviews, ctx.delivery)

    ui.callout(
        "Everything on this page is a <b>hypothesis</b>. The dataset contains no campaigns, "
        "customer contacts, costs, margins or competitor activity, so no intervention has been "
        "tested and no uplift measured. The data can show <i>where</i> to look — not prove that "
        "any action works.", "caution", "Read this first")

    # ------------------------------------------------------------------
    # 1. Segment
    # ------------------------------------------------------------------
    ui.section_header("Which segment?", "One segment at a time; the recommendation, the evidence "
                                        "and the scenario all follow this choice.",
                      eyebrow="Step 1")
    segments = [r.segment for r in recs]
    picked = st.radio("Segment", segments, horizontal=True, label_visibility="collapsed")
    selected = recs[segments.index(picked)]
    seg_row = profile[profile["segment"] == selected.segment]
    r0 = seg_row.iloc[0] if len(seg_row) else None

    if r0 is not None:
        ui.kpi_strip([
            ui.metric("Customers in segment", ui.fmt_int(r0["n_customers"]),
                      f"{ui.fmt_pct(r0['share_customers'])} of the customer base"),
            ui.metric("Historical value", ui.fmt_money(r0["monetary_value"]),
                      f"{ui.fmt_pct(r0['share_value'])} of total merchandise value",
                      "historical revenue, not profit"),
            ui.metric("Average recency", ui.fmt_days(r0["avg_recency_days"]),
                      f"{r0['avg_frequency']:.2f} orders per customer on average"),
            ui.metric("Share with ≥2 orders", ui.fmt_pct(r0["share_repeat"]),
                      "the segment's own repeat base rate"),
        ])

    # ------------------------------------------------------------------
    # 2 + 3 + 4. Evidence, action, KPI
    # ------------------------------------------------------------------
    left, right = st.columns([1.55, 1], gap="large")

    with left:
        ui.section_header("Evidence", "Measured facts about this segment from the loaded dataset.",
                          eyebrow="Step 2")
        for e in selected.evidence:
            st.markdown(f"- {e}")

        ui.section_header("Proposed action", f"{selected.headline}",
                          eyebrow="Step 3")
        st.markdown(
            f'{ui.priority_pill(selected.priority)} &nbsp; '
            f'{ui.pill(selected.evidence_strength)}',
            unsafe_allow_html=True,
        )
        st.markdown(selected.action)
        ui.note(f"<b>Expected mechanism.</b> {selected.mechanism}")

        ui.section_header("KPI to monitor", "What would tell you within one test cycle whether the "
                                            "hypothesis holds.", eyebrow="Step 4")
        st.markdown(selected.kpi)

        with st.expander("Experiment design", expanded=False):
            st.markdown(selected.experiment_design)
            ui.callout(selected.uncertainty, "plain", "Statement of uncertainty")

    with right:
        ui.section_header("Costs and risks", "", eyebrow="Before committing")
        for c in selected.costs_and_risks:
            st.markdown(f"- {c}")
        ui.callout(
            "A negative or null result from the test is a valid answer. It is also the cheapest "
            "way to find out that this segment does not respond.", "plain", "If the KPI does not move")

    # ------------------------------------------------------------------
    # 5. Scenario simulator
    # ------------------------------------------------------------------
    ui.section_header(
        "Scenario model",
        "Arithmetic on assumptions. Only the segment size, its observed order value and its "
        "observed repeat rate come from the data — every other input is yours.",
        eyebrow="Step 5 · hypothetical",
    )

    seg_names = profile["segment"].tolist()
    default_seg = "Lapsing Repeat" if "Lapsing Repeat" in seg_names else seg_names[0]
    top = st.columns([1.1, 1, 1], gap="large")

    with top[0]:
        sim_seg_name = st.selectbox("Segment", seg_names, index=seg_names.index(default_seg),
                                    key="sim_segment")
        sim_row = profile[profile["segment"] == sim_seg_name].iloc[0]
        seg_size = int(sim_row["n_customers"])
        observed_aov = float(sim_row["avg_order_value"])
        ui.note(f"{ui.fmt_int(seg_size)} customers · observed AOV "
                f"{ui.fmt_money(observed_aov, digits=2)}")

    with top[1]:
        participation = st.slider(
            "Participation rate", 0.01, 1.0,
            float(settings.SCENARIO_DEFAULTS["participation_rate"]), 0.01,
            help="Share of the segment that receives and engages with the message.")

    with top[2]:
        incremental = st.slider(
            "Assumed incremental repeat rate", 0.0, 0.30,
            float(settings.SCENARIO_DEFAULTS["incremental_repeat_rate"]), 0.005,
            format="%.3f",
            help="Extra share of contacted customers who order because of the intervention. No "
                 "dataset can supply this — it comes from a test.")

    adv1, adv2 = st.columns([4, 1.1], gap="small")
    with adv2:
        with ui.filters("Assumptions"):
            use_observed_aov = st.checkbox("Use observed average order value", value=True)
            aov_override = None
            if not use_observed_aov:
                aov_override = st.number_input(
                    "Average order value (R$)", min_value=1.0,
                    value=float(round(observed_aov, 2)), step=5.0)
            margin = st.slider("Contribution margin", 0.0, 0.9,
                               float(settings.SCENARIO_DEFAULTS["contribution_margin"]), 0.01,
                               help="Gross margin share of revenue. Not in this dataset — a typical "
                                    "ecommerce assumption only.")
            cost_contact = st.number_input(
                "Cost per contacted customer (R$)", min_value=0.0,
                value=float(settings.SCENARIO_DEFAULTS["cost_per_contact"]), step=0.10)
            incentive = st.number_input(
                "Incentive cost per redemption (R$)", min_value=0.0,
                value=float(settings.SCENARIO_DEFAULTS["incentive_cost_per_redemption"]), step=1.0)
            baseline = st.slider("Baseline repeat rate among contacted", 0.0, 0.30,
                                 float(min(0.10, max(0.0, sim_row["share_repeat"]))), 0.005,
                                 format="%.3f",
                                 help="Defaults to the segment's observed share with 2+ orders. The "
                                      "incentive is also redeemed by these customers, which is why "
                                      "'wasted incentive' is reported.")

    aov = observed_aov if use_observed_aov else float(aov_override)

    with adv1:
        ui.note(
            f"Assumed: AOV {ui.fmt_money(aov, digits=2)} · margin {margin:.0%} · contact "
            f"{ui.fmt_money(cost_contact, digits=2)} · incentive {ui.fmt_money(incentive, digits=2)} "
            f"· baseline repeat {baseline:.3f}. Change these under <i>Assumptions</i>."
        )

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

    ui.kpi_strip([
        ui.metric("Contacted", ui.fmt_int(base["contacted"]),
                  f"of {ui.fmt_int(seg_size)} in segment"),
        ui.metric("Incremental orders", f"{base['incremental_orders']:,.0f}",
                  f"plus {base['baseline_orders']:,.0f} baseline orders"),
        ui.metric("Net contribution", ui.fmt_money(base["net_contribution"]),
                  f"ROI {base['roi']:.2f}" if not np.isnan(base["roi"]) else "ROI n/a",
                  "gross contribution − incentive − contact cost",
                  risk=base["net_contribution"] < 0),
        ui.metric("Revenue at stake", ui.fmt_money(base["incremental_revenue"]),
                  f"gross contribution {ui.fmt_money(base['gross_contribution'])}",
                  "incremental orders × assumed AOV × margin"),
    ])

    col1, col2 = st.columns([1.25, 1], gap="large")
    with col1:
        display = band[["scenario", "incremental_rate", "incremental_orders",
                        "incremental_revenue", "gross_contribution", "total_cost",
                        "net_contribution", "roi"]].rename(columns={
            "scenario": "Scenario", "incremental_rate": "Incremental rate",
            "incremental_orders": "Orders", "incremental_revenue": "Revenue",
            "gross_contribution": "Gross contribution", "total_cost": "Total cost",
            "net_contribution": "Net contribution", "roi": "ROI"})
        ui.styled_table(display, column_config={
            "Incremental rate": st.column_config.NumberColumn(format="%.3f"),
            "Orders": st.column_config.NumberColumn(format="%.0f"),
            "Revenue": st.column_config.NumberColumn(format="R$ %.0f"),
            "Gross contribution": st.column_config.NumberColumn(format="R$ %.0f"),
            "Total cost": st.column_config.NumberColumn(format="R$ %.0f"),
            "Net contribution": st.column_config.NumberColumn(format="R$ %.0f"),
            "ROI": st.column_config.NumberColumn(format="%.2f"),
        })
        ui.note("Low / Base / High scale only the assumed incremental repeat rate "
                "(×0.5 / ×1.0 / ×1.75). Costs are held constant.")

    with col2:
        be = rec.breakeven_incremental_rate(
            segment_size=seg_size, participation_rate=participation, average_order_value=aov,
            contribution_margin=margin, cost_per_contact=cost_contact,
            incentive_cost_per_redemption=incentive, baseline_repeat_rate=baseline)
        be_text = ("impossible — each incremental order loses money at this margin and incentive"
                   if np.isinf(be) else f"<b>{be:.2%}</b> incremental repeat rate")
        ui.definition(
            "Break-even check",
            f"With these inputs the campaign breaks even at {be_text}. You assumed "
            f"{incremental:.2%}. Wasted incentive (paid to customers who would have bought anyway): "
            f"{ui.fmt_money(base['wasted_incentive'])}. Cost per incremental order: "
            f"{ui.fmt_money(base['cost_per_incremental_order'], digits=2)}.")
        ui.callout(
            "Break-even rates are often far above anything a business has observed. If the required "
            "rate is implausible, the correct conclusion is <i>do not run the campaign on these "
            "terms</i> — not to adjust the assumption until the numbers look good.",
            "caution", "Use the break-even, not the ROI")

    ui.callout(
        "<b>These are not results.</b> The simulator only shows what would have to be true for the "
        "campaign to pay for itself. The only way to obtain the incremental rate is a randomised "
        "test with a held-out control group.", "plain", "Simulation ≠ forecast")

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    with st.expander("All recommendations at a glance", expanded=False):
        overview = pd.DataFrame([{
            "Priority": r.priority, "Segment": r.segment, "Recommendation": r.headline,
            "Primary KPI": r.kpi.split(";")[0], "Evidence strength": r.evidence_strength,
        } for r in recs])
        ui.styled_table(overview, height=260)
        ui.download_csv(rec.recommendations_frame(recs), "recommendations.csv",
                        "Download recommendation catalogue (CSV)")
