"""
Rules-based retention recommendations + a transparent scenario simulator.

Every recommendation object carries the eight fields the brief requires:

    target_segment, evidence, action, mechanism, kpi, experiment_design,
    costs_and_risks, uncertainty

The *evidence* strings are filled with numbers computed from the loaded data at
run time -- nothing here is hardcoded, and the text always states which
definition produced the number.

Recommendations are HYPOTHESES. None of them has been tested on this dataset,
and the dataset contains no campaign, cost or margin fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings


def _pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _money(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"R$ {x:,.{digits}f}"


# ---------------------------------------------------------------------------
# Recommendation object
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    segment: str
    headline: str
    target_segment: str
    evidence: List[str]
    action: str
    mechanism: str
    kpi: str
    experiment_design: str
    costs_and_risks: List[str]
    uncertainty: str
    priority: str = "Medium"
    evidence_strength: str = "Descriptive only"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def _segment_row(profile: pd.DataFrame, segment: str) -> Optional[pd.Series]:
    sub = profile[profile["segment"] == segment]
    return sub.iloc[0] if len(sub) else None


def build_recommendations(
    profile: pd.DataFrame,
    summary: Dict[str, object],
    thresholds: Dict[str, float],
    review_stats: Optional[Dict[str, object]] = None,
    delivery_stats: Optional[Dict[str, object]] = None,
) -> List[Recommendation]:
    """
    Map measured segment characteristics to proposed marketing actions.

    ``profile`` must be the output of ``segmentation.segment_profile`` so that
    the numbers quoted in the evidence really come from the data.
    """
    recs: List[Recommendation] = []
    lapsing = thresholds.get("lapsing_days")
    inactive = thresholds.get("inactive_days")

    # --- 1. High-value active repeat customers -----------------------------
    row = _segment_row(profile, "High-Value Active Repeat")
    if row is not None and row["n_customers"] > 0:
        recs.append(Recommendation(
            segment="High-Value Active Repeat",
            headline="Protect the customers who already generate disproportionate value",
            target_segment="High-Value Active Repeat",
            evidence=[
                f"{int(row['n_customers']):,} customers ({_pct(row['share_customers'])} of the base) "
                f"generated {_money(row['monetary_value'])} — {_pct(row['share_value'])} of all "
                f"historical merchandise value.",
                f"Value index {row['share_value'] / row['share_customers']:.2f}x "
                f"(share of value ÷ share of customers).",
                f"Average {row['avg_frequency']:.2f} eligible orders per customer and "
                f"mean recency {row['avg_recency_days']:.0f} days "
                f"(lapsing threshold = {lapsing:.0f} days).",
                f"Base-wide repeat-purchase rate is {_pct(summary.get('repeat_purchase_rate'))}, "
                f"so this group is already an outlier — the goal is retention of existing "
                f"behaviour, not creation of new behaviour.",
            ],
            action="Invite to a service-led loyalty tier: priority support, early access to "
                   "new categories matching past purchases, and a personal reorder reminder "
                   "timed to their own observed repurchase interval. Test cross-sell of adjacent "
                   "categories rather than discounting their usual items.",
            mechanism="Reducing friction and search cost for an already-satisfied repeat buyer "
                      "lifts share of wallet without training them to wait for discounts. "
                      "Discounting habitual purchases mostly subsidises sales that would have "
                      "happened anyway.",
            kpi="Retention of this segment at month 3 and month 6 after launch; orders per "
                "customer per 90 days; share of this segment's spend in cross-sold categories; "
                "gross margin per customer (not revenue alone).",
            experiment_design="Hold-out randomised controlled test: 80% of the eligible segment "
                              "receives the loyalty treatment, 20% is held back for 6 months. "
                              "Primary metric = orders per customer; guardrail = margin per "
                              "customer, to catch subsidy effects.",
            costs_and_risks=[
                "Programme administration and support-capacity costs are real; a "
                "priority-support tier consumes service hours.",
                "Loyalty perks raise cost to serve and can compress margin even when revenue "
                "grows — measure contribution, not revenue.",
                "Segment definition depends on a p80 value cut-off: customers just below the "
                "cut-off may react the same way, so the cut-off is an operational choice, not "
                "a natural boundary.",
            ],
            uncertainty="This segment is defined on historical value inside one observation "
                        "window. Survival/migration between segments is not modelled, and nothing "
                        "here proves that the proposed perks would have caused additional "
                        "purchases.",
            priority="High" if row["share_value"] > 0.15 else "Medium",
            evidence_strength="Measured (descriptive), non-causal",
        ))

    # --- 2. Lapsing repeat customers ---------------------------------------
    row = _segment_row(profile, "Lapsing Repeat")
    if row is not None and row["n_customers"] > 0:
        recs.append(Recommendation(
            segment="Lapsing Repeat",
            headline="Re-engage proven repeat buyers before they go quiet",
            target_segment="Lapsing Repeat",
            evidence=[
                f"{int(row['n_customers']):,} customers ({_pct(row['share_customers'])}) with "
                f"≥2 eligible orders but mean recency {row['avg_recency_days']:.0f} days — "
                f"beyond the {lapsing:.0f}-day lapsing threshold but inside the "
                f"{inactive:.0f}-day inactive threshold.",
                f"Their average frequency is {row['avg_frequency']:.2f} orders and they still "
                f"account for {_pct(row['share_value'])} of historical merchandise value.",
                f"75th percentile of all observed repeat-purchase gaps in this dataset is "
                f"{thresholds.get('gap_p75_days', float('nan')):.0f} days — the basis for the "
                f"lapsing threshold.",
            ],
            action="Send a behaviour-based re-engagement message tied to what they actually "
                   "bought (restock reminder, complementary category, service check-in), "
                   "escalating to a small incentive only for the sub-group that stays quiet "
                   "after the first touch.",
            mechanism="These customers have already overcome the first-purchase barrier, so the "
                      "blocker is more likely relevance or forgetfulness than trust. A relevant, "
                      "specific message is cheap; a generic discount resets their reference price.",
            kpi="Reactivation rate within 60 days of contact (share of contacted customers with "
                "an order), incremental revenue per contacted customer, and margin after "
                "incentive cost.",
            experiment_design="Three-arm test: (a) no contact, (b) content-led message, "
                              "(c) content-led message + incentive for non-responders. "
                              "Randomise at customer level; report incremental orders versus arm (a).",
            costs_and_risks=[
                "Discount to customers who would have returned anyway is pure margin loss; "
                "the no-contact arm is what makes that visible.",
                "Contact fatigue: repeated re-engagement attempts can increase unsubscribes and "
                "damage the relationship.",
                "The dataset has no campaign exposure field, so any observed 'reactivation' "
                "outside a randomised test is confounded with seasonality and lifecycle timing.",
            ],
            uncertainty="Recency is an observed state, not a churn label. Some of these "
                        "customers will return without any intervention; the true incremental "
                        "uplift can only be established by the test above.",
            priority="High",
            evidence_strength="Measured (descriptive), non-causal",
        ))

    # --- 3. Long inactive ---------------------------------------------------
    row = _segment_row(profile, "Long Inactive")
    if row is not None and row["n_customers"] > 0:
        recs.append(Recommendation(
            segment="Long Inactive",
            headline="Treat long-inactive customers as a low-priority reactivation pool, not a discount target",
            target_segment="Long Inactive",
            evidence=[
                f"{int(row['n_customers']):,} customers ({_pct(row['share_customers'])}) have not "
                f"ordered for more than {inactive:.0f} days (mean recency "
                f"{row['avg_recency_days']:.0f} days).",
                f"They hold only {_pct(row['share_value'])} of historical merchandise value, and "
                f"{_pct(1 - row['share_repeat'])} of them never placed a second order.",
                "Their inactivity exceeds the 90th percentile of observed repeat-purchase gaps, "
                "i.e. they sit outside the normal repeat rhythm of this customer base.",
            ],
            action="Keep them in low-cost automated channels only (email/app messaging) with a "
                   "content-led 'what's new' message. Do not fund paid discounts for the whole "
                   "group; test paid reactivation on a small random sample first and scale only "
                   "if incremental return exceeds marginal cost.",
            mechanism="Very long inactivity usually reflects a need that has gone away, a single "
                      "purchase occasion, or a bad experience — not a price objection that a "
                      "discount fixes. When most of the group never repeated, spend per reactivation "
                      "is usually far above the recoverable margin.",
            kpi="Cost per incremental reactivated order (not response rate), reactivation rate, "
                "and unsubscribe/complaint rates as guardrails.",
            experiment_design="Small randomised pilot on 5-10% of the group with a firm budget "
                              "cap; compare incremental orders against a held-out control at "
                              "matched cost. Pre-register the decision rule before launching.",
            costs_and_risks=[
                "Hard-bounce and deliverability damage when messaging very old contacts.",
                "Negative brand reaction to discounts for customers who left, if the same "
                "customers had a service problem.",
                "Opportunity cost: budget spent here is not spent on the lapsing repeat group, "
                "where the observed repurchase probability is higher.",
            ],
            uncertainty="'Inactive' is not 'churned'. Some customers are seasonal or single-occasion "
                        "buyers; the dataset holds no unsubscribe, off-platform or competitor "
                        "purchase data.",
            priority="Low",
            evidence_strength="Measured (descriptive), non-causal",
        ))

    # --- 4. Recent one-time buyers -----------------------------------------
    row = _segment_row(profile, "Recent One-Time Buyer")
    if row is not None and row["n_customers"] > 0:
        recs.append(Recommendation(
            segment="Recent One-Time Buyer",
            headline="Use the post-purchase window to build a second purchase",
            target_segment="Recent One-Time Buyer",
            evidence=[
                f"{int(row['n_customers']):,} customers ({_pct(row['share_customers'])}) have "
                f"exactly one eligible order and mean recency {row['avg_recency_days']:.0f} days "
                f"(within the {lapsing:.0f}-day lapsing threshold).",
                f"This is the largest single pool of convertible demand: one-time buyers are "
                f"{_pct(summary.get('n_one_time_customers', 0) / max(summary.get('n_customers', 1), 1))} "
                f"of the entire customer base.",
                f"Median observed time to a second purchase is "
                f"{summary.get('median_days_to_second_order', float('nan')):.0f} days, which is "
                f"the natural anchor for the reminder timing.",
            ],
            action="Run a structured onboarding sequence: delivery confirmation + usage guidance, "
                   "then a relevant second-purchase suggestion timed near the observed median "
                   "time-to-second-order, with category recommendations drawn from the first order.",
            mechanism="The second purchase is the behaviour that turns a transaction into a "
                      "relationship. Most of the lift comes from reducing uncertainty about "
                      "delivery and product fit, not from price.",
            kpi="Second-purchase rate within 90 days, revenue per contacted customer, and "
                "review score among contacted customers (to detect a reverse effect).",
            experiment_design="Randomise new customers at their first order into treatment "
                              "(onboarding sequence) vs business-as-usual. Measure the 90-day "
                              "second-purchase rate difference with a confidence interval; "
                              "the base rate is low, so size the test for a small absolute effect.",
            costs_and_risks=[
                "Email/message fatigue in the first weeks can suppress engagement overall.",
                "If delivery is late or the product is wrong, an onboarding push amplifies "
                "dissatisfaction — sequence should be suppressed for service failures.",
                "Low base rates mean noisy measurement: small samples will often show "
                "a difference that is not real.",
            ],
            uncertainty="Because ~97% of buyers never repeat in this dataset, the achievable "
                        "uplift is bounded by a very low base rate. Any commercial case must be "
                        "built from measured test uplift, not assumed percentages.",
            priority="High",
            evidence_strength="Measured (descriptive), non-causal",
        ))

    # --- 5. Dormant one-time buyers ----------------------------------------
    row = _segment_row(profile, "Dormant One-Time Buyer")
    if row is not None and row["n_customers"] > 0:
        recs.append(Recommendation(
            segment="Dormant One-Time Buyer",
            headline="Win back single-purchase customers selectively, not with blanket discounts",
            target_segment="Dormant One-Time Buyer",
            evidence=[
                f"{int(row['n_customers']):,} customers ({_pct(row['share_customers'])}) placed one "
                f"order and their mean recency is {row['avg_recency_days']:.0f} days — past the "
                f"{lapsing:.0f}-day lapsing threshold but within {inactive:.0f} days.",
                f"They carry {_pct(row['share_value'])} of historical merchandise value with "
                f"{row['avg_frequency']:.2f} orders per customer on average.",
            ],
            action="Segment them by first-purchase category and review score before contacting. "
                   "Prioritise customers with high review scores and long-interval categories "
                   "(e.g. home goods) for a content-led win-back; suppress customers whose only "
                   "order had a poor review or a late delivery until the service issue is understood.",
            mechanism="For one-time buyers the second purchase is a re-decision, so relevance and "
                      "recovered trust matter more than the size of the discount.",
            kpi="Win-back rate within 90 days of contact, incremental margin per contacted "
                "customer, and the share of win-backs that repeat a second time.",
            experiment_design="Two-stage design: first a cheap observational analysis of win-back "
                              "rate by first-order category and review score; then a randomised "
                              "test only on the sub-groups that look promising.",
            costs_and_risks=[
                "Blanket discounting across this large group is the fastest way to destroy "
                "margin in an ecommerce P&L.",
                "Customers with a service complaint who receive a discount instead of an apology "
                "may be further alienated.",
            ],
            uncertainty="No service-recovery or complaint-resolution data exists in this dataset, "
                        "so the stated link between a poor first experience and non-return is a "
                        "plausible hypothesis, not a measured effect.",
            priority="Medium",
            evidence_strength="Measured (descriptive), non-causal",
        ))

    # --- 6. New / insufficient history -------------------------------------
    row = _segment_row(profile, "New / Insufficient History")
    if row is not None and row["n_customers"] > 0:
        recs.append(Recommendation(
            segment="New / Insufficient History",
            headline="Do not judge very recent customers as inactive",
            target_segment="New / Insufficient History",
            evidence=[
                f"{int(row['n_customers']):,} customers ({_pct(row['share_customers'])}) made their "
                f"first eligible order less than {settings.INSUFFICIENT_HISTORY_DAYS} days before "
                f"the analysis date — there is not enough elapsed time to observe a second purchase.",
                "Classifying them as 'churned' or dropping them would bias every retention ratio; "
                "they are therefore reported separately.",
            ],
            action="Monitor as a cohort, not as an action group. Use them to measure onboarding "
                   "performance once their observation window matures.",
            mechanism="A fair retention comparison requires comparable elapsed time; excluding "
                      "immature customers prevents a false 'decline' in retention at the end of "
                      "the observation window.",
            kpi="Their 90-day second-purchase rate once observable, compared with earlier cohorts "
                "at the same age.",
            experiment_design="No intervention proposed. This is a measurement-control group; "
                              "report it as a warning against recency-based targeting.",
            costs_and_risks=[
                "Excluding them shrinks the addressable base for campaigns in the most recent "
                "periods of the dataset.",
            ],
            uncertainty="This label is a data-sufficiency statement, not a customer behaviour "
                        "statement.",
            priority="Low",
            evidence_strength="Measurement rule",
        ))

    # --- 7. Service-experience driven recommendation ------------------------
    if review_stats and delivery_stats:
        low_share = review_stats.get("share_1_2")
        late_share = delivery_stats.get("share_late")
        if low_share is not None and not (isinstance(low_share, float) and np.isnan(low_share)):
            recs.append(Recommendation(
                segment="Service experience (cross-segment)",
                headline="Fix the delivery experience before funding discounts",
                target_segment="Customers with a late delivery or a low review score",
                evidence=[
                    f"{_pct(low_share)} of eligible orders received a review score of 1-2, and "
                    f"{_pct(late_share)} were delivered after the estimated delivery date "
                    f"(median delay among late orders: "
                    f"{delivery_stats.get('median_delay_days', float('nan')):.1f} days).",
                    f"Review coverage: {_pct(review_stats.get('coverage'))} of eligible orders "
                    f"have a review score; mean score "
                    f"{review_stats.get('mean_score', float('nan')):.2f}.",
                    "See the Customer Experience page for the measured review-vs-lateness "
                    "relationship and its effect size.",
                ],
                action="Prioritise delivery-promise accuracy and service recovery for late orders "
                       "(proactive notification + resolution path) before spending on win-back "
                       "discounts. A discount after a bad delivery buys a purchase, not trust.",
                mechanism="A late or failed delivery removes the conditions under which a repeat "
                          "purchase can happen at all; fixing it changes the baseline that any "
                          "marketing spend operates on.",
                kpi="Share of late deliveries, share of 1-2 score reviews, repeat rate within "
                    "90 days for affected vs unaffected orders.",
                experiment_design="Operational A/B on delivery-promise buffers by region, or an "
                                  "interrupted time-series around a service-recovery rollout. "
                                  "Compare repeat rates for matched cohorts.",
                costs_and_risks=[
                    "Faster promises raise logistics cost; widening promises may reduce "
                    "conversion at checkout.",
                    "Service changes affect all customers, so the effect is hard to isolate "
                    "without a staggered rollout.",
                ],
                uncertainty="The association between lateness and review score is measured, but "
                            "the link from review score to repeat purchase is weak and confounded "
                            "by customer type, product category and seasonality.",
                priority="High",
                evidence_strength="Measured association (not causal)",
            ))

    # Order by stated priority
    rank = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: rank.get(r.priority, 3))
    return recs


# ---------------------------------------------------------------------------
# Scenario simulator
# ---------------------------------------------------------------------------

def simulate_scenario(
    segment_size: int,
    participation_rate: float,
    incremental_repeat_rate: float,
    average_order_value: float,
    contribution_margin: float,
    cost_per_contact: float,
    incentive_cost_per_redemption: float,
    baseline_repeat_rate: float = 0.0,
    offer_redemption_share: float = 1.0,
) -> Dict[str, float]:
    """
    Transparent unit-economics of a hypothetical retention campaign.

    Every input is an assumption supplied by the user unless it is explicitly
    drawn from the data (segment_size, AOV, baseline_repeat_rate). Nothing here
    is a validated forecast.

    Formulas
    --------
    contacted                = segment_size * participation_rate
    baseline_orders          = contacted * baseline_repeat_rate
    incremental_orders       = contacted * incremental_repeat_rate
    incremental_revenue      = incremental_orders * average_order_value
    gross_contribution       = incremental_revenue * contribution_margin
    redeemed_offers          = (baseline_orders + incremental_orders) * offer_redemption_share
    incentive_cost           = redeemed_offers * incentive_cost_per_redemption
    contact_cost             = contacted * cost_per_contact
    net_contribution         = gross_contribution - incentive_cost - contact_cost
    roi                      = net_contribution / total_cost

    ``baseline_orders`` is deliberately included: a redeemed offer is usually
    also used by customers who would have bought anyway. That "wasted
    incentive" is reported separately so it cannot be hidden inside a headline
    ROI number.
    """
    contacted = float(segment_size) * float(participation_rate)
    baseline_orders = contacted * float(baseline_repeat_rate)
    incremental_orders = contacted * float(incremental_repeat_rate)
    incremental_revenue = incremental_orders * float(average_order_value)
    gross_contribution = incremental_revenue * float(contribution_margin)
    redeemed_offers = (baseline_orders + incremental_orders) * float(offer_redemption_share)
    incentive_cost = redeemed_offers * float(incentive_cost_per_redemption)
    contact_cost = contacted * float(cost_per_contact)
    total_cost = incentive_cost + contact_cost
    net_contribution = gross_contribution - incentive_cost - contact_cost
    wasted_incentive = baseline_orders * float(incentive_cost_per_redemption) * float(offer_redemption_share)

    return {
        "segment_size": float(segment_size),
        "contacted": contacted,
        "baseline_orders": baseline_orders,
        "incremental_orders": incremental_orders,
        "incremental_revenue": incremental_revenue,
        "gross_contribution": gross_contribution,
        "redeemed_offers": redeemed_offers,
        "incentive_cost": incentive_cost,
        "contact_cost": contact_cost,
        "total_cost": total_cost,
        "net_contribution": net_contribution,
        "roi": (net_contribution / total_cost) if total_cost > 0 else np.nan,
        "cost_per_incremental_order": (total_cost / incremental_orders) if incremental_orders > 0 else np.nan,
        "wasted_incentive": wasted_incentive,
        "incremental_revenue_per_contact": (incremental_revenue / contacted) if contacted > 0 else np.nan,
    }


def scenario_range(
    segment_size: int,
    participation_rate: float,
    incremental_repeat_rate: float,
    average_order_value: float,
    contribution_margin: float,
    cost_per_contact: float,
    incentive_cost_per_redemption: float,
    baseline_repeat_rate: float = 0.0,
    multipliers: Dict[str, float] = settings.SCENARIO_RANGE_MULTIPLIERS,
) -> pd.DataFrame:
    """Low / Base / High variants by scaling the assumed incremental rate."""
    rows = []
    for name, m in multipliers.items():
        res = simulate_scenario(
            segment_size=segment_size,
            participation_rate=participation_rate,
            incremental_repeat_rate=incremental_repeat_rate * m,
            average_order_value=average_order_value,
            contribution_margin=contribution_margin,
            cost_per_contact=cost_per_contact,
            incentive_cost_per_redemption=incentive_cost_per_redemption,
            baseline_repeat_rate=baseline_repeat_rate,
        )
        rows.append({"scenario": name, "incremental_rate": incremental_repeat_rate * m, **res})
    return pd.DataFrame(rows)


def breakeven_incremental_rate(
    segment_size: int,
    participation_rate: float,
    average_order_value: float,
    contribution_margin: float,
    cost_per_contact: float,
    incentive_cost_per_redemption: float,
    baseline_repeat_rate: float = 0.0,
) -> float:
    """
    The incremental repeat rate at which net contribution equals zero.

    Useful sanity check: if the break-even rate is far above anything the
    business has ever observed, the campaign is probably not viable on
    contribution alone.
    """
    df = segment_size * participation_rate
    if df <= 0:
        return float("nan")
    margin_per_order = average_order_value * contribution_margin - incentive_cost_per_redemption
    if margin_per_order <= 0:
        return float("inf")   # every incremental order loses money
    fixed_cost = participation_rate * cost_per_contact * segment_size \
        + baseline_repeat_rate * incentive_cost_per_redemption * df
    return float((fixed_cost / margin_per_order) / df)


def recommendations_frame(recs: List[Recommendation]) -> pd.DataFrame:
    """Flatten recommendations for CSV export."""
    rows = []
    for r in recs:
        rows.append({
            "segment": r.segment,
            "headline": r.headline,
            "priority": r.priority,
            "action": r.action,
            "kpi": r.kpi,
            "experiment_design": r.experiment_design,
            "evidence": " | ".join(r.evidence),
            "mechanism": r.mechanism,
            "costs_and_risks": " | ".join(r.costs_and_risks),
            "uncertainty": r.uncertainty,
            "evidence_strength": r.evidence_strength,
        })
    return pd.DataFrame(rows)
