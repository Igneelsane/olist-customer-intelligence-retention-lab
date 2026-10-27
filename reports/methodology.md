# Methodology

This document records every analytical decision that changes a number, in the order
the pipeline makes it. If a metric in the app disagrees with this document, the
document and the code need to be reconciled.

All figures quoted here were computed by `scripts/generate_reports.py` against the
real dataset (see `reports/findings_metrics.json`).

---

## 1. Source and scope

| | |
|---|---|
| Dataset | Brazilian E-Commerce Public Dataset by Olist |
| Source | https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce |
| Licence | CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution) |
| Observation window | 15 Sep 2016 → 29 Aug 2018 (714 days) |
| Raw order rows | 99,441 |
| Eligible order rows | 96,478 |
| Distinct eligible customers | 93,358 |

This is a **retrospective historical analysis**. No data after 29 Aug 2018 exists in
the source and none is synthesised.

---

## 2. Order eligibility

**Rule:** `order_status = 'delivered'` **and** a parseable `order_purchase_timestamp`
on or after 2016-01-01.

**Effect:** 2,963 orders excluded (2.98%).

| Status | Orders | Included? | Reason |
|---|---|---|---|
| delivered | 96,478 | yes | Goods received; purchase realised |
| shipped | 1,107 | no | Delivery not confirmed; concentrates at end of window |
| canceled | 625 | no | No purchase realised |
| unavailable | 609 | no | No purchase realised |
| invoiced | 314 | no | In-flight; partially observed |
| processing | 301 | no | In-flight; partially observed |
| created | 5 | no | Never paid |
| approved | 2 | no | In-flight; partially observed |

**Why not "everything except canceled"?** In-flight statuses are only partially
observed and cluster in the final weeks of the dataset. Including them would make the
most recent period look artificially inactive and would distort cohort maturity for
the newest cohorts. Since the retention KPIs depend on comparing cohorts across time,
the definition has to be stable across the whole window.

**Sensitivity:** `config/settings.py::BROAD_ELIGIBLE_STATUSES` adds
`shipped/invoiced/processing/approved`. Anyone can switch it and rerun; every
page recomputes from the same rule. The default stays narrow because a KPI called
"purchase" should mean a completed purchase.

---

## 3. Customer identity (the critical decision)

- `customers.customer_id`: **99,441 distinct values** — one per order.
- `customers.customer_unique_id`: **96,096 distinct values** — one per person.

Therefore **3,345 `customer_id` values belong to people who already had another
`customer_id`**. Using `customer_id` for retention would give a repeat-purchase rate
of exactly 0%, silently and completely.

**Rule:** all customer-level analysis (recency, frequency, monetary, cohorts, RFM,
inactivity, segmentation) uses `customer_unique_id`. Order-level tables keep
`customer_id` for lineage only. Geography is collapsed to one row per person using the
modal city/state.

**Caveat:** `customer_unique_id` is the best available person-level key but is not
guaranteed to be one human — shared devices, household accounts and duplicate
registrations can still merge or split identities.

---

## 4. Table relationships and join safety

| Child table | Rows | Cardinality per order | Handling |
|---|---|---|---|
| `order_items` | 112,650 | 1–21 | Aggregated to one row per order: item count, distinct products, distinct sellers, Σ price, Σ freight |
| `order_payments` | 103,886 | 1–29 | Aggregated: Σ payment value, payment count, max instalments, dominant type |
| `order_reviews` | 99,224 | 1–3 (814 duplicate `review_id` rows) | Latest review per order by `review_answer_timestamp` |
| `orders` | 99,441 | 1 | Grain of the analysis |
| `customers` | 99,441 | 1 per order | Collapsed to `customer_unique_id` |

**Double-counting prevention:** each child table is aggregated *before* joining, joins
use `validate="one_to_one"`, and two checks run on every pipeline execution:

1. order count unchanged after joining item and payment aggregates;
2. an independent re-aggregation of `order_items.price` reconciles with the
   order-level `merchandise_value` to within R$0.01 (observed max difference: 0.000000).

---

## 5. Monetary measures (kept deliberately separate)

| Measure | Definition | Used for |
|---|---|---|
| `merchandise_value` | Σ `order_items.price` | **All KPI value figures** |
| `freight_value` | Σ `order_items.freight_value` | Reported separately, never added |
| `order_total` | merchandise + freight | Not used in KPI or cohort arithmetic |
| `payment_value` | Σ `order_payments.payment_value` | Independent reconciliation only |

Dataset totals: merchandise **R$ 13,221,498.11**, freight **R$ 2,198,275.64**,
payments **R$ 15,422,461.77**. Payments differ from merchandise + freight because the
source records vouchers, instalment rounding and, in some cases, amounts that do not
decompose cleanly. The reconciliation check flags mismatch instead of forcing equality.

**Terminology rule:** these are *historical, revenue-like* values. They are **not
profit, not contribution margin, and not customer lifetime value.**

---

## 6. Repeat purchase and retention definitions

| Metric | Formula | Denominator |
|---|---|---|
| Repeat customer | ≥2 distinct eligible orders, ever | — |
| **Repeat-purchase rate** | repeat customers ÷ unique eligible customers | 93,358 customers |
| Cohort | customers grouped by calendar month of first eligible order | — |
| **Month-n retention** | cohort members with ≥1 eligible order in that specific month ÷ cohort size | cohort size, restricted to cohorts where month *n* is observable |
| Month-1 retention (pooled) | Σ active-in-month-1 ÷ Σ cohort sizes (observable cohorts only) | size-weighted |
| Cumulative repeat rate (12m) | cohort members with ≥1 order in months 1–12 ÷ cohort size | cohorts with ≥12 observable months |
| Days to second purchase | days between first and second eligible order | repeaters only |

**Specific-month retention ≠ cumulative repeat.** A customer who returns in month 5 is
not active in months 1–4, and a customer who never returns is never active again. That
is why pooled month-1 retention is **0.48%** while the ever-repeated rate is **3.00%**.
Both are correct; they answer different questions. The app labels them separately
everywhere and the reports never use one in place of the other.

### Observation-window handling (censoring)

- The dataset ends 29 Aug 2018. A cohort from month *m* can be observed at most
  `(2018−m)` months forward.
- The cohort table is built only over **observable** month indices. Months beyond the
  window are **absent from the table**, not stored as zero.
- The heatmap renders a *specific* cell as blank when it does not exist, and as `0%`
  only when the month is observable but no cohort member purchased in it.
- Curves stop where observation stops; pooled retention weights cohorts by size and
  uses only observable cells.
- 12-month cumulative repeat rate is computed only for the **11 mature cohorts**;
  immature cohorts are excluded rather than being credited with a low rate.

### Validation (also enforced in tests)

- month 0 = 100% for every cohort;
- every cohort has exactly one `cohort_size`;
- `n_active ≤ cohort_size` in every cell;
- retention ∈ [0, 1];
- no duplicate `(cohort_month, month_index)` rows (which would mean a customer was
  counted twice inside one cell);
- customers are de-duplicated within each cohort-month before counting.

---

## 7. Recency and inactivity

**Analysis date:** 29 Aug 2018 — the last day in the dataset with an eligible order.
Recency is measured to this date, not to today, so the analysis is reproducible.

**Thresholds derived from the data** (observed repeat-purchase gaps):

- 75th percentile gap = 121.5 days → **lapsing threshold = max(90, 121.5) = 121.5 days**
- 90th percentile gap = 240.3 days → **inactive threshold = max(180, 240.3) = 240.3 days**

Floors (90 / 180) prevent degenerate thresholds on unusually dense data.

| Label | Condition |
|---|---|
| New / insufficient history | first order < 60 days before analysis date **and** only one order |
| Recently active | recency ≤ 121.5 days |
| Potentially lapsing | 121.5 < recency ≤ 240.3 days |
| Long inactive | recency > 240.3 days |

**These are behavioural reference points, not churn probabilities.** An inactive
customer may be seasonal, single-occasion, or gone; the dataset cannot distinguish
these, and no churn label exists anywhere in the source.

---

## 8. RFM segmentation

**Inputs (all at the customer level, all inside the observation window):**
Recency = days since last eligible order; Frequency = count of distinct eligible
orders; Monetary = Σ merchandise value.

**Method 1 — rule-based (default).** Ordered rules, first match wins:

1. `New / Insufficient History` — first order < 60 days ago AND only one order.
2. `Long Inactive` — recency > 240.3 days.
3. Repeat customers (≥2 orders): `High-Value Active Repeat` if recency ≤ 121.5 **and**
   value ≥ p80; else `Active Repeat` if recency ≤ 121.5; else `Lapsing Repeat`.
4. One-time buyers: `Recent One-Time Buyer` if recency ≤ 121.5; else
   `Dormant One-Time Buyer`.

Result: 7 mutually exclusive, exhaustive segments (460 / 385 / 840 / 12,427 / 25,529 /
41,746 / 11,971 customers, 93,358 total).

**Method 2 — k-means (extension).** `log1p` on frequency and monetary, `log1p` on
recency, standardised; k ∈ [3, 8] selected by silhouette on a 30k sample; seed 42.
The app reports the silhouette curve and cluster profiles and states explicitly that a
cluster is not commercially meaningful merely because it exists. Because frequency is
1 for ~97% of customers, k-means mostly separates on recency and value.

**What is *not* claimed:** these are proposed analytical categories, not Olist's
classifications; monetary value is revenue-like, not profit or CLV; segments are a
snapshot with no migration modelling; boundaries are soft (a customer one day on the
other side of a threshold changes segment).

---

## 9. Service-experience analysis

**Descriptors.** Review score (latest per order), delivery duration, lateness =
`delivered_customer_date − estimated_delivery_date` bucketed into Early / On time /
1–3 / 4–7 / 8+ days.

**Outcome.** For each order: did the same customer place another eligible order within
**90 days**? Orders whose 90-day window extends past 29 Aug 2018 are marked censored
and excluded from every rate (18,741 of 96,478 orders). Denominator: 77,737 orders.

**Statistics.** Wilson 95% confidence intervals for every group rate (stable at small
proportions), χ² test for association, and **Cramér's V** as the effect size. With
tens of thousands of orders, significance is nearly guaranteed; V is the number that
should drive interpretation.

**Result.** Delivery lateness → review score is strong and monotone (mean 4.30 early,
1.73 at 8+ days late). Lateness → repeat purchase within 90 days is not measurable
(V = 0.007, p = 0.45); review band → repeat purchase is negligible (V = 0.011,
p = 0.03). Stated on the page in computed wording rather than asserted narrative.

**Why this is reported honestly.** The intuitive story ("bad service loses
customers") is not supported here at the retention level. Four structural reasons are
listed in the app: confounding by customer/product type, timing (review follows
delivery), review-selection effects, and the absence of any randomisation.

---

## 10. Recommendations and the simulator

The recommendation engine maps measured segment characteristics to actions. Evidence
strings are generated from the computed profile at run time, so the numbers in a
recommendation always match the tables. Each item carries: target segment, evidence,
action, mechanism, KPI, experiment design, costs and risks, uncertainty, priority and
evidence strength.

**Simulator formulas** (all inputs labelled hypothetical except segment size, AOV and
observed repeat share):

```
contacted           = segment_size × participation_rate
baseline_orders     = contacted × baseline_repeat_rate
incremental_orders  = contacted × incremental_repeat_rate
incremental_revenue = incremental_orders × AOV
gross_contribution  = incremental_revenue × contribution_margin
redeemed_offers     = (baseline_orders + incremental_orders) × redemption_share
incentive_cost      = redeemed_offers × incentive_cost_per_redemption
contact_cost        = contacted × cost_per_contact
net_contribution    = gross_contribution − incentive_cost − contact_cost
ROI                 = net_contribution ÷ (incentive_cost + contact_cost)
```

`baseline_orders` is included on purpose: a redeemed offer is usually also used by
customers who would have bought anyway. That "wasted incentive" is displayed
separately. The app also computes the **break-even incremental rate**, which is a more
useful decision aid than a favourable ROI on optimistic assumptions.

---

## 11. Reproducibility

- Deterministic: no random sampling except the k-means silhouette sample and the
  scatter plot sample, both seeded (`random_state=42`).
- `data/processed/` parquet artifacts are invalidated when any raw file's size or
  mtime changes (fingerprint used as the cache key).
- `scripts/generate_reports.py` regenerates `reports/findings_metrics.json`, which is
  the single source for every number quoted in `reports/findings.md` and the README.
- Tests (`pytest`) run on synthetic fixtures only and take ~2 seconds.

---

## 12. Known data-quality artefacts (observed, documented, not "fixed")

| Observation | Count | Treatment |
|---|---|---|
| Orders missing `order_approved_at` | 160 | Not used in any KPI |
| Orders missing carrier hand-off date | 1,783 | Not used; delivery analysis uses customer delivery date |
| Orders missing customer delivery date | 2,965 | Excluded from delivery-duration stats; still eligible orders |
| Duplicate `review_id` rows | 814 | Collapsed to latest review per order |
| Orders with multiple review rows | ~9.7% of reviewed orders | Latest review kept; score summarised separately from value |
| Multi-item, multi-seller, multi-payment orders | common | Aggregated before join (see §4) |
| Early cohorts (Sep–Dec 2016) | 1–262 customers | Shown, flagged as statistically fragile |

Negative values were searched for in price, freight and payment columns during the
audit; the pipeline records zero negative amounts in the eligible population, so none
were removed or clipped silently.
