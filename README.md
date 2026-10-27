# Customer Intelligence & Retention Lab

**From historical purchase behavior to evidence-based customer retention decisions.**

An end-to-end customer analytics product built on the public **Brazilian E-Commerce
Public Dataset by Olist** (≈100k orders, 2016–2018). It audits and cleans the raw
transaction data, builds a customer-level view, quantifies retention with monthly
cohort analysis, segments customers with an interpretable RFM model, and maps each
segment to a marketing action with its evidence, KPI, experiment design and stated
uncertainty.

> **Independence and honesty statement.**
> This is an independent student case study built on a public, historical dataset.
> I am not an employee or consultant of Olist and have no access to its internal
> systems, campaigns or customer records. Nothing here describes Olist's actual
> operations or marketing. The data is **historical, not a real-time feed**, and all
> figures are **revenue-like merchandise value, not profit**. Every number in this
> README is computed from the dataset by `scripts/generate_reports.py`; none of it
> is invented.

---

## 1. Project summary

Ecommerce businesses spend most of their acquisition budget on customers who buy
once. The commercially interesting question is not "how many orders did we get?" but
"who comes back, who is drifting away, and where would an intervention plausibly
change the outcome?"

This project answers that question for one real marketplace dataset, and is explicit
about the boundary between what the data **can** show (measured behaviour) and what it
**cannot** (causal effects of marketing, profit, true CLV).

**Headline results (measured on 96,478 eligible delivered orders, 93,358 customers):**

| Metric | Value |
|---|---|
| Repeat-purchase rate (≥2 orders, ever) | **3.00%** (2,801 customers) |
| Month-1 cohort retention (pooled) | **0.48%** |
| Median days from first to second order (among repeaters) | **29 days** |
| Share of repeat gaps within 30 / 90 days | **50.1% / 68.9%** |
| Historical merchandise value | **R$ 13.22M** (freight R$ 2.20M excluded) |
| Average order value | **R$ 137.04** (median R$ 86.58) |
| Value held by top 10% / top 20% of customers | **41.1% / 56.6%** (Gini 0.52) |

The most decision-relevant finding is uncomfortable: **repeat purchasing is rare
(3%), and it happens fast when it happens at all (median 29 days).** That single
pattern reshapes retention strategy — the leverage sits in the first weeks after a
first order, not in long-run win-back campaigns.

---

## 2. Business questions this supports

1. What proportion of customers place a second order, and how does that vary by first-purchase cohort?
2. How long does a repeat purchase typically take, and how should that shape re-engagement timing?
3. How concentrated is historical value, and which customers should be protected?
4. Which customers are recent, lapsing, dormant or too new to judge?
5. Do service experience (review score, delivery lateness) and later purchasing move together?
6. Which segments warrant a different marketing action — and what experiment would prove it?
7. What data would be needed to estimate campaign profitability or prove that retention interventions work?

---

## 3. Dataset

| | |
|---|---|
| **Source** | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, `olistbr/brazilian-ecommerce`) |
| **Publisher** | Olist — a Brazilian marketplace |
| **Period in this analysis** | 15 Sep 2016 → 29 Aug 2018 (714 days, eligible orders) |
| **Licence** | **CC BY-NC-SA 4.0** — non-commercial use, share-alike, attribution required |
| **Size** | 99,441 orders across 9 CSV tables (~45 MB zipped) |
| **Cost** | Free. No API key, no paid tier, no authentication bypass |

Tables used: `orders`, `customers`, `order_items`, `order_payments`,
`order_reviews`, `products`, `sellers`, `product_category_name_translation`.
`geolocation` (1M rows) ships with the dataset but is not used by any KPI, so the
app deliberately does not load it.

**The dataset does not contain** costs, margins, campaign exposure, contact
history, acquisition channel, CAC, returns reasons, or unsubscribe events. Every
limitation in the reports traces back to this list.

---

## 4. Technical architecture

```
olist-customer-intelligence/
├── app.py                      # Streamlit router: navigation, chrome, empty states
├── config/settings.py          # all tunable rules (eligibility, thresholds, palette)
├── src/
│   ├── data_loader.py          # discovery, zip extraction, typed loading, missing-data UX
│   ├── data_validation.py      # raw audits + analytical quality checks (16 assertions)
│   ├── cleaning.py             # eligibility, child-table aggregation, one-row-per-order tables
│   ├── customer_metrics.py     # KPIs, repeat purchase, intervals, concentration, recency
│   ├── cohort_analysis.py      # monthly cohorts, retention matrix, cumulative repeat
│   ├── segmentation.py         # rule-based RFM + k-means extension
│   ├── service_analysis.py     # reviews, delivery, right-censored repeat outcome, effect sizes
│   ├── recommendations.py      # recommendation catalogue + scenario simulator
│   ├── visualizations.py       # shared Plotly styling and chart builders
│   └── ui/                     # Streamlit components + one module per page
├── scripts/generate_reports.py # regenerates reports/findings_metrics.json
├── notebooks/01_exploratory_analysis.ipynb
├── tests/                      # 76 pytest tests on small synthetic fixtures
├── reports/                    # data dictionary, methodology, findings, limitations
│   └── findings_metrics.json   # machine-readable metrics behind every quoted number
├── assets/screenshots/         # real screenshots from the running app
└── deploy/vercel/              # Vercel build: Flask function + static dashboard
```

**Design principle:** app code contains no business logic. Pages call functions in
`src/`, the notebook calls the same functions, the tests call the same functions, and
`scripts/generate_reports.py` calls the same functions. There is exactly one
implementation of every metric.

### Data pipeline

```
raw CSVs
   │  data_loader: typed load, date parsing
   ▼
orders ── eligibility filter (delivered) ── customer identity join (customer_unique_id)
   │
   ├── order_items  ─ aggregate per order ─┐
   ├── payments     ─ aggregate per order ─┤── join one-to-one ── order_values (1 row/order)
   └── reviews      ─ latest per order ────┘
                                             │
              ┌──────────────────────────────┼───────────────────────────────┐
              ▼                              ▼                               ▼
       customers (1 row/person)        cohort table                  orders + 90-day outcome
              │                              │                               │
       RFM segmentation               retention curves                service associations
```

Child tables are **always** aggregated before joining, so `merchandise_value` can
never be double-counted by a one-to-many join. This is verified on every run by two
independent reconciliation checks.

---

## 5. Methodology (summary — full version in [`reports/methodology.md`](reports/methodology.md))

**Order eligibility.** `order_status = 'delivered'` with a valid purchase timestamp.
99,441 → 96,478 orders (97.0%). Canceled (625) and unavailable (609) orders are not
purchases; in-flight statuses (1,729 orders) are excluded because they concentrate at
the end of the window and would bias cohort maturity. The rule is identical in every
month, and a broader variant is available in `config/settings.py` for sensitivity.

**Customer identity.** The raw `customers` table has 99,441 `customer_id` values but
only **96,096 `customer_unique_id` values** — 3,345 order-scoped duplicates. `customer_id`
is issued *per order*, so using it would make the repeat rate exactly zero by
construction. Every customer-level metric uses `customer_unique_id`.

**Retention.** Monthly first-purchase cohorts. `month_index` = calendar months since
the cohort month. Retention in month *n* = cohort members with ≥1 eligible order in
that specific month ÷ cohort size. Month 0 is 100% by construction. **Unobservable
future months are never created** (rather than being filled with zero), so they cannot
be mistaken for zero retention. Because frequency is so low, "specific-month
retention" (0.48% at M1) and "ever repeated" (3.00%) are reported separately and are
never conflated.

**Segmentation.** Interpretable rule-based RFM with data-derived thresholds
(lapsing = max(90, p75 repeat gap) = 121 days; inactive = max(180, p90 gap) = 240 days;
high value = p80 of historical value). Seven mutually exclusive, exhaustive segments.
A k-means variant (log-transformed, scaled, k chosen by silhouette) ships as an
extension with an explicit warning that clusters are not automatically useful.

**Inactivity.** Labels are `Recently active` / `Potentially lapsing` / `Long inactive` /
`New / insufficient history`. These are **observed states, not churn predictions** — the
dataset has no cancellation or competitor-purchase data.

**Recommendations.** A rules engine maps measured segment characteristics to proposed
actions. Each recommendation states target, evidence (with real numbers), mechanism,
KPI, experiment design, costs/risks and uncertainty. All are **hypotheses**.

---

## 6. Results (computed from the real dataset)

Full detail, tables and chart references: [`reports/findings.md`](reports/findings.md).
Regenerate with `python scripts/generate_reports.py`.

### 6.1 Repeat purchasing is rare, and fast when it happens

- **3.00%** of customers (2,801 of 93,358) ever placed a second eligible order.
- Median time from first to second order: **29 days**; 50.1% of all observed repeat
  gaps occur within 30 days, 68.9% within 90 days. The 75th percentile gap is
  121 days and the 90th is 240 days.
- Implication: re-engagement that fires 6–12 months after a first purchase is aimed at
  a window in which almost nothing normally happens.

### 6.2 Month-level retention is very low, and it is not the same as repeat rate

- Pooled month-1 retention: **0.48%**; month-2: **0.34%**; month-12: **0.17%**.
- Median 12-month cumulative repeat rate across the 11 mature cohorts: **2.91%**.
- 23 cohorts exist; only **11** are mature enough (12+ observable months) to carry a
  12-month figure. Later cohorts are shown with their maturity, never padded.

### 6.3 Value is concentrated, but less extremely than the usual 80/20 story

- Top 1% of customers → **11.5%** of value; top 10% → **41.1%**; top 20% → **56.6%**.
- Gini of customer value: **0.52**.
- Implication: a top-decile-only strategy addresses well under half of historical
  revenue; the one-time-buyer tail still carries most of it.

### 6.4 The RFM segments (rule-based, data-derived thresholds)

| Segment | Customers | Share of customers | Share of value | Avg recency (days) | Avg orders |
|---|---|---|---|---|---|
| High-Value Active Repeat | 460 | 0.49% | 1.44% | 62 | 2.25 |
| Active Repeat | 385 | 0.41% | 0.32% | 62 | 2.03 |
| Lapsing Repeat | 840 | 0.90% | 1.63% | 182 | 2.12 |
| Recent One-Time Buyer | 12,427 | 13.3% | 13.5% | 92 | 1.00 |
| Dormant One-Time Buyer | 25,529 | 27.3% | 26.0% | 181 | 1.00 |
| Long Inactive | 41,746 | 44.7% | 44.7% | 380 | 1.03 |
| New / Insufficient History | 11,971 | 12.8% | 12.4% | 30 | 1.00 |

Because repeat buyers are only 3% of the base, the "high-value repeat" segment is
**small in absolute terms** (460 customers, 1.4% of value). Any recommendation that
looks like "focus on your VIPs" needs that context: this business's problem is not
protecting a large VIP tier, it is that almost nobody returns.

### 6.5 Service experience: strong effect on satisfaction, no measurable effect on returning

- Reviews cover **99.3%** of eligible orders; mean score **4.16**; 59.2% are 5s;
  **12.8%** are 1–2.
- Delivery: median **10.2 days**, 90th percentile 23.1 days; **8.1% of orders arrive
  after the estimated date** (median 5.8 days late when late).
- Lateness and satisfaction are strongly related: mean review score **4.30** for early
  deliveries vs **1.73** for 8+ days late; share of 1–2 scores rises from 9.2% to 78.4%.
- **Lateness and later repeat purchasing are not measurably related** (Cramér's V =
  0.007, p = 0.45, n = 77,735), and review score shows only a negligible association
  (V = 0.011, p = 0.03, n = 77,174 — statistically detectable, practically meaningless).
- Base rate for "another order within 90 days": **2.3%** of the 77,737 orders with a
  fully observed follow-up window (18,741 orders were right-censored and excluded).

This is the most important honest result in the project: service quality clearly
drives **satisfaction**, but in this dataset it does not measurably predict
**retention**. A win-back programme justified by "bad reviews cause churn" would be
resting on a claim this data does not support.

---

## 7. Business recommendations (hypotheses, not proven interventions)

| Priority | Segment | Proposed action | Primary KPI | Experiment |
|---|---|---|---|---|
| High | Lapsing Repeat (840) | Behaviour-based re-engagement tied to past purchases; incentive only for non-responders | 60-day reactivation rate, incremental margin | 3-arm test: no contact / content / content + incentive |
| High | Recent One-Time Buyer (12,427) | Onboarding sequence timed near the 29-day median second-purchase gap | 90-day second-purchase rate | Randomised at first order, held-out control |
| High | Service experience (all segments) | Fix delivery-promise accuracy and service recovery before funding discounts | Late-delivery %, 1–2 score %, repeat rate | Staggered rollout / delivery-buffer A/B |
| Medium | High-Value Active Repeat (460) | Service-led loyalty perks and cross-sell, not discounts on habitual items | Orders per customer, margin per customer | 80/20 hold-out for 6 months |
| Medium | Dormant One-Time Buyer (25,529) | Content-led win-back, prioritised by first-order category and review score; no blanket discount | Win-back rate, incremental margin | Observational screen first, then randomised test |
| Low | Long Inactive (41,746) | Low-cost automated channels only; small budget-capped pilot before any paid reactivation | Cost per incremental order | Budget-capped pilot vs held-out control |
| Low | New / Insufficient History (11,971) | No campaign — measurement control group | 90-day repeat once observable | None (guard against recency bias) |

The scenario simulator shows unit economics for any of these under user-supplied
assumptions, and reports the **break-even incremental rate** next to the ROI, because
a campaign only pays if the required uplift is plausible.

---

## 8. Limitations

- **Historical, not real-time.** The dataset ends 29 Aug 2018. Nothing here is live.
- **Revenue ≠ profit.** All monetary figures are merchandise value. No COGS, margin,
  campaign cost or CAC exists in the source, so no profitability or true CLV claim is made.
- **No campaign data**, so no intervention can be evaluated and no uplift can be attributed.
- **Inactivity ≠ churn.** Recency is an observed state; cancellations and competitor
  purchases are invisible.
- **Associations ≠ causation.** No randomisation, no counterfactual. Delivery/review
  correlations are descriptive only.
- **Censoring at both ends.** Recent customers have had less time to repeat; the
  Sep–Dec 2016 cohorts contain 1–262 customers and are statistically fragile.
- **Identity approximations.** `customer_unique_id` is the best available person key
  but not guaranteed to equal one human (shared devices, duplicate accounts).
- **Segment thresholds are judgement calls**, anchored on observed repurchase gaps.
- **Test fixtures are synthetic** (< 10 rows) and never mix with the real data.

---

## 9. Installation and run

Requires Python 3.10+ (developed on 3.13). No paid services, no API keys.

```bash
git clone https://github.com/Igneelsane/olist-customer-intelligence-retention-lab.git olist-customer-intelligence
cd olist-customer-intelligence

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Get the data (free Kaggle account required)
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
#    Option A: unzip the download into data/raw/
#    Option B: drop the zip in data/raw/ as brazilian-ecommerce.zip (auto-extracted)
#    Option C: point the app anywhere else:
export OLIST_DATA_DIR=/absolute/path/to/olist/csvs     # Windows: set OLIST_DATA_DIR=...

# 2. Run the test suite
pytest

# 3. Launch the app
streamlit run app.py
```

**Drop the dataset in its canonical downloaded form** (`data/raw/brazilian-ecommerce.zip`).
On first launch the loader extracts the CSVs automatically — that path is tested — so the
repository stays small and no raw data is ever committed to Git.

The app opens at `http://localhost:8501`. First launch builds the analytical tables
(~10 s) and caches them as parquet in `data/processed/`; later launches are fast.
If the CSVs are missing, the app shows step-by-step acquisition instructions and
**refuses to substitute synthetic records**.

Useful commands:

```bash
python scripts/generate_reports.py     # regenerate reports/findings_metrics.json
jupyter lab notebooks/01_exploratory_analysis.ipynb
```

### Hosting notes

| What | Where | Status |
|---|---|---|
| **Live dashboard** | **https://olist-customer-intelligence-retenti.vercel.app** | deployed from this repository |
| Full application (6 pages, Streamlit) | Streamlit Community Cloud — free, native, no code change | deployable; sleeps when idle |

The live dashboard is the Vercel build in `deploy/vercel/`: a Flask function that recomputes the
metrics on every request from a shipped, anonymised derived table, plus a static page that
renders only what the function returns. Its figures match the pandas pipeline exactly
(`/api/analytics?view=health` → 96,478 orders / 93,358 customers).

Streamlit **cannot** run on Vercel: Vercel Functions are request-scoped and cannot hold the
persistent WebSocket a Streamlit session needs. Rather than ship a screenshot site dressed up as
a deployment, `deploy/vercel/` contains a Vercel-native build that keeps the computation: a
Python function recomputes every metric on request from a compact anonymised derived table, and
a static dashboard renders only what the function returns (it prints an error instead of numbers
when the API is unavailable).

The deployed numbers were cross-checked against the pandas pipeline on the real dataset — repeat
rate, cohort retention, Gini, segment sizes, service base rates and Cramér's V all agree — and
the live site passed 28 browser checks (all five tabs, every table, both failure paths, zero
console errors). Endpoint list, payload description, platform limits and redeploy steps are in
[`deploy/vercel/README.md`](deploy/vercel/README.md).

---

## 10. Screenshots

Real captures from the running application (`assets/screenshots/`):

| Page | File |
|---|---|
| Executive Overview | `01_executive_overview.png` |
| Retention Intelligence | `02_retention_intelligence.png` |
| Cohort retention heatmap | `02b_cohort_heatmap.png` |
| Retention curves | `02c_retention_curves.png` |
| Customer Segmentation | `03_customer_segmentation.png` |
| Segment profile table | `03b_segment_profile.png` |
| Customer Experience | `04_customer_experience.png` |
| Retention Action Planner | `05_retention_action_planner.png` |
| Methodology & Data Quality | `06_methodology_data_quality.png` |

These are captures of the application as it actually ran locally
(Playwright, 1500×1000–1250 viewport), not mockups.

The Vercel dashboard (`deploy/vercel/public/index.html`) was captured the same way, driven by
the real analytics function on a local server:

| View | File |
|---|---|
| Overview — live site | `deploy_live_overview.png` |
| Retention cohorts — live site | `deploy_live_retention.png` |
| Methodology & integrity checks — live site | `deploy_live_methodology.png` |
| Failure state when the API is unreachable — live site | `deploy_live_api_down.png` |

---

## 11. Tests

76 automated tests, all passing (`pytest`):

| File | Covers |
|---|---|
| `test_cleaning.py` | eligibility, identity join, one-to-many aggregation, reconciliation, review de-duplication |
| `test_data_validation.py` | audit correctness, injected-failure detection (duplicate orders, orphan customers, negatives) |
| `test_customer_metrics.py` | repeat rate denominator, intervals, time-to-second-purchase, Gini/concentration, inactivity labels, threshold floors |
| `test_cohort_analysis.py` | month-0 = 100%, dedup within cohort-month, unobservable ≠ zero, pooled retention, cumulative ≠ sum of monthly |
| `test_segmentation.py` | quantile scoring, reversed recency, rule precedence, profile reconciliation, k-means structure |
| `test_recommendations.py` | simulator arithmetic (hand-computed), break-even identity, required recommendation fields |

Tests use **small synthetic fixtures only** to verify formulas; they never assert a
business finding.

---

## 12. Licence

Code: MIT (see `LICENSE`). Dataset: CC BY-NC-SA 4.0 — this project must not be used
commercially, and any redistribution of derived data must carry the same licence and
attribution to Olist.
