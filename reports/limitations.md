# Limitations

Read this before quoting any number from the project. The analysis is honest about
what it can and cannot establish; the limitations are not boilerplate.

---

## 1. What the data is (and is not)

| | |
|---|---|
| Type | Historical retrospective, one public marketplace dataset |
| Period | 15 Sep 2016 → 29 Aug 2018 |
| Real-time? | **No.** Nothing here is a live feed; the newest record is 29 Aug 2018 |
| Olist's internal data? | **No.** Public release only; no internal systems, campaigns or CRM |
| My affiliation | **None.** Independent student case study |

Nothing in this project describes Olist's current operations, its marketing campaigns
or its proprietary customer intelligence. The segments are analytical constructs
created for this study, not Olist's customer classifications.

---

## 2. Data completeness limits (structural)

The source contains **no**:

- costs, COGS or contribution margin → no profitability, no CLV in the profit sense;
- acquisition cost or channel → no CAC payback, no channel-level retention;
- campaign exposure, contact history, discounts or offers → **no intervention can be
  evaluated, and no uplift can be attributed**;
- returns, refunds or complaint resolution outcomes → service failure cannot be
  separated from demand loss;
- unsubscribe, cancellation-intent or competitor-purchase signals → **inactivity can
  never be called churn**.

Every limitation downstream is a consequence of this list.

---

## 3. Analytical limitations

### 3.1 Revenue-like measures only
`merchandise_value` = Σ item prices. It excludes freight (reported separately) and is
not profit. Customers with a high historical value are high-*revenue* customers, not
necessarily high-*margin* ones.

### 3.2 Inactivity is not churn
Recency thresholds (121.5 / 240.3 days) are anchored on observed repurchase gaps. A
"Long Inactive" label means "no eligible order in that window", not "gone". Seasonal,
single-occasion and off-platform behaviour is invisible.

### 3.3 Right-censoring
- Recent customers have had less time to repeat: 11,971 customers are within 60 days
  of the analysis date and are labelled "insufficient history" rather than inactive.
- 18,741 of 96,478 orders had an incomplete 90-day follow-up window and were excluded
  from the service→repeat analysis.
- Early cohorts (Sep–Dec 2016: 1 to 262 customers) are arithmetically valid but
  statistically fragile; the app flags them.

### 3.4 Specific-month retention is not repeat rate
Month-1 retention (0.48%) counts customers active in that exact month. The
ever-repeated rate (3.00%) counts customers who returned at any time. They answer
different questions; conflating them is the most common error in cohort reporting. The
app labels them separately and never substitutes one for the other.

### 3.5 Associations, not causation
Delivery lateness and review score are strongly associated (mean score 4.30 → 1.73),
but the data contains **no randomisation and no counterfactual**. Lateness is not
assigned at random: remote regions and heavy/fragile categories are harder to serve
and more likely to disappoint. All service results are descriptive.

Notably, the *retention* association is negligible (Cramér's V ≈ 0.01), so the
intuitive "bad service causes churn" claim is **not** supported here — see
`findings.md` F8 for the four plausible explanations and why the study is also
underpowered for small true effects at a 2.3% base rate.

### 3.6 Identity approximation
`customer_unique_id` is the best available person key, not a guaranteed human key.
Shared devices, household accounts and duplicate registrations can merge or split
identities. Using `customer_id` instead would be a total error (3,345 duplicates,
repeat rate → 0%).

### 3.7 Segment boundaries are soft
The p80 high-value cut and the day thresholds are defensible, dated judgement calls.
A customer one day or one real on either side of a boundary changes segment. Segments
are a snapshot at one date; no migration between segments is modelled.

### 3.8 Category simplification
`main_category` is the highest-value item in an order. Multi-category orders (common
in this dataset) are simplified, so category-level conclusions understate mixed baskets.

### 3.9 Predictive modelling deliberately not shipped
A churn/next-purchase model is described in the project plan as a stretch goal. It is
**not** implemented, for three reasons:

1. The base rate is ~3% ever-repeat (2.3% within 90 days). With that class balance, a
   model can reach a respectable PR-AUC while adding little over a simple recency rule,
   and the comparison must be shown honestly to be worth anything.
2. The features that would carry most of the signal (category affinity, campaign
   exposure, service outcomes at contact level) are absent or thin in the source.
3. Presenting an unvalidated model would violate the project's own rule that measured
   findings, predictive estimates and hypothetical scenarios must be kept separate.

The correct next step, if extended, is a temporal split with a recency-based baseline,
reported lift over that baseline, and explicit PR-AUC and calibration — not a headline
accuracy number.

### 3.10 Synthetic test fixtures
Unit tests use small fabricated frames (< 10 rows) to verify formulas. They are never
loaded by the application, never merged with the real data, and never used to state a
business finding.

---

## 4. What this project can responsibly claim

- Measured descriptive statistics about one historical marketplace dataset.
- A reproducible pipeline whose definitions are explicit and tested.
- Evidence-based **hypotheses** about where a marketing manager should look next.
- Explicit statements of what data would be needed to move from description to proof.

## 5. What it must never be used to claim

- That Olist's retention is 3% *today*.
- That any marketing intervention would raise retention by any amount.
- That a segment is "profitable" or has a particular CLV.
- That delivery improvement would have recovered a specific number of customers.
- That the recommendations have been validated, tested or endorsed.
