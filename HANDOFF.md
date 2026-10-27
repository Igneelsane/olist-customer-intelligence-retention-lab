# Handoff report

Status of the **Customer Intelligence & Retention Lab** at the end of the build.
Every claim below corresponds to something that was actually executed in this
workspace; nothing is asserted without evidence.

---

## 1. Dataset — found and used (real data)

| | |
|---|---|
| Source | Official Kaggle endpoint for `olistbr/brazilian-ecommerce` (public download, no auth bypass) |
| Archive | `data/raw/brazilian-ecommerce.zip` (44,717,580 bytes — matches Kaggle's stated size) |
| Extracted CSVs | produced automatically from the zip on first launch (verified: extraction + full pipeline in ~34 s from a clean folder). The workspace keeps only the original archive so the raw CSVs are never committed |
| Rows verified | orders 99,441 · customers 99,441 · order_items 112,650 · payments 103,886 · reviews 99,224 · products 32,951 · sellers 3,095 · geolocation 1,000,163 · category translation 71 |
| Licence | CC BY-NC-SA 4.0 (non-commercial) |

No synthetic record is used anywhere in the analysis. The only synthetic data in the
repository lives in `tests/conftest.py` and is used exclusively for unit tests.

---

## 2. Files created

```
olist-customer-intelligence/
├── app.py                                Streamlit router, nav, empty/error states
├── README.md                             recruiter-facing documentation
├── requirements.txt                      pinned, free-only dependencies
├── LICENSE                               MIT (code) + dataset licence note
├── .gitignore                            excludes data, secrets, caches
├── pytest.ini
├── config/settings.py                    every tunable rule + palette
├── data/raw/                             9 real Olist CSVs + original zip
├── data/processed/                       parquet artifacts (gitignored)
├── notebooks/01_exploratory_analysis.ipynb   executed, 28 code cells, no errors
├── scripts/generate_reports.py           regenerates reports/findings_metrics.json
├── src/
│   ├── data_loader.py                    discovery, zip extraction, typed loading
│   ├── data_validation.py                raw audits + 16 quality checks
│   ├── cleaning.py                       eligibility, aggregation, order grain
│   ├── customer_metrics.py               KPIs, intervals, concentration, recency
│   ├── cohort_analysis.py                cohorts, retention matrix, cumulative repeat
│   ├── segmentation.py                   rule-based RFM + k-means extension
│   ├── service_analysis.py               reviews, delivery, censored outcome, effect sizes
│   ├── recommendations.py               recommendation catalogue + simulator
│   ├── visualizations.py                 shared Plotly styling/builders
│   └── ui/                               components.py, context.py, page_*.py (6 pages)
├── tests/                                6 files, 76 tests
├── reports/                              data_dictionary, methodology, findings,
│                                         limitations, findings_metrics.json
└── assets/screenshots/                   9 real screenshots from the running app
```

---

## 3. Features completed

**Data layer**
- Zip auto-extraction, column-selective loading (keeps memory under 600 MB), typed dtypes, date parsing.
- Missing-data behaviour: clear Kaggle instructions rendered in the app; the app refuses to invent data.
- Configurable dataset directory (`OLIST_DATA_DIR` env var or in-app field).

**Audit and validation**
- Per-table audit: shape, dtypes, missing %, duplicate keys, date ranges, negative values.
- 16 analytical checks re-run on every app start (uniqueness, identity coverage, non-negativity, no fan-out, item reconciliation, payment reconciliation, cohort sanity).
- Cleaning log recording every dropped row with the reason.

**Analytics**
- Order eligibility (delivered), customer identity on `customer_unique_id`.
- Customer-level metrics, repeat purchase, purchase-interval distribution, Gini/concentration.
- Monthly cohorts with explicit censoring (unobservable months absent, never zero).
- Rule-based RFM (7 segments, data-derived thresholds) + k-means extension with silhouette selection.
- Service analysis with Wilson CIs, χ², Cramér's V and right-censored repeat outcome.
- Recommendation catalogue (7 items, each with evidence/KPI/experiment/risks/uncertainty) and break-even scenario simulator.

**Application (6 pages)**
1. Executive Overview — KPIs, new-vs-returning trend, concentration curve, key findings, eligibility funnel.
2. Retention Intelligence — heatmap, curves, pooled retention, time-to-second-purchase, cohort table, filters.
3. Customer Segmentation — distribution, profile table, comparison charts, anonymised drill-down, k-means mode.
4. Customer Experience — reviews, delivery timing, repeat-by-group with CIs, effect-size-driven conclusion.
5. Retention Action Planner — evidence-linked recommendations + low/base/high simulator with break-even.
6. Methodology & Data Quality — source, cleaning log, metric definitions, live checks, raw audit, limitations.

---

## 4. Tests actually run

```
$ pytest
76 passed in 1.86s
```

| File | Tests | Coverage |
|---|---|---|
| `test_cleaning.py` | 10 | eligibility, broad-status variant, identity join, one-to-many aggregation, no fan-out, reconciliation, review dedup, customer rollup, invalid timestamps |
| `test_data_validation.py` | 9 | audit fields, duplicate-key detection, negative detection, pass/fail paths, injected failures |
| `test_customer_metrics.py` | 14 | repeat-rate denominator, intervals, time-to-second, Gini, concentration, activity labels, insufficient-history rule, threshold floors, monthly split, summary |
| `test_cohort_analysis.py` | 16 | month-0 = 100%, dedup, unobservable ≠ zero, observable zero, pooled retention, cumulative ≠ sum, maturity, edge cases (single customer, empty) |
| `test_segmentation.py` | 13 | quantile scoring, reversed recency, constant series, rule precedence, profile reconciliation, value index, k-means structure |
| `test_recommendations.py` | 12 | simulator arithmetic (hand-computed), zero-participation, ROI edge, low/base/high ordering, break-even identity, required fields |

Manual verification also performed:
- all six pages rendered headlessly via `streamlit.testing.AppTest` — zero exceptions;
- k-means mode exercised in the app — zero exceptions;
- notebook executed end-to-end with `jupyter nbconvert --execute` — 38 outputs, zero errors;
- live server launched and screenshotted with Playwright.

---

## 5. Findings verified on the real dataset

Full detail in `reports/findings.md`; machine-readable copy in
`reports/findings_metrics.json`.

| Metric | Value |
|---|---|
| Eligible orders / customers | 96,478 / 93,358 |
| Observation window | 15 Sep 2016 → 29 Aug 2018 |
| Repeat-purchase rate (≥2 orders) | 3.00% (2,801 customers) |
| Month-1 / month-2 pooled retention | 0.483% / 0.336% |
| Median 12-month cumulative repeat rate | 2.91% (11 mature cohorts) |
| Median days to second order | 29.0 days; 50.1% of gaps ≤30d, 68.9% ≤90d |
| Merchandise / freight / payment value | R$ 13.22M / R$ 2.20M / R$ 15.42M |
| Average order value | R$ 137.04 (median R$ 86.58) |
| Value concentration | top 1% = 11.5%, top 10% = 41.1%, top 20% = 56.6%, Gini 0.52 |
| Review coverage / mean score / 1–2 share | 99.3% / 4.16 / 12.8% |
| Late deliveries | 8.1% (median 5.8 days late when late) |
| Review score: early vs 8+ days late | 4.30 vs 1.73 (1–2 share 9.2% → 78.4%) |
| Repeat-within-90-days by review band | 2.41–2.60%, χ² p=0.03, **Cramér's V = 0.011** |
| Repeat-within-90-days by lateness | p=0.45, **V = 0.007** (no measurable association) |
| Segments | 460 / 385 / 840 / 12,427 / 25,529 / 41,746 / 11,971 customers |

---

## 6. Commands

```bash
cd olist-customer-intelligence
pip install -r requirements.txt
pytest                                        # 76 tests
python scripts/generate_reports.py            # regenerate findings_metrics.json
streamlit run app.py                          # http://localhost:8501
jupyter lab notebooks/01_exploratory_analysis.ipynb
```

The app is currently running in this workspace on port 8501 (live preview).

---

## 7. Unresolved issues and honest gaps

1. **Predictive churn/next-purchase model is not implemented.** Deliberate — see
   `reports/limitations.md` §3.9. At a 2.3% base rate a model can look respectable
   while adding nothing over a recency rule, and the features needed for a fair
   comparison are absent from the source.
2. **Service→retention association is negligible**, so the corresponding
   recommendation is framed as "fix service for satisfaction, not for retention
   uplift" rather than a retention intervention. This is a finding, not a defect.
3. **Flow-margin polish:** the selection circle in the sidebar renders flush against
   the page label (looks like a bullet point). Cosmetic only; all navigation works.
4. **First load takes ~10 s** (full pipeline) and then caches to parquet. On a machine
   with < 1 GB free RAM the app would struggle; the loader was trimmed (column
   selection, geolocation excluded, raw frames released after the audit) so peak usage
   is now ~520 MB in the pipeline process.
5. **No deployment configuration** (Docker/Streamlit Cloud) — intentionally out of
   scope, and the dataset must not be committed.

---

## 8. Remaining limitations (summary)

Historical data only · no real-time feed · not Olist's internal data · revenue-like
values, no profit or margin · no campaign/exposure data so no causal claim about any
intervention · inactivity ≠ churn · associations ≠ causation · censoring at both ends
of the window · `customer_unique_id` is an approximation of a person · segment
thresholds are judgement calls · synthetic fixtures used only for tests.

---

## 9. What a reviewer should check first

1. `reports/methodology.md` §2 (eligibility) and §3 (identity) — the two decisions that
   determine whether the numbers mean anything.
2. `reports/findings.md` F8 — the result that contradicts the usual marketing intuition.
3. `src/cleaning.py::build_order_values` and `tests/test_cleaning.py::test_order_values_has_no_fan_out` —
   the double-counting guarantee.
4. `src/cohort_analysis.py::build_cohort_table` and the exhaustive month-0 test —
   the censoring guarantee.
