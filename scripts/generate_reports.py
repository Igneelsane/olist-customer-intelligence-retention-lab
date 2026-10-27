"""
Reproducible findings generator.

Runs the full pipeline against the real dataset and writes every number quoted
in reports/findings.md to reports/findings_metrics.json. Nothing in the written
reports is typed by hand: if the data changes, re-run this script.

Usage:
    python scripts/generate_reports.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings                     # noqa: E402
from src import (                               # noqa: E402
    cleaning, cohort_analysis, customer_metrics, data_loader, data_validation,
    recommendations, segmentation, service_analysis,
)


def _r(x, n=4):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return round(float(x), n)
    except Exception:
        return x


def main() -> dict:
    raw = data_loader.load_raw_tables(tables=(
        "orders", "customers", "order_items", "payments", "reviews",
        "products", "sellers", "category_translation"))
    tables = cleaning.build_analytical_tables(raw)
    orders = tables["orders"]

    metrics = customer_metrics.build_customer_metrics(orders, tables["customers"])
    cohort = cohort_analysis.build_cohort_table(orders)
    seg = segmentation.run_segmentation(
        metrics.customers, method="rule_based",
        lapsing_days=metrics.thresholds["lapsing_days"],
        inactive_days=metrics.thresholds["inactive_days"])

    quality = data_validation.run_quality_checks({
        "orders": orders, "order_values": orders,
        "order_items_sum": tables["order_items_sum"],
        "payments_sum": tables["payments_sum"],
        "customers": tables["customers"], "cohort": cohort})

    kpis = cohort_analysis.retention_kpis(cohort, orders)
    pooled = cohort_analysis.weighted_retention_by_month(cohort, upto=12)
    cumulative = cohort_analysis.cumulative_repeat_rate(cohort, orders, upto=12)

    orders_outcome = service_analysis.add_repeat_outcome(
        orders, settings.SERVICE_REPEAT_WINDOW_DAYS, metrics.dates["as_of_date"])
    orders_outcome = service_analysis.late_delivery_bins(orders_outcome)
    bands = service_analysis.score_bands(orders_outcome)

    by_band = service_analysis.repeat_rate_by_group(bands, "score_band")
    by_late = service_analysis.repeat_rate_by_group(orders_outcome, "late_bin")
    test_band = service_analysis.association_tests(bands, "score_band")
    test_late = service_analysis.association_tests(orders_outcome, "late_bin")

    recs = recommendations.build_recommendations(
        seg.profile, metrics.summary, metrics.thresholds,
        service_analysis.review_summary(orders), service_analysis.delivery_summary(orders))

    audit = [data_validation.audit_table(n, raw[n]) for n in
             ("orders", "customers", "order_items", "payments", "reviews")]

    status_counts = raw["orders"]["order_status"].value_counts().to_dict()
    n_cust_raw = int(raw["customers"]["customer_id"].nunique())
    n_unique_raw = int(raw["customers"]["customer_unique_id"].nunique())

    out = {
        "generated_from": str(data_loader.available_files().get("orders")),
        "summary": {k: _r(v) for k, v in metrics.summary.items()},
        "thresholds": {k: _r(v) for k, v in metrics.thresholds.items()},
        "retention": {k: _r(v) for k, v in kpis.items()},
        "pooled_retention_months_1_to_12": [
            {"month_index": int(r.month_index), "retention_rate": _r(r.retention_rate),
             "n_active": int(r.n_active), "n_eligible": int(r.n_eligible),
             "n_cohorts": int(r.n_cohorts)}
            for r in pooled.itertuples()],
        "cumulative_repeat_12m": [
            {"cohort": str(r.cohort_month.date()), "cohort_size": int(r.cohort_size),
             "repeaters": int(r.n_cumulative_repeaters),
             "rate": _r(r.cumulative_repeat_rate)}
            for r in cumulative.itertuples()],
        "segment_profile": json.loads(seg.profile.to_json(orient="records")),
        "quality": {"passed": quality.passed,
                    "n_checks": len(quality.checks),
                    "failures": [c.__dict__ for c in quality.failures],
                    "warnings": [c.__dict__ for c in quality.warnings]},
        "service": {
            "reviews": {k: ({str(kk): _r(vv) for kk, vv in v.items()}
                            if isinstance(v, dict) else _r(v))
                        for k, v in service_analysis.review_summary(orders).items()},
            "delivery": {k: _r(v) for k, v in service_analysis.delivery_summary(orders).items()},
            "repeat_within_window_by_score": json.loads(by_band.to_json(orient="records")),
            "repeat_within_window_by_lateness": json.loads(by_late.to_json(orient="records")),
            "chi2_score_band": {k: _r(v) for k, v in test_band.items()},
            "chi2_lateness": {k: _r(v) for k, v in test_late.items()},
            "n_orders_used": int(orders_outcome["repeat_within_window"].notna().sum()),
            "n_orders_censored": int(orders_outcome["censored"].sum()),
        },
        "order_status_raw": {str(k): int(v) for k, v in status_counts.items()},
        "identity": {"raw_customer_ids": n_cust_raw, "raw_unique_ids": n_unique_raw,
                     "extra_order_scoped_ids": n_cust_raw - n_unique_raw},
        "cleaning_log": tables["cleaning_log"].steps,
        "audit": [{"table": a["table"], "rows": a["rows"], "columns": a["columns"],
                   "duplicate_key_rows": a["key_uniqueness"].get("duplicate_key_rows")
                   if a["key_uniqueness"] else None,
                   "missing": {k: v for k, v in a["missing_counts"].items() if v}}
                  for a in audit],
        "recommendations": [
            {"segment": r.segment, "priority": r.priority, "headline": r.headline,
             "evidence": r.evidence} for r in recs],
    }

    path = ROOT / "reports" / "findings_metrics.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
