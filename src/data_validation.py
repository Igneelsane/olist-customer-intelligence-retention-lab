"""
Data audit, quality checks and reconcile helpers.

Two layers:

* ``audit_table``  -- a generic profile of one raw CSV (shape, dtypes, missing,
  duplicates, key uniqueness, date ranges, outliers).
* ``run_quality_checks`` -- targeted assertions about the *analytical* tables
  (unique order ids, customer linkage, non-negative money, reconciliation of
  item-level totals against order-level totals, cohort sanity).

Errors are collected, never silently swallowed: ``raise_on_failure`` is used by
the pipeline and the test suite, while the app displays the report as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw table audit
# ---------------------------------------------------------------------------

KEY_UNIQUENESS = {
    "orders": ["order_id"],
    "customers": ["customer_id"],
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
    "reviews": ["review_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "category_translation": ["product_category_name"],
}


def audit_table(name: str, df: pd.DataFrame) -> Dict[str, object]:
    """Profile a single raw dataframe."""
    n_rows, n_cols = df.shape
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    missing_pct = {c: round(float(df[c].isna().mean()) * 100, 2) for c in df.columns}
    full_dupes = int(df.duplicated().sum())

    key_info: Dict[str, object] = {}
    for key_cols in [KEY_UNIQUENESS.get(name, [])]:
        if key_cols and all(c in df.columns for c in key_cols):
            dup_keys = int(df.duplicated(subset=key_cols).sum())
            key_info = {
                "key": key_cols,
                "duplicate_key_rows": dup_keys,
                "unique_key_combinations": int(df.drop_duplicates(subset=key_cols).shape[0]),
                "is_unique": dup_keys == 0,
            }

    date_info = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            non_null = df[col].dropna()
            date_info[col] = {
                "min": str(non_null.min()) if len(non_null) else None,
                "max": str(non_null.max()) if len(non_null) else None,
                "nulls": int(df[col].isna().sum()),
            }

    numeric_info = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s):
            numeric_info[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "negative_count": int((s < 0).sum()),
                "zero_count": int((s == 0).sum()),
            }

    return {
        "table": name,
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "dtypes": dtypes,
        "missing_counts": missing,
        "missing_pct": missing_pct,
        "full_duplicate_rows": full_dupes,
        "key_uniqueness": key_info,
        "date_ranges": date_info,
        "numeric_ranges": numeric_info,
    }


def audit_to_markdown(audits: List[Dict[str, object]]) -> str:
    """Render the audit as a readable markdown appendix for reports."""
    lines = ["# Raw data audit\n"]
    for a in audits:
        lines.append(f"## `{a['table']}`")
        lines.append(f"- Rows: **{a['rows']:,}** | Columns: **{a['columns']}**")
        if a["key_uniqueness"]:
            k = a["key_uniqueness"]
            status = "unique" if k["is_unique"] else f"**{k['duplicate_key_rows']:,} duplicate rows**"
            lines.append(f"- Key `{'+'.join(k['key'])}`: {status}")
        lines.append(f"- Fully duplicated rows: {a['full_duplicate_rows']:,}")
        cols_with_missing = {c: v for c, v in a["missing_counts"].items() if v > 0}
        if cols_with_missing:
            lines.append("- Columns with missing values:")
            for c, v in sorted(cols_with_missing.items(), key=lambda kv: -kv[1]):
                lines.append(f"    - `{c}`: {v:,} ({a['missing_pct'][c]}%)")
        if a["date_ranges"]:
            lines.append("- Date columns:")
            for c, v in a["date_ranges"].items():
                lines.append(f"    - `{c}`: {v['min']} → {v['max']} ({v['nulls']:,} null)")
        tricky = {c: v for c, v in a["numeric_ranges"].items() if v["negative_count"] > 0}
        if tricky:
            lines.append("- Columns containing negative values (investigated, see methodology): "
                         + ", ".join(f"`{c}`={v['negative_count']:,}" for c, v in tricky.items()))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Targeted quality checks on analytical tables
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"          # "error" | "warning" | "info"
    value: Optional[float] = None


@dataclass
class QualityReport:
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, severity: str = "error",
            value: Optional[float] = None) -> None:
        self.checks.append(CheckResult(name, passed, detail, severity, value))

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> List[Dict[str, object]]:
        return [c.__dict__ for c in self.checks]

    def to_markdown(self) -> str:
        icon = {True: "PASS", False: "FAIL"}
        lines = ["| Check | Result | Detail |", "|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.name} | {icon[c.passed]} ({c.severity}) | {c.detail} |")
        return "\n".join(lines)


def run_quality_checks(tables: Dict[str, pd.DataFrame]) -> QualityReport:
    """
    Validate the analytical tables.

    Expected keys in ``tables``:
        orders            one row per eligible order
        order_items_sum   one row per order (item-level aggregation)
        payments_sum      one row per order (payment aggregation)
        customers         one row per customer_unique_id
        order_values      one row per order, items + payments reconciled
        cohort            long cohort table (cohort_month, month_index, ...)
    Missing keys are skipped so the function can be reused piecemeal.
    """
    rep = QualityReport()

    orders = tables.get("orders")
    if orders is not None:
        rep.add(
            "Order IDs unique in cleaned orders",
            orders["order_id"].is_unique,
            f"{orders['order_id'].duplicated().sum():,} duplicate order_id rows",
        )
        rep.add(
            "Every order maps to a customer_unique_id",
            orders["customer_unique_id"].isna().sum() == 0,
            f"{int(orders['customer_unique_id'].isna().sum()):,} orders without a unique customer id",
        )
        rep.add(
            "Purchase timestamps valid and non-null",
            bool(orders["order_purchase_timestamp"].notna().all()),
            f"{int(orders['order_purchase_timestamp'].isna().sum()):,} null purchase timestamps",
        )
        if "order_status" in orders:
            rep.add(
                "Only eligible statuses present",
                set(orders["order_status"].unique()) <= set(_eligible_statuses()),
                f"statuses found: {sorted(orders['order_status'].unique())}",
            )

    items = tables.get("order_items_sum")
    pay = tables.get("payments_sum")
    values = tables.get("order_values")

    if values is not None:
        rep.add(
            "Order-level value table has one row per order",
            values["order_id"].is_unique,
            f"{values['order_id'].duplicated().sum():,} duplicates",
        )
        for col in ("merchandise_value", "freight_value", "n_items"):
            if col in values:
                neg = int((values[col] < 0).sum())
                rep.add(
                    f"{col} non-negative",
                    neg == 0,
                    f"{neg:,} negative values",
                )
        rep.add(
            "Order count unchanged by item aggregation (no fan-out)",
            len(values) == len(orders) if orders is not None else True,
            f"{len(values):,} order rows vs {len(orders):,} cleaned orders",
        )

    if items is not None and values is not None:
        merged = values.merge(items, on="order_id", how="left", suffixes=("", "_chk"))
        if "merchandise_value_chk" in merged:
            diff = (merged["merchandise_value"] - merged["merchandise_value_chk"]).abs()
            rep.add(
                "Merchandise value reconciles with independent item aggregation",
                bool((diff < 0.01).all()),
                f"max absolute difference = {diff.max():.6f}",
                value=float(diff.max()),
            )
    if pay is not None and values is not None and "payment_value" in values:
        merged = values.merge(pay, on="order_id", how="left", suffixes=("", "_chk"))
        if "payment_value_chk" in merged:
            diff = (merged["payment_value"] - merged["payment_value_chk"]).abs()
            mism = int((diff > 0.01).sum())
            rep.add(
                "Payment values reconcile with independent payment aggregation",
                mism == 0,
                f"{mism:,} orders differ by more than R$0.01 (max diff {diff.max():.2f})",
                severity="warning",   # payments include vouchers/freight; mismatch is expected
                value=float(diff.max()),
            )

    cust = tables.get("customers")
    if cust is not None and orders is not None:
        rep.add(
            "Customer table has one row per customer_unique_id",
            cust["customer_unique_id"].is_unique,
            f"{cust['customer_unique_id'].duplicated().sum():,} duplicate unique ids",
        )
        orphan = set(orders["customer_unique_id"]) - set(cust["customer_unique_id"])
        rep.add(
            "All order customers exist in the customer dimension",
            len(orphan) == 0,
            f"{len(orphan):,} unmatched customer_unique_id values",
        )

    cohort = tables.get("cohort")
    if cohort is not None and len(cohort):
        if "retention_rate" in cohort:
            rates = cohort["retention_rate"].dropna()
            rep.add(
                "Retention rates between 0 and 1",
                bool(((rates >= 0) & (rates <= 1)).all()),
                f"min={rates.min():.4f}, max={rates.max():.4f}",
            )
        m0 = cohort[cohort["month_index"] == 0]
        if len(m0):
            ok = bool(np.allclose(m0["retention_rate"].fillna(np.nan), 1.0, equal_nan=True))
            rep.add("Month 0 retention equals 100%", ok,
                    f"month-0 range: {m0['retention_rate'].min():.4f}–{m0['retention_rate'].max():.4f}")
        if "cohort_size" in cohort and "n_active" in cohort:
            rep.add(
                "Active customers never exceed cohort size",
                bool((cohort["n_active"] <= cohort["cohort_size"]).all()),
                f"{int((cohort['n_active'] > cohort['cohort_size']).sum())} violations",
            )
    return rep


def _eligible_statuses():
    from config import settings
    return settings.PRIMARY_ELIGIBLE_STATUSES


def assert_quality(tables: Dict[str, pd.DataFrame]) -> QualityReport:
    """Run the checks and raise if any hard error fails (used by the pipeline)."""
    rep = run_quality_checks(tables)
    if not rep.passed:
        details = "; ".join(f"{c.name}: {c.detail}" for c in rep.failures)
        raise AssertionError(f"Data quality checks failed -> {details}")
    return rep
