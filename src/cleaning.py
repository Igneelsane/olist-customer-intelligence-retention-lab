"""
Cleaning and analytical table construction.

Design rules that this module enforces
-------------------------------------
1. One row per order in every order-level table. One-to-many child tables
   (items, payments, reviews) are aggregated *first*, then joined.
2. Merchandise value, freight and payment value stay separate measures.
3. Customer-level identity uses ``customer_unique_id``; the order-level
   ``customer_id`` is documented as an order-scoped key (it changes when the
   same person orders again).
4. Every drop is counted and returned so it can be reported.

Entry point: ``build_analytical_tables``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from config import settings
from src import data_loader


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CleaningLog:
    """Record of what was dropped and why -- goes straight into the reports."""

    steps: List[Dict[str, object]] = field(default_factory=list)

    def add(self, step: str, rows_before: int, rows_after: int, reason: str) -> None:
        self.steps.append(
            {
                "step": step,
                "rows_before": int(rows_before),
                "rows_after": int(rows_after),
                "rows_removed": int(rows_before - rows_after),
                "reason": reason,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def to_markdown(self) -> str:
        if not self.steps:
            return "_No cleaning steps recorded._"
        lines = ["| Step | Rows before | Rows after | Removed | Reason |", "|---|---|---|---|---|"]
        for s in self.steps:
            lines.append(
                f"| {s['step']} | {s['rows_before']:,} | {s['rows_after']:,} | "
                f"{s['rows_removed']:,} | {s['reason']} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def clean_orders(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    statuses=settings.PRIMARY_ELIGIBLE_STATUSES,
    log: CleaningLog | None = None,
) -> pd.DataFrame:
    """
    Build the eligible-order table (one row per order) with customer identity.

    Eligibility = order_status in ``statuses`` AND a parseable purchase
    timestamp at or after the configured minimum date.
    """
    log = log or CleaningLog()
    n0 = len(orders)

    df = orders.copy()
    df["order_status"] = df["order_status"].astype(str).str.strip().str.lower()

    status_counts = df["order_status"].value_counts().to_dict()
    df = df[df["order_status"].isin(statuses)]
    log.add(
        f"Keep eligible order statuses {list(statuses)}",
        n0,
        len(df),
        "Only completed/delivered orders represent a realised purchase. "
        f"Status distribution before filtering: {status_counts}",
    )

    n1 = len(df)
    valid_ts = df["order_purchase_timestamp"].notna() & (
        df["order_purchase_timestamp"] >= pd.Timestamp(settings.MIN_VALID_PURCHASE_DATE)
    )
    df = df[valid_ts]
    log.add(
        "Drop orders without a valid purchase timestamp",
        n1,
        len(df),
        "Recency, cohort month and time-to-second-purchase are undefined without a purchase date.",
    )

    n2 = len(df)
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    log.add("De-duplicate order_id", before, len(df),
            "A duplicated order_id would double-count revenue and orders.")

    # Join customer identity (customer_id is unique per order in this dataset).
    cust = customers.drop_duplicates(subset=["customer_id"])[
        ["customer_id", "customer_unique_id", "customer_city", "customer_state",
         "customer_zip_code_prefix"]
    ]
    df = df.merge(cust, on="customer_id", how="left")
    n3 = len(df)
    df = df[df["customer_unique_id"].notna()]
    log.add(
        "Drop orders without a resolvable customer_unique_id",
        n3,
        len(df),
        "Customer-level retention analysis needs a stable person-level identifier.",
    )

    # Time features used across the app.
    ts = df["order_purchase_timestamp"]
    df["order_date"] = ts.dt.normalize()
    df["order_month"] = ts.dt.to_period("M").dt.to_timestamp()
    df["order_year"] = ts.dt.year
    df["dow"] = ts.dt.dayofweek

    # Delivery descriptors (kept, but never used to define eligibility).
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    df["delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    df["is_late"] = np.where(
        df["order_delivered_customer_date"].notna()
        & df["order_estimated_delivery_date"].notna(),
        (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(float),
        np.nan,
    )

    log.add("Enrich with customer identity, calendar and delivery features", n2, len(df),
            "Derived columns only; no rows added or removed.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Child table aggregation (aggregate BEFORE joining)
# ---------------------------------------------------------------------------

def aggregate_order_items(
    order_items: pd.DataFrame,
    products: pd.DataFrame | None = None,
    translation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    One row per order: item count, merchandise value, freight, product mix.

    ``price`` is the merchandise price of a unit line; ``freight_value`` is the
    freight charged on that line. They are summed separately and never mixed.
    """
    items = order_items.copy()
    for col in ("price", "freight_value"):
        items[col] = pd.to_numeric(items[col], errors="coerce")

    agg = items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        merchandise_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
    )

    if products is not None and len(products):
        prod = products[["product_id", "product_category_name"]].drop_duplicates("product_id")
        if translation is not None and len(translation):
            prod = prod.merge(translation, on="product_category_name", how="left")
        else:
            prod["product_category_name_english"] = prod["product_category_name"]
        prod["product_category_english"] = (
            prod["product_category_name_english"].fillna(prod["product_category_name"])
            .fillna("unknown")
        )
        enriched = items.merge(prod[["product_id", "product_category_english"]],
                               on="product_id", how="left")
        # Dominant category by merchandise value within the order.
        cat_val = (
            enriched.assign(line_value=enriched["price"] * 1.0)
            .groupby(["order_id", "product_category_english"], as_index=False)["line_value"].sum()
        )
        idx = cat_val.groupby("order_id")["line_value"].idxmax()
        dominant = cat_val.loc[idx, ["order_id", "product_category_english"]].rename(
            columns={"product_category_english": "main_category"}
        )
        agg = agg.merge(dominant, on="order_id", how="left")

    return agg


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """One row per order: total payment value and the payment mix."""
    pay = payments.copy()
    pay["payment_value"] = pd.to_numeric(pay["payment_value"], errors="coerce")
    agg = pay.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        n_payments=("payment_sequential", "count"),
        max_installments=("payment_installments", "max"),
    )
    dominant_type = (
        pay.groupby(["order_id", "payment_type"], as_index=False)["payment_value"].sum()
        .sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "main_payment_type"})
    )
    return agg.merge(dominant_type, on="order_id", how="left")


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    One row per order: latest review score.

    9.9% of orders carry more than one review row. We keep the most recent
    review (by ``review_answer_timestamp``, falling back to creation date) so
    that the order appears exactly once.
    """
    rev = reviews.copy()
    rev["review_score"] = pd.to_numeric(rev["review_score"], errors="coerce")
    rev["_sort_ts"] = rev["review_answer_timestamp"].fillna(rev["review_creation_date"])
    rev = rev.sort_values(["_sort_ts"], na_position="first")
    latest = rev.drop_duplicates(subset=["order_id"], keep="last")
    out = latest[["order_id", "review_score", "review_id"]].copy()
    out["has_comment"] = latest["review_comment_message"].notna().astype(int).values
    return out


# ---------------------------------------------------------------------------
# Order value table
# ---------------------------------------------------------------------------

def build_order_values(orders: pd.DataFrame, items_sum: pd.DataFrame,
                       payments_sum: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the pre-aggregated child tables to the order grain.

    After this join the table still has exactly one row per order, so summing
    ``merchandise_value`` cannot double-count.
    """
    out = orders.merge(items_sum, on="order_id", how="left", validate="one_to_one")
    out = out.merge(payments_sum, on="order_id", how="left", validate="one_to_one")
    out["merchandise_value"] = out["merchandise_value"].fillna(0.0)
    out["freight_value"] = out["freight_value"].fillna(0.0)
    out["n_items"] = out["n_items"].fillna(0).astype(int)
    out["order_total"] = out["merchandise_value"] + out["freight_value"]
    return out


# ---------------------------------------------------------------------------
# Customer dimension
# ---------------------------------------------------------------------------

def build_customer_dimension(customers: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ``customer_unique_id``.

    The raw customer table has one row per ``customer_id`` (one per order).
    Because a person can appear under several ``customer_id`` values, we keep
    the most recent-ish record's geography (mode) for the person.
    """
    cust = customers.copy()

    def _mode(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan

    geo = cust.groupby("customer_unique_id", as_index=False).agg(
        customer_state=("customer_state", _mode),
        customer_city=("customer_city", _mode),
        customer_zip_code_prefix=("customer_zip_code_prefix", _mode),
        n_customer_ids=("customer_id", "nunique"),
    )
    return geo


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_analytical_tables(
    raw: Dict[str, pd.DataFrame],
    statuses=settings.PRIMARY_ELIGIBLE_STATUSES,
) -> Dict[str, object]:
    """
    Turn raw source tables into the documented analytical tables.

    Returns a dict with keys:
        orders, order_items_sum, payments_sum, reviews_sum, order_values,
        customers, cleaning_log
    """
    log = CleaningLog()

    orders = clean_orders(raw["orders"], raw["customers"], statuses=statuses, log=log)

    items_sum = aggregate_order_items(
        raw["order_items"],
        products=raw.get("products"),
        translation=raw.get("category_translation"),
    )
    payments_sum = aggregate_payments(raw["payments"])

    order_values = build_order_values(orders, items_sum, payments_sum)

    if "reviews" in raw:
        reviews_sum = aggregate_reviews(raw["reviews"])
        order_values = order_values.merge(reviews_sum, on="order_id", how="left")
    else:
        reviews_sum = pd.DataFrame(columns=["order_id", "review_score"])
        order_values["review_score"] = np.nan
        order_values["has_comment"] = np.nan

    customers = build_customer_dimension(raw["customers"])

    # Customer-level rollup used by nearly every page.
    cust_orders = order_values.groupby("customer_unique_id", as_index=False).agg(
        n_orders=("order_id", "nunique"),
        merchandise_value=("merchandise_value", "sum"),
        freight_value=("freight_value", "sum"),
        payment_value=("payment_value", "sum"),
        first_order_ts=("order_purchase_timestamp", "min"),
        last_order_ts=("order_purchase_timestamp", "max"),
        mean_review_score=("review_score", "mean"),
        items_purchased=("n_items", "sum"),
    )
    cust_orders = cust_orders.merge(customers, on="customer_unique_id", how="left")
    cust_orders["first_order_date"] = cust_orders["first_order_ts"].dt.normalize()
    cust_orders["last_order_date"] = cust_orders["last_order_ts"].dt.normalize()
    cust_orders["cohort_month"] = cust_orders["first_order_ts"].dt.to_period("M").dt.to_timestamp()
    cust_orders["avg_order_value"] = cust_orders["merchandise_value"] / cust_orders["n_orders"]

    return {
        "orders": order_values,
        "order_items_sum": items_sum,
        "payments_sum": payments_sum,
        "reviews_sum": reviews_sum,
        "order_values": order_values,
        "customers": cust_orders,
        "cleaning_log": log,
    }


def save_processed(tables: Dict[str, object], out_dir=None) -> Dict[str, str]:
    """Persist analytical tables as parquet (small, fast, reproducible)."""
    from pathlib import Path

    out = Path(out_dir) if out_dir is not None else settings.PROCESSED_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("orders", "order_items_sum", "payments_sum", "customers"):
        df = tables[name]
        if isinstance(df, pd.DataFrame):
            path = out / f"{name}.parquet"
            df.to_parquet(path, index=False)
            written[name] = str(path)
    log = tables.get("cleaning_log")
    if isinstance(log, CleaningLog):
        path = out / "cleaning_log.csv"
        log.to_frame().to_csv(path, index=False)
        written["cleaning_log"] = str(path)
    return written


def load_processed(in_dir=None) -> Dict[str, pd.DataFrame] | None:
    """Load parquet artifacts, or None when they have not been built yet."""
    from pathlib import Path
    directory = Path(in_dir) if in_dir is not None else settings.PROCESSED_DATA_DIR
    needed = ["orders", "customers", "order_items_sum", "payments_sum"]
    if not all((directory / f"{n}.parquet").exists() for n in needed):
        return None
    out = {n: pd.read_parquet(directory / f"{n}.parquet") for n in needed}
    log_path = directory / "cleaning_log.csv"
    if log_path.exists():
        out["cleaning_log_frame"] = pd.read_csv(log_path)
    return out
