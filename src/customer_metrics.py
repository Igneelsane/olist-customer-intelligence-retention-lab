"""
Customer-level metrics: KPIs, repeat purchasing, purchase intervals,
revenue concentration and inactivity labelling.

Metric definitions live in one place so that the app, the notebook and the
tests all reference the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import settings


# ---------------------------------------------------------------------------
# Observation window
# ---------------------------------------------------------------------------

def analysis_dates(orders: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    """
    Observation window and the 'as of' date used for recency.

    ``as_of_date`` = the last calendar day on which an eligible order was placed.
    Recency for a customer is measured in days from their last order to this
    date. Using the data's own last date (rather than today's date) keeps the
    analysis internally consistent and reproducible.
    """
    ts = orders["order_purchase_timestamp"]
    start = ts.min().normalize()
    end = ts.max().normalize()
    return {
        "first_order_date": start,
        "as_of_date": end,
        "observation_days": int((end - start).days + 1),
    }


# ---------------------------------------------------------------------------
# Repeat purchasing
# ---------------------------------------------------------------------------

def repeat_purchase_stats(customers: pd.DataFrame, repeat_threshold: int = settings.REPEAT_THRESHOLD) -> Dict[str, float]:
    """
    Repeat-purchase rate.

    Definition
    ----------
    repeat_purchase_rate = # customers with >= ``repeat_threshold`` distinct
                           eligible orders / # customers with >= 1 eligible order

    Denominator = every customer in the eligible population. This is an
    "ever repeated" measure and is NOT the same as month-1 retention.
    """
    n_customers = int(len(customers))
    n_repeaters = int((customers["n_orders"] >= repeat_threshold).sum())
    orders_per_customer = float(customers["n_orders"].mean()) if n_customers else np.nan
    return {
        "n_customers": n_customers,
        "n_repeat_customers": n_repeaters,
        "n_one_time_customers": n_customers - n_repeaters,
        "repeat_purchase_rate": n_repeaters / n_customers if n_customers else np.nan,
        "orders_per_customer": orders_per_customer,
        "repeat_threshold": repeat_threshold,
    }


def purchase_intervals(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Within-customer gaps between consecutive eligible orders, in days.

    Returns one row per observed repeat purchase (i.e. gaps only), which is the
    correct denominator for "how long until a customer buys again".
    """
    df = orders.sort_values(["customer_unique_id", "order_purchase_timestamp"]).copy()
    df["prev_ts"] = df.groupby("customer_unique_id")["order_purchase_timestamp"].shift(1)
    df["order_seq"] = df.groupby("customer_unique_id").cumcount() + 1
    gaps = df[df["prev_ts"].notna()].copy()
    gaps["gap_days"] = (gaps["order_purchase_timestamp"] - gaps["prev_ts"]).dt.total_seconds() / 86400.0
    return gaps[["customer_unique_id", "order_id", "order_seq", "order_purchase_timestamp",
                 "prev_ts", "gap_days"]]


def time_to_second_purchase(orders: pd.DataFrame) -> pd.DataFrame:
    """Days from a customer's first to second eligible order (repeaters only)."""
    gaps = purchase_intervals(orders)
    second = gaps[gaps["order_seq"] == 2].copy()
    second = second.rename(columns={"gap_days": "days_to_second_order"})
    return second[["customer_unique_id", "order_id", "order_purchase_timestamp",
                   "days_to_second_order"]]


def interval_summary(gaps: pd.DataFrame) -> Dict[str, float]:
    """Descriptive statistics of observed purchase gaps."""
    s = pd.to_numeric(gaps["gap_days"], errors="coerce").dropna()
    if not len(s):
        return {}
    return {
        "n_intervals": int(len(s)),
        "mean_days": float(s.mean()),
        "median_days": float(s.median()),
        "p25_days": float(s.quantile(0.25)),
        "p75_days": float(s.quantile(0.75)),
        "p90_days": float(s.quantile(0.90)),
        "p95_days": float(s.quantile(0.95)),
        "share_within_30d": float((s <= 30).mean()),
        "share_within_90d": float((s <= 90).mean()),
    }


# ---------------------------------------------------------------------------
# Revenue concentration
# ---------------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution (0 = equal, 1 = maximal)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0 or v.sum() <= 0:
        return float("nan")
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)


def concentration_stats(customers: pd.DataFrame) -> Dict[str, float]:
    """
    How concentrated is historical merchandise value among customers?

    top_x_share = value generated by the top X% of customers by value /
                  total value.
    """
    v = customers["merchandise_value"].astype(float).sort_values(ascending=False)
    total = float(v.sum())
    out = {"total_merchandise_value": total, "gini": gini(v.values), "n_customers": int(len(v))}
    if total <= 0 or not len(v):
        return out
    for pct in (0.01, 0.05, 0.10, 0.20, 0.50):
        k = max(1, int(np.ceil(len(v) * pct)))
        out[f"top_{int(pct * 100)}pct_share"] = float(v.iloc[:k].sum() / total)
    out["bottom_50pct_share"] = float(v.iloc[-max(1, len(v) // 2):].sum() / total)
    return out


def customer_value_deciles(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Ten equal-count buckets of customers ordered by historical value.

    Accepts either ``monetary_value`` (customer metrics table) or
    ``merchandise_value`` (order-level naming) so the same function serves both
    call sites.
    """
    value_col = "monetary_value" if "monetary_value" in customers.columns else "merchandise_value"
    df = customers[["customer_unique_id", value_col, "n_orders"]].copy()
    df = df.rename(columns={value_col: "merchandise_value"})
    df = df.sort_values("merchandise_value", ascending=False).reset_index(drop=True)
    df["value_decile"] = pd.qcut(
        df["merchandise_value"].rank(method="first", ascending=False),
        10, labels=[f"D{i}" for i in range(1, 11)]
    )
    agg = df.groupby("value_decile", observed=True).agg(
        n_customers=("customer_unique_id", "count"),
        merchandise_value=("merchandise_value", "sum"),
        mean_orders=("n_orders", "mean"),
    ).reset_index()
    agg["share_of_value"] = agg["merchandise_value"] / agg["merchandise_value"].sum()
    agg["share_of_customers"] = agg["n_customers"] / agg["n_customers"].sum()
    return agg


# ---------------------------------------------------------------------------
# Inactivity / recency
# ---------------------------------------------------------------------------

def derive_inactivity_thresholds(
    gaps: pd.DataFrame,
    lapsing_floor: int = settings.LAPSING_FLOOR_DAYS,
    inactive_floor: int = settings.INACTIVE_FLOOR_DAYS,
) -> Dict[str, float]:
    """
    Data-informed recency thresholds.

    Rationale: a customer who has not ordered for longer than the interval in
    which most repeat purchases happen is behaving differently from an active
    repeat buyer. We therefore anchor the thresholds on the *observed*
    distribution of repeat-purchase gaps:

        lapsing  = max(lapsing_floor,  p75 of gap_days)
        inactive = max(inactive_floor, p90 of gap_days)

    Floors prevent pathological thresholds if the data were unusually dense.
    These are behavioural reference points, not churn definitions.
    """
    s = pd.to_numeric(gaps["gap_days"], errors="coerce").dropna()
    p75 = float(s.quantile(0.75)) if len(s) else lapsing_floor
    p90 = float(s.quantile(0.90)) if len(s) else inactive_floor
    lapsing = float(max(lapsing_floor, p75))
    inactive = float(max(inactive_floor, p90, lapsing + 1))
    return {
        "lapsing_days": round(lapsing, 1),
        "inactive_days": round(inactive, 1),
        "gap_p75_days": round(p75, 1),
        "gap_p90_days": round(p90, 1),
        "lapsing_floor": lapsing_floor,
        "inactive_floor": inactive_floor,
    }


def classify_activity(
    customers: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lapsing_days: float,
    inactive_days: float,
    insufficient_history_days: int = settings.INSUFFICIENT_HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Label each customer's current activity state.

    Labels are descriptive statements about the observation window, NOT churn
    predictions:
        New / insufficient history -- first order too recent to have had a fair
                                      chance of a second one
        Recently active            -- last order within ``lapsing_days``
        Potentially lapsing        -- between ``lapsing_days`` and ``inactive_days``
        Long inactive              -- beyond ``inactive_days``
    """
    df = customers.copy()
    as_of = pd.Timestamp(as_of_date).normalize()
    # Accept either pre-computed dates or raw timestamps.
    if "last_order_date" not in df.columns:
        source = df["last_order_ts"] if "last_order_ts" in df.columns else df["last_order_date"]
        df["last_order_date"] = pd.to_datetime(source).dt.normalize()
    if "first_order_date" not in df.columns:
        source = df["first_order_ts"] if "first_order_ts" in df.columns else df["first_order_date"]
        df["first_order_date"] = pd.to_datetime(source).dt.normalize()
    df["recency_days"] = (as_of - pd.to_datetime(df["last_order_date"])).dt.days.astype(float)
    df["days_since_first_order"] = (as_of - pd.to_datetime(df["first_order_date"])).dt.days.astype(float)

    # Only a customer who has NOT yet repeated can be "insufficient history":
    # someone with two orders has demonstrably had the opportunity to return.
    frequency = df["frequency"] if "frequency" in df.columns else df["n_orders"]
    too_new = (df["days_since_first_order"] < insufficient_history_days) & (
        frequency < settings.REPEAT_THRESHOLD)
    conditions = [
        too_new,
        df["recency_days"] <= lapsing_days,
        df["recency_days"] <= inactive_days,
    ]
    choices = ["New / insufficient history", "Recently active", "Potentially lapsing"]
    df["activity_status"] = np.select(conditions, choices, default="Long inactive")
    df["is_insufficient_history"] = too_new
    return df


# ---------------------------------------------------------------------------
# Full customer metric build
# ---------------------------------------------------------------------------

@dataclass
class CustomerMetrics:
    customers: pd.DataFrame
    orders: pd.DataFrame
    summary: Dict[str, object]
    intervals: pd.DataFrame
    second_purchase: pd.DataFrame
    thresholds: Dict[str, float]
    dates: Dict[str, pd.Timestamp]


def build_customer_metrics(
    order_values: pd.DataFrame,
    customers_base: pd.DataFrame,
    lapsing_days: Optional[float] = None,
    inactive_days: Optional[float] = None,
) -> CustomerMetrics:
    """
    Compose the customer-level analytical table used by every page.

    ``monetary_value`` is *historical merchandise value* -- the sum of item
    prices a customer paid over the observed window. It is revenue, not profit,
    and it is not a forward-looking CLV estimate.
    """
    dates = analysis_dates(order_values)
    basis = customers_base.copy()

    gaps = purchase_intervals(order_values)
    thresholds = derive_inactivity_thresholds(gaps)
    lapsing = float(lapsing_days if lapsing_days is not None else thresholds["lapsing_days"])
    inactive = float(inactive_days if inactive_days is not None else thresholds["inactive_days"])

    cust = classify_activity(basis, dates["as_of_date"], lapsing, inactive)
    cust = cust.rename(columns={"merchandise_value": "monetary_value"})
    cust["frequency"] = cust["n_orders"]

    second = time_to_second_purchase(order_values)
    cust = cust.merge(
        second.groupby("customer_unique_id")["days_to_second_order"].min().rename("days_to_second_order"),
        on="customer_unique_id", how="left",
    )

    repeat = repeat_purchase_stats(cust)
    interval_stats = interval_summary(gaps)
    conc = concentration_stats(cust.assign(merchandise_value=cust["monetary_value"]))

    summary = {
        **dates,
        "n_orders": int(order_values["order_id"].nunique()),
        "n_customers": int(len(cust)),
        "n_orders_per_customer": float(order_values["order_id"].nunique() / len(cust)) if len(cust) else np.nan,
        "merchandise_value": float(order_values["merchandise_value"].sum()),
        "freight_value": float(order_values["freight_value"].sum()),
        "payment_value": float(order_values["payment_value"].sum())
        if "payment_value" in order_values.columns else float("nan"),
        "avg_order_value": float(order_values["merchandise_value"].mean()),
        "median_order_value": float(order_values["merchandise_value"].median()),
        "avg_merchandise_per_customer": float(cust["monetary_value"].mean()) if len(cust) else np.nan,
        "repeat_purchase_rate": repeat["repeat_purchase_rate"],
        "n_repeat_customers": repeat["n_repeat_customers"],
        "n_one_time_customers": repeat["n_one_time_customers"],
        "gini_customer_value": conc["gini"],
        "top_1pct_share": conc.get("top_1pct_share"),
        "top_10pct_share": conc.get("top_10pct_share"),
        "top_20pct_share": conc.get("top_20pct_share"),
        "lapsing_days": lapsing,
        "inactive_days": inactive,
        # Interval statistics describe ONLY observed repeat purchases
        # (denominator = observed gaps, not all customers).
        "median_days_to_second_order": float(second["days_to_second_order"].median()) if len(second) else float("nan"),
        "p75_days_to_second_order": float(second["days_to_second_order"].quantile(0.75)) if len(second) else float("nan"),
        "median_repeat_gap_days": interval_stats.get("median_days", float("nan")),
        "share_gaps_within_30d": interval_stats.get("share_within_30d", float("nan")),
        "share_gaps_within_90d": interval_stats.get("share_within_90d", float("nan")),
    }

    return CustomerMetrics(
        customers=cust,
        orders=order_values,
        summary=summary,
        intervals=gaps,
        second_purchase=second,
        thresholds={**thresholds, "lapsing_days": lapsing, "inactive_days": inactive},
        dates=dates,
    )


def monthly_trend(orders: pd.DataFrame) -> pd.DataFrame:
    """Monthly eligible orders, unique customers and merchandise value."""
    orders = orders.copy()
    orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    out = orders.groupby("month", as_index=False).agg(
        n_orders=("order_id", "nunique"),
        n_customers=("customer_unique_id", "nunique"),
        merchandise_value=("merchandise_value", "sum"),
        freight_value=("freight_value", "sum"),
    )
    out["avg_order_value"] = out["merchandise_value"] / out["n_orders"]
    return out


def monthly_new_vs_returning(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly split of unique customers into first-time and returning.

    'Returning' = the customer had at least one earlier eligible order strictly
    before the current month. This is a monthly activity split, not a cohort
    retention rate.
    """
    df = orders[["customer_unique_id", "order_purchase_timestamp"]].copy()
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    first_month = df.groupby("customer_unique_id")["month"].min().rename("first_month")
    df = df.join(first_month, on="customer_unique_id")
    df["is_new"] = df["month"] == df["first_month"]

    # One row per (customer, month): a customer counts once per month, and can
    # only ever be "new" in their own first month.
    per_cust_month = df.drop_duplicates(["customer_unique_id", "month"])
    per = per_cust_month.groupby("month", as_index=False).agg(
        new_customers=("is_new", "sum"),
        returning_customers=("is_new", lambda s: int((~s.astype(bool)).sum())),
        total_customers=("customer_unique_id", "nunique"),
    )
    return per
