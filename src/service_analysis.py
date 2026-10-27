"""
Service-experience analysis: reviews and delivery outcomes, and their
association with *subsequent* purchasing.

Causal language is deliberately avoided. Everything here is an association
measured inside one observational dataset, with explicit denominators and
right-censoring handling.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import settings


# ---------------------------------------------------------------------------
# Review and delivery descriptors
# ---------------------------------------------------------------------------

def review_summary(orders: pd.DataFrame) -> Dict[str, object]:
    """Distribution of review scores over eligible orders."""
    s = pd.to_numeric(orders.get("review_score"), errors="coerce")
    valid = s.dropna()
    dist = valid.value_counts().sort_index()
    return {
        "n_reviews": int(len(valid)),
        "n_orders": int(len(orders)),
        "coverage": float(len(valid) / len(orders)) if len(orders) else np.nan,
        "mean_score": float(valid.mean()) if len(valid) else np.nan,
        "median_score": float(valid.median()) if len(valid) else np.nan,
        "share_5": float((valid == 5).mean()) if len(valid) else np.nan,
        "share_1_2": float((valid <= 2).mean()) if len(valid) else np.nan,
        "distribution": dist.to_dict(),
    }


def delivery_summary(orders: pd.DataFrame) -> Dict[str, object]:
    """Delivery timing descriptors."""
    d = pd.to_numeric(orders.get("delivery_days"), errors="coerce").dropna()
    late = pd.to_numeric(orders.get("delay_days"), errors="coerce").dropna()
    late_only = late[late > 0]
    return {
        "n_delivered_with_date": int(len(d)),
        "median_delivery_days": float(d.median()) if len(d) else np.nan,
        "p90_delivery_days": float(d.quantile(0.9)) if len(d) else np.nan,
        "share_late": float((late > 0).mean()) if len(late) else np.nan,
        # Median of ALL orders' (delivered - estimated); negative = early on average.
        "median_delay_days": float(late.median()) if len(late) else np.nan,
        # Median delay restricted to orders that were actually late.
        "median_late_delay_days": float(late_only.median()) if len(late_only) else np.nan,
    }


def late_delivery_bins(orders: pd.DataFrame) -> pd.DataFrame:
    """Bucket orders by how many days early/late they arrived."""
    df = orders.copy()
    df["late_bin"] = pd.cut(
        pd.to_numeric(df["delay_days"], errors="coerce"),
        bins=settings.LATE_DELIVERY_BINS,
        labels=settings.LATE_DELIVERY_LABELS,
    )
    return df


def score_bands(orders: pd.DataFrame) -> pd.DataFrame:
    """Coarse review bands (1-2, 3, 4, 5) for stable comparisons."""
    df = orders.copy()
    s = pd.to_numeric(df["review_score"], errors="coerce")
    df["score_band"] = pd.cut(s, bins=[0, 2, 3, 4, 5], labels=["1-2", "3", "4", "5"])
    return df


# ---------------------------------------------------------------------------
# Subsequent-purchase outcome (right-censoring aware)
# ---------------------------------------------------------------------------

def add_repeat_outcome(
    orders: pd.DataFrame,
    window_days: int = settings.SERVICE_REPEAT_WINDOW_DAYS,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    For each order, did the same customer place another eligible order within
    ``window_days`` after this order's purchase?

    Orders whose window extends beyond the observation end are marked
    ``censored=True`` and must be excluded from the rate -- otherwise recent
    orders would look artificially non-repeating.

    Note on the base rate: with only ~3% of customers ever repeating, the
    absolute repeat rate inside 90 days is very small. That is reported as-is;
    the *relative* comparison across experience groups is the useful signal.
    """
    df = orders.sort_values(["customer_unique_id", "order_purchase_timestamp"]).copy()
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else \
        df["order_purchase_timestamp"].max().normalize()

    df["censored"] = df["order_purchase_timestamp"] + pd.Timedelta(days=window_days) > as_of

    next_ts = df.groupby("customer_unique_id")["order_purchase_timestamp"].shift(-1)
    gap = (next_ts - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    df["days_to_next_order"] = gap
    df["repeat_within_window"] = np.where(
        df["censored"], np.nan, (gap <= window_days).astype(float)
    )
    return df


def _wilson_ci(successes: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson score interval -- stable for small proportions."""
    p = np.divide(successes, n, out=np.zeros(len(n), dtype=float), where=n > 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return centre - half, centre + half


def repeat_rate_by_group(
    orders: pd.DataFrame,
    group_col: str,
    min_group_size: int = settings.MIN_GRADE_GROUP_SIZE,
) -> pd.DataFrame:
    """
    Repeat-within-window rate by an experience group, with 95% Wilson CIs.

    Only uncensored orders with a non-null outcome are used in the denominator.
    Groups smaller than ``min_group_size`` are flagged, not silently shown.
    """
    df = orders[orders["repeat_within_window"].notna() & orders[group_col].notna()]
    g = df.groupby(group_col, observed=True).agg(
        n_orders=("repeat_within_window", "size"),
        n_repeat=("repeat_within_window", "sum"),
        share_late=("is_late", "mean"),
    ).reset_index()
    g["repeat_rate"] = g["n_repeat"] / g["n_orders"]
    lo, hi = _wilson_ci(g["n_repeat"].to_numpy(), g["n_orders"].to_numpy())
    g["ci_low"], g["ci_high"] = lo, hi
    g["small_group"] = g["n_orders"] < min_group_size
    return g


def association_tests(orders: pd.DataFrame, group_col: str = "score_band") -> Dict[str, float]:
    """
    Chi-square test on (group x repeated) plus Cramer's V effect size.

    A p-value answers "is there any association"; V answers "how large". With
    ~96k orders even trivial associations are 'significant', so V is the number
    that should drive interpretation.
    """
    from scipy import stats

    df = orders[orders["repeat_within_window"].notna() & orders[group_col].notna()]
    if df[group_col].nunique() < 2 or df["repeat_within_window"].nunique() < 2:
        return {}
    table = pd.crosstab(df[group_col], df["repeat_within_window"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n else np.nan
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
            "cramers_v": v, "n": int(n)}


def score_vs_lateness(orders: pd.DataFrame) -> pd.DataFrame:
    """Mean review score by delivery-lateness bucket."""
    df = late_delivery_bins(orders)
    df = df[df["review_score"].notna() & df["late_bin"].notna()]
    g = df.groupby("late_bin", observed=True).agg(
        n_orders=("review_score", "size"),
        mean_review=("review_score", "mean"),
        share_low_score=("review_score", lambda s: float((s <= 2).mean())),
    ).reset_index()
    return g


def category_performance(orders: pd.DataFrame, min_orders: int = 100) -> pd.DataFrame:
    """Order volume, value and service outcome by dominant product category."""
    if "main_category" not in orders.columns:
        return pd.DataFrame()
    df = orders[orders["main_category"].notna()]
    g = df.groupby("main_category").agg(
        n_orders=("order_id", "count"),
        merchandise_value=("merchandise_value", "sum"),
        avg_order_value=("merchandise_value", "mean"),
        mean_review=("review_score", "mean"),
        share_late=("is_late", "mean"),
    ).reset_index()
    g = g[g["n_orders"] >= min_orders].sort_values("merchandise_value", ascending=False)
    return g


def interpret_effect_size(v: float) -> str:
    """
    Cohen's conventional thresholds for Cramer's V (df-adjusted roughly):
    <0.10 negligible, <0.20 weak, <0.40 moderate, >=0.40 relatively strong.
    These are heuristics, not laws -- they are used only to keep the wording
    proportionate to the measured association.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not computable"
    if v < 0.10:
        return "negligible"
    if v < 0.20:
        return "weak"
    if v < 0.40:
        return "moderate"
    return "relatively strong"
