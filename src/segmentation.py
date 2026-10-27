"""
RFM segmentation.

Two methods, both configurable and reproducible:

1. ``rule_based`` (default) -- transparent if/else rules over Recency,
   Frequency and historical Monetary value. Every customer lands in exactly one
   segment and each rule can be explained in a sentence.
2. ``kmeans`` (extension) -- log-transformed, scaled RFM with a
   silhouette-selected k. Clusters are then translated into marketing language
   by profiling them; a cluster is not called "valuable" unless its numbers say
   so.

Terminology note: ``monetary_value`` is historical merchandise value
(revenue-like), not profit and not a lifetime-value forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings


# ---------------------------------------------------------------------------
# RFM base table
# ---------------------------------------------------------------------------

def compute_rfm(customers: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    """
    Build the RFM frame.

    Recency  -- days from the customer's last eligible order to ``as_of_date``
    Frequency-- count of distinct eligible orders
    Monetary -- sum of merchandise value (item prices) in the window
    """
    df = customers.copy()
    if "recency_days" not in df.columns:
        as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp(df["last_order_date"].max())
        df["recency_days"] = (as_of - pd.to_datetime(df["last_order_date"])).dt.days
    df["frequency"] = df["n_orders"] if "frequency" not in df.columns else df["frequency"]
    if "monetary_value" not in df.columns:
        df["monetary_value"] = df["merchandise_value"]
    return df


def quantile_scores(series: pd.Series, quantiles=settings.RFM_QUANTILES, reverse: bool = False) -> pd.Series:
    """
    Score 1-5 by quantile bins.

    ``reverse=True`` is used for recency: a *smaller* number of days is better.
    Ties are handled with ranks so the bins stay approximately equal-sized.
    """
    values = series.astype(float)
    if values.nunique() <= 1:
        return pd.Series(3, index=series.index, dtype="int64")
    ranks = values.rank(method="first", pct=True)
    if reverse:
        ranks = 1 - ranks
    edges = [0.0, *quantiles, 1.0]
    bins = pd.cut(ranks, bins=edges, labels=[1, 2, 3, 4, 5], include_lowest=True)
    return bins.astype("int64")


def add_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["r_score"] = quantile_scores(out["recency_days"], reverse=True)
    out["f_score"] = quantile_scores(out["frequency"], reverse=False)
    out["m_score"] = quantile_scores(out["monetary_value"], reverse=False)
    out["rfm_score"] = out["r_score"] + out["f_score"] + out["m_score"]
    out["rfm_cell"] = (
        out["r_score"].astype(str) + out["f_score"].astype(str) + out["m_score"].astype(str)
    )
    return out


# ---------------------------------------------------------------------------
# Rule-based segments
# ---------------------------------------------------------------------------

def high_value_threshold(monetary: pd.Series, quantile: float = settings.HIGH_VALUE_QUANTILE) -> float:
    return float(monetary.quantile(quantile))


def assign_rule_based_segments(
    df: pd.DataFrame,
    lapsing_days: float,
    inactive_days: float,
    high_value_cut: Optional[float] = None,
    insufficient_days: int = settings.INSUFFICIENT_HISTORY_DAYS,
    repeat_threshold: int = settings.REPEAT_THRESHOLD,
) -> pd.DataFrame:
    """
    Assign one segment per customer using documented, ordered rules.

    Rule order matters -- the first matching rule wins:

    1. New / Insufficient History
       first order less than ``insufficient_days`` ago AND only one order.
       A customer who has not yet had a realistic chance to repurchase should
       not be judged as inactive.
    2. Long Inactive
       recency > ``inactive_days`` (regardless of frequency).
    3. Frequency >= 2 (a repeat customer), split by recency and value:
         - High-Value Active Repeat : recency <= lapsing AND monetary >= p80
         - Active Repeat            : recency <= lapsing
         - Lapsing Repeat           : recency in (lapsing, inactive]
    4. Frequency == 1 (one-time buyer), split by recency:
         - Recent One-Time Buyer    : recency <= lapsing
         - Dormant One-Time Buyer   : recency in (lapsing, inactive]
    """
    out = df.copy()
    hv = high_value_threshold(out["monetary_value"]) if high_value_cut is None else float(high_value_cut)

    rec = out["recency_days"].astype(float)
    freq = out["frequency"].astype(float)
    mon = out["monetary_value"].astype(float)
    since_first = out.get("days_since_first_order", pd.Series(np.nan, index=out.index)).astype(float)

    seg = pd.Series(index=out.index, dtype="object")

    is_new = (since_first < insufficient_days) & (freq < repeat_threshold)
    seg[is_new] = "New / Insufficient History"

    remaining = seg.isna()
    long_inactive = remaining & (rec > inactive_days)
    seg[long_inactive] = "Long Inactive"

    remaining = seg.isna()
    repeat = remaining & (freq >= repeat_threshold)
    seg[repeat & (rec <= lapsing_days) & (mon >= hv)] = "High-Value Active Repeat"
    seg[repeat & (rec <= lapsing_days) & (mon < hv)] = "Active Repeat"
    seg[repeat & (rec > lapsing_days)] = "Lapsing Repeat"

    remaining = seg.isna()
    one_time = remaining & (freq < repeat_threshold)
    seg[one_time & (rec <= lapsing_days)] = "Recent One-Time Buyer"
    seg[one_time & (rec > lapsing_days)] = "Dormant One-Time Buyer"

    out["segment"] = seg.fillna("Unclassified")
    out["high_value_cut"] = hv
    out["is_high_value"] = out["monetary_value"] >= hv
    out["is_repeat"] = out["frequency"] >= repeat_threshold
    return out


# ---------------------------------------------------------------------------
# K-means extension
# ---------------------------------------------------------------------------

def assign_kmeans_segments(
    df: pd.DataFrame,
    k_range=settings.KMEANS_K_RANGE,
    random_state: int = settings.KMEANS_RANDOM_STATE,
    sample_size: int = settings.KMEANS_SAMPLE_SIZE,
) -> Dict[str, object]:
    """
    Log-transform, scale and cluster RFM, choosing k by silhouette score.

    Returns dict with the labelled frame, silhouette table and cluster profile.
    k-means on RFM finds *shape* clusters; the marketing meaning of each cluster
    must be read off its profile, not assumed.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    work = df.copy()
    feats = work[["recency_days", "frequency", "monetary_value"]].astype(float)
    # log1p on frequency and monetary: both are strongly right-skewed.
    X = pd.DataFrame({
        "log_recency": np.log1p(feats["recency_days"].clip(lower=0)),
        "log_frequency": np.log1p(feats["frequency"]),
        "log_monetary": np.log1p(feats["monetary_value"].clip(lower=0)),
    })
    Xs = StandardScaler().fit_transform(X)

    # Silhouette is estimated on a random sample of rows (sklearn samples pairs
    # internally) -- computing the full pairwise matrix on 93k customers would
    # need tens of GB.
    sil_sample = int(min(sample_size, len(Xs)))

    sil_rows = []
    best_k, best_score, best_model = None, -np.inf, None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(Xs)
        score = silhouette_score(Xs, labels, sample_size=sil_sample,
                                 random_state=random_state)
        sil_rows.append({"k": k, "silhouette": float(score), "inertia": float(km.inertia_)})
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    labels = best_model.predict(Xs)
    work["cluster"] = labels
    del X, Xs

    work["segment"] = "Cluster " + work["cluster"].astype(str)

    profile = (
        work.groupby("cluster")
        .agg(n_customers=("customer_unique_id", "count"),
             mean_recency=("recency_days", "mean"),
             mean_frequency=("frequency", "mean"),
             mean_monetary=("monetary_value", "mean"),
             total_monetary=("monetary_value", "sum"))
        .reset_index()
    )
    profile["share_customers"] = profile["n_customers"] / profile["n_customers"].sum()
    profile["share_value"] = profile["total_monetary"] / profile["total_monetary"].sum()

    return {
        "data": work,
        "silhouette": pd.DataFrame(sil_rows),
        "best_k": best_k,
        "best_silhouette": float(best_score),
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# Segment profiles
# ---------------------------------------------------------------------------

def segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required output table: size, share, orders, value, share of value and
    average recency / frequency / order value per segment.
    """
    g = df.groupby("segment", as_index=False).agg(
        n_customers=("customer_unique_id", "count"),
        n_orders=("frequency", "sum"),
        monetary_value=("monetary_value", "sum"),
        avg_recency_days=("recency_days", "mean"),
        median_recency_days=("recency_days", "median"),
        avg_frequency=("frequency", "mean"),
        avg_merchandise_per_customer=("monetary_value", "mean"),
        share_repeat=("is_repeat", "mean"),
        avg_review_score=("mean_review_score", "mean"),
    )
    total_customers = g["n_customers"].sum()
    total_value = g["monetary_value"].sum()
    g["share_customers"] = g["n_customers"] / total_customers if total_customers else np.nan
    g["share_value"] = g["monetary_value"] / total_value if total_value else np.nan
    g["avg_order_value"] = np.where(g["n_orders"] > 0,
                                    g["monetary_value"] / g["n_orders"].replace(0, np.nan),
                                    np.nan)
    order = [s for s in settings.SEGMENT_ORDER if s in set(g["segment"])]
    order += [s for s in g["segment"] if s not in order]
    g["_order"] = g["segment"].map({s: i for i, s in enumerate(order)})
    g = g.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return g


def segment_value_index(profile: pd.DataFrame) -> pd.DataFrame:
    """Value-per-customer index: 1.0 means 'average value per customer'."""
    out = profile[["segment", "n_customers", "monetary_value", "share_value", "share_customers"]].copy()
    out["value_index"] = out["share_value"] / out["share_customers"].replace(0, np.nan)
    return out


def compare_segments(df: pd.DataFrame, metric: str = "avg_recency_days") -> pd.DataFrame:
    """Simple segment comparison table (used on the segmentation page)."""
    cols = ["segment", "n_customers", "avg_recency_days", "avg_frequency",
            "avg_merchandise_per_customer", "share_value"]
    return df[cols].sort_values("share_value", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    data: pd.DataFrame
    profile: pd.DataFrame
    method: str
    metadata: Dict[str, object]


def run_segmentation(
    customers: pd.DataFrame,
    method: str = settings.SEGMENTATION_METHOD,
    lapsing_days: Optional[float] = None,
    inactive_days: Optional[float] = None,
    high_value_quantile: float = settings.HIGH_VALUE_QUANTILE,
) -> SegmentationResult:
    """Score, segment and profile in one call."""
    df = compute_rfm(customers)
    if "days_since_first_order" not in df.columns and "first_order_date" in df.columns:
        df["days_since_first_order"] = (
            pd.to_datetime(df["last_order_date"]).max() - pd.to_datetime(df["first_order_date"])
        ).dt.days

    meta: Dict[str, object] = {}
    if method == "kmeans":
        res = assign_kmeans_segments(df)
        data = res["data"]
        # Keep the same derived flags as the rule-based path so profile
        # functions work identically for both methods.
        if "is_repeat" not in data.columns:
            data["is_repeat"] = data["frequency"] >= settings.REPEAT_THRESHOLD
        if "is_high_value" not in data.columns:
            data["is_high_value"] = data["monetary_value"] >= data["monetary_value"].quantile(
                settings.HIGH_VALUE_QUANTILE)
        meta["silhouette"] = res["silhouette"]
        meta["best_k"] = res["best_k"]
        meta["best_silhouette"] = res["best_silhouette"]
        # The k-means profile describes clusters; the segment profile below
        # uses the same schema and is kept for comparison / export.
        kmeans_profile = res["profile"]
        meta["segment_profile"] = segment_profile(data)
    else:
        lapsing = float(lapsing_days if lapsing_days is not None else settings.LAPSING_FLOOR_DAYS)
        inactive = float(inactive_days if inactive_days is not None else settings.INACTIVE_FLOOR_DAYS)
        data = assign_rule_based_segments(df, lapsing, inactive,
                                          high_value_cut=float(df["monetary_value"].quantile(high_value_quantile)))
        meta["lapsing_days"] = lapsing
        meta["inactive_days"] = inactive
        meta["high_value_quantile"] = high_value_quantile

    data = add_rfm_scores(data)
    profile = meta.get("segment_profile") if method == "kmeans" else segment_profile(data)
    if method == "kmeans":
        profile = kmeans_profile
    return SegmentationResult(data=data, profile=profile, method=method, metadata=meta)
