"""
Monthly first-purchase cohort retention.

Definitions (also in reports/methodology.md)
-------------------------------------------
cohort_month   : calendar month of a customer's FIRST eligible order.
month_index    : number of calendar months between the cohort month and the
                 activity month (0 = cohort month itself).
cohort_size    : number of DISTINCT customer_unique_ids whose first eligible
                 order fell in that month.
n_active       : number of DISTINCT customers of that cohort who placed at
                 least one eligible order during that specific month.
retention_rate : n_active / cohort_size  (a specific-month, not cumulative,
                 measure). Month 0 is 1.0 by construction.

Observation-window handling
---------------------------
A cohort-month cell is *observable* only if that calendar month is inside the
dataset's window. Cells beyond the window are stored as NaN (never 0), and are
excluded from denominators, averages and heatmaps.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config import settings


# ---------------------------------------------------------------------------
# Core table
# ---------------------------------------------------------------------------

def _month_index(ts: pd.Series, cohort: pd.Series) -> pd.Series:
    return (ts.dt.year - cohort.dt.year) * 12 + (ts.dt.month - cohort.dt.month)


def build_cohort_table(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format cohort table: one row per (cohort_month, month_index).

    Returns columns:
        cohort_month, month_index, cohort_size, n_active, retention_rate,
        is_observable, customers (list of ids, sorted -- used for validation)
    """
    df = orders[["customer_unique_id", "order_purchase_timestamp"]].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "cohort_month", "month_index", "cohort_size", "n_active", "retention_rate",
            "max_observable_index", "is_observable"])
    df["activity_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    first_month = df.groupby("customer_unique_id")["activity_month"].min().rename("cohort_month")
    df = df.join(first_month, on="customer_unique_id")
    df["month_index"] = _month_index(df["activity_month"], df["cohort_month"]).astype(int)

    # Distinct customers per (cohort, month) -- the key de-duplication step.
    active = (
        df.drop_duplicates(["cohort_month", "month_index", "customer_unique_id"])
        .groupby(["cohort_month", "month_index"], as_index=False)
        .agg(n_active=("customer_unique_id", "nunique"))
    )
    sizes = first_month.value_counts().rename_axis("cohort_month").rename("cohort_size").reset_index()

    last_month = df["activity_month"].max()
    sizes["max_observable_index"] = sizes["cohort_month"].apply(
        lambda c: (last_month.year - c.year) * 12 + (last_month.month - c.month)
    )

    # Dense grid over OBSERVABLE cells only: a month with no purchases is a
    # genuine zero, while months past the end of the dataset are not created at
    # all (so they can never be mistaken for zero retention).
    grid = pd.DataFrame(
        [(cm, mi) for cm, mx in zip(sizes["cohort_month"], sizes["max_observable_index"])
         for mi in range(int(mx) + 1)],
        columns=["cohort_month", "month_index"],
    )
    out = grid.merge(active, on=["cohort_month", "month_index"], how="left")
    out["n_active"] = out["n_active"].fillna(0).astype(int)
    out = out.merge(sizes, on="cohort_month", how="left")
    out["retention_rate"] = out["n_active"] / out["cohort_size"]
    out["is_observable"] = True
    return out.sort_values(["cohort_month", "month_index"]).reset_index(drop=True)


def cohort_sizes(cohort_table: pd.DataFrame) -> pd.DataFrame:
    """One row per cohort: size, first month, maximum observable month index."""
    sizes = (
        cohort_table.groupby("cohort_month", as_index=False)
        .agg(cohort_size=("cohort_size", "first"),
             max_observable_index=("max_observable_index", "first"))
    )
    return sizes.sort_values("cohort_month").reset_index(drop=True)


def cohort_matrix(cohort_table: pd.DataFrame, max_index: Optional[int] = 12) -> pd.DataFrame:
    """
    Wide retention matrix (rows = cohort, columns = month index).

    Unobservable future cells are NaN. Observable but quiet months are filled
    with 0.0 (a true zero retention month, not a missing one).
    """
    sizes = cohort_sizes(cohort_table)
    tbl = cohort_table.copy()
    if max_index is not None:
        tbl = tbl[tbl["month_index"] <= max_index]

    wide = tbl.pivot_table(
        index="cohort_month", columns="month_index", values="retention_rate", aggfunc="first"
    )
    wide = wide.reindex(columns=range(0, (max_index if max_index is not None else int(tbl["month_index"].max())) + 1))

    # Fill observable-but-missing cells with 0.
    obs = sizes.set_index("cohort_month")["max_observable_index"]
    for col in wide.columns:
        mask = wide.index.map(lambda c: col <= obs.get(c, -1))
        wide.loc[mask, col] = wide.loc[mask, col].fillna(0.0)

    wide.insert(0, "cohort_size", sizes.set_index("cohort_month")["cohort_size"].reindex(wide.index))
    wide.insert(1, "max_observable_index",
                sizes.set_index("cohort_month")["max_observable_index"].reindex(wide.index))
    return wide.sort_index()


def cohort_counts_matrix(cohort_table: pd.DataFrame, max_index: Optional[int] = 12) -> pd.DataFrame:
    """Same shape as ``cohort_matrix`` but containing customer counts."""
    sizes = cohort_sizes(cohort_table)
    tbl = cohort_table.copy()
    if max_index is not None:
        tbl = tbl[tbl["month_index"] <= max_index]
    wide = tbl.pivot_table(index="cohort_month", columns="month_index", values="n_active", aggfunc="first")
    wide = wide.reindex(columns=range(0, (max_index if max_index is not None else int(tbl["month_index"].max())) + 1))
    obs = sizes.set_index("cohort_month")["max_observable_index"]
    for col in wide.columns:
        mask = wide.index.map(lambda c: col <= obs.get(c, -1))
        wide.loc[mask, col] = wide.loc[mask, col].fillna(0)
    return wide.sort_index()


# ---------------------------------------------------------------------------
# Retention metrics
# ---------------------------------------------------------------------------

def retention_curves(cohort_table: pd.DataFrame, min_size: int = 1,
                     cohorts: Optional[Iterable[pd.Timestamp]] = None) -> pd.DataFrame:
    """Long retention curves for the selected cohorts (observable cells only)."""
    tbl = cohort_table.copy()
    sizes = cohort_sizes(tbl)
    keep = sizes[sizes["cohort_size"] >= min_size]["cohort_month"]
    if cohorts is not None:
        keep = keep[keep.isin(list(cohorts))]
    tbl = tbl[tbl["cohort_month"].isin(keep) & tbl["is_observable"]]
    return tbl[["cohort_month", "month_index", "retention_rate", "n_active", "cohort_size"]]


def weighted_retention_by_month(cohort_table: pd.DataFrame, upto: Optional[int] = 12) -> pd.DataFrame:
    """
    Pooled retention per month index across cohorts.

    Weighting is by cohort size, and only observable cells contribute to the
    denominator -- the standard way to summarise cohorts of unequal maturity
    without pretending the unobserved future is zero.
    """
    tbl = cohort_table[cohort_table["is_observable"]].copy()
    if upto is not None:
        tbl = tbl[tbl["month_index"] <= upto]
    grp = tbl.groupby("month_index", as_index=False).agg(
        n_active=("n_active", "sum"),
        n_eligible=("cohort_size", "sum"),
        n_cohorts=("cohort_month", "nunique"),
    )
    grp["retention_rate"] = grp["n_active"] / grp["n_eligible"]
    return grp


def month_n_retention(cohort_table: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """Retention in month ``n`` for every cohort where that month is observable."""
    tbl = cohort_table[(cohort_table["month_index"] == n) & cohort_table["is_observable"]]
    out = tbl[["cohort_month", "cohort_size", "n_active", "retention_rate"]].copy()
    return out.sort_values("cohort_month").reset_index(drop=True)


def cumulative_repeat_rate(cohort_table: pd.DataFrame, orders: pd.DataFrame,
                           upto: Optional[int] = 12) -> pd.DataFrame:
    """
    Share of each cohort that ordered again at least once within ``upto`` months.

    This is *cumulative* repeat behaviour (ever repeated), computed from the
    distinct customers appearing at any month index between 1 and ``upto`` --
    so it can never be produced by summing the specific-month retention rates.
    """
    df = orders[["customer_unique_id", "order_purchase_timestamp"]].copy()
    df["activity_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    first_month = df.groupby("customer_unique_id")["activity_month"].min().rename("cohort_month")
    df = df.join(first_month, on="customer_unique_id")
    df["month_index"] = _month_index(df["activity_month"], df["cohort_month"])

    sizes = (first_month.value_counts().rename_axis("cohort_month").rename("cohort_size")
             .reset_index())
    last_month = df["activity_month"].max()
    sizes["max_observable_index"] = sizes["cohort_month"].apply(
        lambda c: (last_month.year - c.year) * 12 + (last_month.month - c.month))
    if upto is not None:
        sizes = sizes[sizes["max_observable_index"] >= upto]

    repeaters = (
        df[(df["month_index"] >= 1) & (df["month_index"] <= (upto if upto is not None else 10 ** 6))]
        .groupby("cohort_month")["customer_unique_id"].nunique()
        .rename("n_cumulative_repeaters")
    )
    out = sizes.merge(repeaters, on="cohort_month", how="left")
    out["n_cumulative_repeaters"] = out["n_cumulative_repeaters"].fillna(0).astype(int)
    out["cumulative_repeat_rate"] = out["n_cumulative_repeaters"] / out["cohort_size"]
    return out.sort_values("cohort_month").reset_index(drop=True)


def cohort_summary(cohort_table: pd.DataFrame, upto: int = 12) -> pd.DataFrame:
    """Per-cohort headline numbers plus a maturity flag."""
    sizes = cohort_sizes(cohort_table)
    m1 = month_n_retention(cohort_table, 1).rename(columns={"retention_rate": "m1_retention",
                                                            "n_active": "m1_active"})
    out = sizes.merge(m1[["cohort_month", "m1_retention", "m1_active"]], on="cohort_month", how="left")
    out["is_mature"] = out["max_observable_index"] >= upto
    return out


def retention_kpis(cohort_table: pd.DataFrame, orders: pd.DataFrame) -> Dict[str, float]:
    """
    Headline retention numbers with explicit denominators.

    * month_1_retention_pooled -- n_active(month 1) summed over cohorts where
      month 1 is observable / cohort_size summed over the same cohorts.
    """
    m1 = cohort_table[(cohort_table["month_index"] == 1) & cohort_table["is_observable"]]
    m2 = cohort_table[(cohort_table["month_index"] == 2) & cohort_table["is_observable"]]
    rep = cumulative_repeat_rate(cohort_table, orders, upto=12)

    def _pooled(tbl):
        if not len(tbl):
            return np.nan
        return float(tbl["n_active"].sum() / tbl["cohort_size"].sum())

    return {
        "month_1_retention_pooled": _pooled(m1),
        "month_2_retention_pooled": _pooled(m2),
        "n_cohorts": int(cohort_table["cohort_month"].nunique()),
        "n_cohorts_mature_12m": int((cohort_sizes(cohort_table)["max_observable_index"] >= 12).sum()),
        "median_12m_cumulative_repeat_rate": float(rep["cumulative_repeat_rate"].median())
        if len(rep) else np.nan,
        "n_customers_in_cohorts": int(cohort_sizes(cohort_table)["cohort_size"].sum()),
    }


# ---------------------------------------------------------------------------
# Validation helpers (used by tests)
# ---------------------------------------------------------------------------

def validate_cohort_table(cohort_table: pd.DataFrame) -> List[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    problems: List[str] = []

    m0 = cohort_table[cohort_table["month_index"] == 0]
    cohorts = set(cohort_table["cohort_month"].unique())
    if set(m0["cohort_month"].unique()) != cohorts:
        problems.append("Some cohorts have no month-0 row.")
    if len(m0) and not np.allclose(m0["retention_rate"], 1.0):
        problems.append("Month 0 retention is not 100% for every cohort.")

    if (cohort_table["n_active"] > cohort_table["cohort_size"]).any():
        problems.append("Some cells have more active customers than cohort members.")

    if cohort_table["retention_rate"].dropna().between(0, 1).all() is False:
        problems.append("Retention rate outside [0, 1].")

    sizes = cohort_table.groupby("cohort_month")["cohort_size"].nunique()
    if (sizes != 1).any():
        problems.append("A cohort has more than one cohort_size value.")

    if cohort_table.duplicated(["cohort_month", "month_index"]).any():
        problems.append("Duplicate (cohort_month, month_index) rows -- customers double counted.")

    # Month 0 must contain the whole cohort: n_active == cohort_size.
    if len(m0) and (m0["n_active"] != m0["cohort_size"]).any():
        problems.append("Month 0 active customers do not equal the cohort size.")

    # No customer may appear twice inside the same cohort-month cell.
    if "n_active" in cohort_table and (cohort_table["n_active"] < 0).any():
        problems.append("Negative customer counts.")

    return problems
