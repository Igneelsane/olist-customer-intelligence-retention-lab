"""
Cohort construction and retention tests.

Hand-checkable expectations from the synthetic fixture:

    Jan-2017 cohort {A, B, C}, size 3
        month 0 -> 3/3
        month 1 -> 1/3 (A only)
        month 2 -> 2/3 (A and C)
    Feb-2017 cohort {D, E}, size 2
        month 0 -> 2/2
        month 1 -> 0/2 (observable, genuine zero)

    Last month present in the data is 2017-03, so Jan-2017 has
    max_observable_index = 2 and Feb-2017 has 1. Month 2 for Feb-2017 is
    NOT observable and must be missing, not zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cohort_analysis as ca

JAN = pd.Timestamp("2017-01-01")
FEB = pd.Timestamp("2017-02-01")


@pytest.fixture
def cohort(synthetic_orders):
    return ca.build_cohort_table(synthetic_orders)


def test_cohort_sizes_use_distinct_customers(cohort):
    sizes = ca.cohort_sizes(cohort).set_index("cohort_month")["cohort_size"]
    assert sizes[JAN] == 3
    assert sizes[FEB] == 2
    assert sizes.sum() == 5


def test_month_zero_is_one_hundred_percent(cohort):
    m0 = cohort[cohort["month_index"] == 0]
    assert (m0["retention_rate"] == 1.0).all()
    assert (m0["n_active"] == m0["cohort_size"]).all()


def test_specific_month_retention_values(cohort):
    idx = cohort.set_index(["cohort_month", "month_index"])["retention_rate"]
    assert idx[(JAN, 1)] == pytest.approx(1 / 3)
    assert idx[(JAN, 2)] == pytest.approx(2 / 3)
    assert idx[(FEB, 1)] == pytest.approx(0.0)


def test_customer_with_two_orders_in_one_month_counts_once(cohort):
    """Customer E placed two orders in Feb-2017 but must count once in month 0."""
    feb0 = cohort[(cohort["cohort_month"] == FEB) & (cohort["month_index"] == 0)].iloc[0]
    assert feb0["n_active"] == 2          # D and E, not 3


def test_retention_never_exceeds_one(cohort):
    assert cohort["retention_rate"].between(0, 1).all()


def test_no_duplicate_cohort_month_rows(cohort):
    assert not cohort.duplicated(["cohort_month", "month_index"]).any()


def test_unobservable_future_months_are_missing_not_zero(cohort):
    matrix = ca.cohort_matrix(cohort, max_index=6)
    # Feb-2017 cohort can only be observed up to month 1.
    assert not np.isnan(matrix.loc[FEB, 0])
    assert not np.isnan(matrix.loc[FEB, 1])
    assert np.isnan(matrix.loc[FEB, 2]), "unobservable month must be NaN, not 0"
    # Jan-2017 can be observed up to month 2.
    assert not np.isnan(matrix.loc[JAN, 2])
    assert np.isnan(matrix.loc[JAN, 3])


def test_observable_but_quiet_month_is_zero(cohort):
    """Feb-2017 month 1 had no purchases and IS observable -> a real zero."""
    matrix = ca.cohort_matrix(cohort, max_index=3)
    assert matrix.loc[FEB, 1] == 0.0


def test_validate_cohort_table_reports_no_problems(cohort):
    assert ca.validate_cohort_table(cohort) == []


def test_validate_cohort_table_detects_missing_month_zero(cohort):
    broken = cohort[cohort["month_index"] != 0]
    assert ca.validate_cohort_table(broken) != []


def test_pooled_month_one_retention(cohort):
    pooled = ca.weighted_retention_by_month(cohort, upto=1)
    row = pooled.set_index("month_index").loc[1]
    # Jan: 1 active of 3, Feb: 0 of 2 -> 1/5
    assert row["retention_rate"] == pytest.approx(1 / 5)
    assert row["n_eligible"] == 5


def test_cumulative_repeat_rate_is_not_the_sum_of_monthly_rates(synthetic_orders, cohort):
    # upto=2 because the fixture only spans Jan–Mar 2017: a cohort is only
    # eligible when the dataset actually contains the follow-up months.
    cum = ca.cumulative_repeat_rate(cohort, synthetic_orders, upto=2)
    jan = cum.set_index("cohort_month").loc[JAN]
    # A and C both repeated by month 2 -> 2/3
    assert jan["n_cumulative_repeaters"] == 2
    assert jan["cumulative_repeat_rate"] == pytest.approx(2 / 3)

    monthly = (cohort[(cohort["cohort_month"] == JAN) & (cohort["month_index"] >= 1)]
               ["retention_rate"].sum())
    assert not np.isclose(monthly, jan["cumulative_repeat_rate"])


def test_cumulative_repeat_rate_excludes_immature_cohorts(synthetic_orders, cohort):
    """A 12-month requirement cannot be met by a 3-month fixture."""
    assert ca.cumulative_repeat_rate(cohort, synthetic_orders, upto=12).empty


def test_retention_kpis_are_bounded(cohort, synthetic_orders):
    kpis = ca.retention_kpis(cohort, synthetic_orders)
    for key in ("month_1_retention_pooled", "month_2_retention_pooled"):
        assert 0 <= kpis[key] <= 1
    assert kpis["n_cohorts"] == 2


def test_retention_curves_respect_min_size(cohort):
    curves = ca.retention_curves(cohort, min_size=3)
    assert set(curves["cohort_month"].unique()) == {JAN}


def test_cohort_summary_maturity_flag(cohort):
    summary = ca.cohort_summary(cohort, upto=12).set_index("cohort_month")
    assert summary.loc[JAN, "is_mature"] == False     # only 2 observable months
    assert summary.loc[JAN, "max_observable_index"] == 2
    assert summary.loc[FEB, "max_observable_index"] == 1


def test_single_customer_dataset_edge_case():
    one = pd.DataFrame({
        "customer_unique_id": ["solo"],
        "order_id": ["o1"],
        "order_purchase_timestamp": [pd.Timestamp("2017-05-01 10:00:00")],
    })
    table = ca.build_cohort_table(one)
    assert len(table) == 1
    assert table.iloc[0]["retention_rate"] == 1.0
    assert ca.validate_cohort_table(table) == []


def test_no_orders_returns_empty_cohort_table():
    empty = pd.DataFrame({
        "customer_unique_id": pd.Series([], dtype="object"),
        "order_id": pd.Series([], dtype="object"),
        "order_purchase_timestamp": pd.Series([], dtype="datetime64[ns]"),
    })
    table = ca.build_cohort_table(empty)
    assert table.empty
