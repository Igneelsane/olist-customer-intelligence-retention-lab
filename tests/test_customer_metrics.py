"""
Customer-level metric tests.

Expected values are computed by hand from the synthetic fixture in conftest.py:

    customer A: 3 orders, value 220
    customer B: 1 order,  value  30
    customer C: 2 orders, value  60
    customer D: 1 order,  value  60
    customer E: 2 orders, value 170  (two orders 5 days apart)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import customer_metrics as cm


@pytest.fixture
def customers(synthetic_orders):
    return (synthetic_orders.groupby("customer_unique_id", as_index=False)
            .agg(n_orders=("order_id", "nunique"),
                 merchandise_value=("merchandise_value", "sum"),
                 first_order_ts=("order_purchase_timestamp", "min"),
                 last_order_ts=("order_purchase_timestamp", "max")))


def test_repeat_purchase_rate_denominator(customers):
    stats = cm.repeat_purchase_stats(customers)
    # A, C, E repeat -> 3 / 5
    assert stats["n_customers"] == 5
    assert stats["n_repeat_customers"] == 3
    assert stats["repeat_purchase_rate"] == pytest.approx(0.6)
    assert stats["n_one_time_customers"] == 2


def test_purchase_intervals_use_only_observed_gaps(synthetic_orders):
    gaps = cm.purchase_intervals(synthetic_orders)
    # 4 gaps: A(36, 23), C(35), E(5). One-time buyers contribute none.
    assert len(gaps) == 4
    assert set(np.round(gaps["gap_days"], 1)) == {36.0, 23.0, 35.0, 5.0}
    assert (gaps["gap_days"] > 0).all()


def test_time_to_second_purchase(synthetic_orders):
    second = cm.time_to_second_purchase(synthetic_orders)
    assert len(second) == 3                       # A, C, E
    values = second.set_index("customer_unique_id")["days_to_second_order"]
    assert values["A"] == pytest.approx(36.0)
    assert values["C"] == pytest.approx(35.0)
    assert values["E"] == pytest.approx(5.0)


def test_interval_summary_median():
    gaps = pd.DataFrame({"gap_days": [5.0, 23.0, 35.0, 36.0]})
    s = cm.interval_summary(gaps)
    assert s["median_days"] == pytest.approx(29.0)
    assert s["share_within_30d"] == pytest.approx(0.5)


def test_gini_zero_for_equal_values():
    assert cm.gini(np.array([10.0, 10.0, 10.0, 10.0])) == pytest.approx(0.0, abs=1e-9)


def test_gini_high_for_unequal_values():
    assert cm.gini(np.array([0.0, 0.0, 0.0, 100.0])) > 0.7


def test_concentration_top_share(customers):
    stats = cm.concentration_stats(customers.rename(columns={"merchandise_value": "merchandise_value"}))
    # A (220) alone is the top 20% of 5 customers -> 220/540
    assert stats["total_merchandise_value"] == pytest.approx(540.0)
    assert stats["top_20pct_share"] == pytest.approx(220 / 540)
    assert 0 <= stats["gini"] <= 1


def test_analysis_dates_use_last_order_not_today(synthetic_orders):
    dates = cm.analysis_dates(synthetic_orders)
    assert dates["first_order_date"] == pd.Timestamp("2017-01-05")
    assert dates["as_of_date"] == pd.Timestamp("2017-03-05")


def test_classify_activity_labels(customers):
    df = customers.copy()
    df["last_order_date"] = df["last_order_ts"].dt.normalize()
    df["first_order_date"] = df["first_order_ts"].dt.normalize()
    out = cm.classify_activity(df, pd.Timestamp("2017-03-05"),
                               lapsing_days=30, inactive_days=60,
                               insufficient_history_days=7)
    labels = out.set_index("customer_unique_id")["activity_status"]
    assert labels["A"] == "Recently active"      # last order 2017-03-05
    assert labels["B"] == "Potentially lapsing"  # 44 days
    assert labels["E"] == "Recently active"      # 8 days, 2 orders


def test_insufficient_history_only_applies_to_single_order_customers():
    df = pd.DataFrame({
        "customer_unique_id": ["x", "y"],
        "n_orders": [1, 3],
        "first_order_date": [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-01")],
        "last_order_date": [pd.Timestamp("2019-01-05"), pd.Timestamp("2019-01-05")],
    })
    out = cm.classify_activity(df, pd.Timestamp("2019-01-10"), 30, 60, insufficient_history_days=60)
    labels = out.set_index("customer_unique_id")["activity_status"]
    assert labels["x"] == "New / insufficient history"
    assert labels["y"] == "Recently active"


def test_derive_thresholds_respect_floors():
    small = pd.DataFrame({"gap_days": [1.0, 2.0, 3.0, 4.0]})
    t = cm.derive_inactivity_thresholds(small, lapsing_floor=90, inactive_floor=180)
    assert t["lapsing_days"] == 90
    assert t["inactive_days"] == 180


def test_derive_thresholds_use_data_when_larger():
    big = pd.DataFrame({"gap_days": list(range(1, 501))})
    t = cm.derive_inactivity_thresholds(big, lapsing_floor=90, inactive_floor=180)
    assert t["lapsing_days"] > 90
    assert t["inactive_days"] > t["lapsing_days"]


def test_monthly_new_vs_returning_adds_up(synthetic_orders):
    split = cm.monthly_new_vs_returning(synthetic_orders)
    assert (split["total_customers"] == split["new_customers"] + split["returning_customers"]).all()
    # January 2017: A, B, C are all first-time buyers.
    jan = split[split["month"] == pd.Timestamp("2017-01-01")].iloc[0]
    assert jan["new_customers"] == 3
    assert jan["returning_customers"] == 0
    # March 2017: A and C return; nobody is new.
    mar = split[split["month"] == pd.Timestamp("2017-03-01")].iloc[0]
    assert mar["new_customers"] == 0
    assert mar["returning_customers"] == 2


def test_build_customer_metrics_summary(synthetic_orders, customers):
    res = cm.build_customer_metrics(synthetic_orders, customers)
    s = res.summary
    assert s["n_customers"] == 5
    assert s["n_orders"] == 9
    assert s["merchandise_value"] == pytest.approx(540.0)
    assert s["avg_order_value"] == pytest.approx(60.0)
    assert s["repeat_purchase_rate"] == pytest.approx(0.6)
    assert "monetary_value" in res.customers.columns
