"""Scenario-simulator arithmetic and recommendation-catalogue tests (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import recommendations as rec


def test_simulate_scenario_arithmetic_is_exact():
    """
    Hand-computed example:
        size 1000, participation 0.2   -> 200 contacted
        incremental rate 0.05         -> 10 incremental orders
        baseline rate 0.10            -> 20 baseline orders
        AOV 100                       -> incremental revenue 1,000
        margin 0.30                   -> gross contribution 300
        redeemed offers 30, incentive 10 -> incentive cost 300
        contact cost 0.50 * 200 = 100
        net = 300 - 300 - 100 = -100
    """
    out = rec.simulate_scenario(
        segment_size=1000, participation_rate=0.2, incremental_repeat_rate=0.05,
        average_order_value=100.0, contribution_margin=0.30,
        cost_per_contact=0.50, incentive_cost_per_redemption=10.0,
        baseline_repeat_rate=0.10,
    )
    assert out["contacted"] == pytest.approx(200.0)
    assert out["incremental_orders"] == pytest.approx(10.0)
    assert out["baseline_orders"] == pytest.approx(20.0)
    assert out["incremental_revenue"] == pytest.approx(1000.0)
    assert out["gross_contribution"] == pytest.approx(300.0)
    assert out["redeemed_offers"] == pytest.approx(30.0)
    assert out["incentive_cost"] == pytest.approx(300.0)
    assert out["contact_cost"] == pytest.approx(100.0)
    assert out["net_contribution"] == pytest.approx(-100.0)
    assert out["wasted_incentive"] == pytest.approx(200.0)


def test_zero_participation_gives_zero_everything():
    out = rec.simulate_scenario(1000, 0.0, 0.1, 100.0, 0.3, 1.0, 5.0)
    assert out["contacted"] == 0
    assert out["incremental_orders"] == 0
    assert out["net_contribution"] == pytest.approx(0.0)


def test_roi_nan_when_no_cost():
    out = rec.simulate_scenario(100, 1.0, 0.1, 100.0, 0.3, 0.0, 0.0, baseline_repeat_rate=0.0)
    assert np.isnan(out["roi"])
    assert out["net_contribution"] > 0


def test_scenario_range_orders_scenarios():
    band = rec.scenario_range(1000, 0.2, 0.05, 100.0, 0.3, 0.5, 10.0, baseline_repeat_rate=0.1)
    assert list(band["scenario"]) == ["Low", "Base", "High"]
    assert band.loc[0, "incremental_orders"] < band.loc[1, "incremental_orders"]
    assert band.loc[1, "incremental_orders"] < band.loc[2, "incremental_orders"]


def test_breakeven_rate_is_the_zero_net_point():
    kwargs = dict(segment_size=1000, participation_rate=0.2, average_order_value=100.0,
                  contribution_margin=0.4, cost_per_contact=0.5,
                  incentive_cost_per_redemption=5.0, baseline_repeat_rate=0.05)
    be = rec.breakeven_incremental_rate(**kwargs)
    net = rec.simulate_scenario(incremental_repeat_rate=be, **kwargs)["net_contribution"]
    assert net == pytest.approx(0.0, abs=1e-6)


def test_breakeven_infinite_when_margin_below_incentive():
    be = rec.breakeven_incremental_rate(
        segment_size=1000, participation_rate=0.2, average_order_value=10.0,
        contribution_margin=0.1, cost_per_contact=0.5,
        incentive_cost_per_redemption=20.0)
    assert np.isinf(be)


@pytest.fixture
def profile():
    return pd.DataFrame({
        "segment": ["High-Value Active Repeat", "Active Repeat", "Lapsing Repeat",
                    "Recent One-Time Buyer", "Dormant One-Time Buyer", "Long Inactive",
                    "New / Insufficient History"],
        "n_customers": [500, 1000, 800, 20000, 15000, 50000, 6000],
        "n_orders": [1500, 2500, 2000, 20000, 15000, 52000, 6000],
        "monetary_value": [150000.0, 120000.0, 100000.0, 400000.0, 250000.0, 300000.0, 80000.0],
        "share_customers": [0.005, 0.01, 0.008, 0.21, 0.16, 0.53, 0.06],
        "share_value": [0.107, 0.086, 0.071, 0.286, 0.179, 0.214, 0.057],
        "avg_recency_days": [10.0, 20.0, 150.0, 30.0, 200.0, 500.0, 15.0],
        "avg_frequency": [3.0, 2.5, 2.5, 1.0, 1.0, 1.04, 1.0],
        "avg_order_value": [100.0, 48.0, 50.0, 20.0, 16.7, 5.8, 13.3],
        "avg_merchandise_per_customer": [300.0, 120.0, 125.0, 20.0, 16.7, 6.0, 13.3],
        "share_repeat": [1.0, 1.0, 1.0, 0.0, 0.0, 0.04, 0.0],
        "mean_review_score": [4.5, 4.2, 4.0, 4.1, 3.8, 3.5, 4.3],
    })


def test_recommendations_have_all_required_fields(profile):
    summary = {"repeat_purchase_rate": 0.03, "n_customers": 93000,
               "n_one_time_customers": 90000, "median_days_to_second_order": 25.0}
    thresholds = {"lapsing_days": 121.5, "inactive_days": 240.3,
                  "gap_p75_days": 121.5, "gap_p90_days": 240.3}
    recs = rec.build_recommendations(profile, summary, thresholds)
    assert len(recs) >= 5
    for r in recs:
        assert r.target_segment
        assert len(r.evidence) >= 2
        assert r.action and r.mechanism and r.kpi and r.experiment_design
        assert len(r.costs_and_risks) >= 1
        assert r.uncertainty


def test_recommendations_quote_real_numbers(profile):
    summary = {"repeat_purchase_rate": 0.0312, "n_customers": 93000,
               "n_one_time_customers": 90000, "median_days_to_second_order": 25.0}
    recs = rec.build_recommendations(profile, summary,
                                     {"lapsing_days": 121.5, "inactive_days": 240.3})
    text = " ".join(" ".join(r.evidence) for r in recs)
    assert "500" in text            # High-Value Active Repeat customer count
    assert "3.12%" in text          # repeat-purchase rate appears as computed


def test_recommendations_sorted_by_priority(profile):
    recs = rec.build_recommendations(profile, {"repeat_purchase_rate": 0.03},
                                     {"lapsing_days": 100, "inactive_days": 200})
    rank = {"High": 0, "Medium": 1, "Low": 2}
    values = [rank[r.priority] for r in recs]
    assert values == sorted(values)


def test_recommendations_frame_is_flat(profile):
    recs = rec.build_recommendations(profile, {}, {"lapsing_days": 100, "inactive_days": 200})
    frame = rec.recommendations_frame(recs)
    assert "evidence" in frame.columns and len(frame) == len(recs)


def test_empty_profile_produces_no_crash():
    empty = pd.DataFrame(columns=["segment", "n_customers", "share_customers",
                                  "monetary_value", "share_value", "avg_recency_days",
                                  "avg_frequency", "avg_order_value", "share_repeat"])
    recs = rec.build_recommendations(empty, {}, {"lapsing_days": 100, "inactive_days": 200})
    assert recs == []
