"""RFM scoring, rule-based segments and the k-means extension (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import segmentation as sg


@pytest.fixture
def rfm_frame() -> pd.DataFrame:
    """
    Hand-built RFM frame with known segment membership.

    recency / frequency / value / days_since_first_order:
        c_highval   : 10 days, 3 orders, 500  -> High-Value Active Repeat
        c_active    : 20 days, 2 orders,  80  -> Active Repeat
        c_lapsing   : 150 days, 3 orders, 300 -> Lapsing Repeat
        c_new       : 15 days, 1 order,   50  -> New / Insufficient History
        c_recent1   : 100 days, 1 order,  40  -> Recent One-Time Buyer
        c_dormant   : 200 days, 1 order,  60  -> Dormant One-Time Buyer
        c_longgone  : 500 days, 2 orders, 400 -> Long Inactive
    """
    return pd.DataFrame({
        "customer_unique_id": ["c_highval", "c_active", "c_lapsing", "c_new",
                               "c_recent1", "c_dormant", "c_longgone"],
        "recency_days": [10, 20, 150, 15, 100, 200, 500],
        "frequency": [3, 2, 3, 1, 1, 1, 2],
        "monetary_value": [500.0, 80.0, 300.0, 50.0, 40.0, 60.0, 400.0],
        "days_since_first_order": [400, 300, 400, 15, 200, 300, 600],
        "n_orders": [3, 2, 3, 1, 1, 1, 2],
        "mean_review_score": [5.0, 4.0, 4.5, 5.0, 3.0, 4.0, 2.0],
    })


def test_quantile_scores_are_monotonic():
    s = pd.Series(range(100), dtype=float)
    scores = sg.quantile_scores(s)
    assert scores.is_monotonic_increasing
    assert set(scores.unique()) <= {1, 2, 3, 4, 5}


def test_recency_scoring_is_reversed():
    recent = pd.Series([1.0, 10.0, 50.0, 100.0, 500.0])
    scores = sg.quantile_scores(recent, reverse=True)
    assert scores.iloc[0] > scores.iloc[-1]      # fewer days => better score


def test_quantile_scores_handle_constant_series():
    s = pd.Series([5.0] * 10)
    scores = sg.quantile_scores(s)
    assert (scores == 3).all()


def test_rule_based_segments_assign_expected_labels(rfm_frame):
    out = sg.assign_rule_based_segments(
        rfm_frame, lapsing_days=120, inactive_days=250,
        high_value_cut=400.0, insufficient_days=60)
    labels = out.set_index("customer_unique_id")["segment"]
    assert labels["c_highval"] == "High-Value Active Repeat"
    assert labels["c_active"] == "Active Repeat"
    assert labels["c_lapsing"] == "Lapsing Repeat"
    assert labels["c_new"] == "New / Insufficient History"
    assert labels["c_recent1"] == "Recent One-Time Buyer"
    assert labels["c_dormant"] == "Dormant One-Time Buyer"
    assert labels["c_longgone"] == "Long Inactive"


def test_every_customer_gets_exactly_one_segment(rfm_frame):
    out = sg.assign_rule_based_segments(rfm_frame, 120, 250, high_value_cut=400.0)
    assert out["segment"].notna().all()
    assert "Unclassified" not in set(out["segment"])
    assert len(out) == len(rfm_frame)


def test_high_value_flag_matches_threshold(rfm_frame):
    out = sg.assign_rule_based_segments(rfm_frame, 120, 250, high_value_cut=400.0)
    assert out.set_index("customer_unique_id").loc["c_highval", "is_high_value"]
    assert not out.set_index("customer_unique_id").loc["c_active", "is_high_value"]


def test_segment_profile_totals_reconcile(rfm_frame):
    out = sg.assign_rule_based_segments(rfm_frame, 120, 250, high_value_cut=400.0)
    out = sg.add_rfm_scores(out)
    profile = sg.segment_profile(out)

    assert profile["n_customers"].sum() == len(rfm_frame)
    assert profile["monetary_value"].sum() == pytest.approx(rfm_frame["monetary_value"].sum())
    assert profile["share_customers"].sum() == pytest.approx(1.0)
    assert profile["share_value"].sum() == pytest.approx(1.0)
    assert profile["n_orders"].sum() == rfm_frame["frequency"].sum()


def test_segment_profile_order_value_definition(rfm_frame):
    out = sg.assign_rule_based_segments(rfm_frame, 120, 250, high_value_cut=400.0)
    profile = sg.segment_profile(out).set_index("segment")
    # High-value segment: 500 value / 3 orders
    assert profile.loc["High-Value Active Repeat", "avg_order_value"] == pytest.approx(500 / 3)


def test_value_index_identity():
    profile = pd.DataFrame({
        "segment": ["A", "B"],
        "n_customers": [10, 90],
        "share_customers": [0.1, 0.9],
        "monetary_value": [500.0, 500.0],
        "share_value": [0.5, 0.5],
    })
    idx = sg.segment_value_index(profile).set_index("segment")["value_index"]
    assert idx["A"] == pytest.approx(5.0)
    assert idx["B"] == pytest.approx(0.5555, abs=1e-3)


def test_rfm_scores_added():
    df = pd.DataFrame({
        "customer_unique_id": [f"c{i}" for i in range(50)],
        "recency_days": np.arange(1, 51, dtype=float),
        "frequency": np.ones(50),
        "monetary_value": np.arange(1, 51, dtype=float),
        "n_orders": np.ones(50),
    })
    out = sg.add_rfm_scores(df)
    for col in ("r_score", "f_score", "m_score", "rfm_score", "rfm_cell"):
        assert col in out.columns
    assert out["rfm_score"].between(3, 15).all()


def test_kmeans_extension_returns_a_valid_structure(rfm_frame):
    res = sg.assign_kmeans_segments(rfm_frame, k_range=range(2, 4))
    assert res["best_k"] in (2, 3)
    assert set(res["data"]["cluster"]).issubset({0, 1, 2})
    assert len(res["profile"]) == res["best_k"]
    assert -1 <= res["best_silhouette"] <= 1
    assert res["profile"]["n_customers"].sum() == len(rfm_frame)


def test_high_value_threshold_quantile():
    s = pd.Series(np.arange(1, 101, dtype=float))
    assert sg.high_value_threshold(s, 0.8) == pytest.approx(80.2)
