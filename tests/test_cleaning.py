"""
Tests for cleaning and order-level aggregation.

Fixture data is SYNTHETIC test data (see conftest.py), used only to verify
formulas. It never touches the real Olist records.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import cleaning


def test_order_eligibility_keeps_only_delivered(synthetic_raw):
    orders = cleaning.clean_orders(synthetic_raw["orders"], synthetic_raw["customers"])
    # o1, o2, o3, o4 are delivered; o5 is canceled and o6 is shipped.
    assert set(orders["order_id"]) == {"o1", "o2", "o3", "o4"}
    assert (orders["order_status"] == "delivered").all()


def test_eligibility_log_records_dropped_rows(synthetic_raw):
    log = cleaning.CleaningLog()
    cleaning.clean_orders(synthetic_raw["orders"], synthetic_raw["customers"], log=log)
    frame = log.to_frame()
    assert len(frame) >= 1
    assert frame["rows_removed"].sum() == 2          # canceled + shipped


def test_broad_status_variant_adds_shipped(synthetic_raw):
    from config import settings
    orders = cleaning.clean_orders(
        synthetic_raw["orders"], synthetic_raw["customers"],
        statuses=settings.BROAD_ELIGIBLE_STATUSES)
    assert "o6" in set(orders["order_id"])        # shipped now counts
    assert "o5" not in set(orders["order_id"])    # canceled still does not


def test_customer_identity_joined_from_unique_id(synthetic_raw):
    orders = cleaning.clean_orders(synthetic_raw["orders"], synthetic_raw["customers"])
    mapping = orders.set_index("order_id")["customer_unique_id"].to_dict()
    assert mapping["o1"] == "u1"
    assert mapping["o4"] == "u1"     # different customer_id, same person
    assert orders["customer_unique_id"].notna().all()


def test_item_aggregation_sums_one_to_many_correctly(synthetic_raw):
    items = cleaning.aggregate_order_items(synthetic_raw["order_items"],
                                           synthetic_raw["products"],
                                           synthetic_raw["category_translation"])
    assert items["order_id"].is_unique
    row = items.set_index("order_id").loc["o1"]
    assert row["n_items"] == 2
    assert row["merchandise_value"] == pytest.approx(100.0)     # 40 + 60
    assert row["freight_value"] == pytest.approx(12.0)          # 5 + 7
    assert row["n_sellers"] == 2
    assert row["main_category"] in {"health_beauty", "computers_accessories"}


def test_payment_aggregation_sums_multiple_records(synthetic_raw):
    pay = cleaning.aggregate_payments(synthetic_raw["payments"])
    assert pay["order_id"].is_unique
    row = pay.set_index("order_id").loc["o1"]
    assert row["payment_value"] == pytest.approx(110.0)   # 60 + 50
    assert row["n_payments"] == 2


def test_order_values_has_no_fan_out(synthetic_raw):
    """The critical double-counting test: joins must not multiply orders."""
    tables = cleaning.build_analytical_tables(synthetic_raw)
    orders = tables["orders"]
    assert len(orders) == 4                    # 4 delivered orders
    assert orders["order_id"].is_unique
    assert orders["merchandise_value"].sum() == pytest.approx(100 + 100 + 200 + 150)


def test_merchandise_value_reconciles_independently(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    orders = tables["orders"]

    # Independent computation straight from the raw item rows.
    raw = synthetic_raw["order_items"]
    raw = raw[raw["order_id"].isin(set(orders["order_id"]))]
    expected = raw["price"].sum()

    assert orders["merchandise_value"].sum() == pytest.approx(expected)
    assert orders["freight_value"].sum() == pytest.approx(
        raw["freight_value"].sum())


def test_reviews_keep_latest_score_per_order(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    orders = tables["orders"].set_index("order_id")
    # o1 has two reviews (5 then 4); the latest by answer timestamp wins.
    assert orders.loc["o1", "review_score"] == 4
    assert tables["orders"]["order_id"].is_unique


def test_customer_dimension_is_one_row_per_unique_id(synthetic_raw):
    dim = cleaning.build_customer_dimension(synthetic_raw["customers"])
    assert dim["customer_unique_id"].is_unique
    assert len(dim) == 3           # u1, u2, u3
    u1 = dim.set_index("customer_unique_id").loc["u1"]
    assert u1["n_customer_ids"] == 2


def test_customer_rollup_uses_unique_ids(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    cust = tables["customers"]
    assert len(cust) == 3                       # 3 people, not 4 orders
    u1 = cust.set_index("customer_unique_id").loc["u1"]
    assert u1["n_orders"] == 2                  # o1 and o4
    assert u1["merchandise_value"] == pytest.approx(250.0)   # 100 + 150


def test_invalid_timestamps_are_dropped(synthetic_raw):
    raw = {k: (v.copy() if isinstance(v, pd.DataFrame) else v)
           for k, v in synthetic_raw.items()}
    raw["orders"].loc[0, "order_purchase_timestamp"] = pd.NaT
    orders = cleaning.clean_orders(raw["orders"], raw["customers"])
    assert "o1" not in set(orders["order_id"])
