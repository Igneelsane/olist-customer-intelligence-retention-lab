"""
Shared pytest fixtures.

ALL DATA IN THIS FILE IS SYNTHETIC TEST DATA.

It exists only to verify formulas on hand-checkable numbers. It is never loaded
by the application, never combined with the real Olist records, and never used
to produce a business finding.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _dt(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ---------------------------------------------------------------------------
# Minimal order-level frame (used by metric and cohort tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_orders() -> pd.DataFrame:
    """
    Five customers, eight orders, hand-checkable retention.

    customer A: o1 2017-01-05, o2 2017-02-10, o3 2017-03-05
    customer B: o4 2017-01-20
    customer C: o5 2017-01-25, o6 2017-03-01
    customer D: o7 2017-02-15
    customer E: o8 2017-02-20, o9 2017-02-25

    Jan-2017 cohort = {A, B, C} (size 3)
        month 0 -> 3/3 = 1.000
        month 1 -> {A} = 1/3
        month 2 -> {A, C} = 2/3
    Feb-2017 cohort = {D, E} (size 2)
        month 0 -> 2/2 = 1.000
        month 1 -> 0/2 = 0.000 (observable: last month in data is 2017-03)
    """
    rows = [
        ("o1", "c1", "A", "2017-01-05 10:00:00", 100.0),
        ("o2", "c2", "A", "2017-02-10 10:00:00", 50.0),
        ("o3", "c3", "A", "2017-03-05 10:00:00", 70.0),
        ("o4", "c4", "B", "2017-01-20 10:00:00", 30.0),
        ("o5", "c5", "C", "2017-01-25 10:00:00", 20.0),
        ("o6", "c6", "C", "2017-03-01 10:00:00", 40.0),
        ("o7", "c7", "D", "2017-02-15 10:00:00", 60.0),
        ("o8", "c8", "E", "2017-02-20 10:00:00", 80.0),
        ("o9", "c9", "E", "2017-02-25 10:00:00", 90.0),
    ]
    df = pd.DataFrame(rows, columns=[
        "order_id", "customer_id", "customer_unique_id",
        "order_purchase_timestamp", "merchandise_value"])
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_status"] = "delivered"
    df["freight_value"] = 10.0
    df["n_items"] = 1
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    df["review_score"] = 5.0
    return df


# ---------------------------------------------------------------------------
# Raw source tables (used by pipeline / cleaning / validation tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_raw() -> dict:
    """Raw source tables mirroring the Olist schema, with known values."""
    customers = pd.DataFrame([
        # customer_id is order-scoped; u1 and u3 each own two customer_id values
        ("c1", "u1", 1111, "sao paulo", "SP"),
        ("c2", "u2", 2222, "rio de janeiro", "RJ"),
        ("c3", "u3", 3333, "curitiba", "PR"),
        ("c4", "u1", 1111, "sao paulo", "SP"),
        ("c5", "u3", 3333, "curitiba", "PR"),
    ], columns=["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                "customer_city", "customer_state"])

    orders = pd.DataFrame([
        ("o1", "c1", "delivered", "2017-01-05 10:00:00", "2017-01-12 10:00:00", "2017-01-15 00:00:00"),
        ("o2", "c2", "delivered", "2017-01-20 10:00:00", "2017-02-01 10:00:00", "2017-01-28 00:00:00"),
        ("o3", "c3", "delivered", "2017-02-10 10:00:00", "2017-02-18 10:00:00", "2017-02-20 00:00:00"),
        ("o4", "c4", "delivered", "2017-03-15 10:00:00", "2017-03-25 10:00:00", "2017-03-22 00:00:00"),
        ("o5", "c5", "canceled", "2017-03-20 10:00:00", None, "2017-03-30 00:00:00"),
        ("o6", "c1", "shipped", "2017-04-02 10:00:00", None, "2017-04-20 00:00:00"),
    ], columns=["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                "order_delivered_customer_date", "order_estimated_delivery_date"])
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])
    orders["order_approved_at"] = orders["order_purchase_timestamp"]
    orders["order_delivered_carrier_date"] = orders["order_purchase_timestamp"]

    order_items = pd.DataFrame([
        # o1 has two items (one-to-many: this is the double-counting trap)
        ("o1", 1, "p1", "s1", "2017-01-06 10:00:00", 40.0, 5.0),
        ("o1", 2, "p2", "s2", "2017-01-06 10:00:00", 60.0, 7.0),
        ("o2", 1, "p1", "s1", "2017-01-21 10:00:00", 100.0, 12.0),
        ("o3", 1, "p3", "s3", "2017-02-11 10:00:00", 200.0, 15.0),
        ("o4", 1, "p1", "s1", "2017-03-16 10:00:00", 150.0, 8.0),
        ("o5", 1, "p2", "s2", "2017-03-21 10:00:00", 25.0, 4.0),
        ("o6", 1, "p3", "s3", "2017-04-03 10:00:00", 300.0, 20.0),
    ], columns=["order_id", "order_item_id", "product_id", "seller_id",
                "shipping_limit_date", "price", "freight_value"])
    order_items["shipping_limit_date"] = pd.to_datetime(order_items["shipping_limit_date"])

    payments = pd.DataFrame([
        # o1 paid in two instalments (another one-to-many trap)
        ("o1", 1, "credit_card", 2, 60.0),
        ("o1", 2, "voucher", 1, 50.0),
        ("o2", 1, "credit_card", 1, 112.0),
        ("o3", 1, "boleto", 1, 215.0),
        ("o4", 1, "credit_card", 3, 158.0),
        ("o5", 1, "credit_card", 1, 29.0),
        ("o6", 1, "credit_card", 1, 320.0),
    ], columns=["order_id", "payment_sequential", "payment_type",
                "payment_installments", "payment_value"])

    reviews = pd.DataFrame([
        ("r1", "o1", 5, None, "2017-01-13 00:00:00", "2017-01-14 00:00:00"),
        ("r2", "o1", 4, "later review", "2017-01-16 00:00:00", "2017-01-17 00:00:00"),
        ("r3", "o2", 1, "late delivery", "2017-02-02 00:00:00", "2017-02-03 00:00:00"),
        ("r4", "o3", 5, None, "2017-02-19 00:00:00", "2017-02-20 00:00:00"),
        ("r5", "o4", 3, None, "2017-03-26 00:00:00", "2017-03-27 00:00:00"),
    ], columns=["review_id", "order_id", "review_score", "review_comment_message",
                "review_creation_date", "review_answer_timestamp"])
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"])

    products = pd.DataFrame([
        ("p1", "beleza_saude"), ("p2", "informatica_acessorios"), ("p3", "moveis_decoracao"),
    ], columns=["product_id", "product_category_name"])

    translation = pd.DataFrame([
        ("beleza_saude", "health_beauty"),
        ("informatica_acessorios", "computers_accessories"),
        ("moveis_decoracao", "furniture_decor"),
    ], columns=["product_category_name", "product_category_name_english"])

    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "category_translation": translation,
    }
