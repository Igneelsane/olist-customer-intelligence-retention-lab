"""Tests for the data audit and quality-check layer (synthetic test data only)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import cleaning, data_validation as dv


def test_audit_reports_shape_and_missing():
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": ["x", "y", "x", None],
    })
    audit = dv.audit_table("orders", df)
    assert audit["rows"] == 4
    assert audit["columns"] == 2
    assert audit["missing_counts"]["a"] == 1
    assert audit["missing_pct"]["a"] == 25.0


def test_audit_detects_duplicate_key_rows():
    df = pd.DataFrame({"order_id": ["o1", "o1", "o2"]})
    audit = dv.audit_table("orders", df)
    assert audit["key_uniqueness"]["is_unique"] is False
    assert audit["key_uniqueness"]["duplicate_key_rows"] == 1


def test_audit_detects_negative_values():
    df = pd.DataFrame({"order_id": ["o1", "o2"], "price": [10.0, -5.0]})
    audit = dv.audit_table("order_items", df)
    assert audit["numeric_ranges"]["price"]["negative_count"] == 1


def test_quality_checks_pass_on_clean_pipeline_output(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    rep = dv.run_quality_checks({
        "orders": tables["orders"],
        "order_values": tables["orders"],
        "order_items_sum": tables["order_items_sum"],
        "payments_sum": tables["payments_sum"],
        "customers": tables["customers"],
    })
    assert rep.passed, [c.detail for c in rep.failures]


def test_quality_checks_catch_duplicate_orders(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    bad = pd.concat([tables["orders"], tables["orders"].iloc[[0]]], ignore_index=True)
    rep = dv.run_quality_checks({"orders": bad, "order_values": bad})
    assert not rep.passed
    assert any("unique" in c.name.lower() for c in rep.failures)


def test_quality_checks_catch_orphan_customers(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    cust = tables["customers"].iloc[:-1]          # remove a customer
    rep = dv.run_quality_checks({"orders": tables["orders"], "customers": cust})
    assert not rep.passed


def test_quality_checks_catch_negative_values(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    values = tables["orders"].copy()
    values.loc[0, "merchandise_value"] = -1.0
    rep = dv.run_quality_checks({"orders": values, "order_values": values})
    assert not rep.passed


def test_assert_quality_raises_on_failure(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    bad_customers = tables["customers"].iloc[:-1]
    with pytest.raises(AssertionError):
        dv.assert_quality({"orders": tables["orders"], "customers": bad_customers})


def test_quality_report_renders_markdown(synthetic_raw):
    tables = cleaning.build_analytical_tables(synthetic_raw)
    rep = dv.run_quality_checks({"orders": tables["orders"]})
    md = rep.to_markdown()
    assert "| Check | Result | Detail |" in md
