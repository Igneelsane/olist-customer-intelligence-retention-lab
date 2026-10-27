"""
Export a compact, anonymised derived dataset for the Vercel deployment.

Why this exists
---------------
Streamlit cannot run on Vercel (it needs a persistent process and WebSockets;
Vercel Functions are request-scoped). The Vercel build therefore ships a
*Vercel-native* interface: a Python function that recomputes the analytics on
request from a compact derived table, plus a static dashboard that calls it.

This script produces that derived table from the real Olist dataset, using the
exact same pipeline (`src/cleaning.py`) that the Streamlit app uses.

What is exported (one row per eligible order, no identifiers):

    c   customer index        integer, assigned by order of first appearance
    t   order timestamp       SECONDS since the analysis epoch (2016-09-01 00:00)
    v   merchandise value     integer cents (Σ order_items.price)
    f   freight value         integer cents
    r   review score          0 = missing, else 1..5 (latest review per order)
    y   delivery days         HUNDREDTHS of a day (-1 = unknown)
    z   delay vs promise      HUNDREDTHS of a day (999 = unknown, negative = early)
    s   customer state        2-letter code (aggregate geography only)
    k   category index        into the `categories` list in the metadata

No customer_unique_id / customer_id / city / zip / free text is exported.
The customer index is a dense enumeration and cannot be mapped back to a person.

Usage:
    python scripts/export_public_data.py
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings                     # noqa: E402
from src import cleaning, customer_metrics, data_loader, data_validation  # noqa: E402

OUT_DIR = ROOT / "deploy" / "vercel"
DATA_DIR = OUT_DIR / "data"
EPOCH = pd.Timestamp("2016-09-01")
CHUNK_CHARS = 700_000          # ~0.7 MB of base64 per physical file


def build_export() -> tuple[pd.DataFrame, dict]:
    raw = data_loader.load_raw_tables(tables=(
        "orders", "customers", "order_items", "payments", "reviews",
        "products", "category_translation"))
    tables = cleaning.build_analytical_tables(raw)
    orders = tables["orders"].copy()

    metrics = customer_metrics.build_customer_metrics(orders, tables["customers"])

    # Dense customer index -- deliberately not derived from any identifier.
    cust_codes = {cid: i for i, cid in enumerate(
        orders.sort_values("order_purchase_timestamp")["customer_unique_id"].drop_duplicates())}

    cat_list = sorted(orders["main_category"].dropna().unique().tolist())
    cat_index = {c: i + 1 for i, c in enumerate(cat_list)}

    out = pd.DataFrame({
        "c": orders["customer_unique_id"].map(cust_codes).astype(int),
        # Second precision matters: the 90-day repeat window and the
        # right-censoring rule are timestamp comparisons, not date comparisons,
        # and "next order" must be resolved without collapsing same-day orders.
        "t": ((orders["order_purchase_timestamp"] - EPOCH).dt.total_seconds()).astype("int64"),
        "v": (orders["merchandise_value"] * 100).round().astype(int),
        "f": (orders["freight_value"] * 100).round().astype(int),
        "r": orders["review_score"].fillna(0).astype(int),
        # Hundredths of a day: rounding delivery deltas to whole days would
        # change "share late" and the lateness bins.
        "y": (orders["delivery_days"] * 100).round().fillna(-100).astype("int64"),
        "z": (orders["delay_days"] * 100).round().fillna(99900).astype("int64"),
        "s": orders["customer_state"].fillna("??"),
        "k": orders["main_category"].map(cat_index).fillna(0).astype(int),
    })
    out = out.sort_values(["c", "t"]).reset_index(drop=True)

    quality = data_validation.run_quality_checks({
        "orders": orders, "order_values": orders,
        "order_items_sum": tables["order_items_sum"],
        "payments_sum": tables["payments_sum"],
        "customers": tables["customers"]})

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "Brazilian E-Commerce Public Dataset by Olist",
        "dataset_url": settings.DATASET_SOURCE_URL,
        "dataset_license": settings.DATASET_LICENSE,
        "epoch": str(EPOCH.date()),
        "epoch_iso": EPOCH.strftime("%Y-%m-%dT%H:%M:%S"),
        "as_of_seconds": int(
            (pd.Timestamp(metrics.dates["as_of_date"]).normalize() - EPOCH).total_seconds()),
        "as_of_note": (
            "Censoring uses the pipeline's end-of-observation date at 00:00, exactly as "
            "src/service_analysis.py does, so the deployed numbers match the local app."),
        "first_order_date": str(metrics.dates["first_order_date"].date()),
        "as_of_date": str(metrics.dates["as_of_date"].date()),
        "observation_days": int(metrics.dates["observation_days"]),
        "n_orders": int(len(out)),
        "n_customers": int(out["c"].nunique()),
        "eligible_statuses": list(settings.PRIMARY_ELIGIBLE_STATUSES),
        "identity_key": "customer_unique_id",
        "categories": cat_list,
        "states": sorted(out["s"].unique().tolist()),
        "columns": {
            "c": "customer index (dense enumeration, not derived from any identifier)",
            "t": "order purchase timestamp in SECONDS since the epoch (timezone as shipped by Olist)",
            "v": "merchandise value, integer cents (Σ order_items.price)",
            "f": "freight value, integer cents",
            "r": "latest review score for the order (0 = missing)",
            "y": "purchase-to-delivery days x100 (-100 = unknown)",
            "z": "(delivered minus estimated delivery) days x100 (99900 = unknown, negative = early)",
            "s": "customer state (2-letter)",
            "k": "dominant category index (0 = unknown)",
        },
        "pipeline_quality_checks": {
            "checks_run": len(quality.checks),
            "passed": quality.passed,
            "failures": [c.name for c in quality.failures],
        },
        "anonymisation": (
            "No customer_unique_id, customer_id, city, zip code, review text or order id is "
            "exported. Customer and category references are dense integer indices. The table is "
            "not linkable to the public dataset's raw rows."),
        "reproduce": "python scripts/export_public_data.py",
    }
    return out, meta


def write_payload(df: pd.DataFrame, meta: dict) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("orders.*.b64"):
        old.unlink()

    csv_bytes = df.to_csv(index=False, header=False).encode("utf-8")
    gz = gzip.compress(csv_bytes, compresslevel=9)
    b64 = base64.b64encode(gz).decode("ascii")

    chunks = [b64[i:i + CHUNK_CHARS] for i in range(0, len(b64), CHUNK_CHARS)]
    for i, chunk in enumerate(chunks):
        (DATA_DIR / f"orders.{i:02d}.b64").write_text(chunk, encoding="ascii")

    meta = dict(meta)
    meta["payload"] = {
        "encoding": "base64(gzip(csv))",
        "files": [f"orders.{i:02d}.b64" for i in range(len(chunks))],
        "csv_bytes": len(csv_bytes),
        "gz_bytes": len(gz),
        "b64_chars": len(b64),
        "rows": int(len(df)),
        "sha256_of_csv": hashlib.sha256(csv_bytes).hexdigest(),
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> dict:
    df, meta = build_export()
    meta = write_payload(df, meta)
    p = meta["payload"]
    print(f"rows              : {p['rows']:,}")
    print(f"csv bytes         : {p['csv_bytes']:,}")
    print(f"gzip bytes        : {p['gz_bytes']:,}")
    print(f"base64 chars      : {p['b64_chars']:,} in {len(p['files'])} file(s)")
    print(f"quality checks ok : {meta['pipeline_quality_checks']['passed']}")
    print(f"written to        : {DATA_DIR}")
    return meta


if __name__ == "__main__":
    main()
