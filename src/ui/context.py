"""
Application data context.

Loads the raw tables once, builds the analytical tables once, then serves every
page from the same in-memory objects. Streamlit caching is keyed on a
fingerprint of the raw files, so replacing the dataset invalidates the cache
automatically.

If processed parquet artifacts exist and are newer than the raw files they are
reused; otherwise the full pipeline runs and the artifacts are rewritten. Both
paths execute the same functions in ``src/`` -- there is no separate logic for
the app.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from config import settings
from src import (
    cleaning,
    cohort_analysis,
    customer_metrics,
    data_loader,
    data_validation,
    recommendations,
    segmentation,
    service_analysis,
)


@dataclass
class AppContext:
    # ``raw_meta`` holds the small facts the UI needs about the raw files
    # (row counts, status distribution, identity counts, audit results). The
    # raw frames themselves are released after the pipeline runs so the cached
    # context stays small enough for a 2 GB machine.
    raw_meta: Dict[str, Any]
    tables: Dict[str, Any]
    metrics: customer_metrics.CustomerMetrics
    cohort: pd.DataFrame
    seg: segmentation.SegmentationResult
    reviews: Dict[str, Any]
    delivery: Dict[str, Any]
    quality: data_validation.QualityReport
    data_dir: str
    used_cache: bool


def _fingerprint(data_dir: Path) -> str:
    parts = []
    for name, path in sorted(data_loader.available_files(data_dir).items()):
        stat = path.stat()
        parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def _processed_is_fresh(data_dir: Path) -> bool:
    processed = settings.PROCESSED_DATA_DIR
    orders_p = processed / "orders.parquet"
    if not orders_p.exists():
        return False
    newest_raw = max((p.stat().st_mtime for p in data_loader.available_files(data_dir).values()),
                     default=0)
    return orders_p.stat().st_mtime >= newest_raw


# Geolocation (1M rows of lat/lng) is not used by any KPI, so it is excluded
# from the application load to keep memory bounded.
APP_TABLES = ("orders", "customers", "order_items", "payments", "reviews",
              "products", "sellers", "category_translation")


def _build(data_dir: Path, use_cache: bool) -> AppContext:
    raw = data_loader.load_raw_tables(data_dir, tables=APP_TABLES)

    # Facts about the raw files, captured before the frames are released.
    raw_meta: Dict[str, Any] = {
        "row_counts": {name: int(len(df)) for name, df in raw.items()},
        "status_counts": (raw["orders"]["order_status"].value_counts().to_dict()
                          if "orders" in raw else {}),
        "n_customer_ids": int(raw["customers"]["customer_id"].nunique()) if "customers" in raw else 0,
        "n_unique_ids": int(raw["customers"]["customer_unique_id"].nunique()) if "customers" in raw else 0,
        "files": {name: settings.TABLE_FILES[name] for name in raw},
        "audit": [data_validation.audit_table(name, raw[name])
                  for name in ("orders", "customers", "order_items", "payments", "reviews")
                  if name in raw],
    }
    used_cache = False

    if use_cache and _processed_is_fresh(data_dir):
        cached = cleaning.load_processed()
        if cached is not None:
            tables = {
                "orders": cached["orders"],
                "customers": cached["customers"],
                "order_items_sum": cached["order_items_sum"],
                "payments_sum": cached["payments_sum"],
                "order_values": cached["orders"],
                "cleaning_log_frame": cached.get("cleaning_log_frame"),
            }
            used_cache = True

    if not used_cache:
        tables = cleaning.build_analytical_tables(raw)
        try:
            cleaning.save_processed(tables)
        except Exception:      # caching is an optimisation, never a hard failure
            pass

    del raw
    gc.collect()

    metrics = customer_metrics.build_customer_metrics(tables["order_values"], tables["customers"])
    tables_final = {**tables, "cohort": cohort_analysis.build_cohort_table(tables["order_values"])}
    quality = data_validation.run_quality_checks({
        "orders": tables["order_values"],
        "order_values": tables["order_values"],
        "order_items_sum": tables.get("order_items_sum"),
        "payments_sum": tables.get("payments_sum"),
        "customers": tables["customers"],
        "cohort": tables_final["cohort"],
    })

    seg = segmentation.run_segmentation(
        metrics.customers,
        method=settings.SEGMENTATION_METHOD,
        lapsing_days=metrics.thresholds["lapsing_days"],
        inactive_days=metrics.thresholds["inactive_days"],
    )

    orders_with_outcome = service_analysis.add_repeat_outcome(
        tables["order_values"], settings.SERVICE_REPEAT_WINDOW_DAYS, metrics.dates["as_of_date"]
    )
    # Both descriptors are attached once so every page can rely on them.
    orders_with_outcome = service_analysis.late_delivery_bins(orders_with_outcome)
    tables_final["orders_with_outcome"] = orders_with_outcome
    tables_final["score_bands"] = service_analysis.score_bands(orders_with_outcome)

    reviews = service_analysis.review_summary(tables["order_values"])
    delivery = service_analysis.delivery_summary(tables["order_values"])

    gc.collect()
    return AppContext(
        raw_meta=raw_meta, tables=tables_final, metrics=metrics,
        cohort=tables_final["cohort"], seg=seg, reviews=reviews,
        delivery=delivery, quality=quality, data_dir=str(data_dir),
        used_cache=used_cache,
    )


@st.cache_resource(show_spinner=False)
def load_context(data_dir_str: str, fingerprint: str) -> AppContext:
    """
    Cached pipeline execution.

    ``fingerprint`` is only used as a cache key; the actual work is done by
    ``_build``.
    """
    return _build(Path(data_dir_str), use_cache=True)


# Remembers the last fingerprint used per data directory, so that a working
# in-memory context keeps serving the session even if the CSV folder is cleaned
# up (the repo ships the original zip rather than 164 MB of extracted CSVs).
_LAST_FINGERPRINT: Dict[str, str] = {}


def get_context(data_dir: Optional[str] = None) -> AppContext:
    directory = Path(data_dir) if data_dir else settings.RAW_DATA_DIR
    key = str(directory)
    if not data_loader.available_files(directory) and key in _LAST_FINGERPRINT:
        return load_context(key, _LAST_FINGERPRINT[key])
    data_loader.maybe_extract_archive(directory)
    fingerprint = _fingerprint(directory)
    _LAST_FINGERPRINT[key] = fingerprint
    return load_context(key, fingerprint)


def data_available(data_dir: Optional[str] = None) -> bool:
    """
    True when the required tables can be read.

    If only the official zip is present, the CSVs are extracted first -- so a
    fresh clone works with nothing but the downloaded archive in data/raw/.
    """
    directory = Path(data_dir) if data_dir else settings.RAW_DATA_DIR
    found = data_loader.available_files(directory)
    if not all(t in found for t in settings.REQUIRED_TABLES):
        try:
            data_loader.maybe_extract_archive(directory)
        except Exception:      # a corrupt archive must surface as "missing data"
            return False
        found = data_loader.available_files(directory)
    return all(t in found for t in settings.REQUIRED_TABLES)
