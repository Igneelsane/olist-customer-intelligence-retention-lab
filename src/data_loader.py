"""
Dataset discovery and loading.

Responsibilities
----------------
1. Find the Olist CSV files (project data/raw, or $OLIST_DATA_DIR).
2. Extract the official Kaggle zip archive if the CSVs are absent.
3. Load each table with explicit dtypes and parsed dates.
4. Produce a clear, actionable error when files are missing -- the application
   must never silently invent data.

Nothing in this module computes business metrics; it only reads data.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from config import settings


# Only the columns the analysis actually needs are read. This keeps memory
# bounded on small machines and avoids loading long free-text review comments
# (which are never displayed, exported or analysed as text).
USECOLS = {
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
               "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                  "customer_city", "customer_state"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id",
                    "shipping_limit_date", "price", "freight_value"],
    "payments": ["order_id", "payment_sequential", "payment_type",
                 "payment_installments", "payment_value"],
    "reviews": ["review_id", "order_id", "review_score", "review_comment_message",
                "review_creation_date", "review_answer_timestamp"],
    "products": ["product_id", "product_category_name"],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "category_translation": ["product_category_name", "product_category_name_english"],
}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the Olist source tables cannot be located."""

    def __init__(self, message: str, missing: Iterable[str] | None = None):
        self.missing = list(missing or [])
        super().__init__(message)


@dataclass
class DatasetInfo:
    """Where the data came from -- surfaced in the UI and QA report."""

    data_dir: Path
    files_found: Dict[str, Path]
    missing_tables: List[str]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def available_files(data_dir: Path | str | None = None) -> Dict[str, Path]:
    """Map table name -> existing CSV path (only files that actually exist)."""
    directory = Path(data_dir) if data_dir is not None else settings.RAW_DATA_DIR
    found: Dict[str, Path] = {}
    for name, filename in settings.TABLE_FILES.items():
        candidate = directory / filename
        if candidate.exists():
            found[name] = candidate
    return found


def maybe_extract_archive(data_dir: Path | str | None = None) -> bool:
    """
    Extract the official Kaggle zip if the CSVs are not already unpacked.

    Returns True when an extraction happened. The archive is the exact file the
    Kaggle download button produces; it is kept compressed in the repo so that
    the raw CSVs never have to be committed to Git.
    """
    directory = Path(data_dir) if data_dir is not None else settings.RAW_DATA_DIR
    archive = directory / settings.RAW_ARCHIVE_NAME
    if not archive.exists():
        return False
    needed = [f for n, f in settings.TABLE_FILES.items() if n in settings.REQUIRED_TABLES]
    if all((directory / f).exists() for f in needed):
        return False
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".csv"):
                # Flatten any folder prefix inside the archive.
                target = directory / Path(member).name
                if not target.exists():
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
    return True


def describe_dataset(data_dir: Path | str | None = None) -> DatasetInfo:
    directory = Path(data_dir) if data_dir is not None else settings.RAW_DATA_DIR
    found = available_files(directory)
    missing = [t for t in settings.REQUIRED_TABLES if t not in found]
    return DatasetInfo(data_dir=directory, files_found=found, missing_tables=missing)


def missing_data_message(info: DatasetInfo) -> str:
    """Human-readable instructions shown in the app when data is absent."""
    expected = "\n".join(f"  - {settings.TABLE_FILES[t]}" for t in settings.TABLE_FILES)
    return f"""### Dataset files not found

The application looks for the Brazilian E-Commerce Public Dataset by Olist in:

    {info.data_dir}

Missing required table(s): **{', '.join(info.missing_tables)}**

**How to get the data (free, no paid API):**

1. Open {settings.DATASET_SOURCE_URL}
2. Sign in to Kaggle (free account) and accept the dataset terms.
3. Click *Download* to get `archive.zip` (44 MB).
4. Either unzip it into `{info.data_dir}` or rename it to
   `{settings.RAW_ARCHIVE_NAME}` and drop it in that folder -- the app will
   extract it automatically.
5. Alternatively point the app at any folder containing the CSVs:

       export OLIST_DATA_DIR=/absolute/path/to/olist/csvs

Expected files:
{expected}

The application will not substitute synthetic records for the real dataset.
"""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path, table: str) -> pd.DataFrame:
    date_cols = settings.DATE_COLUMNS.get(table, [])
    parse = {c: "datetime64[ns]" for c in date_cols}
    usecols = USECOLS.get(table)
    df = pd.read_csv(path, dtype=str, keep_default_na=True, usecols=usecols)
    for col in parse:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    return df


def _coerce_types(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Explicit, documented type conversions (everything is read as string)."""
    numeric_map = {
        "order_items": {"order_item_id": "int64", "price": "float64", "freight_value": "float64"},
        "payments": {"payment_sequential": "int64", "payment_installments": "float64",
                     "payment_value": "float64"},
        "reviews": {"review_score": "Int64"},
        "customers": {"customer_zip_code_prefix": "Int64"},
        "products": {
            "product_name_lenght": "Int64", "product_description_lenght": "Int64",
            "product_photos_qty": "Int64", "product_weight_g": "Float64",
            "product_length_cm": "Float64", "product_height_cm": "Float64",
            "product_width_cm": "Float64",
        },
        "sellers": {"seller_zip_code_prefix": "Int64"},
        "geolocation": {"geolocation_zip_code_prefix": "Int64",
                        "geolocation_lat": "float64", "geolocation_lng": "float64"},
    }
    for col, dtype in numeric_map.get(table, {}).items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


def load_table(table: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load one source table by logical name."""
    if table not in settings.TABLE_FILES:
        raise KeyError(f"Unknown table '{table}'. Known: {sorted(settings.TABLE_FILES)}")
    directory = Path(data_dir) if data_dir is not None else settings.RAW_DATA_DIR
    path = directory / settings.TABLE_FILES[table]
    if not path.exists():
        raise DatasetNotFoundError(
            f"Missing '{settings.TABLE_FILES[table]}' in {directory}.",
            missing=[settings.TABLE_FILES[table]],
        )
    return _coerce_types(_load_csv(path, table), table)


def load_raw_tables(
    data_dir: Path | str | None = None,
    tables: Iterable[str] | None = None,
    auto_extract: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load several tables at once.

    Raises DatasetNotFoundError (with Kaggle instructions available through
    missing_data_message) when a required table is absent.
    """
    directory = Path(data_dir) if data_dir is not None else settings.RAW_DATA_DIR
    if auto_extract:
        maybe_extract_archive(directory)

    wanted = list(tables) if tables is not None else list(settings.TABLE_FILES)
    found = available_files(directory)
    missing_required = [t for t in settings.REQUIRED_TABLES if t not in found]
    if missing_required:
        info = describe_dataset(directory)
        raise DatasetNotFoundError(missing_data_message(info), missing=missing_required)

    out: Dict[str, pd.DataFrame] = {}
    for name in wanted:
        if name in found:
            out[name] = load_table(name, directory)
    return out


def dataset_fingerprint(data_dir: Path | str | None = None) -> Dict[str, object]:
    """
    Cheap identity of the dataset (row counts + byte sizes).

    Used as a cache key so the Streamlit app does not re-read 100k+ rows on
    every interaction, and displayed on the methodology page for traceability.
    """
    directory = Path(data_dir) if data_dir is not None else settings.RAW_DATA_DIR
    found = available_files(directory)
    return {
        "data_dir": str(directory),
        "files": {
            name: {"path": str(p), "bytes": p.stat().st_size}
            for name, p in sorted(found.items())
        },
    }
