"""
Central configuration for the Olist Customer Intelligence & Retention Lab.

Everything that anyone reading or re-running the analysis may want to change -- where the data
lives, which orders count as a real purchase, how segments are defined -- is
declared here rather than buried inside analysis code.

All thresholds have a short rationale so that the methodology report and the
Streamlit methodology page can stay honest about *why* a rule exists.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The dataset directory can be overridden with an environment variable so that
# the repository never has to contain the raw CSVs.
RAW_DATA_DIR = Path(os.environ.get("OLIST_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.environ.get("OLIST_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
REPORTS_DIR = PROJECT_ROOT / "reports"

# If the CSVs are missing but this archive is present, the loader will extract it.
RAW_ARCHIVE_NAME = "brazilian-ecommerce.zip"

# ---------------------------------------------------------------------------
# 2. Source tables
# ---------------------------------------------------------------------------

TABLE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Tables the core pipeline cannot run without. Geolocation and the category
# translation are optional enrichments.
REQUIRED_TABLES = ("orders", "customers", "order_items", "payments")

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

# ---------------------------------------------------------------------------
# 3. Order eligibility (see reports/methodology.md, section "Order eligibility")
# ---------------------------------------------------------------------------

# Statuses that represent a completed commercial transaction.
# 'delivered' is the default because it is the only status where we can observe
# that the customer actually received the goods. Canceled and unavailable orders
# are excluded; in-flight statuses (shipped/processing/...) belong to the last
# few weeks of the window and are excluded to keep the definition consistent
# across the whole timeline.
PRIMARY_ELIGIBLE_STATUSES = ("delivered",)

# Sensitivity variant used only in the methodology page / data-quality report.
BROAD_ELIGIBLE_STATUSES = ("delivered", "shipped", "invoiced", "processing", "approved")

EXCLUDED_STATUSES = ("canceled", "unavailable")

# A purchase timestamp before this date is treated as invalid for the analysis.
MIN_VALID_PURCHASE_DATE = "2016-01-01"

# ---------------------------------------------------------------------------
# 4. Monetary definitions (see methodology report)
# ---------------------------------------------------------------------------

# We deliberately keep four monetary measures apart:
#   merchandise_value = sum(order_items.price)                 <- KPI default
#   freight_value     = sum(order_items.freight_value)
#   item_total        = merchandise + freight
#   payment_value     = sum(order_payments.payment_value)      <- external check
MONETARY_KPI = "merchandise_value"

# ---------------------------------------------------------------------------
# 5. Retention / cohort settings
# ---------------------------------------------------------------------------

COHORT_PERIOD = "M"          # calendar-month cohorts
COHORT_RETENTION_METRIC = "purchased_in_month"   # activity in a specific month
MIN_COHORT_SIZE_FOR_CURVE = 100                   # avoid meaningless tiny curves

# ---------------------------------------------------------------------------
# 6. Recency / inactivity settings
# ---------------------------------------------------------------------------

# A customer whose first purchase is more recent than this has not had a fair
# chance to buy again, so they are labelled "Insufficient history" rather than
# being pushed into an inactivity bucket.
INSUFFICIENT_HISTORY_DAYS = 60

# Recency band boundaries. `None` means "derive from the data" -- see
# customer_metrics.derive_inactivity_thresholds(). Derivation rule:
#   lapsing  = max(90,  75th percentile of observed repeat-purchase gaps)
#   inactive = max(180, 90th percentile of observed repeat-purchase gaps)
LAPSING_DAYS = None
INACTIVE_DAYS = None

# Floor values used when deriving thresholds from data.
LAPSING_FLOOR_DAYS = 90
INACTIVE_FLOOR_DAYS = 180

# ---------------------------------------------------------------------------
# 7. RFM segmentation settings
# ---------------------------------------------------------------------------

SEGMENTATION_METHOD = "rule_based"     # "rule_based" | "kmeans"

# A customer counts as "high value" when their historical merchandise value is
# at or above this quantile of the whole customer base.
HIGH_VALUE_QUANTILE = 0.80

# "Repeat" means at least this many distinct eligible orders.
REPEAT_THRESHOLD = 2

RFM_QUANTILES = (0.2, 0.4, 0.6, 0.8)

SEGMENT_ORDER = [
    "High-Value Active Repeat",
    "Active Repeat",
    "Lapsing Repeat",
    "Recent One-Time Buyer",
    "Dormant One-Time Buyer",
    "Long Inactive",
    "New / Insufficient History",
]

# K-means extension settings
KMEANS_K_RANGE = range(3, 9)
KMEANS_RANDOM_STATE = 42
KMEANS_SAMPLE_SIZE = 10000   # rows used to estimate the silhouette score

# ---------------------------------------------------------------------------
# 8. Service-experience analysis settings
# ---------------------------------------------------------------------------

# Outcome window (days) after delivery in which we check for a subsequent order.
SERVICE_REPEAT_WINDOW_DAYS = 90
MIN_GRADE_GROUP_SIZE = 200   # suppress unstable comparisons below this

LATE_DELIVERY_BINS = [-10**6, -1, 0, 3, 7, 10**6]
LATE_DELIVERY_LABELS = ["Early", "On time", "1-3 days late", "4-7 days late", "8+ days late"]

# ---------------------------------------------------------------------------
# 9. Scenario simulator defaults (hypothetical inputs, clearly labelled)
# ---------------------------------------------------------------------------

SCENARIO_DEFAULTS = {
    "participation_rate": 0.15,
    "incremental_repeat_rate": 0.03,
    "average_order_value": None,   # None -> use the segment's observed AOV
    "contribution_margin": 0.30,
    "cost_per_contact": 0.50,
    "incentive_cost_per_redemption": 15.0,
}

SCENARIO_RANGE_MULTIPLIERS = {"Low": 0.5, "Base": 1.0, "High": 1.75}

# Currency of the source dataset. Values are nominal BRL as recorded in the file.
CURRENCY_CODE = "BRL"

# ---------------------------------------------------------------------------
# 10. Presentation
# ---------------------------------------------------------------------------

PALETTE = {
    "ink": "#0F172A",
    "slate": "#475569",
    "muted": "#94A3B8",
    "line": "#E2E8F0",
    "surface": "#FFFFFF",
    "background": "#F6F8FB",
    "primary": "#1D4ED8",
    "primary_soft": "#DBEAFE",
    "teal": "#0F766E",
    "amber": "#B45309",
    "rose": "#BE123C",
    "violet": "#6D28D9",
    "green": "#15803D",
}

SEGMENT_COLORS = {
    "High-Value Active Repeat": "#0F766E",
    "Active Repeat": "#1D4ED8",
    "Lapsing Repeat": "#B45309",
    "Recent One-Time Buyer": "#6D28D9",
    "Dormant One-Time Buyer": "#64748B",
    "Long Inactive": "#BE123C",
    "New / Insufficient History": "#94A3B8",
}

APP_TITLE = "Customer Intelligence & Retention Lab"
APP_TAGLINE = "From historical purchase behavior to evidence-based customer retention decisions."

DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
DATASET_LICENSE = "CC BY-NC-SA 4.0"
