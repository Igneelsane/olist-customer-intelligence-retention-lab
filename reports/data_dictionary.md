# Data dictionary

Two layers: the **raw source tables** as published by Olist, and the **analytical
tables** this project builds from them. Row counts are from the actual files used.

---

## 1. Raw source tables (as published)

### `olist_orders_dataset.csv` — 99,441 rows

| Column | Type | Notes |
|---|---|---|
| `order_id` | string | Primary key, unique |
| `customer_id` | string | **Order-scoped.** Changes when the same person orders again |
| `order_status` | string | delivered / shipped / canceled / unavailable / invoiced / processing / created / approved |
| `order_purchase_timestamp` | datetime | Order placed; basis for every time metric |
| `order_approved_at` | datetime | 160 nulls; not used in KPIs |
| `order_delivered_carrier_date` | datetime | 1,783 nulls; not used in KPIs |
| `order_delivered_customer_date` | datetime | 2,965 nulls; used for delivery timing only |
| `order_estimated_delivery_date` | datetime | Promise date; late = delivered after this |

### `olist_customers_dataset.csv` — 99,441 rows

| Column | Type | Notes |
|---|---|---|
| `customer_id` | string | One row per order (99,441 distinct) |
| `customer_unique_id` | string | **Person-level key (96,096 distinct)** — the identity used throughout |
| `customer_zip_code_prefix` | int | 5-digit prefix |
| `customer_city` | string | Lower-case, unnormalised |
| `customer_state` | string | 2-letter UF code |

### `olist_order_items_dataset.csv` — 112,650 rows

| Column | Type | Notes |
|---|---|---|
| `order_id` | string | Many rows per order |
| `order_item_id` | int | Sequence within order (1..n) |
| `product_id` | string | FK to products |
| `seller_id` | string | FK to sellers |
| `shipping_limit_date` | datetime | Seller hand-off deadline |
| `price` | float | **Merchandise value of the line** — the KPI measure |
| `freight_value` | float | Freight for the line; kept separate |

### `olist_order_payments_dataset.csv` — 103,886 rows

| Column | Type | Notes |
|---|---|---|
| `order_id` | string | Many rows per order (instalments/vouchers) |
| `payment_sequential` | int | Sequence within order |
| `payment_type` | string | credit_card, boleto, voucher, debit_card |
| `payment_installments` | float | Number of instalments |
| `payment_value` | float | Amount on that payment record |

### `olist_order_reviews_dataset.csv` — 99,224 rows

| Column | Type | Notes |
|---|---|---|
| `review_id` | string | 814 duplicated values in the source |
| `order_id` | string | Some orders carry 2–3 review rows |
| `review_score` | int | 1–5 |
| `review_comment_title` / `review_comment_message` | string | Mostly null (87,656 / 58,247 missing); **not used** — no text analysis, and no raw text is exposed or exported |
| `review_creation_date` | datetime | Survey sent |
| `review_answer_timestamp` | datetime | Survey answered; used to pick the latest review |

### `olist_products_dataset.csv` — 32,951 rows
`product_id` (key), `product_category_name`, plus physical attributes
(name/description length, photos, weight, dimensions) — only the category is used.

### `product_category_name_translation.csv` — 71 rows
`product_category_name` → `product_category_name_english`.

### `olist_sellers_dataset.csv` — 3,095 rows
`seller_id` (key), zip prefix, city, state. Used only for per-order seller counts.

### `olist_geolocation_dataset.csv` — 1,000,163 rows
Zip prefix → latitude/longitude. **Shipped with the dataset but not used**: no KPI
depends on it, and loading it triples memory for no analytical gain.

---

## 2. Analytical tables (built by `src/cleaning.py`)

### `order_values` — one row per eligible order (96,478)

| Column | Definition |
|---|---|
| `order_id` | Order key (unique) |
| `customer_id` | Order-scoped key (lineage only) |
| `customer_unique_id` | Person key — used for all customer metrics |
| `order_status` | Always `delivered` in the default configuration |
| `order_purchase_timestamp`, `order_date`, `order_month`, `order_year`, `dow` | Time features |
| `delivery_days` | Purchase → customer delivery, in days (null if not delivered) |
| `delay_days` | Delivered − estimated, in days (negative = early) |
| `is_late` | 1 if delivered after estimate, 0 if not, null if unknown |
| `n_items` | Item lines in the order |
| `n_distinct_products`, `n_sellers` | Variety within the order |
| `merchandise_value` | **Σ price** — the KPI measure |
| `freight_value` | Σ freight (never added to merchandise) |
| `payment_value` | Σ payment value (reconciliation only) |
| `order_total` | merchandise + freight (not used in KPI arithmetic) |
| `n_payments`, `max_installments`, `main_payment_type` | Payment descriptors |
| `review_score`, `has_comment` | Latest review per order |
| `main_category` | Highest-value item category in the order (English) |
| `customer_city`, `customer_state`, `customer_zip_code_prefix` | Geography |

### `customers` — one row per `customer_unique_id` (93,358)

| Column | Definition |
|---|---|
| `customer_unique_id` | Person key |
| `n_orders` / `frequency` | Distinct eligible orders |
| `merchandise_value` / `monetary_value` | Σ merchandise value |
| `freight_value`, `payment_value` | Σ freight / payments |
| `first_order_ts`, `last_order_ts`, `first_order_date`, `last_order_date` | Purchase bounds |
| `cohort_month` | Month of first eligible order |
| `items_purchased` | Σ item lines |
| `mean_review_score` | Mean of that customer's order reviews |
| `avg_order_value` | monetary ÷ n_orders |
| `customer_state`, `customer_city`, `customer_zip_code_prefix`, `n_customer_ids` | Modal geography per person |
| `recency_days` | Analysis date − last order date |
| `days_since_first_order` | Analysis date − first order date |
| `days_to_second_order` | First → second order gap (repeaters only) |
| `activity_status` | Recently active / Potentially lapsing / Long inactive / New or insufficient history |
| `r_score`, `f_score`, `m_score`, `rfm_score`, `rfm_cell` | RFM quantile scores (1–5) |
| `segment` | Rule-based RFM segment (or `Cluster k` in k-means mode) |
| `is_high_value`, `is_repeat` | Threshold flags |

### `order_items_sum`, `payments_sum` — one row per order
Intermediate aggregations kept separately so the reconciliation checks can be
recomputed independently of the joined table.

### `cohort` — long format, one row per (cohort_month, month_index)

| Column | Definition |
|---|---|
| `cohort_month` | First-purchase month |
| `month_index` | Months since cohort month (0 = cohort month) |
| `cohort_size` | Distinct customers in the cohort |
| `n_active` | Distinct cohort customers with an order in that specific month |
| `retention_rate` | `n_active / cohort_size` |
| `max_observable_index` | Last month index observable before the dataset ends |
| `is_observable` | Always true in the stored table (unobservable months are absent) |

### `orders_with_outcome` — order grain + service outcome
Adds `censored` (the 90-day window extends past the dataset end),
`days_to_next_order`, `repeat_within_window` (null when censored), `late_bin`,
`score_band`.

---

## 3. Privacy notes

- No customer name, email, phone or address exists in the source beyond a zip-code
  prefix, city and state.
- The application displays customers only through a stable pseudonymous label
  (`sha1(customer_unique_id)[:10]` prefixed `C-`).
- Exports contain aggregates, or pseudonymous rows for one segment (max 5,000). Raw
  identifiers are never exported.
- Review comment text is loaded only as a boolean `has_comment` flag; the text itself
  is never displayed or exported.
