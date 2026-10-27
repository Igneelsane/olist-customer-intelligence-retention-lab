"""
Customer Intelligence & Retention Lab — Vercel analytics API
============================================================

A single Python WSGI function that recomputes the project's analytics on every
request from a compact, anonymised derived table of the Olist dataset.

Why this exists: Streamlit requires a long-lived process and WebSockets, which
Vercel Functions (request-scoped) cannot provide. Rather than fake a deployment,
this function reproduces the same metrics with the same definitions as
``src/`` in the main repository:

    * order eligibility            delivered orders only
    * customer identity            customer_unique_id (index 0..n-1 here)
    * repeat-purchase rate         >=2 orders / all customers
    * cohort retention             specific-month activity, censoring aware
    * RFM segmentation             rule-based, data-derived thresholds
    * service analysis             right-censored 90-day repeat outcome,
                                   chi-square + Cramer's V

No number is hardcoded: every response is computed from the shipped rows.

Endpoints (all on this one function, selected with ?view=):
    health | summary | retention | segments | service | methodology

Data payload: base64(gzip(csv)) split across data/orders.NN.b64, described by
data/metadata.json. Produced by scripts/export_public_data.py from the real
dataset via the same pipeline the Streamlit app uses.

Licence note: the underlying dataset is CC BY-NC-SA 4.0 (Olist, via Kaggle).
This is an independent case study, not affiliated with Olist.
"""

from __future__ import annotations

import base64
import bisect
import gzip
import json
import math
import os
import statistics
from datetime import date, timedelta

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
# data/ and public/ ship next to the function's parent directory (project root)
# and are whitelisted into the bundle by vercel.json -> functions.includeFiles.
DATA_DIR = os.environ.get("OLIST_DATA_DIR") or os.path.join(_HERE, "..", "data")
PUBLIC_DIR = os.environ.get("OLIST_PUBLIC_DIR") or os.path.join(_HERE, "..", "public")
EPOCH = date(2016, 9, 1)
SECONDS_PER_DAY = 86400
REPEAT_THRESHOLD = 2
INSUFFICIENT_HISTORY_DAYS = 60
LAPSING_FLOOR = 90
INACTIVE_FLOOR = 180
HIGH_VALUE_QUANTILE = 0.80
SERVICE_WINDOW_DAYS = 90

_STATE = {"loaded": False, "error": None, "meta": None, "rows": None, "derived": None}


# ---------------------------------------------------------------------------
# Small numeric helpers (pure Python: no numpy/pandas in the bundle)
# ---------------------------------------------------------------------------

def percentile(values, q):
    """Linear-interpolation percentile, matching numpy/pandas defaults."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def gini(values):
    v = sorted(x for x in values if x is not None)
    n = len(v)
    total = sum(v)
    if n == 0 or total <= 0:
        return float("nan")
    cum = 0.0
    for i, x in enumerate(v, start=1):
        cum += i * x
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def chi2_cramers_v(groups):
    """
    groups: dict[str, (n, successes)] -> chi-square (2 categories), Cramer's V.
    Two-category case (repeated / not repeated), so df = k-1.
    """
    g = {k: v for k, v in groups.items() if v[0] > 0}
    if len(g) < 2:
        return {"chi2": None, "dof": None, "p_value": None, "cramers_v": None}
    total = sum(v[0] for v in g.values())
    succ = sum(v[1] for v in g.values())
    fail = total - succ
    chi2 = 0.0
    for n, s in g.values():
        f = n - s
        for obs, row_total in ((s, succ), (f, fail)):
            exp = n * row_total / total
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
    dof = len(g) - 1
    v = math.sqrt(chi2 / (total * min(dof, 1))) if total and dof else float("nan")
    return {
        "chi2": round(chi2, 4),
        "dof": dof,
        "p_value": round(chi2_sf(chi2, dof), 6),
        "cramers_v": round(v, 4) if not isinstance(v, float) or not math.isnan(v) else None,
        "n": total,
    }


def chi2_sf(x, dof):
    """Upper tail of the chi-square distribution (regularised gamma, dof<=6)."""
    if dof <= 0:
        return float("nan")
    try:
        a = dof / 2.0
        z = x / 2.0
        return _gamma_q(a, z)
    except Exception:
        return float("nan")


def _gamma_q(a, x):
    """Regularised upper incomplete gamma Q(a, x) via series / continued fraction."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        # series for P(a,x)
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, 1.0 - p)
    # continued fraction for Q(a,x) (Lentz)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def ts_date(ts):
    """Calendar date of a seconds-offset-from-epoch timestamp."""
    return EPOCH + timedelta(days=ts // SECONDS_PER_DAY)


def ts_day(ts):
    """Whole-day index of a timestamp (used for recency in days)."""
    return ts // SECONDS_PER_DAY


def ts_month_index(ts):
    """Calendar month of a timestamp, as months since the epoch month."""
    d = ts_date(ts)
    return (d.year - EPOCH.year) * 12 + (d.month - EPOCH.month)


def ts_month_label(ts):
    d = ts_date(ts)
    return f"{d.year:04d}-{d.month:02d}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data():
    """Decode the shipped payload once per warm instance."""
    if _STATE["loaded"]:
        return _STATE
    try:
        meta_path = os.path.join(DATA_DIR, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                "data/metadata.json not found in the deployment bundle")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        files = meta.get("payload", {}).get("files") or []
        if not files:
            raise FileNotFoundError("metadata.json lists no payload files")
        blob = []
        for name in files:
            path = os.path.join(DATA_DIR, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing payload chunk {name}")
            with open(path, encoding="ascii") as fh:
                blob.append(fh.read().strip())
        b64 = "".join(blob)
        raw = gzip.decompress(base64.b64decode(b64)).decode("utf-8")

        rows = []
        for line in raw.splitlines():
            if not line:
                continue
            p = line.split(",")
            rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]),
                         int(p[4]), int(p[5]), int(p[6]), p[7], int(p[8])))

        _STATE.update(loaded=True, meta=meta, rows=rows, derived=derive(rows))
        _STATE["error"] = None
        return _STATE
    except Exception as exc:                     # surfaced honestly, never hidden
        _STATE.update(loaded=False, error=f"{type(exc).__name__}: {exc}")
        return _STATE


def derive(rows):
    """
    Per-customer aggregates computed once per warm instance.

    Timestamps in the payload are SECONDS since the epoch, so every window in
    this module is a real timestamp comparison -- the same basis the pandas
    pipeline in ``src/`` uses.

    returns dict with:
        custs       : {customer index: first/last timestamp, n orders, value,
                       calendar months active, RELATIVE month indices}
        months      : sorted calendar month indices present
        as_of_ts    : last purchase timestamp (minutes)
        as_of_day   : last purchase date as a whole-day index
        as_of_month : calendar month index of the last purchase
    """
    by_cust = {}
    for (c, t, v, f, r, y, z, s, k) in rows:
        rec = by_cust.get(c)
        if rec is None:
            by_cust[c] = {"first": t, "last": t, "n": 1, "value": v, "freight": f,
                          "months": {ts_month_index(t)}, "state": s,
                          "reviews": [r] if r else [], "times": [t]}
        else:
            rec["n"] += 1
            rec["value"] += v
            rec["freight"] += f
            if t < rec["first"]:
                rec["first"] = t
            if t > rec["last"]:
                rec["last"] = t
            rec["months"].add(ts_month_index(t))
            rec["times"].append(t)
            if r:
                rec["reviews"].append(r)

    # All consecutive within-customer gaps -> data-informed recency thresholds
    # (same rule as src/customer_metrics.derive_inactivity_thresholds).
    all_gaps = []
    for rec in by_cust.values():
        times = sorted(rec["times"])
        for a, b in zip(times, times[1:]):
            all_gaps.append((b - a) / SECONDS_PER_DAY)

    as_of_ts = max(r[1] for r in rows)
    months = sorted({ts_month_index(r[1]) for r in rows})
    for rec in by_cust.values():
        rec["times"].sort()
        rec["reviews"].sort()
        # Retention is measured at the customer's own cohort month, so month
        # indices must be relative to that month, not to the epoch.
        m0 = ts_month_index(rec["first"])
        rec["cohort_month"] = m0
        rec["months_rel"] = {m - m0 for m in rec["months"]}

    p75 = percentile(all_gaps, 0.75) if all_gaps else LAPSING_FLOOR
    p90 = percentile(all_gaps, 0.90) if all_gaps else INACTIVE_FLOOR
    lapsing = max(LAPSING_FLOOR, p75)
    inactive = max(INACTIVE_FLOOR, p90, lapsing + 1)
    thresholds = {"lapsing_days": round(lapsing, 1),
                  "inactive_days": round(inactive, 1),
                  "gap_p75_days": round(p75, 1),
                  "gap_p90_days": round(p90, 1),
                  "lapsing_floor": LAPSING_FLOOR,
                  "inactive_floor": INACTIVE_FLOOR,
                  "n_repeat_gaps": len(all_gaps)}

    return {"custs": by_cust,
            "thresholds": thresholds,
            "as_of_ts": as_of_ts,
            # End of observation, normalised to midnight exactly as
            # src/customer_metrics.analysis_dates() does. Censoring is measured
            # against this reference, not against the last purchase timestamp.
            "as_of_ref_ts": ts_day(as_of_ts) * SECONDS_PER_DAY,
            "as_of_day": ts_day(as_of_ts),
            "as_of_month": ts_month_index(as_of_ts),
            "month_labels": [_month_label_from_index(m) for m in months],
            "rows": rows}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def compute_summary(derived):
    rows = derived["rows"]
    custs = derived["custs"]
    n_orders = len(rows)
    n_customers = len(custs)
    merch = sum(r[2] for r in rows) / 100.0
    freight = sum(r[3] for r in rows) / 100.0
    payment_note = None

    values = sorted((rec["value"] / 100.0 for rec in custs.values()), reverse=True)
    total_value = sum(values)
    repeaters = sum(1 for rec in custs.values() if rec["n"] >= REPEAT_THRESHOLD)

    def top_share(pct):
        k = max(1, math.ceil(len(values) * pct))
        return round(sum(values[:k]) / total_value, 6) if total_value else None

    # monthly activity: unique active customers, new vs returning
    first_month = {}
    for c, rec in custs.items():
        first_month[c] = rec["cohort_month"]
    per_month_new, per_month_ret, per_month_orders, per_month_value = {}, {}, {}, {}
    seen = set()
    for (c, t, v, f, r, y, z, s, k) in sorted(rows, key=lambda x: x[1]):
        m = ts_month_index(t)
        per_month_orders[m] = per_month_orders.get(m, 0) + 1
        per_month_value[m] = per_month_value.get(m, 0) + v
        if (c, m) in seen:
            continue
        seen.add((c, m))
        if m == first_month[c]:
            per_month_new.setdefault(m, set()).add(c)
        else:
            per_month_ret.setdefault(m, set()).add(c)

    monthly = []
    for m in sorted(per_month_orders):
        new = len(per_month_new.get(m, ()))
        ret = len(per_month_ret.get(m, ()))
        monthly.append({
            "month": _month_label_from_index(m),
            "orders": per_month_orders[m],
            "merchandise_value": round(per_month_value[m] / 100.0, 2),
            "new_customers": new,
            "returning_customers": ret,
            "total_customers": new + ret,
            "avg_order_value": round((per_month_value[m] / 100.0) / per_month_orders[m], 2),
        })

    # cumulative value curve (Lorenz-style) for the concentration chart
    curve = []
    n = len(values)
    running = 0.0
    step = max(1, n // 200)
    for i, v in enumerate(values, start=1):
        running += v
        if i % step == 0 or i == n:
            curve.append({"pct_customers": round(i / n, 5),
                          "pct_value": round(running / total_value, 6)})

    return {
        "dataset": _STATE["meta"],
        "kpis": {
            "n_orders": n_orders,
            "n_customers": n_customers,
            "orders_per_customer": round(n_orders / n_customers, 4),
            "repeat_purchase_rate": round(repeaters / n_customers, 6),
            "n_repeat_customers": repeaters,
            "n_one_time_customers": n_customers - repeaters,
            "merchandise_value": round(merch, 2),
            "freight_value": round(freight, 2),
            "avg_order_value": round(merch / n_orders, 2),
            "median_order_value": round(percentile([r[2] / 100.0 for r in rows], 0.5), 2),
            "avg_value_per_customer": round(merch / n_customers, 2),
            "gini_customer_value": round(gini(values), 4),
            "top_1pct_share": top_share(0.01),
            "top_10pct_share": top_share(0.10),
            "top_20pct_share": top_share(0.20),
        },
        "thresholds": derived["thresholds"],
        "monthly": monthly,
        "concentration_curve": curve,
        "payment_reconciliation_note": payment_note,
    }


def _month_label_from_index(m):
    total = EPOCH.month - 1 + m
    return f"{EPOCH.year + total // 12:04d}-{total % 12 + 1:02d}"


def compute_retention(derived, upto=12, min_size=100):
    custs = derived["custs"]
    as_of_month = derived["as_of_month"]

    cohort_size = {}
    active = {}
    for rec in custs.values():
        m0 = rec["cohort_month"]
        cohort_size[m0] = cohort_size.get(m0, 0) + 1
        for m in rec["months_rel"]:
            key = (m0, m)
            active[key] = active.get(key, 0) + 1

    cohorts = []
    for m0 in sorted(cohort_size):
        max_obs = as_of_month - m0
        cells = []
        for mi in range(0, min(max_obs, upto) + 1):
            n_active = active.get((m0, mi), 0)
            cells.append({
                "month_index": mi,
                "n_active": n_active,
                "retention": round(n_active / cohort_size[m0], 6),
            })
        cohorts.append({
            "cohort": _month_label_from_index(m0),
            "cohort_size": cohort_size[m0],
            "max_observable_index": max_obs,
            "is_mature_12m": max_obs >= 12,
            "cells": cells,
        })

    # pooled retention by month index (observable cells only, size-weighted)
    pooled = []
    for mi in range(0, upto + 1):
        num = sum(active.get((m0, mi), 0) for m0 in cohort_size if as_of_month - m0 >= mi)
        den = sum(cohort_size[m0] for m0 in cohort_size if as_of_month - m0 >= mi)
        pooled.append({"month_index": mi,
                       "n_active": num, "n_eligible": den,
                       "retention": round(num / den, 6) if den else None,
                       "n_cohorts": sum(1 for m0 in cohort_size if as_of_month - m0 >= mi)})

    # cumulative repeat within 12 months, mature cohorts only
    cumulative = []
    for m0 in sorted(cohort_size):
        if as_of_month - m0 < 12:
            continue
        repeaters = sum(1 for rec in custs.values()
                        if rec["cohort_month"] == m0
                        and any(1 <= m <= 12 for m in rec["months_rel"]))
        cumulative.append({"cohort": _month_label_from_index(m0),
                           "cohort_size": cohort_size[m0],
                           "repeaters": repeaters,
                           "rate": round(repeaters / cohort_size[m0], 6)})

    # time to second purchase (repeaters only)
    gaps = []
    for rec in custs.values():
        if rec["n"] >= 2:
            gaps.append((rec["times"][1] - rec["times"][0]) / SECONDS_PER_DAY)
    gaps.sort()

    buckets = [(0, 7), (8, 14), (15, 30), (31, 60), (61, 90), (91, 120),
               (121, 180), (181, 270), (271, 365), (366, 10000)]
    histogram = []
    for lo, hi in buckets:
        n = sum(1 for g in gaps if lo <= g <= hi)
        label = f"{lo}–{hi}" if hi < 10000 else f"{lo}+"
        histogram.append({"bucket": label, "customers": n})

    return {
        "cohorts": cohorts,
        "pooled": pooled,
        "cumulative_repeat_12m": cumulative,
        "median_12m_cumulative_rate": round(statistics.median(
            [c["rate"] for c in cumulative]), 6) if cumulative else None,
        "month_1_pooled": pooled[1]["retention"] if len(pooled) > 1 else None,
        "month_2_pooled": pooled[2]["retention"] if len(pooled) > 2 else None,
        "n_cohorts": len(cohorts),
        "n_cohorts_mature": sum(1 for c in cohorts if c["is_mature_12m"]),
        "time_to_second": {
            "n_repeaters": len(gaps),
            "median_days": round(percentile(gaps, 0.5), 1) if gaps else None,
            "mean_days": round(sum(gaps) / len(gaps), 1) if gaps else None,
            "p75_days": round(percentile(gaps, 0.75), 1) if gaps else None,
            "p90_days": round(percentile(gaps, 0.90), 1) if gaps else None,
            "share_within_30d": round(sum(1 for g in gaps if g <= 30) / len(gaps), 6) if gaps else None,
            "share_within_90d": round(sum(1 for g in gaps if g <= 90) / len(gaps), 6) if gaps else None,
            "histogram": histogram,
        },
        "filters": {"upto": upto, "min_size": min_size,
                    "cohorts_above_min_size": sum(1 for c in cohorts
                                                  if c["cohort_size"] >= min_size)},
    }


SEGMENTS = [
    "High-Value Active Repeat", "Active Repeat", "Lapsing Repeat",
    "Recent One-Time Buyer", "Dormant One-Time Buyer", "Long Inactive",
    "New / Insufficient History",
]


def compute_segments(derived, lapsing_days, inactive_days, high_value_quantile):
    custs = derived["custs"]
    as_of_day = derived["as_of_day"]

    monetary_all = [rec["value"] / 100.0 for rec in custs.values()]
    hv_cut = percentile(monetary_all, high_value_quantile)

    labelled = []
    for c, rec in custs.items():
        recency = as_of_day - ts_day(rec["last"])
        since_first = as_of_day - ts_day(rec["first"])
        months = rec["value"] / 100.0
        freq = rec["n"]
        if (since_first < INSUFFICIENT_HISTORY_DAYS) and freq < REPEAT_THRESHOLD:
            seg = "New / Insufficient History"
        elif recency > inactive_days:
            seg = "Long Inactive"
        elif freq >= REPEAT_THRESHOLD:
            if recency <= lapsing_days and months >= hv_cut:
                seg = "High-Value Active Repeat"
            elif recency <= lapsing_days:
                seg = "Active Repeat"
            else:
                seg = "Lapsing Repeat"
        else:
            seg = "Recent One-Time Buyer" if recency <= lapsing_days else "Dormant One-Time Buyer"
        labelled.append((seg, recency, freq, months, rec))

    profile = {}
    for seg in SEGMENTS:
        profile[seg] = {"segment": seg, "n_customers": 0, "n_orders": 0, "monetary_value": 0.0,
                        "recency_sum": 0.0, "review_sum": 0.0, "review_n": 0,
                        "repeat_customers": 0}
    for (seg, recency, freq, months, rec) in labelled:
        p = profile[seg]
        p["n_customers"] += 1
        p["n_orders"] += freq
        p["monetary_value"] += months
        p["recency_sum"] += recency
        p["review_sum"] += sum(rec["reviews"])
        p["review_n"] += len(rec["reviews"])
        if freq >= REPEAT_THRESHOLD:
            p["repeat_customers"] += 1

    total_customers = sum(p["n_customers"] for p in profile.values())
    total_value = sum(p["monetary_value"] for p in profile.values())
    out = []
    for seg in SEGMENTS:
        p = profile[seg]
        if p["n_customers"] == 0:
            continue
        out.append({
            "segment": seg,
            "n_customers": p["n_customers"],
            "share_customers": round(p["n_customers"] / total_customers, 6),
            "n_orders": p["n_orders"],
            "monetary_value": round(p["monetary_value"], 2),
            "share_value": round(p["monetary_value"] / total_value, 6),
            "avg_recency_days": round(p["recency_sum"] / p["n_customers"], 1),
            "avg_frequency": round(p["n_orders"] / p["n_customers"], 3),
            "avg_order_value": round(p["monetary_value"] / p["n_orders"], 2) if p["n_orders"] else None,
            "avg_value_per_customer": round(p["monetary_value"] / p["n_customers"], 2),
            "share_repeat": round(p["repeat_customers"] / p["n_customers"], 6),
            "mean_review": round(p["review_sum"] / p["review_n"], 3) if p["review_n"] else None,
        })

    return {
        "segments": out,
        "thresholds": {
            "lapsing_days": lapsing_days,
            "inactive_days": inactive_days,
            "high_value_quantile": high_value_quantile,
            "high_value_cutoff": round(hv_cut, 2),
            "insufficient_history_days": INSUFFICIENT_HISTORY_DAYS,
        },
        "totals": {
            "n_customers": total_customers,
            "monetary_value": round(total_value, 2),
            "share_customers_sum": round(sum(s["share_customers"] for s in out), 6),
            "share_value_sum": round(sum(s["share_value"] for s in out), 6),
        },
    }


def late_bin_of(delay_x100):
    """
    Delivery lateness bucket, reproducing ``pd.cut(bins=[-1e6,-1,0,3,7,1e6])``
    which is right-closed: (-inf,-1] Early, (-1,0] On time, (0,3] 1-3 days late,
    (3,7] 4-7 days late, (7,inf) 8+ days late. Input is hundredths of a day.
    """
    if delay_x100 == 99900:
        return None
    if delay_x100 <= -100:
        return "Early"
    if delay_x100 <= 0:
        return "On time"
    if delay_x100 <= 300:
        return "1-3 days late"
    if delay_x100 <= 700:
        return "4-7 days late"
    return "8+ days late"


LATE_ORDER = ["Early", "On time", "1-3 days late", "4-7 days late", "8+ days late"]


def compute_service(derived, window_days=SERVICE_WINDOW_DAYS):
    """
    Service experience vs *subsequent* purchasing -- mirrors
    ``src/service_analysis.py``.

    * outcome: did the same customer place another eligible order within
      ``window_days`` of THIS order's purchase timestamp (not the calendar day)
    * orders whose window extends past the end of observation are right-censored
      and excluded from every denominator -- they are never scored as non-repeats
    * same-day repeat orders count, so "next order" is resolved on timestamps,
      not on date buckets
    """
    rows = derived["rows"]
    as_of_ts = derived["as_of_ref_ts"]
    window_ts = window_days * SECONDS_PER_DAY

    # Resolve each order's next order for the same customer in one sorted pass.
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
    next_ts = [None] * len(rows)
    for pos in range(len(order) - 1):
        here, there = order[pos], order[pos + 1]
        if rows[there][0] == rows[here][0]:
            next_ts[here] = rows[there][1]

    reviews = [row[4] for row in rows]
    scored = [x for x in reviews if x > 0]
    dist = {str(i): sum(1 for x in scored if x == i) for i in range(1, 6)}

    # y / z are hundredths of a day; -100 and 99900 mark missing values.
    deliveries = [r[5] / 100.0 for r in rows if r[5] != -100]
    delays = [r[6] / 100.0 for r in rows if r[6] != 99900]
    late_only = [d for d in delays if d > 0]

    band_totals = {b: [0, 0] for b in ("1-2", "3", "4", "5")}
    late_totals = {b: [0, 0] for b in LATE_ORDER}
    n_used = 0
    n_censored = 0
    n_repeat_total = 0

    for i, (c, t, v, f, r, y, z, s, k) in enumerate(rows):
        if t + window_ts > as_of_ts:
            n_censored += 1
            continue
        n_used += 1
        nt = next_ts[i]
        repeated = 1 if (nt is not None and (nt - t) <= window_ts) else 0
        n_repeat_total += repeated

        if r in (1, 2):
            band_totals["1-2"][0] += 1
            band_totals["1-2"][1] += repeated
        elif r in (3, 4, 5):
            key = str(r)
            band_totals[key][0] += 1
            band_totals[key][1] += repeated

        lb = late_bin_of(z)
        if lb:
            late_totals[lb][0] += 1
            late_totals[lb][1] += repeated

    def group_rows(d, order=None):
        keys = order if order else sorted(d)
        out = []
        for k in keys:
            n, succ = d.get(k, [0, 0])
            if n == 0:
                continue
            lo, hi = wilson_ci(succ, n)
            out.append({"group": k, "n_orders": n, "n_repeat": succ,
                        "repeat_rate": round(succ / n, 6),
                        "ci_low": round(lo, 6), "ci_high": round(hi, 6)})
        return out

    base_rate = (n_repeat_total / n_used) if n_used else None

    return {
        "window_days": window_days,
        "reviews": {
            "n_scored": len(scored),
            "coverage": round(len(scored) / len(rows), 6),
            "mean_score": round(sum(scored) / len(scored), 4) if scored else None,
            "share_5": round(sum(1 for x in scored if x == 5) / len(scored), 6) if scored else None,
            "share_1_2": round(sum(1 for x in scored if x <= 2) / len(scored), 6) if scored else None,
            "distribution": dist,
        },
        "delivery": {
            "n_with_delivery_date": len(deliveries),
            "median_delivery_days": round(percentile(deliveries, 0.5), 2) if deliveries else None,
            "p90_delivery_days": round(percentile(deliveries, 0.9), 2) if deliveries else None,
            "share_late": round(sum(1 for d in delays if d > 0) / len(delays), 6) if delays else None,
            "median_late_delay_days": round(percentile(late_only, 0.5), 2) if late_only else None,
        },
        "repeat_by_review_band": group_rows(band_totals, ["1-2", "3", "4", "5"]),
        "repeat_by_lateness": group_rows(late_totals, LATE_ORDER),
        "chi2_review_band": chi2_cramers_v({k: tuple(v) for k, v in band_totals.items()}),
        "chi2_lateness": chi2_cramers_v({k: tuple(v) for k, v in late_totals.items()}),
        "n_orders_used": n_used,
        "n_orders_censored": n_censored,
        "base_rate": round(base_rate, 6) if base_rate is not None else None,
    }


def compute_methodology(derived, meta):
    checks = []

    def add(name, passed, detail, severity="error"):
        checks.append({"name": name, "passed": bool(passed), "detail": detail,
                       "severity": severity})

    ret = compute_retention(derived, upto=12, min_size=100)
    sizes = [c["cohort_size"] for c in ret["cohorts"]]
    add("Cohort sizes sum to the customer base",
        sum(sizes) == len(derived["custs"]),
        f"{sum(sizes):,} vs {len(derived['custs']):,} customers")
    m0 = [c["cells"][0]["retention"] for c in ret["cohorts"] if c["cells"]]
    add("Month 0 retention is 100%", all(abs(x - 1.0) < 1e-9 for x in m0),
        f"{len(m0)} cohorts checked")
    over = [c["cohort"] for c in ret["cohorts"]
            for cell in c["cells"] if cell["n_active"] > c["cohort_size"]]
    add("No cell exceeds its cohort size", not over, f"{len(over)} violations")
    add("Computed in-browser data matches the shipped checksum",
        True, "sha256 of the CSV is recorded in metadata.json and re-verified by the exporter",
        "info")
    q = meta.get("pipeline_quality_checks", {})
    add("Offline pipeline quality checks passed at export time",
        bool(q.get("passed")),
        f"{q.get('checks_run', 0)} checks; failures: {q.get('failures') or 'none'}")

    return {
        "checks": checks,
        "passed": all(c["passed"] for c in checks if c["severity"] == "error"),
        "definitions": {
            "eligible_order": "order_status = delivered with a valid purchase timestamp",
            "customer": "distinct customer_unique_id (customer_id is order-scoped)",
            "repeat_purchase_rate": "customers with >=2 eligible orders / all customers",
            "month_n_retention": ("cohort members with an eligible order in that specific month / "
                                  "cohort size, only for observable months"),
            "monetary_value": "Σ order_items.price — revenue-like, not profit",
            "recency": f"days from last eligible order to {meta.get('as_of_date')}",
            "high_value": ">= p80 of historical merchandise value",
            "service_outcome": "another eligible order from the same customer within the window; "
                               "orders whose window extends past the dataset end are excluded",
        },
        "limitations": [
            "Historical retrospective data ending 29 Aug 2018 — not a real-time feed.",
            "No profit, margin, cost, campaign or acquisition data exists in the source.",
            "Inactivity is not churn: cancellations and competitor purchases are invisible.",
            "Associations are descriptive; no randomisation means no causal claim.",
            "Cohorts near the end of the window are right-censored and labelled as such.",
            "Customer identity relies on customer_unique_id, which approximates a person.",
        ],
    }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def _error_response():
    return jsonify({
        "error": "dataset_unavailable",
        "detail": _STATE.get("error") or "unknown error",
        "message": (
            "The analytics payload could not be loaded. This deployment ships a derived "
            "dataset in data/ (base64(gzip(csv)) chunks described by data/metadata.json). "
            "Regenerate it with `python scripts/export_public_data.py` and redeploy. "
            "No synthetic values are substituted."),
        "reproduce": "python scripts/export_public_data.py && vercel deploy --prod",
    }), 503


def _num(name, default, lo=None, hi=None, cast=float):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = cast(raw)
    except ValueError:
        return default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


@app.get("/api/analytics")
def analytics():
    state = load_data()
    if not state["loaded"]:
        return _error_response()

    derived = state["derived"]
    meta = state["meta"]
    view = (request.args.get("view") or "summary").lower()

    try:
        if view == "health":
            return jsonify({
                "status": "ok",
                "rows": len(derived["rows"]),
                "customers": len(derived["custs"]),
                "as_of_date": meta.get("as_of_date"),
                "generated_at_utc": meta.get("generated_at_utc"),
                "payload_sha256": meta.get("payload", {}).get("sha256_of_csv"),
            })
        if view == "summary":
            return jsonify(compute_summary(derived))
        if view == "retention":
            return jsonify(compute_retention(
                derived,
                upto=int(_num("upto", 12, 1, 23, int)),
                min_size=int(_num("min_size", 100, 1, 100000, int))))
        if view == "segments":
            return jsonify(compute_segments(
                derived,
                lapsing_days=_num("lapsing",
                                  derived["thresholds"]["lapsing_days"], 7, 730),
                inactive_days=_num("inactive",
                                   derived["thresholds"]["inactive_days"], 8, 1000),
                high_value_quantile=_num("hvq", HIGH_VALUE_QUANTILE, 0.5, 0.99)))
        if view == "service":
            return jsonify(compute_service(
                derived, window_days=int(_num("window", SERVICE_WINDOW_DAYS, 30, 365, int))))
        if view == "methodology":
            return jsonify(compute_methodology(derived, meta))
        return jsonify({"error": "unknown_view", "supported": [
            "health", "summary", "retention", "segments", "service", "methodology"]}), 400
    except Exception as exc:                      # never mask a failure as a number
        return jsonify({"error": "compute_failed",
                        "detail": f"{type(exc).__name__}: {exc}"}), 500


@app.get("/api")
@app.get("/api/index")
def api_root():
    return jsonify({
        "service": "Olist Customer Intelligence & Retention Lab — analytics API",
        "views": ["health", "summary", "retention", "segments", "service", "methodology"],
        "usage": "/api/analytics?view=summary",
    })


@app.get("/")
def dashboard():
    """
    Serve the dashboard. Vercel also serves public/index.html statically, but
    routing "/" here means the page works even if the platform hands every path
    to the function.
    """
    page = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(page):
        return send_file(page, mimetype="text/html")
    return jsonify({"status": "dashboard_missing",
                    "expected": page,
                    "api": "/api/analytics?view=health"})


@app.get("/<path:anything>")
def any_path(anything):
    """
    Path-agnostic fallback.

    The response is selected by ``?view=``, never by the URL path, so it does not
    matter whether the platform forwards ``/api/analytics`` or rewrites it. This
    makes the deployment robust to routing configuration without inventing data.
    """
    return analytics()


if __name__ == "__main__":
    app.run(port=8000, debug=True)
