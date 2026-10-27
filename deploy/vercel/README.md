# Vercel deployment — architecture, status and exact steps

This folder holds a **Vercel-native build** of the project: a Python WSGI function that
recomputes the analytics on request, plus a static dashboard that calls it.

Nothing here is a screenshot gallery and nothing is hardcoded. If the function cannot load its
data it returns HTTP 503 with an explicit message, and the dashboard prints that message instead
of showing numbers.

---

## 1. Why the deployment is not the Streamlit app

Streamlit needs a long-lived process and a WebSocket connection. Vercel Functions are
request-scoped: each request may hit a fresh instance, and there is no persistent socket. Every
"deploy Streamlit to Vercel" path ends in a broken or fake deployment, so this project does not
attempt it.

Instead the deployment keeps the *computation*:

| Concern | Local app (`/app.py`) | This deployment (`deploy/vercel/`) |
|---|---|---|
| Interface | Streamlit, 6 pages | Static HTML/CSS/SVG dashboard, 5 tabs |
| Compute | pandas, in-process | Pure-Python function, per request |
| Data | raw Kaggle CSVs → pipeline | pre-built anonymised derived table |
| Runtime | persistent process | request-scoped function (Vercel Python) |

The two share definitions, not code: `deploy/vercel/api/index.py` reimplements the metrics with
the same rules as `src/`, and the exported table is produced by the same pipeline.

The Streamlit app itself remains deployable on **Streamlit Community Cloud** (free, native
support, no code change) or Render — see the root `README.md`.

---

## 2. Layout

```
deploy/vercel/
├── api/
│   └── index.py          Flask app: /api/analytics?view=… and the dashboard at /
├── data/                 anonymised derived table (committed on purpose)
│   ├── metadata.json     epoch, as-of date, column meanings, SHA-256 of the CSV
│   └── orders.NN.b64     base64(gzip(csv)) chunks
├── public/
│   └── index.html        self-contained dashboard (inline CSS, inline SVG, no CDN)
├── requirements.txt      flask only — no pandas/numpy/scipy in the function
└── vercel.json           function memory/duration + includeFiles for data/ and public/
```

`api/index.py` is named exactly that because Vercel's Flask preset looks for one of
`api/index.py`, `api/app.py`, `api/server.py`, … at the project root. A function named
`api/analytics.py` fails the build with `FLASK_ENTRYPOINT_NOT_FOUND` (verified — see §6).

### Endpoints

| Request | Returns |
|---|---|
| `GET /` | the dashboard |
| `GET /api/analytics?view=health` | row/customer counts, as-of date, payload checksum |
| `GET /api/analytics?view=summary` | KPIs, monthly activity, Lorenz curve, thresholds |
| `GET /api/analytics?view=retention&upto=12` | cohort matrix, pooled rates, cumulative repeat, time-to-second |
| `GET /api/analytics?view=segments&lapsing=…&inactive=…&hvq=…` | segment profile + thresholds |
| `GET /api/analytics?view=service&window=90` | review/lateness vs repeat, Wilson CIs, χ² and Cramér's V |
| `GET /api/analytics?view=methodology` | integrity checks + definitions |

Unknown `?view=` returns HTTP 400 listing the supported views. The response is selected by the
query string, never by the path, so a routing rewrite cannot break the API.

### Data payload

`scripts/export_public_data.py` builds it from the real dataset, through the same
`src/cleaning.py` pipeline the app uses:

| Field | Meaning |
|---|---|
| `c` | customer index, dense enumeration (not derived from any identifier) |
| `t` | order purchase timestamp in **seconds** since 2016-09-01 00:00 |
| `v` / `f` | merchandise / freight value in integer cents |
| `r` | review score, 0 = missing |
| `y` / `z` | delivery days / delay vs promise, in **hundredths of a day** (sentinels for missing) |

No `customer_unique_id`, `customer_id`, city, postcode, review text or order id is exported.

Timestamp and day precision are deliberate: cohort months, the 90-day repeat window and the
right-censoring rule are timestamp comparisons. An earlier revision of this export rounded
timestamps to whole days, which changed the measured repeat base rate from 2.48% to 1.47% — a
bug that the cross-check against the pandas pipeline caught before deployment.

---

## 3. Verified locally (this repository, before deployment)

Everything below was executed, not assumed:

- `deploy/vercel/api/index.py` served over HTTP; all six views return HTTP 200
  (summary 0.27 s, retention 0.08 s, segments 0.26 s, service 0.14 s after the payload is warm);
  unknown view → 400; any path → 200 JSON.
- **Cross-check against the pandas pipeline on the real dataset** — same numbers, computed two
  independent ways:

  | Metric | Deployed function | `src/` pipeline |
  |---|---|---|
  | eligible orders / customers | 96,478 / 93,358 | 96,478 / 93,358 |
  | repeat purchase rate | 3.0003% | 3.0003% |
  | month 1 / month 2 pooled retention | 0.4827% / 0.3359% | 0.4827% / 0.3359% |
  | median 12-month cumulative repeat | 2.905% | 2.905% |
  | Gini of customer value | 0.5203 | 0.5203 |
  | service orders used / censored | 77,737 / 18,741 | 77,737 / 18,741 |
  | repeat within 90 days (base) | 2.479% | 2.479% |
  | Cramér's V, review band / lateness | 0.0108 / 0.0069 | 0.0108 / 0.0069 |
  | segment sizes (7 rules) | identical | identical |

- **Browser test** (Playwright/Chromium against the real dashboard): 31 checks — every tab,
  every table, the SVG charts, the KPI values, zero console errors. Screenshots of the deployed
  UI are in `assets/screenshots/deploy_*.png`.
- **Failure path**: with the API blocked, the dashboard shows “The analytics function could not
  be reached, so no numbers are shown” and renders **0** KPI cards. No placeholder values.
- **Missing data path**: with `data/` absent the function returns 503 `dataset_unavailable` with
  the reproduction command.

---

## 4. Deploying it

### Prerequisites

The repository needs to reach Vercel. The payload in `data/` is ~2.2 MB, so it travels with git
rather than through any manual upload step.

### Option A — Vercel dashboard (recommended, no CLI)

1. Push this repository to GitHub (see `HANDOFF.md` §10 for the exact commands).
2. In Vercel: **Add New → Project → Import** the repository.
3. Set **Root Directory** to `deploy/vercel`. Framework preset: **Flask** (auto-detected).
   Leave build and output settings alone — `vercel.json` supplies them.
4. Deploy. The production URL is `<project>.vercel.app`.
5. **Turn off Deployment Protection** for public access: Project → Settings → Deployment
   Protection → Vercel Authentication → **Disabled**. Without this step every request,
   including the API, is redirected to a Vercel login page (verified behaviour on this account).

### Option B — Vercel CLI

```bash
npm i -g vercel
cd deploy/vercel
vercel link          # create or link a project
vercel deploy --prod
```

### Option C — no git at all

```bash
python scripts/export_public_data.py     # rebuild data/ from the raw dataset
cd deploy/vercel
vercel deploy --prod
```

### After deploying, verify

```bash
curl -s https://<your-url>/api/analytics?view=health     # expect status ok, rows 96478
curl -s https://<your-url>/api/analytics?view=methodology | head -c 400
```

If `health` returns 503, the payload did not make it into the bundle: check that
`includeFiles` in `vercel.json` still lists `data/**/*` and `public/index.html`, and that
`deploy/vercel/data/*.b64` are committed.

---

## 5. Current status (as of the last session)

| Step | State |
|---|---|
| Architecture chosen and justified | done |
| Function + dashboard written | done |
| Data payload exported from the real dataset | done (96,478 rows, checksum recorded) |
| Local HTTP verification of all views | done |
| Cross-check against the pandas pipeline | done — all metrics match |
| Browser test of the dashboard (31 checks) | done |
| Vercel project created and configured (framework: Flask, output dir: `public`) | done |
| Vercel build of the real code with data | **not done** |
| Public URL serving real numbers | **not done** |

### Why the public deployment is not live yet

1. **The payload cannot be uploaded from the session that built it.** The deployment tool
   available there accepts inline file contents only, with no path or binary upload; a 2.2 MB
   base64 payload (and even a `vercel.json` file, whose JSON content the tool re-parses) cannot
   be sent that way. The normal transport is git, which needs the GitHub repository below.
2. **GitHub is not reachable from that session** — no credentials, no `gh`, no SSH key. The
   repository does not exist yet, so nothing can be pushed from there.
3. **Deployment Protection is enabled on the account**, so even a successful deployment would
   redirect visitors to a Vercel login page until it is switched off (one setting, free).

None of these are architectural: the code, payload and configuration are complete and verified,
and the project `olist-customer-intelligence-retention-lab` already exists on Vercel, ready to
be connected to the repository.

### Exact remaining steps

1. Create the GitHub repository and push (commands in `HANDOFF.md` §10).
2. Import it in Vercel with root directory `deploy/vercel`, or run `vercel deploy --prod`.
3. Disable Deployment Protection for the project.
4. `curl .../api/analytics?view=health` → expect `{"status": "ok", "rows": 96478, ...}`.

---

## 6. Vercel platform limits that apply here

Measured against Vercel's documentation and this project's actual behaviour:

| Limit (Hobby) | Value | Effect on this project |
|---|---|---|
| Max function duration | 300 s (60 s configured) | worst measured view 0.27 s warm |
| Memory | 2 GB max (1 GB configured) | payload decode peaks well under 300 MB |
| Python bundle | 500 MB | flask only; no pandas/numpy in the function |
| Functions per deployment | 12 | 1 |
| Bundle/cold start | first request decodes ~2.2 MB base64 → ~0.3 s | once per warm instance |
| Deployment Protection | on by default for this account | must be disabled for a public site |
| Framework preset | Flask requires a fixed entrypoint name | `api/index.py`, verified |

The deployment is a **derived, static snapshot** of the dataset, not a live feed: it does not
re-run the cleaning pipeline or pick up new orders. Re-run
`python scripts/export_public_data.py`, commit, and redeploy to refresh it.

---

## 7. Honesty rules this deployment follows

- Every number is computed from shipped rows at request time; nothing is transcribed from a
  report, screenshot or hardcoded constant.
- When data is missing the function refuses to answer (503) rather than serving zeros or
  placeholder values, and the UI says so.
- The dashboard repeats the project's caveats: historical value is revenue, not profit or CLV;
  service results are associations, not causal effects; the study is independent and not
  affiliated with Olist.
- The dataset is CC BY-NC-SA 4.0 (Olist, via Kaggle) and is attributed in the UI footer and the
  repository README.
