"""
Customer Intelligence & Retention Lab
=====================================

Streamlit entry point. This file only handles navigation, page chrome and
error states -- all analysis lives in ``src/`` and all page content in
``src/ui/page_*.py``.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from config import settings
from src import data_loader
from src.ui import components as ui
from src.ui import (
    page_experience,
    page_methodology,
    page_overview,
    page_planner,
    page_retention,
    page_segmentation,
)
from src.ui.context import data_available, get_context

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()

PAGES = {
    "Executive Overview": page_overview.render,
    "Retention Intelligence": page_retention.render,
    "Customer Segmentation": page_segmentation.render,
    "Customer Experience": page_experience.render,
    "Retention Action Planner": page_planner.render,
    "Methodology & Data Quality": page_methodology.render,
}

PAGE_CAPTIONS = {
    "Executive Overview": "Headline KPIs",
    "Retention Intelligence": "Cohorts & curves",
    "Customer Segmentation": "RFM segments",
    "Customer Experience": "Reviews & delivery",
    "Retention Action Planner": "Evidence → action",
    "Methodology & Data Quality": "Definitions & audit",
}


def sidebar(data_dir: str) -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:2px 0 14px 0;">
              <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.11em;
                          text-transform:uppercase;color:{settings.PALETTE['primary']};">
                Analytics product
              </div>
              <div style="font-size:1.02rem;font-weight:700;color:{settings.PALETTE['ink']};
                          line-height:1.25;margin-top:4px;">
                Customer Intelligence<br>&amp; Retention Lab
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            format_func=lambda p: p,
            label_visibility="collapsed",
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="muted-note"><b>About</b><br>Independent retrospective case study on the '
            'public Olist dataset (CC BY-NC-SA 4.0). Not affiliated with Olist; no private data '
            'is used.</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Data source settings", expanded=False):
            st.markdown(
                f'<div class="muted-note">Currently reading from:<br><code>{data_dir}</code></div>',
                unsafe_allow_html=True,
            )
            override = st.text_input(
                "Dataset directory (optional)",
                value=os.environ.get("OLIST_DATA_DIR", ""),
                placeholder="/absolute/path/to/olist/csvs",
                help="Folder containing the Olist CSV files. Leaving this blank uses data/raw/.",
            )
            if override:
                st.session_state["data_dir_override"] = override.strip()
                st.rerun()

        with st.expander("Glossary", expanded=False):
            st.markdown(
                """
**Eligible order** — status `delivered` with a valid purchase timestamp.

**Customer** — distinct `customer_unique_id`, not `customer_id`.

**Repeat customer** — ≥2 eligible orders ever.

**Month-n retention** — share of a first-purchase cohort that ordered in that
specific month n.

**Monetary value** — historical merchandise value (item prices). Revenue-like,
not profit.

**Recency** — days since the customer's last eligible order.
                """
            )

    return page


def main() -> None:
    override = st.session_state.get("data_dir_override") or os.environ.get("OLIST_DATA_DIR")
    data_dir = override or str(settings.RAW_DATA_DIR)

    page = sidebar(data_dir)

    # ------------------------------------------------------------------
    # Missing-data state -- the app must degrade gracefully, never invent data
    # ------------------------------------------------------------------
    if not data_available(data_dir):
        st.markdown(
            f"""
            <div class="app-masthead">
              <div>
                <h1>{settings.APP_TITLE}</h1>
                <p>{settings.APP_TAGLINE}</p>
              </div>
              <div class="meta"><div><b>Status</b><br>Dataset not found</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        info = data_loader.describe_dataset(data_dir)
        ui.callout(
            "The application will not generate or substitute synthetic customer records. "
            "It only analyses the real, public Olist dataset.",
            "danger", "No real data loaded",
        )
        st.markdown(data_loader.missing_data_message(info))
        st.markdown("---")
        st.markdown(
            "**Meanwhile, you can still explore the code:**\n\n"
            "- `src/cleaning.py` — order eligibility, child-table aggregation, reconciliation\n"
            "- `src/cohort_analysis.py` — cohort construction and retention maths\n"
            "- `src/segmentation.py` — rule-based RFM and the k-means extension\n"
            "- `src/recommendations.py` — recommendation catalogue and scenario simulator\n"
            "- `tests/` — unit tests on small synthetic fixtures\n"
        )
        st.stop()

    # ------------------------------------------------------------------
    # Normal state
    # ------------------------------------------------------------------
    with st.spinner("Building analytical tables from the Olist dataset…"):
        try:
            ctx = get_context(data_dir)
        except data_loader.DatasetNotFoundError as exc:
            st.error(f"Dataset error: {exc}")
            st.markdown(data_loader.missing_data_message(data_loader.describe_dataset(data_dir)))
            st.stop()
        except Exception as exc:  # surfaced, never hidden
            st.error(f"The analysis pipeline failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
            st.stop()

    summary = ctx.metrics.summary
    ui.masthead({
        "period": f"{summary['first_order_date']:%b %Y} – {summary['as_of_date']:%b %Y}",
        "as_of": f"{summary['as_of_date']:%d %b %Y}",
    })

    if not ctx.quality.passed:
        ui.callout(
            f"{len(ctx.quality.failures)} data quality check(s) failed. Results may be unreliable — "
            "see *Methodology & Data Quality*.", "danger", "Quality warning")

    PAGES[page](ctx)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="muted-note">Historical retrospective analysis of the public '
        f'<a href="{settings.DATASET_SOURCE_URL}" target="_blank">Brazilian E-Commerce Public '
        f'Dataset by Olist</a> ({settings.DATASET_LICENSE}). Independent case study — not '
        f'affiliated with Olist. All figures are computed from the loaded data; no value is '
        f'hardcoded.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
