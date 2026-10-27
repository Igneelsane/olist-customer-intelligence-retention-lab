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

# Section label -> (render function, title shown in the page header, one line of context)
PAGES = {
    "Overview": (page_overview.render, "Overview",
                 "How many customers bought, how much value they generated, and how often they "
                 "came back."),
    "Retention": (page_retention.render, "Retention",
                  "Whether customers return after their first purchase, and how fast that signal "
                  "fades."),
    "Customers": (page_segmentation.render, "Customers",
                  "Who the customers are, how historical value splits across segments, and which "
                  "groups are worth acting on."),
    "Customer experience": (page_experience.render, "Customer experience",
                            "What reviews and delivery show about repeat purchasing — and why that "
                            "is not a causal claim."),
    "Action planner": (page_planner.render, "Action planner",
                       "Evidence, proposed action and a scenario model for one segment at a time."),
    "Methodology": (page_methodology.render, "Methodology",
                    "Definitions, cleaning rules, identity handling and the quality checks behind "
                    "every number."),
}


def sidebar(data_dir: str) -> str:
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="kicker">Retention analytics</div>'
            f'<div class="mark">{settings.APP_TITLE}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-label">Sections</div>', unsafe_allow_html=True)
        page = st.radio("Sections", list(PAGES.keys()), label_visibility="collapsed")

        st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
        with st.expander("Data source", expanded=False):
            st.markdown(
                f'<div class="note">Reading from<br><code>{data_dir}</code></div>',
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

        with st.expander("Definitions", expanded=False):
            st.markdown(
                """
**Eligible order** — status `delivered` with a valid purchase timestamp.

**Customer** — distinct `customer_unique_id`, not `customer_id`.

**Repeat customer** — at least two eligible orders.

**Month-n retention** — share of a first-purchase cohort ordering in that
specific month *n*, not cumulatively.

**Value** — historical merchandise value (item prices). Revenue-like, not
profit.

**Recency** — days since the customer's last eligible order.
                """
            )

        st.markdown(
            '<div class="note" style="margin-top:18px">Independent case study on the public '
            'Olist dataset (CC BY-NC-SA 4.0). Not affiliated with Olist; no private data.</div>',
            unsafe_allow_html=True,
        )

    return page


def main() -> None:
    override = st.session_state.get("data_dir_override") or os.environ.get("OLIST_DATA_DIR")
    data_dir = override or str(settings.RAW_DATA_DIR)

    page = sidebar(data_dir)
    _, title, subtitle = PAGES[page]

    # ------------------------------------------------------------------
    # Missing-data state -- the app must degrade gracefully, never invent data
    # ------------------------------------------------------------------
    if not data_available(data_dir):
        ui.page_header(title, subtitle, {"Status": "Dataset not found"})
        info = data_loader.describe_dataset(data_dir)
        ui.callout(
            "The application does not generate or substitute synthetic customer records. "
            "It only analyses the real, public Olist dataset, so the analysis stays switched "
            "off until the files are present.",
            "risk", "No real data loaded",
        )
        st.markdown(data_loader.missing_data_message(info))
        ui.section_header("Where the analysis lives in the meantime",
                          "The pipeline is readable and testable without the dataset; only the "
                          "numbers on the pages need it.")
        st.markdown(
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
            ui.page_header(title, subtitle, {"Status": "Dataset error"})
            st.error(f"Dataset error: {exc}")
            st.markdown(data_loader.missing_data_message(data_loader.describe_dataset(data_dir)))
            st.stop()
        except Exception as exc:  # surfaced, never hidden
            ui.page_header(title, subtitle, {"Status": "Pipeline error"})
            st.error(f"The analysis pipeline failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
            st.stop()

    summary = ctx.metrics.summary
    quality_label = ("All checks passed" if ctx.quality.passed
                     else f"{len(ctx.quality.failures)} check(s) failed")
    ui.page_header(
        title, subtitle,
        {
            "Observation window": f"{summary['first_order_date']:%b %Y} – "
                                  f"{summary['as_of_date']:%b %Y}",
            "Recency as of": f"{summary['as_of_date']:%d %b %Y}",
            "Data quality": quality_label,
        },
    )

    if not ctx.quality.passed:
        ui.callout(
            f"{len(ctx.quality.failures)} data quality check(s) failed. Results may be "
            "unreliable — see *Methodology*.", "risk", "Quality warning")

    PAGES[page][0](ctx)

    st.markdown(
        f'<div class="note" style="margin-top:34px;border-top:1px solid {settings.PALETTE["line"]};'
        f'padding-top:14px">Historical retrospective analysis of the public '
        f'<a href="{settings.DATASET_SOURCE_URL}" target="_blank">Brazilian E-Commerce Public '
        f'Dataset by Olist</a> ({settings.DATASET_LICENSE}). Independent case study — not '
        f'affiliated with Olist. Every figure is computed from the loaded data at run time; '
        f'nothing is hardcoded.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
