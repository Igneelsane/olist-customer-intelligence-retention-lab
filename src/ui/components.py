"""
Shared UI building blocks for the application shell.

Everything visual lives here so the six pages share one language: a warm
off-white canvas, hairline rules instead of boxes, one accent colour, and a
typographic scale with three clear levels (page title → section title →
supporting text). Pages only ever call the helpers in this module, never raw
CSS, which is why the whole app can be re-skinned from one file.

The helpers are deliberately small and student-readable:

``page_header``   compact page header (title, one-line context, right-side meta)
``section_header`` section title + optional one-line explanation
``kpi_strip``     4 primary metrics on one row, separated by hairlines
``stat_list``     secondary metrics as a compact label/value list
``findings``      ranked findings: statement, supporting metric, relevance
``callout``       quiet note block (filled colour reserved for risk states)
``filters``       popover wrapper so secondary controls stay out of the way
``note``          caption text
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings

P = settings.PALETTE


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.0f}"


def fmt_pct(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def fmt_money(x, symbol: str = "R$", digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{symbol} {x:,.{digits}f}"


def fmt_compact_money(x, symbol: str = "R$") -> str:
    """Short form for axis labels and tight rows (R$ 13.2M)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= limit:
            return f"{symbol} {x / limit:,.1f}{suffix}"
    return f"{symbol} {x:,.0f}"


def fmt_days(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.0f} days"


def anonymize_id(value: str, prefix: str = "C-") -> str:
    """
    Stable pseudonymous label for a customer identifier.

    The raw ``customer_unique_id`` is never displayed in the UI or exported.
    A salted-free SHA-1 prefix keeps the label stable across pages and exports
    while removing the original key.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}{digest}"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

CSS = """
<style>
/* ---------- canvas and type ---------- */
html, body, [class*="css"] {{
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}}
.stApp {{ background: {bg}; }}
.block-container {{ padding: 1.5rem 2rem 3.5rem 2rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {ink}; letter-spacing: -0.015em; }}
p, li, .stMarkdown {{ color: {slate}; line-height: 1.6; }}
code {{ background: {surface_soft}; color: {ink}; padding: 1px 5px;
        border-radius: 4px; font-size: 0.84em; }}
a {{ color: {ink}; text-decoration: none; border-bottom: 1px solid {line}; }}
a:hover {{ border-bottom-color: {primary}; }}
hr {{ border: 0; border-top: 1px solid {line}; margin: 1.5rem 0; }}
#MainMenu, footer, header {{ visibility: hidden; }}

/* ---------- page header ---------- */
.page-head {{ display: flex; align-items: flex-end; justify-content: space-between;
              gap: 32px; padding: 0 0 14px 0; border-bottom: 1px solid {line};
              margin-bottom: 26px; flex-wrap: wrap; }}
.page-head .titles {{ flex: 1 1 55%; min-width: 0; }}
.page-head h1 {{ font-size: 1.32rem; font-weight: 600; margin: 0; letter-spacing: -0.02em; }}
.page-head .sub {{ margin: 5px 0 0 0; font-size: 0.86rem; color: {slate};
                   max-width: 62ch; line-height: 1.55; }}
.page-head .meta {{ flex: 0 0 auto; display: flex; gap: 28px; font-size: 0.74rem;
                    color: {muted}; white-space: nowrap; padding-bottom: 3px; }}
.page-head .meta b {{ display: block; color: {slate}; font-weight: 500;
                      text-transform: uppercase; letter-spacing: 0.06em;
                      font-size: 0.66rem; margin-bottom: 3px; }}

/* ---------- section headings ---------- */
.section {{ margin: 34px 0 2px 0; }}
.section:first-of-type {{ margin-top: 10px; }}
.section .eyebrow {{ font-size: 0.66rem; font-weight: 600; letter-spacing: 0.1em;
                     text-transform: uppercase; color: {muted}; }}
.section h2 {{ font-size: 0.98rem; font-weight: 600; margin: 4px 0 0 0; }}
.section p {{ margin: 5px 0 0 0; font-size: 0.85rem; color: {slate};
              max-width: 78ch; }}

/* ---------- primary KPI strip ---------- */
.kpi-strip {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
              border-top: 1px solid {line}; border-bottom: 1px solid {line};
              margin: 8px 0 4px 0; }}
.kpi-strip .kpi {{ padding: 16px 20px 15px 20px; border-left: 1px solid {line}; }}
.kpi-strip .kpi:first-child {{ border-left: 0; padding-left: 0; }}
.kpi-strip .kpi .label {{ font-size: 0.7rem; color: {muted}; letter-spacing: 0.04em;
                          text-transform: uppercase; font-weight: 500; }}
.kpi-strip .kpi .value {{ font-size: 1.5rem; font-weight: 600; color: {ink};
                          letter-spacing: -0.025em; line-height: 1.25;
                          margin-top: 6px; font-variant-numeric: tabular-nums; }}
.kpi-strip .kpi .sub {{ font-size: 0.76rem; color: {slate}; margin-top: 4px; }}
.kpi-strip .kpi .hint {{ font-size: 0.72rem; color: {muted}; margin-top: 3px;
                         line-height: 1.4; }}
.kpi-strip .kpi .value.risk {{ color: {rose}; }}

/* ---------- secondary metric list ---------- */
.stat-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 0; border-top: 1px solid {line_soft}; margin: 4px 0 0 0; }}
.stat-list .stat {{ padding: 11px 16px 11px 0; }}
.stat-list .stat .label {{ font-size: 0.72rem; color: {muted}; }}
.stat-list .stat .value {{ font-size: 0.98rem; font-weight: 550; color: {ink};
                           margin-top: 2px; font-variant-numeric: tabular-nums; }}
.stat-list .stat .sub {{ font-size: 0.7rem; color: {muted}; margin-top: 1px; }}

/* ---------- findings ---------- */
.findings {{ border-top: 1px solid {line}; }}
.findings .finding {{ display: grid; grid-template-columns: 1fr 150px; gap: 24px;
                      padding: 15px 0 14px 0; border-bottom: 1px solid {line_soft}; }}
.findings .finding .stmt {{ font-size: 0.9rem; font-weight: 550; color: {ink};
                            line-height: 1.45; }}
.findings .finding .why {{ font-size: 0.83rem; color: {slate}; margin-top: 5px;
                           line-height: 1.55; max-width: 74ch; }}
.findings .finding .metric {{ text-align: right; font-size: 1.02rem; font-weight: 600;
                              color: {ink}; font-variant-numeric: tabular-nums;
                              white-space: nowrap; }}
.findings .finding .metric span {{ display: block; font-size: 0.68rem; font-weight: 400;
                                   color: {muted}; margin-top: 3px;
                                   text-transform: uppercase; letter-spacing: 0.05em; }}

/* ---------- notes, callouts ---------- */
.note {{ font-size: 0.78rem; color: {muted}; line-height: 1.55; }}
.note b, .note code {{ color: {slate}; }}
.callout {{ padding: 12px 16px; margin: 12px 0; font-size: 0.845rem; line-height: 1.6;
            border-left: 2px solid {line}; background: transparent; color: {slate}; }}
.callout .ttl {{ display: block; font-size: 0.72rem; font-weight: 600;
                 letter-spacing: 0.04em; text-transform: uppercase; color: {slate};
                 margin-bottom: 4px; }}
.callout.note-block {{ border-left-color: {primary}; background: {primary_soft}; }}
.callout.note-block .ttl {{ color: {primary}; }}
.callout.risk {{ border-left-color: {rose}; background: #FBF1EF; color: #6E2A21; }}
.callout.risk .ttl {{ color: {rose}; }}
.callout.caution {{ border-left-color: {amber}; background: #FAF5EA; color: #6B5324; }}
.callout.caution .ttl {{ color: {amber}; }}
.callout.plain {{ border-left-color: {line}; }}
.callout code, .note code {{ background: rgba(0,0,0,0.035); }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {surface}; border-right: 1px solid {line}; }}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    padding: 1.35rem 1rem 2rem 1.15rem; }}
.brand {{ margin-bottom: 22px; }}
.brand .mark {{ font-size: 0.9rem; font-weight: 600; color: {ink};
                letter-spacing: -0.01em; line-height: 1.3; }}
.brand .kicker {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em;
                  color: {muted}; margin-bottom: 6px; }}
.nav-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em;
              color: {muted}; margin: 0 0 6px 2px; }}

/* navigation: radio rendered as a plain list, no dots, quiet active state */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{
    padding: 6px 10px; border-radius: 5px; margin: 0 0 1px 0; width: 100%;
    transition: background 90ms ease; cursor: pointer; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{
    background: {surface_soft}; }}
/* hide the radio dot -- the section is a list, not a form */
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {{
    display: none; padding: 0; margin: 0; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{
    font-size: 0.855rem; color: {slate}; font-weight: 450; margin: 0; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {{
    background: transparent; border-radius: 0;
    box-shadow: inset 2px 0 0 {primary}; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p {{
    color: {ink}; font-weight: 600; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:focus-within {{
    outline: 1px solid {primary}; outline-offset: -1px; }}
section[data-testid="stSidebar"] div[data-testid="stExpander"],
section[data-testid="stSidebar"] div[data-testid="stExpander"] details {{
    border: 0 !important; border-radius: 0 !important; background: transparent !important;
    box-shadow: none !important; }}
section[data-testid="stSidebar"] div[data-testid="stExpander"] {{
    border-top: 1px solid {line} !important; margin: 0; }}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
    padding: 8px 0; font-size: 0.78rem; color: {slate}; }}

/* ---------- widgets ---------- */
div[data-testid="stPopover"] > button {{
    border: 1px solid {line}; border-radius: 5px; background: {surface};
    color: {slate}; font-size: 0.78rem; font-weight: 500; padding: 4px 12px;
    min-height: 30px; }}
div[data-testid="stPopover"] > button:hover {{
    border-color: {muted}; color: {ink}; }}
div[data-testid="stExpander"] {{ border: 1px solid {line}; border-radius: 6px;
                                 background: {surface}; margin-top: 14px; }}
div[data-testid="stExpander"] summary {{ font-size: 0.82rem; color: {slate}; }}
div[data-testid="stExpander"] summary:hover {{ color: {ink}; }}
div[data-testid="stExpander"] p, div[data-testid="stExpander"] li {{ font-size: 0.84rem; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {line}; border-radius: 6px; }}
div[data-testid="stMetricValue"] {{ font-size: 1.25rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 18px; border-bottom: 1px solid {line}; }}
.stTabs [data-baseweb="tab"] {{ padding: 4px 0 8px 0; font-size: 0.84rem;
                                color: {muted}; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: {ink}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {primary}; height: 1.5px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {{
    border: 1px solid {line}; border-radius: 5px; background: {surface};
    color: {slate}; font-size: 0.78rem; font-weight: 500; box-shadow: none; }}
div[data-testid="stDownloadButton"] button:hover, div[data-testid="stButton"] button:hover {{
    border-color: {muted}; color: {ink}; }}
div[data-baseweb="select"] > div {{ border-radius: 5px; border-color: {line};
                                    font-size: 0.82rem; }}
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label, div[data-testid="stRadio"] label,
div[data-testid="stTextInput"] label, div[data-testid="stCheckbox"] label {{
    font-size: 0.78rem; color: {slate}; }}
[data-testid="stWidgetLabel"] p {{ font-size: 0.78rem; color: {slate}; }}
div[data-testid="stAlert"] {{ border-radius: 6px; font-size: 0.83rem; }}

/* ---------- status pill ---------- */
.pill {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.68rem;
         font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase;
         border: 1px solid {line}; color: {slate}; background: {surface_soft}; }}
.pill.risk {{ color: {rose}; border-color: #E4C9C4; background: #FBF1EF; }}
.pill.ok {{ color: {primary}; border-color: #CBD9D3; background: {primary_soft}; }}
.pill.warn {{ color: {amber}; border-color: #E5D9BE; background: #FAF5EA; }}
.pill.high {{ color: {rose}; border-color: #E4C9C4; background: #FBF1EF; }}
.pill.medium {{ color: {amber}; border-color: #E5D9BE; background: #FAF5EA; }}
.pill.low {{ color: {slate}; }}

/* ---------- empty state ---------- */
.empty {{ border-top: 1px solid {line}; border-bottom: 1px solid {line};
          padding: 30px 0; margin: 14px 0; }}
.empty .t {{ font-size: 0.92rem; font-weight: 550; color: {ink}; margin-bottom: 4px; }}
.empty .d {{ font-size: 0.84rem; color: {slate}; max-width: 62ch; }}

/* ---------- tables inside HTML ---------- */
table.plain {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
table.plain th {{ text-align: left; font-weight: 500; font-size: 0.7rem;
                  text-transform: uppercase; letter-spacing: 0.05em; color: {muted};
                  padding: 0 12px 7px 0; border-bottom: 1px solid {line}; }}
table.plain td {{ padding: 8px 12px 8px 0; border-bottom: 1px solid {line_soft};
                  color: {slate}; }}
table.plain td.num {{ text-align: right; font-variant-numeric: tabular-nums; color: {ink}; }}
table.plain th.num {{ text-align: right; }}

/* ---------- responsive ---------- */
@media (max-width: 1100px) {{
    .block-container {{ padding: 1.2rem 1.4rem 3rem 1.4rem; }}
}}
@media (max-width: 860px) {{
    .page-head {{ gap: 14px; }}
    .page-head .meta {{ gap: 18px; }}
    .kpi-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .kpi-strip .kpi {{ padding: 14px 14px 13px 14px; }}
    .kpi-strip .kpi:nth-child(odd) {{ border-left: 0; padding-left: 0; }}
    .kpi-strip .kpi:nth-child(n+3) {{ border-top: 1px solid {line}; }}
    .kpi-strip .kpi .value {{ font-size: 1.32rem; }}
    .findings .finding {{ grid-template-columns: 1fr; gap: 8px; }}
    .findings .finding .metric {{ text-align: left; }}
}}
@media (max-width: 560px) {{
    .block-container {{ padding: 1rem 1rem 3rem 1rem; }}
    .page-head h1 {{ font-size: 1.15rem; }}
    .page-head .meta {{ gap: 14px; font-size: 0.7rem; }}
    .kpi-strip {{ grid-template-columns: 1fr; }}
    .kpi-strip .kpi {{ border-left: 0; padding-left: 0; }}
    .kpi-strip .kpi:nth-child(n+2) {{ border-top: 1px solid {line}; }}
}}
</style>
"""


def inject_css() -> None:
    """Apply the stylesheet once per session."""
    st.markdown(
        CSS.format(
            bg=P["background"], surface=P["surface"], surface_soft=P["surface_soft"],
            ink=P["ink"], slate=P["slate"], muted=P["muted"], line=P["line"],
            line_soft=P["line_soft"], primary=P["primary"], primary_soft=P["primary_soft"],
            rose=P["rose"], amber=P["amber"], green=P["green"],
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "", meta: Optional[dict] = None) -> None:
    """Compact page header. ``meta`` renders as small labelled values on the right."""
    meta_html = "".join(
        f"<div><b>{k}</b>{v}</div>" for k, v in (meta or {}).items()
    )
    sub_html = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="page-head"><div class="titles"><h1>{title}</h1>{sub_html}</div>'
        f'<div class="meta">{meta_html}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section">{eyebrow_html}<h2>{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def metric(label: str, value: str, sub: str = "", hint: str = "", risk: bool = False) -> dict:
    """One metric definition for :func:`kpi_strip` / :func:`stat_list`."""
    return {"label": label, "value": value, "sub": sub, "hint": hint, "risk": risk}


def kpi_strip(items: Sequence[dict]) -> None:
    """Up to four primary metrics on a single row, separated by hairlines."""
    cells = []
    for m in items:
        risk = " risk" if m.get("risk") else ""
        sub = f'<div class="sub">{m["sub"]}</div>' if m.get("sub") else ""
        hint = f'<div class="hint">{m["hint"]}</div>' if m.get("hint") else ""
        cells.append(
            f'<div class="kpi"><div class="label">{m["label"]}</div>'
            f'<div class="value{risk}">{m["value"]}</div>{sub}{hint}</div>'
        )
    st.markdown(f'<div class="kpi-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


def stat_list(items: Sequence[dict]) -> None:
    """Secondary metrics as a compact auto-fitting list."""
    cells = []
    for m in items:
        sub = f'<div class="sub">{m["sub"]}</div>' if m.get("sub") else ""
        cells.append(
            f'<div class="stat"><div class="label">{m["label"]}</div>'
            f'<div class="value">{m["value"]}</div>{sub}</div>'
        )
    st.markdown(f'<div class="stat-list">{"".join(cells)}</div>', unsafe_allow_html=True)


def findings(items: Iterable[dict]) -> None:
    """
    Ranked findings list.

    Each item: ``{"statement": ..., "metric": ..., "metric_label": ..., "why": ...}``
    Keeps the statement (what we saw), the number behind it and its business
    relevance visually separate without boxing anything.
    """
    rows = []
    for item in items:
        rows.append(
            '<div class="finding">'
            f'<div><div class="stmt">{item["statement"]}</div>'
            f'<div class="why">{item.get("why", "")}</div></div>'
            f'<div class="metric">{item.get("metric", "")}'
            f'<span>{item.get("metric_label", "")}</span></div>'
            "</div>"
        )
    st.markdown(f'<div class="findings">{"".join(rows)}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "plain", title: str = "") -> None:
    """
    Note block. ``kind`` is one of ``plain``, ``note-block`` (insight),
    ``caution`` (assumption / caveat) or ``risk`` (data or quality problem).
    Filled colour is reserved for caution and risk so it keeps its meaning.
    """
    css_kind = {"definition": "plain", "insight": "note-block",
                "warning": "caution", "danger": "risk", "success": "note-block"}.get(kind, kind)
    ttl = f'<span class="ttl">{title}</span>' if title else ""
    st.markdown(f'<div class="callout {css_kind}">{ttl}{text}</div>', unsafe_allow_html=True)


def definition(title: str, text: str) -> None:
    callout(text, "plain", title)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def priority_pill(priority: str) -> str:
    return pill(priority, {"High": "high", "Medium": "medium", "Low": "low"}.get(priority, "low"))


def empty_state(title: str, text: str) -> None:
    st.markdown(
        f'<div class="empty"><div class="t">{title}</div><div class="d">{text}</div></div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown("<hr>", unsafe_allow_html=True)


@contextmanager
def filters(label: str = "Filters", active: str = ""):
    """
    Secondary controls live behind a small trigger so the page stays calm.

    The trigger shows how many filters differ from their defaults, which keeps
    the state visible even though the controls are collapsed.
    """
    text = f"{label} · {active}" if active else label
    with st.popover(text, use_container_width=False):
        yield


RESET_KEY = "_pending_reset"


def reset_button(defaults: dict, label: str = "Reset to defaults") -> None:
    """
    Restore widgets to their declared defaults.

    A widget's session-state entry cannot be written once the widget exists in
    the current run, so the values are parked here and applied by
    :func:`apply_pending_reset` at the top of the next run -- which is the same
    pattern Streamlit's own documentation recommends.
    """
    if st.button(label, key=f"reset_{'_'.join(defaults)}", use_container_width=True):
        st.session_state[RESET_KEY] = dict(defaults)
        st.rerun()


def apply_pending_reset() -> None:
    """Apply a queued reset before any widget is created. Call first in a page."""
    pending = st.session_state.pop(RESET_KEY, None)
    if pending:
        for key, value in pending.items():
            st.session_state[key] = value


def active_summary(pairs: Sequence[tuple], default: str = "") -> str:
    """Tiny label for the filter trigger: only non-default settings are shown."""
    shown = [text for differs, text in pairs if differs]
    return " · ".join(shown) if shown else default


def chart(fig: go.Figure, height: Optional[int] = None, title: Optional[str] = None) -> None:
    """
    Render a Plotly figure with the app's chart config (no mode bar clutter).

    Pass ``title=""`` when the section header directly above already names the
    chart — one title, not two.
    """
    if height is not None:
        fig.update_layout(height=height)
    if title == "":
        fig.update_layout(title_text=None, margin_t=16)
    elif title:
        fig.update_layout(title_text=title)
    if len(fig.data) <= 1:
        fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False, "displaylogo": False})


# ---------------------------------------------------------------------------
# Tables and exports
# ---------------------------------------------------------------------------

def pct(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Scale 0-1 fraction columns to 0-100 for display.

    Streamlit's ``format="%.2f%%"`` only appends a sign — it does not multiply —
    so a 0.4827 retention would otherwise render as "0.48%" instead of "48.27%".
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * 100
    return out


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv", key=f"dl_{filename}",
    )


def styled_table(df: pd.DataFrame, column_config: Optional[dict] = None,
                 height: Optional[int] = None, hide_index: bool = True) -> None:
    """Dataframe wrapper that tolerates the optional-height case (Streamlit >= 1.5x)."""
    kwargs = dict(width="stretch", hide_index=hide_index, column_config=column_config or {})
    if height is not None:
        kwargs["height"] = int(height)
    st.dataframe(df, **kwargs)


def html_table(df: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None,
               money_cols: Optional[Sequence[str]] = None) -> None:
    """Read-only HTML table used where a full dataframe would be overkill."""
    numeric = set(numeric_cols or ())
    money = set(money_cols or ())
    head = "".join(
        f'<th class="num">{c}</th>' if c in numeric or c in money else f"<th>{c}</th>"
        for c in df.columns
    )
    body = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col in money:
                txt = fmt_money(val, digits=0) if not isinstance(val, str) else val
            elif col in numeric:
                txt = fmt_int(val) if not isinstance(val, str) else val
            else:
                txt = str(val)
            cls = "num" if col in numeric or col in money else ""
            cells.append(f'<td class="{cls}">{txt}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f'<table class="plain"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>',
        unsafe_allow_html=True,
    )
