"""
Shared UI building blocks: CSS, KPI cards, section headers, callouts, formatters.

Kept separate from the page modules so that the visual language stays
consistent across the whole application.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
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
# CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .stApp {{ background: {P['background']}; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        /* ---------- masthead ---------- */
        .app-masthead {{
            display: flex; align-items: flex-start; justify-content: space-between;
            gap: 24px; padding: 20px 26px 24px 26px; margin-bottom: 14px; min-height: 0;
            background: linear-gradient(120deg, #0F172A 0%, #1E3A8A 100%);
            border-radius: 14px; color: #F8FAFC;
        }}
        .app-masthead h1 {{
            font-size: 1.55rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.02em;
            color: #FFFFFF;
        }}
        .app-masthead p {{ margin: 0; color: #CBD5E1; font-size: 0.92rem; line-height: 1.45; }}
        .app-masthead .meta {{
            text-align: right; font-size: 0.75rem; color: #94A3B8; line-height: 1.5;
            white-space: nowrap; display: flex; gap: 26px;
        }}
        .app-masthead .meta > div {{ margin: 0 !important; }}
        .app-masthead .meta b {{ color: #E2E8F0; font-weight: 600; }}

        /* ---------- KPI cards ---------- */
        .kpi-grid {{ display: grid; gap: 14px; margin-bottom: 6px; }}
        .kpi-card {{
            background: #FFFFFF; border: 1px solid {P['line']}; border-radius: 12px;
            padding: 16px 18px; height: 100%;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }}
        .kpi-card .kpi-label {{
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: {P['muted']}; margin-bottom: 8px;
        }}
        .kpi-card .kpi-value {{
            font-size: 1.65rem; font-weight: 700; color: {P['ink']};
            letter-spacing: -0.02em; line-height: 1.15;
        }}
        .kpi-card .kpi-sub {{ font-size: 0.78rem; color: {P['slate']}; margin-top: 7px; line-height: 1.4; }}
        .kpi-card .kpi-rule {{
            font-size: 0.7rem; color: {P['muted']}; margin-top: 8px; line-height: 1.35;
            border-top: 1px dashed {P['line']}; padding-top: 7px;
        }}
        .kpi-card.accent-primary {{ border-left: 3px solid {P['primary']}; }}
        .kpi-card.accent-teal    {{ border-left: 3px solid {P['teal']}; }}
        .kpi-card.accent-amber   {{ border-left: 3px solid {P['amber']}; }}
        .kpi-card.accent-rose    {{ border-left: 3px solid {P['rose']}; }}
        .kpi-card.accent-violet  {{ border-left: 3px solid {P['violet']}; }}

        /* ---------- section headings ---------- */
        .section-head {{ margin: 26px 0 4px 0; }}
        .section-head .eyebrow {{
            font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
            text-transform: uppercase; color: {P['primary']};
        }}
        .section-head h2 {{
            font-size: 1.18rem; font-weight: 650; color: {P['ink']};
            margin: 3px 0 4px 0; letter-spacing: -0.01em;
        }}
        .section-head p {{ margin: 0 0 6px 0; color: {P['slate']}; font-size: 0.88rem; line-height: 1.5; }}

        /* ---------- callouts ---------- */
        .callout {{
            border-radius: 10px; padding: 13px 16px; margin: 10px 0 14px 0;
            font-size: 0.865rem; line-height: 1.55; border: 1px solid;
        }}
        .callout .ttl {{ font-weight: 650; display: block; margin-bottom: 4px; font-size: 0.8rem;
                         text-transform: uppercase; letter-spacing: 0.05em; }}
        .callout.definition {{ background: #F8FAFC; border-color: {P['line']}; color: {P['slate']}; }}
        .callout.definition .ttl {{ color: {P['slate']}; }}
        .callout.insight {{ background: #EFF6FF; border-color: #BFDBFE; color: #1E3A8A; }}
        .callout.insight .ttl {{ color: {P['primary']}; }}
        .callout.warning {{ background: #FFFBEB; border-color: #FDE68A; color: #78350F; }}
        .callout.warning .ttl {{ color: {P['amber']}; }}
        .callout.danger {{ background: #FFF1F2; border-color: #FECDD3; color: #881337; }}
        .callout.danger .ttl {{ color: {P['rose']}; }}
        .callout.success {{ background: #F0FDF4; border-color: #BBF7D0; color: #14532D; }}
        .callout.success .ttl {{ color: {P['green']}; }}

        /* ---------- misc ---------- */
        .pill {{
            display: inline-block; padding: 2px 9px; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
        }}
        .pill.high {{ background: #FEE2E2; color: #991B1B; }}
        .pill.medium {{ background: #FEF3C7; color: #92400E; }}
        .pill.low {{ background: #E2E8F0; color: #334155; }}
        .muted-note {{ font-size: 0.78rem; color: {P['muted']}; line-height: 1.5; }}
        .divider {{ height: 1px; background: {P['line']}; margin: 18px 0; border: 0; }}

        section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {P['line']}; }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            height: 38px; border-radius: 8px 8px 0 0; padding: 0 16px; font-size: 0.86rem;
        }}
        .stDataFrame {{ border: 1px solid {P['line']}; border-radius: 10px; }}
        div[data-testid="stExpander"] {{ border: 1px solid {P['line']}; border-radius: 10px; background: #fff; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------

def masthead(context_info: dict) -> None:
    st.markdown(
        f"""
        <div class="app-masthead">
          <div>
            <h1>{settings.APP_TITLE}</h1>
            <p>{settings.APP_TAGLINE}</p>
          </div>
          <div class="meta">
            <div><b>Observation window</b><br>{context_info.get('period', '—')}</div>
            <div><b>Recency as of</b><br>{context_info.get('as_of', '—')}</div>
            <div><b>Source</b><br>Olist public dataset</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section-head">{eyebrow_html}<h2>{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", rule: str = "", accent: str = "primary") -> str:
    rule_html = f'<div class="kpi-rule">{rule}</div>' if rule else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card accent-{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}{rule_html}</div>'
    )


def kpi_row(cards: list[str], per_row: int = 4) -> None:
    """Render KPI cards in a responsive grid."""
    rows = [cards[i:i + per_row] for i in range(0, len(cards), per_row)]
    for row in rows:
        cols = st.columns(len(row), gap="medium")
        for col, card in zip(cols, row):
            with col:
                st.markdown(card, unsafe_allow_html=True)


def callout(text: str, kind: str = "definition", title: str = "") -> None:
    ttl = f'<span class="ttl">{title}</span>' if title else ""
    st.markdown(f'<div class="callout {kind}">{ttl}{text}</div>', unsafe_allow_html=True)


def definition(title: str, text: str) -> None:
    callout(text, "definition", title)


def empty_state(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div style="background:#fff;border:1px dashed {P['line']};border-radius:12px;
                    padding:38px 28px;text-align:center;margin:18px 0;">
          <div style="font-size:1rem;font-weight:650;color:{P['ink']};margin-bottom:6px;">{title}</div>
          <div style="font-size:0.88rem;color:{P['slate']};max-width:620px;margin:0 auto;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def priority_pill(priority: str) -> str:
    cls = {"High": "high", "Medium": "medium", "Low": "low"}.get(priority, "low")
    return f'<span class="pill {cls}">{priority} priority</span>'


def note(text: str) -> None:
    st.markdown(f'<div class="muted-note">{text}</div>', unsafe_allow_html=True)


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
