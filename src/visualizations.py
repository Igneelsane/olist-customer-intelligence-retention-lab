"""
Shared Plotly styling and chart builders.

Charts are built here (not inside app.py) so that the notebook, the app and the
reports all use the same visual language and the same definitions. Every
function takes already-computed data -- no business logic lives in this module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import settings

P = settings.PALETTE

BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif",
              size=12, color=P["slate"]),
    title_font=dict(size=13.5, color=P["ink"]),
    title=dict(x=0, xanchor="left", y=0.97, yanchor="top"),
    margin=dict(l=6, r=6, t=54, b=8),
    hoverlabel=dict(bgcolor="white", bordercolor=P["line"], font_size=12,
                    font_family="Inter, sans-serif", font_color=P["ink"]),
    colorway=[P["primary"], P["teal"], P["violet"], P["amber"], P["rose"],
              P["green"], P["muted"]],
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(size=11, color=P["slate"]), bgcolor="rgba(0,0,0,0)"),
    plot_bgcolor="white",
    paper_bgcolor="rgba(0,0,0,0)",
    separators=".,",
)


def style(fig: go.Figure, title: str = "", height: int = 320, y_title: str = "",
          x_title: str = "", show_legend: Optional[bool] = None) -> go.Figure:
    """
    Apply the shared chart language: hairline grid, no frame, one accent.

    ``show_legend=None`` hides the legend automatically when there is only one
    series — a one-item legend is pure decoration.
    """
    fig.update_layout(**BASE_LAYOUT)
    if show_legend is None:
        show_legend = len(fig.data) > 1
    fig.update_layout(height=height, title=title,
                      xaxis_title=x_title, yaxis_title=y_title,
                      showlegend=show_legend)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True,
                     linecolor=P["line"], linewidth=1, ticks="",
                     tickfont=dict(size=11, color=P["muted"]),
                     title_font=dict(size=11, color=P["muted"]))
    fig.update_yaxes(gridcolor=P["line_soft"], griddash="dot", zeroline=False,
                     showline=False, ticks="",
                     tickfont=dict(size=11, color=P["muted"]),
                     title_font=dict(size=11, color=P["muted"]))
    return fig


# ---------------------------------------------------------------------------
# KPI / trend charts
# ---------------------------------------------------------------------------

def trend_area(monthly: pd.DataFrame, y: str, title: str, y_title: str,
               marker: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly[y], mode="lines+markers" if marker else "lines",
        line=dict(width=2.4, color=P["primary"]),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(31,77,63,0.07)",
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
        name=y_title,
    ))
    return style(fig, title, y_title=y_title)


def dual_axis_trend(monthly: pd.DataFrame, bar_col: str, line_col: str,
                    title: str, bar_name: str, line_name: str) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly[bar_col], name=bar_name,
                         marker_color=P["primary"],
                         hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly[line_col], name=line_name,
                             mode="lines+markers",
                             line=dict(color=P["teal"], width=2.4),
                             hovertemplate="%{x|%b %Y}<br>%{y:,.1f}<extra></extra>"),
                  secondary_y=True)
    fig = style(fig, title)
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text=line_name, secondary_y=True, showgrid=False)
    return fig


def new_vs_returning(monthly_split: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_split["month"], y=monthly_split["new_customers"],
                         name="First-time buyers", marker_color=P["muted"],
                         hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=monthly_split["month"], y=monthly_split["returning_customers"],
                         name="Returning buyers", marker_color=P["teal"],
                         hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>"))
    fig.update_layout(barmode="stack")
    return style(fig, "Monthly unique customers: first-time vs returning",
                 y_title="Unique customers")


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def histogram(series: pd.Series, title: str, x_title: str, nbins: int = 40,
              log_y: bool = False, color: str = None, y_title: str = "Orders") -> go.Figure:
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig = go.Figure(go.Histogram(x=s, nbinsx=nbins,
                                 marker_color=color or P["primary"], opacity=0.85,
                                 hovertemplate="%{x}<br>%{y:,}<extra></extra>"))
    fig = style(fig, title, x_title=x_title, y_title=y_title)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def bar_horizontal(df: pd.DataFrame, x: str, y: str, title: str, x_title: str,
                   color: Optional[str] = None, height: int = 380) -> go.Figure:
    d = df.sort_values(x, ascending=True)
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h",
                           marker_color=color or P["primary"], opacity=0.9,
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    return style(fig, title, x_title=x_title, height=height, show_legend=False)


def bar_vertical(df: pd.DataFrame, x: str, y: str, title: str, y_title: str,
                 color: Optional[str] = None, text_fmt: str = ",.0f") -> go.Figure:
    fig = go.Figure(go.Bar(x=df[x], y=df[y], marker_color=color or P["primary"],
                           opacity=0.9,
                           hovertemplate="%{x}<br>%{y:,.3s}<extra></extra>"))
    return style(fig, title, y_title=y_title, show_legend=False)


def concentration_curve(customers: pd.DataFrame) -> go.Figure:
    """Lorenz-style cumulative value share by customer percentile."""
    v = customers["monetary_value"].astype(float).sort_values(ascending=False).to_numpy()
    if v.sum() <= 0:
        return style(go.Figure(), "Customer value concentration")
    cum = np.cumsum(v) / v.sum()
    x = np.arange(1, len(v) + 1) / len(v)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", name="Observed",
                             line=dict(color=P["primary"], width=2.6),
                             hovertemplate="Top %{x:.1%} of customers<br>"
                                           "%{y:.1%} of value<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
                             line=dict(color=P["muted"], width=1.4, dash="dash"),
                             hoverinfo="skip"))
    for pct in (0.1, 0.2):
        k = max(1, int(np.ceil(len(v) * pct)))
        fig.add_annotation(x=pct, y=cum[k - 1], text=f"top {int(pct*100)}% → {cum[k-1]:.1%}",
                           showarrow=True, arrowhead=0, ax=40, ay=-30,
                           font=dict(size=11, color=P["ink"]),
                           bgcolor="white", bordercolor=P["line"], borderwidth=1)
    fig = style(fig, "Customer value concentration (cumulative share of merchandise value)",
                x_title="Share of customers, ranked by value", y_title="Cumulative share of value")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickformat=".0%")
    return fig


# ---------------------------------------------------------------------------
# Cohort charts
# ---------------------------------------------------------------------------

def cohort_heatmap(matrix: pd.DataFrame, max_index: int = 12, value_fmt: str = ".2%") -> go.Figure:
    """
    Heatmap of specific-month retention.

    Two deliberate choices:

    * Month 0 is excluded from the colour scale (it is 100% by construction) and
      capped at the 99th percentile of months 1+, otherwise the whole matrix
      renders pale and later-month differences become invisible.
    * NaN cells (unobservable future months) are rendered blank rather than as
      zero, so the visualisation cannot imply that retention fell to zero.
    """
    cols = [c for c in matrix.columns if isinstance(c, (int, np.integer)) and c <= max_index]
    z = matrix[cols].astype(float)
    labels = [f"M{c}" for c in cols]
    y = [pd.Timestamp(i).strftime("%Y-%m") for i in matrix.index]

    later = z[[c for c in cols if c != 0]]
    finite = later.to_numpy()[~np.isnan(later.to_numpy())]
    zmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    zmax = max(zmax, 1e-6)

    arr = z.to_numpy()
    # Month 0 is forced to the top of the scale so it reads as "the start".
    plot_z = np.where(np.isnan(arr), np.nan,
                      np.where(np.array(cols) == 0, zmax, np.clip(arr, 0, zmax)))

    def _fmt(v):
        if np.isnan(v):
            return ""
        if v == 0:
            return "0"
        if v < 0.0001:
            return "<0.01%"
        return f"{v * 100:.2f}%"

    text = np.vectorize(_fmt)(arr)

    colorscale = [[0, "#FBFAF7"], [0.05, "#E9EFEC"], [0.28, "#C3D6CE"],
                  [0.62, "#6F9587"], [1, "#1F4D3F"]]

    fig = go.Figure(go.Heatmap(
        z=plot_z, x=labels, y=y, text=text, texttemplate="%{text}",
        textfont=dict(size=10, color=P["slate"]),
        colorscale=colorscale,
        zmin=0, zmax=zmax,
        hovertemplate="Cohort %{y}<br>%{x} after first purchase<br>"
                      "Retention %{text}<extra></extra>",
        colorbar=dict(title=dict(text="Retention<br>(M1+)", side="top"),
                      thickness=8, len=0.55, outlinewidth=0,
                      tickformat=".2%", tickvals=[0, zmax/2, zmax],
                      tickfont=dict(size=10, color=P["muted"]),
                      ticks=""),
        xgap=2, ygap=2,
    ))
    fig = style(fig, "Cohort retention heatmap — share of the cohort ordering in each specific month",
                height=max(300, 24 * len(matrix) + 110), show_legend=False)
    fig.update_yaxes(autorange="reversed", title="First-purchase cohort")
    fig.update_xaxes(title="Months since first purchase")
    return fig


def retention_curves(curves: pd.DataFrame, highlight: Optional[Sequence[pd.Timestamp]] = None) -> go.Figure:
    fig = go.Figure()
    cohorts = sorted(curves["cohort_month"].unique())
    for c in cohorts:
        sub = curves[curves["cohort_month"] == c].sort_values("month_index")
        is_hi = highlight is not None and pd.Timestamp(c) in set(pd.Timestamp(h) for h in highlight)
        fig.add_trace(go.Scatter(
            x=sub["month_index"], y=sub["retention_rate"], mode="lines+markers",
            name=pd.Timestamp(c).strftime("%b %Y"),
            line=dict(width=2.2 if is_hi else 1.1,
                      color=P["primary"] if is_hi else "#CFCCC4"),
            marker=dict(size=4.5 if is_hi else 0),
            opacity=1.0 if is_hi else 1.0,
            hovertemplate=f"{pd.Timestamp(c).strftime('%Y-%m')}<br>Month %{{x}}<br>"
                          "%{y:.3%}<extra></extra>",
        ))
    fig = style(fig, "Retention curves by cohort (observable months only)",
                x_title="Months since first purchase", y_title="Share of cohort active")
    fig.update_yaxes(tickformat=".2%")
    return fig


def pooled_retention(curves: pd.DataFrame, weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weights["month_index"], y=weights["retention_rate"],
        marker_color=P["primary"], name="Pooled cohort retention",
        hovertemplate="Month %{x}<br>%{y:.3%}<extra></extra>",
    ))
    fig = style(fig, "Pooled retention by month since first purchase",
                x_title="Months since first purchase", y_title="Retention", show_legend=False)
    fig.update_yaxes(tickformat=".2%")
    return fig


def time_to_second(second_purchase: pd.DataFrame, cutoff: int = 180) -> go.Figure:
    s = pd.to_numeric(second_purchase["days_to_second_order"], errors="coerce").dropna()
    s = s[s <= cutoff]
    fig = go.Figure(go.Histogram(x=s, nbinsx=60, marker_color=P["teal"], opacity=0.85,
                                 hovertemplate="%{x} days<br>%{y:,} customers<extra></extra>"))
    fig = style(fig, f"Days from first to second eligible order (repeaters, ≤{cutoff} days shown)",
                x_title="Days", y_title="Customers")
    return fig


def cohort_size_bars(sizes: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=sizes["cohort_month"], y=sizes["cohort_size"],
                           marker_color=P["primary"], opacity=0.9,
                           hovertemplate="%{x|%b %Y}<br>%{y:,} customers<extra></extra>"))
    return style(fig, "First-purchase cohort sizes", y_title="Customers", show_legend=False)


# ---------------------------------------------------------------------------
# Segmentation charts
# ---------------------------------------------------------------------------

def segment_value_chart(profile: pd.DataFrame) -> go.Figure:
    colors = [settings.SEGMENT_COLORS.get(s, P["muted"]) for s in profile["segment"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=profile["share_customers"], y=profile["segment"], orientation="h",
                         name="Share of customers", marker_color=P["muted"],
                         hovertemplate="%{y}<br>%{x:.2%} of customers<extra></extra>"))
    fig.add_trace(go.Bar(x=profile["share_value"], y=profile["segment"], orientation="h",
                         name="Share of merchandise value", marker_color=colors,
                         hovertemplate="%{y}<br>%{x:.2%} of value<extra></extra>"))
    fig.update_layout(barmode="group")
    fig = style(fig, "Segment share of customers vs share of historical merchandise value",
                x_title="Share", height=420)
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return fig


def segment_scatter(df: pd.DataFrame, sample: int = 6000,
                    lapsing_days: Optional[float] = None,
                    inactive_days: Optional[float] = None) -> go.Figure:
    """
    Recency vs historical value, one dot per customer, coloured by segment.

    The two vertical rules are the segment thresholds actually in force, so the
    chart shows the rule that produced the colours rather than hiding it.
    """
    d = df.sample(min(sample, len(df)), random_state=42)
    fig = px.scatter(
        d, x="recency_days", y="monetary_value", color="segment",
        color_discrete_map=settings.SEGMENT_COLORS,
        hover_data={"n_orders": True, "recency_days": ":,.0f", "monetary_value": ":,.2f"},
        opacity=0.45,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_yaxes(type="log", title="Historical merchandise value (R$, log scale)",
                     dtick=1, minor=dict(ticks="", showgrid=False), tickformat=",.0f")
    fig.update_xaxes(title="Recency — days since last eligible order")
    for value, label in ((lapsing_days, "lapsing"), (inactive_days, "inactive")):
        if value:
            fig.add_vline(x=float(value), line=dict(color=P["muted"], width=1, dash="dot"),
                          annotation_text=f"{label} {value:.0f} d",
                          annotation_position="top", annotation_font_size=10,
                          annotation_font_color=P["muted"])
    fig = style(fig, "", height=340)
    # legend below the axis title so it cannot collide with the threshold labels
    fig.update_layout(legend=dict(title=None, orientation="h", y=-0.34, x=0,
                                  font=dict(size=10)),
                      margin=dict(t=34, b=96))
    return fig


def segment_metric_bars(profile: pd.DataFrame, metric: str, title: str, y_title: str) -> go.Figure:
    colors = [settings.SEGMENT_COLORS.get(s, P["muted"]) for s in profile["segment"]]
    fig = go.Figure(go.Bar(x=profile["segment"], y=profile[metric], marker_color=colors,
                           hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
    fig = style(fig, title, y_title=y_title, show_legend=False)
    fig.update_xaxes(tickangle=-20)
    return fig


# ---------------------------------------------------------------------------
# Customer experience charts
# ---------------------------------------------------------------------------

def score_distribution(dist: Dict[int, int]) -> go.Figure:
    ks = sorted(dist)
    fig = go.Figure(go.Bar(x=ks, y=[dist[k] for k in ks],
                           marker_color=[P["rose"] if k <= 2 else (P["amber"] if k == 3 else P["teal"]) for k in ks],
                           hovertemplate="Score %{x}<br>%{y:,} orders<extra></extra>"))
    return style(fig, "Review score distribution (eligible orders)",
                 x_title="Review score", y_title="Orders", show_legend=False)


def grouped_rate_chart(df: pd.DataFrame, group: str, title: str, x_title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[group].astype(str), y=df["repeat_rate"],
        error_y=dict(type="data",
                     array=(df["ci_high"] - df["repeat_rate"]).clip(lower=0),
                     arrayminus=(df["repeat_rate"] - df["ci_low"]).clip(lower=0),
                     color=P["muted"], thickness=1),
        marker_color=P["primary"], opacity=0.9,
        customdata=np.stack([df["n_orders"], df["n_repeat"]], axis=-1),
        hovertemplate="%{x}<br>Repeat rate %{y:.3%}<br>n = %{customdata[0]:,}"
                      "<br>repeats = %{customdata[1]:,}<extra></extra>",
    ))
    fig = style(fig, title, x_title=x_title, y_title="Share repeating within window",
                show_legend=False)
    fig.update_yaxes(tickformat=".2%")
    return fig


def score_by_lateness(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["late_bin"].astype(str), y=df["mean_review"],
                         marker_color=P["teal"], opacity=0.9,
                         customdata=df[["n_orders"]].to_numpy(),
                         hovertemplate="%{x}<br>Mean score %{y:.2f}<br>n = %{customdata[0]:,}"
                                       "<extra></extra>"))
    fig = style(fig, "Mean review score by delivery lateness", y_title="Mean review score",
                show_legend=False)
    fig.update_yaxes(range=[1, 5])
    return fig
