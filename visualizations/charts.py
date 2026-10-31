
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import THEME

NAVY, BLUE, LAV, ACC = THEME["navy"], THEME["baby_blue"], THEME["lavender"], THEME["accent"]
GOOD, BAD, MUTED = "#2E9E6B", "#D9534F", THEME["muted"]
PALETTE = [ACC, "#E08FB0", "#4FB0A8", "#E0A84F", "#7A8CF0", "#9B7BD9", "#5BA3E0", "#C0804F"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _style(fig, title=None, h=380, legend_below=True):
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16)) if title else None, template="plotly_white", height=h,
        margin=dict(l=10, r=10, t=55 if title else 20, b=10), font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY),
        legend=dict(orientation="h", y=-0.18 if legend_below else 1.02, x=0), paper_bgcolor="white", plot_bgcolor="white",
        hovermode="x unified", autosize=True)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#EEF0F6", zeroline=False)
    return fig


def _vmarker(fig, x, text=None):
    """Vertical dashed marker (shape + annotation; avoids add_vline's datetime pitfalls)."""
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper", line=dict(color=MUTED, width=1, dash="dot"))
    if text:
        fig.add_annotation(x=x, y=1, xref="x", yref="paper", text=text, showarrow=False, yanchor="bottom", font=dict(size=11, color=MUTED))

def history_chart(s, rolling=None, title="Historical series"):
    f = go.Figure(go.Scatter(x=s.index, y=s.values, name="Actual", mode="lines", line=dict(color=NAVY, width=2)))
    if rolling is not None:
        f.add_trace(go.Scatter(x=rolling.index, y=rolling.values, name="Rolling mean", line=dict(color=ACC, dash="dash", width=2)))
    return _style(f, title)


def before_after_chart(before, after, title="Before vs after preparation"):
    f = go.Figure()
    f.add_trace(go.Scatter(x=before.index, y=before.values, name="Original", mode="lines+markers", line=dict(color="#B8BFD6", width=1.5), marker=dict(size=4)))
    f.add_trace(go.Scatter(x=after.index, y=after.values, name="Prepared", line=dict(color=ACC, width=2.2)))
    changed = after.index.intersection(before.index)
    diff = changed[(before.reindex(changed).fillna(-1e18).values != after.reindex(changed).values)]
    if len(diff):
        f.add_trace(go.Scatter(x=diff, y=after.reindex(diff).values, mode="markers", name="Modified points", marker=dict(color=BAD, size=8, symbol="diamond")))
    return _style(f, title)


def rolling_std_chart(df):
    return _style(go.Figure(go.Scatter(x=df.index, y=df["rolling_std"], line=dict(color=LAV, width=2), fill="tozeroy", name="Rolling std")), "Rolling standard deviation", 280)


def pct_change_chart(df):
    colors = [GOOD if (v == v and v >= 0) else BAD for v in df["pct_change"]]
    return _style(go.Figure(go.Bar(x=df.index, y=df["pct_change"], marker_color=colors, name="% change")), "Period-over-period change (%)", 280)


def decomposition_chart(dec):
    cols = list(dec.columns)
    f = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=[c.title() for c in cols], vertical_spacing=0.06)
    for i, c in enumerate(cols, 1):
        f.add_trace(go.Scatter(x=dec.index, y=dec[c], line=dict(color=[NAVY, ACC, "#4FB0A8", "#9AA3BF"][(i - 1) % 4], width=1.8), showlegend=False, name=c), row=i, col=1)
    return _style(f, "Trend and seasonal decomposition", 160 * len(cols) + 80)


def seasonal_bar(profile, labels=None):
    x = labels or [str(i) for i in profile.index]
    colors = [ACC if v >= 1 else LAV for v in profile.values]
    f = go.Figure(go.Bar(x=x, y=profile.values, marker_color=colors, name="Seasonal index"))
    f.add_hline(y=1.0, line_dash="dot", line_color=MUTED)
    return _style(f, "Seasonal profile (1.0 = overall average)", 300)


def acf_chart(vals, n, title):
    f = go.Figure(go.Bar(x=list(range(len(vals))), y=vals, marker_color=ACC, name=title))
    b = 1.96 / np.sqrt(n)
    f.add_hline(y=b, line_dash="dot", line_color="grey"); f.add_hline(y=-b, line_dash="dot", line_color="grey")
    return _style(f, title, 280)


def monthly_pattern_chart(s):
    """Box plot of values per calendar month (shows level, spread and outliers per month)."""
    y = s.dropna()
    f = go.Figure()
    for m in range(1, 13):
        v = y[y.index.month == m]
        if len(v):
            f.add_trace(go.Box(y=v.values, name=MONTHS[m - 1], marker_color=ACC, boxmean=True, showlegend=False))
    return _style(f, "Monthly pattern (distribution per calendar month)", 340)


def quarterly_pattern_chart(s):
    y = s.dropna()
    g = y.groupby(y.index.quarter).mean()
    f = go.Figure(go.Bar(x=[f"Q{q}" for q in g.index], y=g.values, marker_color=[ACC, LAV, BLUE, "#4FB0A8"][:len(g)], name="Average"))
    return _style(f, "Quarterly pattern (average per quarter)", 300)


def yearly_pattern_chart(s):
    y = s.dropna()
    g = y.groupby(y.index.year).sum()
    gr = g.pct_change() * 100
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Bar(x=[str(i) for i in g.index], y=g.values, marker_color=LAV, name="Yearly total"), secondary_y=False)
    f.add_trace(go.Scatter(x=[str(i) for i in g.index], y=gr.values, mode="lines+markers", name="YoY growth (%)", line=dict(color=ACC, width=2.5)), secondary_y=True)
    f.update_yaxes(title_text="YoY growth (%)", secondary_y=True, showgrid=False)
    return _style(f, "Yearly totals and year-over-year growth", 320)


def heatmap_chart(s):
    """Year x month heatmap of average values (needs >= 2 years)."""
    y = s.dropna()
    t = y.groupby([y.index.year, y.index.month]).mean().unstack(1).reindex(columns=range(1, 13))
    f = go.Figure(go.Heatmap(z=t.values, x=MONTHS, y=[str(i) for i in t.index], colorscale=[[0, "#EEF3FF"], [1, NAVY]], colorbar=dict(title="avg")))
    return _style(f, "Year x month heatmap", 320, legend_below=False)


def quality_gauge(score):
    color = GOOD if score >= 85 else "#E0A84F" if score >= 65 else BAD
    f = go.Figure(go.Indicator(mode="gauge+number", value=float(score), number=dict(suffix=" / 100"),
                               gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color), steps=[dict(range=[0, 65], color="#FBEAEA"), dict(range=[65, 85], color="#FFF6E5"), dict(range=[85, 100], color="#E9F7F0")])))
    f.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="white", font=dict(color=NAVY))
    return f



def forecast_chart(hist, fc, title="Forecast", level=0.95, history_tail=None):
    h = hist.dropna() if history_tail is None else hist.dropna().tail(history_tail)
    f = go.Figure()
    f.add_trace(go.Scatter(x=h.index, y=h.values, name="Actual", line=dict(color=NAVY, width=2)))
    f.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f.add_trace(go.Scatter(x=fc["date"], y=fc["lower"], fill="tonexty", fillcolor="rgba(201,184,240,.45)", line=dict(width=0), name=f"{level:.0%} interval"))
    link_x, link_y = [h.index[-1], fc["date"].iloc[0]], [h.values[-1], fc["forecast"].iloc[0]]
    f.add_trace(go.Scatter(x=link_x, y=link_y, showlegend=False, hoverinfo="skip", line=dict(color=ACC, width=1.5, dash="dot")))
    f.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast", mode="lines+markers", line=dict(color=ACC, width=2.5, dash="dash"), marker=dict(size=5)))
    _vmarker(f, h.index[-1], "forecast start")
    return _style(f, title, 430)


def actual_vs_pred_chart(pred_df, models, band=None, history=None):
    f = go.Figure()
    if history is not None:
        f.add_trace(go.Scatter(x=history.index, y=history.values, name="History", line=dict(color="#C5CBE0")))
    f.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["Actual"], name="Actual (hold-out)", line=dict(color=NAVY, width=2.5)))
    for i, m in enumerate(models):
        f.add_trace(go.Scatter(x=pred_df["date"], y=pred_df[m], name=m, mode="lines+markers", line=dict(width=2, dash="dash", color=PALETTE[i % len(PALETTE)])))
    if band is not None:
        f.add_trace(go.Scatter(x=pred_df["date"], y=band[1], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=pred_df["date"], y=band[0], fill="tonexty", fillcolor="rgba(201,184,240,.35)", line=dict(width=0), name="Interval"))
    return _style(f, "Actual vs predicted (hold-out period)", 420)


def actual_vs_pred_scatter(pred_df, model):
    a, p = pred_df["Actual"].values, pred_df[model].values
    lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
    f = go.Figure()
    f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect forecast", line=dict(color=MUTED, dash="dot")))
    f.add_trace(go.Scatter(x=a, y=p, mode="markers", name=model, marker=dict(color=ACC, size=9, line=dict(color="white", width=1)),
                           text=[f"{d:%b %Y}" for d in pred_df["date"]], hovertemplate="%{text}<br>actual %{x:,.0f}<br>predicted %{y:,.0f}<extra></extra>"))
    f.update_xaxes(title_text="Actual"); f.update_yaxes(title_text="Predicted")
    f.update_layout(hovermode="closest")
    return _style(f, f"Actual vs predicted - {model}", 380)


def comparison_chart(cv, hold, metric="RMSE"):
    f = go.Figure()
    f.add_trace(go.Bar(x=cv["Model"], y=cv[metric], name=f"CV {metric}", marker_color=BLUE))
    h = hold.set_index("Model").reindex(cv["Model"])
    f.add_trace(go.Bar(x=h.index, y=h[metric], name=f"Hold-out {metric}", marker_color=LAV))
    f.update_layout(barmode="group")
    f.update_xaxes(tickangle=-30)
    return _style(f, f"Model comparison ({metric}; lower is better)", 380)


def fold_heatmap(fold_details, metric="RMSE"):
    t = fold_details.pivot(index="Model", columns="Fold", values=metric)
    f = go.Figure(go.Heatmap(z=t.values, x=[f"Fold {c}" for c in t.columns], y=list(t.index), colorscale=[[0, "#EEF3FF"], [1, ACC]], colorbar=dict(title=metric)))
    return _style(f, f"{metric} by model and validation fold", 40 * len(t) + 120, legend_below=False)


def residual_chart(res_df, model):
    r = res_df[model]
    f = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=("Residuals over time", "Residual distribution"))
    f.add_trace(go.Bar(x=res_df["date"], y=r, marker_color=[GOOD if v >= 0 else BAD for v in r], name="Residual (actual - predicted)"), row=1, col=1)
    f.add_trace(go.Histogram(y=r, marker_color=ACC, nbinsy=min(15, max(4, len(r) // 2)), name="Distribution", showlegend=False), row=1, col=2)
    f.add_hline(y=0, line_dash="dot", line_color=MUTED, row=1, col=1)
    f.update_layout(hovermode="closest")
    return _style(f, f"Residual analysis - {model}", 340)


def error_distribution_chart(res_df, models):
    f = go.Figure()
    for i, m in enumerate(models):
        f.add_trace(go.Box(y=res_df[m].values, name=m, marker_color=PALETTE[i % len(PALETTE)], boxmean=True))
    f.add_hline(y=0, line_dash="dot", line_color=MUTED)
    f.update_layout(hovermode="closest")
    return _style(f, "Hold-out error distribution by model", 360)


def forecast_growth_chart(hist, fc):
    """Period-over-period change of the forecast, starting from the last actual."""
    vals = np.concatenate([[hist.dropna().values[-1]], fc["forecast"].values])
    pc = pd.Series(vals).pct_change().dropna().values * 100
    f = go.Figure(go.Bar(x=fc["date"], y=pc, marker_color=[GOOD if v >= 0 else BAD for v in pc], name="Forecast change (%)"))
    return _style(f, "Forecast growth and decline (period over period, %)", 300)


def uncertainty_chart(fc):
    width = (fc["upper"] - fc["lower"]) / fc["forecast"].abs().replace(0, np.nan) * 100
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_trace(go.Bar(x=fc["date"], y=fc["upper"] - fc["lower"], marker_color=LAV, name="Interval width"), secondary_y=False)
    f.add_trace(go.Scatter(x=fc["date"], y=width, mode="lines+markers", line=dict(color=ACC, width=2.5), name="Width (% of forecast)"), secondary_y=True)
    f.update_yaxes(title_text="% of forecast", secondary_y=True, showgrid=False)
    return _style(f, "Prediction uncertainty grows with the horizon", 320)


def model_forecasts_chart(hist, fcs, tail=36):
    """Overlay of the future forecast of several models."""
    h = hist.dropna().tail(tail)
    f = go.Figure(go.Scatter(x=h.index, y=h.values, name="Actual", line=dict(color=NAVY, width=2)))
    for i, (m, fc) in enumerate(fcs.items()):
        f.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name=m, line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash="dash")))
    return _style(f, "Future forecasts by model", 420)
