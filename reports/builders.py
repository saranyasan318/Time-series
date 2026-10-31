"""Downloadable report builders (CSV / Markdown / PNG / PDF). All content derives from the run results;
nothing is templated text with invented numbers. matplotlib draws the static charts (no browser/kaleido needed)."""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from business_insights.insights import forecast_kpis, limitations

NAVY, ACC, LAV = "#0B1B3A", "#5B6EE1", "#C9B8F0"


# ------------------------------------------------------------------------------------------------- CSV
def csv_bytes(df, index=False): return df.to_csv(index=index).encode("utf-8")
def forecast_csv(fc): return csv_bytes(fc)
def actual_vs_pred_csv(pred_df): return csv_bytes(pred_df)
def processed_csv(series): return csv_bytes(series.rename(series.name or "value").rename_axis("date").reset_index())


def comparison_csv(cv, hold):
    return csv_bytes(cv.merge(hold, on="Model", suffixes=(" (CV)", " (hold-out)"), how="left"))


def all_forecasts_csv(fcs):
    frames = [df.assign(model=m) for m, df in fcs.items()]
    return csv_bytes(pd.concat(frames, ignore_index=True)) if frames else b""


# ------------------------------------------------------------------------------------------------- Markdown
def _table(df, nd=3):
    d = df.round(nd)
    try:
        return d.to_markdown(index=False)
    except ImportError:
        return d.to_string(index=False)


def evaluation_report_md(ev, horizon, freq, n_obs, season=1):
    t0, t1 = ev["train_period"]; s0, s1 = ev["test_period"]
    sel = ev["selected"]
    row = ev["holdout"][ev["holdout"]["Model"] == sel].iloc[0]
    lines = ["# Model Evaluation Report", "",
             f"- Selected model: **{sel}** (best cross-validated {ev.get('primary_metric', 'RMSE')})",
             f"- Training/validation period: {t0:%Y-%m-%d} to {t1:%Y-%m-%d}",
             f"- Hold-out test period: {s0:%Y-%m-%d} to {s1:%Y-%m-%d} ({ev['test_h']} periods)",
             f"- Forecast horizon requested: {horizon} periods ({freq})",
             f"- Cross-validation: {ev['cv_mode']}, {len(ev['cv_folds'])} folds, {ev['cv_h']} periods per fold",
             "", "## Unbiased final evaluation (selected model, hold-out)",
             f"MAE {row['MAE']:.2f} | RMSE {row['RMSE']:.2f} | MAPE {row['MAPE']:.2f}% | sMAPE {row['sMAPE']:.2f}% | WAPE {row['WAPE']:.2f}% | MASE {row['MASE']:.2f}",
             "", "## Cross-validation ranking (used for selection)", _table(ev["cv"]),
             "", "## Hold-out scores (informational for non-selected models)", _table(ev["holdout"]),
             "", "## Procedure",
             "Hold-out data was never used for model selection. Models were ranked by chronological cross-validation on earlier data; "
             "only then was the hold-out scored once. Pre-processing was fitted on training windows only.",
             "", "## Limitations"] + [f"- {x}" for x in limitations(n_obs, season, sel)]
    if ev["errors"]:
        lines += ["", "## Models that could not be evaluated"] + [f"- {k}: {v}" for k, v in ev["errors"].items()]
    return "\n".join(lines)


def insights_report_md(ins, model, horizon, fc):
    lines = ["# Business Insights Report", "", f"Model: **{model}** | Horizon: {horizon} periods | "
             f"Forecast window: {fc['date'].min():%Y-%m-%d} to {fc['date'].max():%Y-%m-%d}", ""]
    for _, r in ins.iterrows():
        lines.append(f"## {r['Insight']} ({r['Category']})\n{r['Recommendation']}" + (f"\n\n_Evidence: {r['Evidence']}_" if r["Evidence"] else "") + "\n")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- charts (matplotlib)
def _fig_bytes(fig):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=150); plt.close(fig)
    return buf.getvalue()


def _clean_ax(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(alpha=.25)


def forecast_png(hist, fc, title="Forecast", level=0.95):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist.index, hist.values, color=NAVY, lw=1.8, label="Actual")
    ax.plot([hist.index[-1], fc["date"].iloc[0]], [hist.values[-1], fc["forecast"].iloc[0]], color=ACC, lw=1, ls=":")
    ax.plot(fc["date"], fc["forecast"], color=ACC, lw=2, ls="--", label="Forecast")
    ax.fill_between(fc["date"], fc["lower"], fc["upper"], color=LAV, alpha=.45, label=f"{level:.0%} prediction interval")
    ax.set_title(title); ax.legend(frameon=False); _clean_ax(ax)
    return _fig_bytes(fig)


def actual_vs_pred_png(ev, model):
    p = ev["predictions"]; lo, hi = ev["bands"][model]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p["date"], p["Actual"], color=NAVY, lw=2.2, marker="o", label="Actual (hold-out)")
    ax.plot(p["date"], p[model], color=ACC, lw=2, ls="--", marker="s", label=model)
    ax.fill_between(p["date"], lo, hi, color=LAV, alpha=.35, label="Interval")
    ax.set_title(f"Hold-out: actual vs predicted ({model})"); ax.legend(frameon=False); _clean_ax(ax)
    return _fig_bytes(fig)


def comparison_png(ev, metric="RMSE", top=10):
    cv = ev["cv"].head(top)
    h = ev["holdout"].set_index("Model").reindex(cv["Model"])
    x = np.arange(len(cv)); w = .38
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w / 2, cv[metric], w, color="#A9D6F5", label=f"CV {metric}")
    ax.bar(x + w / 2, h[metric], w, color=LAV, label=f"Hold-out {metric}")
    ax.set_xticks(x); ax.set_xticklabels(cv["Model"], rotation=30, ha="right"); ax.legend(frameon=False)
    ax.set_title(f"Model comparison ({metric}, lower is better)"); _clean_ax(ax)
    return _fig_bytes(fig)


def residual_png(ev, model):
    r = ev["residuals"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    a.bar(r["date"], r[model], width=20, color=[("#2E9E6B" if v >= 0 else "#D9534F") for v in r[model]]); a.axhline(0, color="grey", lw=.8)
    a.set_title("Residuals over time"); _clean_ax(a)
    b.hist(r[model], bins=min(12, max(4, len(r) // 2)), color=ACC); b.set_title("Residual distribution"); _clean_ax(b)
    return _fig_bytes(fig)


def history_png(s, title="Historical series"):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(s.index, s.values, color=NAVY, lw=1.8); ax.set_title(title); _clean_ax(ax)
    return _fig_bytes(fig)


# ------------------------------------------------------------------------------------------------- PDF
def _df_table(df, cols, col_widths=None, nd=2, max_rows=12, font=7.5):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    d = df[cols].head(max_rows)
    data = [cols] + [[(f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("-" if isinstance(v, float) else str(v))) for v in row] for row in d.values]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), font), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F8")]),
                           ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#D5D9E6")), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def build_pdf_report(*, series, processed, fc, ev, ins, model, horizon, freq, season, quality=None, prep_log=None,
                     title="Time Series Forecasting Report", dataset_name="dataset", level=0.95):
    """Complete forecasting report as PDF bytes. Every number comes from the arguments."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor(NAVY), fontSize=20, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor(ACC), fontSize=13, spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9, leading=12.5)
    SM = ParagraphStyle("SM", parent=P, fontSize=7.5, textColor=colors.HexColor("#6B7694"), leading=10)

    def img(png, w=17):
        im = Image(io.BytesIO(png)); r = im.imageHeight / im.imageWidth
        im.drawWidth, im.drawHeight = w * cm, w * cm * r
        return im

    row = ev["holdout"].set_index("Model").loc[model]
    kp = forecast_kpis(processed, fc, level, freq)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title=title, author="Time Series Intelligence")
    f = [Paragraph(title, H1), Paragraph(f"Dataset: {dataset_name} &nbsp;|&nbsp; Generated {datetime.now():%Y-%m-%d %H:%M} &nbsp;|&nbsp; Frequency {freq}, {len(series)} periods "
                                         f"({series.index.min():%Y-%m-%d} to {series.index.max():%Y-%m-%d})", SM), Spacer(1, 8)]

    kpis = [["Selected model", model], ["Forecast horizon", f"{horizon} periods ({fc['date'].min():%Y-%m-%d} to {fc['date'].max():%Y-%m-%d})"],
            ["Forecast total (central)", f"{kp['forecast_total']:,.0f}  ({level:.0%} range {kp['total_lower']:,.0f} to {kp['total_upper']:,.0f})"],
            ["Change vs " + kp["basis"], f"{kp['total_change_%']:+.1f}%"],
            ["Peak / lowest period", f"{kp['peak_date']:%b %Y} ({kp['peak_value']:,.0f})  /  {kp['low_date']:%b %Y} ({kp['low_value']:,.0f})"],
            ["Hold-out accuracy", f"MAPE {row['MAPE']:.2f}%  |  RMSE {row['RMSE']:,.1f}  |  MASE {row['MASE']:.2f}"]]
    cell = ParagraphStyle("cell", parent=P, fontSize=8.5, leading=11)
    kpis = [[Paragraph(str(a), cell), Paragraph(str(b), cell)] for a, b in kpis]   # wrap long labels instead of overprinting
    t = Table(kpis, colWidths=[5.5 * cm, 11.5 * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F8")),
                           ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#D5D9E6")),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    f += [Paragraph("Executive summary", H2), t, Spacer(1, 6), img(forecast_png(processed.dropna(), fc, f"{model} forecast", level))]

    if quality:
        f += [Paragraph("Data quality", H2), Paragraph(f"Quality score <b>{quality['score']:.0f}/100</b>. " +
              ("Issues found: " + "; ".join(quality["issues"]) + "." if quality["issues"] else "No quality issues detected."), P)]
    if prep_log:
        f.append(Paragraph("Preparation applied: " + " | ".join(prep_log), SM))

    f += [PageBreak(), Paragraph("Model evaluation", H2),
          Paragraph(f"Models were ranked by <b>{ev['cv_mode']}</b> cross-validation ({len(ev['cv_folds'])} folds) on data before the hold-out "
                    f"({ev['train_period'][0]:%Y-%m-%d} to {ev['train_period'][1]:%Y-%m-%d}) using {ev.get('primary_metric', 'RMSE')}. The hold-out "
                    f"({ev['test_period'][0]:%Y-%m-%d} to {ev['test_period'][1]:%Y-%m-%d}, {ev['test_h']} periods) was scored once, after selection, "
                    f"so the selected model's hold-out score is an unbiased estimate. Training accuracy was never used for selection.", P), Spacer(1, 4)]
    cols = [c for c in ["Rank", "Model", "MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE"] if c in ev["cv"].columns]
    f += [Paragraph("Cross-validation ranking (selection basis)", SM), _df_table(ev["cv"], cols, nd=2), Spacer(1, 6)]
    hc = [c for c in ["Model", "MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE", "R2"] if c in ev["holdout"].columns]
    f += [Paragraph("Hold-out scores (final evaluation; informational for non-selected models)", SM), _df_table(ev["holdout"], hc, nd=2), Spacer(1, 8),
          img(comparison_png(ev, ev.get("primary_metric", "RMSE")), 15.5), img(actual_vs_pred_png(ev, model), 15.5), img(residual_png(ev, model), 15.5)]
    if ev["errors"]:
        f += [Paragraph("Models that could not be evaluated", H2)] + [Paragraph(f"&bull; <b>{k}</b>: {v}", SM) for k, v in ev["errors"].items()]
    cov = ev.get("coverage", {}).get(model)
    if cov is not None:
        f.append(Paragraph(f"Empirical coverage of the {level:.0%} interval on the hold-out: <b>{cov:.0%}</b> ({ev['test_h']} points - a coarse check).", SM))

    f += [PageBreak(), Paragraph("Forecast table", H2)]
    fcd = fc.assign(date=fc["date"].dt.strftime("%Y-%m-%d"))
    f.append(_df_table(fcd, ["date", "forecast", "lower", "upper"], nd=1, max_rows=36, font=8))
    f += [Paragraph("Business insights", H2)]
    for _, r in ins.iterrows():
        f.append(Paragraph(f"<b>{r['Insight']}</b> <font color='#6B7694'>[{r['Category']}]</font> - {r['Recommendation']}", P))
        if r["Evidence"]:
            f.append(Paragraph(f"Evidence: {r['Evidence']}", SM))
        f.append(Spacer(1, 3))
    f += [Paragraph("Limitations", H2)] + [Paragraph(f"&bull; {x}", P) for x in limitations(len(series), season, model)]
    doc.build(f)
    return buf.getvalue()
