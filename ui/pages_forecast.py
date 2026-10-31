
import numpy as np
import pandas as pd
import streamlit as st

from business_insights.insights import forecast_kpis, generate_insights, limitations
from config import CV_MODES, HORIZONS, INTERVAL_LEVELS, RANK_METRICS
from evaluation.selection import evaluate_models
from forecasting.models import MODEL_FAMILY, model_availability
from forecasting.selector import length_class, recommend_models
from models.store import save_run
from preprocessing.cleaning import data_quality_report
from reports import builders as rb
from ui import state as stt
from ui.components import download, fmt, header, kpi, kpi_row, need_data, need_results, note, plot, table
from ui.state import S
from visualizations import charts as ch

def lab():
    header("Forecasting Lab", "Models are chosen for your data, ranked on chronological validation, then refit on all history.")
    s0 = need_data(); f, m, n = S["freq"], stt.season(), len(S["series"])
    s_prep, _ = stt.prepped()
    if s_prep.isna().any():
        st.error("Missing values remain after preparation. Choose a fill method in **Data Preparation** first."); st.stop()
    info = stt.cached_characterise(s_prep, f, m)
    rec, excluded, cls = recommend_models(n, m, f, info["seasonal_flag"], all_positive=bool((s_prep > 0).all()))
    kpi_row([("History", f"{n} periods", f"{cls} series (season length {m})"), ("Seasonality", "yes" if info["seasonal_flag"] else "weak / none", f"strength {fmt(info['seasonal_strength'])}"),
             ("Trend", info["trend"]["direction"], f"strength {fmt(info['trend_strength'])}"), ("Candidate models", len(rec), f"{len(excluded)} excluded")])
    st.write("")
    hz = HORIZONS[f]; labels = list(hz) + ["Custom"]
    a, b, c = st.columns(3)
    hl = a.selectbox("Forecast horizon", labels, index=min(1, len(hz) - 1))
    if hl == "Custom":
        horizon = int(a.number_input("Periods ahead", 1, max(2, min(365, n // 2)), min(12, max(2, n // 4))))
    else:
        horizon = hz[hl]
    mode = b.selectbox("Validation scheme", CV_MODES, help="expanding = growing window; rolling = fixed-size window; walk-forward = one-step-ahead refits")
    nsp = c.slider("Validation folds", 2, 8, 3)
    d, e, g = st.columns(3)
    lv_l = d.selectbox("Prediction interval", list(INTERVAL_LEVELS), index=2)
    pm = e.selectbox("Ranking metric", RANK_METRICS, index=RANK_METRICS.index("RMSE"))
    fam = g.multiselect("Show model families", sorted(set(MODEL_FAMILY.values())), default=sorted(set(MODEL_FAMILY.values())))
    avail = model_availability()
    options = [k for k in avail if avail[k][0] and MODEL_FAMILY[k] in fam]
    models = st.multiselect("Models", options, default=[x for x in rec if x in options],
                            help="Pre-selected from your dataset's length, seasonality and sign. Failures are reported, never hidden.")
    with st.expander(f"Why some models are not pre-selected ({len(excluded)})"):
        for k, v in excluded.items(): st.write(f"• **{k}** - {v}")
    if n < 36: st.info("Short history: complex models are unstable here, so expect baselines and smoothing models to compete well.")
    if st.button("Train, validate and forecast", type="primary", key="lab_run"):
        if not models: st.error("Select at least one model."); st.stop()
        bar = st.progress(0.0, text="Starting...")
        try:
            ev = evaluate_models(S["series"], f, m, S["cfg"], models, horizon, nsp, mode, INTERVAL_LEVELS[lv_l], pm,
                                 progress=lambda p, t: bar.progress(min(float(p) * 0.8, 0.8), text=t))
        except Exception as ex:
            bar.empty(); st.error(f"Evaluation failed: {ex}"); st.stop()
        fcs, metas = {}, {}
        for i, name in enumerate(ev["cv"]["Model"]):
            bar.progress(0.8 + 0.2 * i / max(len(ev["cv"]), 1), text=f"Forecasting the future with {name}")
            try:
                df, mt, _ = stt.cached_future(S["series"], f, m, S["cfg"], name, horizon, INTERVAL_LEVELS[lv_l])
                fcs[name], metas[name] = df, mt
            except Exception as ex:
                ev["errors"][name] = f"future forecast failed - {type(ex).__name__}: {ex}"
        bar.empty()
        if ev["selected"] not in fcs:
            st.error("The selected model could not produce a future forecast."); st.stop()
        S.update(ev=ev, fcs=fcs, meta=metas, horizon=horizon, level=INTERVAL_LEVELS[lv_l])
        try:
            save_run({"selected": ev["selected"], "cv": ev["cv"], "holdout": ev["holdout"], "forecasts": fcs, "meta": metas, "horizon": horizon, "freq": f})
        except Exception as ex:
            st.caption(f"(run artefact not saved: {ex})")
        st.success(f"Done. Best model by {ev['primary_metric']} on {ev['cv_mode']} validation: **{ev['selected']}**.")
    ev = S["ev"]
    if ev:
        for t in ev["notes"]: st.info(t)
        for k_, v_ in ev["errors"].items(): st.warning(f"**{k_}** skipped: {v_}")
        st.write(f"Train/validation: {ev['train_period'][0]:%Y-%m-%d} → {ev['train_period'][1]:%Y-%m-%d}; hold-out: {ev['test_period'][0]:%Y-%m-%d} → {ev['test_period'][1]:%Y-%m-%d}.")
        cols = [c for c in ["Rank", "Model", "RMSE", "MAE", "MAPE", "sMAPE", "WAPE", "MASE", "Fit time (s)"] if c in ev["cv"].columns]
        table(ev["cv"][cols].round(3))
        for mname, mt in S["meta"].items():
            if mname in ("ARIMA", "SARIMA", "Auto ARIMA"): st.caption(f"{mname} configuration (by AIC): {mt}")
        st.button("Go to Model Evaluation & Comparison", on_click=stt.goto, args=("Model Evaluation & Comparison",), key="lab_next")


def evaluation():
    header("Model Evaluation & Comparison", "Selection uses chronological validation only - never training accuracy.")
    s0 = need_data(); ev = need_results(); sel = ev["selected"]
    row = ev["holdout"].set_index("Model").loc[sel]
    kpi_row([(f"{k}" + (" (hold-out)" if k == "MAE" else ""), fmt(row[k], pct=k in ("MAPE", "sMAPE", "WAPE"))) for k in ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE"]])
    note(f"<b>{sel}</b> was selected by {ev['primary_metric']} on {ev['cv_mode']} validation of data <i>before</i> the hold-out. Its hold-out scores above are an unbiased "
         "final estimate; hold-out scores of other models are informational (choosing by them would reintroduce selection bias). "
         "MASE &lt; 1 means better than a naive forecast scaled on the training data.")
    cov = ev["coverage"].get(sel)
    best_b = ev["holdout"][ev["holdout"]["Model"].isin(["Naive", "Seasonal Naive", "Drift"])]
    gain = row.get("RMSE vs best baseline (%)", np.nan)
    if len(best_b) and np.isfinite(gain):
        st.write(f"**Versus best baseline** ({best_b.sort_values('RMSE').iloc[0]['Model']}): {gain:+.1f}% RMSE " + ("improvement." if gain > 0 else "- the selected model does not beat a naive baseline; prefer the simpler model."))
    if cov is not None: st.write(f"**Interval calibration:** {cov:.0%} of the {ev['test_h']} hold-out actuals fall inside the nominal {ev['level']:.0%} interval (a coarse check on few points).")
    t1, t2, t3, t4 = st.tabs(["Rankings", "Predictions", "Residuals & errors", "Validation folds"])
    with t1:
        st.markdown("**Cross-validation ranking (selection basis)**"); table(ev["cv"].round(3))
        st.markdown("**Hold-out scores (final evaluation)**"); table(ev["holdout"].round(3))
        metric = st.selectbox("Comparison metric", ["RMSE", "MAE", "MAPE", "sMAPE", "WAPE", "MASE"], key="ev_metric")
        plot(ch.comparison_chart(ev["cv"], ev["holdout"], metric), key="ev_cmp")
    with t2:
        chosen = st.multiselect("Models to plot", list(ev["bands"]), default=[sel])
        if chosen:
            plot(ch.actual_vs_pred_chart(ev["predictions"], chosen, ev["bands"][chosen[0]] if len(chosen) == 1 else None, s0.loc[:ev["train_period"][1]].tail(36)), key="ev_avp")
            plot(ch.actual_vs_pred_scatter(ev["predictions"], chosen[0]), key="ev_sc")
    with t3:
        pick = st.selectbox("Model", list(ev["bands"]), index=list(ev["bands"]).index(sel), key="ev_res")
        plot(ch.residual_chart(ev["residuals"], pick), key="ev_resid")
        plot(ch.error_distribution_chart(ev["residuals"], list(ev["bands"])[:8]), key="ev_err")
        r = ev["residuals"][pick]
        st.caption(f"Residual mean {r.mean():,.2f} (bias), std {r.std():,.2f}. A mean far from zero suggests systematic over/under-forecasting.")
    with t4:
        fd = ev["fold_details"]
        st.write(f"{len(ev['cv_folds'])} {ev['cv_mode']} folds, {ev['cv_h']} periods each.")
        plot(ch.fold_heatmap(fd, "RMSE"), key="ev_fold")
        table(fd.round({c: 2 for c in fd.select_dtypes('number').columns}))
    st.write("**Limitations**")
    for x in limitations(len(s0), stt.season(), sel): st.write("• " + x)


def prediction():
    header("Prediction Dashboard", "Future forecasts with uncertainty.")
    s0 = need_data(); ev = need_results(); proc, _ = stt.prepped()
    avail = list(S["fcs"]); m = st.sidebar.selectbox("Model", avail, index=avail.index(ev["selected"]))
    hmax = len(S["fcs"][m]); hshow = st.sidebar.slider("Horizon shown (periods)", 1, hmax, hmax)
    tail = st.sidebar.slider("History shown (periods)", 12, max(13, len(proc)), min(60, len(proc)))
    fc = S["fcs"][m].head(hshow); kp = forecast_kpis(proc, fc, ev["level"], S["freq"]); row = ev["holdout"].set_index("Model").loc[m]
    tone = "good" if kp["growth_%"] > 5 else "bad" if kp["growth_%"] < -5 else ""
    kpi_row([("Forecast total", fmt(kp["forecast_total"]), f"{ev['level']:.0%}: {fmt(kp['total_lower'])} - {fmt(kp['total_upper'])}"), ("Growth", f"{kp['growth_%']:+.1f}%", kp["basis"], tone),
             ("Peak", f"{kp['peak_date']:%b %Y}", fmt(kp["peak_value"])), ("Lowest", f"{kp['low_date']:%b %Y}", fmt(kp["low_value"])),
             ("Hold-out MAPE", f"{fmt(row['MAPE'])}%", f"RMSE {fmt(row['RMSE'])}"), ("Best model (CV)", ev["selected"], f"viewing: {m}")])
    st.write("")
    plot(ch.forecast_chart(proc, fc, f"{m}: forecast with {ev['level']:.0%} prediction interval", ev["level"], tail), key="pd_fc")
    st.caption("Intervals are model-based estimates of uncertainty, not guaranteed outcomes. Forecasts are clipped at zero for non-negative targets.")
    c1, c2 = st.columns(2)
    with c1: plot(ch.forecast_growth_chart(proc, fc), key="pd_growth")
    with c2: plot(ch.uncertainty_chart(fc), key="pd_unc")
    others = st.multiselect("Compare future forecasts of other models", [x for x in avail], default=[m] if m == ev["selected"] else [m, ev["selected"]][:2])
    if others: plot(ch.model_forecasts_chart(proc, {k: S["fcs"][k].head(hshow) for k in others}), key="pd_multi")
    table(fc.assign(date=fc["date"].dt.strftime("%Y-%m-%d")).round(2))


def _insights(m):
    ev = S["ev"]; proc, _ = stt.prepped(); row = ev["holdout"].set_index("Model").loc[m]
    return generate_insights(proc, S["fcs"][m], row.to_dict(), S["freq"], m, ev["level"], coverage=ev["coverage"].get(m),
                             baseline_gain=row.get("RMSE vs best baseline (%)", np.nan), mase=row.get("MASE", np.nan))


def insights():
    header("Business Intelligence & Insights", "Every statement below is computed from your history, forecast and validation results.")
    s0 = need_data(); ev = need_results()
    m = st.selectbox("Model", list(S["fcs"]), index=list(S["fcs"]).index(ev["selected"]))
    ins = _insights(m)
    cats = ["All"] + sorted(ins["Category"].unique())
    cat = st.radio("Category", cats, horizontal=True)
    shown = ins if cat == "All" else ins[ins["Category"] == cat]
    for _, r in shown.iterrows():
        ev_html = f'<div class="e">Evidence: {r["Evidence"]}</div>' if r["Evidence"] else ""
        st.markdown(f'<div class="insight {r["Type"]}"><span class="t">{r["Insight"]}</span><span class="c">{r["Category"]}</span><div>{r["Recommendation"]}</div>{ev_html}</div>', unsafe_allow_html=True)
    st.write("**Forecast limitations**")
    for x in limitations(len(s0), stt.season(), m): st.write("• " + x)
    download("Download insights report (Markdown)", rb.insights_report_md(ins, m, len(S["fcs"][m]), S["fcs"][m]).encode(), "business_insights_report.md", "text/markdown", key="ins_dl")


def reports():
    header("Reports & Downloads", "Everything here is generated from the results of your current run.")
    s0 = need_data(); ev = need_results(); proc, log = stt.prepped(); m = ev["selected"]; fc = S["fcs"][m]; hz = S.get("horizon") or len(fc)
    sel = st.selectbox("Model for the report", list(S["fcs"]), index=list(S["fcs"]).index(m)); m = sel; fc = S["fcs"][m]
    ins = _insights(m)
    st.markdown("### Complete report")
    if st.button("Generate PDF report", type="primary", key="rep_pdf"):
        with st.spinner("Building PDF..."):
            try:
                q = data_quality_report(S["raw"], S["ds"]["date_col"], S["ds"]["target_col"], S["freq"], S["ds"]["group_col"], S["ds"]["date_format"], S["cfg"].outlier_threshold, S["ds"]["group_value"])
                S["pdf"] = (m, rb.build_pdf_report(series=s0, processed=proc, fc=fc, ev=ev, ins=ins, model=m, horizon=hz, freq=S["freq"], season=stt.season(),
                                                   quality=q, prep_log=log, dataset_name=S["dataset_name"] or "dataset", level=ev["level"]))
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
    if S.get("pdf") and S["pdf"][0] == m:
        download("Download forecasting report (PDF)", S["pdf"][1], "forecasting_report.pdf", "application/pdf", key="dl_pdf")
    st.markdown("### Data")
    a, b, c = st.columns(3)
    with a: download("Forecast results (CSV)", rb.forecast_csv(fc), "forecast_results.csv", "text/csv")
    with b: download("All models' forecasts (CSV)", rb.all_forecasts_csv(S["fcs"]), "all_model_forecasts.csv", "text/csv")
    with c: download("Processed dataset (CSV)", rb.processed_csv(proc), "processed_dataset.csv", "text/csv")
    a, b, c = st.columns(3)
    with a: download("Model comparison (CSV)", rb.comparison_csv(ev["cv"], ev["holdout"]), "model_comparison.csv", "text/csv")
    with b: download("Actual vs predicted (CSV)", rb.actual_vs_pred_csv(ev["predictions"]), "actual_vs_predicted.csv", "text/csv")
    with c: download("Validation folds (CSV)", rb.csv_bytes(ev["fold_details"]), "validation_folds.csv", "text/csv")
    st.markdown("### Summaries")
    a, b = st.columns(2)
    with a: download("Model evaluation summary (Markdown)", rb.evaluation_report_md(ev, hz, S["freq"], len(s0), stt.season()).encode(), "model_evaluation_report.md", "text/markdown")
    with b: download("Business insights report (Markdown)", rb.insights_report_md(ins, m, hz, fc).encode(), "business_insights_report.md", "text/markdown", key="rep_ins")
    st.markdown("### Visualizations")
    a, b, c = st.columns(3)
    with a: download("Forecast chart (PNG)", rb.forecast_png(proc.dropna(), fc, f"{m} forecast", ev["level"]), "forecast.png", "image/png")
    with b: download("Interactive forecast (HTML)", ch.forecast_chart(proc, fc, f"{m} forecast", ev["level"], 60).to_html(include_plotlyjs="cdn").encode(), "forecast_interactive.html", "text/html")
    with c: download("Interactive model comparison (HTML)", ch.comparison_chart(ev["cv"], ev["holdout"], ev["primary_metric"]).to_html(include_plotlyjs="cdn").encode(), "model_comparison.html", "text/html")
