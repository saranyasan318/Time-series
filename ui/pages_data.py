
import numpy as np
import pandas as pd
import streamlit as st

from business_insights.insights import forecast_kpis
from config import (DATE_FORMATS, DUPLICATE_AGG, FREQUENCIES, MISSING_METHODS, OUTLIER_METHODS, OUTLIER_THRESHOLDS,
                    PERIOD_NAME, SEASON_LENGTH)
from exploratory_analysis import eda
from preprocessing.cleaning import (PrepConfig, consistency_checks, data_quality_report, detect_date_format, detect_frequency,
                                    detect_outliers, difference, interval_profile, missing_timestamps, parse_dates,
                                    suggest_differencing, validate_dataset)
from reports import builders as rb
from ui import state as stt
from ui.components import download, fmt, header, kpi, kpi_row, need_data, note, plot, table
from ui.state import S
from utils.sample_data import make_sample_sales, make_test_scenarios
from visualizations import charts as ch


def overview():
    st.markdown('<div class="hero"><h1>Time Series Intelligence</h1><p>Upload a time series, validate and prepare it, compare forecasting models '
                'with honest chronological validation, and turn the forecast into decisions - every number computed from your data.</p></div>',
                unsafe_allow_html=True)
    has_data, has_fc = stt.ready(), bool(S["ev"] and S["fcs"])
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.button("Get started", type="primary", on_click=stt.goto, args=("Dataset Upload",), key="ov_start")
    if c2.button("Load sample", key="ov_sample"):
        stt.load_dataset(make_sample_sales(), "sample_monthly_sales.csv"); st.rerun()
    if not has_data:
        st.info("No dataset loaded yet. Click **Get started** to upload a CSV or load the built-in sample.")
   
        return
    s = S["series"]; proc, _ = stt.prepped()
    st.markdown("### Dashboard")
    q = data_quality_report(S["raw"], S["ds"]["date_col"], S["ds"]["target_col"], S["freq"], S["ds"]["group_col"], S["ds"]["date_format"],
                            S["cfg"].outlier_threshold, S["ds"]["group_value"])
    items = [("Dataset", S["dataset_name"] or "uploaded", f"{S['ds']['target_col']}"), ("Periods", f"{len(s):,}", f"{s.index.min():%Y-%m-%d} to {s.index.max():%Y-%m-%d}"),
             ("Frequency", S["freq"], f"season length {stt.season()}"), ("Quality score", f"{q['score']:.0f}/100", "; ".join(q["issues"][:1]) or "no issues",
              "good" if q["score"] >= 85 else "warn" if q["score"] >= 65 else "bad"), ("Latest value", fmt(float(s.dropna().iloc[-1])), f"{s.dropna().index[-1]:%Y-%m-%d}")]
    kpi_row(items)
    st.write("")
    if has_fc:
        ev, m = S["ev"], S["ev"]["selected"]; fc = S["fcs"][m]
        kp = forecast_kpis(proc, fc, ev["level"], S["freq"]); hold = ev["holdout"].set_index("Model").loc[m]
        kpi_row([("Best model (CV)", m, f"hold-out MAPE {fmt(hold['MAPE'])}%"), ("Forecast total", fmt(kp["forecast_total"]), f"{kp['total_change_%']:+.1f}% vs prior-year/recent"),
                 ("Peak period", f"{kp['peak_date']:%b %Y}", fmt(kp["peak_value"])), ("Lowest period", f"{kp['low_date']:%b %Y}", fmt(kp["low_value"])),
                 ("Interval width", f"{kp['mean_interval_width_%']:.0f}%", f"{ev['level']:.0%} prediction interval")])
        plot(ch.forecast_chart(proc, fc, f"{m} - forecast with {ev['level']:.0%} interval", ev["level"], history_tail=60), key="ov_fc")
    else:
        plot(ch.history_chart(proc, title=f"{S['ds']['target_col']} - history"), key="ov_hist")
        st.button("Continue to Forecasting Lab", on_click=stt.goto, args=("Forecasting Lab",), key="ov_lab")



def upload():
    header("Dataset Upload", "Your original file is never modified - all cleaning happens on a copy.")
    up = st.file_uploader("Upload a CSV file (a date column and a numeric target)", type=["csv", "txt"])
    if up is not None and S["up_id"] != (up.name, up.size):
        try:
            df = stt.read_csv_cached(up.getvalue())
            if df.shape[1] < 2 or len(df) < 3:
                st.error("The file needs at least two columns and three rows.")
            else:
                S["up_id"] = (up.name, up.size); stt.load_dataset(df, up.name)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
    c1, c2 = st.columns([1, 2])
    if c1.button("Load sample dataset (48 months, 3 categories)", key="up_sample"):
        S["up_id"] = None; stt.load_dataset(make_sample_sales(), "sample_monthly_sales.csv"); st.rerun()
    scen = make_test_scenarios()
    pick = c2.selectbox("Or load a test scenario", ["(choose)"] + list(scen), key="up_scen", help="Datasets with injected problems used to verify the pipeline")
    if pick != "(choose)" and S["dataset_name"] != f"scenario:{pick}":
        S["up_id"] = None; stt.load_dataset(scen[pick], f"scenario:{pick}"); st.rerun()
    raw = S["raw"]
    if raw is None:
        st.info("Upload a file or load a sample to continue."); return
    st.caption(f"**{S['dataset_name']}** - {len(raw):,} rows x {raw.shape[1]} columns")
    table(raw.head(10))
    ds = dict(S["ds"]); cols = list(raw.columns)
    a, b, c = st.columns(3)
    ds["date_col"] = a.selectbox("Date column", cols, index=cols.index(ds["date_col"]) if ds["date_col"] in cols else 0)
    num = [x for x in cols if x != ds["date_col"] and pd.api.types.is_numeric_dtype(raw[x])] or [x for x in cols if x != ds["date_col"]]
    ds["target_col"] = b.selectbox("Target variable", num, index=num.index(ds["target_col"]) if ds["target_col"] in num else 0)
    others = ["(none)"] + [x for x in cols if x not in (ds["date_col"], ds["target_col"])]
    g = c.selectbox("Grouping column (optional)", others, index=others.index(ds["group_col"]) if ds["group_col"] in others else 0)
    ds["group_col"] = None if g == "(none)" else g
    d1, d2, d3, d4 = st.columns(4)
    fmt_guess = None
    if not pd.api.types.is_datetime64_any_dtype(raw[ds["date_col"]]) and not pd.api.types.is_numeric_dtype(raw[ds["date_col"]]):
        fmt_guess = detect_date_format(raw[ds["date_col"]])
    labels = list(DATE_FORMATS)
    cur = next((k for k, v in DATE_FORMATS.items() if v == ds["date_format"]), "Auto-detect")
    ds["date_format"] = DATE_FORMATS[d1.selectbox("Date format", labels, index=labels.index(cur),
                                                  help=f"Best detected: {fmt_guess[0] if fmt_guess else 'n/a'}")]
    flabels = list(FREQUENCIES); cur_f = [k for k, v in FREQUENCIES.items() if v == ds["freq"]][0]
    ds["freq"] = FREQUENCIES[d2.selectbox("Frequency", flabels, index=flabels.index(cur_f), help=f"Detected: {ds.get('detected') or 'unknown'}")]
    if ds["group_col"]:
        vals = ["ALL"] + sorted(raw[ds["group_col"]].astype(str).unique())
        ds["group_value"] = d3.selectbox("Category", vals, index=vals.index(ds["group_value"]) if ds["group_value"] in vals else 0)
    else:
        ds["group_value"] = None
    ds["agg"] = d4.selectbox("Aggregate duplicates with", DUPLICATE_AGG, index=DUPLICATE_AGG.index(ds["agg"]))
    if fmt_guess and fmt_guess[2] and ds["date_format"] is None:
        st.warning("Dates such as 01/02/2024 are ambiguous (day/month vs month/day). Pick the date format explicitly to be safe.")
    S["ds"] = ds
    rep = validate_dataset(raw, ds["date_col"], ds["target_col"], ds["group_col"], ds["freq"], date_format=ds["date_format"])
    for e in rep["errors"]: st.error(e)
    for w in rep["warnings"]: st.warning(w)
    s = stt.build_series()
    if s is not None:
        st.success(f"Series ready: {len(s)} periods ({s.index.min():%Y-%m-%d} to {s.index.max():%Y-%m-%d}). All pages are unlocked.")
        for l in S["build_log"]: st.caption("• " + l)
        plot(ch.history_chart(s, title=f"{ds['target_col']}" + (f" ({ds['group_value']})" if ds["group_value"] else "")), key="up_hist")
        st.button("Continue to Data Validation & Quality", on_click=stt.goto, args=("Data Validation & Quality",), key="up_next")



def validation():
    header("Data Validation & Quality", "Problems are reported, never silently fixed.")
    need_data(); raw, ds = S["raw"], S["ds"]
    q = data_quality_report(raw, ds["date_col"], ds["target_col"], S["freq"], ds["group_col"], ds["date_format"], S["cfg"].outlier_threshold, ds["group_value"])
    c1, c2 = st.columns([1, 2])
    with c1:
        plot(ch.quality_gauge(q["score"]), key="q_gauge")
    with c2:
        kpi_row([("Rows", f"{q['rows']:,}"), ("Missing target", q["missing_target"] + q["non_numeric_target"], None, "warn" if q["missing_target"] else ""),
                 ("Duplicate timestamps", q["duplicate_timestamps"], f"{q['exact_duplicate_rows']} exact rows", "warn" if q["duplicate_timestamps"] else ""),
                 ("Missing periods", q["missing_periods"], None, "warn" if q["missing_periods"] else ""), ("Outliers flagged", q["outliers"], None, "warn" if q["outliers"] else "")])
        st.write("")
        if q["issues"]:
            for i in q["issues"]: st.warning(i)
        else:
            st.success("No data-quality issues detected.")
    with st.expander("How the score is computed"):
        pen = pd.DataFrame({"Penalty": q["penalties"]}).round(2)
        table(pen); st.caption("Score = 100 - sum of capped penalties. It is a screening aid, not a guarantee.")
    iv = q["intervals"]
    if iv.get("modal_gap_days"):
        st.write(f"**Timestamp spacing:** modal gap {iv['modal_gap_days']:.0f} days, {iv['irregular_share']:.0%} irregular, largest gap {iv['max_gap_days']:.0f} days. "
                 f"Detected frequency: **{q['detected_frequency'] or 'undetermined'}** (selected: {S['freq']}).")
    a, b = st.columns(2)
    with a:
        st.markdown("**Missing timestamps on the selected frequency grid**")
        try:
            gaps = missing_timestamps(raw, ds["date_col"], S["freq"], ds["date_format"])
            table(pd.DataFrame({"missing period": gaps.strftime("%Y-%m-%d")}).head(200)) if len(gaps) else st.caption("none")
        except Exception as e:
            st.caption(f"unavailable: {e}")
    with b:
        st.markdown("**Duplicate timestamps (rows involved)**")
        dates = parse_dates(raw, ds["date_col"], ds["date_format"])[0]
        keys = [dates] + ([raw[ds["group_col"]]] if ds["group_col"] else [])
        dup = raw[pd.concat(keys, axis=1).duplicated(keep=False) & dates.notna()]
        table(dup.head(100)) if len(dup) else st.caption("none")
    rep = validate_dataset(raw, ds["date_col"], ds["target_col"], ds["group_col"], S["freq"], date_format=ds["date_format"])
    for e in rep["errors"]: st.error(e)
    for w in rep["warnings"]: st.info(w)
    cc = consistency_checks(S["series"])
    st.caption(f"Series checks - monotonic index: {cc['monotonic_index']}, unique index: {cc['unique_index']}, regular grid: {cc['regular_grid']}, "
               f"missing periods in the regular series: {cc['n_missing']}, negative values: {cc['n_negative']}.")



def exploratory():
    header("Exploratory Data Analysis", "Understand trend, seasonality, stationarity and dependence before modelling.")
    s0 = need_data(); s, _ = stt.prepped(); m = stt.season(); f = S["freq"]
    if s.isna().any():
        st.warning("Missing values remain after preparation; some analyses use the observed points only.")
    info = stt.cached_characterise(s, f, m); tr = info["trend"]
    kpi_row([("Trend", tr["direction"], f"{tr['pct_of_mean_per_period']:+.2f}% of mean / period"), ("Trend strength", fmt(info["trend_strength"]), "decomposition-based, 0-1"),
             ("Seasonal strength", fmt(info["seasonal_strength"]), "seasonal" if info["seasonal_flag"] else "weak / none"),
             ("Variability (CV)", f"{info['cv_%']:.0f}%", "std / mean"), ("Stationarity", info["stationarity"]["verdict"].split(" (")[0], info["stationarity"].get("engine", ""))])
    st.write("")
    win = st.slider("Rolling window", 2, max(3, min(24, len(s) // 2)), min(m, 12) if m > 1 else 3)
    r = eda.add_rolling(s, win)
    note("<b>Why:</b> the rolling mean separates trend from noise; the rolling std shows whether variability changes over time (non-constant variance).")
    plot(ch.history_chart(s, r["rolling_mean"]), key="eda_hist")
    c1, c2 = st.columns(2)
    with c1: plot(ch.rolling_std_chart(r), key="eda_std")
    with c2: plot(ch.pct_change_chart(r), key="eda_pct")
    st.markdown("### Patterns by calendar period")
    t1, t2, t3, t4 = st.tabs(["Monthly", "Quarterly", "Yearly", "Heatmap"])
    with t1: plot(ch.monthly_pattern_chart(s), key="eda_m")
    with t2: plot(ch.quarterly_pattern_chart(s), key="eda_q")
    with t3:
        plot(ch.yearly_pattern_chart(s), key="eda_y")
        yt = eda.period_tables(s)["year"]; table(yt.round(2))
    with t4:
        if s.dropna().index.year.nunique() >= 2: plot(ch.heatmap_chart(s), key="eda_h")
        else: st.info("A heatmap needs at least two calendar years.")
    sc = eda.sudden_changes(s)
    st.write("**Sudden changes (|z| > 3 on period change):** " + (", ".join(f"{d:%Y-%m-%d} ({v:+.0%})" for d, v in sc.items()) or "none detected"))
    cyc = info["cycle"]
    if cyc: st.write(f"**Dominant cycle (FFT of detrended series):** ~{cyc['period']:.1f} {PERIOD_NAME[f]}s, {cyc['power_share']:.0%} of spectral power" + ("" if cyc["power_share"] > .25 else " (weak - indicative only)"))
    st.markdown("### Seasonality and decomposition")
    try:
        prof = eda.seasonal_profile(s, f); plot(ch.seasonal_bar(prof), key="eda_prof")
        st.caption(f"Seasonal range (max - min of the profile): {eda.seasonality_strength(s, f):.2f} x mean.")
    except Exception as e:
        st.info(f"Seasonal profile unavailable: {e}")
    model = st.radio("Decomposition model", ["additive", "multiplicative"], horizontal=True)
    try:
        plot(ch.decomposition_chart(stt.cached_decompose(s, m, model)), key="eda_dec")
    except Exception as e:
        st.info(f"Decomposition unavailable: {e}")
    st.markdown("### Stationarity and differencing")
    t = info["stationarity"]
    if "adf_pvalue" in t:
        st.markdown(f"ADF p = **{t['adf_pvalue']:.3f}**, KPSS p = **{t['kpss_pvalue']:.3f}** → **{t['verdict']}**  _(engine: {t['engine']})_")
    d = suggest_differencing(s, m)
    st.write(f"Suggested differencing: **d = {d['d']}, D = {d['D']}** (method: {d['method']}).")
    if d["d"] or d["D"]:
        diffd = difference(s, d["d"], d["D"], m)
        try: st.markdown(f"After differencing - {eda.stationarity_tests(diffd)['verdict']} (n={len(diffd)})")
        except Exception: pass
        plot(ch.history_chart(diffd, title="Differenced series (diagnostic only; models difference internally)"), key="eda_diff")
    nl = max(2, min(24, len(s) // 2 - 2)); acf, pacf = stt.cached_acf(s.dropna().values, nl)
    c1, c2 = st.columns(2)
    with c1: plot(ch.acf_chart(acf, len(s), "ACF"), key="eda_acf")
    with c2: plot(ch.acf_chart(pacf, len(s), "PACF"), key="eda_pacf")
    st.caption("ACF peaks at the seasonal lag indicate seasonality; slow decay indicates trend. PACF cut-offs suggest the AR order.")


def preparation():
    header("Data Preparation", "Treatments are configurable and re-fitted on every training window during validation (no leakage).")
    s0 = need_data(); cfg = S["cfg"]
    a, b, c = st.columns(3)
    mm = a.selectbox("Missing-value treatment", MISSING_METHODS, index=MISSING_METHODS.index(cfg.missing_method))
    om = b.selectbox("Outlier treatment", OUTLIER_METHODS, index=OUTLIER_METHODS.index(cfg.outlier_method))
    tl = list(OUTLIER_THRESHOLDS); cur = next((k for k, v in OUTLIER_THRESHOLDS.items() if v == cfg.outlier_threshold), tl[1])
    th = OUTLIER_THRESHOLDS[c.selectbox("Outlier sensitivity", tl, index=tl.index(cur))]
    new = PrepConfig(mm, om, th)
    if new != cfg:
        S["cfg"] = new; stt.reset_results()
    s, log = stt.prepped()
    mask = detect_outliers(s0.dropna(), stt.season() if stt.season() > 1 else 7, th, stt.season())
    kpi_row([("Periods", len(s0)), ("Missing (raw grid)", int(s0.isna().sum())), ("Missing after prep", int(s.isna().sum()), None, "bad" if s.isna().any() else "good"),
             ("Outliers flagged", int(mask.sum()), om if om != OUTLIER_METHODS[0] else "flag only"), ("Frequency", S["freq"])])
    st.write("")
    for l in log: st.write("• " + l)
    if s.isna().any(): st.error("Missing values remain; models will refuse to train. Choose a fill method.")
    if mask.any() and om == OUTLIER_METHODS[0]:
        st.warning("Outliers are flagged but kept. On contaminated data this can distort model selection - consider a treatment (flagged points are then excluded from error scoring).")
    if mask.any():
        table(pd.DataFrame({"date": s0.dropna().index[mask.values].strftime("%Y-%m-%d"), "value": s0.dropna().values[mask.values]}))
    plot(ch.before_after_chart(s0, s), key="prep_ba")
    note(f"The original upload ({len(S['raw']):,} rows) is untouched. Download the processed series below; preparation shown here is applied to the full history, "
         "whereas validation applies it separately inside each training window.")
    download("Download processed dataset (CSV)", rb.processed_csv(s), "processed_dataset.csv", "text/csv")
