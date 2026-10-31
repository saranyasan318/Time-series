
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  #
    st = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from q3_core import (SALES, METHODS, FIRST_TEST_WEEK, validate_values, forecast_next,  # noqa: E402,F401
                     expanding_backtest, backtest_summary, select_baseline, leaky_neighbour_mae,
                     audit_weekly_dates, validate_weekly_dates, alert_tolerance, demand_alert)

make_backtest = expanding_backtest
_validated_values = validate_values


def forecast_week_13(values):
    f = forecast_next(values)
    return f[METHODS[0]], f[METHODS[1]]


def page():
    if st is None:
        raise RuntimeError("Streamlit is required to display the Q3 page. Install requirements.txt.")
    st.title("Q3 – Time Series Decision Challenge")
    st.caption("Baseline forecasts, chronological backtest, date-quality audit, and uncertainty-based alert.")

    st.subheader("Input: weekly units sold")
    values_text = st.text_input("Comma-separated weekly units", value=", ".join(map(str, SALES)), key="q3_sales_input")
    try:
        tokens = [x.strip() for x in values_text.split(",")]
        if any(not x for x in tokens):
            raise ValueError("Remove empty entries; separate every weekly value with a comma.")
        values = validate_values([float(x) for x in tokens], minimum=FIRST_TEST_WEEK)
    except Exception as exc:
        st.error(f"Invalid sales sequence: {exc}")
        st.info("Backtesting needs at least 5 finite, non-negative numeric observations.")
        return

    fc = forecast_next(values)
    st.subheader(f"(a) Week {fc['target_week']} forecasts")
    c1, c2 = st.columns(2)
    c1.metric("Naive last value", f"{fc[METHODS[0]]:.2f} units")
    c2.metric("Trailing 3-week average", f"{fc[METHODS[1]]:.2f} units")

    st.subheader("(b) Expanding-window, one-step-ahead backtest (weeks 5 onward)")
    bt = expanding_backtest(values)
    s = backtest_summary(bt)
    chosen = select_baseline(s["naive_mae"], s["avg_mae"])
    st.dataframe(bt.round(3), use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Naive MAE", f"{s['naive_mae']:.3f}")
    c2.metric("3-week average MAE", f"{s['avg_mae']:.3f}")
    c3.metric("Selected baseline", chosen)
    st.caption(f"Naive was closer in {s['naive_wins']} of {s['n_weeks']} weeks; mean signed error "
               f"naive {s['naive_bias']:+.2f}, average {s['avg_bias']:+.2f} (actual - forecast).")
    st.info("Shuffling would let later weeks inform earlier forecasts (leakage) and destroy trend and autocorrelation. "
            f"Illustration: predicting each week from its two neighbours gives MAE {leaky_neighbour_mae(values):.2f}, "
            f"lower than the honest naive MAE of {s['naive_mae']:.2f}.")

    st.subheader("(c) Missing weekly dates and duplicate dates")
    st.write("The assignment values have no timestamps, so gaps and duplicates cannot be observed from them. "
             "Upload a CSV with real dates to run the audit.")
    uploaded = st.file_uploader("Optional: CSV with weekly timestamps", type=["csv"], key="q3_dates")
    if uploaded is not None:
        try:
            date_df = pd.read_csv(uploaded)
            if date_df.empty or len(date_df.columns) == 0:
                st.warning("The uploaded CSV is empty.")
            else:
                col = st.selectbox("Date column", date_df.columns, key="q3_date_col")
                freq = st.selectbox("Week-ending convention", ["W-SUN", "W-MON"], key="q3_freq")
                r = audit_weekly_dates(date_df[col], freq=freq)
                st.write(f"Invalid/blank dates: **{r['invalid']}** · exact duplicates: **{len(r['exact_duplicates'])}** · "
                         f"same-week duplicates: **{len(r['same_week_duplicates'])}** · missing weeks: **{len(r['missing_weeks'])}** · "
                         f"out of order: **{r['out_of_order']}** · off-cadence weekday: **{r['off_cadence']}**")
                for key, label in (("exact_duplicates", "Exact duplicate dates"), ("same_week_duplicates", "Same-week duplicates (week ending)"),
                                   ("missing_weeks", "Missing weeks (week ending)")):
                    if r[key]:
                        st.write(label + ":", [d.strftime("%Y-%m-%d") for d in r[key]])
                (st.success if r["ok"] else st.warning)(r["note"])
        except Exception as exc:
            st.error(f"Could not validate dates: {exc}")

    st.subheader("(d) Demand alert from backtest absolute errors")
    errs = bt["Naive absolute error"] if chosen == METHODS[0] else bt["3-week average absolute error"]
    coverage = st.slider("Target coverage of the alert band", 0.50, 0.95, 0.80, 0.05, key="q3_cov")
    base_fc = fc[chosen]
    actual = st.number_input("New actual demand (units)", min_value=0.0, value=float(base_fc), key="q3_actual")
    a = demand_alert(base_fc, actual, errs, coverage)
    c1, c2, c3 = st.columns(3)
    c1.metric("Band half-width", f"{a['half_width']:.1f} units")
    c2.metric("Band", f"{a['lower']:.1f} – {a['upper']:.1f}")
    c3.metric("Absolute error", f"{a['absolute_error']:.1f}")
    if a["alert"]:
        st.error(f"Demand alert: actual is {a['direction']} the band.")
    else:
        st.success("No alert: actual is within the band.")
    if not a["guarantee_available"]:
        st.warning(f"{a['n_errors']} errors are fewer than the {a['min_errors_needed']} needed for a finite-sample "
                   f"{coverage:.0%} statement. The largest error is used and no coverage is claimed.")
    st.caption("Limits: tiny sample; symmetric band vs one-sided errors in a trend; assumes future errors resemble past ones.")

