# Time Series Intelligence - Forecasting & Decision Intelligence Platform

A Streamlit application that validates a time-series CSV, explores and prepares it, compares up to 21 forecasting
models with honest chronological validation, forecasts with prediction intervals, and turns the results into
data-driven business insights and downloadable reports (CSV, Markdown, PNG, HTML and a PDF report).

## Install & run
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # core + statsmodels (ARIMA/SARIMA)
pip install -r requirements-optional.txt                 # optional: XGBoost, Prophet, Auto ARIMA
streamlit run app.py
python -m unittest discover -s tests -v                  # or: pip install -r requirements-dev.txt && pytest
```
Use **Dataset Upload → Load sample** or any file in `data/`:
`sample_monthly_sales.csv`, `test_scenarios/` (clean, missing, duplicates, outliers, irregular, seasonal, trends, insufficient),
`frequency_samples/` (daily, weekly, quarterly, long/short monthly, `dayfirst_strings` with DD/MM/YYYY dates).

## Pages
Overview Dashboard · Dataset Upload · Data Validation & Quality · Exploratory Data Analysis · Data Preparation ·
Forecasting Lab · Model Evaluation & Comparison · Prediction Dashboard · Business Intelligence & Insights · Reports & Downloads.

## Models
| Family | Models |
|---|---|
| Baselines | Naive, Seasonal Naive, Drift, Simple Moving Average, Rolling Average |
| Exponential smoothing (own numpy/scipy implementation) | SES, Holt, Damped Holt, Holt-Winters (additive / damped / multiplicative), Theta |
| Regression | Harmonic regression - trend + Fourier terms, **multiple seasonalities** (e.g. weekly + yearly for daily data) |
| ARIMA (statsmodels) | ARIMA (AIC grid, d via ADF), SARIMA, Auto ARIMA (needs `pmdarima`) |
| Machine learning | Random Forest, Gradient Boosting, Extra Trees, XGBoost (optional), Prophet (optional) |

`forecasting/selector.py` pre-selects models from series length, seasonal period and sign (e.g. no Holt-Winters without two
seasonal cycles, no multiplicative models for non-positive data, no data-hungry ML on short series) and explains every exclusion.
Packages that are not installed are listed as *unavailable* with the install command - nothing is hidden.

## Validation methodology (no leakage)
1. The last *h* observations are a **hold-out**, never used for selection.
2. Models are ranked by **expanding, rolling or walk-forward** cross-validation on earlier data; pre-processing is re-fitted
   inside each training window. Ranking metric is configurable (RMSE default; MAE, MAPE, sMAPE, WAPE, MASE).
3. The winner is the best *validation* score - never training accuracy. The hold-out is scored once afterwards, so the selected
   model's hold-out score is an unbiased estimate (other models' hold-out scores are informational).
4. A failing model is recorded and shown; it never aborts the run. Forecasts are rejected if non-finite and clipped at zero for
   non-negative targets. Intervals come from analytic ETS variance, statsmodels (ARIMA) or validation residuals; empirical
   hold-out coverage is reported as a calibration check.
5. If you choose an outlier treatment, flagged outliers are excluded from *error scoring* (and listed in a note) - otherwise one data
   error can dominate every model's score (on the `outliers` scenario, raw scoring selected Drift at 60% CV-MAPE; with treatment
   Holt-Winters at 4%).

Metrics: MAE, MSE, RMSE, MAPE (zero-safe), sMAPE, WAPE, MASE (scaled by in-sample seasonal-naive error), R² (NaN when undefined).

## Pre-processing & EDA
Auto date/target detection, explicit date formats with day/month ambiguity warning, duplicate aggregation (sum/mean/last),
frequency inference, missing-period and irregular-spacing detection, seasonality-aware robust outlier detection (configurable
sensitivity), several imputation methods, differencing helpers, a scored data-quality report, ADF/KPSS (statsmodels if installed,
otherwise a built-in approximation), classical additive/multiplicative decomposition, ACF/PACF, trend/seasonal strength, monthly /
quarterly / yearly patterns. **The uploaded DataFrame is never modified.**

## Business insights
`business_insights/insights.py` derives every statement from the data: growth vs the *same periods one year earlier* (so a Jan-Jun
forecast is not compared with the previous Jul-Dec seasonal peak), forecast totals with range, peaks/troughs, seasonal pattern,
momentum, anomalies, downside/upside scenarios, interval width, hold-out reliability, coverage and gain over baseline. Each insight
shows its numeric evidence.

## Project layout
```
app.py                       entry point (navigation, routing)         ui/            pages, state, components, cached helpers
config.py                    frequencies, seasons, horizons, theme     preprocessing/ validation, quality report, imputation, outliers
forecasting/models.py        model registry + availability             forecasting/ets.py   exponential smoothing engine
forecasting/selector.py      dataset-aware model recommendation        forecasting/pipeline.py  fit/forecast + safeguards
validation/splits.py         chronological / rolling / walk-forward    evaluation/    metrics + selection
exploratory_analysis/eda.py  decomposition, stationarity, ACF/PACF     business_insights/  insights
visualizations/charts.py     Plotly figures                            reports/builders.py CSV / MD / PNG / PDF
utils/sample_data.py         dataset generators                        tests/         unit tests
```

## Verification status (honest)
Developed in a sandbox **without network access**: `streamlit`, `plotly` and `statsmodels` could not be installed there.
* **Executed and passing (68 tests, `python -m unittest discover -s tests`)**: metrics, ETS engine, every non-statsmodels model on the test scenarios, selection and leakage
  (corrupting the hold-out does not change CV scores or the selected model), all three CV modes, preprocessing, outlier/quality logic, EDA fallbacks, insights, daily/weekly/quarterly pipelines,
  and a real multi-page **PDF report** (rendered and visually inspected), CSV/Markdown/PNG outputs.
* **Page logic executed with stand-ins**: all 10 pages were run end-to-end on the sample, 9 scenarios and the frequency datasets using a stub `streamlit` and stub `plotly`
  (0 Python exceptions). This catches code errors, **not** visual/layout or real-widget behaviour.
* **Not executed anywhere yet**: the real Streamlit UI and real Plotly rendering; ARIMA, SARIMA (statsmodels); XGBoost, Prophet, Auto ARIMA.
  `tests/test_advanced.py` contains tests for ARIMA/SARIMA and a `streamlit.testing` navigation test that run automatically once those packages are installed - **please run the test-suite after `pip install -r requirements.txt`**.
* Seasonal models are skipped (with a visible reason) when the history is too short, e.g. Holt-Winters needs 2·season+2 observations.
* Daily data with 730 points and 16 models takes roughly 15-20 s to validate; short monthly data takes seconds.

## Limitations
Intervals for ML/regression models are residual-based approximations; ML models forecast recursively so error accumulates over long horizons;
past patterns are assumed to continue (no promotions/price effects); hold-out scores on a handful of points are noisy.


## Q3 – Time Series Decision Challenge
Weekly units 120,130,...,215 → week-13 naive / trailing-3 forecasts, expanding-window one-step backtest (weeks 5-12), MAE baseline choice,
date-gap/duplicate audit, backtest-error demand alert, decision memo.

| File | Purpose |
|---|---|
| `q3_core.py` | All Q3 logic (pure numpy/pandas, no Streamlit). Single source of truth |
| `run_q3.py` | **Runnable script**: `python run_q3.py` prints results and writes `outputs/q3/` (CSV, JSON, PNG chart, memo, date-audit demo, test log) |
| `notebooks/q3_decision_challenge.ipynb` | Notebook with saved outputs and chart (imports `q3_core`) |
| `outputs/q3/decision_memo.md` | Concise decision memo generated from the computed numbers |
| `tests/test_q3_core.py`, `tests/test_q3.py` | 40 tests: hand-computed values, no-leakage, date edge cases, alert edge cases |
| `ui/pages_q3.py` | Streamlit page "Q3 – Time Series Decision Challenge" (UI only) |
| `docs/Q3_EVIDENCE.md` | Assumptions, edge cases, evidence checklist |

Minimal run (no Streamlit): `pip install -r requirements-q3.txt && python run_q3.py && python -m unittest discover -s tests -p "test_q3*.py"`.
Full app: `pip install -r requirements.txt && streamlit run app.py`.

Honest scope notes: the assignment sequence has no timestamps, so missing/duplicate-date detection is exercised on separate dated inputs and is
reported as an *assumption* (not a clean result) for the numeric sequence. The alert band comes from only 8 backtest errors and carries no
calibrated coverage claim. The Streamlit page itself has not been run in a real browser in the build environment.
