# Q3 evidence pack

## Reproduce
```
pip install -r requirements-q3.txt
python run_q3.py                                             # prints results, writes outputs/q3/*
python -m unittest discover -s tests -p "test_q3*.py" -v     # 40 Q3 tests
python -m unittest discover -s tests                         # whole project (108 pass; 4 skipped without statsmodels)
```
Source: `q3_core.py` · Script: `run_q3.py` · Notebook: `notebooks/q3_decision_challenge.ipynb` (outputs saved) · Memo: `outputs/q3/decision_memo.md`

## Configuration
Weeks numbered 1-12 · first test week 5 · trailing window 3 · baseline = lower MAE (tie → naive) · alert coverage target 80% ·
week convention W-SUN (W-MON selectable) · no randomness anywhere (results are deterministic).

## Key results (all computed, see outputs/q3/summary.json)
Week 13: naive 215.00, trailing-3 201.67 · MAE naive 11.375, trailing-3 18.333 · naive closer in 8/8 weeks · both under-forecast on average
(mean signed error +9.38 / +18.33) · week-13 alert band 215 ± 20 = [195, 235].

## Assumptions
1. The 12 values are consecutive weekly observations of units sold (not true demand if stock-outs occurred).
2. No timestamps → no dates invented; gap/duplicate logic is verified on dated test inputs.
3. Errors are scored as actual − forecast; MAE uses absolute values.

## Edge cases covered by tests
<5 observations · negative / NaN / inf / non-numeric / None values · exactly 5 observations (1 backtest row) · constant series (zero error) ·
MAE tie → naive · future value change does not alter past forecasts · shuffled series gives different result · exact-duplicate dates ·
same-week different-day duplicates · missing weeks · unparseable/blank dates · out-of-order dates · empty input · Monday vs Sunday weeks ·
no-timestamp mode labelled as assumption · alert at the band boundary · lower band clipped at 0 · invalid coverage · too few errors for requested coverage.

## Known limits
8 backtest errors from one series; the 80% band equals the single largest error; 90% is not supportable (needs ≥ 10 errors);
symmetric band vs one-sided errors in a trend; the Streamlit UI was exercised only via its logic layer here, not in a browser.
