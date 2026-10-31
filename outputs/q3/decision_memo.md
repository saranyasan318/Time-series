# Q3 decision memo: weekly demand baseline

## Decision
Adopt **Naive last value** as the working baseline for week 13: forecast **215.0 units**.
It had the lower one-step-ahead MAE over the 8 backtest weeks (5-12). This is a choice between two
simple baselines on one short series. It is **not** evidence of how accurate the forecast will be.

## Evidence
| Method | Week 13 forecast | Backtest MAE | Mean signed error | Weeks it was closer |
|---|---|---|---|---|
| Naive last value | 215.0 | 11.375 | +9.38 | 8 of 8 |
| Trailing 3-week average | 201.7 | 18.333 | +18.33 | 0 of 8 |

Signed error = actual - forecast. Both mean signed errors are positive: demand kept rising, so both methods under-forecast on average.
The 3-week average lags more because it averages older, lower weeks. That lag is the likely reason it loses,
and it would reverse on a flat or mean-reverting series.

## Validation design
Expanding window, one step ahead: to score week t the methods see weeks 1..t-1 only.
Shuffling observations would let weeks after t inform the forecast of week t. As an illustration, "predicting"
each week from its two neighbours (one of which is in the future) gives MAE 8.57, lower than the honest
naive forecast here (11.38), even though it uses information no real forecast could have. Shuffling also breaks the trend and autocorrelation the methods depend on.

## Demand alert
Alert if actual demand falls outside forecast +/- **20.0 units** (the
rank-8 of 8 backtest absolute errors of the chosen method, target 80%).
For week 13: 195.0 to 235.0 units.
With 8 errors this reaches the minimum sample for a finite-sample coverage statement, but only if future errors resemble past ones (doubtful on a trending series).

## Limitations
- 8 errors from one series: MAE, the band and the baseline ranking could all change with a few more weeks.
  No significance test is meaningful at this size.
- The band is symmetric, but errors here are one-sided (bias +9.4 for naive). Under a continuing trend
  a symmetric band will flag rises late and miss that the forecast is systematically low.
- The rule assumes future errors look like past errors (exchangeability). Trend changes, promotions,
  stock-outs (sales are not demand) and holidays break this.
- Week numbers are assumed to be consecutive weeks. No timestamps were supplied, so missing or
  duplicate calendar dates cannot be detected on this data; the detector is tested on separate dated inputs.
- Recommendation: use the baseline now, record each new week's error, and re-run the comparison (and test
  a trend-aware challenger such as drift or Holt) once there are enough weeks to support it.
