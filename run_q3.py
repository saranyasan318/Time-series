
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q3_core import (SALES, METHODS, forecast_next, expanding_backtest, backtest_summary, select_baseline,  # noqa: E402
                     leaky_neighbour_mae, audit_weekly_dates, alert_tolerance, demand_alert, decision_memo)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "q3")
COVERAGE = 0.80


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Weekly units:", SALES)

    
    fc = forecast_next(SALES)
    print(f"\n Week {fc['target_week']} forecasts\n    naive last value : {fc[METHODS[0]]:.2f}\n"
          f"    trailing 3-week  : {fc[METHODS[1]]:.2f}")

   
    bt = expanding_backtest(SALES)
    s = backtest_summary(bt)
    chosen = select_baseline(s["naive_mae"], s["avg_mae"])
    print("\n Expanding-window one-step backtest, weeks 5-12")
    print(bt.round(2).to_string(index=False))
    print(f"    Naive MAE = {s['naive_mae']:.3f} | 3-wk average MAE = {s['avg_mae']:.3f} -> baseline: {chosen}")
    print(f"    naive closer in {s['naive_wins']}/{s['n_weeks']} weeks; mean signed error naive {s['naive_bias']:+.2f}, avg {s['avg_bias']:+.2f}")
    print(f"    leakage illustration (neighbour 'forecast' MAE) = {leaky_neighbour_mae(SALES):.2f}  <- unrealistically low")
    bt.to_csv(os.path.join(OUT, "backtest.csv"), index=False)

   
    r0 = audit_weekly_dates(None, n_obs=len(SALES), assumed_start="2024-01-07")
    dated = ["2024-01-07", "2024-01-14", "2024-01-14", "2024-01-28", "2024-02-06", "not-a-date"]
    r1 = audit_weekly_dates(dated)
    lines = ["No-timestamp mode: " + r0["note"], "",
             f"Fault-injected demo input: {dated}",
             f"  invalid={r1['invalid']}  exact_duplicates={[str(d.date()) for d in r1['exact_duplicates']]}",
             f"  missing_weeks(week ending)={[str(d.date()) for d in r1['missing_weeks']]}",
             f"  same_week_duplicates={[str(d.date()) for d in r1['same_week_duplicates']]}  off_cadence_weekday={r1['off_cadence']}"]
    print("\n Date audit\n    " + "\n    ".join(lines))
    open(os.path.join(OUT, "date_audit_demo.txt"), "w").write("\n".join(lines) + "\n")

    
    col = "Naive absolute error" if chosen == METHODS[0] else "3-week average absolute error"
    tol = alert_tolerance(bt[col], COVERAGE)
    tol90 = alert_tolerance(bt[col], 0.90)
    chosen_fc = fc[chosen]
    print(f"\n Alert band for week {fc['target_week']} ({chosen}): {chosen_fc:.1f} +/- {tol['half_width']:.1f} "
          f"= [{chosen_fc - tol['half_width']:.1f}, {chosen_fc + tol['half_width']:.1f}]  (target {COVERAGE:.0%}, n={tol['n_errors']})")
    print(f"    90% target: guarantee_available={tol90['guarantee_available']} (needs >= {tol90['min_errors_needed']} errors)")
    for a in (225, 250, 190):
        r = demand_alert(chosen_fc, a, bt[col], COVERAGE)
        print(f"    actual={a}: {'ALERT ' + r['direction'] if r['alert'] else 'within band'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        wk = list(range(1, len(SALES) + 1))
        ax.plot(wk, SALES, "o-", label="Actual", color="#0B1B3A")
        ax.plot(bt["Week"], bt["Naive forecast"], "s--", label=f"Naive (MAE {s['naive_mae']:.1f})", color="#5B6EE1")
        ax.plot(bt["Week"], bt["3-week average forecast"], "^--", label=f"3-wk avg (MAE {s['avg_mae']:.1f})", color="#C9B8F0")
        ax.errorbar([13], [chosen_fc], yerr=[[tol["half_width"]], [tol["half_width"]]], fmt="D", color="crimson",
                    capsize=5, label=f"Week 13 {chosen_fc:.0f} ± {tol['half_width']:.0f} (alert band)")
        ax.set_xlabel("Week"); ax.set_ylabel("Units sold"); ax.set_title("One-step-ahead backtest and week-13 forecast")
        ax.set_xticks(range(1, 14)); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "backtest_chart.png"), dpi=150); plt.close(fig)
    except ImportError:
        print("    (matplotlib not installed: chart skipped)")

    
    memo = decision_memo(SALES, COVERAGE)
    open(os.path.join(OUT, "decision_memo.md"), "w", encoding="utf-8").write(memo)
    summary = {"week_13": {k: v for k, v in fc.items()}, "backtest": s, "baseline": chosen,
               "alert": {"coverage": COVERAGE, **tol}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(f"\n(e) Evidence written to {OUT}")


if __name__ == "__main__":
    main()
