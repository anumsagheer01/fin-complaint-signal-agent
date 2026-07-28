"""
Time-series anomaly detection on daily complaint volume by category.
Uses STL decomposition (seasonal-trend-residual) + a residual z-score threshold.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "daily_volume_timeseries.csv"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "anomaly_results.csv"
SUMMARY_PATH = Path(__file__).parent.parent / "data" / "processed" / "anomaly_summary.json"

Z_THRESHOLD = 2.5  # flag points where residual z-score exceeds this


def load_series():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def detect_anomalies_for_category(df, category, z_threshold=Z_THRESHOLD):
    sub = df[df["category"] == category].sort_values("date").reset_index(drop=True)
    sub = sub.set_index("date")
    series = sub["complaint_count"].asfreq("D").fillna(0)

    # weekly seasonality (period=7) since complaint volume clearly dips on weekends
    stl = STL(series, period=7, robust=True)
    result = stl.fit()

    resid = result.resid
    resid_std = resid.std()
    resid_mean = resid.mean()
    z_scores = (resid - resid_mean) / resid_std

    anomalies = z_scores[abs(z_scores) > z_threshold]

    rows = []
    for date, val in series.items():
        z = z_scores.loc[date]
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "category": category,
            "actual": int(val),
            "trend": round(result.trend.loc[date], 2),
            "seasonal": round(result.seasonal.loc[date], 2),
            "residual": round(resid.loc[date], 2),
            "z_score": round(z, 2),
            "is_anomaly": bool(abs(z) > z_threshold),
        })

    return rows, anomalies


def run():
    df = load_series()
    categories = df["category"].unique()

    all_rows = []
    summary = {}

    for cat in categories:
        rows, anomalies = detect_anomalies_for_category(df, cat)
        all_rows.extend(rows)
        summary[cat] = {
            "total_days": len(rows),
            "anomalies_flagged": len(anomalies),
            "anomaly_rate": round(len(anomalies) / len(rows), 4),
            "top_anomaly_dates": [
                {"date": d.strftime("%Y-%m-%d"), "z_score": round(z, 2)}
                for d, z in anomalies.abs().sort_values(ascending=False).head(5).items()
                for z in [anomalies.loc[d]]
            ],
        }

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Anomaly Detection Summary ===")
    for cat, s in summary.items():
        print(f"\n{cat}")
        print(f"  Total days: {s['total_days']}")
        print(f"  Anomalies flagged (|z| > {Z_THRESHOLD}): {s['anomalies_flagged']} ({s['anomaly_rate']:.1%})")
        print(f"  Top anomaly dates:")
        for a in s["top_anomaly_dates"]:
            print(f"    {a['date']}  z={a['z_score']}")

    print(f"\nFull results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    run()
