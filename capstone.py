"""
energy_analysis.py
End-to-end energy consumption analysis and visualization.

Usage:
    python energy_analysis.py --data path/to/energy.csv

Expected CSV format (two columns):
    timestamp,consumption
    2024-01-01 00:00:00,0.45
    2024-01-01 00:15:00,0.48
    ...

Notes:
- timestamp: parseable by pandas.to_datetime
- consumption: numeric (kWh for interval energy, or kW for instantaneous power)
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams['figure.figsize'] = (10, 5)
sns.set(style="whitegrid")

# -----------------------------
# Utility functions
# -----------------------------
def load_data(csv_path, ts_col="timestamp", value_col="consumption", tz=None):
    """Load CSV into a pandas DataFrame, parse timestamps, set index."""
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {ts_col}, {value_col}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    if tz:
        df[ts_col] = df[ts_col].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    df = df.set_index(ts_col)
    return df

def resample_if_needed(df, rule="H", value_col="consumption", how="sum"):
    """
    Resample to a regular interval. If original data is already regular,
    you can still resample (e.g., sum 15-min into hourly by rule='H' and how='sum').
    how: 'sum' for interval energy (kWh), 'mean' for power (kW)
    """
    if how == "sum":
        return df.resample(rule).sum()
    else:
        return df.resample(rule).mean()

def summarize(df, value_col="consumption"):
    s = df[value_col].describe()
    totals = df[value_col].sum()
    return s, totals

def detect_anomalies(series, z_thresh=3.0):
    """
    Simple anomaly detection based on z-score of daily totals.
    Returns indices (timestamps) of anomalies.
    """
    z = np.abs(stats.zscore(series.fillna(0)))
    anomalies = series.index[(z > z_thresh)]
    return anomalies, z

# -----------------------------
# Analysis + Visualizations
# -----------------------------
def timeseries_plot(df, value_col="consumption", title="Energy consumption time series"):
    fig, ax = plt.subplots()
    ax.plot(df.index, df[value_col], linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(value_col + " (kWh)")
    ax.set_xlabel("Time")
    # format x ticks nicely
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    plt.tight_layout()
    out = OUTPUT_DIR / "timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

def daily_profile_plot(df, value_col="consumption", title="Average daily profile (hourly)"):
    # group by hour of day and compute mean
    hourly = df[value_col].groupby(df.index.hour).mean()
    fig, ax = plt.subplots()
    ax.plot(hourly.index, hourly.values, marker='o')
    ax.set_xticks(range(0,24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(value_col + " (kWh)")
    ax.set_title(title)
    plt.grid(True)
    plt.tight_layout()
    out = OUTPUT_DIR / "daily_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

def heatmap_hour_day(df, value_col="consumption", aggfunc="mean"):
    # pivot: day of week vs hour
    df2 = df.copy()
    df2['dow'] = df2.index.day_name()
    df2['hour'] = df2.index.hour
    pivot = df2.pivot_table(index='dow', columns='hour', values=value_col, aggfunc=aggfunc)
    # ensure day order Mon..Sun
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(days)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3)
    ax.set_title(f"Heatmap ({aggfunc}) by day/hour")
    plt.tight_layout()
    out = OUTPUT_DIR / "heatmap_day_hour.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return pivot

def boxplot_by_month(df, value_col="consumption"):
    df2 = df.copy()
    df2['month'] = df2.index.to_period('M').astype(str)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x='month', y=value_col, data=df2.reset_index(), ax=ax)
    ax.set_title("Monthly distribution (boxplot)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = OUTPUT_DIR / "boxplot_month.png"
