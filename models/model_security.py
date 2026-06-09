# =============================================================================
# CashContant — Security & Fraud Detection Model
# Analyses ATM transaction data for anomalies and computes a per-transaction
# anomaly score (0–100) in addition to aggregate risk flags.
# =============================================================================

import pandas as pd
import numpy as np
from config import (
    RAPID_WITHDRAWAL_SECONDS,
    IMPOSSIBLE_TRAVEL_SECONDS,
    AFTER_HOURS_START,
    AFTER_HOURS_END,
    HIGH_VALUE_PERCENTILE,
)


def _flag_rapid_withdrawals(data: pd.DataFrame) -> pd.Series:
    """Flag transactions made less than RAPID_WITHDRAWAL_SECONDS after the previous one on the same card."""
    if not {"card_id", "timestamp"}.issubset(data.columns):
        return pd.Series(False, index=data.index)
    df = data.sort_values(["card_id", "timestamp"])
    time_diff = df.groupby("card_id")["timestamp"].diff().dt.total_seconds()
    return time_diff < RAPID_WITHDRAWAL_SECONDS


def _flag_impossible_travel(data: pd.DataFrame) -> pd.Series:
    """Flag transactions where the same card appears at a different ATM within 10 minutes."""
    if not {"card_id", "atm_id", "timestamp"}.issubset(data.columns):
        return pd.Series(False, index=data.index)
    df = data.sort_values(["card_id", "timestamp"])
    prev_atm = df.groupby("card_id")["atm_id"].shift()
    time_gap = df.groupby("card_id")["timestamp"].diff().dt.total_seconds()
    return (df["atm_id"] != prev_atm) & (time_gap < IMPOSSIBLE_TRAVEL_SECONDS)


def _flag_after_hours(data: pd.DataFrame) -> pd.Series:
    """Flag transactions occurring between midnight and 4am."""
    if "timestamp" not in data.columns:
        return pd.Series(False, index=data.index)
    hour = pd.to_datetime(data["timestamp"], errors="coerce").dt.hour
    return hour.between(AFTER_HOURS_START, AFTER_HOURS_END)


def _flag_high_value(data: pd.DataFrame) -> pd.Series:
    """Flag transactions in the top 0.5% by amount using IQR outlier method."""
    if "amount" not in data.columns:
        return pd.Series(False, index=data.index)
    q1 = data["amount"].quantile(0.25)
    q3 = data["amount"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return data["amount"] > upper_bound


def _flag_negative_amounts(data: pd.DataFrame) -> pd.Series:
    """Flag any transactions with negative amounts."""
    if "amount" not in data.columns:
        return pd.Series(False, index=data.index)
    return data["amount"] < 0


def compute_anomaly_scores(data: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a composite anomaly score (0–100) to each transaction.
    Each flag contributes a weighted amount to the score.

    Weights:
        - Impossible travel: 40 pts
        - High-value outlier: 25 pts
        - Rapid withdrawal:   20 pts
        - After-hours:        10 pts
        - Negative amount:     5 pts

    Args:
        data (pd.DataFrame): ATM transaction records.

    Returns:
        pd.DataFrame: Original data with added columns:
            ['flag_rapid', 'flag_travel', 'flag_hours', 'flag_value',
             'flag_negative', 'anomaly_score', 'risk_level']
    """
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")

    df["flag_rapid"]    = _flag_rapid_withdrawals(df)
    df["flag_travel"]   = _flag_impossible_travel(df)
    df["flag_hours"]    = _flag_after_hours(df)
    df["flag_value"]    = _flag_high_value(df)
    df["flag_negative"] = _flag_negative_amounts(df)

    df["anomaly_score"] = (
        df["flag_travel"].astype(int)   * 40 +
        df["flag_value"].astype(int)    * 25 +
        df["flag_rapid"].astype(int)    * 20 +
        df["flag_hours"].astype(int)    * 10 +
        df["flag_negative"].astype(int) * 5
    ).clip(0, 100)

    df["risk_level"] = pd.cut(
        df["anomaly_score"],
        bins=[-1, 0, 30, 60, 100],
        labels=["✅ Normal", "🟡 Low Risk", "🟠 Medium Risk", "🔴 High Risk"]
    )

    return df


def security_analysis(data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Full security analysis pipeline. Returns a markdown summary string
    and the scored DataFrame for display in the app.

    Args:
        data (pd.DataFrame): Raw ATM transaction records.

    Returns:
        tuple:
            - summary (str): Markdown-formatted risk report.
            - scored_df (pd.DataFrame): Transactions with anomaly scores.
    """
    df = compute_anomaly_scores(data)
    results = {}

    # --- Data Quality ---
    results["Total Records"]    = f"{len(df):,}"
    results["Missing Values"]   = f"{df.isnull().sum().sum():,} total missing entries"
    results["Duplicate Rows"]   = f"{df.duplicated().sum():,} duplicate rows detected"

    # --- Flag Counts ---
    results["Rapid Withdrawals (< 1 min apart)"]           = f"{df['flag_rapid'].sum():,} events"
    results["Impossible Travel (diff ATM < 10 min)"]       = f"{df['flag_travel'].sum():,} events"
    results["After-Hours Transactions (00h–04h)"]          = f"{df['flag_hours'].sum():,} events"
    results["High-Value Outliers (IQR method)"]            = f"{df['flag_value'].sum():,} events"
    results["Negative Transactions"]                       = f"{df['flag_negative'].sum():,} events"

    # --- Anomaly Score Distribution ---
    high_risk_count = (df["anomaly_score"] >= 61).sum()
    medium_risk_count = df["anomaly_score"].between(31, 60).sum()
    results["High Risk Transactions (score ≥ 61)"]     = f"{high_risk_count:,}"
    results["Medium Risk Transactions (score 31–60)"]  = f"{medium_risk_count:,}"

    # --- Amount Stats ---
    if "amount" in df.columns:
        results["Total Amount Analysed"]   = f"R {df['amount'].sum():,.2f}"
        results["Average Transaction"]     = f"R {df['amount'].mean():,.2f}"
        results["Largest Transaction"]     = f"R {df['amount'].max():,.2f}"

    # --- Build Summary ---
    summary = "## 🛡️ Security & Fraud Risk Report\n\n"
    for key, value in results.items():
        summary += f"- **{key}:** {value}\n"

    # Overall risk verdict
    if high_risk_count > 10:
        summary += "\n\n---\n### ⚠️ VERDICT: HIGH RISK — Immediate review recommended."
    elif high_risk_count > 0 or medium_risk_count > 5:
        summary += "\n\n---\n### 🟡 VERDICT: MODERATE RISK — Monitor flagged transactions."
    else:
        summary += "\n\n---\n### ✅ VERDICT: LOW RISK — No major anomalies detected."

    return summary, df
