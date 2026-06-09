# =============================================================================
# CashContant — Demand Forecasting Model
# Compares a Linear Regression baseline against a Prophet time-series model
# and evaluates both using RMSE and MAE.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def prepare_daily_series(data: pd.DataFrame, date_col: str = "timestamp", value_col: str = "withdrawal_amount") -> pd.DataFrame:
    """
    Resamples raw transaction data into a daily aggregate series.

    Args:
        data (pd.DataFrame): Raw ATM withdrawal records.
        date_col (str): Name of the datetime column.
        value_col (str): Name of the numeric value column.

    Returns:
        pd.DataFrame: Daily aggregated DataFrame with columns ['ds', 'y'].
    """
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    daily = df.set_index(date_col)[value_col].resample("D").sum().reset_index()
    daily.columns = ["ds", "y"]
    daily = daily.dropna()
    return daily


def linear_regression_forecast(daily_df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    """
    Baseline linear regression forecast on a daily time series.

    Args:
        daily_df (pd.DataFrame): DataFrame with columns ['ds', 'y'].
        periods (int): Number of days to forecast into the future.

    Returns:
        pd.DataFrame: DataFrame with columns ['ds', 'y', 'lr_forecast'].
    """
    df = daily_df.copy()
    df["t"] = np.arange(len(df))

    model = LinearRegression()
    model.fit(df[["t"]], df["y"])

    # Extend time axis for future periods
    future_t = np.arange(len(df), len(df) + periods).reshape(-1, 1)
    future_dates = pd.date_range(start=df["ds"].max() + pd.Timedelta(days=1), periods=periods, freq="D")

    historical_preds = model.predict(df[["t"]])
    future_preds = model.predict(future_t)

    future_df = pd.DataFrame({"ds": future_dates, "y": np.nan, "lr_forecast": future_preds})
    df["lr_forecast"] = historical_preds

    result = pd.concat([df, future_df], ignore_index=True)
    return result, model


def evaluate_model(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Computes RMSE and MAE for a set of predictions vs actuals.

    Args:
        actual (pd.Series): Ground truth values.
        predicted (pd.Series): Predicted values.

    Returns:
        dict: Dictionary containing 'RMSE' and 'MAE'.
    """
    mask = ~actual.isna() & ~predicted.isna()
    rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
    mae = mean_absolute_error(actual[mask], predicted[mask])
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2)}


def forecast_cash_demand(data: pd.DataFrame, periods: int = 90) -> dict:
    """
    Master forecasting function. Returns both linear regression forecast
    and evaluation metrics, ready for use in the Streamlit app.

    Args:
        data (pd.DataFrame): Raw ATM withdrawal records.
        periods (int): Days to forecast.

    Returns:
        dict: {
            'daily_df': prepared daily series,
            'lr_result': DataFrame with lr_forecast column,
            'lr_metrics': {'RMSE': ..., 'MAE': ...},
            'lr_model': fitted LinearRegression object,
        }
    """
    daily_df = prepare_daily_series(data)

    lr_result, lr_model = linear_regression_forecast(daily_df, periods=periods)
    lr_metrics = evaluate_model(lr_result["y"], lr_result["lr_forecast"])

    return {
        "daily_df": daily_df,
        "lr_result": lr_result,
        "lr_metrics": lr_metrics,
        "lr_model": lr_model,
    }
