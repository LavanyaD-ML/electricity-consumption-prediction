import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "outputs" / "cleaned_electricity_data.csv"
MODEL_PATH = ROOT / "models" / "advanced_electricity_model.joblib"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD AND PREPARE DAILY DATA
# ==========================================

def load_daily_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required = ["Date", "Global_active_power"]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )

    df = df.dropna(subset=required)
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")

    daily = (
        df.set_index("Date")["Global_active_power"]
        .resample("D")
        .mean()
    )

    missing_days = int(daily.isna().sum())

    if missing_days:
        print(f"Missing calendar days detected: {missing_days}")

        daily = daily.interpolate(
            method="time",
            limit_area="inside"
        )

    if daily.isna().any():
        raise ValueError(
            "Missing values remain at the beginning or end of the series."
        )

    if len(daily) < 150:
        raise ValueError(
            "At least 150 daily observations are required."
        )

    print(f"Daily observations loaded: {len(daily)}")

    return daily


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(daily):

    data = daily.to_frame(name="target")

    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear
    data["is_weekend"] = (
        data.index.dayofweek >= 5
    ).astype(int)

    for lag in [1, 2, 3, 7, 14, 28]:
        data[f"lag_{lag}"] = data["target"].shift(lag)

    for window in [7, 14, 28]:
        previous_values = data["target"].shift(1).rolling(window)

        data[f"rolling_mean_{window}"] = previous_values.mean()
        data[f"rolling_std_{window}"] = previous_values.std()

    data = data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    return data


# ==========================================
# FORECASTING BACKTEST
# ==========================================

def main():

    print("=" * 60)
    print("ELECTRICITY FORECASTING BACKTEST")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Advanced model not found: {MODEL_PATH}. "
            "Run advanced_modeling.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    # Support both saved model formats
    if isinstance(artifact, dict):
        model = artifact["model"]
        expected_features = artifact.get("features")
    else:
        model = artifact
        expected_features = None

    daily = load_daily_data()
    data = create_features(daily)

    X = data.drop(columns=["target"])
    y = data["target"]

    if expected_features is not None:
        missing_features = [
            feature for feature in expected_features
            if feature not in X.columns
        ]

        if missing_features:
            raise ValueError(
                f"Model features are missing: {missing_features}"
            )

        X = X[expected_features]

    # Use the final 15% as chronological test data
    split_index = int(len(data) * 0.85)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    test_dates = data.index[split_index:]

    if len(X_test) == 0:
        raise ValueError("No testing observations available.")

    # Predictions for the held-out chronological period
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # Save predictions
    results = pd.DataFrame({
        "Date": test_dates,
        "Actual_kW": y_test.to_numpy(),
        "Predicted_kW": predictions,
        "Error_kW": y_test.to_numpy() - predictions
    })

    results_path = OUTPUT_DIR / "forecast_backtest_predictions.csv"
    results.to_csv(results_path, index=False)

    # Compare against a simple previous-day baseline
    baseline_predictions = daily.shift(1).reindex(test_dates)

    valid_baseline = baseline_predictions.notna()

    baseline_mae = mean_absolute_error(
        y_test.loc[valid_baseline.to_numpy()],
        baseline_predictions.loc[valid_baseline].to_numpy()
    )

    metrics = {
        "evaluation": "Chronological holdout test",
        "model": artifact.get("best_model_name", "Saved model")
        if isinstance(artifact, dict) else type(model).__name__,
        "test_observations": int(len(y_test)),
        "MAE_kW": float(mae),
        "RMSE_kW": float(rmse),
        "R2": float(r2),
        "previous_day_baseline_MAE_kW": float(baseline_mae)
    }

    metrics_path = OUTPUT_DIR / "forecast_backtest_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nBACKTEST RESULTS")
    print("-" * 40)
    print(f"Test observations: {len(y_test)}")
    print(f"Model MAE: {mae:.4f} kW")
    print(f"Model RMSE: {rmse:.4f} kW")
    print(f"Model R²: {r2:.4f}")
    print(f"Previous-day baseline MAE: {baseline_mae:.4f} kW")

    if mae < baseline_mae:
        print("Result: Model MAE is lower than the baseline MAE.")
    else:
        print("Result: Model MAE is not lower than the baseline MAE.")

    print("\nSaved files:")
    print("- outputs/forecast_backtest_predictions.csv")
    print("- outputs/forecast_backtest_metrics.json")

    print("\nForecasting backtest completed.")


if __name__ == "__main__":
    main()