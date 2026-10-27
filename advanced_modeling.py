import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# PROJECT PATHS
# =========================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "outputs" / "cleaned_electricity_data.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# DATA PREPROCESSING
# =========================

def create_features(df):
    df = df.copy()

    required_columns = ["Date", "Global_active_power"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Global_active_power"]
    )

    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    daily = (
        df.set_index("Date")["Global_active_power"]
        .resample("D")
        .mean()
    )

    missing_days = int(daily.isna().sum())

    if missing_days > 0:
        print(f"Missing calendar days detected: {missing_days}")

        daily = daily.interpolate(
            method="time",
            limit_area="inside"
        )

        print("Internal missing daily values interpolated.")

    if daily.isna().any():
        raise ValueError(
            "Missing values remain at the start or end of the daily series."
        )

    if len(daily) < 100:
        raise ValueError(
            "Not enough daily observations for reliable model training."
        )

    data = daily.to_frame(name="target")

    # Calendar features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear
    data["is_weekend"] = (
        data.index.dayofweek >= 5
    ).astype(int)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 28]:
        data[f"lag_{lag}"] = data["target"].shift(lag)

    # Rolling features using previous observations only
    for window in [7, 14, 28]:
        previous_values = data["target"].shift(1).rolling(
            window=window
        )

        data[f"rolling_mean_{window}"] = (
            previous_values.mean()
        )

        data[f"rolling_std_{window}"] = (
            previous_values.std()
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(data) < 100:
        raise ValueError(
            "Not enough observations remain after feature creation."
        )

    return data


# =========================
# MODEL TRAINING
# =========================

def main():

    print("=" * 60)
    print("ADVANCED ELECTRICITY CONSUMPTION PREDICTION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Run data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nOriginal dataset shape: {df.shape}")

    data = create_features(df)

    X = data.drop(columns=["target"])
    y = data["target"]

    dates = data.index

    # Chronological split: do not randomly shuffle time-series data
    split_index = int(len(data) * 0.85)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = dates[split_index:]

    print(f"Total observations: {len(data)}")
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    if len(X_train) < 100 or len(X_test) < 10:
        raise ValueError(
            "Insufficient training or testing observations."
        )

    if X.isna().any().any() or y.isna().any():
        raise ValueError(
            "Missing values remain in the modelling dataset."
        )

    # Time-series cross-validation
    cv_splits = min(5, max(2, len(X_train) // 100))

    time_cv = TimeSeriesSplit(n_splits=cv_splits)

    # Model configurations
    model_configurations = {
        "RandomForest": {
            "model": RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),
            "parameters": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [1.0, "sqrt", 0.8]
            }
        },

        "ExtraTrees": {
            "model": ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1
            ),
            "parameters": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [1.0, "sqrt", 0.8]
            }
        }
    }

    results = []
    best_model = None
    best_model_name = None
    best_cv_mae = float("inf")

    # Train and tune both models
    for model_name, configuration in model_configurations.items():

        print(f"\nTraining and tuning: {model_name}")

        search = RandomizedSearchCV(
            estimator=configuration["model"],
            param_distributions=configuration["parameters"],
            n_iter=8,
            scoring="neg_mean_absolute_error",
            cv=time_cv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        current_model = search.best_estimator_

        predictions = current_model.predict(X_test)

        test_mae = mean_absolute_error(
            y_test,
            predictions
        )

        test_rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        test_r2 = r2_score(
            y_test,
            predictions
        )

        cv_mae = -search.best_score_

        results.append({
            "model": model_name,
            "best_cv_MAE_kW": cv_mae,
            "test_MAE_kW": test_mae,
            "test_RMSE_kW": test_rmse,
            "test_R2": test_r2,
            "best_parameters": json.dumps(
                search.best_params_
            )
        })

        print(f"Best CV MAE: {cv_mae:.4f} kW")
        print(f"Test MAE: {test_mae:.4f} kW")
        print(f"Test RMSE: {test_rmse:.4f} kW")
        print(f"Test R²: {test_r2:.4f}")

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_model = current_model
            best_model_name = model_name

    # Save model comparison
    comparison = pd.DataFrame(results)

    comparison_path = (
        OUTPUT_DIR / "advanced_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    # Predictions from the selected model
    final_predictions = best_model.predict(X_test)

    prediction_results = pd.DataFrame({
        "Date": test_dates,
        "Actual_kW": y_test.to_numpy(),
        "Predicted_kW": final_predictions,
        "Error_kW": (
            y_test.to_numpy() - final_predictions
        )
    })

    prediction_path = (
        OUTPUT_DIR / "advanced_test_predictions.csv"
    )

    prediction_results.to_csv(
        prediction_path,
        index=False
    )

    # Save feature importance
    if hasattr(best_model, "feature_importances_"):

        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        feature_importance.to_csv(
            OUTPUT_DIR / "advanced_feature_importance.csv",
            index=False
        )

    # Final metrics
    final_mae = mean_absolute_error(
        y_test,
        final_predictions
    )

    final_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            final_predictions
        )
    )

    final_r2 = r2_score(
        y_test,
        final_predictions
    )

    metrics = {
        "best_model": best_model_name,
        "best_cv_MAE_kW": float(best_cv_mae),
        "test_MAE_kW": float(final_mae),
        "test_RMSE_kW": float(final_rmse),
        "test_R2": float(final_r2),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": list(X.columns)
    }

    with open(
        OUTPUT_DIR / "advanced_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save model and required feature names for the dashboard
    model_artifact = {
        "model": best_model,
        "features": list(X.columns),
        "target": "Global_active_power",
        "target_unit": "kW",
        "best_model_name": best_model_name
    }

    joblib.dump(
        model_artifact,
        MODEL_DIR / "advanced_electricity_model.joblib"
    )

    print("\n" + "=" * 60)
    print("ADVANCED MODELING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Selected model: {best_model_name}")
    print(f"Test MAE: {final_mae:.4f} kW")
    print(f"Test RMSE: {final_rmse:.4f} kW")
    print(f"Test R²: {final_r2:.4f}")

    print("\nSaved files:")
    print("- models/advanced_electricity_model.joblib")
    print("- outputs/advanced_model_comparison.csv")
    print("- outputs/advanced_test_predictions.csv")
    print("- outputs/advanced_feature_importance.csv")
    print("- outputs/advanced_test_metrics.json")


if __name__ == "__main__":
    main()
