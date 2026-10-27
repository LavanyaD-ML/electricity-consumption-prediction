from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_FILE = OUTPUT_DIR / "cleaned_electricity_data.csv"
MODEL_FILE = MODEL_DIR / "electricity_model.joblib"


def load_daily_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Run preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)
    required = ["Date", "Global_active_power"]

    if not all(column in df.columns for column in required):
        raise ValueError(f"Dataset must contain: {required}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )

    df = (
        df.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    if len(df) < 100:
        raise ValueError("Not enough daily records to train the model.")

    return df


def create_features(df):
    data = df.copy()
    target = "Global_active_power"

    data["day_of_week"] = data["Date"].dt.dayofweek
    data["month"] = data["Date"].dt.month
    data["day_of_year"] = data["Date"].dt.dayofyear
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    for lag in [1, 2, 3, 7, 14, 28]:
        data[f"lag_{lag}"] = data[target].shift(lag)

    for window in [7, 14, 28]:
        shifted = data[target].shift(1)
        data[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        data[f"rolling_std_{window}"] = shifted.rolling(window).std()

    data = data.dropna().reset_index(drop=True)

    feature_columns = [
        column for column in data.columns
        if column not in ["Date", target]
    ]

    X = data[feature_columns]
    y = data[target]
    dates = data["Date"]

    if X.empty or y.empty:
        raise ValueError("Feature generation produced no usable rows.")

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Missing values remain after feature generation.")

    return X, y, dates, feature_columns


def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_daily_data()
    X, y, dates, feature_columns = create_features(df)

    split_index = int(len(X) * 0.85)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_test = dates.iloc[split_index:]

    if len(X_train) < 50 or len(X_test) == 0:
        raise ValueError("Insufficient training or test records.")

    if dates_test.min() <= dates.iloc[split_index - 1]:
        raise ValueError("Chronological train/test split validation failed.")

    models = {
        "RandomForest": RandomForestRegressor(
            random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            random_state=42, n_jobs=-1
        ),
    }

    parameter_distributions = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, "sqrt", 0.8],
    }

    cv = TimeSeriesSplit(n_splits=5)
    results = []
    best_name = None
    best_search = None
    best_cv_mae = float("inf")

    for name, estimator in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_distributions,
            n_iter=8,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_train, y_train)

        predictions = search.best_estimator_.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)
        cv_mae = -search.best_score_

        results.append({
            "Model": name,
            "CV_MAE_kW": cv_mae,
            "Test_MAE_kW": metrics["MAE"],
            "Test_RMSE_kW": metrics["RMSE"],
            "Test_R2": metrics["R2"],
        })

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_name = name
            best_search = search

    comparison = pd.DataFrame(results).sort_values("CV_MAE_kW")
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    best_model = best_search.best_estimator_
    predictions = best_model.predict(X_test)

    prediction_df = pd.DataFrame({
        "Date": dates_test.values,
        "Actual_kW": y_test.values,
        "Predicted_kW": predictions,
        "Error_kW": y_test.values - predictions,
    })

    prediction_df.to_csv(
        OUTPUT_DIR / "model_predictions.csv",
        index=False,
    )

    importance_values = best_model.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance_values,
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": best_model,
            "features": feature_columns,
            "target": "Global_active_power",
            "target_unit": "kW",
            "best_model_name": best_name,
        },
        MODEL_FILE,
    )

    metrics = calculate_metrics(y_test, predictions)

    pd.DataFrame([{
        "Best_Model": best_name,
        "CV_MAE_kW": best_cv_mae,
        "Test_MAE_kW": metrics["MAE"],
        "Test_RMSE_kW": metrics["RMSE"],
        "Test_R2": metrics["R2"],
        "Training_Rows": len(X_train),
        "Testing_Rows": len(X_test),
    }]).to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False,
    )

    print("\nModel training completed.")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Test MAE: {metrics['MAE']:.4f} kW")
    print(f"Test RMSE: {metrics['RMSE']:.4f} kW")
    print(f"Test R²: {metrics['R2']:.4f}")
    print(f"Model saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()