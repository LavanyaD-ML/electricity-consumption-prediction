from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "error_analysis_plots"

PREDICTIONS_FILE = OUTPUT_DIR / "model_predictions.csv"


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Predictions not found. Run data/src/model_training.py first."
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required = ["Date", "Actual_kW", "Predicted_kW"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for column in ["Actual_kW", "Predicted_kW"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid prediction records available.")

    df["Error_kW"] = df["Actual_kW"] - df["Predicted_kW"]
    df["Absolute_Error_kW"] = df["Error_kW"].abs()
    df["Squared_Error"] = df["Error_kW"] ** 2
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["Date"].dt.day_name()

    mae = mean_absolute_error(df["Actual_kW"], df["Predicted_kW"])
    rmse = np.sqrt(
        mean_squared_error(df["Actual_kW"], df["Predicted_kW"])
    )

    summary = pd.DataFrame([{
        "Test_Records": len(df),
        "MAE_kW": mae,
        "RMSE_kW": rmse,
        "Mean_Error_kW": df["Error_kW"].mean(),
        "Median_Absolute_Error_kW": df["Absolute_Error_kW"].median(),
        "Maximum_Absolute_Error_kW": df["Absolute_Error_kW"].max(),
        "Error_Std_kW": df["Error_kW"].std(),
    }])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "error_analysis_summary.csv",
        index=False,
    )

    monthly_errors = (
        df.groupby("Month")
        .agg(
            MAE_kW=("Absolute_Error_kW", "mean"),
            Mean_Error_kW=("Error_kW", "mean"),
            Records=("Error_kW", "size"),
        )
        .reset_index()
    )
    monthly_errors.to_csv(
        OUTPUT_DIR / "monthly_error_analysis.csv",
        index=False,
    )

    weekday_errors = (
        df.groupby("DayOfWeek")
        .agg(
            MAE_kW=("Absolute_Error_kW", "mean"),
            Mean_Error_kW=("Error_kW", "mean"),
            Records=("Error_kW", "size"),
        )
        .reset_index()
    )
    weekday_errors.to_csv(
        OUTPUT_DIR / "weekday_error_analysis.csv",
        index=False,
    )

    sns.set_theme(style="whitegrid")

    # 1. Actual versus predicted values
    plt.figure(figsize=(8, 7))
    sns.scatterplot(
        data=df,
        x="Actual_kW",
        y="Predicted_kW",
        alpha=0.6,
    )
    low = min(df["Actual_kW"].min(), df["Predicted_kW"].min())
    high = max(df["Actual_kW"].max(), df["Predicted_kW"].max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title("Actual vs Predicted Electricity Consumption")
    plt.xlabel("Actual (kW)")
    plt.ylabel("Predicted (kW)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 2. Prediction error distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="Error_kW", bins=40, kde=True)
    plt.axvline(0, linestyle="--")
    plt.title("Prediction Error Distribution")
    plt.xlabel("Actual - Predicted (kW)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "error_distribution.png", dpi=150)
    plt.close()

    # 3. Error over time
    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["Error_kW"], linewidth=0.8)
    plt.axhline(0, linestyle="--")
    plt.title("Prediction Errors Over Time")
    plt.xlabel("Date")
    plt.ylabel("Error (kW)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "errors_over_time.png", dpi=150)
    plt.close()

    # 4. Absolute error over time
    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["Absolute_Error_kW"], linewidth=0.8)
    plt.title("Absolute Prediction Error Over Time")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error (kW)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "absolute_errors_over_time.png", dpi=150)
    plt.close()

    # 5. Monthly error comparison
    plt.figure(figsize=(14, 5))
    sns.barplot(data=monthly_errors, x="Month", y="MAE_kW")
    plt.title("Monthly Model Error (MAE)")
    plt.xlabel("Month")
    plt.ylabel("MAE (kW)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "monthly_mae.png", dpi=150)
    plt.close()

    # 6. Error by weekday
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    weekday_errors["DayOfWeek"] = pd.Categorical(
        weekday_errors["DayOfWeek"],
        categories=weekday_order,
        ordered=True,
    )
    weekday_errors = weekday_errors.sort_values("DayOfWeek")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=weekday_errors, x="DayOfWeek", y="MAE_kW")
    plt.title("Prediction Error by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("MAE (kW)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "weekday_mae.png", dpi=150)
    plt.close()

    # Save individual prediction errors
    df.to_csv(
        OUTPUT_DIR / "detailed_prediction_errors.csv",
        index=False,
    )

    print("Error analysis completed successfully.")
    print(summary.to_string(index=False))
    print(f"\nCharts saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()