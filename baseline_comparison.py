from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATA_FILE = OUTPUT_DIR / "cleaned_electricity_data.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "model_predictions.csv"
RESULT_FILE = OUTPUT_DIR / "baseline_comparison.csv"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Model predictions not found. Run data/src/model_training.py first."
        )

    data = pd.read_csv(DATA_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    predictions["Date"] = pd.to_datetime(
        predictions["Date"], errors="coerce"
    )

    data["Global_active_power"] = pd.to_numeric(
        data["Global_active_power"], errors="coerce"
    )
    predictions["Actual_kW"] = pd.to_numeric(
        predictions["Actual_kW"], errors="coerce"
    )

    data = data.dropna(
        subset=["Date", "Global_active_power"]
    ).drop_duplicates("Date")

    predictions = predictions.dropna(
        subset=["Date", "Actual_kW", "Predicted_kW"]
    ).drop_duplicates("Date")

    # Use the exact dates from the model's test set.
    test_data = predictions[["Date", "Actual_kW", "Predicted_kW"]].copy()

    actual_data = data.rename(
        columns={"Global_active_power": "Dataset_Actual_kW"}
    )

    test_data = test_data.merge(
        actual_data[["Date", "Dataset_Actual_kW"]],
        on="Date",
        how="left",
        validate="one_to_one",
    )

    if test_data["Dataset_Actual_kW"].isna().any():
        raise ValueError(
            "Some test dates could not be matched to the daily dataset."
        )

    if not np.allclose(
        test_data["Actual_kW"],
        test_data["Dataset_Actual_kW"],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            "Prediction actual values do not match the dataset."
        )

    actual = test_data["Actual_kW"].to_numpy()

    # Baseline: predict the previous available daily observation.
    ordered_data = data.sort_values("Date").reset_index(drop=True)
    ordered_data["Baseline_kW"] = (
        ordered_data["Global_active_power"].shift(1)
    )

    baseline = test_data[["Date", "Actual_kW"]].merge(
        ordered_data[["Date", "Baseline_kW"]],
        on="Date",
        how="left",
        validate="one_to_one",
    )

    if baseline["Baseline_kW"].isna().any():
        raise ValueError("Baseline predictions are missing for test dates.")

    baseline_values = baseline["Baseline_kW"].to_numpy()
    model_values = test_data["Predicted_kW"].to_numpy()

    rows = []

    for name, values in [
        ("Previous-day baseline", baseline_values),
        ("Trained ML model", model_values),
    ]:
        rows.append({
            "Model": name,
            "MAE_kW": mean_absolute_error(actual, values),
            "RMSE_kW": np.sqrt(mean_squared_error(actual, values)),
            "R2": r2_score(actual, values),
        })

    results = pd.DataFrame(rows)

    baseline_mae = results.loc[
        results["Model"] == "Previous-day baseline", "MAE_kW"
    ].iloc[0]

    model_mae = results.loc[
        results["Model"] == "Trained ML model", "MAE_kW"
    ].iloc[0]

    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae > 0
        else np.nan
    )

    results["MAE_Improvement_vs_Baseline_Percent"] = np.nan
    results.loc[
        results["Model"] == "Trained ML model",
        "MAE_Improvement_vs_Baseline_Percent",
    ] = improvement

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)

    print("Baseline comparison completed.")
    print(results.to_string(index=False))
    print(f"\nMAE improvement over baseline: {improvement:.2f}%")
    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()