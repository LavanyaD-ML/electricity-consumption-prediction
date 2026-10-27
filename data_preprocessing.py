from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "cleaned_electricity_data.csv"


def find_dataset():
    possible_files = [
        DATA_DIR / "household_power_consumption.txt",
        DATA_DIR / "household_power_consumption.csv",
        DATA_DIR / "electricity_data.csv",
        DATA_DIR / "raw" / "household_power_consumption.txt",
        DATA_DIR / "raw" / "household_power_consumption.csv",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        "Raw dataset not found. Place the original dataset inside the data folder."
    )


def load_dataset(file_path):
    if file_path.suffix.lower() == ".txt":
        df = pd.read_csv(
            file_path,
            sep=";",
            na_values=["?", ""],
            low_memory=False,
        )
    else:
        df = pd.read_csv(
            file_path,
            na_values=["?", ""],
            low_memory=False,
        )

    df.columns = df.columns.str.strip()

    required_columns = [
        "Date",
        "Time",
        "Global_active_power",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    return df


def clean_dataset(df):
    df = df.copy()

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Time"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(subset=["DateTime"])

    numeric_columns = [
        column for column in df.columns
        if column not in ["Date", "Time", "DateTime"]
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.sort_values("DateTime")

    df = df.drop_duplicates(
        subset=["DateTime"],
        keep="first",
    )

    df = df.dropna(
        subset=["Global_active_power"]
    )

    df = df[
        df["Global_active_power"] >= 0
    ]

    df = df.set_index("DateTime")

    return df


def create_daily_dataset(df):
    daily = df.resample("D").agg(
        {
            "Global_active_power": "mean",
        }
    )

    daily = daily.rename_axis("Date").reset_index()

    daily = daily.dropna(
        subset=["Global_active_power"]
    )

    return daily


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = find_dataset()

    print(f"Dataset found: {dataset_path}")

    raw_df = load_dataset(dataset_path)

    print(f"Original rows: {len(raw_df):,}")

    cleaned_df = clean_dataset(raw_df)

    daily_df = create_daily_dataset(cleaned_df)

    if daily_df.empty:
        raise ValueError(
            "No valid daily data available."
        )

    daily_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Cleaned rows: {len(cleaned_df):,}")
    print(f"Daily rows: {len(daily_df):,}")
    print(f"Missing values:\n{daily_df.isnull().sum()}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()