from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_FILE = OUTPUT_DIR / "cleaned_electricity_data.csv"
PLOT_DIR = OUTPUT_DIR / "eda_plots"


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run data/src/data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = ["Date", "Global_active_power"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )

    df = df.dropna(subset=required_columns)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])

    if df.empty:
        raise ValueError("No valid data available for analysis.")

    return df


def analyze_data(df):
    df = df.copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["DayOfYear"] = df["Date"].dt.dayofyear

    summary = df["Global_active_power"].describe()
    summary.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    monthly = (
        df.groupby(["Year", "Month"])["Global_active_power"]
        .mean()
        .reset_index()
    )
    monthly.to_csv(OUTPUT_DIR / "monthly_usage_summary.csv", index=False)

    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    weekday = (
        df.groupby("DayOfWeek")["Global_active_power"]
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )
    weekday.to_csv(OUTPUT_DIR / "weekday_usage_summary.csv", index=False)

    return df, monthly, weekday


def create_plots(df, monthly, weekday):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Daily consumption trend
    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["Global_active_power"], linewidth=0.8)
    plt.title("Daily Average Electricity Consumption")
    plt.xlabel("Date")
    plt.ylabel("Global Active Power (kW)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "daily_consumption_trend.png", dpi=150)
    plt.close()

    # 2. Monthly usage pattern
    monthly["Period"] = (
        monthly["Year"].astype(str)
        + "-"
        + monthly["Month"].astype(str).str.zfill(2)
    )

    plt.figure(figsize=(14, 5))
    plt.plot(monthly["Period"], monthly["Global_active_power"], marker=".")
    plt.title("Monthly Average Electricity Consumption")
    plt.xlabel("Year-Month")
    plt.ylabel("Global Active Power (kW)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "monthly_consumption_trend.png", dpi=150)
    plt.close()

    # 3. Consumption distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="Global_active_power",
        bins=40,
        kde=True,
    )
    plt.title("Distribution of Daily Electricity Consumption")
    plt.xlabel("Global Active Power (kW)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "consumption_distribution.png", dpi=150)
    plt.close()

    # 4. Weekday comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=weekday,
        x="DayOfWeek",
        y="Global_active_power",
    )
    plt.title("Average Consumption by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Global Active Power (kW)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "weekday_consumption.png", dpi=150)
    plt.close()

    # 5. Yearly comparison
    yearly = (
        df.groupby("Year")["Global_active_power"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=yearly, x="Year", y="Global_active_power")
    plt.title("Average Electricity Consumption by Year")
    plt.xlabel("Year")
    plt.ylabel("Global Active Power (kW)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "yearly_consumption.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df, monthly, weekday = analyze_data(df)
    create_plots(df, monthly, weekday)

    print("EDA completed successfully.")
    print(f"Daily records analysed: {len(df):,}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print("\nDescriptive statistics:")
    print(df["Global_active_power"].describe())
    print(f"\nCharts saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()