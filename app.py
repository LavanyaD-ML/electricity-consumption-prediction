# ============================================================
# ELECTRICITY CONSUMPTION PREDICTION & ENERGY USAGE ANALYSIS
# ADVANCED END-TO-END MACHINE LEARNING PROJECT
# PART 1/4
# Setup, Data Ingestion, Preprocessing, Feature Engineering
# ============================================================

from pathlib import Path
from io import BytesIO
import json
import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    LogisticRegression,
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
)

from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Electricity Consumption ML",
    page_icon="⚡",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REGRESSION_MODEL_PATH = (
    ARTIFACT_DIR / "best_regression_model.joblib"
)



METADATA_PATH = (
    ARTIFACT_DIR / "model_metadata.json"
)


TARGET_CANDIDATES = [
    "Global_active_power",
    "global_active_power",
    "Global Active Power",
    "active_power",
    "ActivePower",
    "consumption",
    "Consumption",
    "power",
    "Power",
    "energy",
    "Energy",
]


# ============================================================
# 2. STREAMLIT PAGE DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">'
    '⚡ Electricity Consumption Prediction'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'End-to-end machine learning, energy analysis and forecasting'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA INGESTION
# Supports CSV, TXT, semicolon-separated files and ? missing data
# ============================================================

@st.cache_data(show_spinner=False)
def read_uploaded_data(file_bytes, filename):

    suffix = Path(filename).suffix.lower()

    if suffix not in [".csv", ".txt"]:
        raise ValueError(
            "Please upload a CSV or TXT file."
        )

    buffer = BytesIO(file_bytes)

    df = pd.read_csv(
        buffer,
        sep=None,
        engine="python",
        na_values=[
            "?",
            "NA",
            "N/A",
            "null",
            "None",
            "",
        ],
    )

    df.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in df.columns
    ]

    if df.empty:
        raise ValueError(
            "The uploaded file contains no data."
        )

    return df


# ============================================================
# 4. DATETIME COLUMN DETECTION
# ============================================================

def detect_datetime_column(df):

    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    # Case 1: Combined timestamp column.
    combined_candidates = [
        "datetime",
        "timestamp",
        "date_time",
        "date time",
    ]

    for candidate in combined_candidates:

        if candidate in columns:

            original = columns[candidate]

            parsed = pd.to_datetime(
                df[original],
                errors="coerce",
                dayfirst=True,
            )

            if parsed.notna().mean() >= 0.70:
                return parsed

    # Case 2: Separate Date and Time columns.
    date_column = columns.get("date")
    time_column = columns.get("time")

    if date_column is not None:

        date_values = (
            df[date_column]
            .astype("string")
            .str.strip()
        )

        if time_column is not None:

            time_values = (
                df[time_column]
                .astype("string")
                .str.strip()
            )

            combined = date_values + " " + time_values

        else:
            combined = date_values

        parsed = pd.to_datetime(
            combined,
            errors="coerce",
            dayfirst=True,
        )

        if parsed.notna().mean() >= 0.70:
            return parsed

    raise ValueError(
        "A valid Date/Time or Datetime/Timestamp column "
        "is required. Separate date and time columns should "
        "be named Date and Time."
    )


# ============================================================
# 5. AUTOMATIC TARGET COLUMN DETECTION
# ============================================================

def detect_target_column(df):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in TARGET_CANDIDATES:

        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ============================================================
# 6. DATA VALIDATION AND CLEANING
# ============================================================

def validate_and_clean(raw_df, target_column):

    df = raw_df.copy()

    df.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in df.columns
    ]

    original_rows = len(df)

    # Remove fully duplicated records.
    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    # Parse timestamps.
    df["_timestamp"] = detect_datetime_column(df)

    # Validate target column.
    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    # Convert measurement columns to numeric values.
    protected_columns = {
        target_column,
        "_timestamp",
        "Date",
        "Time",
        "datetime",
        "timestamp",
    }

    for column in df.columns:

        if column not in protected_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    # Count invalid timestamps.
    invalid_timestamp_count = int(
        df["_timestamp"].isna().sum()
    )

    # Remove records without timestamps or targets.
    df = df.dropna(
        subset=[
            "_timestamp",
            target_column,
        ]
    ).copy()

    # Negative target values are invalid for this project.
    negative_target_count = int(
        (df[target_column] < 0).sum()
    )

    df = df[
        df[target_column] >= 0
    ].copy()

    if df.empty:

        raise ValueError(
            "No valid observations remain after cleaning."
        )

    # Aggregate readings into hourly observations.
    df["_hour"] = (
        df["_timestamp"].dt.floor("h")
    )

    numeric_columns = (
        df.select_dtypes(include=[np.number])
        .columns.tolist()
    )

    if target_column not in numeric_columns:

        raise ValueError(
            "The selected target must be numeric."
        )

    hourly = (
        df.groupby(
            "_hour",
            sort=True,
        )[numeric_columns]
        .mean()
        .reset_index()
        .rename(
            columns={
                "_hour": "timestamp"
            }
        )
    )

    hourly = (
        hourly.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Cleaning summary.
    report = {
        "Original rows": original_rows,

        "Duplicate rows removed": (
            duplicate_count
        ),

        "Invalid timestamps": (
            invalid_timestamp_count
        ),

        "Negative target rows removed": (
            negative_target_count
        ),

        "Rows after validation": len(df),

        "Hourly observations": len(hourly),

        "Missing target values remaining": int(
            hourly[target_column].isna().sum()
        ),
    }

    return hourly, report


# ============================================================
# 7. LEAKAGE-AWARE FEATURE ENGINEERING
# ============================================================

def build_supervised_dataset(
    hourly_df,
    target_column,
):

    df = (
        hourly_df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = df.dropna(
        subset=[target_column]
    ).copy()

    target = df[target_column].astype(float)

    features = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    features["hour"] = (
        df["timestamp"].dt.hour
    )

    features["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    features["day_of_month"] = (
        df["timestamp"].dt.day
    )

    features["month"] = (
        df["timestamp"].dt.month
    )

    features["is_weekend"] = (
        df["timestamp"].dt.dayofweek >= 5
    ).astype(int)

    # Cyclical encoding for time-related features.
    features["hour_sin"] = np.sin(
        2 * np.pi * features["hour"] / 24
    )

    features["hour_cos"] = np.cos(
        2 * np.pi * features["hour"] / 24
    )

    features["weekday_sin"] = np.sin(
        2 * np.pi * features["day_of_week"] / 7
    )

    features["weekday_cos"] = np.cos(
        2 * np.pi * features["day_of_week"] / 7
    )

    features["month_sin"] = np.sin(
        2 * np.pi * features["month"] / 12
    )

    features["month_cos"] = np.cos(
        2 * np.pi * features["month"] / 12
    )

    # --------------------------------------------------------
    # Historical lag features
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3,
        6,
        12,
        24,
        48,
        168,
    ]:

        features[f"lag_{lag}"] = (
            target.shift(lag)
        )

    # --------------------------------------------------------
    # Rolling statistics
    # Shift BEFORE calculating rolling statistics.
    # This prevents the current target entering its own features.
    # --------------------------------------------------------

    previous = target.shift(1)

    for window in [
        3,
        6,
        12,
        24,
        168,
    ]:

        features[
            f"rolling_mean_{window}"
        ] = (
            previous.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )

    features["rolling_std_24"] = (
        previous.rolling(
            window=24,
            min_periods=24,
        ).std()
    )

    features["rolling_max_24"] = (
        previous.rolling(
            window=24,
            min_periods=24,
        ).max()
    )

    features["rolling_min_24"] = (
        previous.rolling(
            window=24,
            min_periods=24,
        ).min()
    )

    # Exclude rows without complete historical features.
    valid = (
        features.notna().all(axis=1)
        & target.notna()
    )

    X = features.loc[valid].copy()

    y = target.loc[valid].copy()

    timestamps = (
        df.loc[valid, "timestamp"].copy()
    )

    return X, y, timestamps



# ============================================================
# 8. REGRESSION MODELS
# ============================================================

def get_regression_models():

    models = {

        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]),

        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            ),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=3,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=100,
                    min_samples_leaf=3,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    loss="squared_error",
                    random_state=42,
                ),
            ),
        ]),

        "KNN Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=7,
                    weights="distance",
                    n_jobs=-1,
                ),
            ),
        ]),
    }

    return models





# ============================================================
# 10. REGRESSION EVALUATION METRICS
# ============================================================

def calculate_regression_metrics(
    y_true,
    predictions,
):

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }




# ============================================================
# 12. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

def chronological_split(
    X,
    y,
    test_fraction=0.20,
):

    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    split_index = int(
        len(X) * (1 - test_fraction)
    )

    if split_index <= 0 or split_index >= len(X):
        raise ValueError(
            "Not enough observations for train-test splitting."
        )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# 13. REGRESSION TRAINING
# ============================================================

def train_regression_models(
    X_train,
    X_test,
    y_train,
    y_test,
    cv_folds=3,
):

    models = get_regression_models()

    n_splits = min(
        cv_folds,
        len(X_train) - 1,
    )

    if n_splits < 2:
        raise ValueError(
            "Insufficient training observations for "
            "time-series cross-validation."
        )

    time_splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    trained_models = {}
    result_rows = []

    for name, model in models.items():

        with st.spinner(
            f"Training regression model: {name}"
        ):

            # Cross-validation uses training data only.
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=time_splitter,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )

            cv_mae = float(
                -np.mean(cv_scores)
            )

            # Fit on the complete training portion.
            fitted_model = clone(model)

            fitted_model.fit(
                X_train,
                y_train,
            )

            # Evaluate on the untouched test portion.
            predictions = fitted_model.predict(
                X_test
            )

            metrics = calculate_regression_metrics(
                y_test,
                predictions,
            )

            trained_models[name] = fitted_model

            result_rows.append({
                "Model": name,
                "CV MAE": cv_mae,
                "Test MAE": metrics["MAE"],
                "Test RMSE": metrics["RMSE"],
                "Test R2": metrics["R2"],
            })

    results_df = pd.DataFrame(
        result_rows
    ).sort_values(
        "CV MAE",
        ascending=True,
    ).reset_index(drop=True)

    best_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_name]

    best_predictions = best_model.predict(
        X_test
    )

    return {
        "models": trained_models,
        "results": results_df,
        "best_name": best_name,
        "best_model": best_model,
        "predictions": best_predictions,
    }





# ============================================================
# 15. COMPLETE MODEL TRAINING PIPELINE
# ============================================================

def train_all_models(
    hourly_df,
    target_column,
    cv_folds=3,
):
    # Build supervised learning features.
    X, y, timestamps = build_supervised_dataset(
        hourly_df,
        target_column,
    )

    # Check available observations.
    if len(X) < 100:
        raise ValueError(
            "Not enough usable observations after feature engineering."
        )

    # Combine features, target, and timestamps.
    supervised_df = X.copy()
    supervised_df[target_column] = y
    supervised_df["timestamp"] = timestamps

    # Select input features.
    feature_columns = [
        column
        for column in supervised_df.columns
        if column not in ["timestamp", target_column]
    ]

    X = supervised_df[feature_columns].copy()

    y = pd.to_numeric(
        supervised_df[target_column],
        errors="coerce",
    )

    # Remove invalid target values.
    valid_rows = y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    if len(X) < 100:
        raise ValueError(
            "Not enough valid rows for regression training."
        )

    # Chronological train-test split.
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_split(
        X,
        y,
    )

    # Train regression models only.
    regression_result = train_regression_models(
        X_train,
        X_test,
        y_train,
        y_test,
        cv_folds=cv_folds,
    )

    # Save the best regression model.
    joblib.dump(
        regression_result["best_model"],
        REGRESSION_MODEL_PATH,
    )

    # Save regression metadata only.
    metadata = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "best_model": regression_result["best_name"],
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    # Return regression results only.
    return {
        "supervised_df": supervised_df,
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "regression": regression_result,
        "metadata": metadata,
    }

# ============================================================
# 16. PREPARE FEATURES FOR FUTURE PREDICTION
# ============================================================

def prepare_prediction_features(
    hourly_df,
    target_column,
    feature_columns,
):
    """
    Create a single prediction row using the latest
    available historical observations.
    """

    if len(hourly_df) < 168:
        raise ValueError(
            "At least 168 hourly observations are needed "
            "to create the lag and rolling features."
        )

    # build_supervised_dataset returns X, y, timestamps
    X, y, timestamps = build_supervised_dataset(
        hourly_df,
        target_column,
    )

    # Combine features and target into one DataFrame
    supervised_df = X.copy()
    supervised_df[target_column] = y
    supervised_df["timestamp"] = timestamps

    if supervised_df.empty:
        raise ValueError(
            "Unable to create prediction features. "
            "Please check your dataset."
        )

    # Get the latest valid feature row
    latest_row = supervised_df.iloc[[-1]].copy()

    # Check that all required training features exist
    missing_features = [
        col
        for col in feature_columns
        if col not in latest_row.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    # Keep only the features used during training
    X_future = latest_row[feature_columns].copy()

    return X_future
# ============================================================
# 17. DATASET SUMMARY
# ============================================================

def get_dataset_summary(
    df,
    target_column,
):

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    summary = {
        "Rows": int(len(df)),
        "Columns": int(len(df.columns)),
        "Missing Values": int(
            df.isna().sum().sum()
        ),
        "Duplicate Rows": int(
            df.duplicated().sum()
        ),
    }

    if target_column in numeric_df.columns:

        target = numeric_df[
            target_column
        ].dropna()

        if not target.empty:

            summary["Target Mean"] = float(
                target.mean()
            )

            summary["Target Median"] = float(
                target.median()
            )

            summary["Target Minimum"] = float(
                target.min()
            )

            summary["Target Maximum"] = float(
                target.max()
            )

            summary["Target Std"] = float(
                target.std()
            )

    return summary


# ============================================================
# 18. DISPLAY DATASET INFORMATION
# ============================================================

def show_dataset_information(
    df,
    target_column,
):

    st.subheader(
        "Dataset Overview"
    )

    summary = get_dataset_summary(
        df,
        target_column,
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Total Rows",
        f"{summary['Rows']:,}",
    )

    metric_columns[1].metric(
        "Total Columns",
        f"{summary['Columns']:,}",
    )

    metric_columns[2].metric(
        "Missing Values",
        f"{summary['Missing Values']:,}",
    )

    metric_columns[3].metric(
        "Duplicate Rows",
        f"{summary['Duplicate Rows']:,}",
    )

    if target_column in df.columns:

        st.markdown(
            f"**Selected target column:** `{target_column}`"
        )

    with st.expander(
        "View first 20 rows"
    ):

        st.dataframe(
            df.head(20),
            use_container_width=True,
        )

    with st.expander(
        "View column data types"
    ):

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Missing Percentage": [
                round(
                    float(df[col].isna().mean() * 100),
                    2,
                )
                for col in df.columns
            ],
        })

        st.dataframe(
            dtype_df,
            use_container_width=True,
        )


# ============================================================
# 19. MISSING VALUES ANALYSIS
# ============================================================

def show_missing_values_analysis(
    df,
):

    st.subheader(
        "Missing Values Analysis"
    )

    missing_counts = df.isna().sum()

    missing_counts = missing_counts[
        missing_counts > 0
    ].sort_values(
        ascending=False
    )

    if missing_counts.empty:

        st.success(
            "No missing values were detected."
        )

        return

    missing_df = pd.DataFrame({
        "Column": missing_counts.index,
        "Missing Count": missing_counts.values,
        "Missing Percentage": [
            round(
                float(
                    df[column].isna().mean() * 100
                ),
                2,
            )
            for column in missing_counts.index
        ],
    })

    st.dataframe(
        missing_df,
        use_container_width=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    missing_df.plot(
        kind="bar",
        x="Column",
        y="Missing Count",
        ax=ax,
        legend=False,
    )

    ax.set_title(
        "Missing Values by Column"
    )

    ax.set_xlabel(
        "Column"
    )

    ax.set_ylabel(
        "Missing Count"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# 20. TARGET DISTRIBUTION
# ============================================================

def show_target_distribution(
    df,
    target_column,
):

    st.subheader(
        "Electricity Consumption Distribution"
    )

    if target_column not in df.columns:

        st.warning(
            "The selected target column was not found."
        )

        return

    target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    ).dropna()

    if target.empty:

        st.warning(
            "The target column has no usable numeric values."
        )

        return

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.hist(
            target,
            bins=40,
        )

        ax.set_title(
            "Target Value Distribution"
        )

        ax.set_xlabel(
            target_column
        )

        ax.set_ylabel(
            "Frequency"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.boxplot(
            target,
            vert=True,
        )

        ax.set_title(
            "Target Value Box Plot"
        )

        ax.set_ylabel(
            target_column
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    st.write(
        "Descriptive statistics"
    )

    st.dataframe(
        target.describe().to_frame(
            name="Value"
        ),
        use_container_width=True,
    )


# ============================================================
# 21. TIME-SERIES VISUALIZATION
# ============================================================

def show_time_series_chart(
    hourly_df,
    target_column,
):

    st.subheader(
        "Electricity Consumption Over Time"
    )

    if "timestamp" not in hourly_df.columns:

        st.warning(
            "A timestamp column is required for this chart."
        )

        return

    plot_df = hourly_df[
        ["timestamp", target_column]
    ].copy()

    plot_df["timestamp"] = pd.to_datetime(
        plot_df["timestamp"],
        errors="coerce",
    )

    plot_df[target_column] = pd.to_numeric(
        plot_df[target_column],
        errors="coerce",
    )

    plot_df = plot_df.dropna()

    if plot_df.empty:

        st.warning(
            "There is no valid time-series data to display."
        )

        return

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        plot_df["timestamp"],
        plot_df[target_column],
        linewidth=0.8,
    )

    ax.set_title(
        "Electricity Consumption Time Series"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        target_column
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# 22. HOURLY AND DAILY PATTERN ANALYSIS
# ============================================================

def show_usage_patterns(hourly_df, target_column):

    st.subheader("Usage Patterns")

    if "timestamp" not in hourly_df.columns:
        st.warning("Timestamp information is unavailable.")
        return

    pattern_df = hourly_df[
        ["timestamp", target_column]
    ].copy()

    pattern_df["timestamp"] = pd.to_datetime(
        pattern_df["timestamp"],
        errors="coerce"
    )

    pattern_df[target_column] = pd.to_numeric(
        pattern_df[target_column],
        errors="coerce"
    )

    pattern_df = pattern_df.dropna()

    if pattern_df.empty:
        st.warning("No valid data is available.")
        return

    pattern_df["Hour"] = pattern_df["timestamp"].dt.hour

    pattern_df["Day of Week"] = (
        pattern_df["timestamp"].dt.day_name()
    )

    left, right = st.columns(2)

    with left:

        hourly_pattern = pattern_df.groupby(
            "Hour"
        )[target_column].mean()

        fig, ax = plt.subplots(figsize=(8, 4))

        hourly_pattern.plot(
            kind="line",
            marker="o",
            ax=ax
        )

        ax.set_title("Average Consumption by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel(f"Mean {target_column}")
        ax.set_xticks(range(0, 24, 2))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:

        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        weekday_pattern = pattern_df.groupby(
            "Day of Week"
        )[target_column].mean()

        weekday_pattern = weekday_pattern.reindex(
            weekday_order
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        weekday_pattern.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Average Consumption by Day")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel(f"Mean {target_column}")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
def show_correlation_analysis(
    hourly_df,
):

    st.subheader("Feature Correlation Analysis")

    numeric_df = hourly_df.select_dtypes(
        include=[np.number]
    )

    if numeric_df.shape[1] < 2:
        st.info(
            "At least two numeric columns are required."
        )
        return

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False,
        ax=ax,
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)





# ============================================================
# 24. MODEL EVALUATION DISPLAY
# ============================================================

def show_regression_evaluation(
    training_result,
):

    st.subheader("Regression Model Comparison")

    result = training_result["regression"]

    results_df = result["results"]

    st.dataframe(
        results_df,
        use_container_width=True,
    )

    st.success(
        f"Best Regression Model: {result['best_name']}"
    )

    y_true = training_result["y_test"]

    y_pred = result["predictions"]

    metrics = calculate_regression_metrics(
        y_true,
        y_pred,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{metrics['MAE']:.4f}",
    )

    col2.metric(
        "RMSE",
        f"{metrics['RMSE']:.4f}",
    )

    col3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}",
    )

    st.caption(
        "Lower MAE and RMSE are generally better. "
        "An R² closer to 1 generally indicates a better fit."
    )

    # Actual versus predicted values.
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        np.asarray(y_true),
        label="Actual",
        linewidth=1,
    )

    ax.plot(
        np.asarray(y_pred),
        label="Predicted",
        linewidth=1,
    )

    ax.set_title(
        "Actual vs Predicted Consumption"
    )

    ax.set_xlabel(
        "Test Observation"
    )

    ax.set_ylabel(
        "Target Value"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # Residual analysis.
    residuals = (
        np.asarray(y_true)
        - np.asarray(y_pred)
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.scatter(
        y_pred,
        residuals,
        alpha=0.5,
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_title(
        "Residual Plot"
    )

    ax.set_xlabel(
        "Predicted Values"
    )

    ax.set_ylabel(
        "Actual - Predicted"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# 25. CLASSIFICATION EVALUATION DISPLAY
# ============================================================

def show_classification_evaluation(
    training_result,
):
    # Classification evaluation is disabled.
    # This project uses regression only.
    return


# ============================================================
# 26. FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    training_result,
):

    st.subheader(
        "Feature Importance"
    )

    best_model = training_result[
        "regression"
    ]["best_model"]

    feature_names = training_result[
        "feature_columns"
    ]

    estimator = best_model.named_steps[
        "model"
    ]

    if not hasattr(
        estimator,
        "feature_importances_",
    ):

        st.info(
            "The selected regression model does not provide "
            "built-in tree-based feature importance."
        )

        return

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": estimator.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False,
    ).head(20)

    st.dataframe(
        importance_df,
        use_container_width=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
    )

    ax.set_title(
        "Top 20 Important Features"
    )

    ax.set_xlabel(
        "Importance"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# 27. PREDICTION INTERFACE
# ============================================================

def show_prediction_interface(
    hourly_df,
    target_column,
    training_result,
):

    st.subheader(
        "Next-Hour Consumption Prediction"
    )

    st.write(
        "Use the trained regression model to estimate the "
        "next hourly target value from the latest available data."
    )

    model = training_result[
        "regression"
    ]["best_model"]

    feature_columns = training_result[
        "feature_columns"
    ]

    if len(hourly_df) < 168:

        st.warning(
            "At least 168 hourly observations are needed "
            "for the lag features used by this project."
        )

        return

    try:

        X_future = prepare_prediction_features(
            hourly_df,
            target_column,
            feature_columns,
        )

    except Exception as exc:

        st.error(
            f"Could not prepare prediction features: {exc}"
        )

        return

    if st.button(
        "Predict Next-Hour Consumption",
        type="primary",
    ):

        try:

            prediction = float(
                model.predict(
                    X_future
                )[0]
            )

            st.metric(
                "Predicted Target Value",
                f"{prediction:.4f}",
            )

            st.caption(
                "This is a model estimate, not a guaranteed "
                "future measurement."
            )

            prediction_df = pd.DataFrame({
                "Prediction Type": [
                    "Next-hour estimate"
                ],
                "Target Column": [
                    target_column
                ],
                "Predicted Value": [
                    prediction
                ],
            })

            st.download_button(
                "Download Prediction CSV",
                data=prediction_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="electricity_prediction.csv",
                mime="text/csv",
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )


# ============================================================
# 28. DOWNLOAD MODEL RESULTS
# ============================================================

def show_download_options(training_result):

    st.subheader("Download Project Results")

    regression_csv = training_result[
        "regression"
    ]["results"].to_csv(
        index=False
    ).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Regression Results",
            data=regression_csv,
            file_name="regression_results.csv",
            mime="text/csv",
        )

    with col2:
        st.empty()


# ============================================================
# 29. STREAMLIT APPLICATION
# ============================================================

def main():
    st.sidebar.title(
        "Project Navigation"
    )
    st.sidebar.write(
        "Electricity Consumption Prediction"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload electricity dataset",
        type=[
            "csv",
            "txt",
        ],
    )
    st.sidebar.caption(
        "CSV and semicolon-separated TXT datasets are supported."
    )

    if uploaded_file is None:

        st.info(
            "Upload your electricity consumption CSV or TXT file "
            "from the sidebar to begin."
        )
        st.markdown(
    """
    ### Project Features

    - Data preprocessing
    - Missing-value analysis
    - Exploratory data analysis
    - Time-series feature engineering
    - Regression model comparison
    - MAE, RMSE and R² evaluation metrics
    - Actual vs predicted chart
    - Residual plot
    - Feature importance analysis
    - Top 20 important features
    - Next-hour electricity consumption prediction
    - CSV downloads
    """
)

        return

    # Read the uploaded file.
    try:

        raw_bytes = uploaded_file.getvalue()

        raw_df = read_uploaded_data(
            raw_bytes,
            uploaded_file.name,
        )

    except Exception as exc:

        st.error(
            f"Could not read the uploaded file: {exc}"
        )

        return

    if raw_df.empty:

        st.error(
            "The uploaded file contains no data."
        )

        return

    st.sidebar.subheader(
        "Target Selection"
    )

    suggested_target = detect_target_column(
        raw_df
    )

    target_options = [
        col
        for col in raw_df.columns
        if col not in [
            "Date",
            "Time",
            "datetime",
            "timestamp",
        ]
    ]

    if not target_options:

        st.error(
            "No target columns were found."
        )

        return

    if (
        suggested_target in target_options
    ):

        default_index = target_options.index(
            suggested_target
        )

    else:

        default_index = 0

    target_column = st.sidebar.selectbox(
        "Choose the target column",
        target_options,
        index=default_index,
    )

    # Preprocess and aggregate the data.
    try:

        hourly_df, cleaning_report = validate_and_clean(
            raw_df,
            target_column,
        )

    except Exception as exc:

        st.error(
            f"Data preprocessing failed: {exc}"
        )

        return

    if hourly_df.empty:

        st.error(
            "No usable observations remain after preprocessing."
        )

        return

    st.title(
        "Electricity Consumption Prediction & Energy Usage Analysis"
    )

    st.caption(
        "Machine learning dashboard for historical usage "
        "analysis and next-hour prediction."
    )

    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Data Analysis",
            "Model Training",
            "Evaluation",
            "Prediction",
        ],
    )

    # --------------------------------------------------------
    # OVERVIEW PAGE
    # --------------------------------------------------------

    if page == "Overview":

        show_dataset_information(
            raw_df,
            target_column,
        )

        st.subheader(
            "Preprocessing Summary"
        )

        st.write(
            cleaning_report
        )

        st.subheader(
            "Processed Hourly Data"
        )

        st.dataframe(
            hourly_df.head(20),
            use_container_width=True,
        )

        st.download_button(
            "Download Processed Data",
            data=hourly_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="processed_electricity_data.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------
    # DATA ANALYSIS PAGE
    # --------------------------------------------------------

    elif page == "Data Analysis":

        show_missing_values_analysis(
            raw_df
        )

        show_target_distribution(
            hourly_df,
            target_column,
        )

        show_time_series_chart(
            hourly_df,
            target_column,
        )

        show_usage_patterns(
            hourly_df,
            target_column,
        )

        show_correlation_analysis(
            hourly_df
        )

    # --------------------------------------------------------
    # MODEL TRAINING PAGE
    # --------------------------------------------------------

    elif page == "Model Training":

        st.subheader(
            "Train Machine Learning Models"
        )

        cv_folds = st.sidebar.selectbox(
            "Time-series CV folds",
            [2, 3, 4, 5],
            index=1,
        )

        st.warning(
            "Training may take time on large datasets."
        )

        if st.button(
            "Train and Compare Models",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Training models. Please wait..."
                ):

                    result = train_all_models(
                        hourly_df,
                        target_column,
                        cv_folds=cv_folds,
                    )

                st.session_state[
                    "training_result"
                ] = result

                st.session_state[
                    "training_target"
                ] = target_column

                st.success(
                    "Training completed successfully!"
                )

            except Exception as exc:

                st.error(
                    f"Model training failed: {exc}"
                )

        if (
            "training_result" in st.session_state
            and st.session_state.get(
                "training_target"
            ) == target_column
        ):

            result = st.session_state[
                "training_result"
            ]

            st.write(
                "Regression model results"
            )

            st.dataframe(
                result["regression"]["results"],
                use_container_width=True,
            )

            
            

    # --------------------------------------------------------
    # EVALUATION PAGE
    # --------------------------------------------------------

    elif page == "Evaluation":

        if (
            "training_result" not in st.session_state
            or st.session_state.get(
                "training_target"
            ) != target_column
        ):

            st.info(
                "Go to Model Training and train the models first."
            )

        else:

            result = st.session_state[
                "training_result"
            ]

            show_regression_evaluation(
                result
            )

            


            st.divider()

            show_feature_importance(
                result
            )

            st.divider()

            show_download_options(
                result
            )

    # --------------------------------------------------------
    # PREDICTION PAGE
    # --------------------------------------------------------

    elif page == "Prediction":

        if (
            "training_result" not in st.session_state
            or st.session_state.get(
                "training_target"
            ) != target_column
        ):

            st.info(
                "Train the models first from the Model Training page."
            )

        else:

            show_prediction_interface(
                hourly_df,
                target_column,
                st.session_state[
                    "training_result"
                ],
            )


# ============================================================
# 30. RUN THE APPLICATION
# ============================================================

if __name__ == "__main__":

    main()


