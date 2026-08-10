from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_utils import (
    COMPARISON_PATH,
    DATASET_PATH,
    METADATA_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    ensure_dataset,
    get_numeric_and_categorical_columns,
    save_metadata,
)


def build_preprocessor(numeric_columns, categorical_columns):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", numeric_transformer, numeric_columns)]
    if categorical_columns:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_columns))

    return ColumnTransformer(transformers=transformers)


def train_and_compare_models(
    dataset_path: str | Path = DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    comparison_path: str | Path = COMPARISON_PATH,
    metadata_path: str | Path = METADATA_PATH,
):
    df = pd.read_csv(dataset_path) if isinstance(dataset_path, (str, Path)) else ensure_dataset()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"The dataset must contain a target column named '{TARGET_COLUMN}'.")

    numeric_columns, categorical_columns = get_numeric_and_categorical_columns(df)
    if not numeric_columns:
        raise ValueError("The dataset must include at least one numeric feature column.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, random_state=42),
    }

    results = []
    best_score = None
    best_model_name = ""
    best_pipeline = None

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        if best_score is None or rmse < best_score:
            best_score = rmse
            best_model_name = model_name
            best_pipeline = pipeline

    comparison_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    comparison_df.to_csv(comparison_path, index=False)

    if best_pipeline is not None:
        joblib.dump(best_pipeline, model_path)

    metrics = comparison_df.loc[comparison_df["Model"] == best_model_name].iloc[0].to_dict()
    save_metadata(best_model_name, metrics, numeric_columns, categorical_columns)

    print("Model comparison saved to:", comparison_path)
    print("Best model:", best_model_name)
    print("Best metrics:", metrics)

    return comparison_df, best_model_name, best_pipeline


if __name__ == "__main__":
    train_and_compare_models()
