from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = DATA_DIR / "house_data.csv"
MODEL_PATH = MODELS_DIR / "best_house_model.pkl"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS = [
    "median_income",
    "house_age",
    "average_rooms",
    "average_bedrooms",
    "population",
    "average_occupancy",
    "latitude",
    "longitude",
]
TARGET_COLUMN = "price"


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)


def build_real_dataset(path: Path) -> pd.DataFrame:
    """Download and cache the California Housing dataset used by this project."""
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()

    df = df.rename(
        columns={
            "MedInc": "median_income",
            "HouseAge": "house_age",
            "AveRooms": "average_rooms",
            "AveBedrms": "average_bedrooms",
            "Population": "population",
            "AveOccup": "average_occupancy",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "MedHouseVal": "price",
        }
    )

    df["price"] = df["price"] * 100_000
    df = df.dropna().reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def ensure_dataset() -> pd.DataFrame:
    ensure_directories()
    if not DATASET_PATH.exists():
        df = build_real_dataset(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH)
        if set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns):
            return df
        if set({"MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"}).issubset(df.columns):
            df = normalize_real_dataset(df)
        else:
            df = build_real_dataset(DATASET_PATH)
        df.to_csv(DATASET_PATH, index=False)
    return df


def normalize_real_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize real data columns to the app's expected schema."""
    renamed = df.copy()

    rename_map = {
        "MedInc": "median_income",
        "HouseAge": "house_age",
        "AveRooms": "average_rooms",
        "AveBedrms": "average_bedrooms",
        "Population": "population",
        "AveOccup": "average_occupancy",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "MedHouseVal": "price",
    }
    renamed = renamed.rename(columns=rename_map)

    if "price" in renamed.columns:
        renamed["price"] = renamed["price"] * 100_000

    return renamed[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().reset_index(drop=True)


def get_numeric_and_categorical_columns(df: pd.DataFrame):
    target_col = TARGET_COLUMN
    feature_columns = [col for col in df.columns if col != target_col]
    numeric_columns = []
    categorical_columns = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    return numeric_columns, categorical_columns


def save_metadata(best_model_name: str, metrics: dict, numeric_columns: list[str], categorical_columns: list[str]) -> None:
    metadata = {
        "best_model_name": best_model_name,
        "metrics": metrics,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
