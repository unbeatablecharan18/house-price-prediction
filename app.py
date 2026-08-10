from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_utils import DATASET_PATH, MODEL_PATH, MODELS_DIR, ensure_dataset, load_metadata
from src.training import train_and_compare_models


@st.cache_data
def load_dataset():
    return ensure_dataset()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparison():
    path = MODELS_DIR / "model_comparison.csv"
    if not path.exists():
        train_and_compare_models()
    return pd.read_csv(path)


def ensure_project_ready():
    ensure_dataset()
    if not MODEL_PATH.exists() or not (MODELS_DIR / "model_comparison.csv").exists():
        train_and_compare_models()


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def main():
    ensure_project_ready()

    df = load_dataset()
    model = load_model()
    comparison_df = load_comparison()
    metadata = load_metadata()

    st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
    st.title("🏠 House Price Prediction Dashboard")
    st.caption("Predict real-estate prices from the California housing dataset.")

    overview_tab, predict_tab, compare_tab = st.tabs(["Overview", "Predict Price", "Model Comparison"])

    with overview_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Rows", len(df))
        with col2:
            st.metric("Average Price", format_currency(df["price"].mean()))
        with col3:
            st.metric("Best Model", metadata.get("best_model_name", "N/A"))

        st.subheader("Price Distribution")
        price_hist = px.histogram(df, x="price", nbins=30, title="House Price Distribution")
        st.plotly_chart(price_hist, width="stretch")

        col4, col5 = st.columns(2)
        with col4:
            st.subheader("Median Income vs Price")
            scatter = px.scatter(
                df,
                x="median_income",
                y="price",
                title="Income vs House Price",
                opacity=0.7,
            )
            st.plotly_chart(scatter, width="stretch")
        with col5:
            st.subheader("House Age vs Price")
            scatter_age = px.scatter(df, x="house_age", y="price", title="House Age vs Price", opacity=0.7)
            st.plotly_chart(scatter_age, width="stretch")

    with predict_tab:
        st.subheader("Enter Property Details")

        with st.form("prediction_form"):
            col_left, col_right = st.columns(2)

            with col_left:
                median_income = st.number_input("Median Income", min_value=0.0, max_value=15.0, step=0.1, value=3.5)
                house_age = st.slider("House Age (years)", min_value=1, max_value=52, value=20)
                average_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, step=0.5, value=5.0)
                average_bedrooms = st.number_input("Average Bedrooms", min_value=0.5, max_value=8.0, step=0.1, value=1.1)
                population = st.number_input("Population", min_value=1, max_value=50000, step=50, value=1500)

            with col_right:
                average_occupancy = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, step=0.1, value=2.5)
                latitude = st.number_input("Latitude", min_value=30.0, max_value=42.0, step=0.01, value=37.8)
                longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, step=0.01, value=-122.2)

            submitted = st.form_submit_button("Predict Price")

        if submitted:
            user_input = pd.DataFrame(
                [{
                    "median_income": median_income,
                    "house_age": house_age,
                    "average_rooms": average_rooms,
                    "average_bedrooms": average_bedrooms,
                    "population": population,
                    "average_occupancy": average_occupancy,
                    "latitude": latitude,
                    "longitude": longitude,
                }]
            )

            predicted_price = float(model.predict(user_input)[0])
            st.success(f"Predicted House Price: {format_currency(predicted_price)}")

            st.json({
                "median_income": median_income,
                "house_age": house_age,
                "average_rooms": average_rooms,
                "average_bedrooms": average_bedrooms,
                "population": population,
                "average_occupancy": average_occupancy,
                "latitude": latitude,
                "longitude": longitude,
                "predicted_price": predicted_price,
            })

    with compare_tab:
        st.subheader("Model Comparison")
        st.dataframe(comparison_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}), width="stretch")

        chart = px.bar(
            comparison_df,
            x="Model",
            y="RMSE",
            color="Model",
            title="RMSE Comparison Across Models",
        )
        st.plotly_chart(chart, width="stretch")


if __name__ == "__main__":
    main()
