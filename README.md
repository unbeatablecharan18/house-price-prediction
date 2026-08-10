# House Price Prediction

This project predicts median house values using a real public dataset, trains several regression models, compares their performance, and exposes the best model through a Streamlit dashboard.

## Dataset source

The project uses the California Housing dataset from scikit-learn, which is a well-known public dataset for housing price prediction.

- Source: scikit-learn `fetch_california_housing`
- Description: residential housing data from California, with features such as median income, house age, rooms, bedrooms, population, occupancy, latitude, longitude, and median house value.
- The dataset is downloaded automatically on first run and saved to `data/house_data.csv`.

## Dataset description

The real dataset is normalized into the app schema below before training:

- `median_income`
- `house_age`
- `average_rooms`
- `average_bedrooms`
- `population`
- `average_occupancy`
- `latitude`
- `longitude`
- `price`

This keeps the training pipeline consistent with the prediction pipeline and the dashboard inputs.

## Preprocessing

The data pipeline includes:

- loading the real dataset from scikit-learn
- validating the required columns
- normalizing column names to a consistent schema
- filling missing values with median imputation for numeric features
- scaling numeric features with `StandardScaler`
- training all models using the same preprocessor and feature ordering

## Models used

The project trains and compares the following regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

## Evaluation metrics

The model comparison is saved to `models/model_comparison.csv` and includes:

- MAE
- RMSE
- R²

The best model is selected using the lowest RMSE and saved to `models/best_house_model.pkl`.

## Project structure

```text
house prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── house_data.csv
├── models/
│   ├── best_house_model.pkl
│   ├── model_comparison.csv
│   └── model_metadata.json
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   └── training.py
├── tests/
│   └── test_pipeline.py
└── .gitignore
```

## How to run

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Train models and generate the dataset artifacts:

   ```bash
   python -m src.training
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Open the local Streamlit URL shown in the terminal, usually:

   ```text
   http://localhost:8501
   ```

## Notes

- The app automatically creates the dataset and trains the model if the model files are missing.
- The dashboard prediction form uses the same feature names as the model was trained on.
- Model performance is tracked and stored in `models/model_comparison.csv`.
