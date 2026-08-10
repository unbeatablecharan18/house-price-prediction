import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_utils import normalize_real_dataset
from src.training import train_and_compare_models


class TestRealHousePipeline(unittest.TestCase):
    def test_normalize_real_dataset_maps_expected_columns(self):
        df = pd.DataFrame(
            {
                "MedInc": [8.32, 4.10, 3.40],
                "HouseAge": [41.0, 21.0, 52.0],
                "AveRooms": [6.98, 5.50, 4.70],
                "AveBedrms": [1.02, 1.00, 1.12],
                "Population": [322, 560, 845],
                "AveOccup": [2.55, 2.12, 2.20],
                "Latitude": [37.88, 37.86, 37.85],
                "Longitude": [-122.23, -122.22, -122.25],
                "MedHouseVal": [4.52, 3.58, 3.42],
            }
        )

        normalized = normalize_real_dataset(df)

        expected_columns = [
            "median_income",
            "house_age",
            "average_rooms",
            "average_bedrooms",
            "population",
            "average_occupancy",
            "latitude",
            "longitude",
            "price",
        ]

        self.assertListEqual(list(normalized.columns), expected_columns)
        self.assertTrue((normalized["price"] > 0).all())
        self.assertTrue((normalized["median_income"] > 0).all())

    def test_training_pipeline_runs_on_realistic_dataset(self):
        df = pd.DataFrame(
            {
                "median_income": [8.32, 4.10, 3.40, 5.12, 6.67, 2.90, 4.50, 3.00],
                "house_age": [41.0, 21.0, 52.0, 30.0, 15.0, 45.0, 18.0, 27.0],
                "average_rooms": [6.98, 5.50, 4.70, 8.20, 7.10, 4.10, 6.30, 5.80],
                "average_bedrooms": [1.02, 1.00, 1.12, 1.09, 1.40, 1.20, 1.15, 1.05],
                "population": [322, 560, 845, 700, 1150, 600, 980, 420],
                "average_occupancy": [2.55, 2.12, 2.20, 2.60, 3.10, 2.25, 2.35, 2.00],
                "latitude": [37.88, 37.86, 37.85, 37.84, 37.82, 37.81, 37.80, 37.79],
                "longitude": [-122.23, -122.22, -122.25, -122.24, -122.26, -122.27, -122.28, -122.29],
                "price": [452600, 358000, 342000, 470000, 520000, 375000, 485000, 335000],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "house_data.csv"
            df.to_csv(dataset_path, index=False)

            comparison_df, best_model, _ = train_and_compare_models(
                dataset_path=dataset_path,
                model_path=Path(tmp_dir) / "best_house_model.pkl",
                comparison_path=Path(tmp_dir) / "model_comparison.csv",
                metadata_path=Path(tmp_dir) / "model_metadata.json",
            )

            self.assertIn("Model", comparison_df.columns)
            self.assertIn("MAE", comparison_df.columns)
            self.assertIn("RMSE", comparison_df.columns)
            self.assertIn("R2", comparison_df.columns)
            self.assertTrue(best_model)
            self.assertTrue((Path(tmp_dir) / "best_house_model.pkl").exists())
            self.assertTrue((Path(tmp_dir) / "model_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
