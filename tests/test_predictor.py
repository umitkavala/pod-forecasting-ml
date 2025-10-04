import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime


class FakeScaler:
    def transform(self, X):
        # return as-is for testing
        return X


class FakeModel:
    def predict(self, X):
        # Return deterministic predictions for FE and BE
        # Shape: (n_samples, 2)
        n = X.shape[0]
        import numpy as np
        # For testing, use gmv_millions to decide values
        preds = []
        for row in X:
            gmv_millions = row[8]
            fe = max(1, int(round(gmv_millions))) + 1
            be = max(1, int(round(row[9] / 10)))
            preds.append([fe, be])
        return np.array(preds)


class TestPodPredictor(unittest.TestCase):
    @patch("joblib.load")
    def setUp(self, mock_joblib_load):
        # joblib.load called twice: model and scaler
        def side_effect(path):
            if path.endswith("model.pkl"):
                return FakeModel()
            return FakeScaler()

        mock_joblib_load.side_effect = side_effect
        from api.predictor import PodPredictor
        self.predictor = PodPredictor(model_path="models/model.pkl", scaler_path="models/scaler.pkl")

    def test_engineer_features_shape_and_values(self):
        date = datetime(2024, 7, 15)
        features = self.predictor.engineer_features(date, 1_000_000, 10000, 50000)
        self.assertEqual(features.shape, (1, 10))
        # Check derived features
        self.assertAlmostEqual(features[0][3], 1_000_000 / 10000)  # gmv_per_user
        self.assertAlmostEqual(features[0][8], 1_000_000 / 1_000_000)  # gmv_millions

    def test_predict_returns_expected_structure(self):
        date = datetime(2024, 7, 15)
        result = self.predictor.predict(date, 1_500_000, 15000, 60000)
        self.assertIn("predictions", result)
        self.assertIn("confidence_intervals", result)
        self.assertIsInstance(result["predictions"]["frontend_pods"], int)
        self.assertIsInstance(result["predictions"]["backend_pods"], int)

    def test_get_model_info(self):
        info = self.predictor.get_model_info()
        self.assertIn("model_type", info)
        self.assertIn("features", info)


if __name__ == "__main__":
    unittest.main()
