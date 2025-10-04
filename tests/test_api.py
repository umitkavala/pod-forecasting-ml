import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class DummyPredictor:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, date, gmv, users, marketing_cost):
        return {
            "predictions": {"frontend_pods": 5, "backend_pods": 2},
            "confidence_intervals": {"frontend_pods": [4, 6], "backend_pods": [2, 2]}
        }

    def get_model_info(self):
        return {"model_type": "Dummy", "features": [], "n_features": 0}


class TestAPI(unittest.TestCase):
    def test_health_success(self):
        # Patch PodPredictor so startup doesn't load real model files
        with patch("api.main.PodPredictor", new=DummyPredictor), \
             patch("api.main.is_public_endpoint", new=lambda path: True):
            from api import main as main_mod
            from api.main import app
            # ensure the module-level predictor is set (lifespan may not run in TestClient imports)
            main_mod.predictor = DummyPredictor()
            try:
                main_mod.model_loaded.set(1)
            except Exception:
                pass
            client = TestClient(app)

            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("model_loaded", data)
            self.assertTrue(data["model_loaded"])
            self.assertIn("model_info", data)

    def test_predict_single_success(self):
        with patch("api.main.PodPredictor", new=DummyPredictor), \
             patch("api.main.is_public_endpoint", new=lambda path: True):
            from api import main as main_mod
            from api.main import app
            main_mod.predictor = DummyPredictor()
            try:
                main_mod.model_loaded.set(1)
            except Exception:
                pass
            client = TestClient(app)

            payload = {
                "date": "2024-07-15",
                "gmv": 1000000,
                "users": 10000,
                "marketing_cost": 50000
            }

            resp = client.post("/predict", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("predictions", data)
            self.assertEqual(data["predictions"]["frontend_pods"], 5)
            self.assertEqual(data["predictions"]["backend_pods"], 2)

    def test_predict_validation_error(self):
        with patch("api.main.PodPredictor", new=DummyPredictor), \
             patch("api.main.is_public_endpoint", new=lambda path: True):
            from api import main as main_mod
            from api.main import app
            main_mod.predictor = DummyPredictor()
            try:
                main_mod.model_loaded.set(1)
            except Exception:
                pass
            client = TestClient(app)

            payload = {
                "date": "2024-07-15",
                "gmv": -1,
                "users": 10000,
                "marketing_cost": 50000
            }

            resp = client.post("/predict", json=payload)
            # ValidationError maps to 422 Unprocessable Entity
            self.assertEqual(resp.status_code, 422)

    def test_metrics_endpoint(self):
        with patch("api.main.PodPredictor", new=DummyPredictor), \
             patch("api.main.is_public_endpoint", new=lambda path: True):
            from api import main as main_mod
            from api.main import app
            main_mod.predictor = DummyPredictor()
            try:
                main_mod.model_loaded.set(1)
            except Exception:
                pass
            client = TestClient(app)

            resp = client.get("/metrics")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/plain", resp.headers.get("content-type", ""))


if __name__ == "__main__":
    unittest.main()
