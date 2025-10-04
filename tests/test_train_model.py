import unittest
import pandas as pd
import numpy as np
from scripts import train_model

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        # Minimal valid training data
        self.df = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=12),
            'gmv': np.arange(100, 112),
            'users': np.arange(10, 22),
            'marketing_cost': np.arange(5, 17),
            'fe_pods': np.arange(1, 13),
            'be_pods': np.arange(2, 14)
        })

    def test_check_data_sufficiency(self):
        # Should not raise
        train_model.check_data_sufficiency(self.df)
        # Should raise for insufficient samples
        df_small = self.df.head(5)
        with self.assertRaises(train_model.InsufficientDataError):
            train_model.check_data_sufficiency(df_small)

    def test_engineer_features(self):
        df_feat = train_model.engineer_features(self.df)
        self.assertIn('gmv_per_user', df_feat.columns)
        self.assertIn('marketing_eff', df_feat.columns)
        self.assertIn('day_of_week', df_feat.columns)
        self.assertIn('is_weekend', df_feat.columns)
        self.assertIn('gmv_millions', df_feat.columns)
        self.assertIn('users_thousands', df_feat.columns)

    def test_train_model_and_evaluate(self):
        df_feat = train_model.engineer_features(self.df)
        feature_names = [
            'gmv', 'users', 'marketing_cost',
            'gmv_per_user', 'marketing_eff',
            'day_of_week', 'is_weekend', 'month',
            'gmv_millions', 'users_thousands'
        ]
        X = df_feat[feature_names]
        y = df_feat[['fe_pods', 'be_pods']]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = train_model.train_model(X_scaled, y)
        metrics = train_model.evaluate_model(model, X_scaled, y, 'Test')
        self.assertIn('mae_frontend', metrics)
        self.assertIn('r2_backend', metrics)

    def test_validate_model_performance(self):
        # Should not raise for good metrics
        metrics = {
            'mae_frontend': 1,
            'mae_backend': 1,
            'r2_frontend': 0.9,
            'r2_backend': 0.9
        }
        train_model.validate_model_performance(metrics)
        # Should raise for bad metrics
        bad_metrics = {
            'mae_frontend': 20,
            'mae_backend': 20,
            'r2_frontend': 0.1,
            'r2_backend': 0.1
        }
        with self.assertRaises(train_model.ModelValidationError):
            train_model.validate_model_performance(bad_metrics)

if __name__ == "__main__":
    unittest.main()
