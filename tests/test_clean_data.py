import unittest
import pandas as pd
from scripts import clean_data

class TestCleanData(unittest.TestCase):
    def setUp(self):
        self.required_columns = clean_data.REQUIRED_COLUMNS
        self.training_columns = clean_data.TRAINING_COLUMNS

    def test_load_data_file_not_found(self):
        with self.assertRaises(clean_data.DataLoadError):
            clean_data.load_data('non_existent_file.csv')

    def test_load_data_empty_file(self):
        # Create an empty CSV for testing
        path = 'empty_test.csv'
        pd.DataFrame().to_csv(path, index=False)
        try:
            with self.assertRaises(clean_data.DataLoadError):
                clean_data.load_data(path)
        finally:
            import os
            os.remove(path)

    def test_check_missing_values(self):
        df = pd.DataFrame({
            'date': ['2021-01-01', None],
            'gmv': [100, None],
            'users': [10, 20],
            'marketing_cost': [5, 6]
        })
        missing = clean_data.check_missing_values(df, 'Test')
        self.assertIn('date', missing)
        self.assertIn('gmv', missing)

    def test_check_duplicates(self):
        df = pd.DataFrame({
            'date': ['2021-01-01', '2021-01-01'],
            'gmv': [100, 100],
            'users': [10, 10],
            'marketing_cost': [5, 5]
        })
        duplicates = clean_data.check_duplicates(df, 'Test')
        self.assertEqual(duplicates, 1)

    def test_check_outliers(self):
        df = pd.DataFrame({'gmv': [-1, 50, 200_000_000]})
        below, above = clean_data.check_outliers(df, 'gmv', clean_data.MIN_GMV, clean_data.MAX_GMV)
        self.assertEqual(below, 1)
        self.assertEqual(above, 1)

    def test_clean_dataframe_missing_and_duplicates(self):
        df = pd.DataFrame({
            'date': ['2021-01-01', '2021-01-01', None],
            'gmv': ['100', '100', None],
            'users': ['10', '10', None],
            'marketing_cost': ['5', '5', None],
            'fe_pods': ['2', '2', None],
            'be_pods': ['1', '1', None]
        })
        # Should drop missing and duplicate rows
        cleaned = clean_data.clean_dataframe(df, 'Test', has_pods=True)
        self.assertEqual(len(cleaned), 1)
        self.assertListEqual(list(cleaned.columns), self.training_columns)

    def test_validate_data_quality_negative(self):
        df = pd.DataFrame({
            'date': ['2021-01-01'],
            'gmv': [-100],
            'users': [10],
            'marketing_cost': [5],
            'fe_pods': [2],
            'be_pods': [1]
        })
        with self.assertRaises(clean_data.DataQualityError):
            clean_data.validate_data_quality(df, 'Test', has_pods=True)

if __name__ == "__main__":
    unittest.main()
