import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Ensure scripts package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from scripts import fetch_from_sheets


def make_service_with_values(values):
    """Return a mocked Google Sheets service where execute() returns the provided values."""
    mock_execute = MagicMock()
    mock_execute.execute.return_value = {'values': values}

    mock_get = MagicMock()
    mock_get.execute = mock_execute.execute

    mock_values = MagicMock()
    mock_values.get.return_value = mock_get

    mock_sheets = MagicMock()
    mock_sheets.values.return_value = mock_values

    service = MagicMock()
    service.spreadsheets.return_value = mock_sheets
    return service


class TestFetchFromSheets(unittest.TestCase):
    def setUp(self):
        self.expected_historical_columns = fetch_from_sheets.EXPECTED_HISTORICAL_COLUMNS
        self.expected_budget_columns = fetch_from_sheets.EXPECTED_BUDGET_COLUMNS

    def test_validate_dataframe_empty(self):
        df = pd.DataFrame()
        with self.assertRaises(fetch_from_sheets.EmptyDataError):
            fetch_from_sheets.validate_dataframe(df, self.expected_historical_columns, 'Historical')

    def test_validate_dataframe_missing_columns(self):
        df = pd.DataFrame({'date': ["2021-01-01"], 'gmv': [100]})
        with self.assertRaises(fetch_from_sheets.DataValidationError):
            fetch_from_sheets.validate_dataframe(df, self.expected_historical_columns, 'Historical')

    def test_validate_dataframe_null_values(self):
        df = pd.DataFrame({
            'date': ["2021-01-01"],
            'gmv': [None],
            'users': [10],
            'marketing_cost': [5],
            'fe_pods': [2],
            'be_pods': [1]
        })
        # Should not raise, just log warning
        fetch_from_sheets.validate_dataframe(df, self.expected_historical_columns, 'Historical')

    @patch('scripts.fetch_from_sheets.service_account.Credentials.from_service_account_file')
    @patch('scripts.fetch_from_sheets.build')
    def test_authenticate_service_account_success(self, mock_build, mock_creds):
        mock_creds.return_value = MagicMock(service_account_email='test@service.com')
        mock_build.return_value = MagicMock()
        with patch('os.path.exists', return_value=True):
            service = fetch_from_sheets.authenticate_service_account()
            self.assertIsNotNone(service)

    def test_convert_to_dataframe(self):
        values = [
            ['date', 'gmv', 'users', 'marketing_cost', 'fe_pods', 'be_pods'],
            ['2021-01-01', '100', '10', '5', '2', '1']
        ]
        df = fetch_from_sheets.convert_to_dataframe(values, 'Historical')
        self.assertListEqual(list(df.columns), self.expected_historical_columns)
        self.assertEqual(len(df), 1)

    def test_fetch_sheet_data_success(self):
        values = [
            ['date', 'gmv', 'users', 'marketing_cost', 'fe_pods', 'be_pods'],
            ['2021-01-01', '100', '10', '5', '2', '1']
        ]
        service = make_service_with_values(values)
        res = fetch_from_sheets.fetch_sheet_data(service, 'Historical')
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)

    def test_fetch_sheet_data_empty_raises(self):
        service = make_service_with_values([])
        with self.assertRaises(fetch_from_sheets.EmptyDataError):
            fetch_from_sheets.fetch_sheet_data(service, 'Historical')

    def test_fetch_sheet_data_404_raises_sheet_not_found(self):
        # Create an HttpError with resp.status == 404
        from googleapiclient.errors import HttpError

        resp = type('R', (), {'status': 404, 'reason': 'Not Found'})()
        http_err = HttpError(resp, b'not found')

        mock_get = MagicMock()
        mock_get.execute.side_effect = http_err

        mock_values = MagicMock()
        mock_values.get.return_value = mock_get

        mock_sheets = MagicMock()
        mock_sheets.values.return_value = mock_values

        service = MagicMock()
        service.spreadsheets.return_value = mock_sheets

        with self.assertRaises(fetch_from_sheets.SheetNotFoundError):
            fetch_from_sheets.fetch_sheet_data(service, 'NonExistent')

    def test_fetch_sheet_data_403_raises_permission(self):
        from googleapiclient.errors import HttpError

        resp = type('R', (), {'status': 403, 'reason': 'Forbidden'})()
        http_err = HttpError(resp, b'forbidden')

        mock_get = MagicMock()
        mock_get.execute.side_effect = http_err

        mock_values = MagicMock()
        mock_values.get.return_value = mock_get

        mock_sheets = MagicMock()
        mock_sheets.values.return_value = mock_values

        service = MagicMock()
        service.spreadsheets.return_value = mock_sheets

        with self.assertRaises(fetch_from_sheets.PermissionError):
            fetch_from_sheets.fetch_sheet_data(service, 'Historical')

    def test_retry_on_error_eventual_success(self):
        # Simulate transient HttpError then success; ensure decorator retries
        from googleapiclient.errors import HttpError

        resp_500 = type('R', (), {'status': 500, 'reason': 'Server Error'})()
        http_err = HttpError(resp_500, b'error')

        # Create a mock function that fails twice then succeeds
        call_count = {'n': 0}

        def flaky_execute():
            call_count['n'] += 1
            if call_count['n'] < 3:
                raise http_err
            return {'values': [['h'], ['v']]}

        mock_get = MagicMock()
        mock_get.execute.side_effect = flaky_execute

        mock_values = MagicMock()
        mock_values.get.return_value = mock_get

        mock_sheets = MagicMock()
        mock_sheets.values.return_value = mock_values

        service = MagicMock()
        service.spreadsheets.return_value = mock_sheets

        # Patch sleep to avoid delay
        with patch('time.sleep', return_value=None):
            res = fetch_from_sheets.fetch_sheet_data(service, 'Historical')
            self.assertIsInstance(res, list)


if __name__ == "__main__":
    unittest.main()
