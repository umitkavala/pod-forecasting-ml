"""
Pod Forecasting API - Python Integration Demo

Demonstrates how to integrate with the Pod Forecasting API using Python.
"""

import requests
import os
import sys
from datetime import datetime, timedelta

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:5000")
"""
Pod Forecasting API - Python Integration Demo

Demonstrates how to integrate with the Pod Forecasting API using Python.
"""

import requests
import os
import sys
from datetime import datetime, timedelta

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:5000")
API_KEY = os.getenv("API_KEY_NODE_SERVICE")

# Check API key
if not API_KEY:
    print("Error: API_KEY_NODE_SERVICE environment variable not set")
    print("\nSet it with:")
    print("  export API_KEY_NODE_SERVICE='your-api-key-here'")
    sys.exit(1)

# Headers
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def test_health_check():
    """Test health check endpoint (no auth required)."""
    print("\n" + "="*60)
    print("Testing health check...")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/health")

        if response.status_code == 200:
            data = response.json()
            print("Health check passed")
            print(f"  Status: {data['status']}")
            print(f"  Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("Connection failed")
        print("  Make sure API is running: uvicorn api.main:app --port 5000")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n" + "="*60)
    print("Testing single prediction...")
    print("="*60)

    # Prediction data
    data = {
        "date": "2024-07-15",
        "gmv": 9500000,
        "users": 85000,
        "marketing_cost": 175000
    }

    print(f"\nRequest:")
    print(f"  Date: {data['date']}")
    print(f"  GMV: ${data['gmv']:,}")
    print(f"  Users: {data['users']:,}")
    print(f"  Marketing Cost: ${data['marketing_cost']:,}")

    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions']
            confidence = result['confidence_intervals']

            print(f"\nPrediction successful!")
            print(f"  Frontend pods: {predictions['frontend_pods']}")
            print(f"  Backend pods: {predictions['backend_pods']}")
            print(f"  Confidence:")
            print(f"    Frontend: {confidence['frontend_pods']}")
            print(f"    Backend: {confidence['backend_pods']}")
            return True

        elif response.status_code == 401:
            print("Authentication failed")
            print("  Check your API key")
            return False

        elif response.status_code == 503:
            print("Service unavailable")
            print("  Model not loaded - check API logs")
            return False

        else:
            print(f"Request failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n" + "="*60)
    print("Testing batch predictions...")
    print("="*60)

    # Generate 3 days of predictions
    base_date = datetime.strptime("2024-07-15", "%Y-%m-%d")

    predictions = []
    for i in range(3):
        date = base_date + timedelta(days=i)
        predictions.append({
            "date": date.strftime("%Y-%m-%d"),
            "gmv": 9500000 + (i * 500000),
            "users": 85000 + (i * 5000),
            "marketing_cost": 175000 + (i * 5000)
        })

    data = {"predictions": predictions}

    print(f"\nRequest: {len(predictions)} predictions")
    for pred in predictions:
        print(f"  {pred['date']}: GMV=${pred['gmv']:,}, Users={pred['users']:,}")

    try:
        response = requests.post(
            f"{API_URL}/forecast/batch",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            count = result['count']
            results = result['predictions']

            print(f"\nBatch prediction successful!")
            print(f"  Processed: {count} predictions")
            print(f"\n  Results:")

            for pred in results:
                date = pred['date']
                fe = pred['predictions']['frontend_pods']
                be = pred['predictions']['backend_pods']
                print(f"    {date}: FE={fe}, BE={be}")
            return True

        elif response.status_code == 401:
            print("Authentication failed")
            return False

        else:
            print(f"Request failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def test_invalid_auth():
    """Test authentication with invalid key."""
    print("\n" + "="*60)
    print("Testing invalid authentication...")
    print("="*60)

    invalid_headers = {
        "X-API-Key": "invalid-key-123",
        "Content-Type": "application/json"
    }

    data = {
        "date": "2024-07-15",
        "gmv": 9500000,
        "users": 85000,
        "marketing_cost": 175000
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers=invalid_headers,
            json=data
        )

        if response.status_code == 401:
            print("Invalid auth correctly rejected")
            print(f"  Error: {response.json()['error']}")
            return True
        else:
            print(f"Unexpected response: {response.status_code}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Pod Forecasting API - Python Integration Demo")
    print("="*60)
    print(f"\nAPI URL: {API_URL}")
    print(f"API Key: {API_KEY[:20]}..." if len(API_KEY) > 20 else f"API Key: {API_KEY}")

    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Invalid Auth", test_invalid_auth),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nTest '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} - {name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())