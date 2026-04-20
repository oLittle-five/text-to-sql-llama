"""
Test script for the Text-to-SQL FastAPI endpoint.

Run the server first:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Then run this script:
    python scripts/test_api.py [--url http://localhost:8000]
"""

import argparse
import json
import requests


def test_health(base_url: str):
    """Test the health endpoint."""
    print("=" * 60)
    print("TEST: Health Check")
    print("=" * 60)
    resp = requests.get(f"{base_url}/health")
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("PASSED\n")


def test_single_predict(base_url: str):
    """Test a single prediction."""
    print("=" * 60)
    print("TEST: Single Prediction")
    print("=" * 60)

    payload = {
        "question": "How many people live in Tokyo?",
        "columns": ["city", "country", "population"],
        "types": ["text", "text", "real"],
    }
    print(f"Request: {json.dumps(payload, indent=2)}")

    resp = requests.post(f"{base_url}/predict", json=payload)
    print(f"\nStatus: {resp.status_code}")
    result = resp.json()
    print(f"SQL: {result['sql']}")
    print(f"Time: {result['generation_time_ms']:.0f}ms")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Generated tokens: {result['generated_tokens']}")
    assert resp.status_code == 200
    assert "SELECT" in result["sql"].upper()
    print("PASSED\n")


def test_batch_predict(base_url: str):
    """Test batch prediction."""
    print("=" * 60)
    print("TEST: Batch Prediction")
    print("=" * 60)

    payload = {
        "queries": [
            {
                "question": "What is the nationality of Terrence Ross?",
                "columns": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"],
                "types": ["text", "text", "text", "text", "text", "text"],
            },
            {
                "question": "How many schools or teams had Jalen Rose?",
                "columns": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"],
                "types": ["text", "text", "text", "text", "text", "text"],
            },
            {
                "question": "What was the date of the race in Misano?",
                "columns": ["No", "Date", "Circuit", "Pole Position", "Race winner"],
                "types": ["real", "text", "text", "text", "text"],
            },
        ]
    }
    print(f"Sending {len(payload['queries'])} queries...")

    resp = requests.post(f"{base_url}/predict/batch", json=payload)
    print(f"\nStatus: {resp.status_code}")
    result = resp.json()

    for i, r in enumerate(result["results"]):
        print(f"\n  Query {i+1}: {payload['queries'][i]['question']}")
        print(f"  SQL:    {r['sql']}")
        print(f"  Time:   {r['generation_time_ms']:.0f}ms")

    print(f"\n  Total time: {result['total_time_ms']:.0f}ms")
    assert resp.status_code == 200
    assert len(result["results"]) == 3
    print("PASSED\n")


def test_validation_error(base_url: str):
    """Test that mismatched columns/types returns 400."""
    print("=" * 60)
    print("TEST: Validation Error (mismatched columns/types)")
    print("=" * 60)

    payload = {
        "question": "Test question",
        "columns": ["col1", "col2"],
        "types": ["text"],  # intentionally wrong length
    }

    resp = requests.post(f"{base_url}/predict", json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 400
    print("PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="Test the Text-to-SQL API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    args = parser.parse_args()

    print(f"\nTesting API at {args.url}\n")

    test_health(args.url)
    test_single_predict(args.url)
    test_batch_predict(args.url)
    test_validation_error(args.url)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
