# TODO: Boot API, call /predict using tests/sample.json

import pytest
import requests
import json
import time
import subprocess
import threading
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

from io_schemas import PredictionInput

def start_server():
    import uvicorn
    import os
    import sys

    # Principal Directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)

    # Add src to path
    src_dir = os.path.join(project_root, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")


def test_inference():
    """Test the inference endpoint with sample data"""
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(6)
    
    # Prepare sample data
    sample_data = [
        PredictionInput(
            plan_type="Standard",
            contract_type="Monthly",
            autopay="Yes",
            is_promo_user="No",
            add_on_count=2,
            tenure_months=12,
            monthly_usage_gb=45.6,
            avg_latency_ms=120.5,
            support_tickets_30d=1,
            discount_pct=10.0,
            payment_failures_90d=0,
            downtime_hours_30d=2.5
        ),
        PredictionInput(
            plan_type="Pro",
            contract_type="Annual",
            autopay="Yes",
            is_promo_user="Yes",
            add_on_count=3,
            tenure_months=24,
            monthly_usage_gb=78.9,
            avg_latency_ms=85.2,
            support_tickets_30d=0,
            discount_pct=15.0,
            payment_failures_90d=1,
            downtime_hours_30d=1.2
        )
    ]
    
    # Make prediction request
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    data = {"data": [item.dict() for item in sample_data]}
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check response
    assert response.status_code == 200, f"Request failed with status {response.status_code}"
    
    predictions = response.json()["predictions"]
    assert len(predictions) == 2, "Should return 2 predictions"
    
    for pred in predictions:
        assert 0 <= pred["churn_probability"] <= 1, "Probability should be between 0 and 1"
        assert isinstance(pred["churned"], bool), "Churned should be boolean"
    
    print("Inference test passed!")

if __name__ == "__main__":
    test_inference()