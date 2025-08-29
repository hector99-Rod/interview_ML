# Mini-Prod ML Challenge (Starter)

Héctor Rodrigo project, 2025-27-08

after creaing your env run the commands below

# Train
python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/

# Serve
uvicorn src.app:app --port 8000

# Drift
python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv

# Agent Monitor
python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml

# Tests
pytest -q
