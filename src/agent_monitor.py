# TODO: Implement Agentic Monitor (LLM-optional)
# CLI: python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml
import argparse
import json
import yaml
import pandas as pd
from datetime import timedelta

def load_metrics(metrics_file):
    df = pd.read_json(metrics_file, lines=True)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
    return df.sort_values('timestamp')

def analyze_metrics(df):
    if len(df) < 2:
        return []
    latest = df.iloc[-1]
    week_ago = latest['timestamp'] - timedelta(days=7)
    recent = df[df['timestamp'] >= week_ago]
    findings = []
    if 'roc_auc' in latest and len(recent) > 1:
        median_auc = recent['roc_auc'].median()
        drop_pct = (median_auc - latest['roc_auc']) / median_auc * 100
        if drop_pct >= 3:
            findings.append({"roc_auc_drop_pct": round(drop_pct, 1)})
    if 'latency_p95_ms' in latest and len(df) >= 2:
        if all(df.tail(2)['latency_p95_ms'] > 400):
            findings.append({"latency_p95_ms": float(latest['latency_p95_ms'])})
    return findings

def generate_plan(findings, drift_report):
    status = "healthy"
    actions = ["do_nothing"]
    rationale = []

    roc_auc_drop = next((f for f in findings if "roc_auc_drop_pct" in f), None)
    high_latency = next((f for f in findings if "latency_p95_ms" in f), None)
    drift = drift_report.get("overall_drift", False)

    if roc_auc_drop and roc_auc_drop["roc_auc_drop_pct"] >= 6:
        status = "critical"
        actions = ["open_incident", "trigger_retraining", "page_oncall=true"]
    elif drift and roc_auc_drop and roc_auc_drop["roc_auc_drop_pct"] >= 5:
        status = "critical"
        actions = ["open_incident", "trigger_retraining"]
    elif roc_auc_drop and roc_auc_drop["roc_auc_drop_pct"] >= 3:
        status = "warn"
        actions = ["trigger_retraining", "raise_thresholds"]
    elif high_latency:
        status = "warn"
        actions = ["raise_thresholds"]

    if roc_auc_drop:
        rationale.append(f"AUC fell {roc_auc_drop['roc_auc_drop_pct']}% vs 7-day median")
    if high_latency:
        rationale.append("p95 latency > 400ms for two windows")
    if drift:
        rationale.append("overall data drift detected")

    if not rationale:
        rationale = ["No significant issues detected"]

    return {
        "status": status,
        "findings": findings,
        "actions": actions,
        "rationale": "; ".join(rationale)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--drift', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    metrics_df = load_metrics(args.metrics)
    with open(args.drift) as f:
        drift_report = json.load(f)

    findings = analyze_metrics(metrics_df)
    plan = generate_plan(findings, drift_report)

    with open(args.out, 'w') as f:
        yaml.dump(plan, f)

    print(f"Agent plan saved to {args.out}")
    print(f"Status: {plan['status']}")
    print(f"Actions: {', '.join(plan['actions'])}")

if __name__ == "__main__":
    main()
