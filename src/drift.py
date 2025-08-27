import argparse
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
import warnings

warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful. Switching to method=asymp.")

def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[-1] += 1e-6
    expected_counts, _ = np.histogram(expected, breakpoints)
    expected_percents = expected_counts / len(expected)
    actual_counts, _ = np.histogram(actual, breakpoints)
    actual_percents = actual_counts / len(actual)
    expected_percents = np.clip(expected_percents, 1e-6, 1)
    actual_percents = np.clip(actual_percents, 1e-6, 1)
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi

def detect_drift(ref_data, new_data, threshold=0.2):
    results = {
        "threshold": threshold,
        "overall_drift": False,
        "features": {}
    }
    drift_detected = False
    for column in ref_data.columns:
        if column not in new_data.columns:
            continue
        if ref_data[column].dtype in ['float64', 'int64']:
            ks_stat, ks_pvalue = ks_2samp(ref_data[column], new_data[column])
            psi = calculate_psi(ref_data[column], new_data[column])
            feature_drift = psi > threshold or ks_pvalue < 0.05
            results["features"][column] = round(psi, 3)
            if feature_drift:
                drift_detected = True
        else:
            ref_counts = ref_data[column].value_counts(normalize=True).sort_index()
            new_counts = new_data[column].value_counts(normalize=True).sort_index()
            all_categories = list(set(ref_counts.index) | set(new_counts.index))
            ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
            new_aligned = new_counts.reindex(all_categories, fill_value=0)
            tvd = 0.5 * np.sum(np.abs(ref_aligned - new_aligned))
            feature_drift = tvd > threshold
            results["features"][column] = round(tvd, 3)
            if feature_drift:
                drift_detected = True
    results["overall_drift"] = drift_detected
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", default="artifacts/drift_report.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    ref_data = pd.read_csv(args.ref)
    new_data = pd.read_csv(args.new)

    drift_report = detect_drift(ref_data, new_data, args.threshold)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(drift_report, f, indent=2)

    print(json.dumps(drift_report))

if __name__ == "__main__":
    main()
