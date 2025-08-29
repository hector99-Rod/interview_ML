# TODO: Train, assert artifacts exist and ROC-AUC threshold
import pytest
import os
import sys
import json

# Import src files
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.train import train_model

def test_training_artifacts_exist():
    """Test that training produces all required artifacts and meets ROC-AUC threshold."""
    data_path = "data/customer_churn_synth.csv"
    outdir = "artifacts"

    # Saves artifacts
    roc_auc = train_model(data_path, outdir)

    # Verifica existencia de artefactos requeridos
    assert os.path.exists(os.path.join(outdir, "model.pkl"))
    assert os.path.exists(os.path.join(outdir, "feature_pipeline.pkl"))
    assert os.path.exists(os.path.join(outdir, "metrics.json"))
    assert os.path.exists(os.path.join(outdir, "feature_importances.csv"))


    with open(os.path.join(outdir, 'metrics.json')) as f:
        metrics = json.load(f)

    assert "val_metrics" in metrics, "Missing 'val_metrics' in metrics.json"
    assert "roc_auc" in metrics["val_metrics"], "ROC-AUC missing in val_metrics"

    roc_auc = metrics["val_metrics"]["roc_auc"]
    assert roc_auc >= 0.83, f"ROC-AUC {roc_auc} is below required threshold 0.83"

if __name__ == "__main__":
    test_training_artifacts_exist()
    print("All tests passed!")
