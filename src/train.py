# TODO: Implement training script.
# CLI: python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime
import subprocess
import sys

#from features import FeaturePreprocessor
#from models import ChurnModel

from src.features import FeaturePreprocessor
from src.models import ChurnModel

def get_git_sha():
    """Get current git SHA if available"""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except:
        return "unknown"

def train_model(data_path, outdir):
    """Train the churn prediction model"""
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Split features and target
    X = data.drop('churned', axis=1)
    y = data['churned']
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess features
    print("Preprocessing features...")
    preprocessor = FeaturePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Train model
    print("Training model...")
    model = ChurnModel()
    model.fit(X_train_processed, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    train_metrics = model.evaluate(X_train_processed, y_train)
    val_metrics = model.evaluate(X_val_processed, y_val)
    
    # Save artifacts
    print("Saving artifacts...")
    model.save(os.path.join(outdir, 'model.pkl'))
    preprocessor.save(os.path.join(outdir, 'feature_pipeline.pkl'))
    
    # Save metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'git_sha': get_git_sha(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature importances
    feature_importances = pd.DataFrame({
        'feature': preprocessor.preprocessor.get_feature_names_out(),
        'importance': model.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importances.to_csv(os.path.join(outdir, 'feature_importances.csv'), index=False)
    
    print(f"Training completed. Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    return val_metrics['roc_auc']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--outdir', type=str, default='artifacts', help='Output directory')
    
    args = parser.parse_args()
    
    roc_auc = train_model(args.data, args.outdir)
    
    if roc_auc >= 0.83:
        print("✓ Model meets acceptance criteria (ROC-AUC ≥ 0.83)")
        sys.exit(0)
    else:
        print("✗ Model does not meet acceptance criteria")
        sys.exit(1)