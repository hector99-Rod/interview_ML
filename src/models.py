# TODO: Train/save/load utilities
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
import joblib

class ChurnModel:
    """Churn prediction model"""
    
    def __init__(self, model=None):
        self.model = model or xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        probs = self.predict_proba(X)
        preds = self.predict(X)
        
        return {
            'roc_auc': roc_auc_score(y, probs),
            'pr_auc': average_precision_score(y, probs),
            'accuracy': accuracy_score(y, preds)
        }
    
    def save(self, filepath):
        """Save the model"""
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load the model"""
        model = joblib.load(filepath)
        return cls(model)