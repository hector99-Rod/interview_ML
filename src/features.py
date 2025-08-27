# TODO: Implement sklearn ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class FeaturePreprocessor:
    """Preprocess features for churn prediction"""
    
    def __init__(self):
        self.numeric_features = [
            'add_on_count', 'tenure_months', 'monthly_usage_gb', 
            'avg_latency_ms', 'support_tickets_30d', 'discount_pct', 
            'payment_failures_90d', 'downtime_hours_30d'
        ]
        
        self.categorical_features = [
            'plan_type', 'contract_type', 'autopay', 'is_promo_user'
        ]
        
        self.preprocessor = self._build_preprocessor()
    
    def _build_preprocessor(self):
        """Build the preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def fit(self, X, y=None):
        """Fit the preprocessor"""
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X):
        """Transform the data"""
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform the data"""
        return self.preprocessor.fit_transform(X, y)
    
    def save(self, filepath):
        """Save the preprocessor"""
        joblib.dump(self.preprocessor, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load the preprocessor"""
        preprocessor = joblib.load(filepath)
        instance = cls()
        instance.preprocessor = preprocessor
        return instance
