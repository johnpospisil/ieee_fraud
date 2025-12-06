"""
LightGBM Baseline Model for IEEE-CIS Fraud Detection
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Optional
import pickle
import json
from datetime import datetime


class LGBMBaseline:
    """
    LightGBM baseline model for fraud detection.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LightGBM baseline model.
        
        Args:
            params: LightGBM parameters (if None, uses default parameters)
        """
        self.params = params or self.get_default_params()
        self.model = None
        self.feature_importance = None
        self.training_history = {}
        
    @staticmethod
    def get_default_params() -> Dict:
        """
        Get default LightGBM parameters optimized for fraud detection.
        
        Returns:
            Dictionary of LightGBM parameters
        """
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }
    
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: pd.DataFrame,
              y_val: pd.Series,
              num_boost_round: int = 1000,
              early_stopping_rounds: int = 100,
              verbose_eval: int = 50) -> Dict[str, float]:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            verbose_eval: Print evaluation metric every N rounds
            
        Returns:
            Dictionary with training metrics
        """
        print("="*60)
        print("TRAINING LIGHTGBM BASELINE MODEL")
        print("="*60)
        print(f"\nTraining samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"\nTraining fraud rate: {y_train.mean()*100:.2f}%")
        print(f"Validation fraud rate: {y_val.mean()*100:.2f}%")
        print()
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        print("Training model...")
        evals_result = {}
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=verbose_eval),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        # Store training history
        self.training_history = evals_result
        
        # Calculate final scores
        train_pred = self.model.predict(X_train, num_iteration=self.model.best_iteration)
        val_pred = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        
        # Get feature importance
        self.feature_importance = dict(zip(X_train.columns, self.model.feature_importance(importance_type='gain')))
        
        metrics = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'best_iteration': self.model.best_iteration,
            'num_features': X_train.shape[1]
        }
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nBest iteration: {self.model.best_iteration}")
        print(f"Training AUC:   {train_auc:.6f}")
        print(f"Validation AUC: {val_auc:.6f}")
        print("="*60)
        print()
        
        return metrics
    
    def predict(self, X: pd.DataFrame, use_best_iteration: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            use_best_iteration: Whether to use best iteration from early stopping
            
        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        num_iteration = self.model.best_iteration if use_best_iteration else None
        predictions = self.model.predict(X, num_iteration=num_iteration)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Args:
            top_n: Number of top features to return (if None, returns all)
            
        Returns:
            DataFrame with features and importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet!")
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Save LightGBM model
        self.model.save_model(filepath)
        
        # Save additional metadata
        metadata = {
            'params': self.params,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'best_iteration': self.model.best_iteration,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = filepath.replace('.txt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved: {filepath}")
        print(f"✓ Metadata saved: {metadata_path}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        self.model = lgb.Booster(model_file=filepath)
        
        # Load metadata if available
        metadata_path = filepath.replace('.txt', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.params = metadata.get('params', {})
                self.feature_importance = metadata.get('feature_importance', {})
                self.training_history = metadata.get('training_history', {})
            print(f"✓ Model loaded: {filepath}")
            print(f"✓ Metadata loaded: {metadata_path}")
        except FileNotFoundError:
            print(f"✓ Model loaded: {filepath}")
            print(f"⚠ Metadata not found: {metadata_path}")


def train_baseline_model(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: pd.DataFrame,
                         y_val: pd.Series,
                         params: Optional[Dict] = None,
                         num_boost_round: int = 1000,
                         early_stopping_rounds: int = 100) -> Tuple[LGBMBaseline, Dict[str, float]]:
    """
    Quick function to train baseline LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: Optional custom parameters
        num_boost_round: Maximum boosting rounds
        early_stopping_rounds: Early stopping rounds
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    model = LGBMBaseline(params=params)
    metrics = model.train(
        X_train, y_train, X_val, y_val,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )
    
    return model, metrics


if __name__ == "__main__":
    print("LightGBM Baseline Model module loaded successfully!")
    print("\nExample usage:")
    print("  from src.models.lgbm_baseline import LGBMBaseline, train_baseline_model")
    print("  model, metrics = train_baseline_model(X_train, y_train, X_val, y_val)")
