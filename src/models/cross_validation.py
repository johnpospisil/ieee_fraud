"""
Cross-Validation Module for IEEE-CIS Fraud Detection

This module provides advanced cross-validation strategies for fraud detection,
including time-series aware splitting, stability analysis, and performance tracking.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import List, Dict, Tuple, Optional, Any
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesCV:
    """
    Time-series cross-validation with expanding window strategy.
    
    This ensures no data leakage by always training on past data
    and validating on future data, which is critical for fraud detection.
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        """
        Initialize TimeSeriesCV.
        
        Parameters:
        -----------
        n_splits : int
            Number of CV folds
        test_size : float
            Proportion of data to use for validation in each fold
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Generate train/validation indices for time-series CV.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target variable (not used, for compatibility)
        
        Yields:
        -------
        train_idx, val_idx : tuple of arrays
            Training and validation indices
        """
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # Calculate validation start point
            val_start = int(n_samples * (1 - self.test_size * (self.n_splits - i) / self.n_splits))
            val_end = val_start + test_size_samples
            
            # Ensure we don't go beyond data
            if val_end > n_samples:
                val_end = n_samples
            
            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits


class RobustCrossValidator:
    """
    Advanced cross-validation with stability analysis and performance tracking.
    """
    
    def __init__(self, model_params: Dict[str, Any], cv_strategy: str = 'time_series',
                 n_splits: int = 5, random_state: int = 42, verbose: bool = True):
        """
        Initialize RobustCrossValidator.
        
        Parameters:
        -----------
        model_params : dict
            LightGBM model parameters
        cv_strategy : str
            CV strategy: 'time_series', 'stratified', or 'standard'
        n_splits : int
            Number of CV folds
        random_state : int
            Random seed
        verbose : bool
            Print progress messages
        """
        self.model_params = model_params
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        
        # Results storage
        self.cv_results = []
        self.oof_predictions = None
        self.feature_importance = None
        self.models = []
    
    def _get_cv_splitter(self):
        """Get the appropriate CV splitter based on strategy."""
        if self.cv_strategy == 'time_series':
            return TimeSeriesCV(n_splits=self.n_splits)
        elif self.cv_strategy == 'stratified':
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                   random_state=self.random_state)
        else:  # standard
            return KFold(n_splits=self.n_splits, shuffle=True,
                        random_state=self.random_state)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       num_boost_round: int = 2000,
                       early_stopping_rounds: int = 100,
                       categorical_features: List[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation with detailed tracking.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        num_boost_round : int
            Maximum number of boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        categorical_features : list, optional
            List of categorical feature names
        
        Returns:
        --------
        results : dict
            CV results including scores, predictions, and statistics
        """
        if self.verbose:
            print(f"Starting {self.cv_strategy} cross-validation with {self.n_splits} folds...")
            print(f"{'='*70}")
        
        cv_splitter = self._get_cv_splitter()
        self.oof_predictions = np.zeros(len(X))
        fold_scores = []
        fold_feature_importance = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            if self.verbose:
                print(f"\nFold {fold_idx + 1}/{self.n_splits}")
                print(f"{'-'*70}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if self.verbose:
                print(f"Train: {len(X_train):,} samples, Fraud rate: {y_train.mean()*100:.2f}%")
                print(f"Val:   {len(X_val):,} samples, Fraud rate: {y_val.mean()*100:.2f}%")
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train,
                                    categorical_feature=categorical_features)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                                  categorical_feature=categorical_features)
            
            # Train model
            callbacks = [
                lgb.log_evaluation(period=100 if self.verbose else 0),
                lgb.early_stopping(stopping_rounds=early_stopping_rounds)
            ]
            
            model = lgb.train(
                self.model_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )
            
            # Store model
            self.models.append(model)
            
            # Get predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Store OOF predictions
            self.oof_predictions[val_idx] = y_pred_val
            
            # Calculate metrics
            train_auc = roc_auc_score(y_train, y_pred_train)
            val_auc = roc_auc_score(y_val, y_pred_val)
            
            fold_result = {
                'fold': fold_idx + 1,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'best_iteration': model.best_iteration
            }
            
            self.cv_results.append(fold_result)
            fold_scores.append(val_auc)
            
            # Store feature importance
            importance = model.feature_importance(importance_type='gain')
            fold_feature_importance.append(importance)
            
            if self.verbose:
                print(f"\nFold {fold_idx + 1} Results:")
                print(f"  Train AUC: {train_auc:.6f}")
                print(f"  Val AUC:   {val_auc:.6f}")
                print(f"  Best iteration: {model.best_iteration}")
        
        # Calculate overall OOF score
        oof_auc = roc_auc_score(y, self.oof_predictions)
        
        # Average feature importance across folds
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.mean(fold_feature_importance, axis=0),
            'importance_std': np.std(fold_feature_importance, axis=0)
        }).sort_values('importance', ascending=False)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("CROSS-VALIDATION SUMMARY")
            print(f"{'='*70}")
            print(f"OOF AUC:           {oof_auc:.6f}")
            print(f"Mean fold AUC:     {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
            print(f"Min fold AUC:      {np.min(fold_scores):.6f}")
            print(f"Max fold AUC:      {np.max(fold_scores):.6f}")
            print(f"AUC std deviation: {np.std(fold_scores):.6f}")
            print(f"{'='*70}")
        
        return {
            'oof_auc': oof_auc,
            'mean_auc': np.mean(fold_scores),
            'std_auc': np.std(fold_scores),
            'min_auc': np.min(fold_scores),
            'max_auc': np.max(fold_scores),
            'fold_scores': fold_scores,
            'cv_results': self.cv_results,
            'oof_predictions': self.oof_predictions,
            'feature_importance': self.feature_importance
        }
    
    def get_feature_importance(self, top_n: int = 50) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
        
        Returns:
        --------
        importance_df : pd.DataFrame
            Top N features with importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Must run cross_validate() first")
        
        return self.feature_importance.head(top_n)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using all trained models (average).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        
        Returns:
        --------
        predictions : np.ndarray
            Average predictions across all folds
        """
        if not self.models:
            raise ValueError("Must run cross_validate() first")
        
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        
        return predictions / len(self.models)
    
    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze model stability across folds.
        
        Returns:
        --------
        stability_metrics : dict
            Various stability metrics
        """
        if not self.cv_results:
            raise ValueError("Must run cross_validate() first")
        
        fold_scores = [r['val_auc'] for r in self.cv_results]
        
        # Calculate coefficient of variation (lower is more stable)
        cv_coefficient = np.std(fold_scores) / np.mean(fold_scores)
        
        # Calculate max drop (max difference between consecutive folds)
        max_drop = max([abs(fold_scores[i] - fold_scores[i-1]) 
                       for i in range(1, len(fold_scores))])
        
        stability_metrics = {
            'coefficient_of_variation': cv_coefficient,
            'max_fold_difference': np.max(fold_scores) - np.min(fold_scores),
            'max_consecutive_drop': max_drop,
            'stability_score': 1 - cv_coefficient,  # Higher is more stable
        }
        
        return stability_metrics


class EnsembleCrossValidator:
    """
    Cross-validation for ensemble models with multiple base models.
    """
    
    def __init__(self, base_models: List[Dict[str, Any]], cv_strategy: str = 'time_series',
                 n_splits: int = 5, random_state: int = 42, verbose: bool = True):
        """
        Initialize EnsembleCrossValidator.
        
        Parameters:
        -----------
        base_models : list of dict
            List of model configurations, each with 'name' and 'params'
        cv_strategy : str
            CV strategy: 'time_series', 'stratified', or 'standard'
        n_splits : int
            Number of CV folds
        random_state : int
            Random seed
        verbose : bool
            Print progress messages
        """
        self.base_models = base_models
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        
        self.validators = []
        self.ensemble_results = None
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       num_boost_round: int = 2000,
                       early_stopping_rounds: int = 100) -> Dict[str, Any]:
        """
        Cross-validate all base models and ensemble.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        num_boost_round : int
            Maximum number of boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        
        Returns:
        --------
        results : dict
            CV results for all models and ensemble
        """
        if self.verbose:
            print(f"Cross-validating {len(self.base_models)} base models...")
            print(f"{'='*70}\n")
        
        oof_predictions_all = []
        model_results = []
        
        for idx, model_config in enumerate(self.base_models):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"MODEL {idx + 1}/{len(self.base_models)}: {model_config['name']}")
                print(f"{'='*70}")
            
            validator = RobustCrossValidator(
                model_params=model_config['params'],
                cv_strategy=self.cv_strategy,
                n_splits=self.n_splits,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            results = validator.cross_validate(
                X, y,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )
            
            self.validators.append(validator)
            oof_predictions_all.append(results['oof_predictions'])
            
            model_results.append({
                'name': model_config['name'],
                'oof_auc': results['oof_auc'],
                'mean_auc': results['mean_auc'],
                'std_auc': results['std_auc']
            })
        
        # Calculate ensemble OOF predictions (simple average)
        ensemble_oof = np.mean(oof_predictions_all, axis=0)
        ensemble_auc = roc_auc_score(y, ensemble_oof)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("ENSEMBLE CROSS-VALIDATION SUMMARY")
            print(f"{'='*70}")
            print("\nIndividual Model Results:")
            for result in model_results:
                print(f"  {result['name']:20s} OOF AUC: {result['oof_auc']:.6f} "
                      f"(CV: {result['mean_auc']:.6f} ± {result['std_auc']:.6f})")
            print(f"\nEnsemble (Average):              {ensemble_auc:.6f}")
            print(f"Best individual model:           {max([r['oof_auc'] for r in model_results]):.6f}")
            print(f"Improvement:                     +{ensemble_auc - max([r['oof_auc'] for r in model_results]):.6f}")
            print(f"{'='*70}")
        
        self.ensemble_results = {
            'ensemble_oof_auc': ensemble_auc,
            'ensemble_oof_predictions': ensemble_oof,
            'model_results': model_results,
            'individual_oof_predictions': oof_predictions_all
        }
        
        return self.ensemble_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        
        Returns:
        --------
        predictions : np.ndarray
            Ensemble predictions (average of all models)
        """
        if not self.validators:
            raise ValueError("Must run cross_validate() first")
        
        predictions = []
        for validator in self.validators:
            predictions.append(validator.predict(X))
        
        return np.mean(predictions, axis=0)


def quick_cv_comparison(X: pd.DataFrame, y: pd.Series,
                        model_params: Dict[str, Any],
                        strategies: List[str] = ['time_series', 'stratified'],
                        n_splits: int = 5) -> pd.DataFrame:
    """
    Quickly compare different CV strategies.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    model_params : dict
        LightGBM model parameters
    strategies : list of str
        CV strategies to compare
    n_splits : int
        Number of CV folds
    
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison of CV strategies
    """
    print("Comparing CV strategies...")
    print(f"{'='*70}\n")
    
    results = []
    
    for strategy in strategies:
        print(f"\nTesting {strategy} CV...")
        
        validator = RobustCrossValidator(
            model_params=model_params,
            cv_strategy=strategy,
            n_splits=n_splits,
            verbose=False
        )
        
        cv_results = validator.cross_validate(X, y)
        stability = validator.analyze_stability()
        
        results.append({
            'strategy': strategy,
            'oof_auc': cv_results['oof_auc'],
            'mean_auc': cv_results['mean_auc'],
            'std_auc': cv_results['std_auc'],
            'min_auc': cv_results['min_auc'],
            'max_auc': cv_results['max_auc'],
            'stability_score': stability['stability_score'],
            'cv_coefficient': stability['coefficient_of_variation']
        })
        
        print(f"  OOF AUC: {cv_results['oof_auc']:.6f}")
        print(f"  Stability: {stability['stability_score']:.4f}")
    
    comparison_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("CV STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*70}")
    
    return comparison_df


def analyze_cv_consistency(cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze consistency across CV folds.
    
    Parameters:
    -----------
    cv_results : list of dict
        CV results from RobustCrossValidator
    
    Returns:
    --------
    consistency_metrics : dict
        Various consistency metrics
    """
    fold_scores = [r['val_auc'] for r in cv_results]
    
    metrics = {
        'mean_score': np.mean(fold_scores),
        'std_score': np.std(fold_scores),
        'min_score': np.min(fold_scores),
        'max_score': np.max(fold_scores),
        'range': np.max(fold_scores) - np.min(fold_scores),
        'cv_coefficient': np.std(fold_scores) / np.mean(fold_scores),
        'consistency_score': 1 - (np.std(fold_scores) / np.mean(fold_scores))
    }
    
    return metrics
