"""
Test Predictions Module for IEEE-CIS Fraud Detection

This module provides utilities for generating test set predictions,
including preprocessing, feature engineering, and ensemble prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import lightgbm as lgb
import json
import os
import warnings

warnings.filterwarnings('ignore')


class TestPredictor:
    """
    Generate predictions on test set using trained models.
    """
    
    def __init__(self, models: List[lgb.Booster], verbose: bool = True):
        """
        Initialize TestPredictor.
        
        Parameters:
        -----------
        models : list of lgb.Booster
            List of trained LightGBM models (from CV folds)
        verbose : bool
            Print progress messages
        """
        self.models = models
        self.verbose = verbose
        self.feature_names = None
        
        if models:
            self.feature_names = models[0].feature_name()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions by averaging across all models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test feature matrix
        
        Returns:
        --------
        predictions : np.ndarray
            Average predictions across all models
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        # Validate features
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training
            X = X[self.feature_names]
        
        if self.verbose:
            print(f"Generating predictions with {len(self.models)} models...")
        
        predictions = np.zeros(len(X))
        
        for idx, model in enumerate(self.models):
            if self.verbose:
                print(f"  Model {idx + 1}/{len(self.models)}...", end='\r')
            predictions += model.predict(X)
        
        predictions /= len(self.models)
        
        if self.verbose:
            print(f"\n✓ Predictions generated for {len(X):,} samples")
            print(f"  Mean: {predictions.mean():.6f}")
            print(f"  Std:  {predictions.std():.6f}")
            print(f"  Min:  {predictions.min():.6f}")
            print(f"  Max:  {predictions.max():.6f}")
        
        return predictions
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test feature matrix
        
        Returns:
        --------
        predictions : np.ndarray
            Average predictions
        uncertainty : np.ndarray
            Standard deviation of predictions across models
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        # Validate and reorder features
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        if self.verbose:
            print(f"Generating predictions with uncertainty estimates...")
        
        all_predictions = []
        
        for idx, model in enumerate(self.models):
            if self.verbose:
                print(f"  Model {idx + 1}/{len(self.models)}...", end='\r')
            all_predictions.append(model.predict(X))
        
        all_predictions = np.array(all_predictions)
        predictions = np.mean(all_predictions, axis=0)
        uncertainty = np.std(all_predictions, axis=0)
        
        if self.verbose:
            print(f"\n✓ Predictions with uncertainty generated")
            print(f"  Mean prediction: {predictions.mean():.6f}")
            print(f"  Mean uncertainty: {uncertainty.mean():.6f}")
            print(f"  Max uncertainty: {uncertainty.max():.6f}")
        
        return predictions, uncertainty


class EnsembleTestPredictor:
    """
    Generate ensemble predictions using multiple model configurations.
    """
    
    def __init__(self, predictors: List[TestPredictor], 
                 weights: Optional[List[float]] = None,
                 verbose: bool = True):
        """
        Initialize EnsembleTestPredictor.
        
        Parameters:
        -----------
        predictors : list of TestPredictor
            List of test predictors for different models
        weights : list of float, optional
            Weights for each predictor (defaults to equal weights)
        verbose : bool
            Print progress messages
        """
        self.predictors = predictors
        self.verbose = verbose
        
        if weights is None:
            self.weights = np.ones(len(predictors)) / len(predictors)
        else:
            if len(weights) != len(predictors):
                raise ValueError("Number of weights must match number of predictors")
            self.weights = np.array(weights)
            self.weights /= self.weights.sum()  # Normalize
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate weighted ensemble predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test feature matrix
        
        Returns:
        --------
        predictions : np.ndarray
            Weighted ensemble predictions
        """
        if self.verbose:
            print(f"Generating ensemble predictions with {len(self.predictors)} models...")
        
        predictions = np.zeros(len(X))
        
        for idx, (predictor, weight) in enumerate(zip(self.predictors, self.weights)):
            if self.verbose:
                print(f"\nModel {idx + 1}/{len(self.predictors)} (weight: {weight:.3f})")
            
            pred = predictor.predict(X)
            predictions += weight * pred
        
        if self.verbose:
            print(f"\n✓ Ensemble predictions complete")
            print(f"  Mean: {predictions.mean():.6f}")
            print(f"  Std:  {predictions.std():.6f}")
        
        return predictions
    
    def predict_with_diversity(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with diversity measure across models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test feature matrix
        
        Returns:
        --------
        predictions : np.ndarray
            Weighted ensemble predictions
        diversity : np.ndarray
            Standard deviation across different model types
        """
        if self.verbose:
            print(f"Generating ensemble predictions with diversity analysis...")
        
        all_predictions = []
        predictions = np.zeros(len(X))
        
        for idx, (predictor, weight) in enumerate(zip(self.predictors, self.weights)):
            if self.verbose:
                print(f"\nModel {idx + 1}/{len(self.predictors)}")
            
            pred = predictor.predict(X)
            all_predictions.append(pred)
            predictions += weight * pred
        
        all_predictions = np.array(all_predictions)
        diversity = np.std(all_predictions, axis=0)
        
        if self.verbose:
            print(f"\n✓ Ensemble predictions with diversity complete")
            print(f"  Mean prediction: {predictions.mean():.6f}")
            print(f"  Mean diversity:  {diversity.mean():.6f}")
        
        return predictions, diversity


def create_submission_file(test_ids: pd.Series, 
                          predictions: np.ndarray,
                          output_path: str = '../submissions/submission.csv',
                          verbose: bool = True) -> pd.DataFrame:
    """
    Create submission file in competition format.
    
    Parameters:
    -----------
    test_ids : pd.Series
        Test transaction IDs
    predictions : np.ndarray
        Predicted fraud probabilities
    output_path : str
        Path to save submission file
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    submission : pd.DataFrame
        Submission dataframe
    """
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': predictions
    })
    
    # Ensure proper format
    submission['TransactionID'] = submission['TransactionID'].astype(int)
    
    # Clip predictions to valid range [0, 1]
    submission['isFraud'] = submission['isFraud'].clip(0, 1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save submission
    submission.to_csv(output_path, index=False)
    
    if verbose:
        print(f"✓ Submission file created: {output_path}")
        print(f"  Samples: {len(submission):,}")
        print(f"  Columns: {list(submission.columns)}")
        print(f"  Mean prediction: {submission['isFraud'].mean():.6f}")
        print(f"  Min prediction:  {submission['isFraud'].min():.6f}")
        print(f"  Max prediction:  {submission['isFraud'].max():.6f}")
        print(f"\nFirst few rows:")
        print(submission.head(10))
    
    return submission


def validate_submission(submission_path: str,
                       sample_submission_path: str = '../data/sample_submission.csv',
                       verbose: bool = True) -> bool:
    """
    Validate submission file format against sample submission.
    
    Parameters:
    -----------
    submission_path : str
        Path to submission file to validate
    sample_submission_path : str
        Path to sample submission file
    verbose : bool
        Print validation messages
    
    Returns:
    --------
    is_valid : bool
        Whether submission is valid
    """
    try:
        # Load files
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_submission_path)
        
        if verbose:
            print("Validating submission file...")
        
        # Check columns
        if list(submission.columns) != list(sample.columns):
            if verbose:
                print(f"✗ Column mismatch!")
                print(f"  Expected: {list(sample.columns)}")
                print(f"  Got: {list(submission.columns)}")
            return False
        
        # Check number of rows
        if len(submission) != len(sample):
            if verbose:
                print(f"✗ Row count mismatch!")
                print(f"  Expected: {len(sample):,}")
                print(f"  Got: {len(submission):,}")
            return False
        
        # Check TransactionIDs match
        if not submission['TransactionID'].equals(sample['TransactionID']):
            if verbose:
                print("✗ TransactionID mismatch!")
            return False
        
        # Check prediction range
        if submission['isFraud'].min() < 0 or submission['isFraud'].max() > 1:
            if verbose:
                print(f"✗ Predictions out of range [0, 1]!")
                print(f"  Min: {submission['isFraud'].min()}")
                print(f"  Max: {submission['isFraud'].max()}")
            return False
        
        # Check for missing values
        if submission.isnull().any().any():
            if verbose:
                print("✗ Missing values detected!")
            return False
        
        if verbose:
            print("✓ Submission validation passed!")
            print(f"  Rows: {len(submission):,}")
            print(f"  Columns: {list(submission.columns)}")
            print(f"  Prediction range: [{submission['isFraud'].min():.6f}, {submission['isFraud'].max():.6f}]")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Validation error: {str(e)}")
        return False


def analyze_predictions(predictions: np.ndarray,
                       uncertainty: Optional[np.ndarray] = None,
                       percentiles: List[int] = [1, 5, 10, 25, 50, 75, 90, 95, 99],
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze prediction distribution and statistics.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted fraud probabilities
    uncertainty : np.ndarray, optional
        Uncertainty estimates for predictions
    percentiles : list of int
        Percentiles to compute
    verbose : bool
        Print analysis
    
    Returns:
    --------
    analysis : dict
        Prediction statistics
    """
    analysis = {
        'count': len(predictions),
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'percentiles': {}
    }
    
    # Compute percentiles
    for p in percentiles:
        analysis['percentiles'][f'p{p}'] = float(np.percentile(predictions, p))
    
    # Add uncertainty stats if available
    if uncertainty is not None:
        analysis['uncertainty'] = {
            'mean': float(np.mean(uncertainty)),
            'std': float(np.std(uncertainty)),
            'min': float(np.min(uncertainty)),
            'max': float(np.max(uncertainty))
        }
    
    if verbose:
        print("="*70)
        print("PREDICTION ANALYSIS")
        print("="*70)
        print(f"Count:        {analysis['count']:,}")
        print(f"Mean:         {analysis['mean']:.6f}")
        print(f"Std:          {analysis['std']:.6f}")
        print(f"Min:          {analysis['min']:.6f}")
        print(f"Max:          {analysis['max']:.6f}")
        print("\nPercentiles:")
        for p in percentiles:
            print(f"  {p:2d}%: {analysis['percentiles'][f'p{p}']:.6f}")
        
        if uncertainty is not None:
            print("\nUncertainty:")
            print(f"  Mean: {analysis['uncertainty']['mean']:.6f}")
            print(f"  Std:  {analysis['uncertainty']['std']:.6f}")
            print(f"  Min:  {analysis['uncertainty']['min']:.6f}")
            print(f"  Max:  {analysis['uncertainty']['max']:.6f}")
        
        print("="*70)
    
    return analysis


def save_prediction_metadata(predictions: np.ndarray,
                            models_info: Dict[str, Any],
                            output_path: str = '../submissions/prediction_metadata.json',
                            verbose: bool = True):
    """
    Save metadata about predictions for tracking and reproducibility.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted fraud probabilities
    models_info : dict
        Information about models used
    output_path : str
        Path to save metadata
    verbose : bool
        Print progress messages
    """
    metadata = {
        'prediction_stats': {
            'count': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
        },
        'models': models_info,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    if verbose:
        print(f"✓ Prediction metadata saved: {output_path}")


def rank_average_predictions(*predictions_list: np.ndarray) -> np.ndarray:
    """
    Average predictions based on their ranks rather than raw values.
    This can be more robust to different prediction scales.
    
    Parameters:
    -----------
    *predictions_list : np.ndarray
        Variable number of prediction arrays
    
    Returns:
    --------
    rank_avg : np.ndarray
        Rank-averaged predictions
    """
    from scipy.stats import rankdata
    
    ranks = [rankdata(pred) / len(pred) for pred in predictions_list]
    rank_avg = np.mean(ranks, axis=0)
    
    return rank_avg


def blend_predictions(predictions_dict: Dict[str, np.ndarray],
                     weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Blend multiple prediction sets with optional weights.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary mapping model names to prediction arrays
    weights : dict, optional
        Dictionary mapping model names to weights
    
    Returns:
    --------
    blended : np.ndarray
        Blended predictions
    """
    if weights is None:
        # Equal weights
        weights = {name: 1.0 / len(predictions_dict) 
                  for name in predictions_dict.keys()}
    else:
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    
    blended = np.zeros_like(next(iter(predictions_dict.values())))
    
    for name, pred in predictions_dict.items():
        blended += weights[name] * pred
    
    return blended
