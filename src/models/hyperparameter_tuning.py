"""
Hyperparameter Tuning Module for LightGBM

This module provides utilities for systematic hyperparameter optimization using Optuna.
It supports both individual parameter tuning and full grid/random search.

Author: IEEE Fraud Detection Team
Milestone: 11 - Hyperparameter Tuning
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice
)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')


class LGBMTuner:
    """
    Hyperparameter tuning for LightGBM using Optuna optimization.
    
    Supports:
    - Bayesian optimization with Optuna
    - K-fold cross-validation
    - Parameter importance analysis
    - Optimization history visualization
    """
    
    def __init__(self, X_train, y_train, X_val=None, y_val=None, 
                 n_folds=5, random_state=42, verbose=True):
        """
        Initialize the tuner.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features (if provided, uses train/val split instead of CV)
        y_val : pd.Series, optional
            Validation target
        n_folds : int, default=5
            Number of folds for cross-validation
        random_state : int, default=42
            Random state for reproducibility
        verbose : bool, default=True
            Whether to print progress
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.best_params = None
        self.study = None
        
    def _objective(self, trial, fixed_params=None):
        """
        Objective function for Optuna optimization.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        fixed_params : dict, optional
            Fixed parameters not to tune
            
        Returns:
        --------
        float : Mean CV AUC score
        """
        # Define parameter search space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': -1,
        }
        
        # Add tunable parameters
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
        params['num_leaves'] = trial.suggest_int('num_leaves', 32, 512)
        params['max_depth'] = trial.suggest_int('max_depth', 6, 15)
        params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 100)
        params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        params['subsample_freq'] = 1
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 2.0)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 2.0)
        
        # Override with fixed params if provided
        if fixed_params:
            params.update(fixed_params)
        
        # Use validation set if provided, otherwise use CV
        if self.X_val is not None and self.y_val is not None:
            # Single train/val split
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
            
            y_pred = model.predict(self.X_val)
            score = roc_auc_score(self.y_val, y_pred)
            
        else:
            # K-fold cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=False, random_state=self.random_state)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
                X_tr = self.X_train.iloc[train_idx]
                y_tr = self.y_train.iloc[train_idx]
                X_vl = self.X_train.iloc[val_idx]
                y_vl = self.y_train.iloc[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_vl, label=y_vl, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=2000,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                )
                
                y_pred = model.predict(X_vl)
                fold_score = roc_auc_score(y_vl, y_pred)
                scores.append(fold_score)
            
            score = np.mean(scores)
        
        return score
    
    def tune(self, n_trials=100, timeout=None, fixed_params=None):
        """
        Run hyperparameter tuning.
        
        Parameters:
        -----------
        n_trials : int, default=100
            Number of optimization trials
        timeout : int, optional
            Time limit in seconds
        fixed_params : dict, optional
            Fixed parameters not to tune
            
        Returns:
        --------
        dict : Best parameters found
        """
        if self.verbose:
            print("="*70)
            print("HYPERPARAMETER TUNING")
            print("="*70)
            print(f"\nOptimization settings:")
            print(f"  • Number of trials: {n_trials}")
            print(f"  • Timeout: {timeout if timeout else 'None'}")
            print(f"  • Validation strategy: {'Train/Val split' if self.X_val is not None else f'{self.n_folds}-fold CV'}")
            print(f"  • Training samples: {len(self.X_train):,}")
            if self.X_val is not None:
                print(f"  • Validation samples: {len(self.X_val):,}")
            print(f"\nStarting optimization...\n")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, fixed_params),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        if self.verbose:
            print("\n" + "="*70)
            print("OPTIMIZATION COMPLETE")
            print("="*70)
            print(f"\nBest AUC: {self.study.best_value:.6f}")
            print(f"\nBest parameters:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  • {param:25s} {value:.6f}")
                else:
                    print(f"  • {param:25s} {value}")
            print("="*70)
        
        return self.best_params
    
    def get_best_params(self, include_fixed=True):
        """
        Get best parameters with fixed parameters.
        
        Parameters:
        -----------
        include_fixed : bool, default=True
            Whether to include fixed parameters
            
        Returns:
        --------
        dict : Complete parameter dictionary
        """
        if self.best_params is None:
            raise ValueError("No tuning has been performed yet. Call tune() first.")
        
        if include_fixed:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'random_state': self.random_state,
                'n_jobs': -1,
                'subsample_freq': 1,
            }
            params.update(self.best_params)
            return params
        else:
            return self.best_params
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No tuning has been performed yet. Call tune() first.")
        
        fig = plot_optimization_history(self.study)
        fig.update_layout(
            title="Optimization History",
            xaxis_title="Trial",
            yaxis_title="ROC-AUC Score",
            width=1000,
            height=500
        )
        fig.show()
    
    def plot_param_importances(self):
        """Plot parameter importances."""
        if self.study is None:
            raise ValueError("No tuning has been performed yet. Call tune() first.")
        
        fig = plot_param_importances(self.study)
        fig.update_layout(
            title="Parameter Importances",
            width=1000,
            height=500
        )
        fig.show()
    
    def plot_param_relationships(self):
        """Plot parameter relationships."""
        if self.study is None:
            raise ValueError("No tuning has been performed yet. Call tune() first.")
        
        fig = plot_slice(self.study)
        fig.update_layout(
            title="Parameter Relationships",
            width=1200,
            height=800
        )
        fig.show()
    
    def save_best_params(self, filepath):
        """
        Save best parameters to JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to save JSON file
        """
        if self.best_params is None:
            raise ValueError("No tuning has been performed yet. Call tune() first.")
        
        params_to_save = self.get_best_params(include_fixed=True)
        
        with open(filepath, 'w') as f:
            json.dump(params_to_save, f, indent=4)
        
        if self.verbose:
            print(f"✓ Best parameters saved to {filepath}")
    
    def get_study_summary(self):
        """
        Get summary of optimization study.
        
        Returns:
        --------
        dict : Summary statistics
        """
        if self.study is None:
            raise ValueError("No tuning has been performed yet. Call tune() first.")
        
        trials_df = self.study.trials_dataframe()
        
        summary = {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial.number,
            'mean_value': trials_df['value'].mean(),
            'std_value': trials_df['value'].std(),
            'min_value': trials_df['value'].min(),
            'max_value': trials_df['value'].max(),
        }
        
        return summary


def quick_tune(X_train, y_train, X_val=None, y_val=None, 
               n_trials=50, n_folds=5, verbose=True):
    """
    Quick hyperparameter tuning with default settings.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation target
    n_trials : int, default=50
        Number of optimization trials
    n_folds : int, default=5
        Number of CV folds (if no validation set)
    verbose : bool, default=True
        Whether to print progress
        
    Returns:
    --------
    dict : Best parameters
    """
    tuner = LGBMTuner(
        X_train, y_train, X_val, y_val,
        n_folds=n_folds, verbose=verbose
    )
    
    best_params = tuner.tune(n_trials=n_trials)
    
    return best_params


def staged_tuning(X_train, y_train, X_val=None, y_val=None, 
                  n_trials_per_stage=30, verbose=True):
    """
    Staged hyperparameter tuning - tune parameters in groups.
    
    Stage 1: Learning rate and tree structure (num_leaves, max_depth)
    Stage 2: Sampling parameters (subsample, colsample_bytree, min_child_samples)
    Stage 3: Regularization (reg_alpha, reg_lambda)
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation target
    n_trials_per_stage : int, default=30
        Number of trials per stage
    verbose : bool, default=True
        Whether to print progress
        
    Returns:
    --------
    dict : Best parameters from all stages
    """
    if verbose:
        print("="*70)
        print("STAGED HYPERPARAMETER TUNING")
        print("="*70)
    
    best_params = {}
    
    # Stage 1: Learning rate and tree structure
    if verbose:
        print("\n" + "="*70)
        print("STAGE 1: Learning rate and tree structure")
        print("="*70)
    
    class Stage1Objective:
        def __init__(self, tuner):
            self.tuner = tuner
        
        def __call__(self, trial):
            fixed = {
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'min_child_samples': 50,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
            }
            
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 32, 512),
                'max_depth': trial.suggest_int('max_depth', 6, 15),
            }
            
            fixed.update(params)
            return self.tuner._objective(trial, fixed_params=fixed)
    
    tuner = LGBMTuner(X_train, y_train, X_val, y_val, verbose=False)
    study1 = optuna.create_study(direction='maximize')
    study1.optimize(Stage1Objective(tuner), n_trials=n_trials_per_stage, show_progress_bar=verbose)
    
    best_params.update(study1.best_params)
    
    if verbose:
        print(f"\nStage 1 Best AUC: {study1.best_value:.6f}")
        for param, value in study1.best_params.items():
            if isinstance(value, float):
                print(f"  • {param:25s} {value:.6f}")
            else:
                print(f"  • {param:25s} {value}")
    
    # Stage 2: Sampling parameters
    if verbose:
        print("\n" + "="*70)
        print("STAGE 2: Sampling parameters")
        print("="*70)
    
    class Stage2Objective:
        def __init__(self, tuner, stage1_params):
            self.tuner = tuner
            self.stage1_params = stage1_params
        
        def __call__(self, trial):
            fixed = {
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
            }
            fixed.update(self.stage1_params)
            
            params = {
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            }
            
            fixed.update(params)
            return self.tuner._objective(trial, fixed_params=fixed)
    
    study2 = optuna.create_study(direction='maximize')
    study2.optimize(Stage2Objective(tuner, best_params), n_trials=n_trials_per_stage, show_progress_bar=verbose)
    
    best_params.update(study2.best_params)
    
    if verbose:
        print(f"\nStage 2 Best AUC: {study2.best_value:.6f}")
        for param, value in study2.best_params.items():
            if isinstance(value, float):
                print(f"  • {param:25s} {value:.6f}")
            else:
                print(f"  • {param:25s} {value}")
    
    # Stage 3: Regularization
    if verbose:
        print("\n" + "="*70)
        print("STAGE 3: Regularization")
        print("="*70)
    
    class Stage3Objective:
        def __init__(self, tuner, prev_params):
            self.tuner = tuner
            self.prev_params = prev_params
        
        def __call__(self, trial):
            fixed = self.prev_params.copy()
            
            params = {
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            }
            
            fixed.update(params)
            return self.tuner._objective(trial, fixed_params=fixed)
    
    study3 = optuna.create_study(direction='maximize')
    study3.optimize(Stage3Objective(tuner, best_params), n_trials=n_trials_per_stage, show_progress_bar=verbose)
    
    best_params.update(study3.best_params)
    
    if verbose:
        print(f"\nStage 3 Best AUC: {study3.best_value:.6f}")
        for param, value in study3.best_params.items():
            if isinstance(value, float):
                print(f"  • {param:25s} {value:.6f}")
            else:
                print(f"  • {param:25s} {value}")
        
        print("\n" + "="*70)
        print("STAGED TUNING COMPLETE")
        print("="*70)
        print(f"\nFinal Best AUC: {study3.best_value:.6f}")
        print(f"\nFinal parameters:")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"  • {param:25s} {value:.6f}")
            else:
                print(f"  • {param:25s} {value}")
        print("="*70)
    
    return best_params


if __name__ == "__main__":
    print("Hyperparameter Tuning Module")
    print("="*70)
    print("\nUsage:")
    print("  from src.models.hyperparameter_tuning import LGBMTuner, quick_tune")
    print("\n  # Quick tuning")
    print("  best_params = quick_tune(X_train, y_train, X_val, y_val, n_trials=50)")
    print("\n  # Advanced tuning")
    print("  tuner = LGBMTuner(X_train, y_train, X_val, y_val)")
    print("  best_params = tuner.tune(n_trials=100)")
    print("  tuner.plot_optimization_history()")
    print("  tuner.save_best_params('best_params.json')")
