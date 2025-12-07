"""
Ensemble Modeling Module

This module provides advanced ensemble techniques for combining multiple models
to improve prediction accuracy and robustness.

Author: IEEE Fraud Detection Team
Milestone: 13 - Advanced Modeling (Ensembles)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")


class ModelEnsemble:
    """
    Ensemble modeling with stacking, blending, and weighted averaging.
    
    Supports:
    - Multiple base models (LightGBM, XGBoost, CatBoost)
    - Stacking with cross-validation
    - Simple weighted averaging
    - Rank averaging
    - Meta-learner optimization
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the ensemble.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print progress
        """
        self.verbose = verbose
        self.base_models = []
        self.meta_model = None
        self.weights = None
        self.oof_predictions = None
        self.test_predictions = None
        
    def add_lightgbm(self, params=None, name='LightGBM'):
        """
        Add LightGBM model to ensemble.
        
        Parameters:
        -----------
        params : dict, optional
            LightGBM parameters
        name : str, default='LightGBM'
            Model name
        """
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'num_leaves': 256,
                'max_depth': 12,
                'min_child_samples': 50,
                'subsample': 0.9,
                'subsample_freq': 1,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.base_models.append({
            'name': name,
            'type': 'lightgbm',
            'params': params,
            'models': []
        })
        
        if self.verbose:
            print(f"✓ Added {name} to ensemble")
    
    def add_xgboost(self, params=None, name='XGBoost'):
        """
        Add XGBoost model to ensemble.
        
        Parameters:
        -----------
        params : dict, optional
            XGBoost parameters
        name : str, default='XGBoost'
            Model name
        """
        if not XGBOOST_AVAILABLE:
            print(f"Warning: Cannot add {name} - XGBoost not installed")
            return
        
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'learning_rate': 0.01,
                'max_depth': 12,
                'min_child_weight': 50,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.base_models.append({
            'name': name,
            'type': 'xgboost',
            'params': params,
            'models': []
        })
        
        if self.verbose:
            print(f"✓ Added {name} to ensemble")
    
    def add_catboost(self, params=None, name='CatBoost'):
        """
        Add CatBoost model to ensemble.
        
        Parameters:
        -----------
        params : dict, optional
            CatBoost parameters
        name : str, default='CatBoost'
            Model name
        """
        if not CATBOOST_AVAILABLE:
            print(f"Warning: Cannot add {name} - CatBoost not installed")
            return
        
        if params is None:
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'learning_rate': 0.01,
                'depth': 8,
                'l2_leaf_reg': 3,
                'subsample': 0.9,
                'random_state': 42,
                'verbose': False,
                'thread_count': -1
            }
        
        self.base_models.append({
            'name': name,
            'type': 'catboost',
            'params': params,
            'models': []
        })
        
        if self.verbose:
            print(f"✓ Added {name} to ensemble")
    
    def fit_stacking(self, X_train, y_train, n_folds=5, num_boost_round=2000,
                     early_stopping_rounds=100, stratified=True):
        """
        Fit ensemble using stacking with cross-validation.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        n_folds : int, default=5
            Number of CV folds
        num_boost_round : int, default=2000
            Number of boosting rounds
        early_stopping_rounds : int, default=100
            Early stopping rounds
        stratified : bool, default=True
            Use stratified k-fold
            
        Returns:
        --------
        dict : OOF predictions and scores
        """
        if self.verbose:
            print("="*70)
            print("STACKING ENSEMBLE")
            print("="*70)
            print(f"\nTraining {len(self.base_models)} base models with {n_folds}-fold CV...")
        
        # Initialize OOF predictions
        oof_predictions = np.zeros((len(X_train), len(self.base_models)))
        
        # K-Fold cross-validation
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Train each base model
        for model_idx, model_config in enumerate(self.base_models):
            model_name = model_config['name']
            model_type = model_config['type']
            
            if self.verbose:
                print(f"\n{'-'*70}")
                print(f"Training {model_name}")
                print(f"{'-'*70}")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                if self.verbose:
                    print(f"  Fold {fold+1}/{n_folds}...", end=' ')
                
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]
                
                # Train model based on type
                if model_type == 'lightgbm':
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    
                    model = lgb.train(
                        model_config['params'],
                        train_data,
                        num_boost_round=num_boost_round,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                    )
                    
                    y_pred = model.predict(X_val)
                
                elif model_type == 'xgboost':
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    
                    model = xgb.train(
                        model_config['params'],
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=[(dval, 'eval')],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=False
                    )
                    
                    y_pred = model.predict(dval)
                
                elif model_type == 'catboost':
                    model = cb.CatBoostClassifier(**model_config['params'])
                    model.fit(
                        X_tr, y_tr,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                    
                    y_pred = model.predict_proba(X_val)[:, 1]
                
                # Store predictions
                oof_predictions[val_idx, model_idx] = y_pred
                
                # Calculate score
                score = roc_auc_score(y_val, y_pred)
                fold_scores.append(score)
                
                # Store model
                model_config['models'].append(model)
                
                if self.verbose:
                    print(f"AUC: {score:.6f}")
            
            # Calculate average score
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            if self.verbose:
                print(f"  {model_name} CV Score: {avg_score:.6f} ± {std_score:.6f}")
        
        # Store OOF predictions
        self.oof_predictions = oof_predictions
        
        # Calculate ensemble OOF score (simple average)
        ensemble_pred = oof_predictions.mean(axis=1)
        ensemble_score = roc_auc_score(y_train, ensemble_pred)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"ENSEMBLE OOF SCORE: {ensemble_score:.6f}")
            print(f"{'='*70}")
        
        return {
            'oof_predictions': oof_predictions,
            'ensemble_score': ensemble_score
        }
    
    def fit_meta_learner(self, y_train, meta_model=None):
        """
        Fit meta-learner on OOF predictions.
        
        Parameters:
        -----------
        y_train : pd.Series
            Training target
        meta_model : object, optional
            Meta-learner model (default: LogisticRegression)
        """
        if self.oof_predictions is None:
            raise ValueError("Must call fit_stacking first to generate OOF predictions")
        
        if meta_model is None:
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        if self.verbose:
            print("\nTraining meta-learner...")
        
        # Fit meta-learner
        meta_model.fit(self.oof_predictions, y_train)
        
        # Calculate meta score
        meta_pred = meta_model.predict_proba(self.oof_predictions)[:, 1]
        meta_score = roc_auc_score(y_train, meta_pred)
        
        self.meta_model = meta_model
        
        if self.verbose:
            print(f"✓ Meta-learner trained")
            print(f"  Meta-learner score: {meta_score:.6f}")
        
        return meta_score
    
    def predict(self, X_test, use_meta_learner=False):
        """
        Generate predictions on test set.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        use_meta_learner : bool, default=False
            Use meta-learner for final predictions
            
        Returns:
        --------
        np.ndarray : Predictions
        """
        if len(self.base_models[0]['models']) == 0:
            raise ValueError("Models not trained. Call fit_stacking first.")
        
        n_folds = len(self.base_models[0]['models'])
        predictions = np.zeros((len(X_test), len(self.base_models)))
        
        # Get predictions from each base model
        for model_idx, model_config in enumerate(self.base_models):
            model_type = model_config['type']
            fold_predictions = []
            
            # Average predictions across folds
            for model in model_config['models']:
                if model_type == 'lightgbm':
                    pred = model.predict(X_test)
                elif model_type == 'xgboost':
                    dtest = xgb.DMatrix(X_test)
                    pred = model.predict(dtest)
                elif model_type == 'catboost':
                    pred = model.predict_proba(X_test)[:, 1]
                
                fold_predictions.append(pred)
            
            # Average across folds
            predictions[:, model_idx] = np.mean(fold_predictions, axis=0)
        
        # Store base predictions
        self.test_predictions = predictions
        
        # Final prediction
        if use_meta_learner and self.meta_model is not None:
            final_pred = self.meta_model.predict_proba(predictions)[:, 1]
        else:
            # Simple average
            final_pred = predictions.mean(axis=1)
        
        return final_pred
    
    def optimize_weights(self, y_train, method='grid'):
        """
        Optimize ensemble weights on OOF predictions.
        
        Parameters:
        -----------
        y_train : pd.Series
            Training target
        method : str, default='grid'
            Optimization method ('grid' or 'scipy')
            
        Returns:
        --------
        np.ndarray : Optimal weights
        """
        if self.oof_predictions is None:
            raise ValueError("Must call fit_stacking first")
        
        if self.verbose:
            print("\nOptimizing ensemble weights...")
        
        if method == 'grid':
            # Grid search over weights
            best_score = 0
            best_weights = None
            
            # Generate weight combinations
            weight_steps = 11  # 0.0, 0.1, 0.2, ..., 1.0
            
            if len(self.base_models) == 2:
                for w1 in np.linspace(0, 1, weight_steps):
                    weights = np.array([w1, 1-w1])
                    pred = (self.oof_predictions * weights).sum(axis=1)
                    score = roc_auc_score(y_train, pred)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weights
            
            elif len(self.base_models) == 3:
                for w1 in np.linspace(0, 1, weight_steps):
                    for w2 in np.linspace(0, 1-w1, weight_steps):
                        w3 = 1 - w1 - w2
                        weights = np.array([w1, w2, w3])
                        pred = (self.oof_predictions * weights).sum(axis=1)
                        score = roc_auc_score(y_train, pred)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = weights
            
            else:
                # For more models, use equal weights
                best_weights = np.ones(len(self.base_models)) / len(self.base_models)
                pred = (self.oof_predictions * best_weights).sum(axis=1)
                best_score = roc_auc_score(y_train, pred)
        
        self.weights = best_weights
        
        if self.verbose:
            print(f"✓ Optimal weights found")
            print(f"  Weights: {[f'{w:.3f}' for w in best_weights]}")
            print(f"  Weighted score: {best_score:.6f}")
        
        return best_weights
    
    def predict_weighted(self, X_test):
        """
        Generate weighted predictions.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
            
        Returns:
        --------
        np.ndarray : Weighted predictions
        """
        if self.weights is None:
            raise ValueError("Must call optimize_weights first")
        
        if self.test_predictions is None:
            self.predict(X_test)
        
        weighted_pred = (self.test_predictions * self.weights).sum(axis=1)
        return weighted_pred
    
    def get_summary(self):
        """Get ensemble summary."""
        summary = {
            'n_models': len(self.base_models),
            'models': [m['name'] for m in self.base_models],
            'has_meta_learner': self.meta_model is not None,
            'has_weights': self.weights is not None
        }
        
        if self.weights is not None:
            for i, model_config in enumerate(self.base_models):
                summary[f'{model_config["name"]}_weight'] = float(self.weights[i])
        
        return summary


def simple_blend(predictions_dict, weights=None):
    """
    Simple blending of predictions.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary of predictions {name: predictions}
    weights : list, optional
        Weights for each model (default: equal weights)
        
    Returns:
    --------
    np.ndarray : Blended predictions
    """
    predictions = np.column_stack(list(predictions_dict.values()))
    
    if weights is None:
        weights = np.ones(len(predictions_dict)) / len(predictions_dict)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()
    
    blended = (predictions * weights).sum(axis=1)
    return blended


def rank_average(predictions_dict):
    """
    Rank averaging of predictions.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary of predictions {name: predictions}
        
    Returns:
    --------
    np.ndarray : Rank-averaged predictions
    """
    from scipy.stats import rankdata
    
    ranks = []
    for pred in predictions_dict.values():
        ranks.append(rankdata(pred))
    
    avg_rank = np.mean(ranks, axis=0)
    return avg_rank / len(avg_rank)


if __name__ == "__main__":
    print("Ensemble Modeling Module")
    print("="*70)
    print("\nUsage:")
    print("  from src.models.ensemble import ModelEnsemble")
    print("\n  # Create ensemble")
    print("  ensemble = ModelEnsemble()")
    print("  ensemble.add_lightgbm(params)")
    print("  ensemble.add_xgboost(params)")
    print("  ensemble.add_catboost(params)")
    print("\n  # Fit with stacking")
    print("  ensemble.fit_stacking(X_train, y_train, n_folds=5)")
    print("  ensemble.fit_meta_learner(y_train)")
    print("\n  # Predict")
    print("  predictions = ensemble.predict(X_test, use_meta_learner=True)")
