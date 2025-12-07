"""
Feature Selection Module for LightGBM

This module provides various feature selection techniques to improve model
performance and reduce overfitting by removing redundant and low-importance features.

Author: IEEE Fraud Detection Team
Milestone: 12 - Feature Selection
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Feature selection for fraud detection models.
    
    Supports:
    - Correlation-based removal (multicollinearity)
    - Importance-based selection
    - Recursive feature elimination
    - Null importance selection
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print progress
        """
        self.verbose = verbose
        self.selected_features = None
        self.removed_features = {}
        self.feature_importance = None
        
    def remove_missing_features(self, df, threshold=0.9):
        """
        Remove features with high missing value percentage.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        threshold : float, default=0.9
            Remove features with missing % above this threshold
            
        Returns:
        --------
        list : Features to remove
        """
        missing_pct = df.isnull().mean()
        to_remove = missing_pct[missing_pct > threshold].index.tolist()
        
        if self.verbose and to_remove:
            print(f"Removing {len(to_remove)} features with >{threshold*100}% missing:")
            for feat in to_remove[:10]:
                print(f"  • {feat}: {missing_pct[feat]*100:.1f}% missing")
            if len(to_remove) > 10:
                print(f"  ... and {len(to_remove)-10} more")
        
        self.removed_features['missing'] = to_remove
        return to_remove
    
    def remove_single_value_features(self, df):
        """
        Remove features with only one unique value.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        list : Features to remove
        """
        to_remove = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                to_remove.append(col)
        
        if self.verbose and to_remove:
            print(f"\nRemoving {len(to_remove)} single-value features:")
            for feat in to_remove[:10]:
                print(f"  • {feat}")
            if len(to_remove) > 10:
                print(f"  ... and {len(to_remove)-10} more")
        
        self.removed_features['single_value'] = to_remove
        return to_remove
    
    def remove_correlated_features(self, df, threshold=0.95):
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        threshold : float, default=0.95
            Correlation threshold above which features are removed
            
        Returns:
        --------
        list : Features to remove
        """
        if self.verbose:
            print(f"\nFinding correlated features (threshold={threshold})...")
        
        # Calculate correlation matrix for numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_features].corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_remove = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if self.verbose:
            print(f"Found {len(to_remove)} highly correlated features to remove")
            if to_remove:
                print("\nSample correlated feature pairs:")
                count = 0
                for col in to_remove[:5]:
                    corr_partners = upper[col][upper[col] > threshold].index.tolist()
                    if corr_partners:
                        partner = corr_partners[0]
                        corr_val = upper.loc[partner, col]
                        print(f"  • {col} <-> {partner}: {corr_val:.3f}")
                        count += 1
                if len(to_remove) > 5:
                    print(f"  ... and {len(to_remove)-5} more pairs")
        
        self.removed_features['correlated'] = to_remove
        return to_remove
    
    def select_by_importance(self, X_train, y_train, X_val, y_val, 
                            threshold=0.95, params=None):
        """
        Select features based on cumulative importance.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        threshold : float, default=0.95
            Keep features that account for this % of total importance
        params : dict, optional
            LightGBM parameters
            
        Returns:
        --------
        tuple : (selected_features, removed_features)
        """
        if self.verbose:
            print(f"\nSelecting features by importance (threshold={threshold})...")
        
        # Default parameters
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'num_leaves': 256,
                'max_depth': 12,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        importance_df['cumulative_importance'] = (
            importance_df['importance'].cumsum() / importance_df['importance'].sum()
        )
        
        # Select features
        selected = importance_df[
            importance_df['cumulative_importance'] <= threshold
        ]['feature'].tolist()
        
        # Always keep at least top features if threshold is too restrictive
        if len(selected) < 50:
            selected = importance_df['feature'].head(50).tolist()
        
        removed = [f for f in X_train.columns if f not in selected]
        
        if self.verbose:
            print(f"Selected {len(selected)} features ({len(selected)/len(X_train.columns)*100:.1f}%)")
            print(f"Removed {len(removed)} low-importance features ({len(removed)/len(X_train.columns)*100:.1f}%)")
            print(f"\nTop 10 features by importance:")
            for i, row in importance_df.head(10).iterrows():
                print(f"  {i+1:2d}. {row['feature']:40s} {row['importance']:>10,.0f} ({row['cumulative_importance']*100:>5.1f}%)")
        
        self.feature_importance = importance_df
        self.removed_features['low_importance'] = removed
        return selected, removed
    
    def recursive_feature_elimination(self, X_train, y_train, X_val, y_val,
                                     n_features_to_select=None, step=0.1, params=None):
        """
        Recursive feature elimination - iteratively remove least important features.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        n_features_to_select : int, optional
            Number of features to keep (default: 50% of original)
        step : float, default=0.1
            Percentage of features to remove at each iteration
        params : dict, optional
            LightGBM parameters
            
        Returns:
        --------
        tuple : (selected_features, elimination_history)
        """
        if self.verbose:
            print("\nPerforming recursive feature elimination...")
        
        # Default parameters
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'num_leaves': 256,
                'max_depth': 12,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if n_features_to_select is None:
            n_features_to_select = len(X_train.columns) // 2
        
        current_features = X_train.columns.tolist()
        elimination_history = []
        
        while len(current_features) > n_features_to_select:
            # Train model
            X_tr = X_train[current_features]
            X_vl = X_val[current_features]
            
            train_data = lgb.Dataset(X_tr, label=y_train)
            val_data = lgb.Dataset(X_vl, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            
            # Evaluate
            y_pred = model.predict(X_vl)
            auc = roc_auc_score(y_val, y_pred)
            
            # Get importance
            importance = pd.DataFrame({
                'feature': current_features,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            # Remove bottom features
            n_to_remove = max(1, int(len(current_features) * step))
            n_to_remove = min(n_to_remove, len(current_features) - n_features_to_select)
            
            removed = importance.tail(n_to_remove)['feature'].tolist()
            current_features = [f for f in current_features if f not in removed]
            
            elimination_history.append({
                'n_features': len(current_features),
                'auc': auc,
                'removed': removed
            })
            
            if self.verbose:
                print(f"  Features: {len(current_features):4d} | AUC: {auc:.6f} | Removed: {len(removed)}")
        
        if self.verbose:
            print(f"\nRFE complete: {len(current_features)} features selected")
        
        return current_features, elimination_history
    
    def null_importance_selection(self, X_train, y_train, X_val, y_val,
                                  n_iterations=5, threshold=0.95, params=None):
        """
        Select features based on null importance (features that beat random permutations).
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        n_iterations : int, default=5
            Number of null importance iterations
        threshold : float, default=0.95
            Keep features with importance > threshold of null distributions
        params : dict, optional
            LightGBM parameters
            
        Returns:
        --------
        tuple : (selected_features, null_importance_scores)
        """
        if self.verbose:
            print(f"\nPerforming null importance selection ({n_iterations} iterations)...")
        
        # Default parameters
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'num_leaves': 256,
                'max_depth': 12,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Get actual importance
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        actual_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance(importance_type='gain')
        })
        
        # Get null importance (shuffled target)
        null_importance_list = []
        
        for i in range(n_iterations):
            if self.verbose:
                print(f"  Iteration {i+1}/{n_iterations}...")
            
            # Shuffle target
            y_shuffled = y_train.sample(frac=1, random_state=i).reset_index(drop=True)
            
            train_data = lgb.Dataset(X_train, label=y_shuffled)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            
            null_importance = pd.DataFrame({
                'feature': X_train.columns,
                f'importance_{i}': model.feature_importance(importance_type='gain')
            })
            
            null_importance_list.append(null_importance)
        
        # Combine null importance
        null_df = null_importance_list[0]
        for df in null_importance_list[1:]:
            null_df = null_df.merge(df, on='feature')
        
        # Calculate statistics
        importance_cols = [f'importance_{i}' for i in range(n_iterations)]
        null_df['null_mean'] = null_df[importance_cols].mean(axis=1)
        null_df['null_std'] = null_df[importance_cols].std(axis=1)
        null_df['null_max'] = null_df[importance_cols].max(axis=1)
        
        # Merge with actual importance
        comparison = actual_importance.merge(null_df[['feature', 'null_mean', 'null_std', 'null_max']], 
                                            on='feature')
        comparison['importance_ratio'] = comparison['importance'] / (comparison['null_mean'] + 1)
        comparison['zscore'] = (comparison['importance'] - comparison['null_mean']) / (comparison['null_std'] + 1)
        
        # Select features that beat null importance
        selected = comparison[
            (comparison['importance'] > comparison['null_max']) |
            (comparison['importance_ratio'] > 2.0) |
            (comparison['zscore'] > 2.0)
        ]['feature'].tolist()
        
        if self.verbose:
            print(f"\nSelected {len(selected)} features ({len(selected)/len(X_train.columns)*100:.1f}%)")
            print(f"Removed {len(X_train.columns)-len(selected)} features that don't beat null importance")
        
        return selected, comparison
    
    def select_features(self, X_train, y_train, X_val, y_val,
                       methods=['correlation', 'importance'],
                       correlation_threshold=0.95,
                       importance_threshold=0.95,
                       params=None):
        """
        Apply multiple feature selection methods sequentially.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        methods : list, default=['correlation', 'importance']
            Feature selection methods to apply
        correlation_threshold : float, default=0.95
            Correlation threshold
        importance_threshold : float, default=0.95
            Importance threshold
        params : dict, optional
            LightGBM parameters
            
        Returns:
        --------
        list : Selected features
        """
        if self.verbose:
            print("="*70)
            print("FEATURE SELECTION")
            print("="*70)
            print(f"\nStarting features: {len(X_train.columns)}")
        
        X_tr = X_train.copy()
        X_vl = X_val.copy()
        
        # Remove missing features
        if 'missing' in methods:
            to_remove = self.remove_missing_features(X_tr, threshold=0.95)
            X_tr = X_tr.drop(columns=to_remove, errors='ignore')
            X_vl = X_vl.drop(columns=to_remove, errors='ignore')
        
        # Remove single-value features
        if 'single_value' in methods:
            to_remove = self.remove_single_value_features(X_tr)
            X_tr = X_tr.drop(columns=to_remove, errors='ignore')
            X_vl = X_vl.drop(columns=to_remove, errors='ignore')
        
        # Remove correlated features
        if 'correlation' in methods:
            to_remove = self.remove_correlated_features(X_tr, threshold=correlation_threshold)
            X_tr = X_tr.drop(columns=to_remove, errors='ignore')
            X_vl = X_vl.drop(columns=to_remove, errors='ignore')
        
        # Importance-based selection
        if 'importance' in methods:
            selected, _ = self.select_by_importance(
                X_tr, y_train, X_vl, y_val,
                threshold=importance_threshold,
                params=params
            )
            X_tr = X_tr[selected]
            X_vl = X_vl[selected]
        
        self.selected_features = X_tr.columns.tolist()
        
        if self.verbose:
            print("\n" + "="*70)
            print(f"FEATURE SELECTION COMPLETE")
            print("="*70)
            print(f"Final features: {len(self.selected_features)}")
            print(f"Removed: {len(X_train.columns) - len(self.selected_features)} ({(1-len(self.selected_features)/len(X_train.columns))*100:.1f}%)")
            print("="*70)
        
        return self.selected_features
    
    def get_summary(self):
        """Get summary of feature selection."""
        summary = {
            'total_selected': len(self.selected_features) if self.selected_features else 0,
            'removed_by_method': {k: len(v) for k, v in self.removed_features.items()}
        }
        return summary


def quick_feature_selection(X_train, y_train, X_val, y_val,
                           correlation_threshold=0.95,
                           importance_threshold=0.95,
                           verbose=True):
    """
    Quick feature selection with default settings.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    correlation_threshold : float, default=0.95
        Correlation threshold
    importance_threshold : float, default=0.95
        Importance threshold
    verbose : bool, default=True
        Whether to print progress
        
    Returns:
    --------
    list : Selected features
    """
    selector = FeatureSelector(verbose=verbose)
    
    selected = selector.select_features(
        X_train, y_train, X_val, y_val,
        methods=['single_value', 'correlation', 'importance'],
        correlation_threshold=correlation_threshold,
        importance_threshold=importance_threshold
    )
    
    return selected


if __name__ == "__main__":
    print("Feature Selection Module")
    print("="*70)
    print("\nUsage:")
    print("  from src.models.feature_selection import FeatureSelector, quick_feature_selection")
    print("\n  # Quick selection")
    print("  selected = quick_feature_selection(X_train, y_train, X_val, y_val)")
    print("\n  # Advanced selection")
    print("  selector = FeatureSelector()")
    print("  selected = selector.select_features(X_train, y_train, X_val, y_val)")
