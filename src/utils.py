"""
Utility functions for fraud detection project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of DataFrame by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        verbose: Whether to print memory reduction info
        
    Returns:
        DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "ROC Curve") -> None:
    """
    Plot ROC curve and display AUC score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = False) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance: Dict[str, float], top_n: int = 20, title: str = "Feature Importance") -> None:
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary of {feature_name: importance_score}
        top_n: Number of top features to display
        title: Plot title
    """
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), importances, color='steelblue', edgecolor='black', alpha=0.8)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def evaluate_model(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'AUC': auc_score,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn
    }
    
    return metrics


def print_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print("="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    print(f"\nROC-AUC Score:     {metrics['AUC']:.6f}")
    print(f"Accuracy:          {metrics['Accuracy']:.4f}")
    print(f"Precision:         {metrics['Precision']:.4f}")
    print(f"Recall:            {metrics['Recall']:.4f}")
    print(f"F1-Score:          {metrics['F1-Score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['True Positives']:,}")
    print(f"  False Positives: {metrics['False Positives']:,}")
    print(f"  True Negatives:  {metrics['True Negatives']:,}")
    print(f"  False Negatives: {metrics['False Negatives']:,}")
    print("="*60)


def create_submission_file(test_ids: np.ndarray, predictions: np.ndarray, filename: str = 'submission.csv') -> None:
    """
    Create submission file for Kaggle.
    
    Args:
        test_ids: Array of TransactionIDs
        predictions: Array of predicted probabilities
        filename: Output filename
    """
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"✓ Submission file saved: {filename}")
    print(f"  Shape: {submission.shape}")
    print(f"  Mean prediction: {predictions.mean():.6f}")


if __name__ == "__main__":
    print("Utilities module loaded successfully!")
