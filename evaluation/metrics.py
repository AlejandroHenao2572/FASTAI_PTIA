"""
Metrics module for F1 Race Predictor.

Implements essential evaluation metrics for race position prediction:
- Standard regression metrics (MAE, RMSE)
- F1-specific metrics (Podium accuracy)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RaceMetrics:
    """
    Container for essential evaluation metrics.
    
    Attributes:
        mae: Mean Absolute Error (positions)
        rmse: Root Mean Squared Error
        top3_accuracy: Accuracy in predicting podium finishers
    """
    mae: float = 0.0
    rmse: float = 0.0
    top3_accuracy: float = 0.0
    n_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'mae': round(self.mae, 4),
            'rmse': round(self.rmse, 4),
            'top3_accuracy': round(self.top3_accuracy, 4),
            'n_samples': self.n_samples,
        }
    
    def summary(self) -> str:
        """Generate a formatted summary string."""
        return f"""
========================================
       MODEL EVALUATION METRICS
========================================

Regression Metrics:
  - MAE:        {self.mae:.3f} positions
  - RMSE:       {self.rmse:.3f} positions

Top-N Accuracy:
  - Top 3 (Podium): {self.top3_accuracy:6.1%}

Samples Evaluated: {self.n_samples}
"""


def calculate_regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Args:
        y_true: Actual positions
        y_pred: Predicted positions
        
    Returns:
        Dictionary with MAE, RMSE, Median AE
    """
    errors = np.abs(y_true - y_pred)
    
    return {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'median_ae': np.median(errors),
    }


def calculate_top3_accuracy(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Calculate accuracy for top-3 (podium) predictions.
    
    Checks if drivers predicted in top 3 actually finished in top 3.
    
    Args:
        y_true: Actual positions
        y_pred: Predicted positions
        
    Returns:
        Top-3 accuracy as float
    """
    # Find drivers predicted to be in top 3
    pred_top3 = set(np.where(y_pred <= 3)[0])
    # Find drivers actually in top 3
    actual_top3 = set(np.where(y_true <= 3)[0])
    
    if len(pred_top3) == 0:
        return 0.0
    
    # Intersection over predicted top 3
    correct = len(pred_top3.intersection(actual_top3))
    accuracy = correct / len(pred_top3)
    
    return accuracy


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> RaceMetrics:
    """
    Comprehensive evaluation of race position predictions.
    
    Args:
        y_true: Actual race positions
        y_pred: Predicted race positions
        
    Returns:
        RaceMetrics object with essential metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Ensure same length
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    
    logger.info(f"Evaluating {len(y_true)} predictions")
    
    # Calculate metrics
    regression = calculate_regression_metrics(y_true, y_pred)
    top3_acc = calculate_top3_accuracy(y_true, y_pred)
    
    # Create metrics object
    metrics = RaceMetrics(
        mae=regression['mae'],
        rmse=regression['rmse'],
        top3_accuracy=top3_acc,
        n_samples=len(y_true),
    )
    
    return metrics


def print_evaluation_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> RaceMetrics:
    """
    Evaluate and print a formatted report.
    
    Args:
        y_true: Actual race positions
        y_pred: Predicted race positions
        
    Returns:
        RaceMetrics object
    """
    metrics = evaluate_predictions(y_true, y_pred)
    print(metrics.summary())
    return metrics
