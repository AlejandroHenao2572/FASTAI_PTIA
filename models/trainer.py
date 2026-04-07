"""
Model trainer module for F1 Race Predictor.

Implements XGBoost regression for predicting F1 race positions.
Includes time-series cross-validation and model persistence.

Key features:
- XGBoost with configurable hyperparameters
- Monotone constraints for logical relationships
- Time-series cross-validation
- Model saving/loading with metadata
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

from config.settings import Settings, XGBOOST_PARAMS, MONOTONE_CONSTRAINTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1ModelTrainer:
    """
    Trains and manages XGBoost models for F1 race prediction.
    
    Features:
    - Configurable hyperparameters
    - Monotone constraints for domain knowledge
    - Time-series cross-validation
    - Model persistence with metadata
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the model trainer.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        self.model: Optional[XGBRegressor] = None
        self.feature_columns: List[str] = self.settings.feature_columns
        self.cv_results: Dict[str, Any] = {}
        self.training_metadata: Dict[str, Any] = {}
        
    def _create_model(self) -> XGBRegressor:
        """
        Create an XGBoost model with configured parameters.
        
        Returns:
            Configured XGBRegressor instance
        """
        params = self.settings.model_params.copy()
        
        # Build monotone constraints tuple based on feature order
        constraints = []
        for col in self.feature_columns:
            constraint = MONOTONE_CONSTRAINTS.get(col, 0)
            constraints.append(constraint)
        
        # Only add monotone_constraints if any are non-zero
        if any(c != 0 for c in constraints):
            params['monotone_constraints'] = tuple(constraints)
        
        return XGBRegressor(**params)
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> XGBRegressor:
        """
        Train the XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            eval_set: Optional validation set (X_val, y_val)
            
        Returns:
            Trained XGBRegressor
        """
        logger.info(f"Training model with {len(X)} samples and {len(self.feature_columns)} features")
        
        self.model = self._create_model()
        
        # Prepare evaluation set if provided
        fit_params = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
        
        # Train the model
        self.model.fit(X, y, **fit_params)
        
        # Store training metadata
        self.training_metadata = {
            'train_samples': len(X),
            'features': self.feature_columns,
            'trained_at': datetime.now().isoformat(),
            'params': self.settings.model_params,
        }
        
        logger.info("Model training complete")
        return self.model
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.
        
        Uses TimeSeriesSplit to ensure temporal ordering is preserved.
        Training always uses past data, validation uses future data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Starting {n_splits}-fold time-series cross-validation")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Calculate metrics (MAE, RMSE only)
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'mae': mae,
                'rmse': rmse,
            })
            
            logger.info(f"Fold {fold + 1}: MAE={mae:.3f}, RMSE={rmse:.3f}")
        
        # Aggregate results
        self.cv_results = {
            'n_splits': n_splits,
            'folds': fold_results,
            'mean_mae': np.mean([f['mae'] for f in fold_results]),
            'std_mae': np.std([f['mae'] for f in fold_results]),
            'mean_rmse': np.mean([f['rmse'] for f in fold_results]),
            'std_rmse': np.std([f['rmse'] for f in fold_results]),
        }
        
        logger.info(f"CV Complete - Mean MAE: {self.cv_results['mean_mae']:.3f} ± {self.cv_results['std_mae']:.3f}")
        
        return self.cv_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted positions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        
        # Clip predictions to valid position range [1, 20]
        predictions = np.clip(predictions, 1, 20)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (uses default if None)
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.settings.models_dir / f"xgboost_model_{timestamp}.pkl"
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'training_metadata': self.training_metadata,
            'cv_results': self.cv_results,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: Path) -> XGBRegressor:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded XGBRegressor
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.training_metadata = model_data.get('training_metadata', {})
        self.cv_results = model_data.get('cv_results', {})
        
        logger.info(f"Model loaded from: {filepath}")
        return self.model


def train_model(
    X: pd.DataFrame, 
    y: pd.Series,
    settings: Optional[Settings] = None,
    run_cv: bool = True
) -> Tuple[F1ModelTrainer, Dict[str, Any]]:
    """
    Convenience function to train a model with cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        settings: Configuration settings
        run_cv: Whether to run cross-validation
        
    Returns:
        Tuple of (trainer instance, CV results dict)
    """
    trainer = F1ModelTrainer(settings)
    
    cv_results = {}
    if run_cv:
        cv_results = trainer.cross_validate(X, y)
    
    trainer.train(X, y)
    
    return trainer, cv_results
