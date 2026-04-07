"""
Models module for F1 Race Predictor.
Contains XGBoost trainer and model utilities.
"""

from .trainer import F1ModelTrainer, train_model

__all__ = ['F1ModelTrainer', 'train_model']
