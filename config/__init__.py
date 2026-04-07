"""
Configuration module for F1 Race Predictor.
Contains settings and circuit metadata.
"""

from .settings import Settings, XGBOOST_PARAMS
from .circuits import CIRCUITS, get_circuit_info

__all__ = ['Settings', 'XGBOOST_PARAMS', 'CIRCUITS', 'get_circuit_info']
