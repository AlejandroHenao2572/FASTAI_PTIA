"""
Centralized configuration settings for F1 Race Predictor.
All hyperparameters and configuration values are defined here.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# XGBoost hyperparameters
# Justification:
# - n_estimators=150: Balance between performance and overfitting
# - max_depth=4: Prevents overly complex trees
# - learning_rate=0.05: Conservative learning for better generalization
# - subsample/colsample=0.8: Adds randomness to reduce overfitting
# - min_child_weight=5: Prevents splits on very small samples
# - reg_alpha/lambda: L1/L2 regularization for smoother predictions
XGBOOST_PARAMS: Dict[str, Any] = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'n_estimators': 150,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

# Monotone constraints for logical relationships
# 1 = increasing (higher feature value -> higher predicted position)
# -1 = decreasing (higher feature value -> lower predicted position)
# 0 = no constraint
MONOTONE_CONSTRAINTS = {
    'quali_position': 1,           # Worse quali -> worse race result
    'grid_position': 1,            # Worse grid -> worse race result
    'quali_gap_to_pole': 1,        # Bigger gap to pole -> worse result
    'driver_dnf_rate': 1,          # Higher DNF rate -> worse expected result
    'team_avg_position_season': 1, # Worse team avg -> worse result
    'constructor_standing': 1,     # Worse standing -> worse result
}


@dataclass
class Settings:
    """
    Main configuration class for the F1 Predictor.
    All configurable parameters are centralized here.
    """
    
    # Data settings
    training_seasons: List[int] = field(default_factory=lambda: [2023, 2024])
    test_season: int = 2024
    test_races_count: int = 4  # Last N races of test_season for final evaluation
    
    # Feature engineering settings
    min_races_for_history: int = 3  # Minimum races to calculate historical features
    default_position: float = 15.0  # Default for missing historical data
    lookback_races: int = 5  # Races to consider for recent form
    
    # Model settings
    model_params: Dict[str, Any] = field(default_factory=lambda: XGBOOST_PARAMS.copy())
    cv_splits: int = 5  # Time-series cross-validation splits
    
    # Paths
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "saved")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    figures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "figures")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "reports")
    
    # Target variable
    target_column: str = 'finish_position'
    
    # Feature columns (defined after feature engineering)
    feature_columns: List[str] = field(default_factory=lambda: [
        # Qualifying features
        'quali_position',
        'quali_gap_to_pole',
        'quali_gap_to_teammate',
        'made_q3',
        # Historical driver features
        'driver_avg_position_last_5',
        'driver_circuit_avg_position',
        'driver_dnf_rate',
        'driver_experience',
        # Team features
        'team_avg_position_season',
        'team_reliability_rate',
        'constructor_standing',
        # Circuit features
        'circuit_type',
        'circuit_length_km',
        'overtaking_difficulty',
        'number_of_laps',
        # Grid and conditions
        'grid_position',
        'is_wet_session',
        'temperature',
    ])
    
    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
