"""
Feature engineering module for F1 Race Predictor.

Creates features from raw race data for the ML model.
Implements strict temporal ordering to prevent data leakage.

Features created (18 total):
- Qualifying: quali_position, quali_gap_to_pole, quali_gap_to_teammate, made_q3
- Historical driver: driver_avg_position_last_5, driver_circuit_avg_position, 
                    driver_dnf_rate, driver_experience
- Team: team_avg_position_season, team_reliability_rate, constructor_standing
- Circuit: circuit_type, circuit_length_km, overtaking_difficulty, number_of_laps
- Grid/Conditions: grid_position, is_wet_session, temperature
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features for F1 race prediction.
    
    Key principles:
    1. Temporal ordering: Historical features use ONLY past data
    2. No data leakage: Race results not used in features
    3. Handles missing data with sensible defaults
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the feature engineer.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        self.default_position = self.settings.default_position
        self.min_races = self.settings.min_races_for_history
        self.lookback = self.settings.lookback_races
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw race data.
        
        Args:
            df: Raw DataFrame from data_loader
            
        Returns:
            DataFrame with all features added
        """
        logger.info("Starting feature engineering...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure proper sorting for temporal features
        df = df.sort_values(['year', 'round']).reset_index(drop=True)
        
        # Create each feature group
        df = self._create_qualifying_features(df)
        df = self._create_driver_historical_features(df)
        df = self._create_team_features(df)
        df = self._create_circuit_features(df)
        df = self._create_grid_features(df)
        
        # Fill any remaining NaN values
        df = self._handle_missing_values(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def _create_qualifying_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create qualifying-related features.
        
        Features:
        - quali_position: Qualifying position (already exists)
        - quali_gap_to_pole: Gap to pole in seconds
        - quali_gap_to_teammate: Performance vs teammate
        - made_q3: Whether driver made it to Q3
        """
        logger.info("Creating qualifying features...")
        
        # quali_position should already exist
        if 'quali_position' not in df.columns:
            df['quali_position'] = df['grid_position']
        
        # Fill missing quali_gap_to_pole
        if 'quali_gap_to_pole' not in df.columns:
            df['quali_gap_to_pole'] = 0.0
        df['quali_gap_to_pole'] = df['quali_gap_to_pole'].fillna(
            (df['quali_position'] - 1) * 0.3  # Approximate 0.3s per position
        )
        
        # made_q3 should already exist
        if 'made_q3' not in df.columns:
            df['made_q3'] = (df['quali_position'] <= 10).astype(int)
        
        # Calculate gap to teammate
        df['quali_gap_to_teammate'] = df.apply(
            lambda row: self._calc_teammate_gap(df, row), axis=1
        )
        
        return df
    
    def _calc_teammate_gap(self, df: pd.DataFrame, row: pd.Series) -> float:
        """Calculate qualifying gap to teammate."""
        same_race = df[
            (df['year'] == row['year']) & 
            (df['round'] == row['round']) &
            (df['team'] == row['team']) &
            (df['driver_code'] != row['driver_code'])
        ]
        
        if same_race.empty:
            return 0.0
        
        teammate_quali = same_race['quali_position'].values[0]
        return row['quali_position'] - teammate_quali
    
    def _create_driver_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create driver historical features using ONLY past data.
        
        Features:
        - driver_avg_position_last_5: Average finish in last 5 races
        - driver_circuit_avg_position: Average at this circuit
        - driver_dnf_rate: Percentage of DNFs
        - driver_experience: Number of races completed
        """
        logger.info("Creating driver historical features...")
        
        # Initialize columns
        df['driver_avg_position_last_5'] = self.default_position
        df['driver_circuit_avg_position'] = self.default_position
        df['driver_dnf_rate'] = 0.1  # Default 10% DNF rate
        df['driver_experience'] = 0
        
        # Process each row with only past data
        for idx, row in df.iterrows():
            driver = row['driver_code']
            year = row['year']
            round_num = row['round']
            circuit = row.get('circuit_key', '')
            
            # Get all previous races for this driver
            past_races = df[
                (df['driver_code'] == driver) &
                ((df['year'] < year) | 
                 ((df['year'] == year) & (df['round'] < round_num)))
            ]
            
            if len(past_races) >= self.min_races:
                # Average position in last N races
                recent = past_races.tail(self.lookback)
                df.at[idx, 'driver_avg_position_last_5'] = recent['finish_position'].mean()
                
                # DNF rate (non-finished races)
                total_races = len(past_races)
                dnf_count = past_races['status'].apply(
                    lambda x: 0 if 'Finished' in str(x) or '+' in str(x) else 1
                ).sum()
                df.at[idx, 'driver_dnf_rate'] = dnf_count / total_races
                
                # Experience
                df.at[idx, 'driver_experience'] = total_races
                
                # Circuit-specific average
                if circuit:
                    circuit_races = past_races[past_races['circuit_key'] == circuit]
                    if len(circuit_races) >= 1:
                        df.at[idx, 'driver_circuit_avg_position'] = circuit_races['finish_position'].mean()
                    else:
                        df.at[idx, 'driver_circuit_avg_position'] = df.at[idx, 'driver_avg_position_last_5']
        
        return df
    
    def _create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team-related features using past data.
        
        Features:
        - team_avg_position_season: Team's average finishing position this season
        - team_reliability_rate: Team's reliability (% of finishes)
        - constructor_standing: Position in constructors championship
        """
        logger.info("Creating team features...")
        
        # Initialize columns
        df['team_avg_position_season'] = self.default_position
        df['team_reliability_rate'] = 0.9  # Default 90% reliability
        df['constructor_standing'] = 5  # Default mid-pack
        
        for idx, row in df.iterrows():
            team = row['team']
            year = row['year']
            round_num = row['round']
            
            # Get team's past races this season
            team_races = df[
                (df['team'] == team) &
                (df['year'] == year) &
                (df['round'] < round_num)
            ]
            
            if len(team_races) >= 2:  # Need at least 2 results (2 drivers * 1 race)
                # Team average position
                df.at[idx, 'team_avg_position_season'] = team_races['finish_position'].mean()
                
                # Team reliability
                finishes = team_races['status'].apply(
                    lambda x: 1 if 'Finished' in str(x) or '+' in str(x) else 0
                ).sum()
                df.at[idx, 'team_reliability_rate'] = finishes / len(team_races)
            
            # Constructor standing (based on points)
            team_points = df[
                (df['year'] == year) &
                (df['round'] < round_num)
            ].groupby('team')['points'].sum()
            
            if len(team_points) > 0 and team in team_points.index:
                standing = team_points.rank(ascending=False).get(team, 5)
                df.at[idx, 'constructor_standing'] = int(standing)
        
        return df
    
    def _create_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure circuit features exist and are properly typed.
        
        Features (should already exist from data_loader):
        - circuit_type: Street (1), Permanent (2), Hybrid (3)
        - circuit_length_km: Length in kilometers
        - overtaking_difficulty: 1 (easy) to 5 (hard)
        - number_of_laps: Race laps
        """
        logger.info("Validating circuit features...")
        
        # Set defaults if missing
        if 'circuit_type' not in df.columns:
            df['circuit_type'] = 2  # Permanent
        if 'circuit_length_km' not in df.columns:
            df['circuit_length_km'] = 5.0
        if 'overtaking_difficulty' not in df.columns:
            df['overtaking_difficulty'] = 3
        if 'number_of_laps' not in df.columns:
            df['number_of_laps'] = 55
        
        return df
    
    def _create_grid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create grid and condition features.
        
        Features:
        - grid_position: Starting position (already exists)
        - is_wet_session: Whether rain is expected
        - temperature: Air temperature
        """
        logger.info("Creating grid and condition features...")
        
        # grid_position should already exist
        if 'grid_position' not in df.columns:
            df['grid_position'] = df.get('quali_position', self.default_position)
        
        # Handle pit lane starts (position 0)
        df.loc[df['grid_position'] == 0, 'grid_position'] = 20
        
        # is_wet_session should already exist
        if 'is_wet_session' not in df.columns:
            df['is_wet_session'] = 0
        
        # temperature should already exist
        if 'temperature' not in df.columns:
            df['temperature'] = 25.0
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle any remaining missing values in features.
        """
        logger.info("Handling missing values...")
        
        # Define defaults for each feature
        defaults = {
            'quali_position': self.default_position,
            'quali_gap_to_pole': 1.0,
            'quali_gap_to_teammate': 0.0,
            'made_q3': 0,
            'driver_avg_position_last_5': self.default_position,
            'driver_circuit_avg_position': self.default_position,
            'driver_dnf_rate': 0.1,
            'driver_experience': 0,
            'team_avg_position_season': self.default_position,
            'team_reliability_rate': 0.9,
            'constructor_standing': 5,
            'circuit_type': 2,
            'circuit_length_km': 5.0,
            'overtaking_difficulty': 3,
            'number_of_laps': 55,
            'grid_position': self.default_position,
            'is_wet_session': 0,
            'temperature': 25.0,
        }
        
        for col, default in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return the list of feature columns used for training."""
        return self.settings.feature_columns.copy()
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and target (y) for training.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        feature_cols = self.get_feature_columns()
        
        # Verify all columns exist
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            for col in missing_cols:
                df[col] = self.default_position
        
        X = df[feature_cols].copy()
        y = df[self.settings.target_column].copy()
        
        return X, y


def create_features(df: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
    """
    Convenience function to create all features.
    
    Args:
        df: Raw DataFrame from data_loader
        settings: Configuration settings
        
    Returns:
        DataFrame with all features
    """
    engineer = FeatureEngineer(settings)
    return engineer.create_all_features(df)
