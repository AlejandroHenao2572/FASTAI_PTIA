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
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

from config.settings import Settings, FEATURE_DEFAULTS
from config.circuits import CIRCUITS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_finished(status: object) -> bool:
    """Return True if a result status indicates a classified finish."""
    s = str(status)
    return 'Finished' in s or '+' in s


class FeatureEngineer:
    """
    Creates features for F1 race prediction.

    Key principles:
    1. Temporal ordering: Historical features use ONLY past data
    2. No data leakage: Race results not used in features
    3. Handles missing data with sensible defaults
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.default_position = self.settings.default_position
        self.min_races = self.settings.min_races_for_history
        self.lookback = self.settings.lookback_races

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering...")
        df = df.copy()
        df = df.sort_values(['year', 'round']).reset_index(drop=True)

        df = self._create_qualifying_features(df)
        df = self._create_driver_historical_features(df)
        df = self._create_team_features(df)
        df = self._create_circuit_features(df)
        df = self._create_grid_features(df)
        df = self._handle_missing_values(df)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    # -------------------------------------------------------------------------
    # Qualifying features
    # -------------------------------------------------------------------------
    def _create_qualifying_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating qualifying features...")

        if 'quali_position' not in df.columns:
            df['quali_position'] = df['grid_position']

        # Per-circuit gap-per-grid-slot fallback (replaces hardcoded 0.3s).
        gap_per_slot = df['circuit_key'].map(
            lambda k: CIRCUITS.get(k, {}).get(
                'gap_per_grid_slot', FEATURE_DEFAULTS['gap_per_grid_slot']
            )
        ).fillna(FEATURE_DEFAULTS['gap_per_grid_slot'])

        if 'quali_gap_to_pole' not in df.columns:
            df['quali_gap_to_pole'] = np.nan
        df['quali_gap_to_pole'] = df['quali_gap_to_pole'].fillna(
            (df['quali_position'] - 1) * gap_per_slot
        )

        if 'made_q3' not in df.columns:
            df['made_q3'] = (df['quali_position'] <= 10).astype(int)

        # Vectorized teammate gap: within (year, round, team), gap between
        # each driver's quali position and the team's minimum quali position.
        # Returns 0 for the lead driver, positive for the slower teammate.
        team_min = df.groupby(['year', 'round', 'team'])['quali_position'].transform('min')
        df['quali_gap_to_teammate'] = df['quali_position'] - team_min

        return df

    # -------------------------------------------------------------------------
    # Driver historical features (vectorized, leak-free)
    # -------------------------------------------------------------------------
    def _create_driver_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Driver-level historical signals computed via groupby + shift so that
        each row only sees that driver's strictly past races.
        """
        logger.info("Creating driver historical features...")

        df = df.sort_values(['driver_code', 'year', 'round']).reset_index(drop=True)

        df['_is_dnf'] = (~df['status'].apply(_is_finished)).astype(float)
        lookback = self.lookback

        # Use transform so the result is aligned to the source index; the
        # inner lambda shifts to exclude the current race before aggregating.
        df['driver_avg_position_last_5'] = df.groupby('driver_code')['finish_position'].transform(
            lambda s: s.shift(1).rolling(window=lookback, min_periods=1).mean()
        )

        df['driver_dnf_rate'] = df.groupby('driver_code')['_is_dnf'].transform(
            lambda s: s.shift(1).expanding().mean()
        )

        df['driver_experience'] = df.groupby('driver_code').cumcount()

        df['driver_circuit_avg_position'] = df.groupby(
            ['driver_code', 'circuit_key']
        )['finish_position'].transform(
            lambda s: s.shift(1).expanding().mean()
        )

        df = df.drop(columns=['_is_dnf'])

        # Apply min_races floor: rows with fewer past races get defaults.
        below_min = df['driver_experience'] < self.min_races
        df.loc[below_min, 'driver_avg_position_last_5'] = FEATURE_DEFAULTS['driver_avg_position_last_5']
        df.loc[below_min, 'driver_dnf_rate'] = FEATURE_DEFAULTS['driver_dnf_rate']

        # Circuit fallback: if no past races at this circuit, reuse recent avg.
        no_circuit_hist = df['driver_circuit_avg_position'].isna()
        df.loc[no_circuit_hist, 'driver_circuit_avg_position'] = df.loc[
            no_circuit_hist, 'driver_avg_position_last_5'
        ]

        # Restore chronological ordering for downstream feature groups.
        df = df.sort_values(['year', 'round']).reset_index(drop=True)
        return df

    # -------------------------------------------------------------------------
    # Team features (vectorized via groupby)
    # -------------------------------------------------------------------------
    def _create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating team features...")

        df = df.sort_values(['team', 'year', 'round']).reset_index(drop=True)
        df['_is_finished'] = df['status'].apply(_is_finished).astype(float)

        past_finish_count = df.groupby(['team', 'year']).cumcount()

        df['team_avg_position_season'] = df.groupby(['team', 'year'])['finish_position'].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        df['team_reliability_rate'] = df.groupby(['team', 'year'])['_is_finished'].transform(
            lambda s: s.shift(1).expanding().mean()
        )

        # Need >= 2 past observations to trust team_avg (mirrors prior heuristic).
        insufficient = past_finish_count < 2
        df.loc[insufficient, 'team_avg_position_season'] = FEATURE_DEFAULTS['team_avg_position_season']
        df.loc[insufficient, 'team_reliability_rate'] = FEATURE_DEFAULTS['team_reliability_rate']
        df = df.drop(columns=['_is_finished'])

        # Constructor standing: ranking by cumulative team points across past
        # rounds of the same season. Vectorized via per-(year, round) merge.
        df = df.sort_values(['year', 'round']).reset_index(drop=True)
        standings = self._compute_constructor_standings(df)
        df = df.merge(standings, on=['year', 'round', 'team'], how='left')
        df['constructor_standing'] = df['constructor_standing'].fillna(
            FEATURE_DEFAULTS['constructor_standing']
        ).astype(int)

        return df

    @staticmethod
    def _compute_constructor_standings(df: pd.DataFrame) -> pd.DataFrame:
        """
        For each (year, round, team) compute rank by cumulative points
        scored in all earlier rounds of the same year (strictly past).
        """
        # Sum points per (year, round, team) then cumulative within (year, team).
        per_round = (
            df.groupby(['year', 'round', 'team'], as_index=False)['points'].sum()
              .sort_values(['year', 'team', 'round'])
        )
        per_round['cum_points'] = (
            per_round.groupby(['year', 'team'])['points'].cumsum()
                     - per_round['points']  # exclude current round
        )

        per_round['constructor_standing'] = (
            per_round.groupby(['year', 'round'])['cum_points']
                     .rank(method='min', ascending=False)
                     .astype(int)
        )
        return per_round[['year', 'round', 'team', 'constructor_standing']]

    # -------------------------------------------------------------------------
    # Circuit / grid / conditions
    # -------------------------------------------------------------------------
    def _create_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validating circuit features...")
        if 'circuit_type' not in df.columns:
            df['circuit_type'] = FEATURE_DEFAULTS['circuit_type']
        if 'circuit_length_km' not in df.columns:
            df['circuit_length_km'] = FEATURE_DEFAULTS['circuit_length_km']
        if 'overtaking_difficulty' not in df.columns:
            df['overtaking_difficulty'] = FEATURE_DEFAULTS['overtaking_difficulty']
        if 'number_of_laps' not in df.columns:
            df['number_of_laps'] = FEATURE_DEFAULTS['number_of_laps']
        return df

    def _create_grid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating grid and condition features...")
        if 'grid_position' not in df.columns:
            df['grid_position'] = df.get('quali_position', self.default_position)
        df.loc[df['grid_position'] == 0, 'grid_position'] = 20
        if 'is_wet_session' not in df.columns:
            df['is_wet_session'] = FEATURE_DEFAULTS['is_wet_session']
        if 'temperature' not in df.columns:
            df['temperature'] = FEATURE_DEFAULTS['temperature']
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values...")
        for col, default in FEATURE_DEFAULTS.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        return df

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------
    def get_feature_columns(self) -> List[str]:
        return self.settings.feature_columns.copy()

    def prepare_training_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        feature_cols = self.get_feature_columns()
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            for col in missing_cols:
                df[col] = FEATURE_DEFAULTS.get(col, self.default_position)

        X = df[feature_cols].copy()
        y = df[self.settings.target_column].copy()
        return X, y


def create_features(df: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
    """Convenience wrapper."""
    return FeatureEngineer(settings).create_all_features(df)


def export_historical_stats(df: pd.DataFrame) -> dict:
    """
    Build per-driver and per-team historical stats from the engineered df.

    Used to serialize the same stats the trainer saw so prediction at serve
    time consumes identical signals (no train/serve skew).
    """
    df = df.sort_values(['year', 'round'])

    # Per-driver totals computed from raw results (independent of row order).
    is_dnf = (~df['status'].apply(_is_finished)).astype(int)
    driver_groups = df.groupby('driver_code')

    avg_pos = driver_groups['finish_position'].apply(
        lambda s: float(s.tail(5).mean())
    )
    dnf_rate = is_dnf.groupby(df['driver_code']).mean()
    experience = driver_groups.size()

    # Per-driver per-circuit average finish.
    circuit_avg = (
        df.groupby(['driver_code', 'circuit_key'])['finish_position'].mean()
    )

    drivers = {}
    for code, n_races in experience.items():
        circuit_map = {
            ck: float(v)
            for (drv, ck), v in circuit_avg.items()
            if drv == code and ck
        }
        drivers[code] = {
            'avg_pos': float(avg_pos.get(code, FEATURE_DEFAULTS['driver_avg_position_last_5'])),
            'circuit_avg': circuit_map,
            'dnf_rate': float(dnf_rate.get(code, FEATURE_DEFAULTS['driver_dnf_rate'])),
            'experience': int(n_races),
        }

    # Per-team: take the most recent row's engineered values (already temporal).
    team_latest = df.sort_values(['year', 'round']).groupby('team').tail(1)
    teams = {}
    for _, row in team_latest.iterrows():
        teams[row['team']] = {
            'avg_pos': float(row.get('team_avg_position_season',
                                     FEATURE_DEFAULTS['team_avg_position_season'])),
            'reliability': float(row.get('team_reliability_rate',
                                         FEATURE_DEFAULTS['team_reliability_rate'])),
            'standing': int(row.get('constructor_standing',
                                    FEATURE_DEFAULTS['constructor_standing'])),
        }

    return {'drivers': drivers, 'teams': teams}
