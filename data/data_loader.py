"""
Data loader module for F1 Race Predictor.
Fetches race data from FastF1 API with caching support.

FastF1 is the official F1 data library that provides:
- Session data (practice, qualifying, race)
- Lap times and telemetry
- Weather data
- Driver and team information
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import fastf1

from config.settings import Settings, PROJECT_ROOT
from config.circuits import get_circuit_info, get_circuit_key, FASTF1_CIRCUIT_MAPPING

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataLoader:
    """
    Loads F1 race data from FastF1 API.
    
    Handles caching, session loading, and data extraction for:
    - Race results
    - Qualifying results
    - Weather data
    - Driver/team information
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the data loader.
        
        Args:
            settings: Configuration settings (uses default if None)
        """
        self.settings = settings or Settings()
        self._setup_cache()
        
    def _setup_cache(self):
        """Configure FastF1 cache directory."""
        cache_path = self.settings.cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_path))
        logger.info(f"FastF1 cache enabled at: {cache_path}")
    
    def get_season_schedule(self, year: int) -> pd.DataFrame:
        """
        Get the race schedule for a given season.
        
        Args:
            year: Season year
            
        Returns:
            DataFrame with race schedule
        """
        try:
            schedule = fastf1.get_event_schedule(year)
            # Filter only race events (exclude testing)
            races = schedule[schedule['EventFormat'].notna()]
            logger.info(f"Loaded {len(races)} events for {year} season")
            return races
        except Exception as e:
            logger.error(f"Error loading schedule for {year}: {e}")
            return pd.DataFrame()
    
    def load_session(
        self, 
        year: int, 
        race: str | int, 
        session_type: str = 'R'
    ) -> Optional[fastf1.core.Session]:
        """
        Load a specific session (Race, Qualifying, etc.).
        
        Args:
            year: Season year
            race: Race name or round number
            session_type: 'R' (Race), 'Q' (Qualifying), 'FP1', 'FP2', 'FP3', 'S' (Sprint)
            
        Returns:
            FastF1 Session object or None if loading fails
        """
        try:
            session = fastf1.get_session(year, race, session_type)
            session.load()
            logger.info(f"Loaded {session_type} session for {race} {year}")
            return session
        except Exception as e:
            logger.warning(f"Could not load {session_type} for {race} {year}: {e}")
            return None
    
    def get_race_results(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract race results from a session.
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with race results
        """
        try:
            results = session.results.copy()
            
            # Clean and standardize columns
            results_clean = pd.DataFrame({
                'driver_code': results['Abbreviation'],
                'driver_name': results['FullName'],
                'team': results['TeamName'],
                'finish_position': results['Position'].astype(float),
                'grid_position': results['GridPosition'].astype(float),
                'status': results['Status'],
                'points': results['Points'].astype(float),
                'race_time': results.get('Time', pd.NaT),
            })
            
            # Handle DNFs: assign position 20 for non-finishers
            # This is a design decision - DNFs get worst position
            dnf_mask = ~results_clean['status'].str.contains('Finished|\\+', na=False)
            results_clean.loc[dnf_mask, 'finish_position'] = results_clean.loc[
                dnf_mask, 'finish_position'
            ].fillna(20)
            
            return results_clean
            
        except Exception as e:
            logger.error(f"Error extracting race results: {e}")
            return pd.DataFrame()
    
    def get_qualifying_results(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract qualifying results from a session.
        
        Args:
            session: FastF1 qualifying session object
            
        Returns:
            DataFrame with qualifying results
        """
        try:
            results = session.results.copy()
            
            # Extract Q1, Q2, Q3 times
            quali_clean = pd.DataFrame({
                'driver_code': results['Abbreviation'],
                'quali_position': results['Position'].astype(float),
                'q1_time': results.get('Q1', pd.NaT),
                'q2_time': results.get('Q2', pd.NaT),
                'q3_time': results.get('Q3', pd.NaT),
                'team': results['TeamName'],
            })
            
            # Convert timedelta to seconds for best time
            quali_clean['best_quali_time'] = quali_clean.apply(
                self._get_best_quali_time, axis=1
            )
            
            # Calculate gap to pole
            pole_time = quali_clean['best_quali_time'].min()
            quali_clean['quali_gap_to_pole'] = quali_clean['best_quali_time'] - pole_time
            
            # Determine if made Q3
            quali_clean['made_q3'] = quali_clean['q3_time'].notna().astype(int)
            
            return quali_clean
            
        except Exception as e:
            logger.error(f"Error extracting qualifying results: {e}")
            return pd.DataFrame()
    
    def _get_best_quali_time(self, row: pd.Series) -> float:
        """Get the best qualifying time in seconds from Q1/Q2/Q3."""
        times = []
        for q in ['q3_time', 'q2_time', 'q1_time']:
            if pd.notna(row.get(q)):
                try:
                    # Handle timedelta
                    if isinstance(row[q], pd.Timedelta):
                        times.append(row[q].total_seconds())
                    else:
                        times.append(float(row[q]))
                except (ValueError, TypeError):
                    continue
        return min(times) if times else np.nan
    
    def get_weather_data(self, session: fastf1.core.Session) -> Dict[str, Any]:
        """
        Extract weather data from a session.
        
        Args:
            session: FastF1 session object
            
        Returns:
            Dictionary with weather information
        """
        try:
            weather = session.weather_data
            
            if weather is None or weather.empty:
                return {
                    'air_temp': 25.0,  # Default values
                    'track_temp': 35.0,
                    'humidity': 50.0,
                    'rainfall': False,
                    'is_wet': False,
                }
            
            # Get average conditions during session
            return {
                'air_temp': weather['AirTemp'].mean(),
                'track_temp': weather['TrackTemp'].mean(),
                'humidity': weather['Humidity'].mean(),
                'rainfall': weather['Rainfall'].any(),
                'is_wet': weather['Rainfall'].any() or weather.get('TrackTemp', pd.Series([35])).mean() < 20,
            }
            
        except Exception as e:
            logger.warning(f"Error extracting weather data: {e}")
            return {
                'air_temp': 25.0,
                'track_temp': 35.0,
                'humidity': 50.0,
                'rainfall': False,
                'is_wet': False,
            }
    
    def load_race_weekend(
        self, 
        year: int, 
        race: str | int
    ) -> Dict[str, Any]:
        """
        Load all relevant data for a race weekend.
        
        Args:
            year: Season year
            race: Race name or round number
            
        Returns:
            Dictionary containing race results, qualifying, and metadata
        """
        # Load sessions
        race_session = self.load_session(year, race, 'R')
        quali_session = self.load_session(year, race, 'Q')
        
        if race_session is None:
            logger.error(f"Could not load race session for {race} {year}")
            return {}
        
        # Extract data
        race_results = self.get_race_results(race_session)
        quali_results = self.get_qualifying_results(quali_session) if quali_session else pd.DataFrame()
        weather = self.get_weather_data(race_session)
        
        # Get circuit info
        event_name = race_session.event['EventName']
        circuit_key = get_circuit_key(event_name) or get_circuit_key(str(race))
        circuit_info = get_circuit_info(event_name) or {}
        
        return {
            'year': year,
            'round': race_session.event['RoundNumber'],
            'race_name': event_name,
            'circuit_key': circuit_key,
            'circuit_info': circuit_info,
            'race_results': race_results,
            'qualifying_results': quali_results,
            'weather': weather,
            'date': race_session.event['EventDate'],
        }
    
    def load_season_data(self, year: int) -> List[Dict[str, Any]]:
        """
        Load data for all races in a season.
        
        Args:
            year: Season year
            
        Returns:
            List of race weekend data dictionaries
        """
        schedule = self.get_season_schedule(year)
        
        if schedule.empty:
            return []
        
        season_data = []
        
        for idx, event in schedule.iterrows():
            round_num = event['RoundNumber']
            event_name = event['EventName']
            
            # Skip testing events
            if 'test' in event_name.lower():
                continue
                
            logger.info(f"Loading Round {round_num}: {event_name}")
            
            try:
                race_data = self.load_race_weekend(year, round_num)
                if race_data:
                    season_data.append(race_data)
            except Exception as e:
                logger.warning(f"Skipping {event_name}: {e}")
                continue
        
        logger.info(f"Loaded {len(season_data)} races for {year} season")
        return season_data
    
    def load_multiple_seasons(
        self, 
        years: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Load data for multiple seasons.
        
        Args:
            years: List of season years
            
        Returns:
            Combined list of race weekend data
        """
        all_data = []
        
        for year in years:
            logger.info(f"Loading {year} season...")
            season_data = self.load_season_data(year)
            all_data.extend(season_data)
        
        logger.info(f"Total races loaded: {len(all_data)}")
        return all_data
    
    def create_training_dataframe(
        self, 
        race_weekends: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a flat DataFrame from race weekend data for training.
        
        Args:
            race_weekends: List of race weekend dictionaries
            
        Returns:
            DataFrame with one row per driver per race
        """
        rows = []
        
        for race in race_weekends:
            race_results = race.get('race_results', pd.DataFrame())
            quali_results = race.get('qualifying_results', pd.DataFrame())
            weather = race.get('weather', {})
            circuit_info = race.get('circuit_info', {})
            
            if race_results.empty:
                continue
            
            for _, driver_race in race_results.iterrows():
                row = {
                    # Race metadata
                    'year': race['year'],
                    'round': race['round'],
                    'race_name': race['race_name'],
                    'circuit_key': race.get('circuit_key', ''),
                    'date': race.get('date'),
                    
                    # Driver info
                    'driver_code': driver_race['driver_code'],
                    'driver_name': driver_race['driver_name'],
                    'team': driver_race['team'],
                    
                    # Target variable
                    'finish_position': driver_race['finish_position'],
                    
                    # Race features
                    'grid_position': driver_race['grid_position'],
                    'status': driver_race['status'],
                    'points': driver_race['points'],
                    
                    # Weather features
                    'temperature': weather.get('air_temp', 25.0),
                    'is_wet_session': int(weather.get('is_wet', False)),
                    
                    # Circuit features
                    'circuit_type': circuit_info.get('circuit_type', 2),
                    'circuit_length_km': circuit_info.get('length_km', 5.0),
                    'overtaking_difficulty': circuit_info.get('overtaking_difficulty', 3),
                    'number_of_laps': circuit_info.get('laps', 55),
                }
                
                # Add qualifying data if available
                if not quali_results.empty:
                    driver_quali = quali_results[
                        quali_results['driver_code'] == driver_race['driver_code']
                    ]
                    
                    if not driver_quali.empty:
                        driver_quali = driver_quali.iloc[0]
                        row['quali_position'] = driver_quali['quali_position']
                        row['quali_gap_to_pole'] = driver_quali['quali_gap_to_pole']
                        row['made_q3'] = driver_quali['made_q3']
                        row['best_quali_time'] = driver_quali['best_quali_time']
                    else:
                        row['quali_position'] = driver_race['grid_position']
                        row['quali_gap_to_pole'] = np.nan
                        row['made_q3'] = 0
                        row['best_quali_time'] = np.nan
                else:
                    row['quali_position'] = driver_race['grid_position']
                    row['quali_gap_to_pole'] = np.nan
                    row['made_q3'] = 0
                    row['best_quali_time'] = np.nan
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by date and round for temporal ordering
        df = df.sort_values(['year', 'round']).reset_index(drop=True)
        
        logger.info(f"Created training DataFrame with {len(df)} rows")
        return df


# Utility function for quick data loading
def load_training_data(
    seasons: List[int] = [2023, 2024],
    settings: Optional[Settings] = None
) -> pd.DataFrame:
    """
    Convenience function to load training data for specified seasons.
    
    Args:
        seasons: List of season years to load
        settings: Configuration settings
        
    Returns:
        DataFrame ready for feature engineering
    """
    loader = F1DataLoader(settings)
    race_data = loader.load_multiple_seasons(seasons)
    df = loader.create_training_dataframe(race_data)
    return df
