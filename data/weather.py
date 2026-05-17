"""
Weather forecast fetcher for race prediction.

Queries OpenWeatherMap's 5-day forecast API to estimate race-day conditions
at a given circuit. Used by predict.py to populate is_wet_session and
temperature features. Falls back silently if the API key is missing or the
request fails.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests

from config.settings import FEATURE_DEFAULTS

logger = logging.getLogger(__name__)

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"


def fetch_race_weather(
    lat: Optional[float],
    lon: Optional[float],
    race_date: Optional[datetime] = None,
    api_key: Optional[str] = None,
) -> Tuple[int, float]:
    """
    Return (is_wet_session, temperature_c).

    Picks the forecast entry whose timestamp is closest to `race_date`.
    Returns defaults from FEATURE_DEFAULTS on any failure.
    """
    default_wet = FEATURE_DEFAULTS['is_wet_session']
    default_temp = FEATURE_DEFAULTS['temperature']

    if lat is None or lon is None:
        return default_wet, default_temp

    api_key = api_key or os.getenv('OPENWEATHERMAP_API_KEY')
    if not api_key:
        logger.info("OPENWEATHERMAP_API_KEY not set; using default weather.")
        return default_wet, default_temp

    try:
        resp = requests.get(
            OPENWEATHER_URL,
            params={'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'},
            timeout=10,
        )
        resp.raise_for_status()
        forecasts = resp.json().get('list', [])
        if not forecasts:
            return default_wet, default_temp

        target_ts = (race_date or datetime.now(timezone.utc)).timestamp()
        nearest = min(forecasts, key=lambda f: abs(f.get('dt', 0) - target_ts))

        temp = float(nearest.get('main', {}).get('temp', default_temp))
        pop = float(nearest.get('pop', 0.0))
        rain_mm = nearest.get('rain', {}).get('3h', 0.0) if nearest.get('rain') else 0.0
        is_wet = int(pop >= 0.5 or rain_mm >= 0.5)

        logger.info(f"Weather forecast: wet={is_wet} temp={temp:.1f}C (pop={pop:.2f})")
        return is_wet, temp

    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}. Using defaults.")
        return default_wet, default_temp
