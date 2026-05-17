"""
Circuit metadata for F1 races.
Contains information about each circuit used for feature engineering.

Circuit types:
- 1: Street circuit (Monaco, Singapore, etc.)
- 2: Permanent circuit (Silverstone, Spa, etc.)
- 3: Hybrid (semi-permanent like Melbourne)

Overtaking difficulty (1-5):
- 1: Very easy (long straights, multiple DRS zones)
- 5: Very hard (narrow, few overtaking opportunities)

gap_per_grid_slot: approximate seconds-per-grid-slot used as fallback when
qualifying gap data is missing. Calibrated per-circuit because grid spread
varies: tight street tracks compress times, high-downforce circuits spread
them. Defaults to 0.30.

lat / lon: circuit coordinates used to query weather forecast in predict.py.
"""

from typing import Dict, Any, Optional

# Circuit metadata for 2024-2025 calendar
CIRCUITS: Dict[str, Dict[str, Any]] = {
    # Bahrain - Sakhir
    'bahrain': {
        'name': 'Bahrain International Circuit',
        'country': 'Bahrain',
        'city': 'Sakhir',
        'circuit_type': 2,
        'length_km': 5.412,
        'laps': 57,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.30,
        'lat': 26.0325,
        'lon': 50.5106,
    },

    # Saudi Arabia - Jeddah
    'jeddah': {
        'name': 'Jeddah Corniche Circuit',
        'country': 'Saudi Arabia',
        'city': 'Jeddah',
        'circuit_type': 1,
        'length_km': 6.174,
        'laps': 50,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.22,
        'lat': 21.6319,
        'lon': 39.1044,
    },

    # Australia - Melbourne
    'albert_park': {
        'name': 'Albert Park Circuit',
        'country': 'Australia',
        'city': 'Melbourne',
        'circuit_type': 3,
        'length_km': 5.278,
        'laps': 58,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.30,
        'lat': -37.8497,
        'lon': 144.9680,
    },

    # Japan - Suzuka
    'suzuka': {
        'name': 'Suzuka International Racing Course',
        'country': 'Japan',
        'city': 'Suzuka',
        'circuit_type': 2,
        'length_km': 5.807,
        'laps': 53,
        'overtaking_difficulty': 4,
        'gap_per_grid_slot': 0.45,
        'lat': 34.8431,
        'lon': 136.5410,
    },

    # China - Shanghai
    'shanghai': {
        'name': 'Shanghai International Circuit',
        'country': 'China',
        'city': 'Shanghai',
        'circuit_type': 2,
        'length_km': 5.451,
        'laps': 56,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.30,
        'lat': 31.3389,
        'lon': 121.2200,
    },

    # Miami
    'miami': {
        'name': 'Miami International Autodrome',
        'country': 'USA',
        'city': 'Miami',
        'circuit_type': 1,
        'length_km': 5.412,
        'laps': 57,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.28,
        'lat': 25.9581,
        'lon': -80.2389,
    },

    # Emilia Romagna - Imola
    'imola': {
        'name': 'Autodromo Enzo e Dino Ferrari',
        'country': 'Italy',
        'city': 'Imola',
        'circuit_type': 2,
        'length_km': 4.909,
        'laps': 63,
        'overtaking_difficulty': 4,
        'gap_per_grid_slot': 0.35,
        'lat': 44.3439,
        'lon': 11.7167,
    },

    # Monaco
    'monaco': {
        'name': 'Circuit de Monaco',
        'country': 'Monaco',
        'city': 'Monte Carlo',
        'circuit_type': 1,
        'length_km': 3.337,
        'laps': 78,
        'overtaking_difficulty': 5,
        'gap_per_grid_slot': 0.15,
        'lat': 43.7347,
        'lon': 7.4206,
    },

    # Canada - Montreal
    'montreal': {
        'name': 'Circuit Gilles Villeneuve',
        'country': 'Canada',
        'city': 'Montreal',
        'circuit_type': 3,
        'length_km': 4.361,
        'laps': 70,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.25,
        'lat': 45.5000,
        'lon': -73.5228,
    },

    # Spain - Barcelona
    'barcelona': {
        'name': 'Circuit de Barcelona-Catalunya',
        'country': 'Spain',
        'city': 'Barcelona',
        'circuit_type': 2,
        'length_km': 4.657,
        'laps': 66,
        'overtaking_difficulty': 4,
        'gap_per_grid_slot': 0.35,
        'lat': 41.5700,
        'lon': 2.2611,
    },

    # Austria - Spielberg
    'red_bull_ring': {
        'name': 'Red Bull Ring',
        'country': 'Austria',
        'city': 'Spielberg',
        'circuit_type': 2,
        'length_km': 4.318,
        'laps': 71,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.20,
        'lat': 47.2197,
        'lon': 14.7647,
    },

    # Great Britain - Silverstone
    'silverstone': {
        'name': 'Silverstone Circuit',
        'country': 'Great Britain',
        'city': 'Silverstone',
        'circuit_type': 2,
        'length_km': 5.891,
        'laps': 52,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.45,
        'lat': 52.0786,
        'lon': -1.0169,
    },

    # Hungary - Budapest
    'hungaroring': {
        'name': 'Hungaroring',
        'country': 'Hungary',
        'city': 'Budapest',
        'circuit_type': 2,
        'length_km': 4.381,
        'laps': 70,
        'overtaking_difficulty': 5,
        'gap_per_grid_slot': 0.35,
        'lat': 47.5789,
        'lon': 19.2486,
    },

    # Belgium - Spa
    'spa': {
        'name': 'Circuit de Spa-Francorchamps',
        'country': 'Belgium',
        'city': 'Spa',
        'circuit_type': 2,
        'length_km': 7.004,
        'laps': 44,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.45,
        'lat': 50.4372,
        'lon': 5.9714,
    },

    # Netherlands - Zandvoort
    'zandvoort': {
        'name': 'Circuit Zandvoort',
        'country': 'Netherlands',
        'city': 'Zandvoort',
        'circuit_type': 2,
        'length_km': 4.259,
        'laps': 72,
        'overtaking_difficulty': 5,
        'gap_per_grid_slot': 0.35,
        'lat': 52.3888,
        'lon': 4.5409,
    },

    # Italy - Monza
    'monza': {
        'name': 'Autodromo Nazionale Monza',
        'country': 'Italy',
        'city': 'Monza',
        'circuit_type': 2,
        'length_km': 5.793,
        'laps': 53,
        'overtaking_difficulty': 1,
        'gap_per_grid_slot': 0.22,
        'lat': 45.6156,
        'lon': 9.2811,
    },

    # Azerbaijan - Baku
    'baku': {
        'name': 'Baku City Circuit',
        'country': 'Azerbaijan',
        'city': 'Baku',
        'circuit_type': 1,
        'length_km': 6.003,
        'laps': 51,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.20,
        'lat': 40.3725,
        'lon': 49.8533,
    },

    # Singapore
    'singapore': {
        'name': 'Marina Bay Street Circuit',
        'country': 'Singapore',
        'city': 'Singapore',
        'circuit_type': 1,
        'length_km': 4.940,
        'laps': 62,
        'overtaking_difficulty': 4,
        'gap_per_grid_slot': 0.18,
        'lat': 1.2914,
        'lon': 103.8642,
    },

    # USA - Austin (COTA)
    'austin': {
        'name': 'Circuit of the Americas',
        'country': 'USA',
        'city': 'Austin',
        'circuit_type': 2,
        'length_km': 5.513,
        'laps': 56,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.32,
        'lat': 30.1328,
        'lon': -97.6411,
    },

    # Mexico - Mexico City
    'mexico': {
        'name': 'Autódromo Hermanos Rodríguez',
        'country': 'Mexico',
        'city': 'Mexico City',
        'circuit_type': 2,
        'length_km': 4.304,
        'laps': 71,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.28,
        'lat': 19.4042,
        'lon': -99.0907,
    },

    # Brazil - Sao Paulo (Interlagos)
    'interlagos': {
        'name': 'Autódromo José Carlos Pace',
        'country': 'Brazil',
        'city': 'São Paulo',
        'circuit_type': 2,
        'length_km': 4.309,
        'laps': 71,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.25,
        'lat': -23.7036,
        'lon': -46.6997,
    },

    # Las Vegas
    'las_vegas': {
        'name': 'Las Vegas Street Circuit',
        'country': 'USA',
        'city': 'Las Vegas',
        'circuit_type': 1,
        'length_km': 6.201,
        'laps': 50,
        'overtaking_difficulty': 2,
        'gap_per_grid_slot': 0.20,
        'lat': 36.1147,
        'lon': -115.1728,
    },

    # Qatar - Lusail
    'losail': {
        'name': 'Lusail International Circuit',
        'country': 'Qatar',
        'city': 'Lusail',
        'circuit_type': 2,
        'length_km': 5.380,
        'laps': 57,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.30,
        'lat': 25.4900,
        'lon': 51.4542,
    },

    # Abu Dhabi - Yas Marina
    'yas_marina': {
        'name': 'Yas Marina Circuit',
        'country': 'UAE',
        'city': 'Abu Dhabi',
        'circuit_type': 2,
        'length_km': 5.281,
        'laps': 58,
        'overtaking_difficulty': 3,
        'gap_per_grid_slot': 0.30,
        'lat': 24.4672,
        'lon': 54.6031,
    },
}

# Mapping from FastF1 circuit names to our keys
FASTF1_CIRCUIT_MAPPING: Dict[str, str] = {
    'Bahrain': 'bahrain',
    'Bahrain International Circuit': 'bahrain',
    'Sakhir': 'bahrain',
    'Saudi Arabia': 'jeddah',
    'Jeddah': 'jeddah',
    'Jeddah Corniche Circuit': 'jeddah',
    'Australia': 'albert_park',
    'Melbourne': 'albert_park',
    'Albert Park': 'albert_park',
    'Japan': 'suzuka',
    'Suzuka': 'suzuka',
    'China': 'shanghai',
    'Shanghai': 'shanghai',
    'Miami': 'miami',
    'Emilia Romagna': 'imola',
    'Imola': 'imola',
    'Monaco': 'monaco',
    'Monte Carlo': 'monaco',
    'Canada': 'montreal',
    'Montreal': 'montreal',
    'Spain': 'barcelona',
    'Barcelona': 'barcelona',
    'Austria': 'red_bull_ring',
    'Spielberg': 'red_bull_ring',
    'Red Bull Ring': 'red_bull_ring',
    'Great Britain': 'silverstone',
    'Silverstone': 'silverstone',
    'Hungary': 'hungaroring',
    'Budapest': 'hungaroring',
    'Hungaroring': 'hungaroring',
    'Belgium': 'spa',
    'Spa': 'spa',
    'Spa-Francorchamps': 'spa',
    'Netherlands': 'zandvoort',
    'Zandvoort': 'zandvoort',
    'Italy': 'monza',
    'Monza': 'monza',
    'Azerbaijan': 'baku',
    'Baku': 'baku',
    'Singapore': 'singapore',
    'Marina Bay': 'singapore',
    'United States': 'austin',
    'Austin': 'austin',
    'COTA': 'austin',
    'Mexico': 'mexico',
    'Mexico City': 'mexico',
    'Brazil': 'interlagos',
    'São Paulo': 'interlagos',
    'Interlagos': 'interlagos',
    'Las Vegas': 'las_vegas',
    'Qatar': 'losail',
    'Lusail': 'losail',
    'Abu Dhabi': 'yas_marina',
    'Yas Marina': 'yas_marina',
}


def get_circuit_info(circuit_name: str) -> Optional[Dict[str, Any]]:
    """
    Get circuit information by name.

    Args:
        circuit_name: Circuit name (can be FastF1 format or our key)

    Returns:
        Dictionary with circuit metadata or None if not found
    """
    # Try direct lookup first
    if circuit_name.lower().replace(' ', '_') in CIRCUITS:
        return CIRCUITS[circuit_name.lower().replace(' ', '_')]

    # Try mapping from FastF1 names
    circuit_key = FASTF1_CIRCUIT_MAPPING.get(circuit_name)
    if circuit_key:
        return CIRCUITS.get(circuit_key)

    # Try partial match
    for key, info in CIRCUITS.items():
        if circuit_name.lower() in info['name'].lower():
            return info
        if circuit_name.lower() in info['city'].lower():
            return info

    return None


def get_circuit_key(circuit_name: str) -> Optional[str]:
    """
    Get the normalized circuit key from any circuit name format.

    Args:
        circuit_name: Circuit name in any format

    Returns:
        Normalized circuit key or None if not found
    """
    # Try direct lookup
    normalized = circuit_name.lower().replace(' ', '_')
    if normalized in CIRCUITS:
        return normalized

    # Try mapping
    return FASTF1_CIRCUIT_MAPPING.get(circuit_name)
