#!/usr/bin/env python
"""
F1 Race Predictor - Prediction Script.

Loads only the qualifying session for the target race and consumes the
historical_stats.json produced by main.py, so prediction features match the
training pipeline (no train/serve skew).

Usage:
    python predict.py --race "Monaco" --year 2025
    python predict.py --race 6 --year 2025
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config.settings import Settings, FEATURE_DEFAULTS
from config.circuits import get_circuit_info, get_circuit_key
from models.trainer import F1ModelTrainer
from data.weather import fetch_race_weather

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='F1 Race Predictor')
    parser.add_argument('--race', type=str, required=True,
                        help='Race name (e.g., "Monaco") or round number')
    parser.add_argument('--year', type=int, required=True, help='Season year')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (uses latest if omitted)')
    parser.add_argument('--stats', type=str, default=None,
                        help='Path to historical_stats.json (default: outputs/reports/)')
    return parser.parse_args()


def find_latest_model(models_dir: Path) -> Optional[Path]:
    model_files = sorted(models_dir.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    return model_files[0] if model_files else None


def load_historical_stats(stats_path: Path) -> Dict:
    """Load serialized driver/team stats produced during training."""
    if not stats_path.exists():
        logger.warning(
            f"{stats_path} not found. Train first via main.py. "
            "Using FEATURE_DEFAULTS for all drivers/teams."
        )
        return {'drivers': {}, 'teams': {}}

    with open(stats_path) as f:
        return json.load(f)


def get_driver_stats(code: str, circuit_key: str, stats: Dict) -> Dict:
    d = stats.get('drivers', {}).get(code, {})
    circuit_avg = d.get('circuit_avg', {}).get(circuit_key)
    return {
        'avg_pos': d.get('avg_pos', FEATURE_DEFAULTS['driver_avg_position_last_5']),
        'circuit_avg': circuit_avg if circuit_avg is not None
                       else d.get('avg_pos', FEATURE_DEFAULTS['driver_circuit_avg_position']),
        'dnf_rate': d.get('dnf_rate', FEATURE_DEFAULTS['driver_dnf_rate']),
        'experience': d.get('experience', FEATURE_DEFAULTS['driver_experience']),
    }


def get_team_stats(team: str, stats: Dict) -> Dict:
    teams = stats.get('teams', {})
    if team in teams:
        return teams[team]
    # Fuzzy match (handles "Red Bull" vs "Red Bull Racing").
    for key, val in teams.items():
        if key.lower() in team.lower() or team.lower() in key.lower():
            return val
    return {
        'avg_pos': FEATURE_DEFAULTS['team_avg_position_season'],
        'reliability': FEATURE_DEFAULTS['team_reliability_rate'],
        'standing': FEATURE_DEFAULTS['constructor_standing'],
    }


def predict_race(
    race: str,
    year: int,
    model_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str]:
    settings = Settings()

    if model_path is None:
        model_path = find_latest_model(settings.models_dir)
        if model_path is None:
            raise FileNotFoundError(
                f"No model in {settings.models_dir}. Run main.py first."
            )

    if stats_path is None:
        stats_path = settings.reports_dir / 'historical_stats.json'

    print(f"\n[1/4] Loading model from {model_path.name}...")
    trainer = F1ModelTrainer(settings)
    trainer.load_model(model_path)

    print(f"[2/4] Loading historical stats from {stats_path.name}...")
    hist_stats = load_historical_stats(stats_path)

    import fastf1
    for name in ('fastf1', 'fastf1.core', 'fastf1.api', 'fastf1.req'):
        logging.getLogger(name).setLevel(logging.ERROR)
    fastf1.Cache.enable_cache(str(settings.cache_dir))

    try:
        race_id = int(race)
    except ValueError:
        race_id = race

    print(f"[3/4] Loading qualifying data for {race} {year}...")
    quali = fastf1.get_session(year, race_id, 'Q')
    quali.load()
    results = quali.results
    if results.empty:
        raise ValueError("No qualifying results found")

    event_name = quali.event['EventName']
    circuit_key = get_circuit_key(event_name) or ''
    circuit_info = get_circuit_info(event_name) or {}
    gap_per_slot = circuit_info.get('gap_per_grid_slot', FEATURE_DEFAULTS['gap_per_grid_slot'])

    print(f"    Circuit: {event_name}")
    print(f"    Drivers: {len(results)}")

    # Real weather (silent fallback if no API key / request fails).
    race_date = quali.event.get('EventDate')
    if hasattr(race_date, 'to_pydatetime'):
        race_date = race_date.to_pydatetime()
    is_wet, temperature = fetch_race_weather(
        lat=circuit_info.get('lat'),
        lon=circuit_info.get('lon'),
        race_date=race_date,
    )

    print(f"[4/4] Building features and predicting...")

    pole_time = None
    for _, row in results.iterrows():
        q3 = row.get('Q3')
        if pd.notna(q3) and (pole_time is None or q3 < pole_time):
            pole_time = q3

    rows = []
    for _, q_row in results.iterrows():
        code = q_row['Abbreviation']
        team = q_row['TeamName']
        quali_pos = float(q_row['Position'])

        d_stats = get_driver_stats(code, circuit_key, hist_stats)
        t_stats = get_team_stats(team, hist_stats)

        best_time = q_row.get('Q3') or q_row.get('Q2') or q_row.get('Q1')
        if pd.notna(best_time) and pd.notna(pole_time):
            try:
                quali_gap = (best_time - pole_time).total_seconds()
            except Exception:
                quali_gap = (quali_pos - 1) * gap_per_slot
        else:
            quali_gap = (quali_pos - 1) * gap_per_slot

        rows.append({
            'quali_position': quali_pos,
            'quali_gap_to_pole': max(0.0, quali_gap),
            'quali_gap_to_teammate': 0.0,
            'made_q3': 1 if quali_pos <= 10 else 0,

            'driver_avg_position_last_5': d_stats['avg_pos'],
            'driver_circuit_avg_position': d_stats['circuit_avg'],
            'driver_dnf_rate': d_stats['dnf_rate'],
            'driver_experience': d_stats['experience'],

            'team_avg_position_season': t_stats['avg_pos'],
            'team_reliability_rate': t_stats['reliability'],
            'constructor_standing': t_stats['standing'],

            'circuit_type': circuit_info.get('circuit_type', FEATURE_DEFAULTS['circuit_type']),
            'circuit_length_km': circuit_info.get('length_km', FEATURE_DEFAULTS['circuit_length_km']),
            'overtaking_difficulty': circuit_info.get('overtaking_difficulty', FEATURE_DEFAULTS['overtaking_difficulty']),
            'number_of_laps': circuit_info.get('laps', FEATURE_DEFAULTS['number_of_laps']),

            'grid_position': quali_pos,
            'is_wet_session': is_wet,
            'temperature': temperature,

            'driver_code': code,
            'team': team,
        })

    pred_df = pd.DataFrame(rows)

    team_min = pred_df.groupby('team')['quali_position'].transform('min')
    pred_df['quali_gap_to_teammate'] = pred_df['quali_position'] - team_min

    X = pred_df[settings.feature_columns].copy()
    predictions = trainer.predict(X)

    pred_df['predicted_position_raw'] = predictions
    pred_df['predicted_position'] = (
        pred_df['predicted_position_raw'].rank(method='first').astype(int)
    )
    pred_df = pred_df.sort_values('predicted_position')

    return pred_df, event_name


def print_predictions(pred_df: pd.DataFrame, race_name: str, year: int):
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    F1 RACE PREDICTION                         ║
╠═══════════════════════════════════════════════════════════════╣
║  {race_name:<55} ║
║  Season {year}                                                 ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    print("=" * 65)
    print(f"{'POS':>4} {'DRIVER':<6} {'TEAM':<26} {'GRID':>5} {'SCORE':>8}")
    print("=" * 65)

    for _, row in pred_df.iterrows():
        pos = int(row['predicted_position'])
        prefix = f" P{pos}" if pos <= 3 else f"P{pos:02d}"
        print(f"{prefix:>4} {row['driver_code']:<6} {str(row['team'])[:24]:<26} "
              f"{int(row['grid_position']):>5} {row['predicted_position_raw']:>8.2f}")

    print("=" * 65)

    print("\n PREDICTED PODIUM:")
    medals = [' 1st:', ' 2nd:', ' 3rd:']
    for i, (_, row) in enumerate(pred_df.head(3).iterrows()):
        print(f"   {medals[i]} {row['driver_code']} ({row['team']})")

    pred_df = pred_df.copy()
    pred_df['change'] = pred_df['grid_position'] - pred_df['predicted_position']

    gainers = pred_df[pred_df['change'] > 0].nlargest(3, 'change')
    if not gainers.empty:
        print("\n PREDICTED GAINERS:")
        for _, row in gainers.iterrows():
            print(f"   {row['driver_code']}: P{int(row['grid_position'])} -> "
                  f"P{int(row['predicted_position'])} (+{int(row['change'])})")

    losers = pred_df[pred_df['change'] < 0].nsmallest(3, 'change')
    if not losers.empty:
        print("\n PREDICTED LOSERS:")
        for _, row in losers.iterrows():
            print(f"   {row['driver_code']}: P{int(row['grid_position'])} -> "
                  f"P{int(row['predicted_position'])} ({int(row['change'])})")


def main():
    args = parse_args()
    try:
        pred_df, race_name = predict_race(
            race=args.race,
            year=args.year,
            model_path=Path(args.model) if args.model else None,
            stats_path=Path(args.stats) if args.stats else None,
        )
        print_predictions(pred_df, race_name, args.year)
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("Train the model first: python main.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
