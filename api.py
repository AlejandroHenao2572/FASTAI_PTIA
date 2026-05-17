#!/usr/bin/env python
"""
F1 Race Predictor — REST API

FastAPI backend exposing the prediction pipeline as HTTP endpoints.
Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config.settings import Settings
from config.circuits import CIRCUITS, FASTF1_CIRCUIT_MAPPING
from predict import predict_race, find_latest_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Race Predictor API",
    description="Predicts Formula 1 race finishing positions using XGBoost.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = Settings()

# ---------------------------------------------------------------------------
# Curated race list (display name → FastF1 identifier)
# One canonical entry per circuit keyed by circuit_key.
# ---------------------------------------------------------------------------
_CIRCUIT_KEY_TO_FASTF1: dict[str, str] = {}
for fastf1_name, key in FASTF1_CIRCUIT_MAPPING.items():
    if key not in _CIRCUIT_KEY_TO_FASTF1:
        _CIRCUIT_KEY_TO_FASTF1[key] = fastf1_name

RACE_OPTIONS = [
    {
        "display_name": CIRCUITS[key]["country"] + " — " + CIRCUITS[key]["name"],
        "value": _CIRCUIT_KEY_TO_FASTF1.get(key, key),
        "circuit_key": key,
        "city": CIRCUITS[key]["city"],
        "circuit_type": CIRCUITS[key]["circuit_type"],
    }
    for key in CIRCUITS
    if key in _CIRCUIT_KEY_TO_FASTF1
]
RACE_OPTIONS.sort(key=lambda r: r["display_name"])

SUPPORTED_YEARS = [2023, 2024, 2025, 2026]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    race: str
    year: int
    model_path: Optional[str] = None
    stats_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
def get_status():
    """Return whether a trained model and historical stats are available."""
    latest_model = find_latest_model(settings.models_dir)
    stats_path = settings.reports_dir / "historical_stats.json"
    report_path = settings.reports_dir / "training_report.json"

    result = {
        "model_available": latest_model is not None,
        "model_file": latest_model.name if latest_model else None,
        "stats_available": stats_path.exists(),
        "report_available": report_path.exists(),
        "training_metrics": None,
    }

    if report_path.exists():
        import json
        with open(report_path) as f:
            report = json.load(f)
        result["training_metrics"] = {
            "seasons": report["training_info"]["seasons"],
            "train_samples": report["training_info"]["train_samples"],
            "test_samples": report["training_info"]["test_samples"],
            "mae": report["test_metrics"]["mae"],
            "rmse": report["test_metrics"]["rmse"],
            "top3_accuracy": report["test_metrics"]["top3_accuracy"],
            "trained_at": report["training_info"]["timestamp"],
        }

    return result


@app.get("/api/circuits")
def get_circuits():
    """Return the list of available circuits for prediction."""
    return {"circuits": RACE_OPTIONS}


@app.get("/api/years")
def get_years():
    """Return the list of supported prediction years."""
    return {"years": SUPPORTED_YEARS}


@app.post("/api/predict")
def run_prediction(body: PredictRequest):
    """
    Run race prediction for the given race and year.

    Requires:
    - A trained model in models/saved/ (run main.py first)
    - The qualifying session to have already occurred
    """
    if body.year not in SUPPORTED_YEARS:
        raise HTTPException(
            status_code=400,
            detail=f"Year {body.year} not supported. Use one of {SUPPORTED_YEARS}.",
        )

    model_path = Path(body.model_path) if body.model_path else None
    stats_path = Path(body.stats_path) if body.stats_path else None

    try:
        pred_df, race_name = predict_race(
            race=body.race,
            year=body.year,
            model_path=model_path,
            stats_path=stats_path,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    predictions = []
    for _, row in pred_df.iterrows():
        pos = int(row["predicted_position"])
        grid = int(row["grid_position"])
        predictions.append(
            {
                "position": pos,
                "driver_code": row["driver_code"],
                "team": row["team"],
                "grid_position": grid,
                "score": round(float(row["predicted_position_raw"]), 3),
                "position_change": grid - pos,
            }
        )

    pred_copy = pred_df.copy()
    pred_copy["change"] = pred_copy["grid_position"] - pred_copy["predicted_position"]

    gainers = [
        {
            "driver": r["driver_code"],
            "from_pos": int(r["grid_position"]),
            "to_pos": int(r["predicted_position"]),
            "change": int(r["change"]),
        }
        for _, r in pred_copy[pred_copy["change"] > 0]
        .nlargest(3, "change")
        .iterrows()
    ]

    losers = [
        {
            "driver": r["driver_code"],
            "from_pos": int(r["grid_position"]),
            "to_pos": int(r["predicted_position"]),
            "change": int(r["change"]),
        }
        for _, r in pred_copy[pred_copy["change"] < 0]
        .nsmallest(3, "change")
        .iterrows()
    ]

    return {
        "success": True,
        "race_name": race_name,
        "year": body.year,
        "race_input": body.race,
        "predictions": predictions,
        "podium": [p["driver_code"] for p in predictions[:3]],
        "gainers": gainers,
        "losers": losers,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
