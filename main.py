#!/usr/bin/env python
"""
F1 Race Predictor - Main Training Pipeline

This script orchestrates the complete ML pipeline:
1. Load data from FastF1 API
2. Engineer features
3. Train XGBoost model with cross-validation
4. Evaluate on test set
5. Generate visualizations and reports
6. Save trained model

Usage:
    python main.py                    # Run full pipeline
    python main.py --seasons 2023 2024  # Specify training seasons
    python main.py --skip-cv          # Skip cross-validation
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from data.data_loader import F1DataLoader
from features.engineering import FeatureEngineer
from models.trainer import F1ModelTrainer
from evaluation.metrics import evaluate_predictions, print_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='F1 Race Predictor - Training Pipeline'
    )
    parser.add_argument(
        '--seasons', 
        nargs='+', 
        type=int, 
        default=[2023, 2024],
        help='Seasons to use for training (default: 2023 2024)'
    )
    parser.add_argument(
        '--test-races', 
        type=int, 
        default=4,
        help='Number of last races to use for testing (default: 4)'
    )
    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip cross-validation'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     F1 RACE PREDICTOR - TRAINING PIPELINE                    ║
    ║                                                               ║
    ║     Model: XGBoost Regressor                                 ║
    ║     Target: Race Finish Position (1-20)                      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize settings
    settings = Settings()
    settings.training_seasons = args.seasons
    settings.test_races_count = args.test_races
    
    logger.info(f"Training seasons: {settings.training_seasons}")
    logger.info(f"Test races: {settings.test_races_count}")
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading Data from FastF1 API")
    print("="*60)
    
    loader = F1DataLoader(settings)
    
    # Load data for all training seasons
    race_data = loader.load_multiple_seasons(settings.training_seasons)
    
    if not race_data:
        logger.error("No data loaded. Check your internet connection and FastF1 cache.")
        sys.exit(1)
    
    # Create training DataFrame
    df = loader.create_training_dataframe(race_data)
    logger.info(f"Loaded {len(df)} samples from {len(race_data)} races")
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)
    
    engineer = FeatureEngineer(settings)
    df = engineer.create_all_features(df)
    
    logger.info(f"Features created: {len(settings.feature_columns)}")
    logger.info(f"Feature columns: {settings.feature_columns}")
    
    # =========================================================================
    # STEP 3: Train/Test Split (Temporal)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Train/Test Split (Temporal)")
    print("="*60)
    
    # Get unique races sorted by date
    df['race_id'] = df['year'].astype(str) + '_' + df['round'].astype(str)
    unique_races = df.groupby('race_id').first().sort_values(['year', 'round']).index.tolist()
    
    # Last N races for testing
    test_races = unique_races[-settings.test_races_count:]
    train_races = unique_races[:-settings.test_races_count]
    
    logger.info(f"Training races: {len(train_races)}")
    logger.info(f"Test races: {len(test_races)}")
    
    # Split data
    train_df = df[df['race_id'].isin(train_races)].copy()
    test_df = df[df['race_id'].isin(test_races)].copy()
    
    # Prepare features and target
    X_train, y_train = engineer.prepare_training_data(train_df)
    X_test, y_test = engineer.prepare_training_data(test_df)
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # =========================================================================
    # STEP 4: Model Training with Cross-Validation
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: Model Training")
    print("="*60)
    
    trainer = F1ModelTrainer(settings)
    
    # Cross-validation
    cv_results = {}
    if not args.skip_cv:
        print("\nRunning Time-Series Cross-Validation...")
        cv_results = trainer.cross_validate(X_train, y_train, n_splits=5)
        
        print(f"\nCV Results:")
        print(f"  Mean MAE: {cv_results['mean_mae']:.3f} ± {cv_results['std_mae']:.3f}")
        print(f"  Mean RMSE: {cv_results['mean_rmse']:.3f} ± {cv_results['std_rmse']:.3f}")
    
    # Train final model on all training data
    print("\nTraining final model...")
    trainer.train(X_train, y_train)
    
    # =========================================================================
    # STEP 5: Evaluation on Test Set
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: Evaluation on Test Set")
    print("="*60)
    
    # Make predictions
    y_pred = trainer.predict(X_test)
    
    # Round predictions to get positions
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_rounded = np.clip(y_pred_rounded, 1, 20)
    
    # Evaluate
    metrics = print_evaluation_report(y_test.values, y_pred)
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # =========================================================================
    # STEP 6: Save Results
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: Saving Results")
    print("="*60)
    
    # Save model
    model_path = trainer.save_model()
    print(f"Model saved to: {model_path}")
    
    # Save metrics report
    report = {
        'training_info': {
            'seasons': settings.training_seasons,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_races': test_races,
            'features': settings.feature_columns,
            'timestamp': datetime.now().isoformat(),
        },
        'cv_results': cv_results,
        'test_metrics': metrics.to_dict(),
        'feature_importance': feature_importance.to_dict('records'),
    }
    
    report_path = settings.reports_dir / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to: {report_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"""
Summary:
  - Model: XGBoost Regressor
  - Training samples: {len(X_train)}
  - Test samples: {len(X_test)}
  - Features: {len(settings.feature_columns)}
  
Test Set Performance:
  - MAE: {metrics.mae:.2f} positions
  - RMSE: {metrics.rmse:.2f} positions
  - Top-3 Accuracy: {metrics.top3_accuracy:.1%}

Files saved:
  - Model: {model_path}
  - Report: {report_path}
    """)
    
    return trainer, metrics


if __name__ == '__main__':
    trainer, metrics = main()
