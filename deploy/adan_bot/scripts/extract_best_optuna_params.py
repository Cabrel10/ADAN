#!/usr/bin/env python3
"""
Extract best hyperparameters from Optuna study and save to JSON.
Used to transfer optimal params from optimization to training phase.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

import optuna

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_best_params(study_name: str, storage: str = "sqlite:///optuna.db") -> Dict[str, Any]:
    """
    Extract best trial parameters from Optuna study.
    
    Args:
        study_name: Name of the Optuna study
        storage: Storage URL (default: sqlite:///optuna.db)
    
    Returns:
        Dictionary with best parameters
    """
    logger.info(f"Loading study '{study_name}' from {storage}")
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
    except Exception as e:
        logger.error(f"Failed to load study: {e}")
        sys.exit(1)
    
    if not study.trials:
        logger.error("No trials found in study")
        sys.exit(1)
    
    # Get best trial
    best_trial = study.best_trial
    logger.info(f"Best trial: #{best_trial.number}")
    logger.info(f"Best value (Sharpe): {best_trial.value:.4f}")
    
    # Extract parameters
    best_params = best_trial.params
    
    logger.info("Best parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Add metadata
    result = {
        "trial_number": best_trial.number,
        "best_value": float(best_trial.value),
        "total_trials": len(study.trials),
        "study_name": study_name,
        "timestamp": str(best_trial.datetime_complete),
        "parameters": best_params
    }
    
    return result


def save_params(params: Dict[str, Any], output_path: str) -> None:
    """Save parameters to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Parameters saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract best hyperparameters from Optuna study"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="adan_final_v1",
        help="Name of Optuna study (default: adan_final_v1)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna.db",
        help="Storage URL (default: sqlite:///optuna.db)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/best_params_optuna.json",
        help="Output JSON file path (default: config/best_params_optuna.json)"
    )
    
    args = parser.parse_args()
    
    # Extract parameters
    params = extract_best_params(args.study_name, args.storage)
    
    # Save to file
    save_params(params, args.output)
    
    logger.info("✅ Best parameters extracted successfully")
    
    # Print summary
    print("\n" + "="*60)
    print("BEST PARAMETERS SUMMARY")
    print("="*60)
    print(f"Trial: #{params['trial_number']}")
    print(f"Sharpe Ratio: {params['best_value']:.4f}")
    print(f"Total Trials: {params['total_trials']}")
    print(f"Output: {args.output}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
