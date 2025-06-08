#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ADAN System Quick Smoke Test
----------------------------
This script performs an end-to-end smoke test of the ADAN data pipeline and
a short training run using minimal synthetic data. Its purpose is to quickly
verify that the major components of the system integrate correctly after
refactorings, especially concerning data processing and feature scaling.
"""

import os
import sys
import subprocess
import shutil # For cleanup
from pathlib import Path
import pandas as pd
import numpy as np
import yaml # For creating temporary config files
import json # For potentially reading feature order if needed by a step
import logging

# Configure basic logging for the smoke test script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for the smoke test
SMOKE_TEST_DIR_NAME = "temp_smoke_test_data"
SYNTHETIC_ASSETS = ["SYNTH1USDT", "SYNTH2USDT"]
SYNTHETIC_TIMEFRAME = "1m"
SYNTHETIC_PROFILE = "smoke_cpu" # A custom profile for this test
SYNTHETIC_FEATURES_BASE = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'rsi_14'] # OHLCV + 2 indicators
SYNTHETIC_NUM_ROWS = 300 # Number of rows for synthetic data

TEMP_DATA_CONFIG_FILENAME = f"temp_data_config_{SYNTHETIC_PROFILE}.yaml" # More specific temp name
SMOKE_TRAIN_N_STEPS = "64" # String for command line, SB3 default is 2048, make much smaller
SMOKE_MODEL_SUFFIX = "smoke_test"

# Expected output model name from training (using the new timeframe-specific naming)
EXPECTED_MODEL_FILENAME = f"final_model_{SYNTHETIC_TIMEFRAME}_{SMOKE_MODEL_SUFFIX}.zip"

# Constants for training parameters (ensure these are strings for subprocess)
SMOKE_TRAIN_TIMESTEPS = "100"
SMOKE_TRAIN_CAPITAL = "15000"
SMOKE_TRAIN_MAX_EP_STEPS = "50"
SMOKE_TRAIN_SAVE_FREQ = "80"

# Determine project root dynamically
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add pandas_ta import
import pandas_ta as pta

def generate_synthetic_data(base_save_path: Path, assets: list, timeframe_str: str, num_rows: int = 300, features_to_generate: list = None):
    logger.info(f"Generating synthetic data for assets: {assets} at {base_save_path}")
    base_save_path.mkdir(parents=True, exist_ok=True)

    if features_to_generate is None:
        features_to_generate = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'rsi_14']

    for asset_id in assets:
        fixed_end_time_for_test = pd.Timestamp("2023-10-26 12:00:00").floor('min')
        end_time = fixed_end_time_for_test
        start_time = end_time - pd.Timedelta(minutes=num_rows - 1)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')

        df = pd.DataFrame({'timestamp': timestamps})
        np.random.seed(sum(ord(c) for c in asset_id))
        price_drift = np.random.randn(num_rows).cumsum() * 0.1
        price_oscillation = np.sin(np.linspace(0, 10 * np.pi, num_rows)) * 5
        base_price = 100 + price_drift + price_oscillation

        df['open'] = base_price + np.random.uniform(-0.5, 0.5, num_rows)
        df['close'] = base_price + np.random.uniform(-0.5, 0.5, num_rows)
        df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0, 0.5, num_rows)
        df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0, 0.5, num_rows)
        df['volume'] = np.random.uniform(100, 1000, num_rows)

        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        df.loc[df['low'] > df['high'], 'low'] = df['high']

        feature_cols_to_build = [f for f in features_to_generate if f != 'timestamp']
        if 'sma_10' in feature_cols_to_build:
            df['sma_10'] = pta.sma(df['close'], length=10)
        if 'rsi_14' in feature_cols_to_build:
            df['rsi_14'] = pta.rsi(df['close'], length=14)

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)

        columns_to_select_for_file = ['timestamp'] + feature_cols_to_build
        final_columns_in_df = [col for col in columns_to_select_for_file if col in df.columns]

        missing_gen_cols = [col for col in columns_to_select_for_file if col not in df.columns]
        if missing_gen_cols:
            logger.warning(f"For asset {asset_id}, could not generate/find: {missing_gen_cols}. Available: {df.columns.tolist()}")

        df_to_save = df[final_columns_in_df]
        output_file = base_save_path / f"{asset_id}_features.parquet"
        try:
            df_to_save.to_parquet(output_file, index=False)
            logger.info(f"Saved synthetic data for {asset_id} to {output_file} ({len(df_to_save)} rows, {len(df_to_save.columns)} cols)")
        except Exception as e:
            logger.error(f"Failed to save synthetic data for {asset_id} to {output_file}: {e}")
            raise

def create_temporary_configs(data_config_target_path: Path, agent_config_target_path: Path, assets: list,
                                 features_base_names: list, synthetic_data_source_dir: Path,
                                 timeframe: str, num_rows: int, project_root: Path,
                                 temp_smoke_test_dir_name: str): # Renamed from temp_processed_base_dir_name
    logger.info(f"Creating temporary data configuration at: {data_config_target_path}")
    logger.info(f"Creating temporary agent configuration at: {agent_config_target_path}")

    # --- Create Temporary Data Config ---
    fixed_end_time_for_test = pd.Timestamp("2023-10-26 12:00:00").floor('min')
    end_date_for_split = fixed_end_time_for_test
    start_date_for_split = end_date_for_split - pd.Timedelta(minutes=num_rows - 1)

    train_end_date = start_date_for_split + pd.Timedelta(minutes=int(num_rows * 0.5) -1)
    val_start_date = train_end_date + pd.Timedelta(minutes=1)
    val_end_date = val_start_date + pd.Timedelta(minutes=int(num_rows * 0.25) -1)
    test_start_date = val_end_date + pd.Timedelta(minutes=1)
    test_end_date = end_date_for_split

    relative_source_dir = str(synthetic_data_source_dir.relative_to(project_root))
    # This path is relative to the main "data" dir (e.g., data/../temp_smoke_test_data/processed)
    relative_processed_dir = str(Path("..") / temp_smoke_test_dir_name / "processed")
    # This path is relative to project_root for data_loader.py's Path construction
    relative_scalers_dir = str(Path(temp_smoke_test_dir_name) / "scalers_encoders")


    temp_data_config_content = {
        'assets': assets,
        'timeframes_to_process': [timeframe],
        'training_timeframe': timeframe,
        'source_directory': relative_source_dir,
        'base_market_features': features_base_names,
        'indicators_by_timeframe': { timeframe: [], '1h': [], '1d': [] },
        'data_split': {
            timeframe: {
                'train_start_date': start_date_for_split.strftime('%Y-%m-%d %H:%M:%S'),
                'train_end_date': train_end_date.strftime('%Y-%m-%d %H:%M:%S'),
                'validation_start_date': val_start_date.strftime('%Y-%m-%d %H:%M:%S'),
                'validation_end_date': val_end_date.strftime('%Y-%m-%d %H:%M:%S'),
                'test_start_date': test_start_date.strftime('%Y-%m-%d %H:%M:%S'),
                'test_end_date': test_end_date.strftime('%Y-%m-%d %H:%M:%S'),
            }
        },
        'cnn_input_window_size': 10,
        'processed_data_dir': relative_processed_dir,
        'paths': {
            'scalers_encoders_dir': relative_scalers_dir
        },
        'lot_id': "smoke_test_lot" # Use a specific lot_id for smoke test outputs
    }

    try:
        data_config_target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_config_target_path, 'w') as f:
            yaml.dump(temp_data_config_content, f, sort_keys=False)
        logger.info(f"Temporary data configuration saved to {data_config_target_path}")
    except Exception as e:
        logger.error(f"Failed to save temporary data configuration to {data_config_target_path}: {e}")
        raise

    # --- Create Temporary Agent Config ---
    temp_agent_config_data = {
        'agent_name': 'PPO',
        'n_envs': 1,
        'seed': 42,
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': int(SMOKE_TRAIN_N_STEPS), # Use constant, ensure it's int for SB3
            'batch_size': 32,
            'n_epochs': 4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        },
        'policy': {
            'type': 'MultiInputPolicy',
            'kwargs': {
                 'net_arch': dict(pi=[32], vf=[32]),
                 'features_extractor_kwargs': dict(features_dim=32)
            }
        },
        'eval_freq': 50,
        'checkpoint_freq': 50, # Must be multiple of n_steps for SB3 if using EvalCallback with it
        'custom_log_freq_rollouts': 1,
    }
    try:
        agent_config_target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agent_config_target_path, 'w') as f:
            yaml.dump(temp_agent_config_data, f, sort_keys=False)
        logger.info(f"Temporary agent configuration saved to {agent_config_target_path}")
    except Exception as e:
        logger.error(f"Failed to save temporary agent configuration to {agent_config_target_path}: {e}")
        raise

    return data_config_target_path, agent_config_target_path

def run_pipeline_step(command_args: list, step_name: str, cwd: Path = None, env_vars: dict = None) -> bool:
    logger.info(f"🚀 Executing [Step: {step_name}]: {' '.join(map(str, command_args))}")

    command_args = [str(arg) for arg in command_args]

    if command_args[0].lower() == "python":
        command_args[0] = sys.executable

    try:
        current_env = os.environ.copy()
        if env_vars:
            current_env.update(env_vars)

        result = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else str(PROJECT_ROOT),
            env=current_env,
            check=False
        )

        if result.returncode == 0:
            logger.info(f"✅ [Step: {step_name}] completed successfully.")
            return True
        else:
            logger.error(f"🔥 [Step: {step_name}] FAILED with return code {result.returncode}.")
            logger.error(f"Stderr for {step_name}:\n{result.stderr}")
            logger.error(f"Stdout for {step_name}:\n{result.stdout}")
            return False
    except FileNotFoundError as e:
        logger.error(f"🔥 [Step: {step_name}] FAILED. Command not found: {command_args[0]}. Error: {e}")
        logger.error("Ensure the script path is correct and Python executable is valid.")
        return False
    except Exception as e:
        logger.error(f"🔥 [Step: {step_name}] FAILED with an unexpected exception: {e}", exc_info=True)
        return False

def cleanup(temp_data_root_to_delete: Path, temp_config_files_to_delete: list, smoke_model_file_to_delete: Path):
    logger.info(f"Attempting to clean up temporary data directory: {temp_data_root_to_delete}")
    try:
        if temp_data_root_to_delete.exists():
            shutil.rmtree(temp_data_root_to_delete)
            logger.info(f"Successfully removed directory: {temp_data_root_to_delete}")
        else:
            logger.info(f"Temporary data directory not found: {temp_data_root_to_delete}")
    except Exception as e:
        logger.error(f"Error removing temporary data directory {temp_data_root_to_delete}: {e}", exc_info=True)

    for config_file_path_obj in temp_config_files_to_delete:
        if not isinstance(config_file_path_obj, Path):
            config_file_path_obj = Path(config_file_path_obj)

        logger.info(f"Attempting to clean up temporary config file: {config_file_path_obj}")
        try:
            if config_file_path_obj.exists():
                config_file_path_obj.unlink()
                logger.info(f"Successfully removed config file: {config_file_path_obj}")
            else:
                logger.info(f"Temporary config file not found: {config_file_path_obj}")
        except Exception as e:
            logger.error(f"Error removing temporary config file {config_file_path_obj}: {e}", exc_info=True)

    logger.info(f"Attempting to clean up smoke test model file: {smoke_model_file_to_delete}")
    try:
        if smoke_model_file_to_delete.exists():
            smoke_model_file_to_delete.unlink()
            logger.info(f"Successfully removed smoke test model file: {smoke_model_file_to_delete}")
        else:
            logger.info(f"Smoke test model file not found: {smoke_model_file_to_delete}")
    except Exception as e:
        logger.error(f"Error removing smoke test model file {smoke_model_file_to_delete}: {e}", exc_info=True)


def main():
    logger.info("🚀 Starting ADAN System Quick Smoke Test...")

    temp_data_root = PROJECT_ROOT / SMOKE_TEST_DIR_NAME
    temp_new_data_dir = temp_data_root / "new"

    models_output_dir = PROJECT_ROOT / "models" # Standard models dir for output check

    temp_data_config_path = PROJECT_ROOT / "config" / TEMP_DATA_CONFIG_FILENAME
    temp_agent_config_path = PROJECT_ROOT / "config" / f"agent_config_{SYNTHETIC_PROFILE}.yaml"

    temporary_files_to_cleanup = [temp_data_config_path, temp_agent_config_path]
    smoke_model_file_path = models_output_dir / EXPECTED_MODEL_FILENAME


    overall_success = True

    try:
        logger.info(f"Creating temporary directories under: {temp_data_root}")
        temp_new_data_dir.mkdir(parents=True, exist_ok=True)
        # Ensure models directory exists for saving test model
        models_output_dir.mkdir(parents=True, exist_ok=True)

        generate_synthetic_data(temp_new_data_dir, SYNTHETIC_ASSETS, SYNTHETIC_TIMEFRAME,
                                num_rows=SYNTHETIC_NUM_ROWS, features_to_generate=SYNTHETIC_FEATURES_BASE)
        logger.info("Synthetic data generation completed.")

        created_data_cfg_path, created_agent_cfg_path = create_temporary_configs(
            data_config_target_path=temp_data_config_path,
            agent_config_target_path=temp_agent_config_path,
            assets=SYNTHETIC_ASSETS, features_base_names=SYNTHETIC_FEATURES_BASE,
            synthetic_data_source_dir=temp_new_data_dir, timeframe=SYNTHETIC_TIMEFRAME,
            num_rows=SYNTHETIC_NUM_ROWS, project_root=PROJECT_ROOT,
            temp_smoke_test_dir_name=SMOKE_TEST_DIR_NAME
        )
        if not (created_data_cfg_path.exists() and created_agent_cfg_path.exists()):
            raise FileNotFoundError("Temporary data or agent configuration was not created.")
        logger.info(f"Temporary configurations created: {created_data_cfg_path}, {created_agent_cfg_path}")

        cmd_convert = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'convert_real_data.py'),
                       '--data_config', str(temp_data_config_path)]
        if not run_pipeline_step(cmd_convert, "Data Conversion"):
            raise Exception("Data Conversion step failed.")
        logger.info("Data conversion step completed.")

        cmd_merge = [
            sys.executable, str(PROJECT_ROOT / 'scripts' / 'merge_processed_data.py'),
            '--data_config', str(temp_data_config_path), # Use explicit temp data_config
            '--exec_profile', SYNTHETIC_PROFILE, # Still needed for main_config loading within merge script
            '--timeframes', SYNTHETIC_TIMEFRAME,
            '--splits', 'train', 'val', 'test',
            '--training-timeframe', SYNTHETIC_TIMEFRAME
        ]
        if not run_pipeline_step(cmd_merge, "Data Merging"):
            raise Exception("Data Merging step failed.")
        logger.info("Data merging step completed.")

        cmd_train = [
            sys.executable, str(PROJECT_ROOT / 'scripts' / 'train_rl_agent.py'),
            '--data_config', str(temp_data_config_path),      # Use explicit temp data_config
            '--agent_config', str(temp_agent_config_path),   # Use explicit temp agent_config
            '--exec_profile', SYNTHETIC_PROFILE,             # For any other configs it might load by convention
            '--training_timeframe', SYNTHETIC_TIMEFRAME,
            '--total_timesteps', SMOKE_TRAIN_TIMESTEPS,
            '--initial_capital', SMOKE_TRAIN_CAPITAL,
            '--max_episode_steps', SMOKE_TRAIN_MAX_EP_STEPS,
            '--save_freq', SMOKE_TRAIN_SAVE_FREQ,
            '--n_steps', SMOKE_TRAIN_N_STEPS,                 # Pass new n_steps
            '--model_name_suffix', SMOKE_MODEL_SUFFIX,        # Pass new suffix
            '--logging_config', str(PROJECT_ROOT / 'config' / 'logging_config.yaml')
        ]
        if not run_pipeline_step(cmd_train, "Agent Training"):
            logger.error("Agent training script finished with errors or non-zero exit code.")
            overall_success = False
        else:
            logger.info("Agent training script execution completed.")

        if overall_success:
            logger.info("🔬 Performing model output verifications...")
            if smoke_model_file_path.exists() and smoke_model_file_path.is_file():
                logger.info(f"✅ Verification PASSED: Expected model file '{smoke_model_file_path}' found.")
                model_size_kb = smoke_model_file_path.stat().st_size / 1024
                if model_size_kb > 0.5: # Check if > 0.5KB for a tiny model
                    logger.info(f"✅ Model file size is {model_size_kb:.2f} KB.")
                else:
                    logger.warning(f"⚠️ Verification WARNING: Model file size is very small ({model_size_kb:.2f} KB).")
            else:
                logger.error(f"🔥 Verification FAILED: Expected model file '{smoke_model_file_path}' NOT found.")
                overall_success = False
        else:
             logger.warning("Skipping model verification because a previous step indicated failure.")

    except Exception as e:
        logger.error(f"💥 Smoke test execution failed: {e}", exc_info=False)
        overall_success = False

    finally:
        logger.info("🧹 Performing cleanup...")
        cleanup(temp_data_root, temporary_files_to_cleanup, smoke_model_file_path)

        if overall_success:
            logger.info("=====================================")
            logger.info("🎉 SMOKE TEST: ALL STEPS PASSED 🎉")
            logger.info("=====================================")
        else:
            logger.error("====================================")
            logger.error("🔥 SMOKE TEST: FAILED 🔥")
            logger.error("Please review logs above for details on the failed step(s) or verifications.")
            logger.error("====================================")

    if not overall_success:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
