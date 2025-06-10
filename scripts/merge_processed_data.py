#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
import sys
import gc
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.adan_trading_bot.common.utils import get_logger, load_config, ensure_dir_exists

logger = get_logger(__name__)

def load_configs(exec_profile='cpu'):
    logger.info(f"Chargement des configurations avec le profil d'exécution: {exec_profile}")
    main_config_path = 'config/main_config.yaml'
    data_config_path = f'config/data_config_{exec_profile}.yaml'
    main_config = load_config(main_config_path)
    data_config = load_config(data_config_path)
    return main_config, data_config

def get_processed_data_paths(main_config, data_config):
    project_dir = main_config.get('paths', {}).get('base_project_dir_local', os.getcwd())
    data_dir = os.path.join(project_dir, main_config.get('paths', {}).get('data_dir_name', 'data'))
    base_processed_dir = os.path.join(data_dir, data_config.get('data', {}).get('processed_data_dir', 'processed')) # Ensure 'data' key
    
    lot_id = data_config.get('data', {}).get('lot_id', None) # Ensure 'data' key
    unified_segment = 'unified'

    processed_dir = os.path.join(base_processed_dir, lot_id) if lot_id else base_processed_dir
    logger.info(f"Répertoire de base pour les données par actif (avant unified/tf/asset): {processed_dir}")

    merged_dir_base = os.path.join(base_processed_dir, 'merged')
    merged_dir = os.path.join(merged_dir_base, lot_id, unified_segment) if lot_id else os.path.join(merged_dir_base, unified_segment)
    logger.info(f"Répertoire cible pour les données fusionnées: {merged_dir}")
    ensure_dir_exists(merged_dir)
    
    assets = data_config.get('data', {}).get('assets', []) # Ensure 'data' key
    timeframes = data_config.get('data', {}).get('timeframes_to_process', data_config.get('data', {}).get('timeframes', [])) # Ensure 'data' key
    if isinstance(timeframes, dict): timeframes = list(timeframes.keys())
    
    logger.info(f"Actifs disponibles: {assets}")
    logger.info(f"Timeframes disponibles: {timeframes}")
    return processed_dir, merged_dir, assets, timeframes

def merge_data_for_timeframe_split(processed_dir, merged_dir, assets, timeframe, split):
    logger.info(f"Fusion des données pour le timeframe {timeframe}, split {split}...")
    dfs_by_asset = {}
    ohlcv_base_columns = ['open', 'high', 'low', 'close', 'volume'] # Define base OHLCV names

    for asset in assets:
        file_path = Path(processed_dir) / "unified" / timeframe / asset / f"{asset}_{timeframe}_{split}.parquet"
        if not file_path.exists():
            logger.warning(f"Fichier {file_path} non trouvé. L'actif {asset} sera ignoré pour {timeframe}_{split}.")
            continue
        try:
            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # MODIFIED RENAMING LOGIC
            new_columns = {}
            for col_name in df.columns:
                if col_name in ohlcv_base_columns:
                    new_columns[col_name] = f"{col_name}_{asset}"
                else: # It's an indicator
                    new_columns[col_name] = f"{col_name}_{timeframe}_{asset}" # Add timeframe suffix
            df = df.rename(columns=new_columns)
            dfs_by_asset[asset] = df
            logger.info(f"Données préparées pour {asset}: {len(df)} lignes, {len(df.columns)} colonnes. Cols: {list(df.columns[:5])}...")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données pour {asset} ({timeframe}_{split}): {e}\n{traceback.format_exc()}")
            continue
    
    if not dfs_by_asset:
        logger.error(f"Aucune donnée n'a pu être chargée pour {timeframe}_{split}.")
        return False
    
    try:
        merged_df = None
        sorted_assets = sorted(dfs_by_asset.keys())
        for i, asset in enumerate(sorted_assets):
            df_asset = dfs_by_asset[asset]
            if merged_df is None:
                merged_df = df_asset.copy()
            else:
                merged_df = pd.merge(merged_df, df_asset, left_index=True, right_index=True, how='outer', suffixes=(None, f'_dup_{asset}')) # Added suffix handling for safety
            del dfs_by_asset[asset]; gc.collect()
        
        if merged_df is not None and not merged_df.empty:
            merged_df = merged_df.ffill().bfill() # Fill with np.nan then ffill/bfill
            # Only drop rows if all values are NaN AFTER ffill/bfill, which means they were NaN across all assets for that timestamp
            # However, the user asked for dropna() which would remove rows with any NaN.
            # Let's stick to ffill/bfill first, and then drop rows where ALL values are NaN.
            # A more aggressive dropna() might remove too much if some assets have shorter history.
            # For now, let's follow the existing ffill/bfill then dropna() but be mindful.
            nan_count_before_dropna = merged_df.isna().sum().sum()
            if nan_count_before_dropna > 0:
                 logger.info(f"NaN count before dropna for {timeframe}_{split}: {nan_count_before_dropna}")
            merged_df = merged_df.dropna() # This was in original, potentially aggressive.
            logger.info(f"Shape after dropna for {timeframe}_{split}: {merged_df.shape}")


            output_file = Path(merged_dir) / f"{timeframe}_{split}_merged.parquet"
            merged_df.to_parquet(output_file)
            logger.info(f"Données fusionnées sauvegardées: {output_file} ({len(merged_df)} lignes, {len(merged_df.columns)} colonnes)")
            return True
        else:
            logger.error(f"Aucune donnée fusionnée valide pour {timeframe}_{split}.")
            return False
    except Exception as e:
        logger.error(f"Erreur lors de la fusion des données pour {timeframe}_{split}: {e}\n{traceback.format_exc()}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fusion des données traitées.')
    parser.add_argument('--exec_profile', type=str, default='cpu', help="Profil d'exécution.")
    parser.add_argument('--timeframes', nargs='+', help='Timeframes à traiter.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='Splits à traiter.')
    parser.add_argument('--training-timeframe', type=str, help='Timeframe principal pour entraînement.')
    parser.add_argument('--data_config', type=str, default=None, help='Path to data_config YAML.')
    args = parser.parse_args()

    main_config_path = 'config/main_config.yaml'
    main_config = load_config(main_config_path)
    if main_config is None: sys.exit(f"FATAL: Main config not found: {main_config_path}")

    data_config_path = args.data_config if args.data_config else f'config/data_config_{args.exec_profile}.yaml'
    data_config = load_config(data_config_path)
    if data_config is None: sys.exit(f"FATAL: Data config not found: {data_config_path}")

    processed_dir, merged_dir, assets, available_timeframes = get_processed_data_paths(main_config, data_config)
    timeframes_to_run = args.timeframes if args.timeframes else available_timeframes
    
    # Ensure assets list is from data_config.data.assets
    assets_from_cfg = data_config.get('data',{}).get('assets',[])
    if not assets_from_cfg:
        logger.error("FATAL: No assets found in data_config under 'data.assets'. Exiting.")
        sys.exit(1) # Actual exit here is fine as it's a fatal config error for the script's logic
    
    success_count = 0; total_count = 0
    for timeframe in timeframes_to_run:
        for split in args.splits:
            total_count += 1
            # Pass assets_from_cfg instead of assets from get_processed_data_paths if it's more reliable
            if merge_data_for_timeframe_split(processed_dir, merged_dir, assets_from_cfg, timeframe, split):
                success_count += 1
    logger.info(f"Fusion des données terminée. {success_count}/{total_count} fusions réussies.")

if __name__ == "__main__":
    main()
