#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to merge processed data from multiple timeframes into a single
multi-timeframe dataset for each asset and perform train/val/test split.
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
import pandas as pd
import yaml

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.adan_trading_bot.common.utils import load_config

def validate_timeframe_synchronization(dfs: dict, asset_id: str) -> dict:
    """
    Validate timeframe synchronization according to design specifications.
    
    Args:
        dfs: Dictionary of DataFrames by timeframe
        asset_id: Asset identifier for logging
        
    Returns:
        Dictionary containing synchronization validation results
    """
    sync_report = {
        'synchronized': True,
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    try:
        if not dfs:
            sync_report['synchronized'] = False
            sync_report['issues'].append(f"{asset_id}: No timeframe data available")
            return sync_report
        
        # Check data availability for each timeframe
        timeframe_info = {}
        for tf, df in dfs.items():
            if df is None or df.empty:
                sync_report['issues'].append(f"{asset_id} {tf}: Empty DataFrame")
                continue
                
            timeframe_info[tf] = {
                'start_time': df.index.min(),
                'end_time': df.index.max(),
                'row_count': len(df),
                'time_range': df.index.max() - df.index.min()
            }
        
        if len(timeframe_info) < 2:
            sync_report['synchronized'] = False
            sync_report['issues'].append(f"{asset_id}: Insufficient timeframes for synchronization")
            return sync_report
        
        # Check time range alignment
        start_times = [info['start_time'] for info in timeframe_info.values()]
        end_times = [info['end_time'] for info in timeframe_info.values()]
        
        earliest_start = min(start_times)
        latest_start = max(start_times)
        earliest_end = min(end_times)
        latest_end = max(end_times)
        
        # Check for significant misalignment
        start_diff = (latest_start - earliest_start).total_seconds() / 3600  # hours
        end_diff = (latest_end - earliest_end).total_seconds() / 3600  # hours
        
        if start_diff > 24:  # More than 24 hours difference
            sync_report['warnings'].append(f"{asset_id}: Large start time difference ({start_diff:.1f} hours)")
        
        if end_diff > 24:  # More than 24 hours difference
            sync_report['warnings'].append(f"{asset_id}: Large end time difference ({end_diff:.1f} hours)")
        
        # Check data density consistency
        base_tf = list(timeframe_info.keys())[0]
        base_density = timeframe_info[base_tf]['row_count'] / timeframe_info[base_tf]['time_range'].total_seconds()
        
        for tf, info in timeframe_info.items():
            if tf != base_tf:
                tf_density = info['row_count'] / info['time_range'].total_seconds()
                density_ratio = tf_density / base_density if base_density > 0 else 0
                
                if density_ratio < 0.1 or density_ratio > 10:  # Significant density mismatch
                    sync_report['warnings'].append(f"{asset_id}: Data density mismatch between {base_tf} and {tf}")
        
        sync_report['metrics'] = {
            'timeframes_count': len(timeframe_info),
            'start_time_diff_hours': start_diff,
            'end_time_diff_hours': end_diff,
            'timeframe_info': timeframe_info
        }
        
        # Determine overall synchronization status
        if len(sync_report['issues']) > 0:
            sync_report['synchronized'] = False
        elif len(sync_report['warnings']) > 2:
            sync_report['synchronized'] = False
            sync_report['issues'].append(f"{asset_id}: Too many synchronization warnings")
        
    except Exception as e:
        sync_report['synchronized'] = False
        sync_report['issues'].append(f"{asset_id}: Synchronization validation error - {str(e)}")
        logger.error(f"Error validating synchronization for {asset_id}: {e}")
    
    return sync_report

def apply_forward_fill_with_validation(merged_df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    """
    Apply forward-fill with validation according to design specifications.
    
    Args:
        merged_df: Merged DataFrame to apply forward-fill
        asset_id: Asset identifier for logging
        
    Returns:
        DataFrame with forward-fill applied and validation
    """
    logger.info(f"Applying forward-fill with validation for {asset_id}")
    
    # Record initial state
    initial_shape = merged_df.shape
    initial_na_count = merged_df.isna().sum().sum()
    
    # Apply forward-fill
    filled_df = merged_df.ffill()
    
    # Validate forward-fill results
    final_na_count = filled_df.isna().sum().sum()
    fill_effectiveness = ((initial_na_count - final_na_count) / initial_na_count * 100) if initial_na_count > 0 else 100
    
    logger.info(f"{asset_id}: Forward-fill effectiveness: {fill_effectiveness:.1f}% "
               f"({initial_na_count} -> {final_na_count} NAs)")
    
    # Check for remaining NAs
    if final_na_count > 0:
        na_columns = filled_df.columns[filled_df.isna().any()].tolist()
        logger.warning(f"{asset_id}: {final_na_count} NAs remaining in columns: {na_columns[:5]}...")
        
        # Apply backward fill for remaining NAs
        filled_df = filled_df.bfill()
        final_final_na_count = filled_df.isna().sum().sum()
        
        if final_final_na_count > 0:
            logger.warning(f"{asset_id}: {final_final_na_count} NAs still remaining after backward fill")
    
    # Add freshness indicators (minutes_since_update)
    filled_df = add_freshness_indicators(filled_df, asset_id)
    
    return filled_df

def add_freshness_indicators(df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    """
    Add freshness indicators (minutes_since_update) according to design specifications.
    
    Args:
        df: DataFrame to add freshness indicators to
        asset_id: Asset identifier for logging
        
    Returns:
        DataFrame with freshness indicators added
    """
    logger.debug(f"Adding freshness indicators for {asset_id}")
    
    # Extract timeframes from column names
    timeframes = set()
    for col in df.columns:
        if '_' in col:
            tf = col.split('_')[0]
            if tf in ['5m', '1h', '4h']:
                timeframes.add(tf)
    
    # Add minutes_since_update for each timeframe
    for tf in timeframes:
        freshness_col = f"{tf}_minutes_since_update"
        
        # Find a representative column for this timeframe
        tf_columns = [col for col in df.columns if col.startswith(f"{tf}_")]
        if not tf_columns:
            continue
        
        # Use the first available column to detect updates
        ref_col = tf_columns[0]
        
        # Calculate minutes since last update
        # This is a simplified approach - in reality, this would be calculated during data ingestion
        df[freshness_col] = 0  # Initialize with 0 (fresh data)
        
        # For demonstration, add some realistic freshness values
        # In production, this would be calculated based on actual data timestamps
        if tf == '5m':
            df[freshness_col] = 0  # Always fresh for base timeframe
        elif tf == '1h':
            df[freshness_col] = (df.index.minute % 60)  # Minutes since last hour
        elif tf == '4h':
            df[freshness_col] = ((df.index.hour % 4) * 60 + df.index.minute)  # Minutes since last 4h
    
    return df

def process_asset(asset_id: str, config: dict) -> None:
    """
    Charge, fusionne, et sauvegarde les données pour un seul actif avec synchronisation améliorée.
    """
    logger.info(f"--- Traitement de l'actif : {asset_id} ---")
    
    timeframes = config.get('timeframes_for_observation', ['5m', '1h', '4h'])
    processed_dir = Path(config['data_pipeline']['processed_data_dir'])
    final_dir = Path(config['data_pipeline'].get('final_data_dir', 'data/final'))

    dfs = {}
    for tf in timeframes:
        # Essayer d'abord avec le format {ASSET}USDT.parquet dans le dossier du timeframe
        file_path = processed_dir / tf / f"{asset_id}USDT.parquet"
        if not file_path.exists():
            # Essayer avec le format {ASSET}_{TIMEFRAME}.parquet
            file_path = processed_dir / tf / f"{asset_id}_{tf}.parquet"
            
        if file_path.exists():
            logger.info(f"Chargement de {file_path}")
            try:
                df = pd.read_parquet(file_path)
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif not isinstance(df.index, pd.DatetimeIndex):
                    logger.error(f"No valid timestamp index found in {file_path}")
                    continue
                    
                # Validate data before processing
                if df.empty:
                    logger.warning(f"Empty DataFrame loaded from {file_path}")
                    continue
                    
                df = df.add_prefix(f"{tf}_")
                dfs[tf] = df
                logger.info(f"Successfully loaded {tf} data: {df.shape}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        else:
            logger.warning(f"Fichier manquant pour {asset_id} - {tf}, il sera ignoré.")

    if not dfs:
        logger.error(f"Aucune donnée de timeframe trouvée pour {asset_id}. Abandon.")
        return

    # Validate timeframe synchronization according to design specifications
    sync_report = validate_timeframe_synchronization(dfs, asset_id)
    
    if not sync_report['synchronized']:
        logger.error(f"Synchronization validation failed for {asset_id}:")
        for issue in sync_report['issues']:
            logger.error(f"  - {issue}")
        return
    
    if sync_report['warnings']:
        logger.warning(f"Synchronization warnings for {asset_id}:")
        for warning in sync_report['warnings']:
            logger.warning(f"  - {warning}")

    base_tf = timeframes[0]
    if base_tf not in dfs:
        logger.error(f"Le timeframe de base '{base_tf}' n'a pas pu être chargé pour {asset_id}. Abandon.")
        return
    
    logger.info(f"Starting merge process for {asset_id} with {len(dfs)} timeframes")
    merged_df = dfs[base_tf].copy()
    
    for tf in timeframes[1:]:
        if tf in dfs:
            df_to_join = dfs[tf]
            logger.debug(f"Joining {tf} data ({df_to_join.shape}) to merged data ({merged_df.shape})")
            # La jointure externe crée des lignes pour tous les timestamps uniques
            merged_df = merged_df.join(df_to_join, how='outer')

    # Apply forward-fill with validation according to design specifications
    merged_df = apply_forward_fill_with_validation(merged_df, asset_id)
    
    # Conserver uniquement les lignes qui existent dans le timeframe de base
    merged_df = merged_df.loc[dfs[base_tf].index]
    
    # Nettoyage final des NaNs qui pourraient persister au début
    initial_rows = len(merged_df)
    merged_df.dropna(inplace=True)
    final_rows = len(merged_df)
    
    if initial_rows != final_rows:
        logger.info(f"{asset_id}: Dropped {initial_rows - final_rows} rows with remaining NAs")
    
    if merged_df.empty:
        logger.error(f"La fusion a produit un DataFrame vide pour {asset_id}.")
        return

    logger.info(f"Fusion réussie pour {asset_id}. Shape finale: {merged_df.shape}")

    # Création de la variable cible
    future_periods = config.get('target_config', {}).get('future_periods', 6)
    close_col = f"{base_tf}_close"
    merged_df['target_return'] = merged_df[close_col].pct_change(periods=future_periods).shift(-future_periods)
    
    merged_df.dropna(subset=['target_return'], inplace=True)

    # Split des données
    split_config = config['data_split']
    train_end = pd.to_datetime(split_config['train_end_date'])
    val_end = pd.to_datetime(split_config['validation_end_date'])

    train_df = merged_df[merged_df.index <= train_end]
    val_df = merged_df[(merged_df.index > train_end) & (merged_df.index <= val_end)]
    test_df = merged_df[merged_df.index > val_end]

    logger.info(f"Split pour {asset_id}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Sauvegarde
    asset_final_dir = final_dir / asset_id
    asset_final_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(asset_final_dir / "train.parquet")
    val_df.to_parquet(asset_final_dir / "val.parquet")
    test_df.to_parquet(asset_final_dir / "test.parquet")
    
    logger.info(f"Données finales pour {asset_id} sauvegardées dans {asset_final_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fusionne les données multi-timeframe pour ADAN.")
    parser.add_argument('--config', type=str, default='config/data_config.yaml', help='Chemin vers le fichier de configuration des données.')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if not config:
            raise FileNotFoundError(f"Fichier de configuration non trouvé ou vide : {args.config}")

        # Vérifier que les timeframes sont définis
        if 'timeframes_for_observation' not in config:
            config['timeframes_for_observation'] = ['5m', '1h', '4h']
            logger.warning("timeframes_for_observation non défini dans la config, utilisation des valeurs par défaut: %s", 
                         config['timeframes_for_observation'])

        for asset in config['assets']:
            process_asset(asset, config)
            gc.collect()

        logger.info("Pipeline de fusion terminé avec succès !")

    except Exception as e:
        logger.error(f"Une erreur critique est survenue dans le pipeline de fusion : {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
