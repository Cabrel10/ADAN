#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour convertir les données réelles de data/new vers le format pipeline ADAN multi-timeframe.
Supporte la génération de données pour 1m (pré-calculées), 1h et 1d (calculées dynamiquement).
Convertit du format "long" vers le format "wide" avec gestion multi-timeframe.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.data_processing.feature_engineer import add_technical_indicators

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resample_data(df, target_timeframe):
    """
    Ré-échantillonne les données OHLCV de 1m vers 1h ou 1d.
    
    Args:
        df: DataFrame avec colonnes OHLCV et index timestamp
        target_timeframe: '1h' ou '1d'
        
    Returns:
        pd.DataFrame: Données ré-échantillonnées
    """
    logger.info(f"🔄 Ré-échantillonnage vers {target_timeframe}")
    
    # Assurer que l'index est datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Définir la fréquence de ré-échantillonnage
    freq = '1h' if target_timeframe == '1h' else '1D'
    
    # Ré-échantillonner les données OHLCV
    ohlcv_resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    logger.info(f"✅ Ré-échantillonnage: {len(df)} → {len(ohlcv_resampled)} barres")
    return ohlcv_resampled

def process_asset_for_timeframe(asset, timeframe, config):
    """
    Traite un actif pour un timeframe spécifique.
    
    Args:
        asset: Nom de l'actif (ex: 'ADAUSDT')
        timeframe: '1m', '1h', ou '1d'
        config: Configuration complète
        
    Returns:
        pd.DataFrame: Données traitées pour cet actif
    """
    logger.info(f"📊 Traitement {asset} pour {timeframe}")
    
    # Charger les données 1m de base
    source_file = f"data/new/{asset}_features.parquet"
    if not os.path.exists(source_file):
        logger.error(f"❌ Fichier source manquant: {source_file}")
        return None
    
    df = pd.read_parquet(source_file)
    logger.info(f"📈 Données chargées: {df.shape}")
    
    # Assurer que timestamp est l'index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Supprimer les colonnes non numériques
    columns_to_remove = ['symbol']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"🗑️ Colonne {col} supprimée")
    
    if timeframe == '1m':
        # Pour 1m, utiliser les features spécifiées dans base_market_features
        base_features = config.get('data', {}).get('base_market_features', [])
        if not base_features:
            logger.warning("⚠️ 'base_market_features' non défini dans la config. Utilisation de toutes les colonnes.")
            selected_df = df
        else:
            # S'assurer que les colonnes OHLCV de base sont incluses si elles ne sont pas déjà dans base_features
            ohlcv_base = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_base:
                if col not in base_features and col in df.columns:
                    base_features.append(col)

            missing_cols = [col for col in base_features if col not in df.columns]
            if missing_cols:
                logger.warning(f"⚠️ Colonnes manquantes dans le df pour 1m: {missing_cols}. Elles seront ignorées.")

            final_features_1m = [col for col in base_features if col in df.columns]
            selected_df = df[final_features_1m]
            logger.info(f"✅ Sélection des {len(final_features_1m)} features pré-calculées pour 1m: {final_features_1m}")
        return selected_df
    
    else:
        # Pour 1h/1d, ré-échantillonner et recalculer les indicateurs
        logger.info(f"🔄 Ré-échantillonnage et recalcul des indicateurs ({timeframe})")
        
        # Extraire seulement OHLCV pour le ré-échantillonnage
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        available_ohlcv = [col for col in ohlcv_cols if col in df.columns]
        
        if not available_ohlcv:
            logger.error(f"❌ Colonnes OHLCV manquantes pour {asset}")
            return None
        
        df_ohlcv = df[available_ohlcv].copy()
        
        # Ré-échantillonner
        df_resampled = resample_data(df_ohlcv, timeframe)
        
        # Ajouter les indicateurs techniques pour ce timeframe
        indicators_config = config.get('indicators_by_timeframe', {}).get(timeframe, [])
        
        if indicators_config:
            logger.info(f"📊 Calcul de {len(indicators_config)} indicateurs pour {timeframe}")
            df_with_indicators, added_features = add_technical_indicators(df_resampled, indicators_config, timeframe)
            return df_with_indicators
        else:
            logger.warning(f"⚠️ Aucun indicateur configuré pour {timeframe}")
            return df_resampled

def split_data_by_timeframe(df, config, timeframe):
    """
    Divise les données selon les configurations de split pour le timeframe.
    
    Args:
        df: DataFrame à diviser
        config: Configuration complète
        timeframe: Timeframe concerné
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info(f"✂️ Division des données pour {timeframe}")
    
    data_split = config.get('data', {}).get('data_split', {})
    timeframe_split = data_split.get(timeframe, {})
    
    if not timeframe_split:
        logger.warning(f"⚠️ Pas de split configuré pour {timeframe}, utilisation des pourcentages par défaut")
        # Split par pourcentages
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.2)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
    else:
        # Split par dates
        train_start = pd.to_datetime(timeframe_split.get('train_start_date'))
        train_end = pd.to_datetime(timeframe_split.get('train_end_date'))
        val_start = pd.to_datetime(timeframe_split.get('validation_start_date'))
        val_end = pd.to_datetime(timeframe_split.get('validation_end_date'))
        test_start = pd.to_datetime(timeframe_split.get('test_start_date'))
        test_end = pd.to_datetime(timeframe_split.get('test_end_date'))
        
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        val_df = df[(df.index >= val_start) & (df.index <= val_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
    
    logger.info(f"📊 Split résultats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def save_asset_data_split(df_split, split_name, asset, timeframe, asset_data_dir):
    """
    Sauvegarde un split de données (train, val, ou test) pour un actif et timeframe spécifique.
    
    Args:
        df_split: DataFrame du split de données à sauvegarder.
        split_name: Nom du split ("train", "val", "test").
        asset: Nom de l'actif (ex: 'BTCUSDT').
        timeframe: Timeframe traité (ex: '1h').
        asset_data_dir: Chemin du répertoire où sauvegarder le fichier (Path object).
                       Ex: data/processed/unified/BTCUSDT/
    """
    output_file = asset_data_dir / f"{asset}_{timeframe}_{split_name}.parquet"
    try:
        df_split.to_parquet(output_file)
        logger.info(f"✅ Data split saved: {output_file} ({df_split.shape})")
    except Exception as e:
        logger.error(f"❌ Failed to save data split {output_file}: {e}")

def process_unified_pipeline(config, exec_profile):
    """
    Pipeline unifié pour traiter tous les timeframes.
    
    Args:
        config: Configuration complète
        exec_profile: Profil d'exécution (cpu/gpu)
    """
    logger.info("🚀 Démarrage du pipeline unifié multi-timeframe")
    
    # Récupérer la configuration
    assets = config.get('assets', [])
    timeframes_to_process = config.get('timeframes_to_process', ['1m'])
    
    logger.info(f"📊 Assets: {assets}")
    # logger.info(f"📊 Assets: {assets}") # Duplicate log
    logger.info(f"⏰ Timeframes to process: {timeframes_to_process}")

    for timeframe in timeframes_to_process:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Timeframe: {timeframe}")
        logger.info(f"{'='*70}")

        # Create base directories for the current timeframe
        timeframe_unified_data_dir = Path("data/processed/unified") / timeframe
        timeframe_scalers_dir = Path("data/scalers_encoders") / timeframe

        timeframe_unified_data_dir.mkdir(parents=True, exist_ok=True)
        timeframe_scalers_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Base data directory for {timeframe}: {timeframe_unified_data_dir}")
        logger.info(f"Base scalers directory for {timeframe}: {timeframe_scalers_dir}")

        for asset in assets:
            logger.info(f"\n--- Processing Asset: {asset} for Timeframe: {timeframe} ---")

            # Create specific directories for the asset within the timeframe directory
            asset_specific_data_dir = timeframe_unified_data_dir / asset
            asset_specific_scaler_dir = timeframe_scalers_dir / asset
            asset_specific_data_dir.mkdir(parents=True, exist_ok=True)
            asset_specific_scaler_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Asset data save path: {asset_specific_data_dir}")
            logger.debug(f"Asset scaler save path: {asset_specific_scaler_dir}")

            try:
                asset_df = process_asset_for_timeframe(asset, timeframe, config)
                if asset_df is None:
                    logger.error(f"❌ No data processed for {asset} at {timeframe}. Skipping.")
                    continue

                train_df, val_df, test_df = split_data_by_timeframe(asset_df, config, timeframe)

                if train_df.empty or val_df.empty or test_df.empty:
                    logger.warning(f"⚠️ Data splitting for {asset} - {timeframe} resulted in one or more empty dataframes. Skipping normalization and saving.")
                    logger.warning(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
                    continue

                # Normalization is removed from this script. Saving raw (but feature-engineered) data.
                # train_norm, val_norm, test_norm, scaler = normalize_features(train_df, val_df, test_df, config)

                logger.info(f"Skipping normalization for {asset} - {timeframe}. Saving data as is after feature engineering.")

                # Pass the new asset_specific_data_dir
                # Saving unnormalized data directly
                save_asset_data_split(train_df, "train", asset, timeframe, asset_specific_data_dir)
                save_asset_data_split(val_df, "val", asset, timeframe, asset_specific_data_dir)
                save_asset_data_split(test_df, "test", asset, timeframe, asset_specific_data_dir)

                # Scaler saving is removed
                # save_scaler(scaler, asset, timeframe, asset_specific_scaler_dir)

                logger.info(f"✅ Successfully processed and saved (unnormalized) {asset} for {timeframe}")

            except Exception as e:
                logger.error(f"❌ Error processing {asset} for {timeframe}: {e}", exc_info=True)
        
        logger.info(f"✅ Timeframe {timeframe} processed successfully for all assets!")

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(description="Pipeline unifié de traitement des données ADAN")
    parser.add_argument("--exec_profile", type=str, default="cpu", 
                       choices=["cpu", "gpu"], help="Profil d'exécution (utilisé si --data_config n'est pas fourni)")
    parser.add_argument('--data_config', type=str, default=None,
                        help='Path to a specific data_config YAML file. Overrides --exec_profile for data config loading.')
    
    args = parser.parse_args()
    
    try:
        # Charger la configuration
        data_config_to_load = None
        if args.data_config:
            data_config_to_load = args.data_config
            logger.info(f"Using explicit data_config from: {data_config_to_load}")
        else:
            data_config_to_load = f"config/data_config_{args.exec_profile}.yaml"
            logger.info(f"Using data_config derived from exec_profile '{args.exec_profile}': {data_config_to_load}")

        try:
            config = load_config(data_config_to_load) # load_config should return the dict
            if config is None: # load_config might return None on error
                raise FileNotFoundError(f"Configuration file {data_config_to_load} not found or empty.")
            logger.info(f"Data configuration loaded successfully from {data_config_to_load}")
        except FileNotFoundError:
            logger.error(f"FATAL: Data configuration file not found at {data_config_to_load}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"FATAL: Error loading data configuration from {data_config_to_load}: {e}")
            sys.exit(1)
        
        # Exécuter le pipeline unifié
        process_unified_pipeline(config, args.exec_profile)
        
        logger.info("🎉 Pipeline unifié terminé avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {e}")
        raise

if __name__ == "__main__":
    main()