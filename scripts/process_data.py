#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour traiter les données brutes et générer le dataset final avec les indicateurs techniques.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import yaml
import logging
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from src.adan_trading_bot.common.utils import load_config, get_path
from src.adan_trading_bot.data_processing.feature_engineer import FeatureEngineer, add_technical_indicators
import gc
from sklearn.preprocessing import StandardScaler

# Configuration du logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_features(df, scaler=None, scaler_path=None, fit=True, numeric_cols=None):
    """
    Normalise les features spécifiées du DataFrame.
    
    Args:
        df: DataFrame à normaliser
        scaler: Scaler à utiliser (si None, un nouveau sera créé)
        scaler_path: Chemin pour charger/sauvegarder le scaler
        fit: Si True, ajuste le scaler, sinon utilise un scaler existant
        numeric_cols: Liste des noms de colonnes numériques à normaliser
        
    Returns:
        Tuple[DataFrame, StandardScaler]: DataFrame normalisé et scaler utilisé
    """
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    if not numeric_cols:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        logger.warning(f"Aucune colonne numérique spécifiée, utilisation de toutes les colonnes numériques: {numeric_cols}")
    
    if not numeric_cols:
        logger.warning("Aucune colonne numérique à normaliser.")
        return df, None
    
    # Vérifier que les colonnes existent
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes non trouvées pour la normalisation: {missing_cols}")
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        if not numeric_cols:
            return df, None
    
    # Créer une copie du DataFrame pour éviter les modifications inattendues
    df_normalized = df.copy()
    
    # Initialiser ou utiliser le scaler fourni
    if scaler is None and fit:
        scaler = StandardScaler()
        logger.info("Création d'un nouveau StandardScaler")
    
    # Normaliser les données
    if fit:
        logger.info(f"Ajustement du scaler sur {len(numeric_cols)} colonnes numériques")
        df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Sauvegarder le scaler s'il y a un chemin spécifié
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler sauvegardé dans {scaler_path}")
    else:
        if scaler is None:
            if not scaler_path or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Fichier de scaler non trouvé et aucun scaler fourni: {scaler_path}")
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler chargé depuis {scaler_path}")
        
        logger.info(f"Application du scaler sur {len(numeric_cols)} colonnes numériques")
        df_normalized[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df_normalized, scaler

def normalize_features_chunked(df, features_to_normalize, scaler_path, chunk_size, fit=True):
    """
    Normalise les features par chunks pour éviter les problèmes de mémoire.
    """
    from sklearn.preprocessing import StandardScaler
    
    if not features_to_normalize:
        return df, None
    
    # Créer le scaler
    scaler = StandardScaler()
    df_normalized = df.copy()
    
    if fit:
        logger.info(f"Fitting scaler sur {len(df)} lignes par chunks de {chunk_size}")
        # Fit le scaler par chunks
        for i in range(0, len(df), chunk_size):
            end_idx = min(i + chunk_size, len(df))
            chunk = df.iloc[i:end_idx][features_to_normalize]
            chunk_clean = chunk.ffill().bfill()
            
            if i == 0:
                scaler.partial_fit(chunk_clean)
            else:
                scaler.partial_fit(chunk_clean)
            
            logger.info(f"Processed chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}")
            gc.collect()  # Forcer la collecte des ordures
        
        # Sauvegarder le scaler
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
    
    # Transformer par chunks
    logger.info(f"Transforming data par chunks de {chunk_size}")
    for i in range(0, len(df), chunk_size):
        end_idx = min(i + chunk_size, len(df))
        chunk = df_normalized.iloc[i:end_idx][features_to_normalize]
        chunk_clean = chunk.ffill().bfill()
        
        df_normalized.iloc[i:end_idx, df_normalized.columns.get_indexer(features_to_normalize)] = scaler.transform(chunk_clean)
        
        if (i // chunk_size + 1) % 10 == 0:  # Log tous les 10 chunks
            logger.info(f"Transformed chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}")
        gc.collect()
    
    logger.info(f"Normalized {len(features_to_normalize)} features using chunked processing")
    return df_normalized, scaler

def normalize_features_chunked_transform(df, features_to_normalize, scaler, chunk_size):
    """
    Transforme les données avec un scaler déjà ajusté, par chunks.
    """
    if not features_to_normalize or scaler is None:
        return df
    
    df_normalized = df.copy()
    
    logger.info(f"Transforming {len(df)} lignes par chunks de {chunk_size}")
    for i in range(0, len(df), chunk_size):
        end_idx = min(i + chunk_size, len(df))
        chunk = df_normalized.iloc[i:end_idx][features_to_normalize]
        chunk_clean = chunk.ffill().bfill()
        
        df_normalized.iloc[i:end_idx, df_normalized.columns.get_indexer(features_to_normalize)] = scaler.transform(chunk_clean)
        
        if (i // chunk_size + 1) % 10 == 0:  # Log tous les 10 chunks
            logger.info(f"Transformed chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}")
        gc.collect()
    
    logger.info(f"Normalized {len(features_to_normalize)} features using chunked transform")
    return df_normalized

def find_data_source_for_asset_timeframe(asset, timeframe, data_sources, data_dir):
    """
    Trouve la source de données appropriée pour un actif et timeframe donné.
    
    Args:
        asset: Symbole de l'actif
        timeframe: Intervalle de temps
        data_sources: Liste des sources de données configurées
        data_dir: Répertoire de base des données
        
    Returns:
        tuple: (source_info, file_path) ou (None, None) si non trouvé
    """
    logger.debug(f"Recherche de la source pour {asset} ({timeframe}) parmi {len(data_sources)} sources")
    
    for i, source in enumerate(data_sources):
        logger.debug(f"Source {i+1}: {source.get('name', 'sans nom')}")
        logger.debug(f"  Assets: {source.get('assets', [])}")
        logger.debug(f"  Timeframes: {source.get('timeframes', [])}")
        
        if asset in source.get('assets', []) and timeframe in source.get('timeframes', []):
            # Construire le chemin du fichier selon le pattern
            filename = source.get('filename_pattern', '').format(ASSET=asset, TIMEFRAME=timeframe)
            directory = source.get('directory', '').format(TIMEFRAME=timeframe)
            file_path = os.path.join(data_dir, directory, filename)
            
            logger.debug(f"  Chemin construit: {file_path}")
            logger.debug(f"  Le fichier existe: {os.path.exists(file_path)}")
            
            if os.path.exists(file_path):
                logger.debug(f"  Fichier trouvé: {file_path}")
                return source, file_path
            else:
                logger.debug(f"  Fichier non trouvé: {file_path}")
    
    logger.warning(f"Aucune source de données trouvée pour {asset} ({timeframe})")
    return None, None

def process_asset_data(asset, timeframe, config, data_dir, processed_data_dir, scalers_dir):
    """
    Traite les données d'un actif depuis les sources configurées et génère le dataset final.
    
    Args:
        asset: Symbole de l'actif
        timeframe: Intervalle de temps
        config: Configuration des données
        data_dir: Répertoire de base des données
        processed_data_dir: Répertoire des données traitées
        scalers_dir: Répertoire des scalers
        
    Returns:
        bool: True si succès, False sinon
    """
    # Configuration du traitement
    
    # Configuration pour traitement par chunks (éviter les erreurs de mémoire)
    chunk_size = config.get('processing_chunk_size', 50000)  # Par défaut 50k lignes par chunk
    try:
        # Trouver la source de données appropriée
        data_sources = config.get('data_sources', [])
        source_info, raw_file = find_data_source_for_asset_timeframe(asset, timeframe, data_sources, data_dir)
        
        if source_info is None:
            logger.error(f"Aucune source de données trouvée pour {asset} ({timeframe})")
            return False
        
        if not os.path.exists(raw_file):
            logger.error(f"Fichier de données non trouvé: {raw_file}")
            return False
        
        # Lire le fichier CSV avec le bon séparateur et gestion des dates
        df = pd.read_csv(raw_file, parse_dates=['timestamp'], index_col='timestamp')
        logger.info(f"Données chargées pour {asset} ({timeframe}) depuis {source_info['group_name']}: {len(df)} lignes")
        
        # Gérer l'index temporel pour les nouvelles données
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"Conversion de la colonne timestamp en index pour {asset} ({timeframe})")
            df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index, utc=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Aucune colonne timestamp trouvée et index non temporel pour {asset} ({timeframe})")
        
        # Vérifier que les colonnes nécessaires sont présentes
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Colonnes OHLCV manquantes dans les données pour {asset}")
            return False
        
        # Vérifier si les features sont déjà calculées (nouvelles données)
        features_ready = source_info.get('features_ready', False)
        
        if features_ready:
            logger.info(f"Features déjà calculées pour {asset} ({timeframe}) - pas besoin de calcul d'indicateurs")
            # Pour les nouvelles données, toutes les colonnes sauf timestamp et symbol sont des features
            exclude_cols = ['timestamp', 'symbol', 'date', 'datetime']
            available_features = [col for col in df.columns if col not in exclude_cols]
            added_features = [col for col in available_features if col not in required_cols]
            
            # Supprimer la colonne symbol des données traitées car elle n'est pas numérique
            if 'symbol' in df.columns:
                df = df.drop('symbol', axis=1)
                logger.info(f"Colonne 'symbol' supprimée pour {asset} ({timeframe})")
        else:
            # Ajouter les indicateurs techniques spécifiques au timeframe
            indicators_by_timeframe = config.get('indicators_by_timeframe', {})
            indicators_config = indicators_by_timeframe.get(timeframe, [])
            
            if not indicators_config:
                logger.warning(f"❌ Aucun indicateur trouvé pour le timeframe {timeframe} dans la configuration.")
                logger.error(f"Vérifiez que indicators_by_timeframe['{timeframe}'] existe dans la config.")
                return False
            
            logger.info(f"🔧 Calcul de {len(indicators_config)} indicateurs pour {asset} ({timeframe})")
            
            # Ajouter les indicateurs et récupérer la liste des features ajoutées
            df, added_features = add_technical_indicators(df, indicators_config, timeframe)
            logger.info(f"✅ Indicateurs techniques calculés pour {asset} ({timeframe}): {len(added_features)} features")
        
        logger.info(f"Features disponibles après traitement (premières 10): {df.columns.tolist()[:10]}")
        
        # Construire dynamiquement la liste des features à normaliser pour ce timeframe
        # NE PAS normaliser les colonnes OHLC de base pour conserver les prix réels
        features_to_normalize = []
        
        # Ajouter uniquement les indicateurs calculés et le volume
        for feature in added_features:
            features_to_normalize.append(feature)
        
        # Ajouter le volume à la liste des features à normaliser
        if 'volume' in df.columns:
            features_to_normalize.append('volume')
            
        logger.info(f"OHLC ne seront PAS normalisés pour conserver les prix réels")
        logger.info(f"Features à normaliser pour {asset} ({timeframe}): {len(features_to_normalize)} features")
        
        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_features = [col for col in features_to_normalize if col not in df.columns]
        if missing_features:
            logger.warning(f"Colonnes manquantes dans le DataFrame: {missing_features}")
            logger.debug(f"Colonnes disponibles: {df.columns.tolist()}")
        
        # Gérer les valeurs manquantes
        missing_values_handling = config.get('missing_values_handling', 'ffill')
        if missing_values_handling == 'ffill':
            df = df.ffill().bfill()  # Utiliser ffill() et bfill() au lieu de fillna(method=...)
        elif missing_values_handling == 'drop':
            df.dropna(inplace=True)
        
        logger.info(f"Période des données pour {asset} ({timeframe}): {df.index.min()} à {df.index.max()}")
        
        # Supprimer les lignes restantes avec des valeurs manquantes
        df.dropna(inplace=True)
        
        # Diviser les données en ensembles d'entraînement, de validation et de test
        data_split = config.get('data_split', {})
        
        # Obtenir les dates spécifiques pour ce timeframe
        tf_split = data_split.get(timeframe, {})
        if not tf_split:
            logger.warning(f"Aucune configuration de split pour le timeframe {timeframe}. Utilisation des valeurs par défaut.")
            # Get timezone info from the DataFrame index if available
            index_tz = getattr(df.index, 'tz', None)
            
            train_start = pd.to_datetime('2024-01-01')
            train_end = pd.to_datetime('2025-03-31')
            val_start = pd.to_datetime('2025-04-01')
            val_end = pd.to_datetime('2025-04-30')
            test_start = pd.to_datetime('2025-05-01')
            test_end = pd.to_datetime('2025-05-20')
            
            # Apply timezone only if the DataFrame index has timezone
            if index_tz is not None:
                train_start = train_start.tz_localize('UTC')
                train_end = train_end.tz_localize('UTC')
                val_start = val_start.tz_localize('UTC')
                val_end = val_end.tz_localize('UTC')
                test_start = test_start.tz_localize('UTC')
                test_end = test_end.tz_localize('UTC')
        else:
            # Get timezone info from the DataFrame index if available
            index_tz = getattr(df.index, 'tz', None)
            
            train_start = pd.to_datetime(tf_split.get('train_start_date', '2024-01-01'))
            train_end = pd.to_datetime(tf_split.get('train_end_date', '2025-03-31'))
            val_start = pd.to_datetime(tf_split.get('validation_start_date', '2025-04-01'))
            val_end = pd.to_datetime(tf_split.get('validation_end_date', '2025-04-30'))
            test_start = pd.to_datetime(tf_split.get('test_start_date', '2025-05-01'))
            test_end = pd.to_datetime(tf_split.get('test_end_date', '2025-05-20'))
            
            # Apply timezone only if the DataFrame index has timezone
            if index_tz is not None:
                train_start = train_start.tz_localize('UTC')
                train_end = train_end.tz_localize('UTC')
                val_start = val_start.tz_localize('UTC')
                val_end = val_end.tz_localize('UTC')
                test_start = test_start.tz_localize('UTC')
                test_end = test_end.tz_localize('UTC')
        
        # S'assurer que l'index est un DatetimeIndex (déjà fait plus haut)
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"Impossible de convertir l'index en DatetimeIndex pour {asset} ({timeframe})")
            return False
        
        # Filtrer les données pour chaque ensemble
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        val_df = df[(df.index >= val_start) & (df.index <= val_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        
        logger.info(f"Données divisées pour {asset} ({timeframe}):")
        logger.info(f"  - Entraînement: {len(train_df)} lignes ({train_start} à {train_end})")
        logger.info(f"  - Validation: {len(val_df)} lignes ({val_start} à {val_end})")
        logger.info(f"  - Test: {len(test_df)} lignes ({test_start} à {test_end})")
        
        # Vérifier à nouveau que toutes les colonnes nécessaires sont présentes
        missing_features = [col for col in features_to_normalize if col not in train_df.columns]
        if missing_features:
            logger.error(f"Colonnes toujours manquantes après correction: {missing_features}")
            return False
        
        # Normaliser les features
        normalization_method = config.get('normalization', {}).get('method', 'standard')
        
        # Créer le répertoire des scalers pour cet actif s'il n'existe pas
        asset_scalers_dir = os.path.join(scalers_dir, asset)
        os.makedirs(asset_scalers_dir, exist_ok=True)
        
        # Normaliser les données par chunks pour éviter les problèmes de mémoire
        scaler_path = os.path.join(asset_scalers_dir, f"{asset}_{timeframe}_scaler.joblib")
        
        # Pour les gros datasets (comme lot2), traiter par chunks
        if len(train_df) > chunk_size:
            logger.info(f"Dataset volumineux ({len(train_df)} lignes), traitement par chunks de {chunk_size}")
            train_df_normalized, scaler = normalize_features_chunked(
                train_df, features_to_normalize, scaler_path, chunk_size, fit=True
            )
            val_df_normalized = normalize_features_chunked_transform(
                val_df, features_to_normalize, scaler, chunk_size
            )
            test_df_normalized = normalize_features_chunked_transform(
                test_df, features_to_normalize, scaler, chunk_size
            )
        else:
            # Traitement normal pour les datasets plus petits
            train_df_normalized, scaler = normalize_features(
                train_df, 
                scaler=None, 
                scaler_path=scaler_path, 
                fit=True, 
                numeric_cols=features_to_normalize
            )
            
            val_df_normalized, _ = normalize_features(
                val_df, 
                scaler=scaler, 
                fit=False, 
                numeric_cols=features_to_normalize
            )
            
            test_df_normalized, _ = normalize_features(
                test_df, 
                scaler=scaler, 
                fit=False, 
                numeric_cols=features_to_normalize
            )
        
        # Sauvegarder les données traitées
        asset_dir = os.path.join(processed_data_dir, asset)
        os.makedirs(asset_dir, exist_ok=True)
        
        train_df_normalized.to_parquet(os.path.join(asset_dir, f"{asset}_{timeframe}_train.parquet"))
        val_df_normalized.to_parquet(os.path.join(asset_dir, f"{asset}_{timeframe}_val.parquet"))
        test_df_normalized.to_parquet(os.path.join(asset_dir, f"{asset}_{timeframe}_test.parquet"))
        
        logger.info(f"Données traitées sauvegardées pour {asset} ({timeframe})")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement des données pour {asset} ({timeframe}): {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process raw market data and generate the final dataset.")
    parser.add_argument('--main_config', type=str, default='config/main_config.yaml', help='Path to the main configuration file.')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to the data configuration file.')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Configurer le niveau de journalisation
    logging.basicConfig(level=getattr(logging, args.log_level),
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        logger.info(f"Chargement de la configuration principale: {args.main_config}")
        logger.info(f"Chargement de la configuration des données: {args.data_config}")
        
        # Charger les configurations
        main_cfg = load_config(args.main_config)
        logger.debug(f"Contenu de la configuration principale: {main_cfg}")
        
        data_cfg = load_config(args.data_config)
        logger.debug(f"Contenu de la configuration des données: {data_cfg}")
    except FileNotFoundError as e:
        logger.error(f"Erreur de configuration : {e}. Assurez-vous que les chemins sont corrects.")
        return
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return

    # Obtenir les chemins des répertoires
    try:
        project_root = main_cfg.get('paths', {}).get('base_project_dir_local', os.getcwd())
        data_dir = os.path.join(project_root, main_cfg.get('paths', {}).get('data_dir_name', 'data'))
        
        processed_data_dir = os.path.join(data_dir, data_cfg.get('processed_data_dir', 'processed'))
        scalers_dir = os.path.join(data_dir, data_cfg.get('scalers_encoders_dir', 'scalers_encoders'))
        
        os.makedirs(processed_data_dir, exist_ok=True)
        os.makedirs(scalers_dir, exist_ok=True)
        
        logger.info(f"Répertoire de base des données: {data_dir}")
        logger.info(f"Répertoire des données traitées: {processed_data_dir}")
        logger.info(f"Répertoire des scalers: {scalers_dir}")
        
        # Afficher les sources de données configurées
        data_sources = data_cfg.get('data_sources', [])
        logger.info(f"Sources de données configurées: {len(data_sources)}")
        for source in data_sources:
            source_dir = os.path.join(data_dir, source['directory'])
            logger.info(f"  - {source['group_name']}: {source_dir} ({source['assets'][:3]}{'...' if len(source['assets']) > 3 else ''})")
            
    except Exception as e:
        logger.error(f"Erreur lors de la configuration des chemins de données : {e}")
        return

    # Traiter les données pour chaque actif et timeframe
    assets = data_cfg.get('assets', [])
    # Utiliser timeframes_to_process au lieu de timeframes pour être cohérent avec la nouvelle structure
    timeframes = data_cfg.get('timeframes_to_process', data_cfg.get('timeframes', ["1h"]))
    
    logger.info(f"Actifs à traiter: {assets}")
    logger.info(f"Timeframes à traiter: {timeframes}")
    
    success_count = 0
    total_count = len(assets) * len(timeframes)
    
    for asset in assets:
        for timeframe in timeframes:
            logger.info(f"Traitement des données pour {asset} ({timeframe})...")
            if process_asset_data(asset, timeframe, data_cfg, data_dir, processed_data_dir, scalers_dir):
                success_count += 1
    
    logger.info(f"Traitement des données terminé. {success_count}/{total_count} actifs traités avec succès.")

if __name__ == "__main__":
    main()
