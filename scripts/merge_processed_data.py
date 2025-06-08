#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour fusionner les données traitées par actif en un seul DataFrame par timeframe et split.
Ce script lit les fichiers de données traitées individuellement depuis data/processed/{ASSET}/{ASSET}_{TIMEFRAME}_{SPLIT}.parquet,
les fusionne en un seul DataFrame par timeframe et split, et sauvegarde les résultats dans data/processed/merged/.
"""

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

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.adan_trading_bot.common.utils import get_logger, load_config, ensure_dir_exists

logger = get_logger(__name__)

def load_configs(exec_profile='cpu'):
    """
    Charge les configurations nécessaires pour la fusion des données.
    
    Args:
        exec_profile (str): Profil d'exécution ('cpu' ou 'gpu')
    
    Returns:
        tuple: (main_config, data_config)
    """
    logger.info(f"Chargement des configurations avec le profil d'exécution: {exec_profile}")
    main_config_path = 'config/main_config.yaml'
    data_config_path = f'config/data_config_{exec_profile}.yaml'
    
    logger.info(f"Chargement de la configuration principale: {main_config_path}")
    logger.info(f"Chargement de la configuration des données: {data_config_path}")
    
    main_config = load_config(main_config_path)
    data_config = load_config(data_config_path)
    return main_config, data_config

def get_processed_data_paths(main_config, data_config):
    """
    Obtient les chemins des données traitées et les informations sur les actifs et timeframes.
    
    Args:
        main_config (dict): Configuration principale
        data_config (dict): Configuration des données
        
    Returns:
        tuple: (processed_dir, merged_dir, assets, timeframes)
    """
    # Obtenir le chemin du projet depuis la configuration
    project_dir = main_config.get('paths', {}).get('base_project_dir_local', os.getcwd())
    
    # Construire le chemin vers les données traitées
    data_dir = os.path.join(project_dir, main_config.get('paths', {}).get('data_dir_name', 'data'))
    base_processed_dir = os.path.join(data_dir, data_config.get('processed_data_dir', 'processed'))
    
    # Support des lots de données
    lot_id = data_config.get('lot_id', None)
    unified_segment = 'unified'

    # Source directory for individual asset files
    # This will now be data/processed/ or data/processed/{lot_id}/
    # The "unified/{timeframe}/{asset}" will be added in merge_data_for_timeframe_split
    if lot_id:
        processed_dir = os.path.join(base_processed_dir, lot_id)
        logger.info(f"Utilisation du lot de données pour la source des données par actif: {lot_id}")
        logger.info(f"Répertoire de base pour les données par actif (avant unified/tf/asset): {processed_dir}")
    else:
        processed_dir = base_processed_dir
    logger.info(f"Répertoire de base pour les données par actif (avant unified/tf/asset): {processed_dir}")

    # Target directory for merged files (remains .../merged/{lot_id}/unified or .../merged/unified)
    if lot_id:
        merged_dir = os.path.join(base_processed_dir, 'merged', lot_id, unified_segment)
        logger.info(f"Utilisation du lot de données pour la sortie fusionnée: {lot_id}")
    else:
        merged_dir = os.path.join(base_processed_dir, 'merged', unified_segment)

    logger.info(f"Répertoire cible pour les données fusionnées: {merged_dir}")

    # Créer le répertoire pour les données fusionnées s'il n'existe pas
    ensure_dir_exists(merged_dir)
    # logger.info(f"Répertoire des données fusionnées: {merged_dir}") # Duplicate log line
    
    # Obtenir la liste des actifs et timeframes
    assets = data_config.get('assets', [])
    
    # Utiliser timeframes_to_process comme source principale des timeframes
    # avec fallback sur timeframes pour la compatibilité avec l'ancien format
    timeframes = data_config.get('timeframes_to_process', data_config.get('timeframes', []))
    
    # Gérer les timeframes qui peuvent être une liste ou un dictionnaire
    if isinstance(timeframes, dict):
        timeframes = list(timeframes.keys())
    
    logger.info(f"Actifs disponibles: {assets}")
    logger.info(f"Timeframes disponibles: {timeframes}")
    
    return processed_dir, merged_dir, assets, timeframes

def merge_data_for_timeframe_split(processed_dir, merged_dir, assets, timeframe, split):
    """
    Fusionne les données pour un timeframe et un split spécifiques.
    
    Args:
        processed_dir (str): Répertoire des données traitées
        merged_dir (str): Répertoire où sauvegarder les données fusionnées
        assets (list): Liste des actifs à fusionner
        timeframe (str): Timeframe à traiter
        split (str): Split à traiter (train, val, test)
        
    Returns:
        bool: True si la fusion a réussi, False sinon
    """
    logger.info(f"Fusion des données pour le timeframe {timeframe}, split {split}...")
    
    # Dictionnaire pour stocker les DataFrames chargés par actif
    dfs_by_asset = {}
    
    # Charger les données pour chaque actif
    for asset in assets:
        # Construct the full path to the asset-specific file, including timeframe
        # processed_dir is now data/processed/ or data/processed/{lot_id}/
        file_path = os.path.join(processed_dir, "unified", timeframe, asset, f"{asset}_{timeframe}_{split}.parquet")
        
        if not os.path.exists(file_path):
            logger.warning(f"Fichier {file_path} non trouvé. L'actif {asset} sera ignoré pour {timeframe}_{split}.")
            continue
        
        try:
            logger.info(f"Chargement des données pour {asset} ({timeframe}_{split})...")
            df = pd.read_parquet(file_path)
            logger.info(f"Données chargées pour {asset}: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # S'assurer que l'index est bien un DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.info(f"Conversion de l'index en DatetimeIndex pour {asset}")
                df.index = pd.to_datetime(df.index)
            
            # Renommer les colonnes pour inclure le nom de l'actif
            logger.info(f"Renommage des colonnes pour {asset}...")
            df = df.rename(columns={col: f"{col}_{asset}" for col in df.columns})
            
            # Ajouter au dictionnaire des DataFrames
            dfs_by_asset[asset] = df
            logger.info(f"Données préparées pour {asset}: {len(df)} lignes, {len(df.columns)} colonnes")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données pour {asset} ({timeframe}_{split}): {e}")
            logger.error(traceback.format_exc())
            continue
    
    if not dfs_by_asset:
        logger.error(f"Aucune donnée n'a pu être chargée pour {timeframe}_{split}.")
        return False
    
    # Fusionner les DataFrames de manière itérative
    try:
        logger.info(f"Début de la fusion itérative pour {timeframe}_{split}...")
        merged_df = None
        
        # Trier les actifs pour une fusion déterministe
        sorted_assets = sorted(dfs_by_asset.keys())
        
        for i, asset in enumerate(sorted_assets):
            logger.info(f"Fusion de l'actif {asset} ({i+1}/{len(sorted_assets)}) pour {timeframe}_{split}...")
            df_asset = dfs_by_asset[asset]
            
            if merged_df is None:
                merged_df = df_asset.copy()
                logger.info(f"Premier DataFrame assigné: {asset}, forme: {merged_df.shape}")
            else:
                # Fusionner avec le DataFrame accumulé
                logger.info(f"Fusion de {asset} avec le DataFrame accumulé (forme actuelle: {merged_df.shape})...")
                merged_df = pd.merge(merged_df, df_asset, left_index=True, right_index=True, how='outer')
                logger.info(f"Fusion réalisée, nouvelle forme: {merged_df.shape}")
            
            # Libérer la mémoire
            del dfs_by_asset[asset]
            gc.collect()
        
        # Gérer les valeurs manquantes
        if merged_df is not None and not merged_df.empty:
            logger.info(f"Gestion des valeurs manquantes pour {timeframe}_{split}...")
            
            # Compter les NaN avant remplissage
            nan_count_before = merged_df.isna().sum().sum()
            logger.info(f"Nombre de NaN avant remplissage: {nan_count_before}")
            
            # Forward fill puis backward fill (méthode non dépréciée)
            logger.info("Application de ffill()...")
            merged_df = merged_df.ffill()
            
            logger.info("Application de bfill()...")
            merged_df = merged_df.bfill()
            
            # Vérifier s'il reste des NaN
            nan_count_after = merged_df.isna().sum().sum()
            logger.info(f"Nombre de NaN après remplissage: {nan_count_after}")
            
            if nan_count_after > 0:
                logger.warning(f"Il reste {nan_count_after} valeurs NaN dans les données fusionnées pour {timeframe}_{split}.")
                
                # Compter les NaN par ligne et par colonne pour le débogage
                nan_rows = merged_df.isna().sum(axis=1)
                rows_with_nan = nan_rows[nan_rows > 0]
                if len(rows_with_nan) > 0:
                    logger.warning(f"{len(rows_with_nan)} lignes contiennent des NaN.")
                    
                nan_cols = merged_df.isna().sum()
                cols_with_nan = nan_cols[nan_cols > 0]
                if len(cols_with_nan) > 0:
                    logger.warning(f"{len(cols_with_nan)} colonnes contiennent des NaN.")
                    logger.warning(f"Premières colonnes avec NaN: {cols_with_nan.index[:5].tolist() if len(cols_with_nan) > 5 else cols_with_nan.index.tolist()}")
                
                # Supprimer les lignes avec des NaN restants
                logger.info("Suppression des lignes avec NaN restants...")
                merged_df = merged_df.dropna()
                logger.info(f"Après suppression des NaN: {len(merged_df)} lignes")
        
            # Supprimer la colonne 'asset' si elle existe (elle n'est pas nécessaire dans les données fusionnées)
            if 'asset' in merged_df.columns:
                logger.info("Suppression de la colonne 'asset' générique des données fusionnées...")
                merged_df = merged_df.drop(columns=['asset'])
                logger.info("Colonne 'asset' supprimée.")
            else:
                logger.info("Aucune colonne 'asset' générique trouvée dans les données fusionnées (c'est normal).")
            
            # Vérifier si des colonnes contiennent le mot 'asset' (pour détecter d'autres colonnes potentiellement inutiles)
            asset_columns = [col for col in merged_df.columns if 'asset' in col.lower()]
            if asset_columns:
                logger.warning(f"Colonnes contenant 'asset' détectées: {asset_columns}")
                logger.warning("Ces colonnes pourraient être inutiles. Vérifiez si elles doivent être supprimées.")
            
            # Vérification finale : s'assurer qu'aucune colonne 'asset' n'existe
            final_asset_check = [col for col in merged_df.columns if col == 'asset' or col.lower() == 'asset']
            if final_asset_check:
                logger.error(f"ERREUR : Des colonnes 'asset' persistent après suppression : {final_asset_check}")
                # Forcer la suppression
                merged_df = merged_df.drop(columns=final_asset_check, errors='ignore')
                logger.info("Suppression forcée des colonnes 'asset' restantes.")
            
            # Vérifier les types de données avant sauvegarde
            logger.info("Vérification des types de données avant sauvegarde...")
            for col, dtype in merged_df.dtypes.items():
                if dtype == 'object':
                    logger.warning(f"Colonne {col} a un type 'object', ce qui peut causer des problèmes avec Parquet.")
                    # Tenter de convertir en float si possible
                    try:
                        merged_df[col] = merged_df[col].astype('float64')
                        logger.info(f"Colonne {col} convertie en float64.")
                    except Exception as e:
                        logger.error(f"Impossible de convertir la colonne {col} en float64: {e}")
            
            # Sauvegarder le DataFrame fusionné
            try:
                output_file = os.path.join(merged_dir, f"{timeframe}_{split}_merged.parquet")
                logger.info(f"Sauvegarde des données fusionnées dans {output_file}...")
                merged_df.to_parquet(output_file)
                logger.info(f"Données fusionnées sauvegardées: {len(merged_df)} lignes, {len(merged_df.columns)} colonnes")
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde des données fusionnées: {e}")
                logger.error(traceback.format_exc())
                return False
        else:
            logger.error(f"Aucune donnée fusionnée valide pour {timeframe}_{split}.")
            return False
    except Exception as e:
        logger.error(f"Erreur lors de la fusion des données pour {timeframe}_{split}: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Fonction principale pour la fusion des données.
    """
    parser = argparse.ArgumentParser(description='Fusion des données traitées par actif en un seul DataFrame par timeframe et split.')
    parser.add_argument(
        '--exec_profile', 
        type=str, 
        default='cpu_lot1',
        choices=['cpu', 'gpu', 'cpu_lot1', 'cpu_lot2', 'gpu_lot1', 'gpu_lot2', 'smoke_cpu'],
        help="Profil d'exécution ('cpu', 'gpu', 'cpu_lot1', 'cpu_lot2', etc.) pour charger data_config_{profile}.yaml."
    )
    parser.add_argument('--timeframes', nargs='+', help='Liste des timeframes à traiter (par défaut: tous)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='Liste des splits à traiter (par défaut: train, val, test)')
    parser.add_argument('--training-timeframe', type=str, help='Timeframe principal pour l\'entraînement (pour générer un fichier spécial)')
    parser.add_argument('--data_config', type=str, default=None,
                        help='Path to a specific data_config YAML file. Overrides --exec_profile for data config loading.')
    args = parser.parse_args()
    
    # Charger les configurations
    main_config_path = 'config/main_config.yaml' # Or get from args if made configurable
    logger.info(f"Loading main configuration from: {main_config_path}")
    main_config = load_config(main_config_path)
    if main_config is None:
        logger.error(f"FATAL: Main configuration file not found or empty at {main_config_path}")
        sys.exit(1)

    data_config_to_load = None
    if args.data_config:
        data_config_to_load = args.data_config
        logger.info(f"Using explicit data_config from: {data_config_to_load}")
    else:
        data_config_to_load = f'config/data_config_{args.exec_profile}.yaml'
        logger.info(f"Using data_config derived from exec_profile '{args.exec_profile}': {data_config_to_load}")
    
    logger.info(f"Loading data configuration from: {data_config_to_load}")
    try:
        data_config = load_config(data_config_to_load)
        if data_config is None:
            raise FileNotFoundError(f"Data configuration file {data_config_to_load} not found or empty.")
        logger.info(f"Data configuration loaded successfully from {data_config_to_load}")
    except FileNotFoundError:
        logger.error(f"FATAL: Data configuration file not found at {data_config_to_load}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: Error loading data configuration from {data_config_to_load}: {e}")
        sys.exit(1)

    # Obtenir les chemins et informations
    processed_dir, merged_dir, assets, available_timeframes = get_processed_data_paths(main_config, data_config)
    
    # Filtrer les timeframes si spécifiés
    timeframes = args.timeframes if args.timeframes else available_timeframes
    
    # Obtenir le timeframe d'entraînement depuis les arguments ou la configuration
    training_timeframe = args.training_timeframe if args.training_timeframe else data_config.get('training_timeframe', '1h')
    logger.info(f"Timeframe principal pour l'entraînement: {training_timeframe}")
    
    # Fusionner les données pour chaque timeframe et split
    success_count = 0
    total_count = 0
    
    for timeframe in timeframes:
        for split in args.splits:
            total_count += 1
            if merge_data_for_timeframe_split(processed_dir, merged_dir, assets, timeframe, split):
                success_count += 1
                
                # Si c'est le timeframe d'entraînement, créer une copie avec un nom spécial
                if timeframe == training_timeframe:
                    source_file = os.path.join(merged_dir, f"{timeframe}_{split}_merged.parquet")
                    target_file = os.path.join(merged_dir, f"training_{split}_merged.parquet")
                    try:
                        # Copier le fichier
                        import shutil
                        shutil.copy2(source_file, target_file)
                        logger.info(f"Fichier {source_file} copié vers {target_file} (timeframe d'entraînement)")
                    except Exception as e:
                        logger.error(f"Erreur lors de la copie du fichier d'entraînement: {e}")
    
    logger.info(f"Fusion des données terminée. {success_count}/{total_count} fusions réussies.")

if __name__ == "__main__":
    main()
