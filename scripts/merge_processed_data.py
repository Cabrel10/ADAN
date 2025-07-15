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

def process_asset(asset_id: str, config: dict) -> None:
    """Charge, fusionne, et sauvegarde les données pour un seul actif."""
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
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df = df.add_prefix(f"{tf}_")
            dfs[tf] = df
        else:
            logger.warning(f"Fichier manquant pour {asset_id} - {tf}, il sera ignoré.")

    if not dfs:
        logger.error(f"Aucune donnée de timeframe trouvée pour {asset_id}. Abandon.")
        return

    base_tf = timeframes[0]
    if base_tf not in dfs:
        logger.error(f"Le timeframe de base '{base_tf}' n'a pas pu être chargé pour {asset_id}. Abandon.")
        return
    
    merged_df = dfs[base_tf].copy()
    
    for tf in timeframes[1:]:
        if tf in dfs:
            df_to_join = dfs[tf]
            # La jointure externe crée des lignes pour tous les timestamps uniques
            merged_df = merged_df.join(df_to_join, how='outer')

    # Le forward-fill propage les valeurs des timeframes lents
    logger.info("Application du forward-fill pour propager les features des timeframes supérieurs...")
    merged_df.ffill(inplace=True)
    
    # Conserver uniquement les lignes qui existent dans le timeframe de base
    merged_df = merged_df.loc[dfs[base_tf].index]
    
    # Nettoyage final des NaNs qui pourraient persister au début
    merged_df.dropna(inplace=True)
    
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
