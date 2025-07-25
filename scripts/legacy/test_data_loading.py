#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier le chargement et le formatage des données pour l'entraînement.
"""
import os
import sys
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_validation.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Charge la configuration depuis le fichier YAML."""
    try:
        with open("config/main_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        sys.exit(1)

def check_asset_data(asset, split="train", data_dir=None, timeframes=None, features_per_tf=None):
    """
    Vérifie les données pour un actif et un split donnés.
    
    Args:
        asset (str): Nom de l'actif à vérifier
        split (str): Type de split ('train', 'val', 'test')
        data_dir (Path): Chemin vers le dossier des données
        timeframes (list): Liste des timeframes à vérifier
        features_per_tf (dict): Dictionnaire des features par timeframe
        
    Returns:
        bool: True si la vérification est réussie, False sinon
    """
    if not all([data_dir, timeframes, features_per_tf]):
        logger.error("Paramètres manquants pour la vérification des données")
        return False
        
    file_path = data_dir / asset / f"{split}.parquet"
    
    if not file_path.exists():
        logger.warning(f"Fichier non trouvé: {file_path}")
        return False
    
    try:
        # Charger les données
        logger.info(f"\n=== Vérification des données pour {asset} ({split}) ===")
        df = pd.read_parquet(file_path)
        
        # Afficher les informations de base
        logger.info(f"- Taille: {len(df)} lignes x {len(df.columns)} colonnes")
        if len(df) > 0:
            logger.info(f"- Période: {df.index[0]} à {df.index[-1]}")
        
        # Vérifier les timeframes
        logger.info("\nTimeframes présents dans les données:")
        tf_present = []
        for tf in timeframes:
            # Vérifier si au moins une colonne commence par ce timeframe
            tf_columns = [col for col in df.columns if col.startswith(f"{tf}_")]
            has_tf = len(tf_columns) > 0
            status = "✓" if has_tf else "✗"
            logger.info(f"  {status} {tf} (colonnes: {tf_columns})")
            if has_tf:
                tf_present.append(tf)
        
        # Vérifier les features pour chaque timeframe
        logger.info("\nVérification des features par timeframe:")
        for tf in tf_present:
            logger.info(f"\n  Timeframe: {tf}")
            
            # Récupérer les features attendues pour ce timeframe
            expected_features = features_per_tf.get(tf, [])
            
            # Vérifier chaque feature attendue
            for feature in expected_features:
                col_name = f"{tf}_{feature}"
                if col_name in df.columns:
                    # Vérifier les valeurs manquantes
                    missing = df[col_name].isnull().sum()
                    if missing > 0:
                        logger.warning(f"    ✗ {col_name}: {missing} valeurs manquantes")
                    else:
                        logger.info(f"    ✓ {col_name}: OK")
                else:
                    logger.warning(f"    ✗ Colonne non trouvée: {col_name}")
        
        # Vérifier les valeurs infinies
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_values = np.isinf(df[numeric_cols]).sum().sum()
        if inf_values > 0:
            logger.warning(f"\nATTENTION: {inf_values} valeurs infinies détectées")
            for col in numeric_cols:
                if np.isinf(df[col]).any():
                    logger.warning(f"  - {col}: {np.isinf(df[col]).sum()} valeurs infinies")
        
        # Afficher les statistiques de base pour les principales colonnes
        main_columns = []
        for tf in tf_present:
            for feat in ['open', 'high', 'low', 'close', 'volume']:
                col = f"{tf}_{feat}"
                if col in df.columns:
                    main_columns.append(col)
        
        if main_columns:
            logger.info("\nStatistiques descriptives des principales colonnes:")
            stats = df[main_columns].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            logger.info(f"\n{stats.round(6).to_string()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de {file_path}: {str(e)}", exc_info=True)
        return False

def main():
    """Fonction principale du script."""
    # Charger la configuration
    config = load_config()
    
    # Configuration des chemins et paramètres
    data_dir = Path("data/final")
    assets = config["data"]["assets"]
    timeframes = config["data"]["timeframes"]
    features_per_tf = config["data"]["features_per_timeframe"]
    
    # Vérifier que le dossier des données existe
    if not data_dir.exists():
        logger.error(f"Le dossier des données n'existe pas: {data_dir}")
        sys.exit(1)
    
    logger.info(f"Démarrage de la vérification des données dans: {data_dir.absolute()}")
    logger.info(f"Actifs à vérifier: {', '.join(assets)}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    
    # Vérifier tous les actifs pour chaque split
    for split in ["train", "val", "test"]:
        logger.info(f"\n{'='*80}")
        logger.info(f"VÉRIFICATION DU SPLIT: {split.upper()}")
        logger.info(f"{'='*80}")
        
        for asset in tqdm(assets, desc=f"Vérification des actifs ({split})"):
            check_asset_data(
                asset=asset,
                split=split,
                data_dir=data_dir,
                timeframes=timeframes,
                features_per_tf=features_per_tf
            )
    
    logger.info("\nVérification terminée. Consultez le fichier data_validation.log pour les détails complets.")

if __name__ == "__main__":
    main()
