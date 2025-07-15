#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test d'intégration du ChunkedDataLoader et du StateBuilder.

Ce script charge des données à partir des fichiers générés, construit des observations
et vérifie que tout fonctionne comme prévu.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.adan_trading_bot.data_processing.chunked_loader import ChunkedDataLoader
from src.adan_trading_bot.environment.state_builder import StateBuilder

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_state_builder.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_integration():
    """Teste l'intégration entre ChunkedDataLoader et StateBuilder."""
    # Chemins des fichiers
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / 'config' / 'data_config.yaml'
    data_dir = project_dir / 'data' / 'final'
    
    # Charger la configuration
    config = load_config(config_path)
    
    # Configuration des timeframes et des fonctionnalités
    timeframes = ['5m', '1h', '4h']
    
    # Définir les fonctionnalités par timeframe
    features_per_timeframe = {
        '5m': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update'],
        '1h': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update'],
        '4h': ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update']
    }
    
    # Initialiser le ChunkedDataLoader
    logger.info("Initialisation du ChunkedDataLoader...")
    loader = ChunkedDataLoader(
        data_dir=data_dir,
        chunk_size=1000,  # Taille de chunk plus petite pour les tests
        assets_list=['BTC', 'ETH'],  # Tester avec quelques actifs pour commencer
        features_by_timeframe=features_per_timeframe,
        split='train',
        timeframes=timeframes
    )
    
    # Initialiser le StateBuilder
    logger.info("Initialisation du StateBuilder...")
    state_builder = StateBuilder(
        window_size=64,  # Taille de fenêtre pour les données historiques
        timeframes=timeframes,
        features_per_timeframe=features_per_timeframe
    )
    
    # Charger un chunk de données
    logger.info("Chargement d'un chunk de données...")
    chunk = loader.load_chunk(0)
    
    # Vérifier que nous avons bien chargé des données
    if not chunk:
        logger.error("Aucune donnée chargée. Vérifiez les chemins et les fichiers de données.")
        return
    
    # Afficher les actifs et timeframes chargés
    for asset, tf_data in chunk.items():
        logger.info(f"Actif: {asset}")
        for tf, df in tf_data.items():
            if not df.empty:
                logger.info(f"  - {tf}: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                logger.info(f"     Colonnes: {', '.join(df.columns)}")
            else:
                logger.warning(f"  - {tf}: Aucune donnée")
    
    # Tester la construction d'une observation pour un actif
    test_asset = next(iter(chunk.keys()))
    logger.info(f"\nTest de construction d'observation pour l'actif: {test_asset}")
    
    # Vérifier que nous avons des données pour cet actif
    if not chunk[test_asset]:
        logger.error(f"Aucune donnée pour l'actif {test_asset}")
        return
    
    # Préparer les données pour le StateBuilder
    # Nous devons avoir un seul DataFrame avec des colonnes préfixées par le timeframe
    merged_data = None
    
    for tf, df in chunk[test_asset].items():
        if df is None or df.empty:
            logger.warning(f"Aucune donnée pour le timeframe {tf} de l'actif {test_asset}")
            continue
            
        # Faire une copie pour éviter les modifications accidentelles
        df_tf = df.copy()
        
        # Ajouter le préfixe du timeframe aux noms des colonnes
        df_tf = df_tf.add_prefix(f"{tf}_")
        
        # Ajouter une colonne timestamp pour la fusion
        if 'timestamp' in df_tf.columns:
            # Si la colonne timestamp existe déjà, on la renomme pour éviter les conflits
            df_tf = df_tf.rename(columns={f"{tf}_timestamp": 'timestamp'})
        
        # Fusionner avec les données existantes
        if merged_data is None:
            merged_data = df_tf
        else:
            # Fusionner sur la colonne timestamp si elle existe
            if 'timestamp' in merged_data.columns and 'timestamp' in df_tf.columns:
                merged_data = pd.merge(merged_data, df_tf, on='timestamp', how='outer')
            else:
                # Sinon, concaténer les colonnes
                merged_data = pd.concat([merged_data, df_tf], axis=1)
    
    if merged_data is None or merged_data.empty:
        logger.error(f"Impossible de préparer les données pour l'actif {test_asset}")
        return
    
    # Trier par timestamp si disponible
    if 'timestamp' in merged_data.columns:
        merged_data = merged_data.sort_values('timestamp')
    
    # Remplacer les valeurs manquantes par 0 (ou une autre stratégie de remplissage)
    merged_data = merged_data.fillna(0.0)
    
    # Afficher les premières lignes des données fusionnées
    logger.info(f"\nAperçu des données fusionnées pour {test_asset}:")
    logger.info(merged_data.head())
    
    # Afficher les colonnes disponibles
    logger.info(f"\nColonnes disponibles: {', '.join(merged_data.columns)}")
    
    # Construire une observation
    logger.info("\nConstruction d'une observation...")
    observation = state_builder.build_observation(merged_data)
    
    # Afficher les informations sur l'observation
    logger.info(f"Taille de l'observation: {observation.shape}")
    logger.info(f"Type de données: {observation.dtype}")
    
    # Vérifier les valeurs manquantes
    nan_count = np.isnan(observation).sum()
    logger.info(f"Valeurs manquantes dans l'observation: {nan_count}")
    
    # Afficher un résumé des valeurs
    logger.info("\nRésumé des valeurs par canal (timeframe):")
    for i, tf in enumerate(timeframes):
        tf_data = observation[i]
        logger.info(f"  - {tf}: min={tf_data.min():.6f}, max={tf_data.max():.6f}, mean={tf_data.mean():.6f}, std={tf_data.std():.6f}")
    
    # Afficher les premières valeurs de chaque feature pour le premier timeframe
    tf_idx = 0
    tf = timeframes[tf_idx]
    logger.info(f"\nDétails des features pour le timeframe {tf} (premières valeurs):")
    for feat_idx, feature in enumerate(features_per_timeframe[tf]):
        values = observation[tf_idx, :, feat_idx]
        logger.info(f"  - {feature}: {values[:5].tolist()}...")
    
    # Afficher la forme des données après normalisation
    logger.info("\nForme des données après normalisation:")
    logger.info(f"- Nombre de timeframes: {observation.shape[0]}")
    logger.info(f"- Taille de la fenêtre: {observation.shape[1]}")
    logger.info(f"- Nombre de features par timeframe: {observation.shape[2]}")
    
    # Vérifier la plage des valeurs normalisées
    logger.info("\nPlage des valeurs normalisées:")
    for i, tf in enumerate(timeframes):
        tf_data = observation[i]
        logger.info(f"  - {tf}: min={tf_data.min():.6f}, max={tf_data.max():.6f}")
    
    # Vérifier les valeurs aberrantes
    logger.info("\nValeurs aberrantes (en dehors de [-10, 10]):")
    for i, tf in enumerate(timeframes):
        tf_data = observation[i]
        outliers = np.sum((tf_data < -10) | (tf_data > 10))
        logger.info(f"  - {tf}: {outliers} valeurs aberrantes")
    
    logger.info("\nTest d'intégration terminé avec succès!")

def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='Test d\'intégration du ChunkedDataLoader et du StateBuilder')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Niveau de log (par défaut: INFO)')
    return parser.parse_args()

if __name__ == "__main__":
    # Parser les arguments en ligne de commande
    args = parse_args()
    
    # Configurer le niveau de log
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Afficher la configuration utilisée
    logger.info(f"Démarrage du test avec le niveau de log: {args.log_level}")
    
    # Exécuter le test
    test_integration()
