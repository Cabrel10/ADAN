#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour le chargement des données avec le ChunkedDataLoader.
"""

import os
import sys
from pathlib import Path
import logging
import yaml

# Ajouter le répertoire racine au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Charge la configuration depuis le fichier config.yaml."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_data_loading():
    """Teste le chargement des données avec le ChunkedDataLoader."""
    try:
        # Charger la configuration
        config = load_config()
        
        # Configuration spécifique pour le test
        worker_config = {
            'timeframes': ['5m', '1h'],
            'data_split_override': 'train',
            'assets': ['BTCUSDT', 'ETHUSDT'],  # Noms des actifs en majuscules
            'chunk_sizes': {'5m': 100, '1h': 100},
            'features_per_timeframe': {
                '5m': ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'EMA_20'],
                '1h': ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'EMA_20']
            }
        }
        
        logger.info("Initialisation du ChunkedDataLoader...")
        loader = ChunkedDataLoader(config, worker_config)
        
        logger.info("Chargement d'un chunk de données...")
        data = loader.load_chunk(chunk_idx=0)
        
        # Vérifier que les données ont été chargées correctement
        if not data:
            raise ValueError("Aucune donnée n'a été chargée.")
        
        logger.info("Données chargées avec succès pour les actifs : %s", 
                   ', '.join(data.keys()))
        
        # Afficher des informations sur les données chargées
        for asset, timeframes in data.items():
            logger.info("\nActif: %s", asset)
            for tf, df in timeframes.items():
                if df is not None and not df.empty:
                    # Afficher les premières et dernières lignes pour vérification
                    logger.info("  %s: %d lignes, colonnes: %s", 
                              tf, len(df), ', '.join(df.columns))
                    logger.info("  Première ligne: %s", df.index[0] if hasattr(df.index, '__len__') and len(df.index) > 0 else 'N/A')
                    logger.info("  Dernière ligne: %s", df.index[-1] if hasattr(df.index, '__len__') and len(df.index) > 0 else 'N/A')
                    logger.info("  Exemple de données (Open, Close, RSI_14, EMA_20):")
                    if len(df) > 0:
                        sample = df.iloc[0]
                        logger.info("    Open: %.2f, Close: %.2f, RSI_14: %.2f, EMA_20: %.2f",
                                  sample.get('Open', float('nan')), 
                                  sample.get('Close', float('nan')),
                                  sample.get('RSI_14', float('nan')),
                                  sample.get('EMA_20', float('nan')))
                else:
                    logger.warning("  %s: Aucune donnée chargée", tf)
        
        return True
    
    except Exception as e:
        logger.error("Erreur lors du test de chargement des données : %s", str(e), 
                   exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Début du test de chargement des données...")
    success = test_data_loading()
    
    if success:
        logger.info("Test de chargement des données réussi !")
        sys.exit(0)
    else:
        logger.error("Le test de chargement des données a échoué.")
        sys.exit(1)
