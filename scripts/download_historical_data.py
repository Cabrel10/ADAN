#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour télécharger les données historiques via l'API CCXT.
"""
import os
import sys
import ccxt
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
import yaml
import argparse
from typing import Dict, Any, List, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download_historical_data.log')
    ]
)
logger = logging.getLogger(__name__)

# Chemin par défaut vers le fichier de configuration
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'data_config.yaml'

class DataDownloadError(Exception):
    """Exception levée en cas d'erreur lors du téléchargement des données."""
    pass

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Charge et valide le fichier de configuration.
    
    Args:
        config_path: Chemin vers le fichier de configuration YAML
        
    Returns:
        Dictionnaire contenant la configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Vérifier les clés requises
        required_keys = ['data_pipeline']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Clé de configuration requise manquante: {key}")
                
        return config
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        raise

def initialize_exchange(exchange_id: str, is_testnet: bool = False) -> ccxt.Exchange:
    """
    Initialise une instance d'échange CCXT.
    
    Args:
        exchange_id: ID de l'échange (ex: 'binance', 'bitget')
        is_testnet: Si True, utilise l'environnement de test
        
    Returns:
        Instance de l'échange CCXT
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange_params = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Marché au comptant
                'adjustForTimeDifference': True,
            }
        }
        
        # Configuration spécifique au testnet
        if is_testnet and exchange_id == 'binance':
            exchange_params.update({
                'urls': {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    }
                }
            })
            
        exchange = exchange_class(exchange_params)
        exchange.load_markets()
        return exchange
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de l'échange {exchange_id}: {e}")
        raise

def download_historical_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: datetime,
    output_dir: Path,
    limit: int = 1000
) -> bool:
    """
    Télécharge les données historiques pour une paire de trading et un timeframe donnés.
    
    Args:
        exchange: Instance de l'échange CCXT
        symbol: Paire de trading (ex: 'BTC/USDT')
        timeframe: Période des bougies (ex: '1m', '1h', '1d')
        since: Date de début
        output_dir: Répertoire de sortie pour les fichiers CSV
        limit: Nombre maximum de bougies à récupérer par requête
        
    Returns:
        bool: True si le téléchargement a réussi, False sinon
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vérifier si la paire est disponible sur l'échange
        if symbol not in exchange.symbols:
            logger.warning(f"Paire non disponible sur {exchange.id}: {symbol}")
            return False
            
        # Vérifier si le timeframe est supporté
        if timeframe not in exchange.timeframes:
            logger.warning(f"Timeframe non supporté sur {exchange.id}: {timeframe}")
            return False
        
        # Convertir la date de début en timestamp en millisecondes
        since_ts = int(since.timestamp() * 1000)
        
        # Créer le répertoire du timeframe s'il n'existe pas
        tf_dir = output_dir / timeframe
        tf_dir.mkdir(parents=True, exist_ok=True)
        
        # Nom du fichier de sortie
        symbol_formatted = symbol.replace('/', '')
        output_file = tf_dir / f"{symbol_formatted}.csv"
        
        logger.info(f"Téléchargement des données pour {symbol} {timeframe} depuis {since.date()}...")
        
        all_ohlcv = []
        
        while True:
            try:
                # Télécharger les données par lots (limité par l'API)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
                
                if not ohlcv:
                    break
                    
                # Mettre à jour le timestamp pour la prochaine requête
                since_ts = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)
                
                # Afficher la progression
                last_date = pd.to_datetime(ohlcv[-1][0], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"Données téléchargées jusqu'à {last_date}")
                
                # Sortir si nous avons atteint la date actuelle
                if len(ohlcv) < limit or since_ts > int(time.time() * 1000):
                    break
                    
                # Respecter la limite de taux de l'API
                time.sleep(exchange.rateLimit / 1000)
                
            except ccxt.NetworkError as e:
                logger.warning(f"Erreur réseau: {e}. Nouvelle tentative dans 5 secondes...")
                time.sleep(5)
                continue
                
            except ccxt.ExchangeError as e:
                logger.error(f"Erreur de l'échange: {e}")
                break
                
            except Exception as e:
                logger.error(f"Erreur inattendue: {e}")
                break
        
        if not all_ohlcv:
            logger.warning("Aucune donnée téléchargée.")
            return False
        
        # Convertir en DataFrame
        df = pd.DataFrame(
            all_ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convertir les timestamps en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Trier par timestamp (au cas où)
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        
        # Sauvegarder les données
        df.to_csv(output_file, index=False)
        logger.info(f"Données sauvegardées dans {output_file} ({len(df)} bougies)")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données pour {symbol}: {e}", exc_info=True)
        return False

def main():
    """Fonction principale."""
    try:
        # Configuration des arguments en ligne de commande
        parser = argparse.ArgumentParser(description='Télécharger des données historiques depuis un échange cryptographique.')
        parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH),
                          help=f'Chemin vers le fichier de configuration (défaut: {DEFAULT_CONFIG_PATH})')
        parser.add_argument('--since', type=str, 
                          help='Date de début au format YYYY-MM-DD (défaut: 30 jours avant aujourd\'hui)')
        parser.add_argument('--timeframes', type=str, nargs='+',
                          help='Liste des timeframes à télécharger (ex: 1m 1h 1d)')
        args = parser.parse_args()
        
        # Charger la configuration
        config = load_config(Path(args.config))
        data_pipeline = config['data_pipeline']
        
        # Déterminer la source des données
        if data_pipeline['source'] != 'ccxt':
            raise ValueError("Ce script ne prend en charge que la source de données 'ccxt'")
            
        ccxt_config = data_pipeline['ccxt_download']
        exchange_id = ccxt_config['exchange']
        symbols = ccxt_config['symbols']
        timeframes = args.timeframes or ccxt_config.get('timeframes', ['5m', '1h', '4h'])
        
        # Déterminer la date de début
        since = args.since or ccxt_config.get('since')
        if since:
            since_dt = datetime.strptime(since, '%Y-%m-%d')
        else:
            # Par défaut, télécharger les 30 derniers jours
            since_dt = datetime.now() - timedelta(days=30)
        
        # Initialiser l'échange
        exchange = initialize_exchange(
            exchange_id=exchange_id,
            is_testnet=ccxt_config.get('testnet', False)
        )
        
        # Télécharger les données pour chaque paire et chaque timeframe
        success_count = 0
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    logger.info(f"Traitement de {symbol} {timeframe}...")
                    if download_historical_data(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since_dt,
                        output_dir=Path('data/raw')
                    ):
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {symbol} {timeframe}: {e}", exc_info=True)
        
        logger.info(f"Téléchargement terminé. {success_count} jeux de données téléchargés avec succès.")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur dans le script de téléchargement: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
