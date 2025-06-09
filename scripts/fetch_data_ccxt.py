#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour télécharger les données historiques de marché depuis Binance en utilisant ccxt.
"""
import os
import sys
import argparse
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config, get_path
import logging

# Configuration du logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_ohlcv_data(exchange, symbol, timeframe, since, limit=1000):
    """
    Récupère les données OHLCV d'un échange en utilisant ccxt.
    
    Args:
        exchange: Instance d'échange ccxt
        symbol: Symbole de la paire (ex: 'BTC/USDT')
        timeframe: Intervalle de temps (ex: '1h', '1d')
        since: Timestamp Unix en millisecondes pour la date de début
        limit: Nombre maximum de bougies à récupérer par requête
        
    Returns:
        pd.DataFrame: DataFrame contenant les données OHLCV
    """
    all_candles = []
    
    # Première requête
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        all_candles.extend(candles)
        logger.info(f"Récupéré {len(candles)} bougies pour {symbol} ({timeframe})")
        
        # Si on a récupéré moins de bougies que la limite, on a tout récupéré
        if len(candles) < limit:
            logger.info(f"Toutes les données récupérées pour {symbol} ({timeframe})")
        else:
            # Sinon, on continue à récupérer les données par pagination
            while len(candles) == limit:
                # Attendre pour respecter les limites de rate
                time.sleep(exchange.rateLimit / 1000)  # rateLimit est en millisecondes
                
                # Récupérer la date de la dernière bougie + 1 ms
                last_timestamp = candles[-1][0] + 1
                
                # Récupérer les bougies suivantes
                candles = exchange.fetch_ohlcv(symbol, timeframe, last_timestamp, limit)
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                logger.info(f"Récupéré {len(candles)} bougies supplémentaires pour {symbol} ({timeframe})")
                
                # Si on a récupéré moins de bougies que la limite, on a tout récupéré
                if len(candles) < limit:
                    logger.info(f"Toutes les données récupérées pour {symbol} ({timeframe})")
                    break
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
        return pd.DataFrame()
    
    # Convertir en DataFrame
    if all_candles:
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Fetch historical market data using ccxt.")
    parser.add_argument(
        '--exec_profile', 
        type=str, 
        default='cpu',
        choices=['cpu', 'gpu', 'smoke_cpu'],
        help="Profil d'exécution ('cpu' ou 'gpu' ou 'smoke_cpu') pour charger data_config_{profile}.yaml."
    )
    parser.add_argument('--main_config', type=str, default='config/main_config.yaml', help='Path to the main configuration file.')
    parser.add_argument('--data_config', type=str, default=None, help='Path to the data configuration file (default: config/data_config_{profile}.yaml).')
    args = parser.parse_args()

    try:
        # Construire le chemin du fichier de configuration des données en fonction du profil
        profile = args.exec_profile
        logger.info(f"Utilisation du profil d'exécution: {profile}")
        
        data_config_path = args.data_config if args.data_config else f'config/data_config_{profile}.yaml'
        logger.info(f"Chargement de la configuration principale: {args.main_config}")
        logger.info(f"Chargement de la configuration des données: {data_config_path}")
        
        # Charger les configurations
        main_cfg = load_config(args.main_config)
        data_cfg = load_config(data_config_path)
    except FileNotFoundError as e:
        logger.error(f"Erreur de configuration : {e}. Assurez-vous que les chemins sont corrects.")
        return
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return

    # Obtenir le chemin du répertoire des données brutes
    try:
        project_root = main_cfg.get('paths', {}).get('base_project_dir_local', os.getcwd())
        raw_data_dir = os.path.join(project_root, main_cfg.get('paths', {}).get('data_dir_name', 'data'), 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)
        logger.info(f"Les données brutes seront sauvegardées dans : {raw_data_dir}")
    except Exception as e:
        logger.error(f"Erreur lors de la configuration des chemins de données : {e}")
        return

    # Créer une instance de l'échange Binance
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,  # Respecter les limites de rate de l'API
        })
        logger.info(f"Connexion à {exchange.name} établie")
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à l'échange: {e}")
        return

    assets = data_cfg.get('assets', [])
    timeframes = data_cfg.get('timeframes', ["1h"])
    timeframe_periods = data_cfg.get('timeframe_periods', {})
    
    # Date de début par défaut (1 an en arrière si non spécifiée)
    default_start_date_str = data_cfg.get('default_fetch_start_date', (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
    default_end_date_str = datetime.now().strftime("%Y-%m-%d")

    for asset in assets:
        # Formater le symbole pour ccxt (ajouter '/' entre la crypto et USDT)
        if 'USDT' in asset:
            base, quote = asset.split('USDT')
            symbol = f"{base}/USDT"
        else:
            symbol = asset  # Au cas où le format est déjà correct
        
        for timeframe in timeframes:
            # Obtenir les dates spécifiques pour ce timeframe
            tf_config = timeframe_periods.get(timeframe, {})
            start_date_str = tf_config.get('start_date', default_start_date_str)
            end_date_str = tf_config.get('end_date', default_end_date_str)
            
            # Convertir en timestamp Unix (millisecondes)
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            since = int(start_date.timestamp() * 1000)
            
            logger.info(f"Téléchargement des données pour {symbol} avec l'intervalle {timeframe} depuis {start_date_str} jusqu'à {end_date_str}...")
            
            df = fetch_ohlcv_data(exchange, symbol, timeframe, since)
            
            if not df.empty:
                # Filtrer les données jusqu'à la date de fin
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                df = df[df.index <= end_date]
                
                # Nom du fichier: ASSET_TIMEFRAME_raw.parquet
                filename = f"{asset}_{timeframe}_raw.parquet"
                filepath = os.path.join(raw_data_dir, filename)
                try:
                    df.to_parquet(filepath)
                    logger.info(f"Données pour {asset} ({timeframe}) sauvegardées dans {filepath}: {len(df)} lignes de {df.index.min()} à {df.index.max()}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde du fichier {filepath}: {e}")
            else:
                logger.warning(f"Aucune donnée à sauvegarder pour {asset} ({timeframe}).")
    
    logger.info("Téléchargement des données terminé.")

if __name__ == "__main__":
    main()
