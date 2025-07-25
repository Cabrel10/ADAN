#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour générer des données d'exemple pour le trading.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    num_candles: int = 10000,
    output_dir: str = 'data/raw'
) -> None:
    """
    Génère des données d'exemple pour une paire de trading et un timeframe donnés.
    
    Args:
        symbol: Paire de trading (ex: 'BTC/USDT')
        timeframe: Période des bougies (ex: '1m', '1h', '1d')
        start_date: Date de début au format 'YYYY-MM-DD'
        num_candles: Nombre de bougies à générer
        output_dir: Répertoire de sortie pour les fichiers CSV
    """
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convertir la date de début en datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Déterminer le pas de temps en fonction du timeframe
    if timeframe.endswith('m'):
        minutes = int(timeframe.replace('m', ''))
        delta = timedelta(minutes=minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe.replace('h', ''))
        delta = timedelta(hours=hours)
    elif timeframe.endswith('d'):
        days = int(timeframe.replace('d', ''))
        delta = timedelta(days=days)
    else:
        raise ValueError(f"Format de timeframe non pris en charge: {timeframe}")
    
    # Générer les timestamps
    timestamps = [start_dt + i * delta for i in range(num_candles)]
    
    # Générer des prix initiaux aléatoires
    np.random.seed(42)  # Pour la reproductibilité
    base_price = 30000 + np.random.normal(0, 1000)
    
    # Générer les prix OHLCV
    prices = []
    current_price = base_price
    
    for _ in range(num_candles):
        # Générer des variations de prix aléatoires
        change_pct = np.random.normal(0, 0.01)  # Environ 1% de variation
        current_price *= (1 + change_pct)
        
        # Générer des prix OHLC réalistes
        open_price = current_price
        close_price = current_price * (1 + np.random.normal(0, 0.005))
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.lognormal(mean=0, sigma=1) * 100
        
        prices.append([open_price, high, low, close_price, volume])
        current_price = close_price
    
    # Créer le DataFrame
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = timestamps
    
    # Réorganiser les colonnes
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Nom du fichier de sortie
    symbol_formatted = symbol.replace('/', '')
    output_file = output_path / f"{symbol_formatted}_{timeframe}.csv"
    
    # Sauvegarder en CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Données d'exemple sauvegardées dans {output_file} ({len(df)} bougies)")

def main():
    """Fonction principale."""
    # Configuration
    config = {
        'symbols': ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"],  # Paires de trading à générer
        'timeframes': ['1m', '1h', '3h'],  # Timeframes à générer
        'start_date': '2022-01-01',  # Date de début
        'num_candles': 10000,  # Nombre de bougies à générer par timeframe
        'output_dir': 'data/raw'  # Répertoire de sortie
    }
    
    # Générer des données pour chaque paire et chaque timeframe
    for symbol in config['symbols']:
        for timeframe in config['timeframes']:
            generate_sample_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=config['start_date'],
                num_candles=config['num_candles'],
                output_dir=config['output_dir']
            )

if __name__ == "__main__":
    main()
