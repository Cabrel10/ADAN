#!/usr/bin/env python3
"""
STRESS TEST DATA PREPARATION
Télécharge et prépare les données pour les 5 scénarios mortels
Calcule les indicateurs EXACTS que le modèle attend
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer


# Scénarios critiques
STRESS_SCENARIOS = {
    "BEAR_2018": {
        "asset": "BTCUSDT",
        "start": "2017-12-01",
        "end": "2018-12-31",
        "description": "Bear Market 2018 (BTC -85%)",
        "expected_crash": -85,
    },
    "COVID_CRASH": {
        "asset": "BTCUSDT",
        "start": "2020-02-15",
        "end": "2020-04-15",
        "description": "Crash COVID Mars 2020 (BTC -60% en 2 jours)",
        "expected_crash": -60,
    },
    "ALT_MASSACRE": {
        "asset": "XRPUSDT",
        "start": "2021-11-01",
        "end": "2022-12-31",
        "description": "Altcoin Massacre 2022 (XRP -80%)",
        "expected_crash": -80,
    },
    "DEAD_RANGE": {
        "asset": "BTCUSDT",
        "start": "2019-06-01",
        "end": "2019-12-01",
        "description": "Dead Range 6 mois (0 volatilité, piège à traders)",
        "expected_crash": 0,
    },
}


def download_data(asset: str, start: str, end: str, timeframe: str = "5m"):
    """
    Télécharge les données OHLCV depuis une source fiable
    Utilise ccxt ou yfinance selon disponibilité
    """
    logger.info(f"Téléchargement {asset} {timeframe} de {start} à {end}...")
    
    try:
        import ccxt
        exchange = ccxt.binance()
        
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        all_data = []
        current = start_dt
        
        while current < end_dt:
            try:
                since = int(current.timestamp() * 1000)
                ohlcv = exchange.fetch_ohlcv(asset, timeframe, since=since, limit=1000)
                
                if not ohlcv:
                    break
                
                df_chunk = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                all_data.append(df_chunk)
                
                current = df_chunk['timestamp'].max() + timedelta(minutes=5)
                logger.info(f"  Downloaded up to {current}")
                
            except Exception as e:
                logger.warning(f"  Erreur lors du téléchargement: {e}")
                break
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Téléchargé {len(df)} candles pour {asset}")
            return df
        else:
            logger.error(f"❌ Aucune donnée téléchargée pour {asset}")
            return None
            
    except ImportError:
        logger.error("❌ ccxt non installé. Installation: pip install ccxt")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return None


def calculate_indicators(df: pd.DataFrame, config: dict, timeframe: str) -> pd.DataFrame:
    """
    Calcule les indicateurs EXACTS que le modèle attend
    """
    logger.info(f"Calcul des indicateurs pour {timeframe}...")
    
    try:
        fe = FeatureEngineer(config, ".")
        df_with_indicators = fe.calculate_indicators_for_single_timeframe(df, timeframe)
        
        logger.info(f"✅ {len(df_with_indicators.columns)} colonnes calculées")
        logger.info(f"   Indicateurs: {[col for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
        
        return df_with_indicators
        
    except Exception as e:
        logger.error(f"❌ Erreur calcul indicateurs: {e}")
        return df


def save_parquet(df: pd.DataFrame, asset: str, timeframe: str, scenario: str):
    """
    Sauvegarde les données en parquet dans la structure attendue
    """
    output_dir = Path(f"data/processed/indicators/stress_tests/{scenario}/{asset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{timeframe}.parquet"
    df.to_parquet(output_file)
    
    logger.info(f"✅ Sauvegardé: {output_file}")
    logger.info(f"   Taille: {len(df)} rows, {len(df.columns)} cols")
    
    return output_file


def prepare_stress_test_data():
    """
    Prépare les données pour tous les scénarios
    """
    logger.info("=" * 80)
    logger.info("STRESS TEST DATA PREPARATION")
    logger.info("=" * 80)
    
    # Charger config
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")
    
    results = {}
    
    for scenario_name, scenario_config in STRESS_SCENARIOS.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"SCÉNARIO: {scenario_name}")
        logger.info(f"Description: {scenario_config['description']}")
        logger.info(f"{'='*80}")
        
        asset = scenario_config["asset"]
        start = scenario_config["start"]
        end = scenario_config["end"]
        
        # Télécharger données
        df = download_data(asset, start, end, "5m")
        
        if df is None or df.empty:
            logger.error(f"❌ Impossible de télécharger les données pour {scenario_name}")
            results[scenario_name] = {"status": "FAILED", "reason": "Download failed"}
            continue
        
        # Calculer indicateurs pour chaque timeframe
        for timeframe in ["5m", "1h", "4h"]:
            try:
                df_indicators = calculate_indicators(df, config, timeframe)
                
                if df_indicators is None or df_indicators.empty:
                    logger.warning(f"⚠️ Pas d'indicateurs pour {timeframe}")
                    continue
                
                # Sauvegarder
                output_file = save_parquet(df_indicators, asset, timeframe, scenario_name)
                
            except Exception as e:
                logger.error(f"❌ Erreur pour {timeframe}: {e}")
                continue
        
        # Statistiques du scénario
        logger.info(f"\n📊 Statistiques {scenario_name}:")
        logger.info(f"   Période: {df.index.min()} → {df.index.max()}")
        logger.info(f"   Candles: {len(df)}")
        logger.info(f"   Prix min: ${df['close'].min():.2f}")
        logger.info(f"   Prix max: ${df['close'].max():.2f}")
        
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        logger.info(f"   Changement: {price_change:.2f}%")
        
        results[scenario_name] = {
            "status": "SUCCESS",
            "rows": len(df),
            "price_change": price_change,
            "expected_crash": scenario_config.get("expected_crash", 0),
        }
    
    # Résumé final
    logger.info(f"\n{'='*80}")
    logger.info("RÉSUMÉ PRÉPARATION")
    logger.info(f"{'='*80}")
    
    for scenario_name, result in results.items():
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        logger.info(f"{status_icon} {scenario_name}: {result['status']}")
        
        if result["status"] == "SUCCESS":
            logger.info(f"   Rows: {result['rows']}, Change: {result['price_change']:.2f}%")
    
    return results


if __name__ == "__main__":
    try:
        results = prepare_stress_test_data()
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ PRÉPARATION TERMINÉE")
        logger.info(f"{'='*80}")
        logger.info("\nLes données sont prêtes dans: data/processed/indicators/stress_tests/")
        logger.info("\nProchaine étape: Lancer les backtests")
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)
