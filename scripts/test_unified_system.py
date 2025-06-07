#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour valider le système unifié ADAN avec 8 actifs et 42 indicateurs.
Teste le pipeline complet : données brutes → indicateurs → fusion → entraînement.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.data_processing.feature_engineer import add_technical_indicators
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.environment.order_manager import OrderManager

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_unified():
    """Test 1: Vérification de la configuration unifiée"""
    logger.info("🧪 TEST 1: Configuration unifiée")

    try:
        config = load_config('config/data_config_cpu.yaml')

        # Vérifier les 8 actifs
        assets = config.get('assets', [])
        expected_assets = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"]

        if len(assets) != 8:
            logger.error(f"❌ Nombre d'actifs incorrect: {len(assets)}/8")
            return False

        for asset in expected_assets:
            if asset not in assets:
                logger.error(f"❌ Actif manquant: {asset}")
                return False

        # Vérifier le lot_id unifié
        lot_id = config.get('lot_id')
        if lot_id != 'unified':
            logger.error(f"❌ lot_id incorrect: {lot_id} (attendu: unified)")
            return False

        # Vérifier data_source_type
        data_source_type = config.get('data_source_type')
        if data_source_type != 'calculate_from_raw':
            logger.error(f"❌ data_source_type incorrect: {data_source_type}")
            return False

        # Vérifier les 3 timeframes
        timeframes_to_process = config.get('timeframes_to_process', [])
        expected_timeframes = ['1m', '1h', '1d']

        for tf in expected_timeframes:
            if tf not in timeframes_to_process:
                logger.error(f"❌ Timeframe manquant: {tf}")
                return False

        # Vérifier les indicateurs par timeframe
        indicators_by_timeframe = config.get('indicators_by_timeframe', {})

        for tf in expected_timeframes:
            indicators = indicators_by_timeframe.get(tf, [])
            if len(indicators) < 30:  # Au moins 30 indicateurs (42 attendus)
                logger.error(f"❌ Indicateurs insuffisants pour {tf}: {len(indicators)}")
                return False

        logger.info(f"✅ Configuration unifiée validée: {len(assets)} actifs, {len(timeframes_to_process)} timeframes")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors du test de configuration: {e}")
        return False

def test_indicators_calculation():
    """Test 2: Calcul des 42 indicateurs techniques"""
    logger.info("🧪 TEST 2: Calcul des indicateurs techniques")

    try:
        config = load_config('config/data_config_cpu.yaml')
        indicators_by_timeframe = config.get('indicators_by_timeframe', {})

        # Créer des données OHLCV fictives
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        np.random.seed(42)

        df = pd.DataFrame({
            'open': 100 + np.random.randn(1000).cumsum() * 0.1,
            'high': 100 + np.random.randn(1000).cumsum() * 0.1 + 0.5,
            'low': 100 + np.random.randn(1000).cumsum() * 0.1 - 0.5,
            'close': 100 + np.random.randn(1000).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)

        # Garantir que high >= low >= close >= open
        df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(1000) * 0.1)
        df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(1000) * 0.1)

        # Tester pour timeframe 1m
        timeframe = '1m'
        indicators_config = indicators_by_timeframe.get(timeframe, [])

        if not indicators_config:
            logger.error(f"❌ Aucun indicateur configuré pour {timeframe}")
            return False

        logger.info(f"🔧 Test de {len(indicators_config)} indicateurs pour {timeframe}")

        # Calculer les indicateurs
        df_with_indicators, added_features = add_technical_indicators(df, indicators_config, timeframe)

        if len(added_features) < 30:  # Au moins 30 features ajoutées
            logger.error(f"❌ Trop peu d'indicateurs calculés: {len(added_features)}")
            return False

        # Vérifier que les noms contiennent le timeframe
        time
