#!/usr/bin/env python3
"""
Diagnostic complet de la santé du modèle ADAN.
Vérifie: données réelles, signaux, trades, latences réseau.
"""

import os
import sys
import time
import yaml
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def diagnose_api_connection():
    """Test 1: Connexion API Binance"""
    logger.info("=" * 80)
    logger.info("TEST 1: API BINANCE CONNECTION")
    logger.info("=" * 80)
    
    try:
        from src.adan_trading_bot.exchange_api.connector import get_exchange_client
        
        with open('config/config.yaml') as f:
            config = yaml.safe_load(f)
        
        start = time.time()
        client = get_exchange_client(config)
        latency = (time.time() - start) * 1000
        
        logger.info(f"✅ API Client créé en {latency:.2f}ms")
        
  