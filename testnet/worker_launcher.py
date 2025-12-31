#!/usr/bin/env python3
"""
Worker Launcher - Respecte l'architecture du projet
Lance paper_trading_monitor.py avec les bonnes configurations
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def launch_worker(worker_id, testnet=True, capital=10, cycles=100):
    """Lance un worker avec paper_trading_monitor.py"""
    
    # Vérifier les clés API
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ Clés API manquantes")
        return False
    
    # Importer le script existant
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    
    try:
        from paper_trading_monitor import RealPaperTradingMonitor
        
        logger.info(f"🚀 Lancement du worker {worker_id}")
        logger.info(f"   Testnet: {testnet}")
        logger.info(f"   Capital: ${capital}")
        logger.info(f"   Cycles: {cycles}")
        
        # Initialiser le monitor
        monitor = RealPaperTradingMonitor(
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Charger la configuration
        if not monitor.load_config():
            logger.error("❌ Erreur chargement config")
            return False
        
        # Setup exchange
        if not monitor.setup_exchange():
            logger.error("❌ Erreur setup exchange")
            return False
        
        # Setup pipeline
        if not monitor.setup_pipeline():
            logger.error("❌ Erreur setup pipeline")
            return False
        
        logger.info(f"✅ {worker_id} prêt pour le trading")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="ADAN Worker Launcher")
    parser.add_argument("--worker", type=str, required=True, help="Worker ID (w1, w2, w3, w4)")
    parser.add_argument("--testnet", action="store_true", default=True, help="Use testnet")
    parser.add_argument("--capital", type=float, default=10, help="Capital in USD")
    parser.add_argument("--cycles", type=int, default=100, help="Number of trading cycles")
    
    args = parser.parse_args()
    
    # Vérifier worker_id
    if args.worker not in ["w1", "w2", "w3", "w4"]:
        logger.error(f"❌ Worker invalide: {args.worker}")
        sys.exit(1)
    
    # Lancer le worker
    success = launch_worker(
        worker_id=args.worker,
        testnet=args.testnet,
        capital=args.capital,
        cycles=args.cycles
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
