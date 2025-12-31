#!/usr/bin/env python3
"""
ADAN Orchestrator - Consensus-based trading
Respecte l'architecture du projet et les normes de codage
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testnet/adan_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ADANOrchestrator:
    """
    ADAN: Autonomous Distributed Adaptive Network
    Orchestre les 4 workers avec consensus voting
    """
    
    def __init__(self, api_key, api_secret, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Importer le script existant
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from paper_trading_monitor import RealPaperTradingMonitor
        
        self.monitor_class = RealPaperTradingMonitor
        self.workers = {}
        self.results = {
            "orchestrator": "ADAN",
            "timestamp": datetime.now().isoformat(),
            "total_cycles": 0,
            "consensus_decisions": [],
            "metrics": {
                "consensus_reached": 0,
                "consensus_failed": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0
            }
        }
    
    def initialize_workers(self):
        """Initialise les 4 workers"""
        logger.info("🔧 Initialisation des 4 workers...")
        
        for worker_id in ["w1", "w2", "w3", "w4"]:
            try:
                monitor = self.monitor_class(
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                
                if not monitor.load_config():
                    logger.error(f"❌ Erreur config {worker_id}")
                    continue
                
                if not monitor.setup_exchange():
                    logger.error(f"❌ Erreur exchange {worker_id}")
                    continue
                
                if not monitor.setup_pipeline():
                    logger.error(f"❌ Erreur pipeline {worker_id}")
                    continue
                
                self.workers[worker_id] = monitor
                logger.info(f"✅ {worker_id} initialisé")
                
            except Exception as e:
                logger.error(f"❌ Erreur {worker_id}: {e}")
        
        if len(self.workers) < 4:
            logger.warning(f"⚠️ Seulement {len(self.workers)}/4 workers initialisés")
        
        return len(self.workers) > 0
    
    def get_worker_decisions(self):
        """Récupère les décisions de tous les workers"""
        decisions = {}
        
        for worker_id, monitor in self.workers.items():
            try:
                # Récupérer les données
                monitor.fetch_data()
                
                # Construire l'état
                state = monitor.state_builder.build_state(
                    monitor.latest_raw_data,
                    portfolio_state={"balance": 100, "position": 0}
                )
                
                if state is not None:
                    # Inférence du modèle
                    # (Utiliser le premier worker comme exemple)
                    if worker_id in monitor.workers:
                        model = monitor.workers[worker_id]
                        action, _ = model.predict(state, deterministic=False)
                        
                        # Convertir en vote
                        action_value = float(action[0]) if hasattr(action, '__getitem__') else float(action)
                        
                        if action_value > 0.5:
                            vote = "BUY"
                        elif action_value < -0.5:
                            vote = "SELL"
                        else:
                            vote = "HOLD"
                        
                        decisions[worker_id] = {
                            "action": action_value,
                            "vote": vote
                        }
                        
                        logger.info(f"  {worker_id}: {vote} (action: {action_value:.4f})")
                
            except Exception as e:
                logger.debug(f"⚠️ Erreur décision {worker_id}: {e}")
                decisions[worker_id] = {"action": 0, "vote": "HOLD"}
        
        return decisions
    
    def consensus_voting(self, decisions):
        """Implémente le consensus ADAN"""
        vote_counts = defaultdict(int)
        
        for worker_id, decision in decisions.items():
            vote_counts[decision["vote"]] += 1
        
        # Chercher le consensus (3+ votes identiques)
        for vote, count in vote_counts.items():
            if count >= 3:
                return vote, count, True
        
        # Pas de consensus = HOLD
        return "HOLD", 0, False
    
    def execute(self, cycles=50):
        """Boucle principale d'orchestration"""
        logger.info("🚀 Démarrage de l'orchestrateur ADAN")
        logger.info(f"📊 Cycles: {cycles}")
        
        if not self.initialize_workers():
            logger.error("❌ Impossible d'initialiser les workers")
            return False
        
        try:
            for cycle in range(cycles):
                logger.info(f"\n{'='*60}")
                logger.info(f"Cycle {cycle + 1}/{cycles}")
                logger.info(f"{'='*60}")
                
                # Récupérer les décisions
                logger.info("🗳️  Votes des workers:")
                decisions = self.get_worker_decisions()
                
                if not decisions:
                    logger.warning("⚠️ Pas de décisions")
                    time.sleep(5)
                    continue
                
                # Consensus voting
                consensus_vote, vote_count, consensus_reached = self.consensus_voting(decisions)
                
                if consensus_reached:
                    logger.info(f"✅ CONSENSUS: {consensus_vote} ({vote_count}/4)")
                    self.results["metrics"]["consensus_reached"] += 1
                else:
                    logger.info(f"⚠️ Pas de consensus - {consensus_vote}")
                    self.results["metrics"]["consensus_failed"] += 1
                
                # Enregistrer la décision
                decision_record = {
                    "cycle": cycle,
                    "timestamp": datetime.now().isoformat(),
                    "consensus_vote": consensus_vote,
                    "consensus_reached": consensus_reached,
                    "worker_votes": decisions
                }
                
                self.results["consensus_decisions"].append(decision_record)
                self.results["total_cycles"] = cycle + 1
                
                # Mettre à jour les métriques
                if consensus_vote == "BUY":
                    self.results["metrics"]["buy_signals"] += 1
                elif consensus_vote == "SELL":
                    self.results["metrics"]["sell_signals"] += 1
                else:
                    self.results["metrics"]["hold_signals"] += 1
                
                time.sleep(5)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Sauvegarder les résultats
            results_path = Path("testnet/adan_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ ADAN terminé")
            logger.info(f"Résultats: {results_path}")
            logger.info(f"Cycles: {self.results['total_cycles']}")
            logger.info(f"Consensus: {self.results['metrics']['consensus_reached']}")
            logger.info(f"{'='*60}")

def main():
    # Récupérer les clés
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ Clés API manquantes")
        sys.exit(1)
    
    # Lancer l'orchestrateur
    orchestrator = ADANOrchestrator(api_key, api_secret, testnet=True)
    success = orchestrator.execute(cycles=50)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
