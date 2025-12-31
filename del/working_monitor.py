#!/usr/bin/env python3
"""
Monitor ADAN fonctionnel avec données préchargées
Version simplifiée qui fonctionne immédiatement
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import time
from datetime import datetime
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkingADANMonitor:
    """Monitor ADAN fonctionnel avec adaptation légère"""
    
    def __init__(self):
        self.data_dir = Path("historical_data")
        self.data = {}
        self.portfolio = {
            'balance': 29.0,
            'position': 0.0,
            'trades_today': 0,
            'total_trades': 0,
            'pnl': 0.0
        }
        
        # Workers avec adaptation légère
        self.workers = {
            'w1': {'weight': 0.25, 'performance': []},
            'w2': {'weight': 0.25, 'performance': []},
            'w3': {'weight': 0.25, 'performance': []},
            'w4': {'weight': 0.25, 'performance': []}
        }
        
        self.global_confidence = 0.75
        self.last_decision = None
        self.adaptation_enabled = True
        
    def load_data(self):
        """Charge les données préchargées"""
        logger.info("📂 Chargement des données préchargées...")
        
        timeframes = ['5m', '1h', '4h']
        for tf in timeframes:
            file_path = self.data_dir / f"BTC_USDT_{tf}_data.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                self.data[tf] = df
                logger.info(f"✅ {tf}: {len(df)} périodes chargées")
            else:
                logger.error(f"❌ Fichier manquant: {file_path}")
                return False
        
        return len(self.data) == 3
    
    def get_market_signals(self):
        """Analyse les signaux de marché"""
        signals = {}
        
        for tf, df in self.data.items():
            latest = df.iloc[-1]
            
            # Signal RSI
            rsi = latest.get('rsi', 50)
            if rsi > 70:
                rsi_signal = -1  # Survente
            elif rsi < 30:
                rsi_signal = 1   # Surachat
            else:
                rsi_signal = 0   # Neutre
            
            # Signal ADX (force de tendance)
            adx = latest.get('adx', 25)
            trend_strength = min(adx / 50, 1.0)  # Normaliser
            
            # Signal volatilité
            volatility = latest.get('volatility', 0.5)
            vol_factor = min(volatility * 100, 1.0)  # Normaliser
            
            # Tendance prix (5 dernières périodes)
            if len(df) >= 5:
                recent_prices = df['close'].tail(5)
                price_trend = 1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1
            else:
                price_trend = 0
            
            signals[tf] = {
                'rsi_signal': rsi_signal,
                'trend_strength': trend_strength,
                'volatility_factor': vol_factor,
                'price_trend': price_trend,
                'rsi_value': rsi,
                'adx_value': adx,
                'price': latest['close']
            }
        
        return signals
    
    def simulate_worker_decisions(self, signals):
        """Simule les décisions des workers avec variabilité"""
        decisions = {}
        
        for worker_id, worker_data in self.workers.items():
            # Chaque worker a une sensibilité différente
            if worker_id == 'w1':  # Sensible au RSI court terme
                primary_signal = signals['5m']['rsi_signal']
                noise = random.uniform(-0.3, 0.3)
            elif worker_id == 'w2':  # Suit la tendance moyen terme
                primary_signal = signals['1h']['price_trend']
                noise = random.uniform(-0.2, 0.2)
            elif worker_id == 'w3':  # Sensible à la volatilité
                vol_factor = signals['4h']['volatility_factor']
                primary_signal = 1 if vol_factor > 0.7 else -1 if vol_factor < 0.3 else 0
                noise = random.uniform(-0.4, 0.4)
            else:  # w4 - Conservateur
                # Moyenne des signaux
                avg_signal = np.mean([s['rsi_signal'] for s in signals.values()])
                primary_signal = 1 if avg_signal > 0.3 else -1 if avg_signal < -0.3 else 0
                noise = random.uniform(-0.1, 0.1)
            
            # Ajouter du bruit et de la variabilité
            final_decision = primary_signal + noise
            
            # Convertir en action discrète
            if final_decision > 0.2:
                action = 1  # BUY
            elif final_decision < -0.2:
                action = -1  # SELL
            else:
                action = 0  # HOLD
            
            decisions[worker_id] = action
        
        return decisions
    
    def calculate_ensemble_decision(self, worker_decisions):
        """Calcule la décision d'ensemble avec poids adaptatifs"""
        weighted_sum = 0
        total_weight = 0
        
        for worker_id, decision in worker_decisions.items():
            weight = self.workers[worker_id]['weight']
            weighted_sum += decision * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_score = weighted_sum / total_weight
        else:
            ensemble_score = 0
        
        # Convertir en décision finale
        if ensemble_score > 0.3:
            final_action = 1  # BUY
        elif ensemble_score < -0.3:
            final_action = -1  # SELL
        else:
            final_action = 0  # HOLD
        
        # Calculer la confiance
        agreement = sum(1 for d in worker_decisions.values() if d == final_action)
        confidence = agreement / len(worker_decisions)
        
        return {
            'action': final_action,
            'confidence': confidence,
            'ensemble_score': ensemble_score,
            'worker_decisions': worker_decisions
        }
    
    def adapt_worker_weights(self, trade_result):
        """Adaptation légère des poids basée sur les performances"""
        if not self.adaptation_enabled:
            return
        
        learning_rate = 0.01
        
        for worker_id, worker_data in self.workers.items():
            # Ajouter le résultat à l'historique
            worker_data['performance'].append(trade_result)
            
            # Garder seulement les 20 derniers trades
            if len(worker_data['performance']) > 20:
                worker_data['performance'].pop(0)
            
            # Ajuster le poids basé sur la performance récente
            if len(worker_data['performance']) >= 5:
                recent_perf = np.mean(worker_data['performance'][-5:])
                adjustment = learning_rate * recent_perf
                
                # Appliquer l'ajustement
                new_weight = worker_data['weight'] + adjustment
                worker_data['weight'] = max(0.05, min(0.95, new_weight))
        
        # Renormaliser les poids
        total_weight = sum(w['weight'] for w in self.workers.values())
        if total_weight > 0:
            for worker_data in self.workers.values():
                worker_data['weight'] /= total_weight
        
        # Mettre à jour la confiance globale
        self.update_global_confidence()
    
    def update_global_confidence(self):
        """Met à jour la confiance globale basée sur la stabilité des poids"""
        weights = [w['weight'] for w in self.workers.values()]
        weight_std = np.std(weights)
        
        # Plus les poids sont équilibrés, plus la confiance est élevée
        balance_factor = 1.0 - min(weight_std * 4, 0.3)
        
        # Ajuster selon les performances récentes
        recent_performances = []
        for worker_data in self.workers.values():
            if worker_data['performance']:
                recent_performances.extend(worker_data['performance'][-3:])
        
        if recent_performances:
            avg_perf = np.mean(recent_performances)
            perf_factor = 1.0 + min(avg_perf * 5, 0.2)
        else:
            perf_factor = 1.0
        
        # Calculer la nouvelle confiance
        base_confidence = np.mean(weights)
        self.global_confidence = min(max(base_confidence * balance_factor * perf_factor, 0.3), 0.95)
    
    def execute_trade(self, decision):
        """Simule l'exécution d'un trade"""
        action = decision['action']
        confidence = decision['confidence']
        
        if action == 1:  # BUY
            logger.info(f"🟢 SIGNAL BUY (confiance: {confidence:.2f})")
            # Simuler un résultat de trade
            trade_result = random.uniform(-0.5, 1.0)  # -0.5% à +1.0%
            
        elif action == -1:  # SELL
            logger.info(f"🔴 SIGNAL SELL (confiance: {confidence:.2f})")
            trade_result = random.uniform(-0.5, 1.0)
            
        else:  # HOLD
            logger.info(f"⚪ SIGNAL HOLD (confiance: {confidence:.2f})")
            trade_result = 0
        
        # Mettre à jour le portfolio
        if trade_result != 0:
            self.portfolio['pnl'] += trade_result
            self.portfolio['balance'] += trade_result
            self.portfolio['total_trades'] += 1
            self.portfolio['trades_today'] += 1
            
            # Adapter les poids
            self.adapt_worker_weights(trade_result)
            
            logger.info(f"💰 Trade PnL: {trade_result:+.2f}% | Balance: ${self.portfolio['balance']:.2f}")
        
        return trade_result
    
    def log_status(self, signals, decision):
        """Log du statut complet"""
        logger.info("="*60)
        logger.info("📊 STATUT ADAN")
        logger.info("="*60)
        
        # Signaux de marché
        logger.info("📈 Signaux de marché:")
        for tf, signal in signals.items():
            logger.info(f"   {tf}: RSI={signal['rsi_value']:.1f}, ADX={signal['adx_value']:.1f}, Prix=${signal['price']:.2f}")
        
        # Décisions des workers
        logger.info("🤖 Décisions des workers:")
        for worker_id, worker_decision in decision['worker_decisions'].items():
            weight = self.workers[worker_id]['weight']
            action_str = {1: "BUY", -1: "SELL", 0: "HOLD"}[worker_decision]
            logger.info(f"   {worker_id}: {action_str} (poids: {weight:.3f})")
        
        # Portfolio
        logger.info("💰 Portfolio:")
        logger.info(f"   Balance: ${self.portfolio['balance']:.2f}")
        logger.info(f"   PnL total: {self.portfolio['pnl']:+.2f}%")
        logger.info(f"   Trades aujourd'hui: {self.portfolio['trades_today']}")
        logger.info(f"   Confiance globale: {self.global_confidence:.2f}")
    
    def run(self):
        """Boucle principale du monitor"""
        logger.info("🚀 DÉMARRAGE ADAN MONITOR AVEC DONNÉES PRÉCHARGÉES")
        logger.info("="*60)
        
        # Charger les données
        if not self.load_data():
            logger.error("❌ Impossible de charger les données")
            return
        
        logger.info("✅ Données chargées avec succès")
        logger.info(f"📊 Timeframes disponibles: {list(self.data.keys())}")
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                logger.info(f"\n🔄 CYCLE {cycle} - {datetime.now().strftime('%H:%M:%S')}")
                
                # 1. Analyser les signaux de marché
                signals = self.get_market_signals()
                
                # 2. Simuler les décisions des workers
                worker_decisions = self.simulate_worker_decisions(signals)
                
                # 3. Calculer la décision d'ensemble
                ensemble_decision = self.calculate_ensemble_decision(worker_decisions)
                
                # 4. Exécuter le trade
                trade_result = self.execute_trade(ensemble_decision)
                
                # 5. Logger le statut
                self.log_status(signals, ensemble_decision)
                
                # 6. Attendre avant le prochain cycle
                wait_time = random.randint(30, 120)  # 30s à 2min
                logger.info(f"⏳ Attente {wait_time}s avant prochain cycle...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")

def main():
    """Fonction principale"""
    monitor = WorkingADANMonitor()
    monitor.run()

if __name__ == "__main__":
    main()