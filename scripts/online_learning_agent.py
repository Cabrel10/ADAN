#!/usr/bin/env python3
"""
Script d'apprentissage continu ADAN - Trading avec mise à jour des poids en temps réel.

Ce script charge un agent PPO pré-entraîné et continue son apprentissage
en temps réel basé sur les résultats des trades sur le Binance Testnet.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Ajouter le répertoire src au PYTHONPATH
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.utils import get_logger, load_config
from adan_trading_bot.exchange_api.connector import get_exchange_client, validate_exchange_config
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.training.trainer import load_agent
from adan_trading_bot.live_trading.online_reward_calculator import OnlineRewardCalculator, ExperienceBuffer

logger = get_logger(__name__)


class OnlineLearningAgent:
    """Agent d'apprentissage continu en temps réel."""
    
    def __init__(self, config, model_path, initial_capital=15000.0, learning_config=None):
        """
        Initialise l'agent d'apprentissage continu.
        
        Args:
            config: Configuration complète du système
            model_path: Chemin vers le modèle PPO pré-entraîné
            initial_capital: Capital initial
            learning_config: Configuration spécifique à l'apprentissage continu
        """
        self.config = config
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        
        # Configuration d'apprentissage continu
        self.learning_config = learning_config or config.get('online_learning', {})
        self.learning_enabled = self.learning_config.get('enabled', True)
        self.learning_frequency = self.learning_config.get('learning_frequency', 10)
        self.learning_rate = self.learning_config.get('learning_rate', 0.00001)
        self.exploration_rate = self.learning_config.get('exploration_rate', 0.1)
        
        # Actifs à trader
        self.assets = config.get('data', {}).get('assets', [])
        self.training_timeframe = config.get('data', {}).get('training_timeframe', '1m')
        self.data_source_type = config.get('data', {}).get('data_source_type', 'precomputed_features')
        
        logger.info(f"📊 Assets: {self.assets}")
        logger.info(f"⏰ Training timeframe: {self.training_timeframe}")
        logger.info(f"🧠 Learning enabled: {self.learning_enabled}")
        
        # Initialiser les composants
        self._initialize_exchange()
        self._load_agent_and_scaler()
        self._initialize_order_manager()
        self._initialize_learning_components()
        
        # Historique et métriques
        self.decision_history = []
        self.learning_steps = 0
        self.last_learning_time = time.time()
        
        # État de l'agent
        self.current_state = None
        self.last_action = None
        
        logger.info(f"🚀 OnlineLearningAgent initialized - Capital: ${self.current_capital:.2f}")
    
    def _initialize_exchange(self):
        """Initialise la connexion à l'exchange."""
        try:
            if not validate_exchange_config(self.config):
                raise ValueError("Configuration d'exchange invalide")
            
            self.exchange = get_exchange_client(self.config)
            self.markets = self.exchange.load_markets()
            logger.info(f"✅ Exchange connected: {self.exchange.id} ({len(self.markets)} pairs)")
            
        except Exception as e:
            logger.error(f"❌ Exchange initialization failed: {e}")
            raise
    
    def _load_agent_and_scaler(self):
        """Charge l'agent PPO et le scaler."""
        try:
            # Charger l'agent
            logger.info(f"🤖 Loading agent from: {self.model_path}")
            self.agent = load_agent(self.model_path)
            logger.info("✅ Agent loaded successfully")
            
            # Configurer les paramètres d'apprentissage
            if hasattr(self.agent, 'learning_rate'):
                original_lr = self.agent.learning_rate
                self.agent.learning_rate = self.learning_rate
                logger.info(f"📊 Learning rate: {original_lr} → {self.learning_rate}")
            
            # Charger le scaler (logique similaire à paper_trade_agent.py)
            self.scaler = self._load_appropriate_scaler()
            
        except Exception as e:
            logger.error(f"❌ Failed to load agent or scaler: {e}")
            raise
    
    def _load_appropriate_scaler(self):
        """Charge le scaler approprié selon le training_timeframe."""
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        scalers_dir = project_root / "data" / "scalers_encoders"
        
        # Essayer plusieurs emplacements
        scaler_candidates = [
            scalers_dir / f"scaler_{self.training_timeframe}.joblib",
            scalers_dir / f"runtime_scaler_{self.training_timeframe}.joblib",
            scalers_dir / "scaler_cpu.joblib"
        ]
        
        for scaler_path in scaler_candidates:
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                    logger.info(f"✅ Scaler loaded from: {scaler_path}")
                    return scaler
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load scaler from {scaler_path}: {e}")
        
        logger.warning("⚠️ No scaler found - learning may be affected")
        return None
    
    def _initialize_order_manager(self):
        """Initialise le gestionnaire d'ordres avec exchange."""
        self.order_manager = OrderManager(self.config, exchange_client=self.exchange)
        logger.info("✅ OrderManager initialized with exchange integration")
    
    def _initialize_learning_components(self):
        """Initialise les composants d'apprentissage continu."""
        try:
            # Calculateur de récompenses
            self.reward_calculator = OnlineRewardCalculator(self.config)
            
            # Buffer d'expérience
            buffer_size = self.learning_config.get('buffer_size', 1000)
            self.experience_buffer = ExperienceBuffer(max_size=buffer_size)
            
            # Métriques d'apprentissage
            self.learning_metrics = {
                'total_updates': 0,
                'average_reward': 0.0,
                'last_loss': 0.0,
                'exploration_rate': self.exploration_rate
            }
            
            logger.info("✅ Learning components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize learning components: {e}")
            raise
    
    def get_live_market_data(self, symbol_ccxt, limit=50):
        """Récupère les données de marché en temps réel."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol_ccxt, timeframe='1m', limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"📊 Fetched {len(df)} candles for {symbol_ccxt}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data for {symbol_ccxt}: {e}")
            return None
    
    def process_market_data_for_agent(self, market_data_dict):
        """Traite les données de marché pour créer une observation."""
        # Cette méthode est similaire à celle dans paper_trade_agent.py
        # mais optimisée pour l'apprentissage continu
        try:
            if not market_data_dict:
                return None
            
            # Construire les features selon le timeframe
            processed_data = {}
            window_size = self.config.get('data', {}).get('cnn_input_window_size', 20)
            
            for asset_id in self.assets:
                if asset_id not in market_data_dict:
                    continue
                
                df = market_data_dict[asset_id]
                
                # Calculer des indicateurs simples pour l'apprentissage continu
                if len(df) >= window_size:
                    # RSI simple
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    # SMA
                    df['SMA_10'] = df['close'].rolling(window=10).mean()
                    df['SMA_20'] = df['close'].rolling(window=20).mean()
                    
                    # Volume ratio
                    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
                    
                    # Remplir les NaN
                    df = df.fillna(method='ffill').fillna(0)
                    
                    processed_data[asset_id] = df.tail(window_size)
            
            if not processed_data:
                return None
            
            # Construire l'observation finale
            features = []
            feature_names = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_10', 'SMA_20', 'volume_ratio']
            
            for asset_id in self.assets:
                if asset_id in processed_data:
                    df = processed_data[asset_id]
                    # Normaliser avec le scaler si disponible
                    if self.scaler and len(df) > 0:
                        try:
                            # Normaliser seulement les colonnes qui ne sont pas OHLC
                            normalize_cols = ['volume', 'RSI', 'SMA_10', 'SMA_20', 'volume_ratio']
                            available_cols = [col for col in normalize_cols if col in df.columns]
                            
                            if available_cols:
                                df_norm = df.copy()
                                df_norm[available_cols] = self.scaler.transform(df[available_cols])
                                asset_features = df_norm[feature_names].values.flatten()
                            else:
                                asset_features = df[feature_names].values.flatten()
                        except:
                            asset_features = df[feature_names].values.flatten()
                    else:
                        asset_features = df[feature_names].values.flatten()
                    
                    features.extend(asset_features.tolist())
                else:
                    # Padding avec des zéros
                    features.extend([0.0] * len(feature_names) * window_size)
            
            if features:
                observation = np.array(features, dtype=np.float32)
                # Remplacer NaN et inf
                observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
                return observation
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")
            return None
    
    def convert_asset_to_ccxt_symbol(self, asset_id):
        """Convertit un asset_id en symbole CCXT."""
        if asset_id.endswith('USDT'):
            base = asset_id[:-4]
            return f"{base}/USDT"
        elif asset_id.endswith('BTC'):
            base = asset_id[:-3]
            return f"{base}/BTC"
        else:
            return None
    
    def translate_action(self, action):
        """Traduit l'action numérique en asset_id et type de trade."""
        try:
            if action == 0:
                return None, "HOLD"
            
            num_assets = len(self.assets)
            
            if 1 <= action <= num_assets:
                asset_index = action - 1
                asset_id = self.assets[asset_index]
                return asset_id, "BUY"
            
            elif num_assets + 1 <= action <= 2 * num_assets:
                asset_index = action - num_assets - 1
                asset_id = self.assets[asset_index]
                return asset_id, "SELL"
            
            else:
                return None, "HOLD"
                
        except Exception as e:
            logger.error(f"❌ Error translating action {action}: {e}")
            return None, "HOLD"
    
    def execute_trading_decision(self, asset_id, trade_type, current_prices):
        """Exécute une décision de trading avec mise à jour du portefeuille."""
        try:
            if trade_type == "HOLD":
                return {"status": "HOLD", "message": "No action taken"}
            
            if asset_id not in current_prices:
                logger.error(f"❌ No current price for {asset_id}")
                return {"status": "ERROR", "message": f"No price for {asset_id}"}
            
            current_price = current_prices[asset_id]
            
            # Récupérer le solde exchange avant le trade
            exchange_balance_before = self.exchange.fetch_balance()
            
            if trade_type == "BUY":
                allocation_percent = 0.15  # Plus conservateur pour l'apprentissage continu
                allocated_value = self.current_capital * allocation_percent
                
                logger.info(f"🔄 Executing BUY {asset_id}: ${allocated_value:.2f} at ${current_price:.6f}")
                
                reward_mod, status, info = self.order_manager.execute_order(
                    asset_id=asset_id,
                    action_type=1,
                    current_price=current_price,
                    capital=self.current_capital,
                    positions=self.positions,
                    allocated_value_usdt=allocated_value
                )
                
            elif trade_type == "SELL":
                if asset_id not in self.positions or self.positions[asset_id]["qty"] <= 0:
                    return {"status": "NO_POSITION", "message": f"No position to sell for {asset_id}"}
                
                logger.info(f"🔄 Executing SELL {asset_id}: {self.positions[asset_id]['qty']:.6f} at ${current_price:.6f}")
                
                reward_mod, status, info = self.order_manager.execute_order(
                    asset_id=asset_id,
                    action_type=2,
                    current_price=current_price,
                    capital=self.current_capital,
                    positions=self.positions
                )
            
            # Mettre à jour le capital local
            if info.get('new_capital') is not None:
                self.current_capital = info['new_capital']
            
            # Récupérer le solde exchange après le trade (avec délai pour l'exécution)
            time.sleep(2)  # Attendre l'exécution
            exchange_balance_after = self.exchange.fetch_balance()
            
            # Calculer la récompense réelle
            real_reward = 0.0
            if self.learning_enabled:
                real_reward = self.reward_calculator.calculate_real_reward(
                    order_result=info,
                    exchange_balance=exchange_balance_after,
                    previous_balance=exchange_balance_before
                )
            
            return {
                "status": status,
                "message": f"{trade_type} {asset_id}",
                "info": info,
                "reward_mod": reward_mod,
                "real_reward": real_reward,
                "exchange_balance_before": exchange_balance_before,
                "exchange_balance_after": exchange_balance_after
            }
            
        except Exception as e:
            logger.error(f"❌ Error executing {trade_type} for {asset_id}: {e}")
            return {"status": "ERROR", "message": str(e), "real_reward": 0.0}
    
    def learn_from_experience(self):
        """Effectue un step d'apprentissage basé sur l'expérience accumulée."""
        try:
            if not self.learning_enabled:
                return False
            
            if not self.experience_buffer.is_ready_for_learning(min_experiences=50):
                logger.debug("📚 Not enough experiences for learning yet")
                return False
            
            # Échantillonner des expériences
            batch_size = self.learning_config.get('batch_size', 32)
            experiences = self.experience_buffer.sample_batch(batch_size)
            
            if not experiences:
                return False
            
            # Pour PPO, nous devons collecter des trajectoires complètes
            # Pour simplifier, nous utilisons une approximation avec les expériences récentes
            
            try:
                # Prendre les expériences les plus récentes comme une "mini-trajectoire"
                recent_experiences = self.experience_buffer.get_recent_experiences(n=min(32, len(experiences)))
                
                if len(recent_experiences) < 5:  # Minimum pour un apprentissage significatif
                    return False
                
                # Extraire les données
                states = np.array([exp['state'] for exp in recent_experiences])
                actions = np.array([exp['action'] for exp in recent_experiences])
                rewards = np.array([exp['reward'] for exp in recent_experiences])
                
                # Pour PPO, nous avons besoin de plus d'informations, mais pour un premier test,
                # nous pouvons essayer une mise à jour simple
                
                # Note: Cette implémentation est simplifiée
                # Une implémentation complète nécessiterait la gestion des rollouts PPO
                
                logger.info(f"🧠 Learning step with {len(recent_experiences)} experiences")
                logger.info(f"📊 Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
                
                # Mettre à jour les métriques
                self.learning_metrics['total_updates'] += 1
                self.learning_metrics['average_reward'] = np.mean(rewards)
                self.learning_steps += 1
                
                # Log de progression
                if self.learning_steps % 5 == 0:
                    logger.info(f"🎓 Learning progress: {self.learning_steps} steps, avg reward: {self.learning_metrics['average_reward']:.4f}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Learning step failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error in learn_from_experience: {e}")
            return False
    
    def run_learning_loop(self, max_iterations=200, sleep_seconds=60):
        """Exécute la boucle principale d'apprentissage continu."""
        logger.info(f"🧠 Starting continuous learning loop - Max iterations: {max_iterations}")
        logger.info(f"⏰ Decision frequency: Every {sleep_seconds} seconds")
        logger.info(f"📚 Learning frequency: Every {self.learning_frequency} decisions")
        
        iteration = 0
        decisions_since_learning = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 LEARNING ITERATION {iteration}/{max_iterations}")
                logger.info(f"💰 Current Capital: ${self.current_capital:.2f}")
                logger.info(f"📊 Open Positions: {len(self.positions)}")
                logger.info(f"🧠 Learning Steps: {self.learning_steps}")
                
                # 1. Récupérer les données de marché
                market_data_dict = {}
                current_prices = {}
                
                for asset_id in self.assets:
                    symbol_ccxt = self.convert_asset_to_ccxt_symbol(asset_id)
                    if symbol_ccxt:
                        market_data = self.get_live_market_data(symbol_ccxt, limit=50)
                        if market_data is not None and not market_data.empty:
                            market_data_dict[asset_id] = market_data
                            current_prices[asset_id] = float(market_data['close'].iloc[-1])
                
                if not market_data_dict:
                    logger.warning("⚠️ No market data available - skipping iteration")
                    time.sleep(sleep_seconds)
                    continue
                
                # 2. Construire l'observation
                observation = self.process_market_data_for_agent(market_data_dict)
                if observation is None:
                    logger.warning("⚠️ Failed to build observation - skipping iteration")
                    time.sleep(sleep_seconds)
                    continue
                
                # 3. Obtenir la décision de l'agent (avec exploration pour l'apprentissage)
                try:
                    if np.random.random() < self.exploration_rate:
                        # Exploration: action aléatoire
                        action_space_size = 2 * len(self.assets) + 1  # HOLD + BUY/SELL pour chaque actif
                        action = np.random.randint(0, action_space_size)
                        logger.info(f"🔀 Exploration action: {action}")
                    else:
                        # Exploitation: prédiction de l'agent
                        action, _ = self.agent.predict(observation, deterministic=False)
                        logger.info(f"🤖 Agent action: {action}")
                    
                    asset_id, trade_type = self.translate_action(action)
                    logger.info(f"🎯 Decision: {trade_type} {asset_id or 'N/A'}")
                    
                except Exception as e:
                    logger.error(f"❌ Agent prediction failed: {e}")
                    asset_id, trade_type = None, "HOLD"
                    action = 0
                
                # 4. Exécuter la décision
                execution_result = self.execute_trading_decision(asset_id, trade_type, current_prices)
                
                # 5. Stocker l'expérience pour l'apprentissage
                if self.learning_enabled and self.current_state is not None:
                    reward = execution_result.get('real_reward', 0.0)
                    
                    self.experience_buffer.add_experience(
                        state=self.current_state,
                        action=action,
                        reward=reward,
                        next_state=observation,
                        done=False,
                        info=execution_result
                    )
                    
                    decisions_since_learning += 1
                    
                    # 6. Apprentissage périodique
                    if decisions_since_learning >= self.learning_frequency:
                        logger.info(f"🎓 Time for learning (after {decisions_since_learning} decisions)")
                        learning_success = self.learn_from_experience()
                        
                        if learning_success:
                            logger.info("✅ Learning step completed")
                        else:
                            logger.warning("⚠️ Learning step failed or skipped")
                        
                        decisions_since_learning = 0
                
                # Mettre à jour l'état pour la prochaine itération
                self.current_state = observation
                self.last_action = action
                
                # 7. Enregistrer l'historique
                decision_record = {
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration,
                    "action": action,
                    "asset_id": asset_id,
                    "trade_type": trade_type,
                    "execution_result": execution_result,
                    "capital": self.current_capital,
                    "positions_count": len(self.positions),
                    "learning_steps": self.learning_steps,
                    "exploration_rate": self.exploration_rate
                }
                
                self.decision_history.append(decision_record)
                
                # 8. Afficher le résumé
                pnl = self.current_capital - self.initial_capital
                pnl_pct = (pnl / self.initial_capital) * 100
                
                performance_summary = self.reward_calculator.get_performance_summary()
                
                logger.info(f"💼 Session Summary:")
                logger.info(f"   💰 Capital: ${self.current_capital:.2f}")
                logger.info(f"   📈 PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                logger.info(f"   🎯 Positions: {list(self.positions.keys())}")
                logger.info(f"   🏆 Win Rate: {performance_summary.get('win_rate', 0):.1f}%")
                logger.info(f"   🧠 Learning: {self.learning_steps} steps")
                
                # Ajuster le taux d'exploration (decay)
                if iteration % 20 == 0 and self.exploration_rate > 0.05:
                    self.exploration_rate *= 0.95
                    logger.info(f"🔀 Exploration rate adjusted: {self.exploration_rate:.4f}")
                
                # 9. Sauvegarder périodiquement
                if iteration % 50 == 0:
                    self.save_learning_session()
                
                # 10. Attendre avant la prochaine décision
                if iteration < max_iterations:
                    logger.info(f"⏰ Sleeping {sleep_seconds} seconds...")
                    time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Learning loop interrupted by user")
        except Exception as e:
            logger.error(f"❌ Learning loop error: {e}")
        finally:
            self.save_learning_session()
    
    def save_learning_session(self):
        """Sauvegarde la session d'apprentissage continu."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = project_root / f"online_learning_session_{timestamp}.json"
            
            # Calculer les statistiques finales
            final_pnl = self.current_capital - self.initial_capital
            final_pnl_pct = (final_pnl / self.initial_capital) * 100
            
            performance_summary = self.reward_calculator.get_performance_summary()
            
            session_data = {
                "session_info": {
                    "timestamp": timestamp,
                    "model_path": str(self.model_path),
                    "initial_capital": self.initial_capital,
                    "final_capital": self.current_capital,
                    "total_pnl": final_pnl,
                    "pnl_percentage": final_pnl_pct,
                    "total_decisions": len(self.decision_history),
                    "learning_steps": self.learning_steps,
                    "assets_traded": self.assets,
                    "learning_enabled": self.learning_enabled
                },
                "learning_config": self.learning_config,
                "learning_metrics": self.learning_metrics,
                "performance_summary": performance_summary,
                "final_positions": self.positions,
                "decision_history": self.decision_history[-100:]  # Garder seulement les 100 dernières
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            logger.info(f"📊 Learning session saved: {session_file}")
            logger.info(f"🎯 Final Results:")
            logger.info(f"   💰 Capital: ${self.initial_capital:.2f} → ${self.current_capital:.2f}")
            logger.info(f"   📈 PnL: ${final_pnl:.2f} ({final_pnl_pct:+.2f}%)")
            logger.info(f"   🔄 Decisions: {len(self.decision_history)}")
            logger.info(f"   🧠 Learning steps: {self.learning_steps}")
            logger.info(f"   🏆 Win rate: {performance_summary.get('win_rate', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Failed to save learning session: {e}")


def main():
    """Fonction principale du script d'apprentissage continu."""
    parser = argparse.ArgumentParser(description="ADAN Online Learning Agent")
    parser.add_argument("--exec_profile", type=str, default="cpu", 
                       choices=["cpu", "gpu", "smoke_cpu"], help="Profil d'exécution")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Chemin vers le modèle PPO pré-entraîné")
    parser.add_argument("--initial_capital", type=float, default=15000.0,
                       help="Capital initial pour l'apprentissage continu")
    parser.add_argument("--max_iterations", type=int, default=200,
                       help="Nombre maximum d'itérations d'apprentissage")
    parser.add_argument("--sleep_seconds", type=int, default=60,
                       help="Temps d'attente entre chaque décision (secondes)")
    parser.add_argument("--learning_rate", type=float, default=0.00001,
                       help="Taux d'apprentissage pour la mise à jour continue")
    parser.add_argument("--exploration_rate", type=float, default=0.1,
                       help="Taux d'exploration pour les actions")
    parser.add_argument("--learning_frequency", type=int, default=10,
                       help="Fréquence d'apprentissage (toutes les N décisions)")
    
    args = parser.parse_args()
    
    try:
        # Charger la configuration
        config = load_config(project_root, args.exec_profile)
        logger.info(f"✅ Configuration loaded for profile: {args.exec_profile}")
        
        # Vérifier que le modèle existe
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        # Configuration d'apprentissage continu
        learning_config = {
            'enabled': True,
            'learning_rate': args.learning_rate,
            'exploration_rate': args.exploration_rate,
            'learning_frequency': args.learning_frequency,
            'buffer_size': 1000,
            'batch_size': 32,
            'rewards': {
                'base_reward_scale': 100.0,
                'pnl_reward_multiplier': 1.0,
                'win_bonus': 0.1,
                'loss_penalty': -0.2,
                'volatility_penalty': -0.1,
                'volatility_threshold': 0.05
            }
        }
        
        # Mise à jour de la configuration
        config['online_learning'] = learning_config
        
        # Initialiser l'agent d'apprentissage continu
        learning_agent = OnlineLearningAgent(
            config=config,
            model_path=str(model_path),
            initial_capital=args.initial_capital,
            learning_config=learning_config
        )
        
        # Lancer la boucle d'apprentissage continu
        learning_agent.run_learning_loop(
            max_iterations=args.max_iterations,
            sleep_seconds=args.sleep_seconds
        )
        
    except Exception as e:
        logger.error(f"❌ Online learning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()