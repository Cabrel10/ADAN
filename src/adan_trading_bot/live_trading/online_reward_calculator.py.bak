"""
Module de calcul de récompenses en temps réel pour l'apprentissage continu.
Calcule les récompenses basées sur les résultats réels des trades sur l'exchange.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class OnlineRewardCalculator:
    """
    Calculateur de récompenses pour l'apprentissage continu en temps réel.
    Utilise les résultats réels des trades sur l'exchange pour calculer les récompenses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le calculateur de récompenses.
        
        Args:
            config: Configuration complète du système
        """
        self.config = config
        
        # Configuration des récompenses
        reward_config = config.get('online_learning', {}).get('rewards', {})
        
        # Paramètres de base
        self.base_reward_scale = reward_config.get('base_reward_scale', 100.0)
        self.pnl_reward_multiplier = reward_config.get('pnl_reward_multiplier', 1.0)
        self.win_bonus = reward_config.get('win_bonus', 0.1)
        self.loss_penalty = reward_config.get('loss_penalty', -0.2)
        
        # Pénalités spéciales
        self.volatility_penalty = reward_config.get('volatility_penalty', -0.1)
        self.volatility_threshold = reward_config.get('volatility_threshold', 0.05)
        self.large_loss_penalty = reward_config.get('large_loss_penalty', -0.5)
        self.large_loss_threshold = reward_config.get('large_loss_threshold', 0.03)
        
        # Bonus temporels
        self.quick_profit_bonus = reward_config.get('quick_profit_bonus', 0.2)
        self.quick_profit_time_threshold = reward_config.get('quick_profit_time_threshold', 300)  # 5 minutes
        
        # Historique pour calculs
        self.previous_portfolio_value = None
        self.trade_history = []
        self.performance_window = []
        self.window_size = reward_config.get('performance_window_size', 10)
        
        # Métriques de suivi
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.cumulative_reward = 0.0
        
        logger.info(f"✅ OnlineRewardCalculator initialized with scale={self.base_reward_scale}")
    
    def calculate_real_reward(self, 
                            order_result: Dict[str, Any], 
                            exchange_balance: Dict[str, float],
                            previous_balance: Optional[Dict[str, float]] = None,
                            market_context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calcule la récompense basée sur les résultats réels de l'exchange.
        
        Args:
            order_result: Résultat de l'ordre exécuté (de OrderManager)
            exchange_balance: Solde actuel du compte exchange
            previous_balance: Solde précédent pour comparaison
            market_context: Contexte de marché (prix, volatilité, etc.)
            
        Returns:
            float: Récompense calculée
        """
        try:
            # Calculer la valeur actuelle du portefeuille
            current_value = self._calculate_portfolio_value(exchange_balance)
            
            if self.previous_portfolio_value is None:
                self.previous_portfolio_value = current_value
                logger.debug(f"📊 Initial portfolio value: ${current_value:.2f}")
                return 0.0
            
            # Calculer le changement de valeur
            portfolio_change = current_value - self.previous_portfolio_value
            portfolio_change_pct = portfolio_change / self.previous_portfolio_value if self.previous_portfolio_value > 0 else 0.0
            
            # Récompense de base basée sur le changement de valeur
            base_reward = portfolio_change_pct * self.base_reward_scale
            
            # Analyser les détails du trade
            trade_reward = self._calculate_trade_specific_reward(order_result, portfolio_change)
            
            # Bonus/malus contextuels
            context_reward = self._calculate_context_reward(market_context, portfolio_change_pct)
            
            # Récompense de performance temporelle
            temporal_reward = self._calculate_temporal_reward(order_result, portfolio_change)
            
            # Récompense totale
            total_reward = base_reward + trade_reward + context_reward + temporal_reward
            
            # Appliquer les limites
            total_reward = np.clip(total_reward, -2.0, 2.0)
            
            # Mettre à jour l'historique
            self._update_history(order_result, portfolio_change, total_reward)
            
            # Mettre à jour les métriques
            self._update_metrics(portfolio_change, total_reward)
            
            self.previous_portfolio_value = current_value
            
            logger.info(f"💰 Reward calculated: {total_reward:.4f} (base={base_reward:.4f}, trade={trade_reward:.4f}, context={context_reward:.4f}, temporal={temporal_reward:.4f})")
            logger.debug(f"📊 Portfolio: ${self.previous_portfolio_value:.2f} → ${current_value:.2f} ({portfolio_change_pct:+.2%})")
            
            return total_reward
            
        except Exception as e:
            logger.error(f"❌ Error calculating reward: {e}")
            return 0.0
    
    def _calculate_portfolio_value(self, balance: Dict[str, float]) -> float:
        """
        Calcule la valeur totale du portefeuille en USDT.
        
        Args:
            balance: Soldes par devise
            
        Returns:
            float: Valeur totale en USDT
        """
        try:
            # Pour simplifier, on suppose que la valeur principale est en USDT
            usdt_value = balance.get('USDT', 0.0)
            
            # TODO: Ajouter la conversion des autres devises en USDT
            # Pour l'instant, on utilise seulement USDT comme proxy
            
            return usdt_value
            
        except Exception as e:
            logger.error(f"❌ Error calculating portfolio value: {e}")
            return 0.0
    
    def _calculate_trade_specific_reward(self, order_result: Dict[str, Any], portfolio_change: float) -> float:
        """Calcule la récompense spécifique au type de trade."""
        try:
            trade_reward = 0.0
            
            status = order_result.get('status', '')
            
            if 'BUY' in status:
                # Bonus/malus pour les achats
                if portfolio_change > 0:
                    trade_reward += self.win_bonus
                    logger.debug(f"📈 BUY win bonus: +{self.win_bonus}")
                else:
                    trade_reward += self.loss_penalty * 0.5  # Pénalité réduite pour BUY
                    logger.debug(f"📉 BUY loss penalty: {self.loss_penalty * 0.5}")
                    
            elif 'SELL' in status:
                # Bonus/malus pour les ventes
                if portfolio_change > 0:
                    trade_reward += self.win_bonus * 1.5  # Bonus augmenté pour prendre des profits
                    logger.debug(f"📈 SELL profit bonus: +{self.win_bonus * 1.5}")
                else:
                    trade_reward += self.loss_penalty  # Pénalité normale pour vendre à perte
                    logger.debug(f"📉 SELL loss penalty: {self.loss_penalty}")
            
            return trade_reward
            
        except Exception as e:
            logger.error(f"❌ Error calculating trade specific reward: {e}")
            return 0.0
    
    def _calculate_context_reward(self, market_context: Optional[Dict[str, Any]], portfolio_change_pct: float) -> float:
        """Calcule les récompenses/pénalités contextuelles."""
        try:
            context_reward = 0.0
            
            # Pénalité pour volatilité excessive
            if abs(portfolio_change_pct) > self.volatility_threshold:
                context_reward += self.volatility_penalty
                logger.debug(f"⚡ Volatility penalty: {self.volatility_penalty} (change: {portfolio_change_pct:.2%})")
            
            # Pénalité pour grosses pertes
            if portfolio_change_pct < -self.large_loss_threshold:
                context_reward += self.large_loss_penalty
                logger.debug(f"💥 Large loss penalty: {self.large_loss_penalty} (loss: {portfolio_change_pct:.2%})")
            
            # Bonus pour performance consistante
            if len(self.performance_window) >= 3:
                recent_changes = self.performance_window[-3:]
                if all(change > 0 for change in recent_changes):
                    context_reward += 0.1
                    logger.debug(f"🎯 Consistency bonus: +0.1")
            
            return context_reward
            
        except Exception as e:
            logger.error(f"❌ Error calculating context reward: {e}")
            return 0.0
    
    def _calculate_temporal_reward(self, order_result: Dict[str, Any], portfolio_change: float) -> float:
        """Calcule les récompenses basées sur le timing."""
        try:
            temporal_reward = 0.0
            
            # Bonus pour profits rapides
            if portfolio_change > 0 and len(self.trade_history) > 0:
                last_trade_time = self.trade_history[-1].get('timestamp', time.time())
                time_since_last_trade = time.time() - last_trade_time
                
                if time_since_last_trade < self.quick_profit_time_threshold:
                    temporal_reward += self.quick_profit_bonus
                    logger.debug(f"⚡ Quick profit bonus: +{self.quick_profit_bonus} (time: {time_since_last_trade:.0f}s)")
            
            return temporal_reward
            
        except Exception as e:
            logger.error(f"❌ Error calculating temporal reward: {e}")
            return 0.0
    
    def _update_history(self, order_result: Dict[str, Any], portfolio_change: float, reward: float):
        """Met à jour l'historique des trades et performances."""
        try:
            # Ajouter à l'historique des trades
            trade_record = {
                'timestamp': time.time(),
                'order_result': order_result,
                'portfolio_change': portfolio_change,
                'reward': reward,
                'portfolio_value': self.previous_portfolio_value
            }
            
            self.trade_history.append(trade_record)
            
            # Limiter la taille de l'historique
            max_history = 1000
            if len(self.trade_history) > max_history:
                self.trade_history = self.trade_history[-max_history:]
            
            # Mettre à jour la fenêtre de performance
            self.performance_window.append(portfolio_change)
            if len(self.performance_window) > self.window_size:
                self.performance_window.pop(0)
            
        except Exception as e:
            logger.error(f"❌ Error updating history: {e}")
    
    def _update_metrics(self, portfolio_change: float, reward: float):
        """Met à jour les métriques de suivi."""
        try:
            self.total_trades += 1
            self.total_pnl += portfolio_change
            self.cumulative_reward += reward
            
            if portfolio_change > 0:
                self.winning_trades += 1
            
            # Log des métriques périodiquement
            if self.total_trades % 10 == 0:
                win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
                avg_reward = self.cumulative_reward / self.total_trades if self.total_trades > 0 else 0
                
                logger.info(f"📊 Metrics Update:")
                logger.info(f"   🔢 Total trades: {self.total_trades}")
                logger.info(f"   🎯 Win rate: {win_rate:.1f}%")
                logger.info(f"   💰 Total PnL: ${self.total_pnl:.2f}")
                logger.info(f"   🏆 Avg reward: {avg_reward:.4f}")
                logger.info(f"   📈 Cumulative reward: {self.cumulative_reward:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error updating metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances."""
        try:
            win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
            avg_reward = self.cumulative_reward / self.total_trades if self.total_trades > 0 else 0
            avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
            
            recent_performance = np.mean(self.performance_window) if self.performance_window else 0
            
            return {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'average_pnl': avg_pnl,
                'cumulative_reward': self.cumulative_reward,
                'average_reward': avg_reward,
                'recent_performance': recent_performance,
                'current_portfolio_value': self.previous_portfolio_value
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {}
    
    def reset_metrics(self):
        """Remet à zéro les métriques de suivi."""
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.cumulative_reward = 0.0
        self.trade_history.clear()
        self.performance_window.clear()
        self.previous_portfolio_value = None
        
        logger.info("🔄 Reward calculator metrics reset")


class ExperienceBuffer:
    """
    Buffer d'expérience pour l'apprentissage continu.
    Stocke les transitions (state, action, reward, next_state, done) pour l'apprentissage.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialise le buffer d'expérience.
        
        Args:
            max_size: Taille maximale du buffer
        """
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        
        logger.info(f"✅ ExperienceBuffer initialized with max_size={max_size}")
    
    def add_experience(self, 
                      state: np.ndarray, 
                      action: int, 
                      reward: float, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Optional[Dict[str, Any]] = None):
        """
        Ajoute une expérience au buffer.
        
        Args:
            state: État initial
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indique si l'épisode est terminé
            info: Informations supplémentaires
        """
        try:
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'timestamp': time.time(),
                'info': info or {}
            }
            
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                # Remplacer l'ancienne expérience (buffer circulaire)
                self.buffer[self.position] = experience
                self.position = (self.position + 1) % self.max_size
            
            logger.debug(f"📝 Experience added to buffer (size: {len(self.buffer)})")
            
        except Exception as e:
            logger.error(f"❌ Error adding experience to buffer: {e}")
    
    def sample_batch(self, batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Échantillonne un batch d'expériences pour l'apprentissage.
        
        Args:
            batch_size: Taille du batch
            
        Returns:
            List[Dict]: Batch d'expériences
        """
        try:
            if len(self.buffer) < batch_size:
                logger.warning(f"⚠️ Buffer size ({len(self.buffer)}) < batch_size ({batch_size}), returning all")
                return self.buffer.copy()
            
            import random
            batch = random.sample(self.buffer, batch_size)
            
            logger.debug(f"📦 Sampled batch of {len(batch)} experiences")
            return batch
            
        except Exception as e:
            logger.error(f"❌ Error sampling batch: {e}")
            return []
    
    def get_recent_experiences(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Retourne les N expériences les plus récentes.
        
        Args:
            n: Nombre d'expériences à retourner
            
        Returns:
            List[Dict]: Expériences récentes
        """
        try:
            if not self.buffer:
                return []
            
            # Trier par timestamp et prendre les plus récentes
            sorted_buffer = sorted(self.buffer, key=lambda x: x['timestamp'], reverse=True)
            return sorted_buffer[:n]
            
        except Exception as e:
            logger.error(f"❌ Error getting recent experiences: {e}")
            return []
    
    def clear(self):
        """Vide le buffer d'expérience."""
        self.buffer.clear()
        self.position = 0
        logger.info("🗑️ Experience buffer cleared")
    
    def size(self) -> int:
        """Retourne la taille actuelle du buffer."""
        return len(self.buffer)
    
    def is_ready_for_learning(self, min_experiences: int = 100) -> bool:
        """
        Vérifie si le buffer contient assez d'expériences pour l'apprentissage.
        
        Args:
            min_experiences: Nombre minimum d'expériences requis
            
        Returns:
            bool: True si prêt pour l'apprentissage
        """
        return len(self.buffer) >= min_experiences