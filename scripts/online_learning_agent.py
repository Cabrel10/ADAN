#!/usr/bin/python3
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
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

# Ajouter le répertoire src au PYTHONPATH
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.utils import get_logger, load_config
from adan_trading_bot.exchange_api.connector import get_exchange_client, validate_exchange_config
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.training.trainer import load_agent
from adan_trading_bot.live_trading.online_reward_calculator import OnlineRewardCalculator
from adan_trading_bot.live_trading import PrioritizedExperienceReplayBuffer
from adan_trading_bot.training import HyperparameterModulator

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
        self.model_path = Path(model_path)
        self.initial_capital = initial_capital
        
        # Configuration de l'apprentissage
        self.learning_config = learning_config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning rate scheduling parameters
        self.initial_lr = self.learning_config.get('learning_rate', 0.00001)
        self.lr_decay_steps = self.learning_config.get('lr_decay_steps', 10000)
        self.min_lr = self.learning_config.get('min_lr', 1e-6)
        
        # Initialisation du buffer d'expérience prioritaire
        self.buffer = PrioritizedExperienceReplayBuffer(
            buffer_size=self.learning_config.get('buffer_size', 10000),
            alpha=self.learning_config.get('alpha', 0.6),
            beta=self.learning_config.get('beta', 0.4),
            beta_increment=self.learning_config.get('beta_increment', 0.001),
            epsilon=self.learning_config.get('epsilon', 1e-6)
        )
        
        # Charger ou initialiser le modèle
        self.agent, self.agent_config = self._load_or_initialize_agent()
        
        # Compteurs et états
        self.episode_count = 0
        self.step_count = 0
        self.last_save_time = time.time()
        self.save_interval = self.learning_config.get('save_interval', 3600)  # 1h par défaut
        
        # Répertoire de sauvegarde
        self.save_dir = Path(self.learning_config.get('save_dir', 'saved_models/online_learning'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisation des états
        self.current_capital = initial_capital
        self.positions = {}
        self.last_episode_metrics = {}
        
        # Initialisation du client d'échange
        self.exchange_client = self._initialize_exchange_client()
        
        # Initialisation du calculateur de récompense
        self.reward_calculator = OnlineRewardCalculator(self.config)
        
        logger.info(f"OnlineLearningAgent initialisé avec {len(self.buffer)} expériences dans le buffer")
        logger.info(f"Modèle chargé depuis : {self.model_path}")
    
    def run_episode(self, env, max_steps=1000):
        """
        Exécute un épisode complet d'interaction avec l'environnement.
        
        Args:
            env: L'environnement de trading
            max_steps: Nombre maximum d'étapes par épisode
            
        Returns:
            Dictionnaire contenant les métriques de l'épisode
        """
        # Réinitialiser l'environnement
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        metrics = {
            'episode_reward': 0,
            'episode_steps': 0,
            'portfolio_value': [env.portfolio_manager.portfolio_value],
            'actions': [],
            'rewards': [],
            'positions': [],
            'market_prices': [],
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        # Boucle principale de l'épisode
        while not done and episode_steps < max_steps:
            # Sélectionner une action
            action, _states = self.agent.predict(state, deterministic=False)
            
            # Exécuter l'action dans l'environnement
            next_state, reward, done, truncated, info = env.step(action)
            
            # Mettre à jour les métriques
            episode_reward += reward
            episode_steps += 1
            self.step_count += 1
            
            # Enregistrer les métriques
            metrics['episode_reward'] = episode_reward
            metrics['episode_steps'] = episode_steps
            metrics['portfolio_value'].append(env.portfolio_manager.portfolio_value)
            metrics['actions'].append(action)
            metrics['rewards'].append(reward)
            metrics['positions'].append(info.get('open_positions', 0))
            metrics['market_prices'].append(info.get('prices', {}).get(env.assets[0], 0)) # Assuming single asset for simplicity
            
            # Note: trade_result is not directly available in info from MultiAssetEnv step
            # You might need to add it to the info dict in MultiAssetEnv or calculate it here
            # For now, we'll skip trade_result specific metrics
            
            # Calculer le drawdown actuel
            current_value = metrics['portfolio_value'][-1]
            peak_value = max(metrics['portfolio_value'])
            current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
            metrics['max_drawdown'] = max(metrics['max_drawdown'], current_drawdown)
            
            # Ajouter l'expérience au buffer
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }
            self.buffer.add(experience)
            
            # Apprentissage
            if self.learning_config.get('enabled', True) and episode_steps % self.learning_config.get('learning_frequency', 10) == 0:
                self.learn_from_experience()
            
            # Sauvegarder périodiquement
            if self._should_save_checkpoint():
                self._save_checkpoint_if_needed()
            
            # Mettre à jour l'état
            state = next_state
        
        # Calculer le ratio de Sharpe
        returns = np.diff(metrics['portfolio_value']) / np.array(metrics['portfolio_value'][:-1])
        if len(returns) > 1:
            metrics['sharpe_ratio'] = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-8)
        
        # Update trade metrics (placeholder, needs actual trade tracking in env)
        metrics['total_trades'] = env.portfolio_manager.trades_count # Assuming trades_count is available
        metrics['winning_trades'] = env.portfolio_manager.win_count # Assuming win_count is available
        metrics['losing_trades'] = env.portfolio_manager.loss_count # Assuming loss_count is available
        metrics['win_rate'] = metrics['winning_trades'] / max(1, metrics['total_trades'])
        
        # Enregistrer les métriques
        logger.info(f"Épisode {self.episode_count} terminé - "
                   f"Récompense: {episode_reward:.2f}, "
                   f"Durée: {episode_steps} étapes, "
                   f"Valeur du portefeuille: {metrics['portfolio_value'][-1]:.2f}, "
                   f"Trades: {metrics['total_trades']} (G: {metrics['winning_trades']}, P: {metrics['losing_trades']}, "
                   f"Win Rate: {metrics['win_rate']*100:.1f}%), "
                   f"Drawdown Max: {metrics['max_drawdown']*100:.2f}%")
        
        # Incrémenter le compteur d'épisodes
        self.episode_count += 1
        self.last_episode_metrics = metrics
        
        return metrics
    
    
    
    def _load_or_initialize_agent(self):
        """
        Charge un modèle existant ou initialise un nouveau modèle.
        
        Returns:
            Tuple contenant l'agent et sa configuration
        """
        try:
            # Essayer de charger le modèle et le buffer
            agent, agent_config = load_agent(self.model_path, self.device)
            
            # Vérifier s'il existe un état sauvegardé
            state_path = self.save_dir / 'agent_state.pkl'
            if state_path.exists():
                try:
                    with open(state_path, 'rb') as f:
                        state = pickle.load(f)
                    
                    # Charger l'état du buffer
                    buffer_path = self.save_dir / 'experience_buffer.pkl'
                    if buffer_path.exists():
                        self.buffer = PrioritizedExperienceReplayBuffer.load(buffer_path)
                    
                    # Charger les compteurs
                    self.episode_count = state.get('episode_count', 0)
                    self.step_count = state.get('step_count', 0)
                    
                    logger.info(f"État de l'agent chargé depuis {state_path}")
                    logger.info(f"Buffer d'expérience chargé avec {len(self.buffer)} expériences")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de l'état de l'agent : {e}")
            
            return agent, agent_config
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {self.model_path}: {e}")
            logger.info("Initialisation d'un nouvel agent...")
            # Ici, vous devriez initialiser un nouvel agent si le chargement échoue
            raise NotImplementedError("L'initialisation d'un nouvel agent n'est pas encore implémentée")
    
    def _initialize_exchange_client(self):
        """Initialise le client d'échange."""
        try:
            exchange_config = self.config['exchange']
            validate_exchange_config(exchange_config)
            return get_exchange_client(exchange_config, testnet=True)
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client d'échange : {e}")
            raise
    
    def save_state(self):
        """Sauvegarde l'état actuel de l'agent et du buffer d'expérience."""
        try:
            # Sauvegarder l'état de l'agent
            agent_state = {
                'episode_count': self.episode_count,
                'step_count': self.step_count,
                'current_capital': self.current_capital,
                'last_episode_metrics': self.last_episode_metrics,
                'model_version': self.agent_config.get('version', '1.0.0')
            }
            
            # Sauvegarder l'état de l'agent
            state_path = self.save_dir / 'agent_state.pkl'
            with open(state_path, 'wb') as f:
                pickle.dump(agent_state, f)
            
            # Sauvegarder le buffer d'expérience
            buffer_path = self.save_dir / 'experience_buffer.pkl'
            self.buffer.save(buffer_path)
            
            # Sauvegarder le modèle
            model_version = agent_state['model_version']
            model_save_path = self.save_dir / f'model_v{model_version}.pth'
            torch.save({
                'model_state_dict': self.agent.policy.state_dict(),
                'optimizer_state_dict': self.agent.policy.optimizer.state_dict(),
                'config': self.agent_config
            }, model_save_path)
            
            logger.info(f"État de l'agent sauvegardé dans {self.save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état de l'agent : {e}")
            return False
    
    def _should_save_checkpoint(self):
        """Détermine si un checkpoint doit être sauvegardé."""
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time
        return time_since_last_save >= self.save_interval
    
    def _save_checkpoint_if_needed(self):
        """Sauvegarde un checkpoint si nécessaire."""
        if self._should_save_checkpoint():
            self.save_state()
            self.last_save_time = time.time()
    
    def _calculate_td_error(self, batch):
        """
        Calcule l'erreur TD (Temporal Difference) pour un batch d'expériences.
        
        Args:
            batch: Dictionnaire contenant les données du batch
            
        Returns:
            Tableau d'erreurs TD pour chaque expérience du batch
        """
        with torch.no_grad():
            # Convertir les données en tenseurs
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)
            
            # Obtenir les valeurs d'état actuelles et suivantes
            current_values = self.agent.policy.predict_values(states)
            next_values = self.agent.policy.predict_values(next_states)
            
            # Calculer les cibles et l'erreur TD
            targets = rewards + (1 - dones) * self.agent.gamma * next_values
            td_errors = torch.abs(targets - current_values).squeeze()
            
            return td_errors.cpu().numpy()

    def _adjust_learning_rate(self):
        """
        Adjusts the learning rate using a linear decay schedule.
        """
        if self.learning_steps >= self.lr_decay_steps:
            new_lr = self.min_lr
        else:
            decay_factor = 1 - (self.learning_steps / self.lr_decay_steps)
            new_lr = self.initial_lr * decay_factor
        
        # Ensure learning rate doesn't go below minimum
        new_lr = max(new_lr, self.min_lr)
        
        # Update the agent's learning rate
        if hasattr(self.agent.optimizer, 'param_groups'):
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            # Fallback for agents without param_groups (e.g., some custom optimizers)
            self.agent.learning_rate = new_lr # Assuming agent has a direct learning_rate attribute

        logger.debug(f"Learning rate adjusted to: {new_lr:.6f}")
    
    def learn_from_experience(self, batch_size=256, num_epochs=4):
        """
        Effectue une étape d'apprentissage sur un échantillon du buffer d'expérience.
        
        Args:
            batch_size: Taille du batch d'apprentissage
            num_epochs: Nombre d'époques d'entraînement
            
        Returns:
            Dictionnaire contenant les métriques d'apprentissage
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"Pas assez d'expériences dans le buffer ({len(self.buffer)} < {batch_size})")
            return {}
        
        metrics = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'explained_variance': [],
            'mean_td_error': []
        }
        
        for epoch in range(num_epochs):
            # Échantillonner un batch d'expériences
            batch, batch_indices, weights = self.buffer.sample(batch_size)
            
            # Convertir les données en tenseurs
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
            
            # Effectuer une étape d'apprentissage avec PPO
            batch_metrics = self.agent.learn(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                weights=weights
            )
            
            # Calculer les nouvelles priorités basées sur l'erreur TD
            with torch.no_grad():
                td_errors = self._calculate_td_error(batch)
                self.buffer.update_priorities(batch_indices, td_errors)
            
            # Mettre à jour les métriques
            for k, v in batch_metrics.items():
                if k in metrics:
                    metrics[k].append(v)
            
            metrics['mean_td_error'].append(float(td_errors.mean()))
        
        # Calculer les moyennes des métriques sur toutes les époques
        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
        
        # Enregistrer les métriques
        logger.info(f"Apprentissage - Loss: {avg_metrics['loss']:.4f}, "
                   f"Policy Loss: {avg_metrics['policy_loss']:.4f}, "
                   f"Value Loss: {avg_metrics['value_loss']:.4f}, "
                   f"Entropy: {avg_metrics['entropy']:.4f}, "
                   f"Explained Var: {avg_metrics['explained_variance']:.4f}, "
                   f"Mean TD Error: {avg_metrics['mean_td_error']:.4f}")
        
        return avg_metrics
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
        
        # Initialisation du modulateur d'hyperparamètres
        self.hyperparam_modulator = HyperparameterModulator(
            agent=self.agent,
            config=self.learning_config.get('hyperparameter_modulation', {})
        )
        
        # Initialisation du buffer d'expérience prioritaires
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
            
            # Buffer d'expérience prioritaire
            buffer_size = self.learning_config.get('buffer_size', 1000)
            alpha = self.learning_config.get('per_alpha', 0.6)  # Priorité des échantillons
            beta = self.learning_config.get('per_beta', 0.4)    # Importance du sampling
            
            self.experience_buffer = PrioritizedExperienceReplayBuffer(
                max_size=buffer_size,
                alpha=alpha,
                beta=beta
            )
            
            # Métriques d'apprentissage
            self.learning_metrics = {
                'total_updates': 0,
                'average_reward': 0.0,
                'last_loss': 0.0,
                'exploration_rate': self.exploration_rate,
                'supervised_learning_triggers': 0
            }
            
            # Configuration pour l'apprentissage supervisé
            self.supervised_learning_config = {
                'trigger_pct': self.learning_config.get('supervised_trigger_pct', 0.7),
                'batch_size': self.learning_config.get('supervised_batch_size', 32),
                'max_samples': self.learning_config.get('supervised_max_samples', 1000)
            }
            
            logger.info("✅ Learning components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize learning components: {e}")
            raise
            
    def _should_apply_supervised_learning(self, metrics: Dict[str, Any]) -> bool:
        """
        Détermine si un apprentissage supervisé doit être déclenché.
        
        Args:
            metrics: Métriques d'apprentissage actuelles
            
        Returns:
            bool: True si l'apprentissage supervisé doit être déclenché, False sinon
        """
        # Vérifier si nous avons suffisamment de données
        if len(self.experience_buffer) < self.supervised_learning_config['batch_size'] * 2:
            return False
            
        # Vérifier la performance par rapport au seuil
        avg_reward = metrics.get('avg_reward', 0)
        avg_loss = metrics.get('avg_loss', float('inf'))
        
        # Seuil de déclenchement basé sur la configuration
        reward_threshold = -0.1  # Seuil arbitraire à ajuster
        loss_threshold = 1.0     # Seuil arbitraire à ajuster
        
        return (avg_reward < reward_threshold or 
                avg_loss > loss_threshold)
    
    def _apply_supervised_learning(self, env, current_prices: Dict[str, float]) -> None:
        """
        Applique un apprentissage supervisé basé sur les meilleurs trades possibles.
        
        Args:
            env: L'environnement de trading
            current_prices: Dictionnaire des prix actuels des actifs
        """
        try:
            logger.info("🎓 Démarrage de l'apprentissage supervisé...")
            
            # 1. Identifier les meilleurs trades possibles sur les données récentes
            optimal_trades = self._identify_optimal_trades(env, current_prices)
            
            if not optimal_trades:
                logger.warning("⚠️ Aucun trade optimal identifié pour l'apprentissage supervisé")
                return
                
            # 2. Créer des expériences supervisées
            supervised_experiences = []
            
            for trade in optimal_trades:
                # Créer une expérience supervisée pour ce trade
                state = trade['state']
                action = trade['optimal_action']
                reward = trade['expected_reward']
                
                # Ajouter au buffer avec une priorité élevée
                self.experience_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state,  # État suivant identique car c'est un objectif
                    done=False,
                    priority=1.0  # Haute priorité
                )
                
                supervised_experiences.append({
                    'state': state,
                    'action': action,
                    'reward': reward
                })
            
            # 3. Effectuer une étape d'apprentissage supplémentaire
            if supervised_experiences:
                # Mélanger les expériences
                np.random.shuffle(supervised_experiences)
                
                # Prendre un sous-ensemble selon la taille du batch
                batch = supervised_experiences[:self.supervised_learning_config['batch_size']]
                
                # Préparer les données pour l'entraînement
                states = np.array([exp['state'] for exp in batch])
                actions = np.array([exp['action'] for exp in batch])
                rewards = np.array([exp['reward'] for exp in batch])
                
                # Entraîner le modèle avec ces données supervisées
                # Note: Cette partie dépend de l'implémentation spécifique de votre modèle
                # Voici un exemple générique
                try:
                    # Exemple avec un modèle Stable Baselines 3
                    if hasattr(self.agent, 'policy'):
                        # Convertir les actions en format one-hot si nécessaire
                        if len(actions.shape) == 1:
                            n_actions = self.agent.action_space.n
                            actions_one_hot = np.eye(n_actions)[actions]
                            
                        # Calculer la perte d'entropie croisée
                        # Note: Cette partie est un exemple et doit être adaptée
                        # à votre implémentation spécifique
                        policy = self.agent.policy
                        if hasattr(policy, 'evaluate_actions'):
                            # Évaluer les actions avec la politique actuelle
                            dist = policy.get_distribution(states)
                            log_probs = dist.log_prob(actions)
                            
                            # Calculer la perte (négative car on maximise la vraisemblance)
                            loss = -(log_probs * rewards).mean()
                            
                            # Mettre à jour les poids
                            policy.optimizer.zero_grad()
                            loss.backward()
                            policy.optimizer.step()
                            
                            logger.info(f"✅ Apprentissage supervisé terminé - Perte: {loss.item():.4f}")
                            
                            # Mettre à jour les métriques
                            self.learning_metrics['supervised_learning_triggers'] += 1
                            self.learning_metrics['last_supervised_loss'] = loss.item()
                            
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'apprentissage supervisé: {e}")
        
        except Exception as e:
            logger.error(f"❌ Échec de l'apprentissage supervisé: {e}")
    
    def _identify_optimal_trades(self, env, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Identifie les meilleurs trades possibles sur les données récentes.
        
        Args:
            env: L'environnement de trading
            current_prices: Dictionnaire des prix actuels des actifs
            
        Returns:
            Liste des trades optimaux avec leurs états et actions
        """
        optimal_trades = []
        
        try:
            # 1. Récupérer les données récentes
            lookback = min(100, len(self.experience_buffer))  # Nombre d'états récents à considérer
            
            if lookback == 0:
                return []
                
            # 2. Pour chaque état récent, déterminer l'action optimale
            for i in range(max(0, len(self.experience_buffer) - lookback), len(self.experience_buffer)):
                experience = self.experience_buffer.buffer[i]
                state = experience['state']
                
                # 3. Évaluer chaque action possible
                best_action = None
                best_reward = -float('inf')
                
                # Pour chaque action possible
                for action_idx in range(self.agent.action_space.n):
                    # Simuler l'action et obtenir la récompense attendue
                    # Note: Cette partie dépend de votre implémentation spécifique
                    # Voici un exemple générique
                    try:
                        # Obtenir la distribution de probabilité des actions
                        if hasattr(self.agent.policy, 'get_distribution'):
                            dist = self.agent.policy.get_distribution(state.reshape(1, -1))
                            action_probs = dist.distribution.probs.detach().numpy()[0]
                            
                            # La récompense attendue est la probabilité de l'action
                            # pondérée par la qualité de l'état (à adapter)
                            state_quality = 1.0  # À remplacer par une métrique de qualité d'état
                            expected_reward = action_probs[action_idx] * state_quality
                            
                            if expected_reward > best_reward:
                                best_reward = expected_reward
                                best_action = action_idx
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur lors de l'évaluation de l'action {action_idx}: {e}")
                
                # Si une action optimale a été trouvée, l'ajouter à la liste
                if best_action is not None and best_reward > 0:
                    optimal_trades.append({
                        'state': state,
                        'optimal_action': best_action,
                        'expected_reward': best_reward,
                        'timestamp': time.time()
                    })
                    
                    # Limiter le nombre d'échantillons
                    if len(optimal_trades) >= self.supervised_learning_config['max_samples']:
                        break
            
            logger.info(f"🔍 {len(optimal_trades)} trades optimaux identifiés pour l'apprentissage supervisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'identification des trades optimaux: {e}")
        
        return optimal_trades
    
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
        """
        Effectue un step d'apprentissage basé sur l'expérience accumulée avec PER.
        
        Returns:
            bool: True si l'apprentissage a été effectué, False sinon
            dict: Métriques d'apprentissage (loss, etc.)
        """
        try:
            if not self.learning_enabled:
                return False, {}
            
            batch_size = self.learning_config.get('batch_size', 64)
            min_experiences = max(batch_size * 2, 100)  # Au moins 2x le batch size
            
            if not self.experience_buffer.is_ready_for_learning(min_experiences=min_experiences):
                logger.debug(f"📚 Not enough experiences for learning (need {min_experiences}, have {self.experience_buffer.size})")
                return False, {}
            
            # Échantillonner un batch avec PER
            batch = self.experience_buffer.sample_batch(batch_size=batch_size)
            if not batch:
                logger.warning("⚠️ Failed to sample batch from experience buffer")
                return False, {}
            # Extraire les données du batch
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            next_states = batch['next_states']
            dones = batch['dones']
            weights = batch['weights']
            batch_indices = batch['indices']
            
            # Vérifier les dimensions des données
            if len(states) != batch_size or len(actions) != batch_size:
                logger.error(f"❌ Batch size mismatch: states={len(states)}, actions={len(actions)}")
                return False, {}
            
            try:
                # 1. Calculer les valeurs cibles avec le modèle cible (si disponible)
                with torch.no_grad():
                    # Pour PPO, nous utilisons le modèle actuel pour l'évaluation
                    # Dans une implémentation complète, on utiliserait un modèle cible
                    _, values, _ = self.agent.policy.evaluate_actions(
                        torch.FloatTensor(states).to(self.agent.device),
                        torch.LongTensor(actions).to(self.agent.device)
                    )
                    values = values.cpu().numpy()
                
                # 2. Calculer les avantages et les retours (simplifié pour PPO)
                # Dans une implémentation complète, on utiliserait GAE (Generalized Advantage Estimation)
                advantages = rewards - values.squeeze()
                returns = rewards + self.agent.gamma * (1 - dones) * values.squeeze()
                
                # 3. Normaliser les avantages
                if len(advantages) > 1:  # Éviter la division par zéro
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 4. Convertir en tenseurs PyTorch
                states_tensor = torch.FloatTensor(states).to(self.agent.device)
                actions_tensor = torch.LongTensor(actions).to(self.agent.device)
                old_log_probs_tensor = torch.FloatTensor(values).to(self.agent.device)
                returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.agent.device)
                advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1).to(self.agent.device)
                weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.agent.device)
                
                # 5. Effectuer la mise à jour du modèle avec PPO
                policy_loss, value_loss, entropy_loss = self.agent._update_policy(
                    states_tensor,
                    actions_tensor,
                    returns_tensor,
                    advantages_tensor,
                    old_log_probs_tensor,
                    weights_tensor
                )
                
                # 6. Calculer les nouvelles priorités basées sur l'erreur TD
                with torch.no_grad():
                    _, new_values, _ = self.agent.policy.evaluate_actions(
                        states_tensor,
                        actions_tensor
                    )
                    td_errors = (returns_tensor - new_values).abs().cpu().numpy().flatten()
                
                # 7. Mettre à jour les priorités dans le buffer
                self.experience_buffer.update_priorities(batch_indices, td_errors)
                
                # 8. Mettre à jour les métriques
                self.learning_metrics.update({
                    'total_updates': self.learning_metrics.get('total_updates', 0) + 1,
                    'average_reward': float(np.mean(rewards)),
                    'policy_loss': float(policy_loss),
                    'value_loss': float(value_loss),
                    'entropy_loss': float(entropy_loss),
                    'avg_td_error': float(np.mean(td_errors)),
                    'exploration_rate': self.exploration_rate
                })
                
                self.learning_steps += 1
                
                # Adjust learning rate
                self._adjust_learning_rate()
                
                # 9. Ajuster le taux d'exploration
                self._adjust_exploration_rate()
                
                # 10. Log des métriques
                if self.learning_steps % 5 == 0:
                    logger.info(f"🎓 Learning step {self.learning_steps}:")
                    logger.info(f"   📊 Avg reward: {self.learning_metrics['average_reward']:.4f}")
                    logger.info(f"   📉 Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")
                    logger.info(f"   🎲 Exploration rate: {self.exploration_rate:.4f}")
                    logger.info(f"   🔄 Avg TD error: {np.mean(td_errors):.4f}")
                
                return True, self.learning_metrics
                
            except Exception as e:
                logger.error(f"❌ Learning step failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False, {}
            
        except Exception as e:
            logger.error(f"❌ Error in learn_from_experience: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, {}
    
    def _adjust_exploration_rate(self):
        """
        Ajuste dynamiquement le taux d'exploration.
        
        Le taux d'exploration diminue progressivement selon un planning linéaire
        jusqu'à atteindre une valeur minimale.
        """
        min_exploration = self.learning_config.get('min_exploration', 0.01)
        decay_steps = self.learning_config.get('exploration_decay_steps', 1000)
        
        if self.learning_steps >= decay_steps:
            self.exploration_rate = min_exploration
        else:
            # Décroissance linéaire
            decay_rate = 1.0 - (self.learning_steps / decay_steps)
            self.exploration_rate = max(
                min_exploration,
                self.learning_config.get('initial_exploration', 0.1) * decay_rate
            )
        
        # Mettre à jour le taux d'exploration dans l'agent
        if hasattr(self.agent, 'exploration_rate'):
            self.agent.exploration_rate = self.exploration_rate
        
        return self.exploration_rate
    
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
                
                # Log additional KPIs
                current_metrics = self.reward_calculator.get_performance_summary()
                logger.info(f"📈 Current PnL: ${current_metrics.get('total_pnl', 0.0):.2f} ({current_metrics.get('win_rate', 0.0):.1f}% Win Rate)")
                logger.info(f"📉 Max Drawdown: {current_metrics.get('max_drawdown', 0.0):.2f}%")
                logger.info(f"📊 Sharpe Ratio: {current_metrics.get('sharpe_ratio', 0.0):.2f}")

                # Save metrics to file periodically
                if iteration % self.learning_config.get('metrics_save_frequency', 10) == 0:
                    self._save_metrics_to_file(iteration, current_metrics)
                
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
                        
                        # Récupérer la modulation du DBE si disponible
                        dbe_modulation = {}
                        if hasattr(env, 'dbe') and hasattr(env.dbe, 'get_current_modulation'):
                            dbe_modulation = env.dbe.get_current_modulation()
                            
                            # Ajuster les hyperparamètres en fonction de la modulation du DBE
                            param_changes = self.hyperparam_modulator.adjust_params(dbe_modulation)
                            if param_changes:
                                logger.info(f"🔄 Hyperparameters adjusted: {param_changes}")
                        
                        # Effectuer l'apprentissage
                        learning_result = self.learn_from_experience()
                        
                        # Vérifier si learn_from_experience retourne un tuple (success, metrics) ou juste un booléen
                        if isinstance(learning_result, tuple):
                            learning_success, metrics = learning_result
                        else:
                            learning_success = learning_result
                            metrics = {}
                        
                        # Vérifier la performance et appliquer un apprentissage supervisé si nécessaire
                        if learning_success and self._should_apply_supervised_learning(metrics):
                            self._apply_supervised_learning(env, current_prices)
                        
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

    def _save_metrics_to_file(self, iteration: int, metrics: Dict[str, Any]):
        """Saves key performance metrics to a JSON file."""
        metrics_path = self.save_dir / "realtime_metrics.json"
        data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "capital": self.current_capital,
            "learning_steps": self.learning_steps,
            "exploration_rate": self.exploration_rate,
            "metrics": metrics
        }
        
        # Append to file if it exists, otherwise create new
        if metrics_path.exists():
            with open(metrics_path, 'r+') as f:
                try:
                    file_data = json.load(f)
                    if not isinstance(file_data, list):
                        file_data = [file_data] # Ensure it's a list
                except json.JSONDecodeError:
                    file_data = []
                file_data.append(data)
                f.seek(0)
                json.dump(file_data, f, indent=2, default=str)
                f.truncate()
        else:
            with open(metrics_path, 'w') as f:
                json.dump([data], f, indent=2, default=str)
        logger.debug(f"Metrics saved to {metrics_path}")


def main():
    """Fonction principale du script d'apprentissage continu."""
    parser = argparse.ArgumentParser(description="ADAN Online Learning Agent")
    parser.add_argument("--exec_profile", type=str, default="cpu", 
                       choices=["cpu", "gpu"], help="Profil d'exécution")
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