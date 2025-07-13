#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'entraînement avancé pour l'agent de trading ADAN avec RL.

Fonctionnalités principales :
- Chargement et validation de la configuration
- Initialisation de l'environnement de trading personnalisé
- Entraînement avec suivi détaillé des performances
- Callbacks personnalisés pour le suivi du trading
- Visualisation et sauvegarde des résultats
"""

import argparse
import logging
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple
import yaml
from datetime import datetime

# Stable Baselines 3
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Ajout du répertoire racine au PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import des modules personnalisés
from src.adan_trading_bot.environment.multi_asset_env import AdanTradingEnv
from src.adan_trading_bot.utils.visualization import TradingVisualizer

class TradingMetricsCallback(BaseCallback):
    """Callback pour suivre les métriques de trading personnalisées."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.portfolio_values = []
        self.actions_history = []
        self.rewards_history = []
        self.episode_returns = []
        self.current_episode_returns = []
        self.trades = []
        
    def _on_step(self) -> bool:
        # Récupérer les infos de l'environnement
        infos = self.locals.get('infos')
        if not infos or not isinstance(infos, dict):
            return True
            
        # Extraire les métriques de trading
        for info in infos.get('episode', [{}]):
            if 'portfolio_value' in info:
                self.portfolio_values.append(info['portfolio_value'])
            if 'trades' in info:
                self.trades.extend(info['trades'])
        
        # Historique des actions et récompenses
        actions = self.locals.get('actions')
        rewards = self.locals.get('rewards')
        
        if actions is not None:
            self.actions_history.extend(actions)
        if rewards is not None:
            self.rewards_history.append(rewards)
            self.current_episode_returns.append(rewards)
        
        # Vérifier si l'épisode est terminé
        dones = self.locals.get('dones')
        if dones and any(dones):
            episode_return = sum(self.current_episode_returns)
            self.episode_returns.append(episode_return)
            self.current_episode_returns = []
            
            # Log des métriques
            logger.info(f"Episode {len(self.episode_returns)} - Return: {episode_return:.2f}")
            
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques collectées."""
        return {
            'portfolio_values': self.portfolio_values,
            'actions': self.actions_history,
            'rewards': self.rewards_history,
            'episode_returns': self.episode_returns,
            'trades': self.trades
        }

class TradingEvalCallback(EvalCallback):
    """Callback pour l'évaluation périodique du modèle."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Log des résultats d'évaluation
        if self.evaluations_times:
            logger.info(f"Evaluation - Mean reward: {self.best_mean_score:.2f} ± {self.last_mean_reward:.2f}")
            
        return result

def load_config(path: str) -> Dict[str, Any]:
    """Charge un fichier de configuration YAML avec support des inclusions."""
    config_path = Path(path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Gestion des inclusions de fichiers
    if 'imports' in config:
        config_dir = config_path.parent
        for key, import_path in config['imports'].items():
            import_full_path = config_dir / import_path
            if import_full_path.exists():
                with open(import_full_path, 'r') as f:
                    imported_config = yaml.safe_load(f)
                    config[key] = imported_config
    
    return config

def setup_environment(config: Dict[str, Any]) -> Tuple[DummyVecEnv, Dict[str, Any]]:
    """Initialise et configure l'environnement de trading."""
    logger.info("Initialisation de l'environnement de trading...")
    
    # Création de l'environnement
    env = AdanTradingEnv(config=config)
    
    # Enveloppement avec Monitor pour le suivi des récompenses
    env = Monitor(env)
    
    # Création d'un environnement vectorisé (nécessaire pour SB3)
    vec_env = DummyVecEnv([lambda: env])
    
    # Normalisation des observations
    if config.get('environment', {}).get('normalize_observations', True):
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0
        )
    
    return vec_env, env.get_metadata()

def setup_model(env: DummyVecEnv, config: Dict[str, Any]) -> PPO:
    """Initialise le modèle d'apprentissage par renforcement."""
    logger.info("Initialisation du modèle d'apprentissage par renforcement...")
    
    # Configuration de l'agent
    agent_config = config.get('agent_config_cpu', {}).get('agent', {})
    policy_config = agent_config.get('policy', {})
    
    # Paramètres de la politique
    policy_kwargs = {
        'net_arch': policy_config.get('net_arch', [256, 256]),
        'activation_fn': getattr(torch.nn, policy_config.get('activation_fn', 'ReLU')),
        'ortho_init': policy_config.get('ortho_init', True)
    }
    
    # Création du modèle
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=agent_config.get('learning_rate', 3e-4),
        n_steps=agent_config.get('n_steps', 2048),
        batch_size=agent_config.get('batch_size', 64),
        n_epochs=agent_config.get('n_epochs', 10),
        gamma=agent_config.get('gamma', 0.99),
        gae_lambda=agent_config.get('gae_lambda', 0.95),
        clip_range=agent_config.get('clip_range', 0.2),
        clip_range_vf=agent_config.get('clip_range_vf', None),
        ent_coef=agent_config.get('ent_coef', 0.0),
        vf_coef=agent_config.get('vf_coef', 0.5),
        max_grad_norm=agent_config.get('max_grad_norm', 0.5),
        use_sde=agent_config.get('use_sde', False),
        sde_sample_freq=agent_config.get('sde_sample_freq', -1),
        target_kl=agent_config.get('target_kl', None),
        tensorboard_log=config.get('paths', {}).get('tensorboard_log', 'logs/tensorboard'),
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    return model

def setup_callbacks(model: PPO, config: Dict[str, Any], eval_env: Optional[DummyVecEnv] = None) -> CallbackList:
    """Configure les callbacks pour l'entraînement."""
    logger.info("Configuration des callbacks...")
    
    callbacks = []
    
    # Callback de sauvegarde des modèles
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('checkpointing', {}).get('save_freq', 10000),
        save_path=config.get('paths', {}).get('checkpoint_dir', 'models/checkpoints'),
        name_prefix=config.get('checkpointing', {}).get('name_prefix', 'adan_ppo')
    )
    callbacks.append(checkpoint_callback)
    
    # Callback pour les métriques de trading
    metrics_callback = TradingMetricsCallback()
    callbacks.append(metrics_callback)
    
    # Callback d'évaluation si un environnement d'évaluation est fourni
    if eval_env is not None:
        eval_callback = TradingEvalCallback(
            eval_env,
            best_model_save_path=config.get('paths', {}).get('best_model_dir', 'models/best'),
            log_path=config.get('paths', {}).get('eval_logs_dir', 'logs/eval'),
            eval_freq=config.get('evaluation', {}).get('eval_freq', 10000),
            deterministic=True,
            render=False,
            n_eval_episodes=config.get('evaluation', {}).get('n_eval_episodes', 5),
            warn=False
        )
        callbacks.append(eval_callback)
    
    return CallbackList(callbacks)

def train_model():
    """Fonction principale pour l'entraînement du modèle."""
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Entraînement de l'agent de trading ADAN avec RL")
    parser.add_argument(
        "--config",
        type=str,
        default="config/main_config.yaml",
        help="Chemin vers le fichier de configuration principal"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un modèle à reprendre pour l'entraînement"
    )
    args = parser.parse_args()
    
    # Chargement de la configuration
    logger.info(f"Chargement de la configuration depuis {args.config}")
    config = load_config(args.config)
    
    # Configuration des chemins
    config.setdefault('paths', {})
    config['paths'].setdefault('tensorboard_log', 'logs/tensorboard')
    config['paths'].setdefault('checkpoint_dir', 'models/checkpoints')
    config['paths'].setdefault('best_model_dir', 'models/best')
    config['paths'].setdefault('eval_logs_dir', 'logs/eval')
    
    # Création des répertoires nécessaires
    for path in config['paths'].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Initialisation de l'environnement
    env, env_metadata = setup_environment(config)
    
    # Initialisation du modèle
    if args.resume:
        logger.info(f"Reprise de l'entraînement à partir de {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = setup_model(env, config)
    
    # Configuration des callbacks
    callbacks = setup_callbacks(model, config)
    
    # Entraînement du modèle
    total_timesteps = config.get('training', {}).get('total_timesteps', 1_000_000)
    logger.info(f"Début de l'entraînement pour {total_timesteps} pas...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=config.get('training', {}).get('tb_log_name', 'adan_training'),
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("Entraînement interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}", exc_info=True)
        raise
    
    # Sauvegarde du modèle final
    final_model_path = os.path.join(
        config.get('paths', {}).get('trained_models_dir', 'models'),
        f"adan_ppo_final_{int(datetime.now().timestamp())}"
    )
    model.save(final_model_path)
    logger.info(f"Modèle final sauvegardé dans {final_model_path}")
    
    # Fermeture de l'environnement
    env.close()
    
    logger.info("Entraînement terminé avec succès!")
    
    # Récupération des métriques pour analyse
    metrics = {}
    for callback in callbacks.callbacks:
        if hasattr(callback, 'get_metrics'):
            metrics.update(callback.get_metrics())
    
    # Génération du rapport de trading si des données sont disponibles
    if metrics.get('portfolio_values') and metrics.get('actions') and metrics.get('rewards'):
        try:
            from src.adan_trading_bot.utils.visualization import generate_training_report
            
            # Création du répertoire de rapports
            report_dir = os.path.join('reports', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            os.makedirs(report_dir, exist_ok=True)
            
            # Génération du rapport
            generate_training_report(
                training_metrics={
                    'reward': metrics.get('rewards', []),
                    'portfolio_value': metrics.get('portfolio_values', []),
                },
                portfolio_values=metrics.get('portfolio_values', []),
                actions=metrics.get('actions', []),
                prices=[p[0] for p in metrics.get('prices', [])],  # Prix du premier actif
                returns=np.diff(metrics.get('portfolio_values', [0, 0])) / np.array(metrics.get('portfolio_values', [1, 1])[:-1]),
                output_dir=report_dir
            )
            
            logger.info(f"Rapport de trading généré dans {os.path.abspath(report_dir)}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {str(e)}", exc_info=True)
    
    return metrics

if __name__ == "__main__":
    train_metrics = train_model()
    
    # Sauvegarde des métriques d'entraînement
    if train_metrics:
        metrics_path = os.path.join('logs', f'training_metrics_{int(datetime.now().timestamp())}.json')
        with open(metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)
        logger.info(f"Métriques d'entraînement sauvegardées dans {metrics_path}")
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    logger.info(f"Models will be saved every {save_freq} steps in '{save_path}'")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    # 5. Save the Final Model
    final_model_path = Path(save_path) / f"{model_name}_final.zip"
    model.save(final_model_path)

    logger.info(f"Training finished. Final model saved to: {final_model_path}")
    logger.info("To evaluate the model, run the evaluation script.")

if __name__ == "__main__":
    main()
