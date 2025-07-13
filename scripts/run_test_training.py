#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour lancer un entraînement de test sur un sous-ensemble des données.
"""
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from src.adan_trading_bot.common.utils import get_logger
from src.adan_trading_bot.environment.multi_asset_env import AdanTradingEnv
from src.adan_trading_bot.models.feature_extractors import CustomCNNFeatureExtractor

# Configurer le logger
logger = get_logger()

def load_configs():
    """Charge les configurations nécessaires."""
    # Chemin vers les fichiers de configuration
    config_dir = Path("config")
    
    # Charger la configuration principale
    with open(config_dir / "main_config.yaml", 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Charger la configuration de l'agent
    with open(config_dir / "agent_config_cpu.yaml", 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Charger la configuration de l'environnement
    with open(config_dir / "environment_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    
    return main_config, agent_config, env_config

def create_environment(env_config, main_config):
    """Crée l'environnement de trading."""
    # Configuration de base de l'environnement
    env_kwargs = {
        "initial_balance": main_config["environment"]["initial_balance"],
        "commission": main_config["environment"]["trading_fees"],
        "window_size": main_config["agent"]["window_size"],
        "frame_stack": env_config.get("frame_stack", 4),
        "use_technical_indicators": True,
        "use_sentiment": False,
        "use_volume_profile": False,
        "use_order_book": False,
        "use_time_features": True,
        "normalize_observation": True,
        "reward_scale": 1.0,
        "mode": "train",
        "verbose": 1,
        # Configuration de l'état
        "state": {
            "window_size": main_config["agent"]["window_size"],
            "use_technical_indicators": True,
            "use_time_features": True,
            "use_volume": True
        }
    }
    
    # Créer un environnement simplifié qui charge directement les données
    # au lieu d'utiliser le ChunkedDataLoader
    class SimpleTradingEnv(gym.Env):
        """Un environnement de trading simplifié pour le test."""
        
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.mode = config.get("mode", "train")
            self.window_size = config["window_size"]
            
            # Charger les données d'entraînement pour tous les actifs
            self.assets = ["BTC", "ETH", "SOL", "XRP", "ADA"]
            self.data = {}
            for asset in self.assets:
                file_path = f"data/processed/{asset}/{asset}_1h_train.parquet"
                if os.path.exists(file_path):
                    self.data[asset] = pd.read_parquet(file_path)
                    print(f"Chargé {len(self.data[asset])} lignes pour {asset}")
            
            if not self.data:
                raise ValueError("Aucune donnée trouvée pour l'entraînement")
            
            # Définir les espaces d'observation et d'action
            # (simplifié pour le test)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(len(self.assets), self.window_size, 5),  # OHLCV
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(3)  # Vendre, Rien, Acheter
            
            self.current_step = self.window_size
            
        def reset(self, seed=None, options=None):
            # Reset l'état de l'environnement
            self.current_step = self.window_size
            
            # Gérer la graine aléatoire si fournie
            if seed is not None:
                np.random.seed(seed)
                
            # Retourner l'observation et un dictionnaire d'infos vide
            observation = self._get_observation()
            info = {}
            
            return observation, info
            
        def step(self, action):
            # Simulation simple d'un pas d'environnement
            self.current_step += 1
            
            # Vérifier si l'épisode est terminé
            terminated = all(self.current_step >= len(df) - 1 for df in self.data.values())
            truncated = False  # Pas de troncature pour l'instant
            
            # Récompense aléatoire pour le test
            reward = np.random.uniform(-1, 1)
            
            # Récupérer l'observation
            observation = self._get_observation()
            info = {}  # Dictionnaire d'informations supplémentaires
            
            return observation, reward, terminated, truncated, info
            
        def _get_observation(self):
            # Retourne les données de la fenêtre actuelle
            obs = []
            for asset in self.assets:
                df = self.data[asset]
                start_idx = max(0, self.current_step - self.window_size)
                end_idx = self.current_step
                window_data = df.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values
                
                # Si la fenêtre est trop petite, on la remplit avec des zéros
                if len(window_data) < self.window_size:
                    padding = np.zeros((self.window_size - len(window_data), 5))
                    window_data = np.vstack([padding, window_data])
                
                obs.append(window_data)
            
            return np.array(obs, dtype=np.float32)
    
    # Créer l'environnement simplifié
    env = SimpleTradingEnv(env_kwargs)
    
    # Encapsuler dans un DummyVecEnv pour la compatibilité avec SB3
    env = DummyVecEnv([lambda: env])
    
    # Normaliser les observations
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    return env

def train_model():
    """Lance l'entraînement du modèle."""
    # Charger les configurations
    main_config, agent_config, env_config = load_configs()
    
    # Créer les répertoires de sortie
    log_dir = Path("logs/test_training")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer l'environnement
    env = create_environment(env_config, main_config)
    
    # Configuration du modèle
    agent_params = agent_config["agent"]
    policy_kwargs = {
        "features_extractor_class": CustomCNNFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 512,
            "channels": 5,  # 5 actifs (BTC, ETH, SOL, XRP, ADA)
            "kernel_sizes": [3, 3, 3],
            "strides": [1, 1, 1],
            "paddings": [1, 1, 1],
            "activation_fn": torch.nn.ReLU,
            "use_batch_norm": True,
            "dropout": 0.1
        },
        "net_arch": agent_config.get("policy", {}).get("net_arch", [256, 128]),
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True
    }
    
    # Créer le modèle PPO
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=agent_params.get("learning_rate", 3e-4),
        n_steps=agent_params.get("n_steps", 2048),
        batch_size=agent_params.get("batch_size", 64),
        n_epochs=agent_params.get("n_epochs", 10),
        gamma=agent_params.get("gamma", 0.99),
        gae_lambda=agent_params.get("gae_lambda", 0.95),
        clip_range=agent_params.get("clip_range", 0.2),
        ent_coef=agent_params.get("ent_coef", 0.0),
        vf_coef=agent_params.get("vf_coef", 0.5),
        max_grad_norm=agent_params.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard")
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="adan_ppo"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "evaluations"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Entraînement
    logger.info("Démarrage de l'entraînement...")
    model.learn(
        total_timesteps=100000,  # Réduit pour le test
        callback=callbacks,
        tb_log_name="adan_ppo_test"
    )
    
    # Sauvegarder le modèle final
    model.save(log_dir / "adan_ppo_final")
    env.save(log_dir / "vec_normalize.pkl")
    
    logger.info("Entraînement terminé avec succès!")

if __name__ == "__main__":
    train_model()
