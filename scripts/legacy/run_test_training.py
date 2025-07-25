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
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
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
    config = {
        "data": main_config["data"],
        "env": env_config["environment"],
        "portfolio": main_config["portfolio"],
        "trading": main_config["trading"],
        "rewards": main_config["rewards"]
    }
    
    # Créer l'environnement MultiAssetChunkedEnv
    env = MultiAssetChunkedEnv(config=config)
    
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
