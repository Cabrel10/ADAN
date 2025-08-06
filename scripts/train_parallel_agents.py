#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement parallèle pour 4 instances ADAN avec conditions différentes.

Stratégie d'entraînement parallèle :
- Instance 1: Capital faible (1000$) - Apprentissage conservateur
- Instance 2: Capital moyen (5000$) - Apprentissage équilibré
- Instance 3: Capital élevé (15000$) - Apprentissage agressif
- Instance 4: Capital variable - Apprentissage adaptatif

Chaque instance utilise des paramètres de risque différents mais contribue
au même modèle global via un mécanisme de partage d'expérience.
"""

import logging
import os
import sys
import time
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Définir le chemin absolu du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Import local après modification du PYTHONPATH
from adan_trading_bot.environment.multi_asset_chunked_env import (  # noqa: E402
    MultiAssetChunkedEnv,
)


def setup_logging() -> logging.Logger:
    """Configure le logging pour l'entraînement parallèle.

    Returns:
        logging.Logger: L'objet logger configuré
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/parallel_training_{timestamp}.log"

    # Créer le dossier logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)

    # Configurer le format des logs
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all messages
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite log file
            logging.StreamHandler(),
        ],
    )

    # Configurer le logger pour ce module
    logger = logging.getLogger(__name__)
    
    # Set specific log levels for verbose modules
    logging.getLogger('adan_trading_bot').setLevel(logging.DEBUG)
    logging.getLogger(
        'adan_trading_bot.environment.multi_asset_chunked_env'
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        'adan_trading_bot.data_processing.state_builder'
    ).setLevel(logging.DEBUG)
    
    # Disable excessive logging from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('stable_baselines3').setLevel(logging.INFO)

    return logger


def load_base_config(
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Charge la configuration de base et résout les variables de chemin.

    Args:
        config_override: Dictionnaire de paramètres pour écraser la configuration de base.

    Returns:
        Dict[str, Any]: Dictionnaire contenant la configuration chargée et fusionnée.
    """
    # Charger la configuration de base
    with open(os.path.join(PROJECT_ROOT, "config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Résoudre les variables de chemin
    def resolve_paths(node, config_root):
        if isinstance(node, dict):
            for key, value in node.items():
                node[key] = resolve_paths(value, config_root)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                node[i] = resolve_paths(item, config_root)
        elif isinstance(node, str):
            import re
            # Loop to handle multiple and nested variables
            # We limit to 10 iterations to prevent infinite loops
            for _ in range(10):
                match = re.search(r'\$\{(.+?)\}', node)
                if not match:
                    break
                
                path_variable = match.group(1)
                keys = path_variable.split('.')
                resolved_value = config_root
                try:
                    for k in keys:
                        resolved_value = resolved_value[k]
                    
                    # If the resolved value is a path and not absolute, make it absolute
                    if isinstance(resolved_value, str) and path_variable.startswith('paths.') and not os.path.isabs(resolved_value):
                        resolved_value = os.path.join(PROJECT_ROOT, resolved_value)

                    # Replace the variable with its resolved value
                    node = node.replace(match.group(0), str(resolved_value))
                except (KeyError, TypeError):
                    # If the key is not found, leave the variable as is and break the loop
                    break
        return node

    config = resolve_paths(config, config)

    # Appliquer les paramètres de l'override si fournis
    if config_override:
        for key, value in config_override.items():
            if key in config:
                config[key] = value

    return config


def train_single_instance(
    instance_id: int,
    total_timesteps: int,
    config_override: Optional[Dict[str, Any]] = None,
    shared_model_path: str = None,
) -> Dict[str, Any]:
    """
    Entraîne une instance spécifique du modèle avec sa configuration de worker dédiée.

    Args:
        instance_id: Identifiant numérique de l'instance (1-4)
        total_timesteps: Nombre total de pas d'entraînement
        config_override: Configuration de remplacement optionnelle
        shared_model_path: Chemin vers un modèle partagé pour le fine-tuning

    Returns:
        Dict contenant les résultats de l'entraînement
    """
    logger = logging.getLogger(f"Instance_{instance_id}")
    logger.info(f"🚀 Démarrage de l'entraînement pour l'instance {instance_id}")

    try:
        # 1. Charger la configuration de base
        base_config = load_base_config(config_override)

        # 2. Valider et extraire la configuration du worker
        worker_id_str = f"w{instance_id}"
        if "workers" not in base_config or worker_id_str not in base_config["workers"]:
            raise ValueError(
                f"Configuration pour le worker '{worker_id_str}' introuvable dans config.yaml"
            )

        # Créer une copie profonde pour éviter les effets de bord
        worker_config = copy.deepcopy(base_config["workers"][worker_id_str])

        # 3. Fusionner les configurations (si nécessaire)
        # Ici, nous pourrions ajouter une logique pour fusionner des configurations
        # spécifiques du worker avec la configuration de base

        # 4. Journalisation des informations de configuration
        logger.info(f"Instance {instance_id} - {worker_config.get('name', 'Sans nom')}")
        logger.info(f"  - Actifs: {', '.join(worker_config.get('assets', []))}")
        logger.info(f"  - Timeframes: {', '.join(worker_config.get('timeframes', []))}")
        logger.info(f"  - Jeu de données: {worker_config.get('data_split', 'train')}")
        
        # 5. S'assurer que la configuration de l'environnement est correctement structurée
        if 'environment' not in base_config:
            base_config['environment'] = {}
        
        # 6. Créer l'environnement avec la configuration fusionnée
        env = MultiAssetChunkedEnv(config=base_config, worker_config=worker_config)

        # --- Validation dimensionnelle ---
        logger.info("Performing state dimension validation...")
        try:
            # Réinitialiser l'environnement pour charger les premières données
            initial_obs, _ = env.reset()
            # Valider les dimensions
            current_data = env.data_loader.load_chunk()  # Utiliser load_chunk() au lieu de get_current_chunk()
            # La validation des dimensions est déjà effectuée dans l'environnement
            logger.info("✅ State dimension validation successful.")
        except ValueError as e:
            logger.error(f"❌ State dimension validation failed: {e}")
            raise e
        # --- Fin de la validation ---

        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")

        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])

        policy_class = "MultiInputPolicy"
        logger.info(f"Using policy: {policy_class}")
        
        # Forcer l'observation space à être un dictionnaire si ce n'est pas déjà le cas
        if not isinstance(env.observation_space, gym.spaces.Dict):
            logger.warning("L'espace d'observation n'est pas un dictionnaire, conversion en cours...")
            # Créer un nouvel espace d'observation de type Dict
            env.observation_space = gym.spaces.Dict({
                'observation': env.observation_space,
                'portfolio_state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
            })

        # Les paramètres de l'agent sont maintenant dans la config du worker
        agent_config = worker_config.get("agent_config", {})
        if not agent_config:
            raise ValueError(f"'agent_config' not found for worker {worker_id_str}")
            
        # S'assurer que le taux d'apprentissage est défini et valide
        learning_rate = float(agent_config.get("learning_rate", 0.0003))
        n_steps = int(agent_config.get("n_steps", 2048))
        batch_size = int(agent_config.get("batch_size", 64))
        gamma = float(agent_config.get("gamma", 0.99))
        
        logger.info(f"Agent configuration for worker {worker_id_str}:")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - N steps: {n_steps}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Gamma: {gamma}")

        if shared_model_path and os.path.exists(shared_model_path):
            logger.info(f"Loading shared model from {shared_model_path}")
            model = PPO.load(shared_model_path, env=vec_env)
            # Ajuster les paramètres d'apprentissage pour cette instance
            model.learning_rate = learning_rate
            model.batch_size = batch_size
            model.n_steps = n_steps
            model.gamma = gamma
        else:
            logger.info(f"Creating new model with {policy_class}")
            model = PPO(
                policy=policy_class,
                env=vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                verbose=1,
                tensorboard_log=f"logs/tensorboard/instance_{instance_id}",
            )

        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=f"instance_{instance_id}_{worker_config['name'].lower()}",
            progress_bar=False,
        )
        training_time = time.time() - start_time

        instance_model_path = (
            f"models/instance_{instance_id}_{worker_config['name'].lower()}_final.zip"
        )
        model.save(instance_model_path)

        obs = vec_env.reset()
        total_reward = 0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            if done[0]:
                obs = vec_env.reset()

        avg_reward = total_reward / 100

        vec_env.close()

        results = {
            "instance_id": instance_id,
            "name": worker_config["name"],
            "initial_capital": base_config["environment"]["initial_balance"],
            "training_time": training_time,
            "avg_reward": avg_reward,
            "model_path": instance_model_path,
            "timesteps": total_timesteps,
        }

        logger.info(
            f"✅ Instance {instance_id} completed - Avg Reward: {avg_reward:.4f}, Time: {training_time:.1f}s"
        )
        return results

    except Exception as e:
        logger.error(f"❌ Instance {instance_id} failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"instance_id": instance_id, "error": str(e), "success": False}


def main(config_path: str = "config/config.yaml"):
    """Fonction principale d'entraînement parallèle"""
    logger = setup_logging()
    logger.info("🚀 Starting ADAN Parallel Training")

    # Charger la configuration
    config = load_base_config()  # Load base config without override initially

    num_instances = config["training"]["num_instances"]
    timesteps_per_instance = config["training"]["timesteps_per_instance"]

    # Créer les répertoires nécessaires
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    logger.info("Training configuration:")
    logger.info(f"  - Timesteps per instance: {timesteps_per_instance}")
    logger.info(f"  - Parallel workers: {num_instances}")
    logger.info(f"  - Total training steps: {timesteps_per_instance * num_instances}")

    # Lancement de l'entraînement parallèle
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=num_instances) as executor:
        # Soumettre les tâches d'entraînement
        futures = {
            executor.submit(train_single_instance, i, timesteps_per_instance, None): i
            for i in range(1, num_instances + 1)
        }

        # Collecter les résultats
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Instance {i} completed successfully")
            except Exception as e:
                logger.error(f"Instance {i} failed with error: {e}")
                results.append({"instance_id": i, "error": str(e), "success": False})

    total_time = time.time() - start_time

    # Analyser les résultats
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]

    logger.info("📊 Training Results Summary:")
    logger.info(f"  - Total time: {total_time:.1f}s")
    logger.info(f"  - Successful instances: {len(successful_results)}/{len(results)}")
    logger.info(f"  - Failed instances: {len(failed_results)}/{len(results)}")

    if successful_results:
        logger.info("  - Instance Performance:")
        for result in successful_results:
            logger.info(
                f"    * {result['name']}: Reward={result['avg_reward']:.4f}, Time={result['training_time']:.1f}s"
            )

        # Fusionner les modèles réussis
        model_paths = [
            r["model_path"]
            for r in successful_results
            if os.path.exists(r["model_path"])
        ]
        if len(model_paths) > 1:
            # Assuming merge_models function exists and is imported
            # from .utils import merge_models # Example import
            # merge_models(model_paths, merged_model_path)
            logger.info(
                "Skipping model merge for now. Implement merge_models if needed."
            )

    # Sauvegarder les résultats détaillés
    results_path = (
        f"logs/parallel_training_results_{int(datetime.now().timestamp())}.json"
    )
    import json

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"📋 Detailed results saved to: {results_path}")
    logger.info("🎉 Parallel training completed!")

    return len(successful_results) == len(results)


if __name__ == "__main__":
    # Parse arguments if run from command line
    import argparse

    parser = argparse.ArgumentParser(
        description="Lancement de l'entraînement parallèle ADAN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration maître.",
    )

    args = parser.parse_args()

    success = main(args.config)
    sys.exit(0 if success else 1)
