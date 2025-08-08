#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script d'entraînement parallèle pour instances ADAN.

Ce script permet d'entraîner plusieurs instances du modèle ADAN en parallèle,
chacune avec une configuration de worker différente.
"""

import argparse
import copy
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import gym
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.utils.caching_utils import DataCacheManager


# Définir le chemin absolu du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def make_env(instance_id: int, config: Dict, cache: DataCacheManager):
    """Crée un environnement avec les données mises en cache.

    Args:
        instance_id: ID de l'instance du worker
        config: Configuration complète
        cache: Instance de DataCacheManager

    Returns:
        Environnement configuré avec les données mises en cache
    """
    logger = logging.getLogger(f"Instance_{instance_id}")

    try:
        # Vérifier la configuration des workers
        worker_keys = list(config["workers"].keys())
        worker_key = worker_keys[instance_id % len(worker_keys)]
        worker_config = copy.deepcopy(config["workers"][worker_key])

        # Sélectionner un actif et un timeframe pour ce worker
        if not worker_config.get("assets") or not worker_config.get("timeframes"):
            raise ValueError(
                "La configuration du worker doit contenir 'assets' et 'timeframes'"
            )

        # Utiliser l'ID de l'instance pour sélectionner un actif et un timeframe de manière déterministe
        asset = worker_config["assets"][instance_id % len(worker_config["assets"])]
        timeframe = worker_config["timeframes"][
            instance_id % len(worker_config["timeframes"])
        ]

        logger.info(
            "Chargement des données pour %s - Asset: %s, Timeframe: %s",
            worker_key,
            asset,
            timeframe,
        )

        # Initialisation du chargeur de données
        data_loader = ChunkedDataLoader(config, worker_config)

        # Chargement des données
        logger.info("Chargement des données pour %s - %s...", asset, timeframe)
        data_dict = data_loader.load_chunk()

        # Vérification des données
        if asset not in data_dict or timeframe not in data_dict[asset]:
            err_msg = f"Aucune donnée pour {asset} sur {timeframe}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        data = data_dict[asset][timeframe]

        # Création d'un scaler factice (à remplacer par un vrai scaler si nécessaire)
        scaler = None

        logger.info(
            "Données chargées: %d points pour %s (%s)", len(data), asset, timeframe
        )

        # Création de l'environnement
        return MultiAssetChunkedEnv(
            config=config,
            worker_config=worker_config,
            data_loader_instance=data_loader,
            **worker_config.get("env_params", {}),
        )

    except Exception as e:
        logger.error("Erreur lors de la création de l'environnement: %s", str(e))
        logger.exception("Détails de l'erreur:")
        raise


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
            logging.FileHandler(log_file, mode="w"),  # Overwrite log file
            logging.StreamHandler(),
        ],
    )

    # Configurer le logger pour ce module
    logger = logging.getLogger(__name__)

    # Set specific log levels for verbose modules
    logging.getLogger("adan_trading_bot").setLevel(logging.DEBUG)
    logging.getLogger("adan_trading_bot.environment.multi_asset_chunked_env").setLevel(
        logging.DEBUG
    )
    logging.getLogger("adan_trading_bot.data_processing.state_builder").setLevel(
        logging.DEBUG
    )

    # Disable excessive logging from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("stable_baselines3").setLevel(logging.INFO)

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
                match = re.search(r"\$\{(.+?)\}", node)
                if not match:
                    break

                path_variable = match.group(1)
                keys = path_variable.split(".")
                resolved_value = config_root
                try:
                    for k in keys:
                        resolved_value = resolved_value[k]

                    # If the resolved value is a path and not absolute, make it absolute
                    if (
                        isinstance(resolved_value, str)
                        and path_variable.startswith("paths.")
                        and not os.path.isabs(resolved_value)
                    ):
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


class TimeoutException(Exception):
    """Exception levée quand le timeout est atteint"""

    pass


def timeout_handler(signum, frame):
    """Gestionnaire de signal pour le timeout"""
    raise TimeoutException("Temps d'entraînement écoulé")


def save_checkpoint(model, optimizer, epoch: int, path: str):
    """
    Sauvegarde un checkpoint complet incluant :
    - État du modèle
    - État de l'optimiseur
    - État du générateur de nombres aléatoires
    - Métadonnées
    """
    import random

    import numpy as np
    import torch

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "timestamp": time.time(),
    }

    # Créer le répertoire parent si nécessaire
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Sauvegarder avec tolérance aux erreurs
    try:
        torch.save(checkpoint, path)
        return path
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du checkpoint: {e}")
        return None


def load_checkpoint(model, optimizer, path: str):
    """
    Charge un checkpoint complet et restaure :
    - L'état du modèle
    - L'état de l'optimiseur
    - Les états des générateurs de nombres aléatoires

    Returns:
        int: L'époque à partir de laquelle reprendre l'entraînement
    """
    import random

    import numpy as np
    import torch

    if not os.path.exists(path):
        logging.warning(
            f"Aucun checkpoint trouvé à {path}, démarrage d'un nouvel entraînement"
        )
        return 0

    try:
        # Charger le checkpoint
        device = model.device if hasattr(model, "device") else "cpu"
        checkpoint = torch.load(path, map_location=device)

        # Restaurer les états du modèle et de l'optimiseur
        model.policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restaurer les états des générateurs aléatoires
        if "rng_state" in checkpoint:
            random.setstate(checkpoint["rng_state"]["python"])
            np.random.set_state(checkpoint["rng_state"]["numpy"])
            torch.set_rng_state(checkpoint["rng_state"]["torch"].to("cpu"))

        logging.info(f"Checkpoint chargé depuis {path} (époque {checkpoint['epoch']})")
        return checkpoint["epoch"]

    except Exception as e:
        logging.error(f"Erreur lors du chargement du checkpoint {path}: {e}")
        return 0


def train_single_instance(
    instance_id: int,
    total_timesteps: int,
    config_override: Optional[Dict[str, Any]] = None,
    shared_model_path: str = None,
    checkpoint_path: str = None,
    timeout: int = None,
) -> Dict[str, Any]:
    """Entraîne une instance spécifique du modèle avec sa configuration de worker dédiée.

    Gère le timeout et la sauvegarde automatique des checkpoints.
    Utilise un cache pour les données et les scalers.

    Args:
        instance_id: Identifiant numérique de l'instance (1-4)
        total_timesteps: Nombre total de pas d'entraînement
        config_override: Configuration de remplacement optionnelle
        shared_model_path: Chemin vers un modèle partagé pour le fine-tuning
        checkpoint_path: Chemin pour sauvegarder les checkpoints
        timeout: Délai maximal d'exécution en secondes

    Returns:
        Dict contenant les résultats de l'entraînement
    """
    logger = setup_logging()
    start_time = time.time()
    last_checkpoint_time = start_time
    checkpoint_interval = 300  # 5 minutes en secondes

    try:
        # Charger la configuration
        config = load_base_config(config_override)

        # Initialiser le cache des données
        cache_dir = os.path.join(PROJECT_ROOT, "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache = DataCacheManager(cache_dir)

        logger.info(f"🚀 Démarrage de l'entraînement pour l'instance {instance_id}")
        logger.info(f"Utilisation du cache dans : {cache_dir}")

        # Créer l'environnement avec les données mises en cache
        env = make_env(instance_id, config, cache)

        # Configuration spécifique au worker
        worker_keys = list(config["workers"].keys())
        worker_key = worker_keys[instance_id % len(worker_keys)]
        worker_config = copy.deepcopy(config["workers"][worker_key])
        agent_config = config.get("agent", {})

        # Ajouter la clé du worker à la configuration pour référence
        worker_config["worker_key"] = worker_key

        # 3. Fusionner les configurations (si nécessaire)
        # Ici, nous pourrions ajouter une logique pour fusionner des configurations
        # spécifiques du worker avec la configuration de base

        # 4. Journalisation des informations de configuration
        logger.info(f"Instance {instance_id} - {worker_config.get('name', 'Sans nom')}")
        logger.info(f"  - Actifs: {', '.join(worker_config.get('assets', []))}")
        logger.info(f"  - Timeframes: {', '.join(worker_config.get('timeframes', []))}")
        logger.info(f"  - Jeu de données: {worker_config.get('data_split', 'train')}")

        # 5. Charger la configuration de base
        base_config = load_base_config(config_override)

        # 6. S'assurer que la configuration de l'environnement est correctement structurée
        if "environment" not in base_config:
            base_config["environment"] = {}

        # 7. Créer l'environnement avec la configuration fusionnée
        env = MultiAssetChunkedEnv(config=base_config, worker_config=worker_config)

        # --- Validation dimensionnelle ---
        logger.info("Performing state dimension validation...")
        try:
            # Réinitialiser l'environnement pour charger les premières données
            initial_obs, _ = env.reset()
            # Valider les dimensions
            current_data = (
                env.data_loader.load_chunk()
            )  # Utiliser load_chunk() au lieu de get_current_chunk()
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
            logger.warning(
                "L'espace d'observation n'est pas un dictionnaire, conversion en cours..."
            )
            # Créer un nouvel espace d'observation de type Dict
            # Extraire la forme de l'espace d'observation existant
            if hasattr(env.observation_space, "shape"):
                obs_shape = env.observation_space.shape
                # Créer un espace Box pour l'observation avec les mêmes caractéristiques
                obs_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
                )
                # Créer l'espace d'observation final
                env.observation_space = gym.spaces.Dict(
                    {
                        "observation": obs_space,
                        "portfolio_state": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
                        ),
                    }
                )
            else:
                # Si on ne peut pas déterminer la forme, utiliser l'espace tel quel
                env.observation_space = gym.spaces.Dict(
                    {
                        "observation": env.observation_space,
                        "portfolio_state": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
                        ),
                    }
                )

        # Les paramètres de l'agent sont maintenant dans la config du worker
        agent_config = worker_config.get("agent_config", {})
        if not agent_config:
            raise ValueError(f"'agent_config' not found for worker {instance_id}")

        # Initialiser le modèle avec MultiInputPolicy pour les espaces d'observation de type dictionnaire
        policy_class = "MultiInputPolicy"
        logger.info(f"Using policy: {policy_class} (for dict observation space)")

        # Créer ou charger le modèle
        if shared_model_path and os.path.exists(shared_model_path):
            logger.info(f"Chargement du modèle partagé depuis {shared_model_path}")
            model = PPO.load(shared_model_path, env=vec_env, verbose=1)
        else:
            logger.info("Création d'un nouveau modèle")
            # Créer une copie de la configuration sans la clé 'policy' si elle existe
            model_config = {k: v for k, v in agent_config.items() if k != "policy"}
            model = PPO(
                policy=policy_class,
                env=vec_env,
                verbose=1,
                tensorboard_log=f"logs/tensorboard/instance_{instance_id}",
                **model_config,
            )

        # Configuration de l'entraînement
        n_steps = agent_config.get("n_steps", 2048)
        epochs = (total_timesteps // n_steps) + 1
        checkpoint_interval = 300  # 5 minutes

        # Charger le checkpoint s'il existe
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(
                model, model.policy.optimizer, checkpoint_path
            )
            logger.info(
                f"Reprise de l'entraînement à partir de l'epoch {start_epoch+1}/{epochs}"
            )

        # Boucle d'entraînement
        try:
            for epoch in range(start_epoch, epochs):
                current_time = time.time()
                time_elapsed = current_time - start_time

                # Vérifier le timeout
                if timeout and time_elapsed >= timeout:
                    logger.info(f"Timeout atteint après {time_elapsed:.1f} secondes")
                    break

                logger.info(f"Début de l'epoch {epoch+1}/{epochs}")

                # Entraînement sur une epoch
                model.learn(
                    total_timesteps=n_steps,
                    tb_log_name=f"instance_{instance_id}_{worker_config['name'].lower()}",
                    progress_bar=True,
                    reset_num_timesteps=False,
                )

                # Sauvegarder le checkpoint périodiquement
                current_time = time.time()
                if (
                    checkpoint_path
                    and (current_time - last_checkpoint_time) >= checkpoint_interval
                ):
                    save_checkpoint(
                        model, model.policy.optimizer, epoch, checkpoint_path
                    )
                    logger.info(f"Checkpoint sauvegardé à {checkpoint_path}")
                    last_checkpoint_time = current_time

            # Sauvegarder le modèle final
            if checkpoint_path:
                save_checkpoint(model, model.policy.optimizer, epoch, checkpoint_path)
                logger.info(f"Checkpoint final sauvegardé à {checkpoint_path}")

        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise

        training_time = time.time() - start_time
        worker_name = worker_config["name"].lower().replace(" ", "_")
        instance_model_path = f"models/instance_{instance_id}_{worker_name}_final.zip"
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


def main(
    config_path: str = "config/config.yaml",
    timeout: int = None,
    checkpoint_dir: str = "checkpoints",
):
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

    # Créer le dossier de checkpoints si nécessaire
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_instances) as executor:
        # Soumettre les tâches d'entraînement avec les paramètres de timeout
        futures = {
            executor.submit(
                train_single_instance,
                instance_id=i,
                total_timesteps=timesteps_per_instance,
                config_override=None,
                shared_model_path=None,
                checkpoint_path=os.path.join(
                    checkpoint_dir, f"instance_{i}_checkpoint.pt"
                )
                if checkpoint_dir
                else None,
                timeout=timeout,
            ): i
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
        description="Train ADAN trading bot with timeout and checkpoint support"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Maximum training time in seconds"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    args = parser.parse_args()

    # Call main with all arguments
    success = main(
        config_path=args.config,
        timeout=args.timeout,
        checkpoint_dir=args.checkpoint_dir,
    )
    sys.exit(0 if success else 1)
