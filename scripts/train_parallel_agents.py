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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

from typing import Optional, Dict, Any, List
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Définir le chemin absolu du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Import local après modification du PYTHONPATH
from adan_trading_bot.environment.multi_asset_chunked_env import (  # noqa: E402
    MultiAssetChunkedEnv
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
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(
                f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler()
        ]
    )

    # Configurer le logger pour ce module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger


def load_base_config(config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Charge la configuration de base et résout les variables de chemin.
    
    Args:
        config_override: Dictionnaire de paramètres pour écraser la configuration de base.
    
    Returns:
        Dict[str, Any]: Dictionnaire contenant la configuration chargée et fusionnée.
    """
    # Charger la configuration de base
    with open(os.path.join(PROJECT_ROOT, 'config/config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # Résoudre les variables de chemin
    def resolve_paths(node):
        if isinstance(node, dict):
            for key, value in node.items():
                node[key] = resolve_paths(value)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                node[i] = resolve_paths(item)
        elif isinstance(node, str):
            # Regex pour trouver les variables comme ${section.key}
            import re
            match = re.search(r"\$\{(.+?)\}", node)
            if match:
                path_variable = match.group(1)
                keys = path_variable.split('.')
                value = config
                try:
                    for k in keys:
                        value = value[k]
                    # Remplacer la variable par sa valeur résolue
                    if path_variable.startswith('paths.') or path_variable.startswith('data.data_dir'):
                        # Pour tous les chemins, utiliser le chemin absolu du projet
                        # Supprimer le préfixe ADAN/ si présent
                        if isinstance(value, str) and value.startswith('ADAN/'):
                            value = value[5:]  # Remove 'ADAN/' prefix
                        # Si c'est un chemin relatif, le rendre absolu
                        if isinstance(value, str) and not os.path.isabs(value):
                            value = os.path.join(PROJECT_ROOT, value)
                        return value
                    return node.replace(match.group(0), value)
                except KeyError:
                    # Laisser la variable telle quelle si non trouvée
                    return node
        return node

    config = resolve_paths(config)
    
    # Appliquer les paramètres de l'override si fournis
    if config_override:
        for key, value in config_override.items():
            if key in config:
                config[key] = value
    
    return config

def create_instance_config(base_config: Dict[str, Any], instance_id: int) -> Dict[str, Any]:
    """Crée une configuration spécifique pour une instance d'entraînement.
    
    Args:
        base_config: Configuration de base chargée depuis le fichier
        instance_id: ID de l'instance (1-4)
        
    Returns:
        Configuration spécifique à l'instance
    """
    config = base_config.copy()
    
    # Configuration spécifique à chaque instance
    instance_configs = [
        {
            "name": "Conservative",
            "initial_capital": 1000.0,
            "risk_level": 0.5,
            "learning_rate": 0.0003,
            "ent_coef": 0.01
        },
        {
            "name": "Balanced",
            "initial_capital": 5000.0,
            "risk_level": 1.0,
            "learning_rate": 0.001,
            "ent_coef": 0.02
        },
        {
            "name": "Aggressive",
            "initial_capital": 15000.0,
            "risk_level": 1.5,
            "learning_rate": 0.003,
            "ent_coef": 0.03
        },
        {
            "name": "Adaptive",
            "initial_capital": 10000.0,
            "risk_level": 1.2,
            "learning_rate": 0.0005,
            "ent_coef": 0.015
        }
    ]
    
    # Appliquer la configuration spécifique à l'instance
    instance_cfg = instance_configs[instance_id - 1]  # -1 car les IDs commencent à 1
    
    # Mettre à jour la configuration de base
    if "environment" not in config:
        config["environment"] = {}
    if "training" not in config:
        config["training"] = {}
    if "agent" not in config:
        config["agent"] = {}
    
    # Configuration de l'environnement
    config["environment"]["initial_capital"] = instance_cfg["initial_capital"]
    config["environment"]["risk_level"] = instance_cfg["risk_level"]
    
    # Configuration de l'agent
    config["agent"]["learning_rate"] = instance_cfg["learning_rate"]
    config["agent"]["batch_size"] = 64  # Taille de lot par défaut
    config["agent"]["n_steps"] = 2048    # Nombre de pas par mise à jour
    config["agent"]["gamma"] = 0.99      # Facteur de remise
    config["agent"]["ent_coef"] = instance_cfg["ent_coef"]
    
    # Ajouter les infos de l'instance à la config
    config["instance"] = {
        "id": instance_id,
        "name": instance_cfg["name"]
    }
    
    return config

def train_single_instance(instance_id: int, total_timesteps: int, config_override: Optional[Dict[str, Any]] = None, shared_model_path: str = None) -> Dict[str, Any]:
    """Entraîner une instance spécifique"""
    logger = logging.getLogger(f"Instance_{instance_id}")
    logger.info(f"🚀 Starting training for Instance {instance_id}")
    
    try:
        # Charger la configuration
        base_config = load_base_config(config_override)
        config = create_instance_config(base_config, instance_id)
        
        logger.info(f"Instance {instance_id} ({config['instance']['name']}) - Capital: ${config['environment']['initial_capital']}")
        
        # Créer l'environnement
        env = MultiAssetChunkedEnv(config=config)
        
        # --- Validation dimensionnelle ---
        logger.info("Performing state dimension validation...")
        try:
            # Réinitialiser l'environnement pour charger les premières données
            initial_obs, _ = env.reset()
            # Valider les dimensions
            # Utiliser la méthode get_current_chunk() au lieu de current_chunk
            current_data = env.data_loader.get_current_chunk()
            env.state_builder.validate_dimension(current_data)
            logger.info("✅ State dimension validation successful.")
        except ValueError as e:
            logger.error(f"❌ State dimension validation failed: {e}")
            # Arrêter l'entraînement pour cette instance
            raise e
        # --- Fin de la validation ---
        
        # Afficher les espaces d'observation et d'action pour le débogage
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # Envelopper l'environnement avec Monitor et DummyVecEnv
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        
        # Choisir automatiquement la politique en fonction de l'espace d'observation
        if isinstance(env.observation_space, gym.spaces.Dict):
            policy_class = "MultiInputPolicy"
        else:
            policy_class = "MlpPolicy"
            
        logger.info(f"Using policy: {policy_class}")
        
        # Créer ou charger le modèle
        if shared_model_path and os.path.exists(shared_model_path):
            logger.info(f"Loading shared model from {shared_model_path}")
            model = PPO.load(shared_model_path, env=vec_env)
            # Ajuster les paramètres d'apprentissage pour cette instance
            model.learning_rate = config['agent']['learning_rate']
            model.batch_size = config['agent']['batch_size']
            model.n_steps = config['agent']['n_steps']
            model.gamma = config['agent']['gamma']
        else:
            logger.info(f"Creating new model with {policy_class}")
            model = PPO(
                policy=policy_class,
                env=vec_env,
                learning_rate=config['agent']['learning_rate'],
                n_steps=config['agent']['n_steps'],
                batch_size=config['agent']['batch_size'],
                gamma=config['agent']['gamma'],
                verbose=1,  # Augmenter la verbosité pour le débogage
                tensorboard_log=f"logs/tensorboard/instance_{instance_id}"
            )
        
        # Entraînement
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=f"instance_{instance_id}_{config['instance']['name'].lower()}",
            progress_bar=False
        )
        training_time = time.time() - start_time
        
        # Sauvegarde du modèle de l'instance
        instance_model_path = f"models/instance_{instance_id}_{config['instance']['name'].lower()}_final.zip"
        model.save(instance_model_path)
        
        # Évaluation rapide
        obs = vec_env.reset()
        total_reward = 0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            if done[0]:
                obs = vec_env.reset()
        
        avg_reward = total_reward / 100
        
        # Fermeture
        vec_env.close()
        
        results = {
            'instance_id': instance_id,
            'name': config['instance']['name'],
            'initial_capital': config['environment']['initial_capital'],
            'training_time': training_time,
            'avg_reward': avg_reward,
            'model_path': instance_model_path,
            'timesteps': total_timesteps
        }
        
        logger.info(f"✅ Instance {instance_id} completed - Avg Reward: {avg_reward:.4f}, Time: {training_time:.1f}s")
        return results
        
    except Exception as e:
        logger.error(f"❌ Instance {instance_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'instance_id': instance_id,
            'error': str(e),
            'success': False
        }

def main(config_path: str = 'config/config.yaml'):
    """Fonction principale d'entraînement parallèle"""
    logger = setup_logging()
    logger.info("🚀 Starting ADAN Parallel Training")
    
    # Charger la configuration
    config = load_base_config() # Load base config without override initially
    
    num_instances = config['training']['num_instances']
    timesteps_per_instance = config['training']['timesteps_per_instance']
    
    # Créer les répertoires nécessaires
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    
    logger.info(f"Training configuration:")
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
                results.append({
                    'instance_id': i,
                    'error': str(e),
                    'success': False
                })
    
    total_time = time.time() - start_time
    
    # Analyser les résultats
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    logger.info("📊 Training Results Summary:")
    logger.info(f"  - Total time: {total_time:.1f}s")
    logger.info(f"  - Successful instances: {len(successful_results)}/{len(results)}")
    logger.info(f"  - Failed instances: {len(failed_results)}/{len(results)}")
    
    if successful_results:
        logger.info("  - Instance Performance:")
        for result in successful_results:
            logger.info(f"    * {result['name']}: Reward={result['avg_reward']:.4f}, Time={result['training_time']:.1f}s")
        
        # Fusionner les modèles réussis
        model_paths = [r['model_path'] for r in successful_results if os.path.exists(r['model_path'])]
        if len(model_paths) > 1:
            # Assuming merge_models function exists and is imported
            # from .utils import merge_models # Example import
            # merge_models(model_paths, merged_model_path)
            logger.info("Skipping model merge for now. Implement merge_models if needed.")
    
    # Sauvegarder les résultats détaillés
    results_path = f"logs/parallel_training_results_{int(datetime.now().timestamp())}.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📋 Detailed results saved to: {results_path}")
    logger.info("🎉 Parallel training completed!")
    
    return len(successful_results) == len(results)

if __name__ == "__main__":
    # Parse arguments if run from command line
    import argparse
    parser = argparse.ArgumentParser(description="Lancement de l'entraînement parallèle ADAN.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Chemin vers le fichier de configuration maître.')
    
    args = parser.parse_args()
    
    success = main(args.config)
    sys.exit(0 if success else 1)