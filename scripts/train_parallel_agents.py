#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script d'entraînement parallèle pour instances ADAN."""

import logging
import os
import signal
import sys
import time
import traceback
import warnings
import yaml
import numpy as np
import torch  # utilisé dans policy_kwargs / model init - s'assurer présent
try:
    import gymnasium as gym
except Exception:
    # Si gymnasium n'est pas installé, on essaie gym (legacy) — mais SB3 recommande gymnasium
    import gym  # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Optional, List, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch as th

gym.logger.set_level(40)  # Suppress Gym warnings

# Désactiver les avertissements spécifiques
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module='stable_baselines3'
)
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning
)

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecNormalize
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FlattenObservation
import gymnasium.spaces as spaces

class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Wrapper minimal pour adapter un env Gymnasium (reset -> (obs,info), step -> (obs,rew,term,trunc,info))
    à l'API Gym attendue par certains composants (SB3 DummyVecEnv / Monitor).
    - reset() retourne obs (dropping info)
    - step(action) retourne (obs, reward, done, info) where done = terminated or truncated
    """
    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        # Gymnasium returns (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            logger.debug("GymnasiumToGymWrapper.reset: returning obs only (dropping info).")
            return obs
        return out

    def step(self, action):
        out = super().step(action)
        # gymnasium step returns (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        # pass-through if already gym-like (obs, reward, done, info)
        if isinstance(out, tuple) and len(out) == 4:
            return out
        raise RuntimeError(f"Unexpected step return shape from wrapped env: {type(out)} / len={len(out) if isinstance(out, tuple) else 'N/A'}")

def validate_env_observation(env):
    """
    Validation simple après création de env.
    Lève une exception si format incorrect.
    """
    obs = env.reset()
    # If wrapper hasn't been applied and reset returns (obs, info) handle that
    if isinstance(obs, tuple) and len(obs) == 2:
        obs = obs[0]
    if not isinstance(obs, dict):
        raise TypeError(f"Invalid observation type from env.reset(): {type(obs)} — expected dict with keys 'observation' and 'portfolio_state'.")
    required = {'observation', 'portfolio_state'}
    if not required.issubset(set(obs.keys())):
        raise KeyError(f"Observation dict missing required keys. Keys present: {list(obs.keys())}")
    market = obs['observation']
    portfolio = obs['portfolio_state']
    if not isinstance(market, np.ndarray):
        raise TypeError(f"'observation' must be np.ndarray, got {type(market)}")
    if market.ndim != 3:
        raise ValueError(f"'observation' must be 3D (timeframes, window, features). Got ndim={market.ndim}")
    if not isinstance(portfolio, np.ndarray):
        raise TypeError(f"'portfolio_state' must be np.ndarray, got {type(portfolio)}")
    if portfolio.ndim != 1:
        raise ValueError(f"'portfolio_state' must be 1D vector (17,), got shape={portfolio.shape}")
    logger.info("Env observation validated successfully.")
    return True
import numpy as np
import torch as th

# Désactiver les avertissements spécifiques
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module='stable_baselines3'
)
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning
)

from adan_trading_bot.utils.caching_utils import DataCacheManager


class TimeoutException(Exception):
    """Exception levée quand le timeout est atteint."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Temps d'exécution dépassé")

# Configuration du logger
logger = logging.getLogger(__name__)

# Définir le répertoire racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
sys.path.append(PROJECT_ROOT)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Extrait les caractéristiques à partir des observations de type Dict.
    Hérite de BaseFeaturesExtractor pour une meilleure intégration avec SB3.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,  # Dimension de sortie requise par SB3
        cnn_output_dim: int = 64,
        mlp_extractor_net_arch: Optional[List[int]] = None
    ) -> None:
        # Appel au constructeur parent avec la dimension de sortie
        super().__init__(observation_space, features_dim=features_dim)
        
        extractors = {}
        total_concat_size = 0
        
        # Pour chaque clé de l'espace d'observation
        for key, subspace in observation_space.spaces.items():
            if key == "observation":  # Traitement des données d'image
                # Calcul de la taille après aplatissement
                n_flatten = 1
                for i in range(len(subspace.shape)):
                    n_flatten *= subspace.shape[i]
                
                # Réseau pour traiter les données d'image
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_flatten, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
                total_concat_size += 128
            else:  # Traitement des données vectorielles
                # Utilisation d'un MLP simple pour les données vectorielles
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )
                total_concat_size += 32
        
        self.extractors = nn.ModuleDict(extractors)
        
        # Couche linéaire finale pour adapter à la dimension de sortie souhaitée
        self.fc = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensor_list = []
        
        # Extraire les caractéristiques pour chaque clé
        for key, extractor in self.extractors.items():
            if key in observations:
                # S'assurer que les observations sont au bon format
                x = observations[key]
                if isinstance(x, np.ndarray):
                    x = th.as_tensor(x, device=self.device, dtype=th.float32)
                encoded_tensor_list.append(extractor(x))
        
        # Concaténer toutes les caractéristiques et appliquer la couche finale
        return self.fc(th.cat(encoded_tensor_list, dim=1))




class GymnasiumToSB3Wrapper(gym.Wrapper):
    """Wrapper pour convertir un environnement Gymnasium en un format compatible avec Stable Baselines 3."""
    def __init__(self, env: gym.Env) -> None:
        """Initialize the Gymnasium to SB3 wrapper.
        
        Args:
            env: The Gymnasium environment to wrap
        """
        super().__init__(env)
        
        # Convertir l'espace d'observation en un format compatible
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Pour les espaces de type Dict, on conserve la structure
            self.observation_space = env.observation_space
        else:
            # Pour les autres types d'espaces, on essaie de les convertir
            self.observation_space = env.observation_space
            
        self.action_space = env.action_space
        self.metadata = getattr(env, 'metadata', {'render_modes': []})
        
        # Activer le mode vectorisé si nécessaire
        self.is_vector_env = hasattr(env, 'num_envs')

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation and info."""
        obs, info = self.env.reset(**kwargs)
        
        # S'assurer que l'observation est au bon format
        if isinstance(obs, dict) and 'observation' in obs and 'portfolio_state' in obs:
            # Déjà au bon format
            return obs, info
            
        # Gérer les autres formats d'observation si nécessaire
        return obs, info

    def step(self, action):
        """Take an action in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # S'assurer que l'observation est au bon format
        if isinstance(obs, dict) and 'observation' in obs and 'portfolio_state' in obs:
            # Déjà au bon format
            return obs, reward, terminated or truncated, False, info
            
        # Gérer les autres formats d'observation si nécessaire
        return obs, reward, terminated or truncated, False, info

    def render(self, mode: str = "human"):
        return self.env.render(mode)


# Local application imports
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv, 
    SubprocVecEnv, 
    VecNormalize, 
    VecCheckNan,
    VecTransposeImage
)
from stable_baselines3.common.utils import set_random_seed

def _normalize_obs_for_sb3(obs: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize observation for Stable Baselines 3 compatibility.
    
    Handles Gym's tuple (obs, info) and ensures proper numpy arrays.
    Converts observations to the format expected by SB3.
    
    Args:
        obs: Observation to normalize (can be tuple, dict, numpy array).
            L'observation à normaliser (peut être un tuple, un dict,
            un tableau numpy).
            
    Returns:
        Union[np.ndarray, Dict[str, np.ndarray]]: Normalized observation for SB3.
    """
    # Handle case where obs is a tuple (obs, info) from Gymnasium
    if isinstance(obs, tuple) and len(obs) >= 1:
        obs = obs[0]  # Take only the observation part
    
    # Handle dict observations (for MultiInputPolicy)
    if isinstance(obs, dict):
        # Ensure all values are numpy arrays and handle potential tuples
        normalized_obs = {}
        for k, v in obs.items():
            if isinstance(v, tuple):
                v = v[0]  # Take first element if it's a tuple
            normalized_obs[k] = np.asarray(v, dtype=np.float32)
        return normalized_obs
    
    # Handle numpy arrays
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False)
    
    # Handle PyTorch tensors
    if hasattr(obs, 'numpy'):
        return obs.detach().cpu().numpy()
    
    # Handle lists and other array-like objects
    try:
        return np.asarray(obs, dtype=np.float32)
    except Exception as e:
        error_msg = f"Could not convert observation to numpy array: {obs}, error: {e}"
        raise ValueError(error_msg) from e


def safe_predict(model, obs, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Wrapper sécurisé autour de model.predict pour gérer différents formats d'obs.
    
    Gère les différences d'API entre Gym et SB3, et assure une conversion
    robuste des observations en tableaux numpy.
    
    Args:
        model: Le modèle SB3 à utiliser pour la prédiction
        obs: L'observation d'entrée (peut être un tuple, un dict, un tableau numpy)
        deterministic: Si True, utilise une politique déterministe
        
    Returns:
        Tuple (action, state) comme retourné par model.predict
    """
    try:
        # Normaliser l'observation pour SB3
        normalized_obs = _normalize_obs_for_sb3(obs)
        
        # Gestion spéciale pour les espaces d'observation de type Dict
        if hasattr(model, 'observation_space') and isinstance(model.observation_space, spaces.Dict):
            # Si l'observation est déjà un dict, on le laisse tel quel
            if not isinstance(normalized_obs, dict):
                # Sinon, on essaie de convertir en dict si nécessaire
                normalized_obs = {"observation": normalized_obs}
        
        # Effectuer la prédiction avec gestion des erreurs
        try:
            return model.predict(normalized_obs, deterministic=deterministic)
        except (ValueError, TypeError) as e:
            # Gestion des erreurs spécifiques à SB3
            if "You have passed a tuple" in str(e):
                if isinstance(normalized_obs, tuple):
                    normalized_obs = normalized_obs[0]
                return model.predict(normalized_obs, deterministic=deterministic)
            raise
            
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {e}")
        logging.debug(traceback.format_exc())
        
        # Retourner une action par défaut en cas d'erreur
        if hasattr(model, 'action_space') and hasattr(model.action_space, 'sample'):
            return model.action_space.sample(), None
        return np.zeros(1), None


def configure_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Configure et retourne un logger avec les handlers appropriés.

    Args:
        name: Nom du logger
        log_level: Niveau de log (par défaut: logging.INFO)

    Returns:
        Instance de logger configurée
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Crée un formateur de logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Crée un handler pour la sortie console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Crée un répertoire de logs s'il n'existe pas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Crée un handler pour le fichier de log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Ajoute les handlers au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Évite la propagation des logs vers le logger racine
    logger.propagate = False

    return logger


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Charge et valide la configuration complète."""
    if config_path is None:
        possible_paths = [
            'config/config.yaml',
            'config.yaml',
            '/home/morningstar/Documents/trading/bot/config/config.yaml',
            '/home/morningstar/Documents/trading/bot/config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError("Fichier de configuration non trouvé")
    
    logger.info(f"Chargement de la configuration depuis: {config_path}")
    
    try:
        # Charger la configuration avec chemins fixes
        config = load_base_config(config_path)
        
        # Validation des sections requises
        required_sections = ['general', 'paths', 'workers', 'environment', 'agent']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Section manquante dans la configuration: {section}")
        
        return config
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        raise

# Définir le répertoire racine du projet
PROJECT_ROOT = "/home/morningstar/Documents/trading/bot"
sys.path.append(PROJECT_ROOT)

# Configuration du logger principal
logger = configure_logger(__name__)

# Ajouter le répertoire src au chemin Python
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def make_env_fn(env_config, rank=0, use_vecnormalize=True):
    """Crée une fonction d'initialisation d'environnement.
    
    Args:
        env_config: Configuration pour MultiAssetChunkedEnv
        rank: Identifiant du worker
        use_vecnormalize: Si True, applique VecNormalize
        
    Returns:
        Fonction d'initialisation d'environnement
    """
    def _init():
        # Créer une configuration de travailleur minimale si non fournie
        worker_config = env_config.get('worker_config', {
            'assets': env_config.get('assets', ['BTCUSDT']),
            'timeframes': env_config.get('timeframes', ['1h']),
            'window_size': env_config.get('window_size', 100)
        })
    
    return _init

def build_vector_envs(
    config: Dict[str, Any],
    num_envs: int = 4,
    start_method: Optional[str] = None,
    normalize: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    gamma: float = 0.99,
    seed: Optional[int] = None,
    log_dir: Optional[str] = None,
) -> Tuple[Union[DummyVecEnv, SubprocVecEnv], Optional[VecNormalize]]:
    """
    Build a vectorized environment for parallel training.

    Args:
        config: Configuration dictionary
        num_envs: Number of parallel environments
        start_method: Method to start subprocesses ('fork', 'spawn', or 'forkserver')
        normalize: Whether to normalize observations and rewards
        norm_obs: Whether to normalize observations
        norm_reward: Whether to normalize rewards
        clip_obs: Maximum absolute value for observation normalization
        clip_reward: Maximum absolute value for reward normalization
        gamma: Discount factor for reward normalization
        seed: Random seed for reproducibility
        log_dir: Directory to save normalization statistics

    Returns:
        Tuple containing the vectorized environment and the normalization wrapper
    """
    # Set start method for subprocesses if specified
    if start_method is not None:
        logger.info(f"Setting multiprocessing start method to '{start_method}'")
        try:
            import multiprocessing as mp
            mp_ctx = mp.get_context(start_method)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to set start method '{start_method}': {e}")
            mp_ctx = None
    else:
        mp_ctx = None

    # Create a list of environment creation functions
    env_fns = []
    for i in range(num_envs):
        # Create a new config for each environment with a unique seed
        env_config = deepcopy(config)
        env_config["seed"] = seed + i if seed is not None else None
        env_config["rank"] = i
        
        # Create environment creation function with proper closure
        def make_env(rank: int, env_config: Dict[str, Any]):
            def _init():
                # Create a deep copy of the config to avoid sharing state
                local_config = deepcopy(env_config)
                
                # Create the base environment
                env = make_env_from_config(local_config)
                
                # Log environment details
                logger.info(f"[Worker {rank}] Environment created successfully")
                logger.info(f"[Worker {rank}] Observation space: {env.observation_space}")
                logger.info(f"[Worker {rank}] Action space: {env.action_space}")
                
                # Apply our custom compatibility wrapper first
                env = SB3GymCompatibilityWrapper(env)
                
                # Apply monitoring for episode statistics
                env = Monitor(env)
                
                # Apply Gymnasium to Gym wrapper if needed (for older SB3 versions)
                try:
                    from stable_baselines3.common.monitor import Monitor
                    env = Monitor(env)
                    
                    # Try to wrap with Gymnasium compatibility
                    try:
                        from gymnasium.wrappers import EnvCompatibility
                        env = EnvCompatibility(env)
                    except ImportError:
                        logger.warning("Gymnasium EnvCompatibility wrapper not available")
                        
                except Exception as e:
                    logger.warning(f"Could not apply Monitor wrapper: {e}")
                
                return env
            
            return _init
        
        env_fns.append(make_env(i, env_config))

    # Create the vectorized environment
    if num_envs == 1 or mp_ctx is None:
        logger.info(f"Creating DummyVecEnv with {num_envs} environments")
        env = DummyVecEnv(env_fns)
    else:
        logger.info(f"Creating SubprocVecEnv with {num_envs} environments using {mp_ctx.get_start_method()}")
        env = SubprocVecEnv(env_fns, start_method=start_method)

    # Apply normalization if requested
    vec_norm = None
    if normalize:
        try:
            # Get the path for saving normalization statistics
            norm_path = None
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                norm_path = os.path.join(log_dir, "vec_normalize.pkl")
            
            # Log normalization settings
            logger.info(f"Applying normalization with config: norm_obs={norm_obs}, "
                      f"norm_reward={norm_reward}, clip_obs={clip_obs}, "
                      f"clip_reward={clip_reward}, gamma={gamma}")
            
            # Create the normalization wrapper
            vec_norm = VecNormalize(
                env,
                training=True,
                norm_obs=norm_obs,
                norm_reward=norm_reward,
                clip_obs=clip_obs,
                clip_reward=clip_reward,
                gamma=gamma,
                norm_obs_keys=None,  # Normalize all observation keys if dict
            )
            
            # Save the normalization parameters if a path was provided
            if norm_path is not None:
                try:
                    vec_norm.save(norm_path)
                    logger.info(f"Saved normalization parameters to {norm_path}")
                except Exception as e:
                    logger.error(f"Failed to save normalization parameters: {e}")
            
            # Wrap with VecCheckNan to catch numerical instabilities
            env = VecCheckNan(vec_norm, raise_exception=True)
            logger.info("Successfully applied VecNormalize and VecCheckNan wrappers")
            
        except Exception as e:
            logger.error(f"Error during normalization setup: {e}")
            logger.warning("Continuing without normalization")
            
            # If normalization fails, still apply VecCheckNan to the original env
            env = VecCheckNan(env, raise_exception=True)
    
    return env, vec_norm


def make_env(instance_id: int, config: Dict[str, Any], cache: DataCacheManager, is_eval: bool = False) -> gym.Env:
    """
    Crée et configure un environnement d'entraînement avec validation et wrappers de compatibilité.

    Args:
        instance_id: Identifiant unique de l'instance
        config: Configuration de l'environnement
        cache: Gestionnaire de cache de données
        is_eval: Si True, crée un environnement d'évaluation

    Returns:
        Environnement d'entraînement configuré avec les wrappers nécessaires
    """
    try:
        # Configuration du logger pour cette instance
        logger = setup_logging(instance_id)
        logger.info(f"Initialisation de l'environnement {instance_id} (évaluation: {is_eval})")
        
        # Récupération de la configuration du worker
        worker_keys = list(config["workers"].keys())
        worker_key = worker_keys[instance_id % len(worker_keys)]
        worker_config = config["workers"][worker_key]
        
        # Ajout du gestionnaire de cache dans la configuration
        worker_config["cache_manager"] = cache
        
        # Création de l'environnement de base
        env = MultiAssetChunkedEnv(
            config=config,
            worker_config=worker_config,
            data_loader_instance=None,
            shared_buffer=None,
            worker_id=instance_id
        )
        
        # 1. Validation initiale de l'environnement avant tout wrapping
        try:
            # Test de l'environnement de base
            obs, _ = env.reset(seed=42 + instance_id)
            if not isinstance(obs, dict) or 'observation' not in obs or 'portfolio_state' not in obs:
                raise ValueError("L'environnement ne retourne pas une observation au format attendu (dict avec 'observation' et 'portfolio_state')")
                
            # Vérification des dimensions
            if not isinstance(obs['observation'], np.ndarray) or obs['observation'].ndim != 3:
                raise ValueError(f"L'observation du marché doit être un tableau numpy 3D, reçu: {type(obs['observation'])} avec forme {getattr(obs['observation'], 'shape', 'N/A')}")
                
            if not isinstance(obs['portfolio_state'], np.ndarray) or obs['portfolio_state'].ndim != 1:
                raise ValueError(f"L'état du portefeuille doit être un tableau numpy 1D, reçu: {type(obs['portfolio_state'])} avec forme {getattr(obs['portfolio_state'], 'shape', 'N/A')}")
                
        except Exception as e:
            logger.error("Échec de la validation initiale de l'environnement:")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            raise
            
        # 2. Application du wrapper de suivi des performances (avant la conversion Gymnasium->Gym)
        env = Monitor(env)
        
        # 3. Application du wrapper de compatibilité Gymnasium vers Gym
        env = GymnasiumToGymWrapper(env)
        
        # 4. Configuration du seed pour la reproductibilité
        env.reset(seed=42 + instance_id)
        
        logger.info(f"Environnement {instance_id} initialisé avec succès")
        return env
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de l'environnement {instance_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def setup_logging(instance_id: int) -> logging.Logger:
    """Configure le logging pour l'entraînement parallèle.

    Returns:
        logging.Logger: L'objet logger configuré
    """
    log_dir = os.path.join(PROJECT_ROOT, "logs", f"instance_{instance_id}")
    os.makedirs(log_dir, exist_ok=True)

    # Configurer le logger pour cette instance
    instance_logger = logging.getLogger(f"instance_{instance_id}")
    instance_logger.setLevel(logging.INFO)

    # Créer un gestionnaire de fichier
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Créer un formateur et l'ajouter au gestionnaire
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Ajouter le gestionnaire au logger
    instance_logger.addHandler(file_handler)

    return instance_logger




def load_base_config(config_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Charge la configuration de base avec résolution simple des chemins.
    
    Args:
        config_input: Soit un chemin vers un fichier YAML, soit un dictionnaire de configuration
        
    Returns:
        Dictionnaire de configuration avec les chemins résolus
    """
    try:
        # Si l'entrée est un dictionnaire, l'utiliser directement
        if isinstance(config_input, dict):
            config = config_input.copy()
        # Sinon, essayer de charger depuis un fichier
        else:
            with open(config_input, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        
        # Définir le répertoire de base de manière fixe
        base_dir = "/home/morningstar/Documents/trading/bot"
        
        # Correction des chemins dans la section paths
        if 'paths' not in config:
            config['paths'] = {}
            
        config['paths'].update({
            'base_dir': base_dir,
            'data_dir': f"{base_dir}/data",
            'raw_data_dir': f"{base_dir}/data/raw",
            'processed_data_dir': f"{base_dir}/data/processed",
            'indicators_data_dir': f"{base_dir}/data/processed/indicators",
            'final_data_dir': f"{base_dir}/data/final",
            'models_dir': f"{base_dir}/models",
            'trained_models_dir': f"{base_dir}/models/rl_agents",
            'logs_dir': f"{base_dir}/logs",
            'reports_dir': f"{base_dir}/reports",
            'figures_dir': f"{base_dir}/reports/figures",
            'metrics_dir': f"{base_dir}/reports/metrics"
        })
        
        # Correction des autres chemins dans la config
        if 'model' in config and 'diagnostics' in config['model']:
            config['model']['diagnostics']['attention_map_dir'] = f"{base_dir}/models/attention_maps"
        
        if 'data' in config:
            config['data']['data_dir'] = f"{base_dir}/data/processed/indicators"
        
        if 'training' in config and 'checkpointing' in config['training']:
            config['training']['checkpointing']['save_path'] = f"{base_dir}/models/rl_agents/adan_model"
        
        if 'reward_shaping' in config and 'tier_rewards' in config['reward_shaping']:
            config['reward_shaping']['tier_rewards']['checkpoint_dir'] = f"{base_dir}/models/rl_agents/checkpoints"
        
        # Créer les répertoires nécessaires
        for path in config['paths'].values():
            if isinstance(path, str):
                os.makedirs(path, exist_ok=True)
        
        # Créer aussi les autres répertoires nécessaires
        os.makedirs(f"{base_dir}/models/attention_maps", exist_ok=True)
        os.makedirs(f"{base_dir}/models/rl_agents/checkpoints", exist_ok=True)
        
        logger.info("Configuration chargée avec chemins corrigés")
        return config
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        raise

        raise
    except TypeError as e:
        logger.error(f"Type de configuration invalide: {str(e)}")
        raise
    except Exception as e:
        logger.critical(f"Erreur critique dans load_base_config: {str(e)}", exc_info=True)
        raise


class TimeoutException(Exception):
    """Exception levée quand le timeout est atteint"""

    pass


def timeout_handler(signum, frame):
    """Gestionnaire de signal pour le timeout"""
    raise TimeoutException("Temps d'entraînement écoulé")


def save_checkpoint(model, optimizer, epoch: int, path: str):
    """
    Sauvegarde un checkpoint du modèle et de l'optimiseur.
    
    Args:
        model: Modèle à sauvegarder (PPO de Stable Baselines 3)
        optimizer: Optimiseur à sauvegarder
        epoch: Numéro d'epoch actuel
        path: Chemin de base pour la sauvegarde (sans extension)
    """
    try:
        # Créer le répertoire s'il n'existe pas
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Pour les modèles PPO de Stable Baselines 3, utiliser la méthode save intégrée
        if hasattr(model, 'save'):
            # Sauvegarder le modèle complet
            model.save(path)
            
            # Si un optimiseur est fourni, le sauvegarder séparément
            if optimizer is not None:
                checkpoint = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()
                }
                th.save(checkpoint, f"{path}_optimizer.pt")
        else:
            # Pour les modèles PyTorch standard
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None
            }
            th.save(checkpoint, f"{path}.pt")
        
        logging.info(f"Checkpoint sauvegardé dans {path} (epoch {epoch})")
        
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du checkpoint {path}: {e}")
        raise


def load_checkpoint(model, optimizer, path: str):
    """
    Charge un checkpoint de modèle et d'optimiseur.
    
    Args:
        model: Modèle à mettre à jour (PPO de Stable Baselines 3)
        optimizer: Optimiseur à mettre à jour
        path: Chemin de base du checkpoint (sans extension)
        
    Returns:
        L'epoch à partir de laquelle reprendre l'entraînement
    """
    try:
        # Vérifier que le fichier du modèle existe
        if not os.path.exists(f"{path}.zip"):
            raise FileNotFoundError(f"Checkpoint non trouvé: {path}.zip")
        
        epoch = 0
        
        # Pour les modèles PPO de Stable Baselines 3, utiliser la méthode load intégrée
        if hasattr(model, 'load'):
            model = model.load(path, device=model.device)
            
            # Charger l'optimiseur séparément s'il existe
            optimizer_path = f"{path}_optimizer.pt"
            if optimizer is not None and os.path.exists(optimizer_path):
                checkpoint = th.load(optimizer_path, map_location=model.device)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint.get('epoch', 0)
        else:
            # Pour les modèles PyTorch standard
            checkpoint = th.load(f"{path}.pt", map_location=model.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            epoch = checkpoint.get('epoch', 0)
        
        logging.info(f"Checkpoint chargé depuis {path} (epoch {epoch})")
        return epoch
        
    except Exception as e:
        logging.error(f"Erreur lors du chargement du checkpoint {path}: {e}")
        return 0  # Reprendre au début en cas d'erreur


def emergency_save(model, env, save_path: str):
    """Sauvegarde d'urgence du modèle et de l'environnement.

    Args:
        model: Modèle à sauvegarder
        env: Environnement à sauvegarder
        save_path: Chemin de sauvegarde
    """
    try:
        # Créer le répertoire de sauvegarde s'il n'existe pas
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Sauvegarder le modèle
        model.save(f"{save_path}_emergency")

        # Sauvegarder l'environnement s'il a une méthode de sauvegarde
        if hasattr(env, 'save'):
            env.save(f"{save_path}_vecnormalize_emergency.pkl")

        logging.info(f"Sauvegarde d'urgence effectuée dans {save_path}_emergency")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde d'urgence: {str(e)}")







def train_single_instance(
    instance_id: int,
    total_timesteps: int,
    config_override: Optional[Union[Dict[str, Any], str]] = None,
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
        config_override: Configuration de remplacement optionnelle (chemin ou dict)
        shared_model_path: Chemin vers un modèle partagé pour le fine-tuning
        checkpoint_path: Chemin pour sauvegarder les checkpoints
        timeout: Délai maximal d'exécution en secondes

    Returns:
        Dict contenant les résultats de l'entraînement
    """
    # Initialisation du timer
    start_time = time.time()

    # Configuration du logging pour cette instance
    setup_logging(instance_id)

    try:
        # Gestion du timeout
        if timeout:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        # Charger la configuration
        if config_override is None:
            config_path = os.path.join(
                os.getenv('TRADING_BOT_DIR', '.'), 
                'config', 
                'config.yaml'
            )
            config = load_base_config(config_path)
        elif isinstance(config_override, str):
            config = load_base_config(config_override)
        elif isinstance(config_override, dict):
            config = load_base_config()  # Charge la config par défaut
            # Fusionne avec les overrides
            for key, value in config_override.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value

        # Initialisation du cache de données si nécessaire
        if 'data_cache' not in locals():
            data_cache = {}

        # Création de l'environnement
        env = make_env(instance_id, config, data_cache)

        # Vérification de la compatibilité Gymnasium
        try:
            check_env(env)
        except Exception as e:
            logger.warning(
                f"Avertissement de compatibilité Gymnasium: {e}"
            )

        # Enveloppement dans un wrapper de compatibilité SB3 si nécessaire
        if not hasattr(env, 'step'):
            env = SB3GymCompatibilityWrapper(env)

        # Ajout du wrapper Monitor pour le suivi
        env = Monitor(env, f"instance_{instance_id}")

        # Création ou chargement du modèle
        if shared_model_path and os.path.exists(shared_model_path):
            logger.info(f"Chargement du modèle partagé depuis {shared_model_path}")
            model = load_checkpoint(shared_model_path, env=env)
        else:
            # Création d'un nouveau modèle
            policy_kwargs = dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64])
            )

            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=os.path.join(
                    config['training']['log_dir'], 
                    f"instance_{instance_id}"
                )
            )

        # Configuration des callbacks
        callbacks = []

        # Callback pour le suivi personnalisé
        callbacks.append(CustomTrainingInfoCallback(
            verbose=1,
            instance_id=instance_id,
            log_interval=config['training'].get('log_interval', 100)
        ))
        
        # Callback pour les sauvegardes périodiques
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
            callbacks.append(CheckpointCallback(
                save_freq=config['training'].get('save_freq', 10000),
                save_path=checkpoint_path,
                name_prefix=f"instance_{instance_id}",
                verbose=1
            ))
            
        # Entraînement du modèle
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Sauvegarde finale du modèle
        final_model_path = None
        if checkpoint_path:
            final_model_path = os.path.join(
                checkpoint_path, 
                f"instance_{instance_id}_final"
            )
            save_checkpoint(model, env, final_model_path)
            logger.info(f"Modèle final sauvegardé dans {final_model_path}")
            
        # Retourner les résultats
        return {
            "instance_id": instance_id,
            "status": "completed",
            "model_path": final_model_path,
            "timesteps_completed": total_timesteps,
            "training_time": time.time() - start_time
        }
            
    except TimeoutException as e:
        logger.error(f"Temps d'exécution dépassé pour l'instance {instance_id}")
        return {
            "instance_id": instance_id,
            "status": "timeout",
            "error": str(e),
            "timesteps_completed": (
                model.num_timesteps if (
                    'model' in locals() and 
                    hasattr(model, 'num_timesteps')
                ) else 0
            ),
            "training_time": time.time() - start_time
        }
            
    except Exception as e:
        logger.error(f"Erreur critique dans l'instance {instance_id}: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Sauvegarde d'urgence si possible
        if 'model' in locals() and 'env' in locals() and checkpoint_path:
            try:
                emergency_save(
                    model, 
                    env, 
                    os.path.join(
                        checkpoint_path, 
                        f"crash_save_instance_{instance_id}"
                    )
                )
            except Exception as save_error:
                logger.error(f"Échec de la sauvegarde d'urgence: {str(save_error)}")
        
        return {
            "instance_id": instance_id,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timesteps_completed": (
                model.num_timesteps if (
                    'model' in locals() and 
                    hasattr(model, 'num_timesteps')
                ) else 0
            ),
            "training_time": time.time() - start_time
        }
        
    finally:
        # Nettoyage
        try:
            if 'env' in locals():
                env.close()
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de l'environnement: {str(e)}")

def train_single_instance(
    instance_id: int,
    total_timesteps: int,
    config_override: Optional[Union[Dict[str, Any], str]] = None,
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
    instance_logger = setup_logging(instance_id)
    start_time = time.time()
    last_checkpoint_time = start_time
    checkpoint_interval = 300  # 5 minutes en secondes

    # Variables pour la gestion des interruptions
    model = None
    env = None
    eval_env = None

    def signal_handler(signum, frame):
        nonlocal model, env, instance_id
        msg = f"Signal {signum} reçu, sauvegarde d'urgence..."
        instance_logger.warning(msg)

        # Créer le répertoire de sauvegarde s'il n'existe pas
        save_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        # Définir le chemin de sauvegarde avec un horodatage
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(
            save_dir,
            f"emergency_save_instance_{instance_id}_{timestamp}"
        )

        try:
            if model is not None and env is not None:
                emergency_save(model, env, save_path)
                instance_logger.info(f"Sauvegarde d'urgence effectuée dans {save_path}")
            else:
                instance_logger.warning("Impossible de sauvegarder: modèle ou environnement non initialisé")
        except Exception as e:
            instance_logger.error(f"Erreur lors de la sauvegarde d'urgence: {str(e)}")

        sys.exit(0)

    # Enregistrer le gestionnaire de signal pour SIGINT (Ctrl+C)
    import signal
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialiser les variables pour la gestion des interruptions
        model = None
        env = None
        eval_env = None

        # Définir le chemin de sauvegarde d'urgence
        emergency_save_path = os.path.join(
            os.path.dirname(checkpoint_path) if checkpoint_path else
            os.path.join(PROJECT_ROOT, "models"),
            f"emergency_save_instance_{instance_id}"
        )

        # Charger la configuration
        config = load_base_config(config_override)

        # Initialiser le cache des données
        cache_dir = os.path.join(PROJECT_ROOT, "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache = DataCacheManager(cache_dir)

        instance_logger.info(f"🚀 Démarrage de l'entraînement pour l'instance {instance_id}")
        instance_logger.info(f"Utilisation du cache dans : {cache_dir}")

        # Configuration spécifique au worker
        worker_keys = list(config["workers"].keys())
        worker_key = worker_keys[instance_id % len(worker_keys)]
        worker_config = config["workers"][worker_key]
        agent_config = config.get("agent", {})

        # Initialiser les environnements à None
        env = None
        eval_env = None

        # Définir le répertoire de logs
        log_dir = os.path.join(
            PROJECT_ROOT,
            "logs",
            f"instance_{instance_id}",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "tensorboard"), exist_ok=True)

        # Ajouter la clé du worker à la configuration pour référence
        worker_config["worker_key"] = worker_key

        # 3. Fusionner les configurations (si nécessaire)
        # Ici, nous pourrions ajouter une logique pour fusionner des configurations
        # spécifiques du worker avec la configuration de base

        # 4. Journalisation des informations de configuration
        instance_logger.info(f"Instance {instance_id} - {worker_config.get('name', 'Sans nom')}")
        instance_logger.info(f"  - Actifs: {', '.join(worker_config.get('assets', []))}")
        instance_logger.info(f"  - Timeframes: {', '.join(worker_config.get('timeframes', []))}")
        instance_logger.info(f"  - Jeu de données: {worker_config.get('data_split', 'train')}")

        # 5. Charger la configuration de base
        base_config = load_base_config(config_override)

        # 6. S'assurer que la configuration de l'environnement est correctement structurée
        if "environment" not in base_config:
            base_config["environment"] = {}

        # 7. Créer l'environnement avec la configuration fusionnée
        from adan_trading_bot.environment.compat import SB3GymCompatibilityWrapper
        # S'assurer que worker_config contient les champs requis
        if not isinstance(worker_config, dict):
            worker_config = {}
        
        # Définir les valeurs par défaut si non spécifiées
        worker_config.setdefault('assets', base_config.get('assets', ['BTCUSDT']))
        worker_config.setdefault('timeframes', base_config.get('timeframes', ['1h']))
        worker_config.setdefault('window_size', base_config.get('window_size', 100))
        
        # Créer l'environnement de base
        env = MultiAssetChunkedEnv(config=base_config, worker_config=worker_config)
        
        # Afficher les informations sur les espaces d'observation et d'action
        instance_logger.info(f"Espace d'observation initial: {env.observation_space}")
        instance_logger.info(f"Espace d'action initial: {env.action_space}")
        
        # Configurer la politique en fonction du type d'espace d'observation
        if isinstance(env.observation_space, gym.spaces.Dict):
            policy = "MultiInputPolicy"
            instance_logger.info(f"Utilisation de {policy} pour l'espace d'observation de type Dict")
            
            # Vérifier que les clés attendues sont présentes
            if not ('observation' in env.observation_space.spaces and 'portfolio_state' in env.observation_space.spaces):
                raise ValueError("L'espace d'observation Dict doit contenir les clés 'observation' et 'portfolio_state'")
                
            # Afficher les détails de l'espace d'observation
            obs_space = env.observation_space.spaces['observation']
            portfolio_space = env.observation_space.spaces['portfolio_state']
            instance_logger.info(f"  - Observation shape: {obs_space.shape}, type: {type(obs_space).__name__}")
            instance_logger.info(f"  - Portfolio state shape: {portfolio_space.shape}, type: {type(portfolio_space).__name__}")
        else:
            policy = "MlpPolicy"
            instance_logger.info(f"Utilisation de {policy} pour l'espace d'observation standard")
        
        # Envelopper l'environnement avec le wrapper de compatibilité
        env = SB3GymCompatibilityWrapper(env)
        instance_logger.info(f"Espace d'observation après wrapper: {env.observation_space}")
        instance_logger.info(f"Espace d'action après wrapper: {env.action_space}")
        
        # Configuration du modèle avec gestion des espaces d'observation de type Dict
        policy_kwargs = {
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64]
            },
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True
        }

        # Configuration de base du modèle
        model_config = {
            "learning_rate": config.get("learning_rate", 3e-4),
            "n_steps": config.get("n_steps", 2048),
            "batch_size": config.get("batch_size", 64),
            "n_epochs": config.get("n_epochs", 10),
            "gamma": config.get("gamma", 0.99),
            "gae_lambda": config.get("gae_lambda", 0.95),
            "clip_range": config.get("clip_range", 0.2),
            "clip_range_vf": config.get("clip_range_vf", None),
            "normalize_advantage": config.get("normalize_advantage", True),
            "ent_coef": config.get("ent_coef", 0.0),
            "vf_coef": config.get("vf_coef", 0.5),
            "max_grad_norm": config.get("max_grad_norm", 0.5),
            "use_sde": config.get("use_sde", False),
            "sde_sample_freq": config.get("sde_sample_freq", -1),
            "target_kl": config.get("target_kl", None),
            "tensorboard_log": os.path.join(log_dir, "tensorboard"),
            "verbose": 1,
            "seed": config.get("seed", 42) + instance_id,
            "device": config.get("device", "auto")
        }

        # Créer un environnement vectorisé avec gestion des espaces d'observation de type Dict
        def make_env():
            # Créer une nouvelle instance de l'environnement
            env = MultiAssetChunkedEnv(config=base_config, worker_config=worker_config)
            
            # Appliquer le wrapper de compatibilité
            env = SB3GymCompatibilityWrapper(env)
            
            # Appliquer le wrapper Monitor pour le suivi des épisodes
            env = Monitor(env)
            
            return env
        
        # Créer l'environnement vectorisé avec un seul worker
        vec_env = DummyVecEnv([make_env])
        
        # Configurer la politique en fonction du type d'espace d'observation
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Utiliser une politique adaptée aux espaces d'observation de type Dict
            policy = "MultiInputPolicy"
            
            # Configurer l'extracteur de caractéristiques personnalisé
            policy_kwargs.update({
                "features_extractor_class": CustomCombinedExtractor,
                "features_extractor_kwargs": {
                    "cnn_output_dim": 64,
                    "mlp_extractor_net_arch": [64, 64],
                    "observation_space": env.observation_space
                },
                "net_arch": {
                    "pi": [64, 64],
                    "vf": [64, 64]
                },
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True
            })
            
            # Pour les espaces d'observation complexes, utiliser un wrapper supplémentaire
            # si nécessaire pour la transposition des images
            if len(env.observation_space.spaces['observation'].shape) >= 2:
                vec_env = VecTransposeImage(vec_env)
                
            # Afficher les informations sur l'environnement vectorisé
            instance_logger.info(f"Environnement vectorisé créé avec succès")
            instance_logger.info(f"  - Espace d'observation: {vec_env.observation_space}")
            instance_logger.info(f"  - Espace d'action: {vec_env.action_space}")
            
        else:
            # Pour les espaces d'observation simples, utiliser une politique standard
            policy = "MlpPolicy"
            policy_kwargs.update({
                "net_arch": {
                    "pi": [64, 64],
                    "vf": [64, 64]
                },
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True
            })
            
            # Afficher les informations sur l'environnement vectorisé
            instance_logger.info(f"Environnement vectorisé créé avec succès (espace d'observation simple)")
            instance_logger.info(f"  - Espace d'observation: {vec_env.observation_space}")
            instance_logger.info(f"  - Espace d'action: {vec_env.action_space}")
        
        # Mettre à jour la configuration du modèle avec les paramètres de la politique
        model_config.update({
            "policy": policy,
            "env": vec_env,
            "policy_kwargs": policy_kwargs,
            "verbose": 1,
            "tensorboard_log": os.path.join(log_dir, "tensorboard"),
            "seed": config.get("seed", 42) + instance_id,
            "device": config.get("device", "auto")
        })
        
        # Afficher la configuration du modèle pour le débogage
        instance_logger.info("Configuration du modèle:")
        for key, value in model_config.items():
            if key != 'policy_kwargs':
                instance_logger.info(f"  - {key}: {value}")
        
        instance_logger.info("Configuration de la politique (policy_kwargs):")
        for key, value in policy_kwargs.items():
            instance_logger.info(f"  - {key}: {value}")
        
        # Création du modèle avec gestion des erreurs
        try:
            model = PPO(**model_config)
        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle: {e}")
            raise

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
            instance_logger.info(
                f"Reprise de l'entraînement à partir de l'epoch {start_epoch+1}/{epochs}"
            )

        # Boucle d'entraînement
        try:
            # Réinitialisation initiale
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Extraire l'observation du tuple (obs, info)

            for epoch in range(start_epoch, epochs):
                current_time = time.time()
                time_elapsed = current_time - start_time

                # Vérifier le timeout
                if timeout and time_elapsed >= timeout:
                    instance_logger.info(f"Timeout atteint après {time_elapsed:.1f} secondes")
                    break

                instance_logger.info(f"Début de l'epoch {epoch+1}/{epochs}")

                # Entraînement sur une epoch
                model.learn(
                    total_timesteps=n_steps,
                    tb_log_name=f"instance_{instance_id}_{worker_config['name'].lower()}",
                    progress_bar=True,
                    reset_num_timesteps=False,
                )

                # Sauvegarder le checkpoint périodiquement
                current_time = time.time()
                if checkpoint_path and (current_time - last_checkpoint_time) >= checkpoint_interval:
                    save_checkpoint(model, model.policy.optimizer, epoch, checkpoint_path)
                    logger.info("Checkpoint sauvegardé à %s", checkpoint_path)
                    last_checkpoint_time = current_time

                # Réinitialisation pour la prochaine époque
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

            # Sauvegarder le modèle final
            if checkpoint_path:
                save_checkpoint(model, model.policy.optimizer, epoch, checkpoint_path)
                logger.info("Checkpoint final sauvegardé à %s", checkpoint_path)

        except Exception as e:
            logger.error("Erreur pendant l'entraînement: %s", str(e))
            # Tenter de sauvegarder en cas d'erreur
            if checkpoint_path:
                emergency_path = f"{checkpoint_path}_emergency"
                save_checkpoint(model, model.policy.optimizer, epoch, emergency_path)
                logger.info("Sauvegarde d'urgence effectuée à %s", emergency_path)
            raise

        training_time = time.time() - start_time
        worker_name = worker_config["name"].lower().replace(" ", "_")
        instance_model_path = "models/instance_{}_{}_final.zip".format(
            instance_id, worker_name
        )
        model.save(instance_model_path)

        # Évaluation du modèle
        obs = env.reset()  # On récupère uniquement l'observation
        total_reward = 0
        num_episodes = 0

        for _ in range(100):
            # Prédiction sécurisée avec gestion des erreurs
            try:
                action, _ = safe_predict(model, obs, deterministic=True)
                # Exécution du pas d'environnement avec gestion du retour (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Mise à jour de la récompense totale
                if isinstance(reward, (list, np.ndarray)) and len(reward) > 0:
                    total_reward += float(reward[0])
                else:
                    total_reward += float(reward)

                # Log metrics
                instance_logger.info(f"Trades: {info.get('total_trades', 0)}")
                instance_logger.info(f"Win/Loss Ratio: {info.get('win_loss_ratio', 0)}")

                # Mise à jour de l'observation pour la prochaine itération
                obs = next_obs

                # Réinitialisation si l'épisode est terminé
                if done:
                    obs = env.reset()
                    num_episodes += 1
                    if num_episodes >= 10:  # Limite le nombre d'épisodes
                        break

            except Exception as e:
                logger.error("Erreur lors de l'évaluation: %s", str(e))
                logger.debug("Détails de l'erreur:", exc_info=True)
                break

        # Calcul de la récompense moyenne
        avg_reward = total_reward / 100 if num_episodes > 0 else 0

        env.close()

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
            "✅ Instance %d completed - Avg Reward: %.4f, Time: %.1fs",
            instance_id, avg_reward, training_time
        )
        return results

    except KeyboardInterrupt:
        logger.warning(f"Interruption utilisateur détectée pour l'instance {instance_id}")
        if model is not None and env is not None:
            logger.info(f"Sauvegarde d'urgence de l'instance {instance_id}...")
            emergency_save(model, env, emergency_save_path)
        return {"status": "interrupted", "instance_id": instance_id}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement de l'instance {instance_id}: {str(e)}")
        logger.error(traceback.format_exc())
        # Tenter une sauvegarde d'urgence en cas d'erreur non gérée
        if model is not None and env is not None:
            logger.info(f"Tentative de sauvegarde d'urgence après erreur pour l'instance {instance_id}...")
            try:
                emergency_save(model, env, f"{emergency_save_path}_error")
            except Exception as save_error:
                logger.error(f"Échec de la sauvegarde d'urgence: {str(save_error)}")
        return {"status": "error", "error": str(e), "instance_id": instance_id}


def main(
    config_path: str = "config/config.yaml",
    timeout: int = None,
    checkpoint_dir: str = "checkpoints",
    shared_model_path: str = None,
):
    """Fonction principale d'entraînement parallèle"""
    # Initialisation du logger principal avec instance_id=0
    logger = setup_logging(instance_id=0)
    logger.info("🚀 Starting ADAN Parallel Training")

    # Charger la configuration
    config = load_base_config(config_path)  # Load base config with provided path

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
                config_override=config,  # Passer la configuration déjà chargée
                shared_model_path=shared_model_path,
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


class SB3GymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper pour assurer la compatibilité entre Gymnasium et Stable Baselines 3.
    Gère spécifiquement les espaces d'observation de type Dict pour les environnements de trading.
    """
    def __init__(self, env):
        # Appeler le constructeur parent avec l'environnement Gymnasium
        super().__init__(env)
        
        # Enregistrer l'espace d'observation original
        self.original_obs_space = env.observation_space
        
        # S'assurer que l'espace d'action est correctement défini
        if not isinstance(env.action_space, (spaces.Discrete, spaces.Box, 
                                         spaces.MultiDiscrete, spaces.MultiBinary)):
            raise ValueError(
                f"Type d'espace d'action non supporté: {type(env.action_space)}"
            )
        # Pour les environnements de trading avec espace d'observation de type Dict
        if isinstance(env.observation_space, spaces.Dict):
            # Vérifier si nous avons les clés attendues
            if 'observation' in env.observation_space.spaces and 'portfolio_state' in env.observation_space.spaces:
                # Enregistrer l'espace d'observation original
                self.observation_space = env.observation_space
                
                # Extraire les espaces pour le traitement
                obs_space = env.observation_space.spaces['observation']
                portfolio_space = env.observation_space.spaces['portfolio_state']
                
                # Vérifier les dimensions
                if not (isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3 and
                       isinstance(portfolio_space, spaces.Box) and len(portfolio_space.shape) == 1):
                    raise ValueError("Format d'observation non supporté. Attendu: observation (Box 3D) et portfolio_state (Box 1D)")
                
                # Enregistrer les dimensions pour le traitement des observations
                self.obs_shape = obs_space.shape
                self.portfolio_dim = portfolio_space.shape[0]
                
                logger.info(f"SB3GymCompatibilityWrapper: Espace d'observation configuré avec succès. "
                          f"Forme de l'observation: {obs_space.shape}, "
                          f"Dimension du portefeuille: {portfolio_space.shape}")
            else:
                raise ValueError("L'espace d'observation Dict doit contenir les clés 'observation' et 'portfolio_state'")
        else:
            # Pour les autres types d'espaces, utiliser l'espace d'observation tel quel
            self.observation_space = env.observation_space
    
    def reset(self, **kwargs):
        # Appeler reset sur l'environnement sous-jacent
        obs, info = self.env.reset(**kwargs)
        
        # Traiter l'observation pour la compatibilité avec SB3
        processed_obs = self._process_obs(obs)
        
        # Journalisation pour le débogage
        if hasattr(self, 'obs_shape') and hasattr(self, 'portfolio_dim'):
            logger.debug(
                f"Reset - Type d'observation: {type(obs)}, "
                f"Type traité: {type(processed_obs)}, "
                f"Forme: {getattr(processed_obs, 'shape', 'N/A')}"
            )
        
        return processed_obs, info
    
    def step(self, action):
        # Appeler step sur l'environnement sous-jacent
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Traiter l'observation pour la compatibilité avec SB3
        processed_obs = self._process_obs(obs)
        
        # Journalisation pour le débogage
        if hasattr(self, 'obs_shape') and hasattr(self, 'portfolio_dim'):
            logger.debug(
                f"Step - Type d'observation: {type(obs)}, "
                f"Type traité: {type(processed_obs)}, "
                f"Forme: {getattr(processed_obs, 'shape', 'N/A')}, "
                f"Récompense: {reward:.4f}, "
                f"Terminé: {terminated}, Tronqué: {truncated}"
            )
        
        return processed_obs, reward, terminated, truncated, info
    
    def _process_obs(self, obs):
        """
        Traite l'observation pour la compatibilité avec SB3.
        Convertit un dictionnaire d'observations en un format compatible avec SB3.
        """
        try:
            # Si l'observation est déjà un tableau numpy, la retourner telle quelle
            if isinstance(obs, np.ndarray):
                return obs
                
            # Si c'est un dictionnaire avec les clés attendues
            if isinstance(obs, dict) and 'observation' in obs and 'portfolio_state' in obs:
                # Créer une copie du dictionnaire pour éviter de modifier l'original
                processed_obs = {}
                
                # Traiter l'observation de marché (3D)
                if isinstance(obs['observation'], np.ndarray):
                    obs_tensor = obs['observation'].astype(np.float32)
                else:
                    obs_tensor = np.array(obs['observation'], dtype=np.float32)
                
                # Traiter l'état du portefeuille (1D)
                if isinstance(obs['portfolio_state'], np.ndarray):
                    portfolio_tensor = obs['portfolio_state'].astype(np.float32)
                else:
                    portfolio_tensor = np.array(obs['portfolio_state'], dtype=np.float32)
                
                # Vérifier et corriger les valeurs NaN ou infinies
                if np.any(np.isnan(obs_tensor)) or np.any(np.isinf(obs_tensor)):
                    logger.warning("L'observation de marché contient des valeurs NaN ou infinies. Remplacement par des zéros.")
                    obs_tensor = np.nan_to_num(obs_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                
                if np.any(np.isnan(portfolio_tensor)) or np.any(np.isinf(portfolio_tensor)):
                    logger.warning("L'état du portefeuille contient des valeurs NaN ou infinies. Remplacement par des zéros.")
                    portfolio_tensor = np.nan_to_num(portfolio_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Créer un nouveau dictionnaire avec les tableaux traités
                processed_obs = {
                    'observation': obs_tensor,
                    'portfolio_state': portfolio_tensor
                }
                
                return processed_obs
            
            # Si l'observation n'est pas un dictionnaire, essayer de la convertir en tableau numpy
            if not isinstance(obs, np.ndarray):
                try:
                    obs = np.array(obs, dtype=np.float32)
                    return obs
                except (TypeError, ValueError) as e:
                    logger.error(f"Impossible de convertir l'observation en tableau numpy: {e}")
                    raise ValueError(f"Format d'observation non supporté: {type(obs)}")
            
            return obs
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'observation: {e}")
            logger.error(f"Type d'observation: {type(obs)}")
            if hasattr(obs, 'shape'):
                logger.error(f"Forme de l'observation: {obs.shape}")
            if isinstance(obs, dict):
                logger.error(f"Clés de l'observation: {obs.keys()}")
                for k, v in obs.items():
                    logger.error(f"  {k}: type={type(v).__name__}, shape={getattr(v, 'shape', 'N/A')}")
            raise


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
        "--shared-model",
        type=str,
        default=None,
        help="Path to shared model for distributed training",
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
        shared_model_path=args.shared_model,
    )
    sys.exit(0 if success else 1)