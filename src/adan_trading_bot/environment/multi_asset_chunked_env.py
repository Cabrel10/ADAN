#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environnement de trading multi-actifs avec chargement par morceaux.

Ce module implémente un environnement de trading pour plusieurs actifs
avec chargement efficace des données par lots.
"""
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.data_processing.state_builder import StateBuilder, TimeframeConfig
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

# Importations locales
from .dynamic_behavior_engine import DynamicBehaviorEngine
from .order_manager import OrderManager
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager as Portfolio
from .reward_calculator import RewardCalculator
from .reward_shaper import RewardShaper


# Configuration du logger
logger = logging.getLogger(__name__)


class MultiAssetChunkedEnv(gym.Env):
    """Environnement de trading multi-actifs avec chargement par morceaux.

    Cet environnement gère plusieurs actifs et intervalles de temps, avec
    support pour des espaces d'actions discrets et continus. Il utilise un
    constructeur d'état pour créer l'espace d'observation et un gestionnaire
    de portefeuille pour suivre les positions et le PnL.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Constantes pour les actions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        config: Dict[str, Any],
        worker_config: Dict[str, Any],
        data_loader_instance: Optional[Any] = None,
        shared_buffer: Optional[Any] = None,
        worker_id: int = 0,
    ) -> None:
        """Initialise l'environnement de trading multi-actifs.

        Args:
            config: Configuration principale de l'application (déjà résolue).
            worker_config: Configuration spécifique au worker (déjà résolue).
            data_loader_instance: Instance de ChunkedDataLoader (optionnel).
            shared_buffer: Instance du SharedExperienceBuffer (optionnel).
        """
        super().__init__()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.worker_config = worker_config
        self.data_loader_instance = data_loader_instance
        self.shared_buffer = shared_buffer

        # Initialize self.assets from worker_config
        self.assets = worker_config.get("assets", [])
        if not self.assets:
            raise ValueError("No assets specified in worker_config.")

        # Configuration du cache d'observations
        self._observation_cache = {}  # Cache des observations
        self._max_cache_size = 1000  # Taille maximale du cache
        self._cache_hits = 0  # Succès du cache
        self._cache_misses = 0  # Échecs du cache
        self._current_observation = None  # Observation courante
        self._cache_access = {}  # Suivi de l'utilisation

        # Initialisation des compteurs et états
        self.current_chunk = 0
        self.current_chunk_idx = 0
        self.done = False  # État done
        self.global_step = 0  # Compteur global d'étapes
        self.current_step = 0  # Étape courante dans l'épisode

        # Suivi des paliers
        self.current_tier = None
        self.previous_tier = None
        self.episode_count = 0
        self.episodes_in_tier = 0
        self.best_portfolio_value = 0.0
        self.last_tier_change_step = 0
        self.tier_history = []  # Historique des paliers

        # Suivi des trades
        self.last_trade_step = -1  # Dernière étape où un trade a été effectué (-1 = aucun trade)

        # Initialisation des composants critiques
        self._is_initialized = False  # Standardisation sur _is_initialized
        try:
            self._initialize_components()
            self._is_initialized = True
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation: %s", str(e))
            raise

    def _initialize_components(self) -> None:
        """Initialize all environment components in the correct order."""
        # 1. Initialize data loader FIRST to know the data structure
        if self.data_loader_instance is not None:
            self.data_loader = self.data_loader_instance
        else:
            self._init_data_loader(self.assets)

        # 2. Create TimeframeConfig from loaded data
        timeframe_configs = []
        if self.data_loader.features_by_timeframe:
            for tf_name, features in self.data_loader.features_by_timeframe.items():
                config = TimeframeConfig(
                    timeframe=tf_name, features=features, window_size=100
                )
                timeframe_configs.append(config)
        else:
            raise ValueError("No feature configuration in data loader.")

        # 3. Initialize portfolio manager
        portfolio_config = self.config.copy()
        env_config = self.config.get("environment", {})
        portfolio_config["trading_rules"] = self.config.get("trading_rules", {})
        portfolio_config["capital_tiers"] = self.config.get("capital_tiers", [])
        # Utiliser portfolio.initial_balance en priorité, puis environment.initial_balance
        # avec une valeur par défaut de 20.0
        portfolio_balance = self.config.get("portfolio", {}).get(
            "initial_balance", env_config.get("initial_balance", 20.0)
        )
        portfolio_config["initial_capital"] = portfolio_balance

        # Map asset names to full names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
        }
        # Create a mapped assets list for the data loader
        mapped_assets = [asset_mapping.get(asset, asset) for asset in self.assets]

        # Initialize portfolio with mapped asset names
        self.portfolio = PortfolioManager(
            env_config=portfolio_config, assets=mapped_assets
        )
        # Create alias for backward compatibility
        self.portfolio_manager = self.portfolio
        self.assets = mapped_assets  # Update self.assets

        # Convert list of TimeframeConfig objects to dictionary
        timeframe_configs_dict = {
            tf_config.timeframe: tf_config for tf_config in timeframe_configs
        }

        # 4. Initialize StateBuilder with dynamic config
        features_config = {
            tf: config.features for tf, config in timeframe_configs_dict.items()
        }

        # Récupérer les tailles de fenêtres spécifiques à chaque timeframe
        env_obs_cfg = self.config.get("environment", {}).get("observation", {})
        window_sizes = env_obs_cfg.get(
            "window_sizes",
            {"5m": 20, "1h": 10, "4h": 5}  # Valeurs par défaut si non spécifiées
        )
        
        # Utiliser la taille de fenêtre du timeframe 5m comme valeur par défaut
        default_window_size = window_sizes.get("5m", 20)
        
        # Configurer les tailles de fenêtres spécifiques pour chaque timeframe
        timeframe_configs = {}
        for tf in features_config.keys():
            tf_window_size = window_sizes.get(tf, default_window_size)
            timeframe_configs[tf] = TimeframeConfig(
                timeframe=tf,
                features=features_config[tf],
                window_size=tf_window_size,
                normalize=True
            )
            self.logger.info(f"Configuration de la fenêtre pour {tf}: {tf_window_size} périodes")
        
        # Initialiser le StateBuilder avec la configuration des timeframes
        self.state_builder = StateBuilder(
            features_config=features_config,
            window_size=default_window_size,  # Valeur par défaut pour la rétrocompatibilité
            include_portfolio_state=True,
            normalize=True
        )
        
        # Configurer les tailles de fenêtres spécifiques dans le StateBuilder
        for tf, config in timeframe_configs.items():
            self.state_builder.set_timeframe_config(tf, config.window_size, config.features)
            self.logger.info(f"Configuration appliquée pour {tf}: fenêtre={config.window_size}, features={len(config.features)}")

        # 5. Setup action and observation spaces (requires state_builder)
        self._setup_spaces()

        # 6. Initialize max_steps and max_chunks_per_episode from config
        self.max_steps = self.config.get("environment", {}).get("max_steps", 1000)
        self.max_chunks_per_episode = self.config.get("environment", {}).get("max_chunks_per_episode", 10)
        self.logger.info(f"Initialized max_steps to {self.max_steps} and max_chunks_per_episode to {self.max_chunks_per_episode}")
        
        # Log the chunking configuration
        self.logger.info(f"Chunk configuration - Total chunks: {self.total_chunks}, Max chunks per episode: {self.max_chunks_per_episode}")

        # 7. Initialize other components using worker_config where available
        trading_rules = self.config.get("trading_rules", {})
        penalties = self.config.get("environment", {}).get("penalties", {})
        self.order_manager = OrderManager(
            trading_rules=trading_rules, penalties=penalties
        )

        # Get reward config with fallback to main config
        env_section = self.config.get("environment", {})
        reward_cfg = self.worker_config.get(
            "reward_config", env_section.get("reward_config", {})
        )

        # Create env config with reward shaping
        env_config = {"reward_shaping": reward_cfg}
        self.reward_calculator = RewardCalculator(env_config=env_config)

        # Initialize observation validator (will be initialized if needed)
        self.observation_validator = None

        # Initialize DBE with config from worker or main config
        dbe_config = self.worker_config.get("dbe_config", self.config.get("dbe", {}))
        self.dbe = DynamicBehaviorEngine(
            config=dbe_config,
            finance_manager=getattr(self.portfolio, "finance_manager", None),
        )

    def _init_data_loader(self, assets: List[str]) -> Any:
        """Initialize the chunked data loader using worker-specific config.

        Returns:
            Initialized ChunkedDataLoader instance

        Raises:
            ValueError: If configuration is invalid or no assets are available
        """
        if not self.worker_config:
            raise ValueError(
                "worker_config must be provided to initialize the data loader."
            )

        # Ensure paths are resolved
        if not hasattr(self, "config") or not self.config:
            raise ValueError("Configuration not properly initialized")

        # Mapping for asset names to file system names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
        }
        # Create a mapped assets list for the data loader
        mapped_assets = [asset_mapping.get(asset, asset) for asset in assets]

        if not mapped_assets:
            raise ValueError("No assets specified in worker or environment config")

        # Get timeframes from config with fallback to worker config
        global_data_timeframes = self.config.get("data", {}).get("timeframes", [])
        worker_timeframes = self.worker_config.get("timeframes", [])

        # Use worker timeframes if specified, otherwise fallback to global config
        self.timeframes = worker_timeframes or global_data_timeframes

        if not self.timeframes:
            raise ValueError(
                f"No timeframes defined: global={global_data_timeframes}, "
                f"worker={worker_timeframes}"
            )

        # Create a copy of the config with resolved paths
        loader_config = {
            **self.config,
            "data": {
                **self.config.get("data", {}),
                "features_per_timeframe": self.config.get("data", {}).get(
                    "features_per_timeframe", {}
                ),
                "assets": mapped_assets,  # Use mapped_assets here
            },
        }

        # Ensure we have features configured for each timeframe
        features_config = loader_config["data"]["features_per_timeframe"]
        for tf in self.timeframes:
            if tf not in features_config:
                logger.warning(
                    f"No features configured for timeframe {tf}, "
                    "using default features"
                )
                features_config[tf] = ["open", "high", "low", "close", "volume"]

        # Initialize the data loader
        self.data_loader = ChunkedDataLoader(
            config=loader_config,
            worker_config={
                **self.worker_config,
                "assets": mapped_assets,  # Use mapped_assets here
                "timeframes": self.timeframes,
            },
        )

        # Get total chunks from the data loader
        self.total_chunks = self.data_loader.total_chunks
        logger.debug(f"MultiAssetChunkedEnv timeframes: {self.timeframes}")

        logger.info(
            f"Initialized data loader with {len(self.assets)} assets: {', '.join(self.assets)}"
        )
        logger.debug(f"Available timeframes: {', '.join(self.timeframes)}")

        return self.data_loader

    def _setup_spaces(self) -> None:
        """Set up action and observation spaces.

        Raises:
            ValueError: If the observation space cannot be properly configured
        """
        # Action space: Continuous actions in [-1, 1] for each asset
        # -1 = max sell, 0 = hold, 1 = max buy
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.assets),),  # One action per asset
            dtype=np.float32,
        )

        # Configure observation space based on StateBuilder
        try:
            # Get the observation shape from StateBuilder (returns 3 values)
            (
                n_timeframes,
                window_size,
                n_features,
            ) = self.state_builder.get_observation_shape()

            # Get portfolio state dimension from StateBuilder
            portfolio_dim = self.state_builder.get_portfolio_state_dim()

            # Store the shape information
            self.observation_shape = (n_timeframes, window_size, n_features)
            self.portfolio_state_dim = portfolio_dim

            # Log the dimensions for debugging
            logger.info(f"Observation shape: {self.observation_shape}")
            logger.info(f"Portfolio state dimension: {self.portfolio_state_dim}")

            # Create observation space dictionary
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=self.observation_shape,
                        dtype=np.float32,
                    ),
                    "portfolio_state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.portfolio_state_dim,),
                        dtype=np.float32,
                    ),
                }
            )

            logger.info(f"Observation space: {self.observation_space}")

        except Exception as e:
            logger.error(f"Error setting up observation space: {str(e)}")
            raise

    def reset(self, *, seed=None, options=None):
        """Reset the environment to start a new episode.

        Args:
            seed: Optional seed for the random number generator
            options: Additional options for reset

        Returns:
            tuple: (observation, info) containing the initial observation and info
        """
        super().reset(seed=seed)

        # Reset episode-specific variables
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.step_in_chunk = 0

        # Reset portfolio and load initial data chunk
        # Ensure starting from configured initial balance at each new episode
        # and clear last trade tracking
        if hasattr(self, "last_trade_step"):
            self.last_trade_step = -1
        self.portfolio.reset(new_epoch=True)
        self.current_chunk_idx = 0
        self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)

        # Get initial observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _apply_tier_reward(self, reward: float, current_value: float) -> float:
        """Applique les récompenses et pénalités liées aux changements de palier.

        Args:
            reward: Récompense actuelle à modifier
            current_value: Valeur actuelle du portefeuille

        Returns:
            float: Récompense modifiée
        """
        if not hasattr(self, 'current_tier') or self.current_tier is None:
            return reward

        # Mettre à jour le meilleur portefeuille pour ce palier
        if current_value > self.best_portfolio_value:
            self.best_portfolio_value = current_value

        # Vérifier si le palier a changé
        has_changed, is_promotion = self._update_tier(current_value)

        if not has_changed:
            return reward

        # Appliquer les bonus/malus de changement de palier
        tier_rewards = self.config.get('reward_shaping', {}).get('tier_rewards', {})

        if is_promotion:
            promotion_bonus = tier_rewards.get('promotion_bonus', 0.0)
            logger.info(f"Applying promotion bonus: {promotion_bonus}")
            reward += promotion_bonus

            # Sauvegarder le modèle si configuré
            if tier_rewards.get('checkpoint_on_promotion', False):
                self._save_checkpoint_on_promotion()
        else:
            demotion_penalty = tier_rewards.get('demotion_penalty', 0.0)
            logger.info(f"Applying demotion penalty: {demotion_penalty}")
            reward -= demotion_penalty

        # Appliquer le multiplicateur de performance du palier
        performance_multiplier = self.current_tier.get('performance_multiplier', 1.0)
        if performance_multiplier != 1.0:
            reward *= performance_multiplier
            logger.info(f"Applied tier performance multiplier: {performance_multiplier}")

        return reward

    def _save_checkpoint_on_promotion(self) -> None:
        """Sauvegarde un point de contrôle complet lors d'une promotion de palier.

        Cette méthode sauvegarde à la fois le modèle et l'état de l'environnement.
        """
        if not hasattr(self, 'model') or self.model is None:
            logger.warning("Cannot save checkpoint: model not available")
            return

        # Créer le répertoire de checkpoints s'il n'existe pas
        tier_rewards = self.config.get('reward_shaping', {}).get('tier_rewards', {})
        checkpoint_dir = tier_rewards.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Générer un nom de fichier unique avec le timestamp et le palier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tier_name = self.current_tier['name'].lower().replace(' ', '_')
        checkpoint_base = os.path.join(
            checkpoint_dir,
            f"model_{tier_name}_promo_{timestamp}"
        )

        try:
            # 1. Sauvegarder le modèle
            model_path = f"{checkpoint_base}_model"
            self.model.save(model_path)
            logger.info(f"Model checkpoint saved to {model_path}")

            # 2. Sauvegarder l'état de l'environnement
            env_checkpoint = self._save_checkpoint()
            env_checkpoint['model_path'] = model_path

            # 3. Sauvegarder les métadonnées supplémentaires
            metadata = {
                'tier': self.current_tier['name'],
                'timestamp': timestamp,
                'portfolio_value': self.portfolio.get_total_value(),
                'episode': self.episode_count,
                'step': self.current_step,
                'checkpoint_type': 'promotion',
                'tier_info': {
                    'current_tier': self.current_tier['name'],
                    'min_value': self.current_tier['min_value'],
                    'max_value': self.current_tier.get('max_value', float('inf')),
                    'episodes_in_tier': self.episodes_in_tier,
                    'last_tier_change_step': self.last_tier_change_step
                }
            }

            # 4. Fusionner les métadonnées avec le checkpoint
            env_checkpoint['metadata'] = metadata

            # 5. Sauvegarder le checkpoint complet
            checkpoint_path = f"{checkpoint_base}_full.pkl"
            with open(checkpoint_path, 'wb') as f:
                import pickle
                pickle.dump(env_checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Full environment checkpoint saved to {checkpoint_path}")

            # 6. Mettre à jour l'historique des checkpoints
            if not hasattr(self, 'checkpoint_history'):
                self.checkpoint_history = []

            self.checkpoint_history.append({
                'timestamp': timestamp,
                'path': checkpoint_path,
                'tier': self.current_tier['name'],
                'portfolio_value': self.portfolio.get_total_value()
            })

            # 7. Garder uniquement les N derniers checkpoints
            max_checkpoints = tier_rewards.get('max_checkpoints', 5)
            if len(self.checkpoint_history) > max_checkpoints:
                oldest_checkpoint = self.checkpoint_history.pop(0)
                try:
                    os.remove(oldest_checkpoint['path'])
                    logger.info(f"Removed old checkpoint: {oldest_checkpoint['path']}")
                except Exception as e:
                    logger.error(f"Failed to remove old checkpoint: {e}")

        except Exception as e:
            logger.error(f"Failed to save promotion checkpoint: {e}")
            raise

    def _update_tier(self, current_value: float) -> Tuple[bool, bool]:
        """Met à jour le palier actuel en fonction de la valeur du portefeuille.

        Args:
            current_value: Valeur actuelle du portefeuille

        Returns:
            Tuple[bool, bool]: (has_tier_changed, is_promotion) indiquant
                              si le palier a changé et si c'est une promotion
        """
        if not hasattr(self, 'portfolio'):
            return False, False

        current_tier = self.portfolio.get_current_tier()

        # Si c'est la première initialisation
        if self.current_tier is None:
            self.current_tier = current_tier
            self.best_portfolio_value = current_value
            self.tier_history.append({
                'step': self.current_step,
                'tier': current_tier['name'],
                'portfolio_value': current_value,
                'episode': self.episode_count,
                'is_promotion': False
            })
            return False, False

        # Vérifier si le palier a changé
        if current_tier['name'] != self.current_tier['name']:
            self.previous_tier = self.current_tier
            self.current_tier = current_tier
            self.last_tier_change_step = self.current_step
            self.episodes_in_tier = 0

            # Déterminer s'il s'agit d'une promotion
            prev_min = (self.previous_tier.get('min_capital', 0)
                       if self.previous_tier else 0)
            is_promotion = current_tier['min_capital'] > prev_min

            # Mettre à jour l'historique
            self.tier_history.append({
                'step': self.current_step,
                'tier': current_tier['name'],
                'portfolio_value': current_value,
                'episode': self.episode_count,
                'is_promotion': is_promotion
            })

            prev_name = self.previous_tier['name']
            curr_name = current_tier['name']
            logger.info(
                f"Tier changed from {prev_name} to {curr_name} "
                f"(Promotion: {is_promotion}) at step {self.current_step}"
            )

            return True, is_promotion

        return False, False

        # Reset portfolio and order manager
        # Use new_epoch=hard_reset to control whether to reset capital
        self.portfolio.reset(new_epoch=hard_reset)
        self.order_manager.reset()

        # Load initial chunk of data
        self.current_chunk_idx = 0
        self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)

        # Fit scalers on the first chunk of data
        if hasattr(self.state_builder, "fit_scalers"):
            if callable(self.state_builder.fit_scalers):
                logger.info("Fitting scalers on initial data...")
                try:
                    # Prepare data in the format expected by fit_scalers
                    scaler_data = {}
                    for asset, timeframes in self.current_data.items():
                        for tf, df in timeframes.items():
                            if tf not in scaler_data:
                                scaler_data[tf] = []
                            scaler_data[tf].append(df)

                    # Combine data for each timeframe
                    combined_data = {}
                    for tf, dfs in scaler_data.items():
                        if dfs:  # Only concatenate if there are DataFrames
                            combined_data[tf] = pd.concat(dfs, axis=0)

                    # Fit scalers on the combined data
                    if combined_data:
                        self.state_builder.fit_scalers(combined_data)
                        logger.info("Successfully fitted scalers on initial data")
                    else:
                        logger.warning("No data available to fit scalers")
                except Exception as e:
                    logger.error("Error fitting scalers: %s", str(e))
                    raise

        # Get configuration
        state_config = self.config.get("state", {})
        env_config = self.config.get("environment", {})
        self.window_size = state_config.get("window_size", 50)
        self.warmup_steps = env_config.get("warmup_steps", self.window_size)

        # Log configuration
        logger.info(f"[CONFIG] Window size: {self.window_size}")
        logger.info(f"[CONFIG] Warmup steps: {self.warmup_steps}")

        # Get max steps configuration
        self.max_steps = env_config.get("max_steps", 1000)
        logger.info(f"[CONFIG] Max steps per episode: {self.max_steps}")

        # Initialize last trade step
        self.last_trade_step = 0
        logger.info("[INIT] Last trade step initialized to 0")

        if self.warmup_steps < self.window_size:
            msg = (
                f"warmup_steps ({self.warmup_steps}) is less than "
                f"window_size ({self.window_size}). Setting warmup_steps to "
                f"{self.window_size}"
            )
            logger.warning(msg)
            self.warmup_steps = self.window_size

        first_asset = next(iter(self.current_data.keys()))
        first_timeframe = next(iter(self.current_data[first_asset].keys()))
        data_length = len(self.current_data[first_asset][first_timeframe])

        if data_length < self.warmup_steps:
            raise ValueError(
                f"Le premier chunk ({data_length} steps) est plus petit "
                f"que la période de warm-up requise "
                f"({self.warmup_steps} steps)."
            )

        self.step_in_chunk = 0

        for _ in range(self.warmup_steps - 1):
            self.step_in_chunk += 1
            self.current_step += 1
            if self.step_in_chunk >= data_length:
                self.current_chunk_idx += 1
                if self.current_chunk_idx >= self.total_chunks:
                    raise ValueError(
                        "Reached end of data during warm-up period. "
                        f"Current chunk: {self.current_chunk_idx}, "
                        f"Total chunks: {self.data_loader.total_chunks}"
                    )
                self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)
                self.step_in_chunk = 0
                first_asset = next(iter(self.current_data.keys()))
                first_timeframe = next(iter(self.current_data[first_asset].keys()))
                data_length = len(self.current_data[first_asset][first_timeframe])

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        """Execute one time step within the environment.
        
        This method handles the main environment loop, including:
        - Action validation and processing
        - Portfolio updates and trading
        - Reward calculation
        - Episode termination conditions
        - Chunk transitions and surveillance mode management
        
        Args:
            action: Array of actions for each asset in the portfolio
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Initialize Rich console once per environment if not done already
        if not hasattr(self, "_rich_initialized"):
            try:
                from rich.console import Console
                from rich.table import Table
                from rich.text import Text
                self._rich_console = Console(force_terminal=True, force_interactive=True)
                self._rich_table = Table
                self._rich_text = Text
                self._rich_initialized = True
                self._rich_last_print = 0
                self._rich_print_interval = max(1, int(os.getenv("ADAN_RICH_STEP_EVERY", "10")))
            except Exception as e:
                self._rich_console = None
                self._rich_initialized = True
                self.logger.debug(f"Rich console disabled: {e}")

        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
            
        # Vérifier les conditions d'urgence avant l'exécution de l'étape
        if hasattr(self, "portfolio_manager"):
            emergency_reset = self.portfolio_manager.check_emergency_condition(self.current_step)
            if emergency_reset:
                logger.critical("🆘 EMERGENCY RESET TRIGGERED - Terminating episode")
                observation = self._get_observation()
                info = self._get_info()
                info["termination_reason"] = "emergency_reset"
                return observation, 0.0, True, False, info

        # Validate action
        if not self._check_array("action", action):
            self.logger.warning("Invalid action detected, using no-op action")
            action = np.zeros_like(action, dtype=np.float32)

        # Nettoyage et validation de l'action
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self.current_step += 1
        self.global_step += 1
        self.step_in_chunk += 1

        # Log current step and action with detailed information
        chunk_info = f"chunk {self.current_chunk_idx + 1}/{self.total_chunks}" if hasattr(self, 'total_chunks') else ""
        logger.debug("[STEP LOG] step=%d, action=%s, current_chunk=%d, step_in_chunk=%d",
                    self.current_step, np.array2string(action, precision=6), 
                    self.current_chunk_idx, self.step_in_chunk)
        logger.info(f"[STEP {self.current_step} - {chunk_info}] Executing step with action: {action}")

        # Log portfolio value at the start of the step
        if hasattr(self, "portfolio_manager"):
            try:
                pv = float(self.portfolio_manager.get_portfolio_value())
                
                # Vérifier l'état de surveillance et mettre à jour si nécessaire
                if hasattr(self.portfolio_manager, '_check_surveillance_status'):
                    needs_reset = self.portfolio_manager._check_surveillance_status(self.current_step)
                    if needs_reset:
                        logger.warning("🔁 Surveillance mode reset required - ending episode")
                        observation = self._get_observation()
                        info = self._get_info()
                        info["termination_reason"] = "surveillance_reset"
                        return observation, 0.0, True, False, info
                        
                # Log surveillance status if in surveillance mode
                if hasattr(self.portfolio_manager, '_surveillance_mode') and self.portfolio_manager._surveillance_mode:
                    logger.warning(
                        "👁️  SURVEILLANCE MODE - Survived chunks: %d/2, Current value: %.2f, Start value: %.2f",
                        getattr(self.portfolio_manager, '_survived_chunks', 0),
                        pv,
                        getattr(self.portfolio_manager, 'surveillance_chunk_start_balance', 0.0)
                    )
                logger.info(
                    f"[STEP {self.current_step}] Portfolio value: {pv:.2f}"
                )
            except Exception as _e:
                logger.warning("[STEP] Failed to read portfolio value: %s", str(_e))
        else:
            logger.warning("[STEP] Portfolio manager or portfolio_value not available")

        if self.done:
            # Reset automatiquement l'environnement si l'épisode est terminé
            logger.info("Episode ended. Automatically resetting environment.")
            initial_obs, info = self.reset()
            return initial_obs, 0.0, False, False, info

        try:
            # Préparation de l'action
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

            if action.shape != (len(self.assets),):
                raise ValueError(
                    f"Action shape {action.shape} does not match "
                    f"expected shape (n_assets={len(self.assets)},)"
                )

            # Early risk check before executing any trades
            # Ensure local info dict exists to attach spot protection context
            info = {}
            current_prices = self._get_current_prices()
            try:
                # Update portfolio with current prices and enforce protection limits
                if hasattr(self, "portfolio_manager"):
                    self.portfolio_manager.update_market_price(current_prices)
                    protection_triggered = self.portfolio_manager.check_protection_limits(current_prices)
                    if protection_triggered:
                        if getattr(self.portfolio_manager, "futures_enabled", False):
                            # In futures mode, terminate on protection (e.g., liquidation)
                            info = {
                                "termination_reason": "Risk protection triggered",
                                "current_prices": current_prices,
                                "protection": "futures_liquidation_or_breach",
                            }
                            observation = self._get_observation()
                            logger.warning("[TERMINATION] Risk protection triggered at step %d (futures)", self.current_step)
                            self.done = True
                            return observation, 0.0, True, False, info
                        else:
                            # Spot mode: protection disables new buys; continue episode
                            logger.warning("[PROTECTION] Spot drawdown breach: new BUY orders disabled. Continuing episode.")
                            info.update({"protection": "spot_drawdown", "trading_disabled": True})
            except Exception as risk_e:
                logger.error("Early risk check failed: %s", str(risk_e), exc_info=True)

            # Mise à jour de l'état DBE et calcul de la modulation
            self._update_dbe_state()
            dbe_modulation = self.dbe.compute_dynamic_modulation()

            # Exécution des trades avec modulation DBE et récupération du PnL réalisé
            # Capture positions snapshot before executing trades to detect activity
            positions_before = None
            try:
                if hasattr(self, "portfolio_manager") and hasattr(self.portfolio_manager, "get_metrics"):
                    m_before = self.portfolio_manager.get_metrics() or {}
                    positions_before = {k: (v.get("quantity") or v.get("size") or 0.0)
                                        for k, v in (m_before.get("positions", {}) or {}).items()}
            except Exception as _e:
                logger.debug(f"[STEP] Failed capturing positions before trade: {_e}")

            trade_start_time = time.time()
            realized_pnl = self._execute_trades(action, dbe_modulation)
            trade_end_time = time.time()
            logger.debug(
                f"_execute_trades took {trade_end_time - trade_start_time:.4f} seconds"
            )

            # Detect trade activity by comparing positions snapshots
            try:
                if positions_before is not None and hasattr(self, "portfolio_manager") and hasattr(self.portfolio_manager, "get_metrics"):
                    m_after = self.portfolio_manager.get_metrics() or {}
                    positions_after = {k: (v.get("quantity") or v.get("size") or 0.0)
                                       for k, v in (m_after.get("positions", {}) or {}).items()}
                    if positions_after != positions_before:
                        self.last_trade_step = self.current_step
                        logger.debug(f"[TRADE] Positions changed at step {self.current_step} -> last_trade_step updated")
            except Exception as _e:
                logger.debug(f"[STEP] Failed detecting trade activity: {_e}")

            # Journalisation du PnL réalisé
            logger.info(f"[REWARD] Realized PnL for step: ${realized_pnl:.2f}")

            self.step_in_chunk += 1

            first_asset = next(iter(self.current_data.keys()))
            data_length = len(self.current_data[first_asset])

            MIN_EPISODE_STEPS = 500  # Minimum absolu avant évaluation
            done = False
            termination_reason = ""

            # Log current state before checking termination conditions
            steps_since_trade = (
                "-" if (self.last_trade_step is None or self.last_trade_step < 0)
                else str(self.current_step - self.last_trade_step)
            )
            logger.info(
                f"[TERMINATION CHECK] Step: {self.current_step}, "
                f"Max Steps: {self.max_steps}, "
                f"Portfolio Value: {self.portfolio_manager.get_portfolio_value():.2f}, "
                f"Initial Equity: {self.portfolio_manager.initial_equity:.2f}, "
                f"Steps Since Last Trade: {steps_since_trade}"
            )

            if self.current_step < MIN_EPISODE_STEPS:
                done = False
                termination_reason = (
                    f"Min steps not reached ({self.current_step} < {MIN_EPISODE_STEPS})"
                )
                logger.info(f"[TERMINATION] {termination_reason}")
            elif self.current_step >= self.max_steps:
                done = True
                termination_reason = (
                    f"Max steps reached ({self.current_step} >= {self.max_steps})"
                )
                logger.info(f"[TERMINATION] {termination_reason}")
            elif self.portfolio_manager.get_portfolio_value() <= self.portfolio_manager.initial_equity * 0.70:
                done = True
                termination_reason = (
                    f"Max drawdown exceeded ({self.portfolio_manager.get_portfolio_value():.2f} "
                    f"<= {self.portfolio_manager.initial_equity * 0.70:.2f})"
                )
                logger.warning(f"[TERMINATION] {termination_reason}")
            elif self.current_step - self.last_trade_step > 300:
                done = True
                termination_reason = f"Max inactive steps reached ({self.current_step - self.last_trade_step} > 300)"
                logger.warning(f"[TERMINATION] {termination_reason}")
            elif self.current_step >= data_length - 1:
                done = True
                termination_reason = f"End of chunk reached (step {self.current_step} >= {data_length - 1})"
                logger.info(f"[TERMINATION] {termination_reason}")

            # Ensure environment done flag is set when a termination condition is met
            if done:
                self.done = True

            # Vérifier si nous avons atteint la fin du chunk actuel
            if self.step_in_chunk >= data_length:
                self.current_chunk_idx += 1
                self.current_chunk += 1
                
                # Vérifier si on a atteint le nombre maximum de chunks pour cet épisode
                chunks_limit = min(self.total_chunks, self.max_chunks_per_episode)
                
                if self.current_chunk_idx >= chunks_limit:
                    done = True
                    self.done = True
                    termination_reason = (
                        f"Max chunks per episode reached ({self.current_chunk_idx} >= {self.max_chunks_per_episode})"
                    )
                    logger.info(f"[TERMINATION] {termination_reason}")
                else:
                    # Charger le prochain chunk
                    logger.debug(
                        f"[CHUNK] Loading next chunk {self.current_chunk_idx + 1}/"
                        f"{chunks_limit}"
                    )
                    self.current_data = self.data_loader.load_chunk(self.current_chunk_idx)
                    self.step_in_chunk = 0
                    
                    # Réinitialiser les composants pour le nouveau chunk si nécessaire
                    if hasattr(self.dbe, "_reset_for_new_chunk"):
                        logger.debug("[DBE] Resetting DBE for new chunk")
                        self.dbe._reset_for_new_chunk()
                    
                    logger.info(
                        f"[CHUNK] Successfully loaded chunk {self.current_chunk_idx + 1}/"
                        f"{chunks_limit}"
                    )
            
            # Log final decision and handle episode termination
            if done:
                logger.info(
                    f"[EPISODE END] Episode ending. Reason: {termination_reason}"
                )
                logger.info(
                    f"[EPISODE STATS] Total steps: {self.current_step}, "
                    f"Final portfolio value: {self.portfolio_manager.get_portfolio_value():.2f}, "
                    f"Return: {(self.portfolio_manager.get_portfolio_value()/self.portfolio_manager.initial_equity - 1)*100:.2f}%"
                )
            else:
                logger.debug(
                    f"[TERMINATION] Episode continues. Current step: {self.current_step}"
                )

            # Build observations and validate
            current_observation = self._get_observation()
            if not self._check_array(
                "observation",
                np.concatenate([v.flatten() for v in current_observation.values()]),
            ):
                self.logger.error("Invalid observation detected, resetting environment")
                obs_reset, info_reset = self.reset()
                return obs_reset, 0.0, True, False, {
                    "nan_detected": True,
                    "nan_source": "observation",
                }

            # Calculate reward using internal shaper (includes risk penalties/tier adjustments)
            reward = self._calculate_reward(action)
            # Use local 'done' to signal termination for this step
            terminated = done
            truncated = False

            max_steps = getattr(self, "_max_episode_steps", float("inf"))
            if self.current_step >= max_steps:
                truncated = True
                self.done = True

            info = self._get_info()

            if hasattr(self, "_last_reward_components"):
                info.update({"reward_components": self._last_reward_components})

            # --- Minimal structured JSON-lines logging for multicolumn visualization ---
            try:
                # Prepare JSON metrics using available fields; null for unavailable ones
                pm = getattr(self, "portfolio_manager", None)
                pm_metrics = pm.get_metrics() if pm and hasattr(pm, "get_metrics") else {}
                portfolio_value = pm_metrics.get("total_value") or pm_metrics.get("total_capital")
                cash = pm_metrics.get("cash")
                sharpe = pm_metrics.get("sharpe_ratio")
                max_dd = pm_metrics.get("max_drawdown")
                trading_disabled = bool(getattr(pm, "trading_disabled", False)) if pm else False
                futures_enabled = bool(getattr(pm, "futures_enabled", False)) if pm else False
                current_prices = info.get("market", {}).get("current_prices") or {}
                # Derive a basic protection event label for quick filtering
                protection_event = (
                    "futures_liquidation" if futures_enabled and self.done else (
                        "spot_drawdown" if (not futures_enabled and trading_disabled) else "none"
                    )
                )
                # Compose compact positions list: symbol:size:entry_price:side if available
                positions_compact = []
                for sym, pos in pm_metrics.get("positions", {}).items():
                    size = pos.get("size") or pos.get("quantity")  # Préférer 'size', avec fallback sur 'quantity' pour rétrocompatibilité
                    entry = pos.get("entry_price") or pos.get("avg_price")
                    side = "LONG" if (size or 0) >= 0 else "SHORT"
                    positions_compact.append(f"{sym}:{float(size or 0):.8f}:{float(entry or 0):.8f}:{side}")
                reward_components = info.get("reward_components") or {}
                event_tags = []
                if trading_disabled:
                    event_tags.append("[PROTECTION]")
                # Detect tier change
                current_tier = (pm_metrics or {}).get("tier")
                last_tier = getattr(self, "_last_tier", None)
                tier_changed = (current_tier is not None and current_tier != last_tier)
                if tier_changed:
                    event_tags.append("[TIER]")
                setattr(self, "_last_tier", current_tier)
                # Pull potential sizer outputs from info if available
                sizer_final_val = info.get("sizer_final")
                sizer_reason_val = info.get("sizer_reason")
                sizer_clamped = (sizer_final_val == 0) or (sizer_reason_val is not None)
                if sizer_clamped:
                    event_tags.append("[SIZER]")
                # Build record
                record = {
                    "timestamp": self._get_safe_timestamp(),
                    "step": int(self.current_step),
                    "env_id": int(getattr(self, "worker_id", 0)),
                    "episode_id": int(getattr(self, "episode_count", 0)),
                    "chunk_id": int(getattr(self, "current_chunk", 0)),
                    "action": action.tolist() if isinstance(action, np.ndarray) else action,
                    "action_meaning": "VECTOR",
                    "price_reference": None,
                    "sizer_raw": None,
                    "sizer_final": sizer_final_val if sizer_final_val is not None else None,
                    "sizer_reason": sizer_reason_val if sizer_reason_val is not None else None,
                    "available_cash": float(cash) if cash is not None else None,
                    "portfolio_value": float(portfolio_value) if portfolio_value is not None else None,
                    "cash": float(cash) if cash is not None else None,
                    "positions_value": info.get("portfolio", {}).get("total_position_value"),
                    "unrealized_pnl": None,
                    "realized_pnl": float(realized_pnl) if 'realized_pnl' in locals() and realized_pnl is not None else None,
                    "cum_realized_pnl": None,
                    "num_positions": int(info.get("portfolio", {}).get("num_positions", 0)),
                    "positions": positions_compact,
                    "order_notional": None,
                    "order_status": None,
                    "commission": None,
                    "slippage": None,
                    "reward": float(reward),
                    "reward_components": reward_components,
                    "drawdown_value": float(max_dd) if max_dd is not None else None,
                    "drawdown_pct": float(max_dd) if max_dd is not None else None,
                    "max_drawdown_pct": None,
                    "tier": str(getattr(self, "current_tier", "")) if getattr(self, "current_tier", None) is not None else None,
                    "trading_disabled": trading_disabled,
                    "protection_event": protection_event,
                    "protection_msg": None,
                    "dbE_regime": None,
                    "dbe_params": None,
                    "ppo_metrics": None,
                    "learning_rate": None,
                    "grad_norm": None,
                    "num_trades_step": None,
                    "cum_num_trades": None,
                    "num_wins": None,
                    "num_losses": None,
                    "winrate": None,
                    "avg_win": None,
                    "avg_loss": None,
                    "avg_trade_duration": None,
                    "last_trade_entry_step": None,
                    "last_trade_exit_step": None,
                    "metrics_sharpe": float(sharpe) if sharpe is not None else None,
                    "metrics_volatility": None,
                    "throughput": info.get("performance", {}).get("steps_per_second"),
                    "memory_usage": None,
                    "custom_tags": event_tags,
                    "notes": None,
                }
                # Sampling control to reduce noise: default every 10 steps, always on protection events
                jsonl_every_env = os.getenv("ADAN_JSONL_EVERY", "")
                jsonl_every_cfg = 10
                try:
                    jsonl_every_cfg = int((self.config or {}).get("logging", {}).get("jsonl_every", 10)) if hasattr(self, "config") else 10
                except Exception:
                    jsonl_every_cfg = 10
                jsonl_every = int(jsonl_every_env) if jsonl_every_env.isdigit() else jsonl_every_cfg
                should_write = (
                    (self.current_step % max(1, jsonl_every) == 0)
                    or (protection_event != "none")
                    or sizer_clamped
                    or tier_changed
                )
                if should_write:
                    logs_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "logs")
                    )
                    os.makedirs(logs_dir, exist_ok=True)
                    jsonl_path = os.path.join(logs_dir, "training_events.jsonl")
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, separators=(",", ":")) + "\n")
            except Exception as _log_e:
                logger.debug("[JSONL] Failed to write training event: %s", str(_log_e))

            # Quiet verbose DEBUG logs after initial checks (one-time)
            try:
                if not getattr(self, "_quiet_after_init", False):
                    # default ON; set ADAN_QUIET_AFTER_INIT=0 to disable
                    _quiet_env = os.getenv("ADAN_QUIET_AFTER_INIT", "1").lower() in ("1", "true", "yes", "on")
                    if _quiet_env and int(self.current_step) >= 1:
                        try:
                            import logging as _logging
                            logger.setLevel(_logging.INFO)
                        except Exception:
                            pass
                        self._quiet_after_init = True
            except Exception:
                pass

            # --- Rich Summary Table ---
            if hasattr(self, "_rich_console") and self._rich_console is not None:
                    # Get configuration
                    rich_cfg = (self.config or {}).get("logging", {}) if hasattr(self, "config") else {}
                    env_enabled = os.getenv("ADAN_RICH_STEP_TABLE", "").lower() in ("1", "true", "yes", "on")
                    rich_enabled = rich_cfg.get("rich_step_table", True) if "rich_step_table" in rich_cfg else env_enabled

                    # If rich table is not enabled, skip rendering here and continue
                    if not rich_enabled:
                        pass

                    # Check if we should print on this step
                    print_interval = getattr(self, '_rich_print_interval', 10)
                    should_print = (self.current_step % print_interval == 0) or \
                                 (self.current_step - getattr(self, '_rich_last_print', 0) >= print_interval)

                    if should_print:
                        self._rich_last_print = self.current_step

                        # Local import for Text to avoid scope issues
                        from rich.text import Text

                        # Helpers
                        def _fmt(v):
                            if v is None:
                                return "-"
                            if isinstance(v, float):
                                return f"{v:.6g}"
                            return str(v)

                        def _dd_cell(v):
                            if not isinstance(v, (int, float)):
                                return Text("-")
                            if v < 0.05:
                                return Text(f"{v:.3f}", style="green")
                            if v < 0.20:
                                return Text(f"{v:.3f}", style="yellow")
                            if v < 0.50:
                                return Text(f"{v:.3f}", style="orange3")
                            return Text(f"{v:.3f}", style="red")

                        def _reward_cell(v, avg=None):
                            if not isinstance(v, (int, float)):
                                return Text("-")
                            style = None
                            if v > 0:
                                style = "green3"
                            elif avg is not None and isinstance(avg, (int, float)) and v < -1.0 * abs(avg):
                                style = "red"
                            elif v < 0:
                                style = "orange3"
                            return Text(f"{v:.4g}", style=style)

                        def _prot_cell(p):
                            if p and p != "none":
                                return Text(str(p), style="orange3")
                            return Text("none")

                        def _reason_cell(reason: str):
                            if not reason:
                                return Text("-")
                            r = str(reason)
                            if "insufficient_cash" in r:
                                return Text(r, style="magenta")
                            if "min_notional" in r:
                                return Text(r, style="orange3")
                            if ("step_size" in r) or ("precision" in r):
                                return Text(r, style="gold1")
                            if "trading_disabled" in r:
                                return Text(r, style="red")
                            return Text(r)

                        def _winrate_cell(w):
                            if not isinstance(w, (int, float)):
                                return Text("-")
                            if w >= 0.6:
                                return Text(f"{w:.2f}", style="green3")
                            if w >= 0.4:
                                return Text(f"{w:.2f}")
                            return Text(f"{w:.2f}", style="orange3")

                        def _loss_cell(cur, prev):
                            if not isinstance(cur, (int, float)):
                                return Text("-")
                            if isinstance(prev, (int, float)):
                                delta = cur - prev
                                if delta > 0:
                                    # big increase vs previous -> orange, extremely large -> red
                                    return Text(f"{cur:.3g}", style="red" if delta > abs(prev) * 2 else "orange3")
                            return Text(f"{cur:.3g}")

                        # Gather row fields
                        ts = record.get("timestamp")
                        ts_short = ts[11:19] if isinstance(ts, str) and len(ts) >= 19 else "--:--:--"
                        step_id = _fmt(record.get("step"))
                        env_id = _fmt(record.get("env_id"))
                        ep_id = _fmt(record.get("episode_id"))
                        pv = record.get("portfolio_value")
                        ddv = record.get("drawdown_pct")
                        tier = _fmt(record.get("tier"))
                        td_flag = bool(record.get("trading_disabled"))
                        prot = record.get("protection_event")
                        reward_val = record.get("reward")
                        # Rolling average of reward for magnitude-based coloring
                        avg_reward = getattr(self, "_reward_avg", None)
                        try:
                            if isinstance(reward_val, (int, float)):
                                if avg_reward is None:
                                    avg_reward = float(reward_val)
                                else:
                                    # EMA with smoothing factor
                                    beta = 0.1
                                    avg_reward = (1 - beta) * float(avg_reward) + beta * float(reward_val)
                                setattr(self, "_reward_avg", avg_reward)
                        except Exception:
                            pass
                        sizer_f = record.get("sizer_final")
                        sizer_r = record.get("sizer_reason")
                        trades_step = record.get("num_trades_step")
                        trades_cum = record.get("cum_num_trades")
                        winrate = record.get("winrate")
                        sharpe = record.get("metrics_sharpe")
                        ppo = info.get("ppo_metrics", {}) if isinstance(info, dict) else {}
                        pol_loss = ppo.get("policy_loss")
                        val_loss = ppo.get("value_loss")
                        prev_pol_loss = getattr(self, "_prev_policy_loss", None)
                        prev_val_loss = getattr(self, "_prev_value_loss", None)
                        self._prev_policy_loss = pol_loss
                        self._prev_value_loss = val_loss

                        # Build compact live table
                        from rich import box
                        table = self._rich_table(
                            title=f"Step {self.current_step} - {self._get_safe_timestamp()}",
                            box=box.SIMPLE,
                            show_header=True,
                            header_style="bold magenta",
                            show_lines=True,
                            title_justify="left",
                            expand=False
                        )
                        table.add_column("t", justify="left")
                        table.add_column("step", justify="right")
                        table.add_column("env", justify="right")
                        table.add_column("ep", justify="right")
                        table.add_column("pv", justify="right")
                        table.add_column("dd%", justify="right")
                        table.add_column("tier", justify="center")
                        table.add_column("TD", justify="center")
                        table.add_column("prot", justify="left")
                        table.add_column("reward", justify="right")
                        table.add_column("sizer", justify="right")
                        table.add_column("trades", justify="right")
                        table.add_column("winrate", justify="right")
                        table.add_column("sharpe", justify="right")
                        table.add_column("polL", justify="right")
                        table.add_column("valL", justify="right")
                        table.add_column("tags", justify="left")

                        row_style = "bold white on red" if td_flag else None
                        table.add_row(
                            Text(ts_short),
                            Text(str(step_id)),
                            Text(str(env_id)),
                            Text(str(ep_id)),
                            Text(_fmt(pv)),
                            _dd_cell(ddv),
                            Text(str(tier)),
                            Text("T" if td_flag else "F"),
                            _prot_cell(prot),
                            _reward_cell(reward_val, avg_reward),
                            _reason_cell(_fmt(sizer_r)) if sizer_r else Text(_fmt(sizer_f)),
                            Text(f"{_fmt(trades_step)}|{_fmt(trades_cum)}"),
                            _winrate_cell(winrate),
                            Text(_fmt(sharpe)),
                            _loss_cell(pol_loss, prev_pol_loss),
                            _loss_cell(val_loss, prev_val_loss),
                            Text("".join(event_tags)),
                            style=row_style,
                        )
                        self._rich_console.print(table)

            if self.shared_buffer is not None:
                experience = {
                    "state": current_observation,
                    "action": action,
                    "reward": float(reward),
                    "next_state": current_observation,
                    "done": terminated or truncated,
                    "info": info,
                    "timestamp": self._get_safe_timestamp() or str(self.current_step),
                    "worker_id": self.worker_id,
                }
                self.shared_buffer.add(experience)

            return current_observation, float(reward), terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in step(): {str(e)}", exc_info=True)
            self.done = True
            observation = self._get_observation()
            info = self._get_info()
            info["error"] = str(e)
            return observation, 0.0, True, False, info

    def _update_dbe_state(self) -> None:
        """Update the DBE state with current market conditions."""
        try:
            current_prices = self._get_current_prices()
            portfolio_metrics = self.portfolio.get_metrics()

            live_metrics = {
                "step": self.current_step,
                "current_prices": current_prices,
                "portfolio_value": portfolio_metrics.get("total_capital", 0.0),
                "cash": portfolio_metrics.get("cash", 0.0),
                "positions": portfolio_metrics.get("positions", {}),
                "returns": portfolio_metrics.get("returns", 0.0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0.0),
            }

            if hasattr(self, "current_data") and self.current_data:
                first_asset = next(iter(self.current_data.keys()))
                if first_asset in self.current_data and self.current_data[first_asset]:
                    first_tf = next(iter(self.current_data[first_asset].keys()))
                    df = self.current_data[first_asset][first_tf]

                    if not df.empty and self.current_step < len(df):
                        current_row = df.iloc[self.current_step]
                        live_metrics.update(
                            {
                                "rsi": current_row.get("rsi", 50.0),
                                "adx": current_row.get("adx", 20.0),
                                "atr": current_row.get("atr", 0.0),
                                "atr_pct": current_row.get("atr_pct", 0.0),
                                "ema_ratio": current_row.get("ema_ratio", 1.0),
                            }
                        )
            if hasattr(self, "dbe"):
                self.dbe.update_state(live_metrics)

        except Exception as e:
            logger.warning(f"Failed to update DBE state: {e}")

    def _check_array(self, name: str, arr: np.ndarray) -> bool:
        """Vérifie la présence de NaN/Inf dans un tableau et enregistre un rapport détaillé.

        Args:
            name: Nom de la variable pour les logs
            arr: Tableau NumPy à vérifier

        Returns:
            bool: True si le tableau est valide, False sinon
        """
        if not isinstance(arr, np.ndarray):
            self.logger.warning(f"{name} is not a numpy array, got {type(arr)}")
            return True

        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))

        if has_nan or has_inf:
            issues = []
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")

            self.logger.error(
                f"Invalid values detected in {name} at step {self.current_step}: {' and '.join(issues)}"
            )

            # Enregistrement du contexte
            try:
                dump_path = os.path.join(
                    os.getcwd(),
                    f"nan_dump_{name}_step{self.current_step}.npz"
                )
                np.savez(dump_path, arr=arr)
                self.logger.info(f"Dumped {name} state to {dump_path}")
            except Exception as e:
                self.logger.error(f"Failed to dump {name} state: {e}")

            return False

        return True

    def _get_current_prices(self) -> Dict[str, float]:
        """Get the current prices for all assets with caching."""
        current_time = time.time()
        prices = {}

        # Vérifier si le cache est activé
        perf_config = self.config.get("trading", {}).get("performance", {})
        enable_caching = perf_config.get("enable_data_caching", True)

        for _asset, timeframe_data in self.current_data.items():
            # Vérifier si le prix est en cache et toujours valide
            cache_valid = (
                enable_caching
                and hasattr(self, "_price_cache")
                and _asset in self._price_cache
                and hasattr(self, "_last_price_update")
                and _asset in self._last_price_update
                and current_time - self._last_price_update[_asset] < self._cache_ttl
            )

            if cache_valid:
                prices[_asset] = self._price_cache[_asset]
                continue

            # Calculer le prix courant à partir du timeframe de base '5m' si disponible
            try:
                base_df = None
                if isinstance(timeframe_data, dict) and "5m" in timeframe_data:
                    base_df = timeframe_data["5m"]
                else:
                    # fallback: premier timeframe disponible
                    first_tf = next(iter(timeframe_data.keys())) if timeframe_data else None
                    if first_tf is not None:
                        base_df = timeframe_data[first_tf]

                if base_df is not None and len(base_df) > 0:
                    idx = min(self.current_step, len(base_df) - 1)
                    if "close" in base_df.columns:
                        prices[_asset] = float(base_df.iloc[idx]["close"])
                    else:
                        numeric_cols = base_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            prices[_asset] = float(base_df.iloc[idx][numeric_cols[0]])
            except Exception as e:
                logger.warning(f"Error getting price for {_asset}: {str(e)}")

        return prices

    def _validate_market_data(self, prices: Dict[str, float]) -> bool:
        """Valide les données de marché avant l'exécution des trades.

        Args:
            prices: Dictionnaire des prix actuels par actif

        Returns:
            bool: True si les données sont valides, False sinon
        """
        if not prices:
            logger.error("No market data available")
            return False

        invalid_assets = [
            asset
            for asset, price in prices.items()
            if price <= 0 or not np.isfinite(price)
        ]

        if invalid_assets:
            invalid_list = ", ".join(invalid_assets)
            logger.error("Invalid prices for assets: %s", invalid_list)
            return False

        return True

    def _log_trade_error(
        self, asset: str, action_value: float, price: float, error: str
    ) -> None:
        """Enregistre les erreurs de trading pour analyse ultérieure.

        Args:
            asset: Symbole de l'actif concerné
            action_value: Valeur de l'action (-1 à 1)
            price: Prix au moment de l'erreur
            error: Message d'erreur détaillé
        """
        # Déterminer le type d'action
        if action_value > 0.1:
            action = "BUY"
        elif action_value < -0.1:
            action = "SELL"
        else:
            action = "HOLD"

        # Préparer les informations d'erreur
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "asset": asset,
            "action": action,
            "action_value": float(action_value),
            "price": float(price) if price is not None else None,
            "error": error,
        }

        # Ajouter la valeur du portefeuille si disponible
        if hasattr(self.portfolio, "portfolio_value"):
            error_info["portfolio_value"] = float(self.portfolio.portfolio_value)

        # Logger l'erreur
        logger.error(f"Trade error: {error_info}")

        # Enregistrer dans un fichier si configuré
        log_config = self.config.get("logging", {})
        error_log_path = log_config.get("error_log_path", "trade_errors.log")

        try:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_info) + "\n")
        except IOError as e:
            logger.error(f"Failed to write to error log: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error writing to log: {str(e)}")

    def _get_current_timestamp(self) -> pd.Timestamp:
        """Get the current timestamp."""
        for _asset, timeframe_data in self.current_data.items():
            if not timeframe_data:
                continue
            tf = next(iter(timeframe_data.keys()))
            df = timeframe_data[tf]
            if not df.empty and self.current_step < len(df):
                return df.index[self.current_step]
        raise RuntimeError("No timestamp data available")

    def _get_safe_timestamp(self) -> Optional[str]:
        """Get the current timestamp safely."""
        try:
            return self._get_current_timestamp().isoformat()
        except Exception:
            return None

    def _manage_cache(self, key: str, value: np.ndarray = None) -> Optional[np.ndarray]:
        """Gère le cache d'observations avec une politique LRU."""
        if key in self._observation_cache:
            self._cache_access[key] = time.time()
            self._cache_hits += 1
            return self._observation_cache[key]

        self._cache_misses += 1

        if len(self._observation_cache) >= self._max_cache_size:
            sorted_keys = sorted(
                self._cache_access.keys(), key=lambda k: self._cache_access[k]
            )
            num_to_remove = max(1, int(self._max_cache_size * 0.1))
            for k in sorted_keys[:num_to_remove]:
                self._observation_cache.pop(k, None)
                self._cache_access.pop(k, None)

        if value is not None:
            self._observation_cache[key] = value
            self._cache_access[key] = time.time()

        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation du cache."""
        total = self._cache_hits + self._cache_misses
        hit_ratio = self._cache_hits / total if total > 0 else 0.0

        return {
            "cache_enabled": True,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._observation_cache),
            "max_size": self._max_cache_size,
            "hit_ratio": hit_ratio,
        }

    def _process_assets(
        self, feature_config: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Process asset data for the current step.

        Args:
            feature_config: Dictionary mapping timeframes to lists of feature names

        Returns:
            Dictionary mapping assets to dictionaries of {timeframe: DataFrame}

            Format attendu par StateBuilder.build_observation():
            {
                'asset1': {
                    '5m': DataFrame,
                    '1h': DataFrame,
                    '4h': DataFrame
                },
                'asset2': {
                    '5m': DataFrame,
                    '1h': DataFrame,
                    '4h': DataFrame
                },
                ...
            }
        """
        processed_data = {asset: {} for asset in self.assets}

        for asset in self.assets:
            for timeframe in self.timeframes:
                if timeframe not in feature_config:
                    logger.warning(
                        f"No feature configuration for timeframe {timeframe}"
                    )
                    continue

                features = feature_config[timeframe]
                if not features:
                    logger.warning(f"No features specified for timeframe {timeframe}")
                    continue

                # Get data for this asset and timeframe
                asset_data = self.current_data.get(asset, {}).get(timeframe)
                if asset_data is None or asset_data.empty:
                    logger.debug(f"No data for {asset} {timeframe}")
                    continue

                # Log all available columns for debugging
                logger.debug(
                    f"Available columns in {asset} {timeframe} data: {asset_data.columns.tolist()}"
                )
                logger.debug(f"Requested features for {timeframe}: {features}")

                # Create a mapping of uppercase column names to actual column names
                column_mapping = {col.upper(): col for col in asset_data.columns}

                # Find available features in the asset data (case-insensitive)
                available_features = []
                missing_features = []
                for f in features:
                    upper_f = f.upper()
                    if upper_f in column_mapping:
                        available_features.append(column_mapping[upper_f])
                        logger.debug(
                            f"Found feature: '{f}' -> '{column_mapping[upper_f]}'"
                        )
                    else:
                        missing_features.append(f)
                        logger.debug(
                            f"Missing feature: '{f}' (not in DataFrame columns)"
                        )

                if missing_features:
                    logger.warning(
                        f"Missing {len(missing_features)} features for {asset} {timeframe}: {missing_features}"
                    )
                    logger.debug(
                        f"Available columns in {asset} {timeframe}: {asset_data.columns.tolist()}"
                    )
                    logger.debug(f"Available features: {available_features}")

                if not available_features:
                    logger.warning(
                        f"None of the requested features found for {asset} {timeframe}"
                    )
                    continue

                try:
                    # Select only the requested features using their original case
                    asset_df = asset_data[available_features].copy()

                    # Ensure column names are in uppercase for consistency
                    asset_df.columns = [col.upper() for col in asset_df.columns]

                    # Store the DataFrame in the processed data
                    processed_data[asset][timeframe] = asset_df

                except Exception as e:
                    logger.error(f"Error processing {asset} {timeframe}: {str(e)}")
                    logger.debug(f"Available columns: {asset_data.columns.tolist()}")
                    logger.debug(f"Available features: {available_features}")

        # Remove assets with no data
        return {k: v for k, v in processed_data.items() if v}

    def _create_empty_dataframe(
        self, timeframe: str, window_size: int = None
    ) -> pd.DataFrame:
        """Create an empty DataFrame with required features for a given timeframe.

        Args:
            timeframe: The timeframe for which to create the empty DataFrame
            window_size: Number of rows to include in the empty DataFrame

        Returns:
            DataFrame with required columns and zero values
        """
        try:
            # Get required features for this timeframe
            features = self.state_builder.get_feature_names(timeframe)
            if not features:
                logger.warning(f"No features defined for timeframe {timeframe}")
                features = ["close"]  # Fallback to basic column

            # Use window_size if provided, otherwise default to 1
            rows = window_size if window_size is not None else 1

            # Create empty DataFrame with required features
            empty_data = np.zeros((rows, len(features)))
            df = pd.DataFrame(empty_data, columns=features)

            # Add timestamp column if not present
            if (
                "timestamp" not in df.columns
                and "timestamp" in self.features[timeframe]
            ):
                df["timestamp"] = pd.Timestamp.now()

            logger.debug(
                f"Created empty DataFrame for {timeframe} with shape {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error creating empty DataFrame for {timeframe}: {str(e)}")
            # Fallback to minimal DataFrame
            return pd.DataFrame(columns=["timestamp", "close"])

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construit et retourne l'observation actuelle sous forme de dictionnaire.

        Returns:
            Dict[str, np.ndarray]: Dictionnaire contenant :
                - 'observation': np.ndarray de forme (timeframes, window_size, features)
                - 'portfolio_state': np.ndarray de forme (17,) avec les métriques du portefeuille
        """
        try:
            # 1. Récupérer l'observation du marché depuis StateBuilder
            # Préparer les données pour build_observation
            data = self._process_assets(self.state_builder.features_config)
            market_obs = self.state_builder.build_observation(
                current_idx=self.current_step, data=data
            )

            # 2. Récupérer l'état du portefeuille
            portfolio_state = self.portfolio_manager.get_state()

            # 3. Vérifier et convertir les types
            market_obs = np.asarray(market_obs, dtype=np.float32)
            portfolio_state = np.asarray(portfolio_state, dtype=np.float32)

            # 4. Vérifier les dimensions
            if len(market_obs.shape) != 3:
                raise ValueError(
                    f"L'observation du marché doit être 3D, reçu : {market_obs.shape}"
                )

            if len(portfolio_state.shape) != 1 or portfolio_state.shape[0] != 17:
                raise ValueError(
                    f"L'état du portefeuille doit être de dimension (17,), reçu : {portfolio_state.shape}"
                )

            # 5. Créer et retourner le dictionnaire d'observation
            observation = {
                "observation": market_obs,
                "portfolio_state": portfolio_state,
            }

            # Mettre en cache l'observation actuelle
            self._current_observation = observation

            return observation

        except Exception as e:
            logger.error(f"Erreur dans _get_observation : {str(e)}")
            logger.error(traceback.format_exc())

            # En cas d'erreur, retourner une observation vide avec les bonnes dimensions
            num_timeframes = len(self.state_builder.features_config)
            # Utiliser max_features au lieu de la somme pour être cohérent avec StateBuilder
            num_features = (
                max(len(feats) for feats in self.state_builder.features_config.values())
                if self.state_builder.features_config
                else 1
            )
            return {
                "observation": np.zeros(
                    (num_timeframes, self.window_size, num_features), dtype=np.float32
                ),
                "portfolio_state": np.zeros(17, dtype=np.float32),
            }

    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calcule la récompense pour l'étape actuelle.

        La récompense est calculée comme suit :
        - Récompense de base : rendement du portefeuille
        - Pénalité de risque : basée sur le drawdown maximum
        - Pénalité de transaction : basée sur le turnover
        - Pénalité de concentration : pénalise les positions trop importantes
        - Pénalité de régularité des actions : pénalise les changements brusques

        Args:
            action: Vecteur d'actions du modèle

        Returns:
            float: Valeur de la récompense
        """
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Récupération des métriques du portefeuille
        portfolio_metrics = self.portfolio_manager.get_metrics()
        returns = portfolio_metrics.get("returns", 0.0)
        max_drawdown = portfolio_metrics.get("max_drawdown", 0.0)
        reward_config = self.config.get("reward", {})

        # Configuration des paramètres de récompense
        return_scale = reward_config.get("return_scale", 1.0)
        risk_aversion = reward_config.get("risk_aversion", 1.5)

        # Calcul de la récompense de base
        base_reward = returns * return_scale
        risk_penalty = risk_aversion * max_drawdown

        # Calcul de la pénalité de transaction
        transaction_penalty = 0.0
        if hasattr(self, "_last_portfolio_value"):
            last_value = self._last_portfolio_value
            current_value = portfolio_metrics.get("total_value", 0.0)
            turnover = abs(current_value - last_value) / max(1.0, last_value)
            transaction_penalty = (
                reward_config.get("transaction_cost_penalty", 0.1) * turnover
            )

        # Calcul de la pénalité de concentration
        position_concentration = 0.0
        if portfolio_metrics.get("total_value", 0) > 0:
            positions = portfolio_metrics.get("positions", {})
            position_values = [p.get("value", 0) for p in positions.values()]
            if position_values:
                max_position = max(position_values)
                position_concentration = (
                    max_position / portfolio_metrics["total_value"]
                ) ** 2

        concentration_penalty = (
            reward_config.get("concentration_penalty", 0.5) * position_concentration
        )

        # Calcul de la pénalité de régularité des actions
        action_smoothness_penalty = 0.0
        if hasattr(self, "_last_action") and self._last_action is not None:
            action_diff = np.mean(np.abs(action - self._last_action))
            action_smoothness_penalty = (
                reward_config.get("action_smoothness_penalty", 0.1) * action_diff
            )

        # Calcul de la récompense finale avec toutes les pénalités
        base_reward = returns * return_scale
        total_reward = base_reward - (
            risk_penalty +
            transaction_penalty +
            concentration_penalty +
            action_smoothness_penalty
        )

        # Mise à jour des métriques
        self.episode_reward += total_reward

        # Mettre à jour les valeurs pour la prochaine itération
        self._last_portfolio_value = portfolio_metrics.get("total_value", 0.0)
        self._last_action = action.copy()

        return total_reward

    def _save_checkpoint(self) -> Dict[str, Any]:
        """Sauvegarde l'état actuel de l'environnement et du portefeuille.

        Returns:
            Dict contenant l'état sauvegardé
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_state': self.portfolio.get_state(),
            'env_state': {
                'current_step': self.current_step,
                'current_chunk': self.current_chunk,
                'episode_count': self.episode_count,
                'episode_reward': self.episode_reward,
                'best_portfolio_value': self.best_portfolio_value
            },
            'tier_info': {
                'current_tier': self.current_tier['name'] if self.current_tier else None,
                'episodes_in_tier': self.episodes_in_tier,
                'last_tier_change_step': self.last_tier_change_step
            }
        }

    def _load_checkpoint_on_demotion(self) -> bool:
        """Charge un point de contrôle précédent en cas de rétrogradation.

        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        if not hasattr(self, 'checkpoint_history') or not self.checkpoint_history:
            logger.warning("No checkpoint history available for demotion")
            return False

        try:
            # Charger le dernier checkpoint
            last_checkpoint = self.checkpoint_history[-1]

            # Restaurer l'état du portefeuille
            if 'portfolio_state' in last_checkpoint:
                self.portfolio.set_state(last_checkpoint['portfolio_state'])

            # Restaurer l'état de l'environnement
            if 'env_state' in last_checkpoint:
                env_state = last_checkpoint['env_state']
                self.current_step = env_state.get('current_step', 0)
                self.current_chunk = env_state.get('current_chunk', 0)
                self.episode_count = env_state.get('episode_count', 0)
                self.episode_reward = env_state.get('episode_reward', 0.0)
                self.best_portfolio_value = env_state.get(
                    'best_portfolio_value',
                    self.portfolio.get_total_value()
                )

            # Restaurer les informations de palier
            if 'tier_info' in last_checkpoint:
                tier_info = last_checkpoint['tier_info']
                self.current_tier = next(
                    (t for t in self.tiers if t['name'] == tier_info.get('current_tier')),
                    self.tiers[0] if self.tiers else None
                )
                self.episodes_in_tier = tier_info.get('episodes_in_tier', 0)
                self.last_tier_change_step = tier_info.get('last_tier_change_step', 0)

            logger.info("Successfully loaded checkpoint after demotion")

            # Recharger les données du chunk actuel si nécessaire
            if hasattr(self, 'current_chunk'):
                self.current_data = self.data_loader.load_chunk(self.current_chunk)

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

        # Journalisation des composantes de la récompense
        self._last_reward_components = {
            "base_reward": float(base_reward),
            "risk_penalty": float(risk_penalty),
            "transaction_penalty": float(transaction_penalty),
            "concentration_penalty": float(concentration_penalty),
            "action_smoothness_penalty": float(action_smoothness_penalty),
            "total_reward": float(reward)
        }

        return float(reward)

    def _execute_trades(self, action: np.ndarray, dbe_modulation: Dict[str, float]) -> float:
        """
        Exécute les trades en fonction des actions du modèle et de la modulation DBE.

        Args:
            action: Vecteur d'actions normalisées [-1, 1] pour chaque actif
            dbe_modulation: Dictionnaire de modulation du DBE

        Returns:
            float: PnL réalisé lors de cette étape
        """
        if not hasattr(self, 'portfolio_manager'):
            self.logger.error("Portfolio manager non initialisé")
            return 0.0

        # Récupérer les prix actuels
        try:
            current_prices = self._get_current_prices()
            if not current_prices or not self._validate_market_data(current_prices):
                self.logger.warning("Données de marché invalides, aucun trade exécuté")
                return 0.0
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des prix: {str(e)}")
            return 0.0

        realized_pnl = 0.0
        trade_executed = False

        # Parcourir chaque actif et son action correspondante
        for i, (asset, price) in enumerate(current_prices.items()):
            if i >= len(action):
                break

            action_value = action[i]
            position = self.portfolio_manager.positions.get(asset)

            # Appliquer la modulation DBE si disponible
            if dbe_modulation and 'risk_multiplier' in dbe_modulation:
                action_value *= dbe_modulation['risk_multiplier']
                action_value = np.clip(action_value, -1.0, 1.0)

            try:
                # Décision de trading basée sur la valeur d'action
                if action_value > 0.1:  # Acheter
                    if position and position.is_open:
                        continue  # Position déjà ouverte

                    # Calculer la taille de la position basée sur le risque
                    risk_per_trade = 0.01  # 1% de risque par trade
                    stop_loss_pct = 0.02   # 2% de stop loss

                    # Log de la tentative d'ouverture
                    self.logger.debug(
                        "[TRADE ATTEMPT] asset=%s, action=BUY, price=%.8f, action_value=%.4f, "
                        "cash_available=%.8f",
                        asset, price, action_value, self.portfolio_manager.cash
                    )

                    # Calculer la taille de la position
                    position_size = self.portfolio_manager.calculate_position_size(
                        price=price,
                        stop_loss_pct=stop_loss_pct,
                        risk_per_trade=risk_per_trade,
                        account_risk_multiplier=1.0
                    )

                    # Ouvrir la position
                    if position_size > 0:
                        entry_price = price  # Prix d'entrée pour le log
                        self.portfolio_manager.open_position(asset, price, position_size)
                        trade_executed = True
                        # Log de l'exécution
                        self.logger.debug(
                            "[TRADE EXECUTED] asset=%s, action=BUY, entry_price=%.8f, "
                            "size=%.8f, value=%.8f, cash_after=%.8f",
                            asset, entry_price, position_size, position_size * entry_price,
                            self.portfolio_manager.cash
                        )
                    else:
                        self.logger.debug(
                            "[TRADE SKIPPED] asset=%s, reason=position_size=%.8f <= 0",
                            asset, position_size
                        )

                elif action_value < -0.1:  # Vendre
                    if position and position.is_open:
                        # Log de la tentative de fermeture
                        self.logger.debug(
                            "[TRADE ATTEMPT] asset=%s, action=SELL, price=%.8f, "
                            "entry_price=%.8f, size=%.8f, current_value=%.8f",
                            asset, price, position.entry_price, position.size,
                            position.size * price
                        )
                        # Fermer la position existante
                        pnl = self.portfolio_manager.close_position(asset, price)
                        if pnl is not None:
                            realized_pnl += pnl
                            trade_executed = True
                            # Log de l'exécution
                            self.logger.debug(
                                "[TRADE EXECUTED] asset=%s, action=SELL, exit_price=%.8f, "
                                "pnl=%.8f, realized_pnl=%.8f, cash_after=%.8f",
                                asset, price, pnl, realized_pnl, self.portfolio_manager.cash
                            )
                        else:
                            self.logger.warning(
                                "[TRADE FAILED] asset=%s, action=SELL, reason=close_position_returned_none",
                                asset
                            )

            except Exception as e:
                self.logger.error(
                    f"Erreur lors de l'exécution du trade pour {asset}: {str(e)}"
                )
                self._log_trade_error(asset, action_value, price, str(e))

        # Mettre à jour la valeur du portefeuille
        self.portfolio_manager.update_market_price(current_prices)

        # Mettre à jour l'étape du dernier trade si nécessaire
        if trade_executed:
            try:
                self.last_trade_step = int(self.current_step)
            except Exception:
                self.last_trade_step = 0

        # Vérifier les ordres de protection, mais seulement après la période de warmup
        # ou après l'exécution du premier trade pour éviter les arrêts prématurés
        warmup = getattr(self, "warmup_steps", 0) or 0
        if (self.current_step >= max(warmup, 0)) or (self.last_trade_step is not None and self.last_trade_step >= 0):
            self.portfolio_manager.check_protection_limits(current_prices)
            self.portfolio_manager.check_protection_orders(current_prices)

        return realized_pnl

    def _get_info(self) -> Dict[str, Any]:
        """
        Récupère des informations supplémentaires sur l'état de l'environnement.

        Returns:
            Dict[str, Any]: Dictionnaire contenant des informations détaillées sur l'état
                actuel du portefeuille et de l'environnement.
        """
        portfolio_metrics = self.portfolio_manager.get_metrics()
        current_prices = self._get_current_prices()
        position_values = {}
        total_position_value = 0.0

        # Calculer les valeurs des positions actuelles
        for asset, pos_info in portfolio_metrics.get("positions", {}).items():
            if asset in current_prices:
                qty = pos_info.get("size", pos_info.get("quantity", 0))  # Préférer 'size', avec fallback sur 'quantity' pour rétrocompatibilité
                price = current_prices[asset]
                value = qty * price
                position_values[asset] = {
                    "quantity": qty,
                    "price": price,
                    "value": value,
                    "weight": (
                        value / portfolio_metrics.get("total_value", 1.0)
                        if portfolio_metrics.get("total_value", 0) > 0
                        else 0.0
                    ),
                }
                total_position_value += value
        reward_components = {}
        if hasattr(self, "_last_reward_components"):
            reward_components = self._last_reward_components
        action_stats = {}
        if hasattr(self, "_last_action") and self._last_action is not None:
            action = self._last_action
            action_stats = {
                "action_mean": float(np.mean(action)),
                "action_std": float(np.std(action)),
                "action_min": float(np.min(action)),
                "action_max": float(np.max(action)),
                "num_assets": len(action),
            }
        info = {
            "step": self.current_step,
            "chunk": self.current_chunk,
            "done": self.done,
            "portfolio": {
                "total_value": portfolio_metrics.get("total_value", 0.0),
                "cash": portfolio_metrics.get("cash", 0.0),
                "returns": portfolio_metrics.get("returns", 0.0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0.0),
                "sortino_ratio": portfolio_metrics.get("sortino_ratio", 0.0),
                "total_position_value": total_position_value,
                "leverage": portfolio_metrics.get("leverage", 0.0),
                "num_positions": len(portfolio_metrics.get("positions", {})),
            },
            "positions": position_values,
            "market": {
                "num_assets": len(current_prices),
                "assets": list(current_prices.keys()),
                "current_prices": current_prices,
            },
            "action_stats": action_stats,
            "reward_components": reward_components,
            "performance": {
                "timestamp": self._get_safe_timestamp(),
                "steps_per_second": (
                    self.current_step
                    / max(0.0001, time.time() - self._episode_start_time)
                    if hasattr(self, "_episode_start_time")
                    else 0.0
                ),
            },
        }
        return info

    def render(self, mode: str = "human") -> None:
        """Affiche l'état actuel de l'environnement."""
        if mode == "human":
            portfolio_value = self.portfolio.get_portfolio_value()
            print(
                f"Étape: {self.current_step}, "
                f"Valeur du portefeuille: {portfolio_value:.2f}, "
                f"Espèces: {self.portfolio.cash:.2f}, "
                f"Positions: {self.portfolio.positions}"
            )

    def close(self) -> None:
        """Nettoie les ressources de l'environnement."""
        pass
