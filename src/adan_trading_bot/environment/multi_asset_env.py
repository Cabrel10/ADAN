"""
Multi-asset trading environment for the ADAN trading bot.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich import box
import os
import time

from ..common.utils import get_logger, ensure_dir_exists
from ..common.constants import (
    HOLD, BUY, SELL,
    ACTION_HOLD, ACTION_BUY_ASSET_0, ACTION_SELL_ASSET_0,
    ORDER_TYPE_MARKET, PENALTY_TIME
)
from .state_builder import StateBuilder
from .order_manager import OrderManager
from .reward_calculator import RewardCalculator

logger = get_logger()
console = Console()

class MultiAssetEnv(gym.Env):
    """
    Multi-asset trading environment for reinforcement learning.
    
    This environment simulates a trading scenario with multiple assets,
    allowing an agent to learn optimal trading strategies through interaction.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df_received, config, scaler=None, encoder=None, max_episode_steps_override=None):
        """
        Initialize the trading environment.
        
        Args:
            df_received: DataFrame with market data.
            config: Configuration dictionary.
            scaler: Optional pre-fitted scaler for market features.
            encoder: Optional pre-fitted encoder for dimensionality reduction.
            max_episode_steps_override: Optional override for max episode steps.
        """
        super(MultiAssetEnv, self).__init__()
        
        # Stocker la configuration
        self.config = config
        
        # Initialiser les assets AVANT toute autre chose
        self.assets = sorted(list(set(self.config.get('data', {}).get('assets', []))))
        if not self.assets:
            logger.critical("MultiAssetEnv __init__: ERREUR CRITIQUE - Aucun actif défini dans data_config['assets']!")
            raise ValueError("Aucun actif défini dans la configuration.")
        logger.info(f"MultiAssetEnv __init__: Utilisation des actifs de data_config: {self.assets}")
        
        # Logs détaillés pour le diagnostic du DataFrame reçu EN ARGUMENT
        logger.info(f"MultiAssetEnv __init__ - DataFrame REÇU EN ARGUMENT (df_received). Shape: {df_received.shape if df_received is not None else 'None'}")
        if df_received is not None and not df_received.empty:
            logger.info(f"MultiAssetEnv __init__ - Colonnes du DataFrame REÇU (premières 30): {df_received.columns.tolist()[:30]}")
            if not any(col.startswith('open_') and col.endswith(tuple(config.get('data', {}).get('assets', []))) for col in df_received.columns):
                logger.error("MultiAssetEnv __init__ - ERREUR CRITIQUE : Le DataFrame REÇU par __init__ ne semble PAS être fusionné (pas de colonnes comme open_ADAUSDT) !")
            else:
                logger.info("MultiAssetEnv __init__ - DataFrame REÇU semble correct (colonnes fusionnées).")
            
            # Vérifier la présence des colonnes clés pour les 2 premiers actifs
            for asset in self.assets[:2]:  # Juste vérifier les 2 premiers actifs pour éviter trop de logs
                logger.info(f"MultiAssetEnv __init__ - df_received contient-il 'open_{asset}' ? {'open_' + asset in df_received.columns}")
                logger.info(f"MultiAssetEnv __init__ - df_received contient-il 'close_{asset}' ? {'close_' + asset in df_received.columns}")
        else:
            logger.critical("MultiAssetEnv __init__ - ERREUR CRITIQUE : DataFrame REÇU est vide ou None!")
            # Lever une exception pour arrêter proprement
            raise ValueError("MultiAssetEnv a reçu un DataFrame vide ou None.")
        
        # Stocker le DataFrame (travailler sur une copie pour éviter les modifications inattendues)
        self.df = df_received.copy()
        
        # Logs détaillés pour le diagnostic du DataFrame APRÈS affectation
        logger.info(f"MultiAssetEnv __init__ - self.df APRÈS copie. Shape: {self.df.shape if self.df is not None else 'None'}")
        if self.df is not None and not self.df.empty:
            logger.info(f"MultiAssetEnv __init__ - self.df.shape: {self.df.shape}")
            logger.info(f"MultiAssetEnv __init__ - Premières colonnes (10 max): {self.df.columns.tolist()[:10]}")
            logger.info(f"MultiAssetEnv __init__ - Aperçu des 2 premières lignes:")
            for index, row in self.df.head(2).iterrows():
                logger.info(f"  Index: {index}, Premières valeurs (5 max): {[str(val)[:15] for val in row.values[:5]]}")
        else:
            logger.critical("MultiAssetEnv __init__ - ERREUR CRITIQUE - self.df APRÈS copie est vide ou None!")
        
        self.scaler = scaler
        self.encoder = encoder
        
        # Environment configuration
        env_config = config.get('environment', {})
        
        # Charger les configurations depuis le fichier environment_config.yaml
        # Capital initial
        self.initial_capital = env_config.get('initial_capital', 10000.0)
        
        # Configurations des transactions, ordres et pénalités
        self.transaction_config = env_config.get('transaction', {})
        self.order_rules_config = env_config.get('order_rules', {})
        self.penalties_config = env_config.get('penalties', {})
        
        # Charger les paramètres de transaction
        self.fee_percent = self.transaction_config.get('fee_percent', 0.001)
        self.fixed_fee = self.transaction_config.get('fixed_fee', 0.0)
        
        # Charger les règles d'ordre
        self.min_order_value_tolerable = self.order_rules_config.get('min_value_tolerable', 10.0)
        self.min_order_value_absolute = self.order_rules_config.get('min_value_absolute', 9.0)
        
        # Afficher les premières colonnes du DataFrame pour le débogage
        logger.info(f"Colonnes disponibles dans le DataFrame fusionné (premières 15): {self.df.columns.tolist()[:15]}")
        
        # Vérification de la présence des colonnes attendues pour les actifs
        expected_columns = []
        for asset in self.assets:
            expected_columns.extend([f"open_{asset}", f"high_{asset}", f"low_{asset}", f"close_{asset}"])
        
        found_columns = [col for col in expected_columns if col in self.df.columns]
        missing_columns = [col for col in expected_columns if col not in self.df.columns]
        
        if found_columns:
            logger.info(f"Trouvé {len(found_columns)}/{len(expected_columns)} colonnes attendues pour les actifs")
            logger.info(f"Exemples de colonnes trouvées: {found_columns[:5]}")
        
        if missing_columns:
            logger.error(f"ATTENTION: {len(missing_columns)} colonnes attendues sont MANQUANTES!")
            logger.error(f"Exemples de colonnes manquantes: {missing_columns[:10]}")
            logger.error("Le DataFrame ne contient pas toutes les colonnes attendues pour les actifs définis dans la configuration.")
            # Ne pas lever d'exception ici pour permettre l'utilisation de notre solution temporaire
            logger.warning(f"Continuons avec les actifs définis dans la configuration: {self.assets}")
        
        # Limit to 5 assets maximum (for the 11 discrete actions)
        if len(self.assets) > 5:
            logger.warning(f"More than 5 assets found ({len(self.assets)}). Using only the first 5: {self.assets[:5]}")
            self.assets = self.assets[:5]
        elif not self.assets:  # Si la liste est vide pour une raison quelconque
            logger.error("CRITICAL: No assets defined or detected. Defaulting to 5 placeholder assets.")
            self.assets = [f'ASSET_{i}' for i in range(5)]
        
        logger.info(f"Final assets being used by environment: {self.assets}")
        
        # Get CNN configuration from data_config
        data_config = config.get('data', {})
        self.cnn_input_window_size = data_config.get('cnn_input_window_size', 20)
        
        # Get training timeframe
        self.training_timeframe = data_config.get('training_timeframe', '1h')
        logger.info(f"Training timeframe: {self.training_timeframe}")
        
        # Get base market features
        self.base_feature_names = data_config.get('base_market_features', 
                                                 ['open', 'high', 'low', 'close', 'volume', 'macd'])
        
        # Log des noms de base des features pour le diagnostic
        logger.info(f"Base feature names from config: {self.base_feature_names}")
        
        # Stockage de la configuration des indicateurs par timeframe pour référence future
        # mais nous utilisons directement base_market_features comme source de vérité
        self.indicators_by_timeframe = data_config.get('indicators_by_timeframe', {})
        
        # Note: Les indicateurs spécifiques au timeframe sont maintenant directement inclus
        # dans base_market_features dans le fichier de configuration data_config_{profile}.yaml
        # et n'ont plus besoin d'être ajoutés dynamiquement ici
        
        # Determine the number of market features per step
        self.num_market_features_per_step = len(self.base_feature_names) * len(self.assets)
        
        # Define the image shape for CNN
        self.num_input_channels = 1  # Default to 1 channel
        self.image_shape = (self.num_input_channels, self.cnn_input_window_size, self.num_market_features_per_step)
        
        logger.info(f"CNN input shape: {self.image_shape}")
        logger.info(f"Base feature names: {self.base_feature_names}")
        
        # Initialize components
        self.state_builder = StateBuilder(config, self.assets, scaler, encoder, 
                                         base_feature_names=self.base_feature_names, 
                                         cnn_input_window_size=self.cnn_input_window_size)
        self.order_manager = OrderManager(config)
        self.reward_calculator = RewardCalculator(config)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(1 + 2 * len(self.assets))  # HOLD + BUY/SELL for each asset
        
        # Get observation space dimensions
        obs_space_dims = self.state_builder.get_observation_space_dim()
        
        # Define observation space as a dictionary
        self.observation_space = spaces.Dict({
            "image_features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=obs_space_dims["image_features"], 
                dtype=np.float32
            ),
            "vector_features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(obs_space_dims["vector_features"],), 
                dtype=np.float32
            )
        })
        
        # Initialize state variables (will be reset before use)
        self.capital = self.initial_capital
        self.positions = {}  # {asset_id: {"qty": quantity, "price": price}}
        self.current_step = 0
        self.history = []
        self.trade_log = []
        self.cumulative_reward = 0.0
        
        # Determine the number of steps in an episode
        self.max_steps = len(self.df) if 'pair' not in self.df.columns else len(self.df) // len(self.assets)
        
        # Override max steps if provided
        if max_episode_steps_override is not None:
            self.max_steps = min(max_episode_steps_override, self.max_steps)
            logger.info(f"Max episode steps overridden to: {self.max_steps}")
        
        # Export settings
        self.export_history = env_config.get('export_history', True)
        self.export_dir = env_config.get('export_dir', None)
        
        logger.info(f"MultiAssetEnv initialized with {len(self.assets)} assets and {self.max_steps} steps")
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.capital = self.initial_capital
        self.positions = {}
        self.current_step = 0
        self.history = []
        self.trade_log = []
        self.cumulative_reward = 0.0
        
        # Clear pending orders
        self.order_manager.clear_pending_orders()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial info
        info = {
            "portfolio_value": self.initial_capital,
            "capital": self.capital,
            "positions": self.positions.copy(),
            "step": self.current_step
        }
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment by executing the given action.
        
        Args:
            action: Action to take (index in the action space).
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure action is valid
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Store portfolio value before action
        old_portfolio_value = self._calculate_portfolio_value()
        
        # Get current prices for all assets
        current_prices = self._get_current_prices()
        
        # Process pending orders
        reward_mod, executed_orders, self.capital, self.positions = self.order_manager.process_pending_orders(
            current_prices, self.capital, self.positions, self.current_step
        )
        
        # Log executed orders
        for order_info in executed_orders:
            self.trade_log.append({
                "step": self.current_step,
                "type": "EXECUTED_ORDER",
                **order_info
            })
        
        # Translate action to order parameters
        asset_id, action_type = self._translate_action(action)
        
        # Execute the action if not HOLD
        trade_info = {}
        if action_type != HOLD:
            # Get current price for the asset
            if asset_id in current_prices:
                current_price = current_prices[asset_id]
                
                # Get current tier
                current_tier = self.reward_calculator.get_current_tier(self.capital)
                logger.info(f"[step] Action: {action}, Asset: {asset_id}, Action Type: {action_type}")
                logger.info(f"[step] Current capital: ${self.capital:.2f}, Current price: ${current_price:.4f}")
                logger.info(f"[step] Current tier: {current_tier}")
                logger.info(f"[step] Current positions: {self.positions}")
                
                # Check max positions constraint for BUY
                if action_type == BUY:
                    if len(self.positions) >= current_tier["max_positions"] and asset_id not in self.positions:
                        # Can't open new position, max reached
                        logger.warning(f"[step] Cannot open new position: max positions ({current_tier['max_positions']}) reached")
                        # Utiliser la pénalité configurée pour max_positions_reached
                        penalty = self.penalties_config.get('max_positions_reached', -0.2)
                        reward_mod_action, status, trade_info = penalty, "INVALID_MAX_POSITIONS", {
                            "reason": f"Max positions ({current_tier['max_positions']}) reached",
                            "reward_mod": penalty
                        }
                    else:
                        # Execute BUY order
                        logger.info(f"[step] Calculating allocation value for {asset_id}")
                        allocated_value_usdt = self._get_position_size(asset_id, current_price, current_tier)
                        logger.info(f"[step] Allocated value: ${allocated_value_usdt:.2f}")
                        
                        reward_mod_action, status, trade_info = self.order_manager.execute_order(
                            asset_id, action_type, current_price, self.capital, self.positions,
                            allocated_value_usdt=allocated_value_usdt, order_type=ORDER_TYPE_MARKET, current_step=self.current_step
                        )
                        logger.info(f"[step] Order execution result: status={status}, reward_mod={reward_mod_action}")
                        
                        # Update capital if order was successful
                        if status == "BUY_EXECUTED":
                            old_capital = self.capital
                            self.capital = self.capital - trade_info["total_cost"]
                            logger.info(f"[step] BUY executed: capital ${old_capital:.2f} -> ${self.capital:.2f}, cost: ${trade_info['total_cost']:.2f}")
                            logger.info(f"[step] New position: {asset_id}, quantity: {trade_info['quantity']:.6f}, price: ${trade_info['price']:.4f}")
                else:
                    # Execute SELL order
                    reward_mod_action, status, trade_info = self.order_manager.execute_order(
                        asset_id, action_type, current_price, self.capital, self.positions,
                        order_type=ORDER_TYPE_MARKET, current_step=self.current_step
                    )
                    
                    # Update capital if order was successful
                    if status == "SELL_EXECUTED":
                        old_capital = self.capital
                        self.capital = self.capital + trade_info["value"] - trade_info["fee"]
                        logger.info(f"[step] SELL executed: capital ${old_capital:.2f} -> ${self.capital:.2f}, value: ${trade_info['value']:.2f}, fee: ${trade_info['fee']:.2f}")
                        logger.info(f"[step] Position closed: {asset_id}")
                
                # Add to reward modifier
                reward_mod += reward_mod_action
                
                # Log the trade
                self.trade_log.append({
                    "step": self.current_step,
                    "action": action,
                    "action_type": action_type,
                    "asset_id": asset_id,
                    "status": status,
                    "reward_mod": reward_mod_action,
                    **trade_info
                })
            else:
                # Asset price not available
                logger.warning(f"[step] Cannot execute order: price not available for {asset_id}")
                # Utiliser la pénalité configurée pour price_not_available
                penalty = self.penalties_config.get('price_not_available', -0.1)
                reward_mod_action, status, trade_info = penalty, "PRICE_NOT_AVAILABLE", {
                    "reason": f"Price not available for {asset_id}",
                    "reward_mod": penalty
                }
        else:
            # HOLD action
            # Appliquer une petite pénalité pour chaque pas de temps (time_step penalty)
            reward_mod += self.penalties_config.get('time_step', -0.001)
            trade_info = {"reason": "HOLD action"}
        
        # Calculate new portfolio value
        new_portfolio_value = self._calculate_portfolio_value()
        
        # Calculate reward
        current_tier = self.reward_calculator.get_current_tier(self.capital)
        reward = self.reward_calculator.calculate_reward(
            old_portfolio_value, new_portfolio_value, penalties=-reward_mod, tier=current_tier
        )
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = False
        
        # Bankruptcy check (capital too low and no positions)
        if self.capital <= self.min_order_value_absolute and not self.positions:
            done = True
        
        # End of data check
        if self.current_step >= self.max_steps - 1:
            done = True
        
        # Record state in history
        self.history.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "portfolio_value": new_portfolio_value,
            "capital": self.capital,
            "positions": self.positions.copy(),
            "old_portfolio_value": old_portfolio_value,
            "reward_mod": reward_mod,
            "done": done
        })
        
        # Display trading table
        self._display_trading_table(
            action, old_portfolio_value, new_portfolio_value, 
            reward, reward_mod, trade_info
        )
        
        # Prepare info dictionary
        info = {
            "portfolio_value": new_portfolio_value,
            "capital": self.capital,
            "positions": self.positions.copy(),
            "step": self.current_step,
            "reward_mod": reward_mod,
            "trade_info": trade_info,
            "tier": current_tier
        }
        
        # Increment step counter
        self.current_step += 1
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            dict: Observation dictionary with 'image_features' and 'vector_features'.
        """
        # Get market data window for CNN
        market_data_window = self._get_market_data_window()
        
        # Check if we have data in the window
        if market_data_window.empty:
            logger.warning(f"Empty market data window at step {self.current_step}. Using zeros for observation.")
            # Create a dummy observation with zeros
            image_features = np.zeros(self.image_shape, dtype=np.float32)
            vector_features = np.zeros(1 + len(self.assets), dtype=np.float32)
            # Set normalized capital to 1.0 (initial state)
            vector_features[0] = 1.0
            
            return {
                "image_features": image_features,
                "vector_features": vector_features
            }
        
        # Build observation using state builder
        observation = self.state_builder.build_observation(
            market_data_window, 
            self.capital, 
            self.positions,
            self.image_shape
        )
        
        return observation
        
    def _get_market_data_window(self):
        """
        Get a window of market data for the CNN.
        
        Returns:
            pandas.DataFrame: Window of market data.
        """
        # Calculate the start index for the window
        start_idx = max(0, self.current_step - self.cnn_input_window_size + 1)
        end_idx = self.current_step + 1  # +1 because end index is exclusive
        
        # Extract the window from the dataframe
        window = self.df.iloc[start_idx:end_idx].copy()
        
        # Log the columns in the window for debugging (toujours, pas seulement en DEBUG)
        logger.info(f"_get_market_data_window: Shape de la fenêtre: {window.shape}")
        logger.info(f"_get_market_data_window: Premières colonnes (10 max): {window.columns.tolist()[:10]}")
        logger.info(f"_get_market_data_window: Index de la fenêtre (type): {type(window.index)}")
        
        # Vérifier que les colonnes contiennent bien les noms d'actifs dans le format attendu
        expected_columns = []
        for asset in self.assets:
            for base_feature in self.base_feature_names:
                expected_columns.append(f"{base_feature}_{asset}")
        
        # Vérifier les colonnes attendues
        found_columns = [col for col in expected_columns if col in window.columns]
        missing_columns = [col for col in expected_columns if col not in window.columns]
        
        if found_columns:
            logger.info(f"Trouvé {len(found_columns)}/{len(expected_columns)} colonnes attendues")
            logger.info(f"Exemples de colonnes trouvées: {found_columns[:5]}")
        
        if missing_columns:
            logger.error(f"ATTENTION: {len(missing_columns)} colonnes attendues sont MANQUANTES!")
            logger.error(f"Exemples de colonnes manquantes: {missing_columns[:10]}")
        
        # Vérifier les colonnes avec noms d'actifs (méthode générale)
        asset_columns = [col for col in window.columns if any(asset in col for asset in self.assets)]
        if not asset_columns:
            logger.error(f"ERREUR CRITIQUE: Aucune colonne avec les noms d'actifs trouvée dans market_data_window!")
            logger.error(f"Assets attendus: {self.assets}")
            logger.error(f"Colonnes disponibles (premières 15): {window.columns.tolist()[:15]}")
        else:
            logger.info(f"Trouvé {len(asset_columns)} colonnes contenant les noms d'actifs sur {len(window.columns)} colonnes totales")
            logger.info(f"Exemples de colonnes avec actifs: {asset_columns[:5]}")
        
        # Log specific information about timeframe indicators
        timeframe_columns = [col for col in window.columns if self.training_timeframe in col]
        if timeframe_columns:
            logger.info(f"Trouvé {len(timeframe_columns)} colonnes avec timeframe {self.training_timeframe}")
            logger.info(f"Exemples: {timeframe_columns[:5]}")
        
        # Vérification finale des colonnes de la fenêtre retournée
        logger.info(f"MultiAssetEnv _get_market_data_window - Colonnes de la 'window' retournée (premières 30): {window.columns.tolist()[:30] if window is not None and not window.empty else 'Window vide/None'}")
        if window is not None and not window.empty and not any(col.startswith('open_') and col.endswith(tuple(self.assets)) for col in window.columns):
            logger.error("MultiAssetEnv _get_market_data_window - ERREUR : La 'window' retournée n'a PAS les colonnes fusionnées !")
        elif window is not None and not window.empty:
            logger.info("MultiAssetEnv _get_market_data_window - 'window' retournée semble avoir les colonnes fusionnées.")
        
        return window
    
    def _get_current_data_row(self):
        """
        Get the current data row based on the current step.
        
        Returns:
            pandas.Series: Current data row.
        """
        # For merged data, each row already contains data for all assets
        data_row = self.df.iloc[self.current_step]
        
        # Log détaillé pour le diagnostic
        logger.info(f"MultiAssetEnv _get_current_data_row - Type de data_row: {type(data_row)}")
        logger.info(f"MultiAssetEnv _get_current_data_row - Index de data_row (premières 30): {data_row.index.tolist()[:30] if data_row is not None and not data_row.empty else 'data_row vide/None'}")
        
        # Vérification de la présence des colonnes fusionnées dans l'index
        if data_row is not None and not any(idx.startswith('open_') and idx.endswith(tuple(self.assets)) for idx in data_row.index):
            logger.error("MultiAssetEnv _get_current_data_row - ERREUR : L'index de 'data_row' n'a PAS les noms de colonnes fusionnées !")
        elif data_row is not None:
            logger.info("MultiAssetEnv _get_current_data_row - L'index de 'data_row' semble avoir les colonnes fusionnées.")
        
        return data_row
    
    def _get_current_prices(self):
        """
        Get current prices for all assets.
        
        Returns:
            dict: Dictionary mapping asset IDs to current prices.
        """
        current_prices = {}
        
        # Get current data row
        data_row = self._get_current_data_row()
        
        # Log les colonnes disponibles dans data_row pour le débogage
        logger.info(f"MultiAssetEnv _get_current_prices - Index de data_row REÇU (premières 30): {data_row.index.tolist()[:30] if data_row is not None else 'data_row vide/None'}")
        
        # Détecter le format des données
        format_multi_asset = any(f"_{asset}" in col for col in data_row.index for asset in self.assets)
        format_with_asset_column = 'asset' in data_row
        
        logger.info(f"_get_current_prices: Format multi-actif détecté: {format_multi_asset}")
        logger.info(f"_get_current_prices: Format avec colonne 'asset' détecté: {format_with_asset_column}")
        
        # Vérifier le nombre de prix que nous sommes capables de trouver
        found_prices = 0
        missing_prices = []
        
        # Traitement selon le format détecté
        for asset in self.assets:
            # Format principal: multi-actif avec colonnes comme 'close_ADAUSDT'
            close_col = f'close_{asset}'
            if format_multi_asset and close_col in data_row:
                price = data_row[close_col]
                logger.info(f"MultiAssetEnv _get_current_prices - Prix trouvé pour {asset} via '{close_col}': {price}")
                current_prices[asset] = price
                found_prices += 1
                continue
            
            # Essayer de trouver une colonne alternative si le nom exact n'a pas été trouvé
            if format_multi_asset:
                possible_close_columns = [col for col in data_row.index if col.startswith('close_') and asset in col]
                if possible_close_columns:
                    chosen_col = possible_close_columns[0]
                    price = data_row[chosen_col]
                    logger.info(f"MultiAssetEnv _get_current_prices - Prix trouvé pour {asset} via '{chosen_col}' (alternative): {price}")
                    current_prices[asset] = price
                    found_prices += 1
                    continue
                else:
                    logger.error(f"MultiAssetEnv _get_current_prices - ERREUR : Aucune colonne 'close_*' trouvée pour {asset}")
                    logger.error(f"MultiAssetEnv _get_current_prices - Colonnes disponibles (extrait): {sorted([col for col in data_row.index if 'close' in col.lower()][:10])}")
                    missing_prices.append(asset)
                    # NE PAS utiliser de valeur par défaut, retourner np.nan pour ce prix
                    current_prices[asset] = np.nan
                    continue
        
        # Générer un message récapitulatif
        if found_prices == len(self.assets):
            logger.info(f"MultiAssetEnv _get_current_prices - SUCCÈS: Tous les prix ({found_prices}/{len(self.assets)}) ont été trouvés correctement")
        elif found_prices > 0:
            logger.warning(f"MultiAssetEnv _get_current_prices - INCOMPLET: Seulement {found_prices}/{len(self.assets)} prix trouvés.")
            logger.warning(f"MultiAssetEnv _get_current_prices - Prix manquants pour: {missing_prices}")
        else:
            logger.error(f"MultiAssetEnv _get_current_prices - ÉCHEC TOTAL: Aucun prix ({found_prices}/{len(self.assets)}) n'a été trouvé!")
            logger.error(f"MultiAssetEnv _get_current_prices - Vérifiez que les fichiers fusionnés ont bien été créés et contiennent les bonnes colonnes close_ASSET")
            logger.error(f"MultiAssetEnv _get_current_prices - Le format de données multi-actif est-il détecté? {format_multi_asset}")
        
        # Log summary of found prices
        found_prices = {k: v for k, v in current_prices.items() if not np.isnan(v)}
        logger.info(f"MultiAssetEnv _get_current_prices - Résumé des prix trouvés: {len(found_prices)}/{len(self.assets)} actifs")
        logger.info(f"MultiAssetEnv _get_current_prices - Tous les prix: {current_prices}")
        
        # Vérification des prix manquants
        missing_prices = {k: v for k, v in current_prices.items() if np.isnan(v)}
        if missing_prices:
            logger.error(f"MultiAssetEnv _get_current_prices - ERREUR CRITIQUE : Prix manquants pour {len(missing_prices)} actifs: {list(missing_prices.keys())}")
        
        return current_prices
    
    def _translate_action(self, action):
        """
        Translate action index to asset ID and action type.
        
        Args:
            action: Action index.
        
        Returns:
            tuple: (asset_id, action_type)
        """
        if action == ACTION_HOLD:
            return None, HOLD
        
        num_assets = len(self.assets)
        
        if ACTION_BUY_ASSET_0 <= action < ACTION_BUY_ASSET_0 + num_assets:
            # BUY action
            asset_idx = action - ACTION_BUY_ASSET_0
            return self.assets[asset_idx], BUY
        
        if ACTION_SELL_ASSET_0 <= action < ACTION_SELL_ASSET_0 + num_assets:
            # SELL action
            asset_idx = action - ACTION_SELL_ASSET_0
            return self.assets[asset_idx], SELL
        
        # Should not reach here
        logger.error(f"Invalid action: {action}")
        return None, HOLD
    
    def _calculate_portfolio_value(self):
        """
        Calculate the current portfolio value (capital + positions).
        
        Returns:
            float: Total portfolio value.
        """
        # Start with available capital (limité pour éviter les overflows)
        MAX_CAPITAL = 1e6  # 1 million USD maximum
        capped_capital = min(self.capital, MAX_CAPITAL)
        total_value = capped_capital
        
        # Add value of all positions
        current_prices = self._get_current_prices()
        for asset, position in self.positions.items():
            if asset in current_prices:
                # Limiter la taille des positions pour éviter les overflows
                MAX_POSITION_SIZE = 1e8  # 100 millions de jetons maximum
                position_qty = min(position['qty'], MAX_POSITION_SIZE)
                position_value = position_qty * current_prices[asset]
                
                # Limiter la valeur de la position pour éviter les overflows
                MAX_POSITION_VALUE = 1e6  # 1 million USD maximum par position
                capped_position_value = min(position_value, MAX_POSITION_VALUE)
                total_value += capped_position_value
        
        return total_value
    
    def _get_position_size(self, asset_id, price, current_tier):
        """
        Calculate the position size for a given asset based on the current tier.
        
        Args:
            asset_id: Asset ID.
            price: Current price of the asset.
            current_tier: Current tier configuration.
            
        Returns:
            float: Position size in USDT.
        """
        # Log input parameters for debugging
        logger.info(f"[_get_position_size] Input parameters: asset_id={asset_id}, price=${price:.7f}")
        logger.info(f"[_get_position_size] Tier details: {current_tier}")
        
        # Get allocation fraction from tier
        allocation_frac = current_tier.get('allocation_frac_per_pos', 0.95)
        logger.info(f"[_get_position_size] Allocation fraction from tier: {allocation_frac:.7f}")
        
        # Limiter le capital à une valeur raisonnable pour éviter les overflows
        MAX_CAPITAL = 1e6  # 1 million USD maximum pour les calculs d'allocation
        capped_capital = min(self.capital, MAX_CAPITAL)
        
        # Calculate allocated value
        allocated_value_usdt = capped_capital * allocation_frac
        
        # Log allocation details
        logger.info(f"[_get_position_size] Capital original: ${self.capital:.7f}, capital plafonné: ${capped_capital:.7f}")
        logger.info(f"[_get_position_size] Allocation_frac: {allocation_frac:.7f}, allocated_value_usdt: ${allocated_value_usdt:.7f}")
        logger.info(f"[_get_position_size] Asset: {asset_id}, price: ${price:.7f}, min_order_value_tolerable: ${self.min_order_value_tolerable:.4f}, min_order_value_absolute: ${self.min_order_value_absolute:.4f}")
        
        return allocated_value_usdt
    
    def _interpret_action_for_display(self, action):
        """
        Traduit l'action numérique en une chaîne de caractères lisible.
        
        Args:
            action: Action numérique.
            
        Returns:
            str: Chaîne de caractères décrivant l'action.
        """
        if action == ACTION_HOLD:
            return "[bold yellow]HOLD[/bold yellow]"
        
        num_assets = len(self.assets)
        
        if ACTION_BUY_ASSET_0 <= action < ACTION_BUY_ASSET_0 + num_assets:
            # BUY action
            asset_idx = action - ACTION_BUY_ASSET_0
            asset_id = self.assets[asset_idx]
            return f"[bold green]BUY[/bold green] {asset_id}"
        
        if ACTION_SELL_ASSET_0 <= action < ACTION_SELL_ASSET_0 + num_assets:
            # SELL action
            asset_idx = action - ACTION_SELL_ASSET_0
            asset_id = self.assets[asset_idx]
            return f"[bold red]SELL[/bold red] {asset_id}"
        
        return f"[bold red]UNKNOWN ACTION: {action}[/bold red]"
    
    def _display_trading_table(self, action, old_value, new_value, reward, reward_mod, trade_info):
        """
        Display a trading table with current state information.
        
        Args:
            action: Action taken.
            old_value: Portfolio value before action.
            new_value: Portfolio value after action.
            reward: Reward received.
            reward_mod: Reward modifier.
            trade_info: Trade information.
        """
        # Obtenir l'interprétation de l'action
        action_str = self._interpret_action_for_display(action)
        
        # Créer un titre pour la table principale
        title = f"[bold blue]Step {self.current_step}/{self.max_steps}[/bold blue]"
        
        # Créer un panneau pour l'action
        action_panel = Panel.fit(
            action_str,
            title="Action",
            border_style="green" if ACTION_BUY_ASSET_0 <= action < ACTION_BUY_ASSET_0 + len(self.assets) else 
                      "red" if ACTION_SELL_ASSET_0 <= action < ACTION_SELL_ASSET_0 + len(self.assets) else "yellow"
        )
        
        # Afficher le panneau d'action
        console.print(Rule(title))
        console.print(action_panel)
        
        # Créer une table pour la performance du portefeuille
        portfolio_table = Table(title="Performance du Portefeuille", show_header=True, box=box.ROUNDED)
        portfolio_table.add_column("Métrique", style="cyan")
        portfolio_table.add_column("Avant", style="yellow")
        portfolio_table.add_column("Après", style="green")
        portfolio_table.add_column("Variation", style="magenta")
        
        # Calculer les variations de valeur
        value_change = new_value - old_value
        value_change_pct = (value_change / old_value) * 100 if old_value > 0 else 0
        value_change_style = "green" if value_change >= 0 else "red"
        
        # Ajouter les lignes de performance
        portfolio_table.add_row(
            "Valeur Totale",
            f"${old_value:.2f}",
            f"${new_value:.2f}",
            Text(f"${value_change:.2f} ({value_change_pct:.2f}%)", style=value_change_style)
        )
        
        # Ajouter la ligne de capital
        capital_before = self.history[-1]['capital'] if self.history else self.initial_capital
        capital_change = self.capital - capital_before
        capital_change_style = "green" if capital_change >= 0 else "red"
        portfolio_table.add_row(
            "Capital Disponible",
            f"${capital_before:.2f}",
            f"${self.capital:.2f}",
            Text(f"${capital_change:.2f}", style=capital_change_style)
        )
        
        # Afficher la table de performance
        console.print(portfolio_table)
        
        # Créer une table pour les positions actives
        positions_table = Table(title="Positions Actives", show_header=True, box=box.ROUNDED)
        positions_table.add_column("Actif", style="cyan")
        positions_table.add_column("Quantité", style="yellow")
        positions_table.add_column("Prix d'Entrée", style="green")
        positions_table.add_column("Prix Actuel", style="blue")
        positions_table.add_column("Valeur", style="magenta")
        positions_table.add_column("PnL Latent", style="white")
        
        # Obtenir les prix actuels
        current_prices = self._get_current_prices()
        
        # Ajouter les positions à la table
        if self.positions:
            for asset_id, position in self.positions.items():
                current_price = current_prices.get(asset_id, position["price"])
                position_value = position["qty"] * current_price
                pnl = position["qty"] * (current_price - position["price"])
                pnl_pct = (pnl / (position["qty"] * position["price"])) * 100 if position["qty"] * position["price"] > 0 else 0
                pnl_style = "green" if pnl >= 0 else "red"
                
                positions_table.add_row(
                    asset_id,
                    f"{position['qty']:.6f}",
                    f"${position['price']:.4f}",
                    f"${current_price:.4f}",
                    f"${position_value:.2f}",
                    Text(f"${pnl:.2f} ({pnl_pct:.2f}%)", style=pnl_style)
                )
        else:
            positions_table.add_row("Aucune position active", "", "", "", "", "")
        
        # Afficher la table des positions
        console.print(positions_table)
        
        # Créer une table pour les trades exécutés
        if trade_info and 'status' in trade_info:
            trades_table = Table(title="Trade Exécuté", show_header=True, box=box.ROUNDED)
            trades_table.add_column("Statut", style="cyan")
            trades_table.add_column("Type", style="yellow")
            trades_table.add_column("Actif", style="green")
            trades_table.add_column("Quantité", style="blue")
            trades_table.add_column("Prix", style="magenta")
            trades_table.add_column("Valeur", style="white")
            trades_table.add_column("Frais", style="red")
            
            # Déterminer le style du statut
            status_style = "green" if trade_info['status'].endswith("_EXECUTED") else "red"
            
            # Extraire les informations du trade
            asset_id = trade_info.get('asset_id', 'N/A')
            action_type = trade_info.get('action_type', 'N/A')
            quantity = trade_info.get('quantity', 0)
            price = trade_info.get('price', 0)
            value = trade_info.get('value', trade_info.get('total_cost', 0))
            fee = trade_info.get('fee', 0)
            
            # Ajouter la ligne à la table
            trades_table.add_row(
                Text(trade_info['status'], style=status_style),
                "BUY" if action_type == BUY else "SELL" if action_type == SELL else "N/A",
                asset_id,
                f"{quantity:.6f}" if quantity else "N/A",
                f"${price:.4f}" if price else "N/A",
                f"${value:.2f}" if value else "N/A",
                f"${fee:.2f}" if fee else "N/A"
            )
            
            # Afficher la table des trades
            console.print(trades_table)
            
            # Si le trade a échoué, afficher la raison
            if 'reason' in trade_info and not trade_info['status'].endswith("_EXECUTED"):
                console.print(Panel(f"[bold red]Raison: {trade_info['reason']}[/bold red]", title="Échec du Trade"))
        
        # Créer une table pour les ordres en attente
        if hasattr(self.order_manager, 'pending_orders') and self.order_manager.pending_orders:
            pending_orders_table = Table(title="Ordres en Attente", show_header=True, box=box.ROUNDED)
            pending_orders_table.add_column("ID", style="cyan")
            pending_orders_table.add_column("Type", style="yellow")
            pending_orders_table.add_column("Actif", style="green")
            pending_orders_table.add_column("Quantité", style="blue")
            pending_orders_table.add_column("Prix", style="magenta")
            pending_orders_table.add_column("Expiration", style="white")
            
            # Ajouter les ordres en attente à la table
            for order_id, order in self.order_manager.pending_orders.items():
                pending_orders_table.add_row(
                    str(order_id),
                    order.get('action_type_str', 'BUY' if order.get('action_type') == BUY else 'SELL' if order.get('action_type') == SELL else 'N/A'),
                    order.get('asset_id', 'N/A'),
                    f"{order.get('quantity', 0):.6f}",
                    f"${order.get('price', 0):.4f}",
                    f"Step {order.get('expiration', 'N/A')}"
                )
            
            # Afficher la table des ordres en attente
            console.print(pending_orders_table)
        
        # Créer une table pour la récompense
        reward_table = Table(title="Récompense", show_header=True, box=box.ROUNDED)
        reward_table.add_column("Type", style="cyan")
        reward_table.add_column("Valeur", style="yellow")
        
        # Déterminer les styles de récompense
        reward_style = "green" if reward >= 0 else "red"
        cum_reward_style = "green" if self.cumulative_reward >= 0 else "red"
        
        # Ajouter les lignes de récompense
        reward_table.add_row("Modificateur", f"{reward_mod:.6f}")
        reward_table.add_row("Récompense Nette", Text(f"{reward:.6f}", style=reward_style))
        reward_table.add_row("Récompense Cumulative", Text(f"{self.cumulative_reward:.6f}", style=cum_reward_style))
        
        # Afficher la table de récompense
        console.print(reward_table)
        
        # Créer une table pour les informations du palier actuel
        current_tier = self.reward_calculator.get_current_tier(self.capital)
        tier_table = Table(title="Palier Actuel", show_header=True, box=box.ROUNDED)
        tier_table.add_column("Paramètre", style="cyan")
        tier_table.add_column("Valeur", style="yellow")
        
        # Debug: Loguer les détails du palier actuel
        logger.info(f"[_display_trading_table] Capital actuel: ${self.capital:.2f}")
        logger.info(f"[_display_trading_table] Palier actuel: {current_tier}")
        alloc_frac_debug = current_tier.get('allocation_frac_per_pos', -1.0)  # -1.0 pour signaler une erreur si la clé manque
        logger.info(f"[_display_trading_table] allocation_frac_per_pos: {alloc_frac_debug:.4f}")
        
        # Ajouter les informations du palier
        tier_table.add_row("Seuil", f"${current_tier.get('threshold', 0):.2f}")
        tier_table.add_row("Positions Max", str(current_tier.get('max_positions', 0)))
        tier_table.add_row("Allocation par Position", f"{current_tier.get('allocation_frac_per_pos', 0) * 100:.2f}%")
        tier_table.add_row("Multiplicateur Récompense", f"{current_tier.get('reward_pos_mult', 1):.2f}x")
        tier_table.add_row("DEBUG alloc_frac_per_pos", f"{alloc_frac_debug * 100:.2f}%")
        tier_table.add_row("Capital utilisé pour tier", f"${self.capital:.2f}")
        
        # Afficher la table du palier
        console.print(tier_table)
        
        # Afficher une ligne de séparation
        console.print(Rule())
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode.
            
        Returns:
            None
        """
        # Rendering is handled by _display_trading_table in step()
        pass
    
    def close(self):
        """
        Clean up resources.
        """
        # Export trading data if requested
        if self.export_history and self.history:
            self.export_trading_data(self.export_dir)
    
    def export_trading_data(self, export_dir=None):
        """
        Export trading history and logs to files.
        
        Args:
            export_dir: Directory to export data to.
        """
        if export_dir is None:
            export_dir = os.path.join(os.getcwd(), 'exports')
        
        ensure_dir_exists(export_dir)
        
        # Generate timestamp
        timestamp = int(time.time())
        
        # Export history
        if self.history:
            history_df = pd.DataFrame(self.history)
            history_path = os.path.join(export_dir, f'trading_history_{timestamp}.csv')
            history_df.to_csv(history_path, index=False)
            logger.info(f"Exported trading history to {history_path}")
        
        # Export trade log
        if self.trade_log:
            trade_log_df = pd.DataFrame(self.trade_log)
            trade_log_path = os.path.join(export_dir, f'trade_log_{timestamp}.csv')
            trade_log_df.to_csv(trade_log_path, index=False)
            logger.info(f"Exported trade log to {trade_log_path}")
        
        # Calculate and export performance metrics
        metrics = self._calculate_performance_metrics()
        metrics_path = os.path.join(export_dir, f'performance_metrics_{timestamp}.csv')
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        logger.info(f"Exported performance metrics to {metrics_path}")
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics from trading history.
        
        Returns:
            dict: Performance metrics.
        """
        if not self.history:
            return {}
        
        # Extract relevant data
        portfolio_values = [h['portfolio_value'] for h in self.history]
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        # Calculate metrics
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns (assuming each step is a day)
        returns = [(v2 - v1) / v1 for v1, v2 in zip(portfolio_values[:-1], portfolio_values[1:])]
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Maximum drawdown
        max_drawdown = 0
        peak = portfolio_values[0]
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate
        trades = [t for t in self.trade_log if t.get('pnl') is not None]
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Average PnL
        avg_pnl = np.mean([t.get('pnl', 0) for t in trades]) if trades else 0
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'num_trades': len(trades),
            'num_steps': len(self.history)
        }
