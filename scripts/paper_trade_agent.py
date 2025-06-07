#!/usr/bin/env python3
"""
Script de Paper Trading pour ADAN - Trading en temps réel avec agent RL pré-entraîné.

Ce script charge un agent PPO pré-entraîné et l'utilise pour prendre des décisions
de trading sur le Binance Testnet en temps réel.
"""

import os
import sys
import argparse
import time
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.utils import get_logger, load_config
from adan_trading_bot.exchange_api.connector import get_exchange_client, validate_exchange_config
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.environment.state_builder import StateBuilder
from adan_trading_bot.agent.ppo_agent import load_agent

logger = get_logger(__name__)

class PaperTradingAgent:
    """Agent de paper trading en temps réel."""
    
    def __init__(self, config, model_path, initial_capital=15000.0):
        """
        Initialise l'agent de paper trading.
        
        Args:
            config: Configuration complète du système
            model_path: Chemin vers le modèle PPO pré-entraîné
            initial_capital: Capital initial pour le paper trading
        """
        self.config = config
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        
        # Actifs à trader
        self.assets = config.get('data', {}).get('assets', [])
        logger.info(f"📊 Assets configured: {self.assets}")
        
        # Configuration du trading
        self.training_timeframe = config.get('data', {}).get('training_timeframe', '1m')
        self.data_source_type = config.get('data', {}).get('data_source_type', 'precomputed_features')
        
        # Initialiser les composants
        self._initialize_exchange()
        self._load_agent_and_scaler()
        self._initialize_order_manager()
        self._initialize_state_builder()
        
        # Historique des trades
        self.trade_history = []
        self.decision_history = []
        
        logger.info(f"🚀 PaperTradingAgent initialized - Capital: ${self.current_capital:.2f}")
    
    def _initialize_exchange(self):
        """Initialise la connexion à l'exchange."""
        try:
            # Valider la configuration
            if not validate_exchange_config(self.config):
                raise ValueError("Configuration d'exchange invalide")
            
            # Créer le client d'exchange
            self.exchange = get_exchange_client(self.config)
            logger.info(f"✅ Exchange connected: {self.exchange.id}")
            
            # Charger les marchés
            self.markets = self.exchange.load_markets()
            logger.info(f"📈 Markets loaded: {len(self.markets)} pairs")
            
        except Exception as e:
            logger.error(f"❌ Exchange initialization failed: {e}")
            logger.warning("🔧 Falling back to simulation mode")
            self.exchange = None
            self.markets = None
    
    def _load_agent_and_scaler(self):
        """Charge l'agent PPO et le scaler correspondant au training_timeframe."""
        try:
            # Charger l'agent
            logger.info(f"🤖 Loading agent from: {self.model_path}")
            self.agent = load_agent(self.model_path)
            logger.info("✅ Agent loaded successfully")
            
            # Charger le scaler approprié pour le training_timeframe
            self.scaler = self._load_appropriate_scaler()
            
        except Exception as e:
            logger.error(f"❌ Failed to load agent or scaler: {e}")
            raise
    
    def _load_appropriate_scaler(self):
        """Charge le scaler approprié selon le training_timeframe et data_source_type."""
        scalers_dir = project_root / "data" / "scalers_encoders"
        
        # Strategy 1: Scaler spécifique au timeframe
        scaler_candidates = [
            scalers_dir / f"scaler_{self.training_timeframe}.joblib",
            scalers_dir / f"scaler_{self.training_timeframe}_cpu.joblib",
            scalers_dir / f"unified_scaler_{self.training_timeframe}.joblib"
        ]
        
        # Strategy 2: Fallback scaler générique
        fallback_candidates = [
            scalers_dir / "scaler_cpu.joblib",
            scalers_dir / "scaler.joblib",
            scalers_dir / "unified_scaler.joblib"
        ]
        
        all_candidates = scaler_candidates + fallback_candidates
        
        for scaler_path in all_candidates:
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                    logger.info(f"✅ Scaler loaded from: {scaler_path}")
                    
                    # Validation du scaler
                    if hasattr(scaler, 'transform'):
                        logger.info(f"📊 Scaler features: {getattr(scaler, 'n_features_in_', 'Unknown')}")
                        return scaler
                    else:
                        logger.warning(f"⚠️ Invalid scaler format in {scaler_path}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load scaler from {scaler_path}: {e}")
        
        # Strategy 3: Créer un scaler à partir des données d'entraînement
        logger.warning("⚠️ No pre-saved scaler found, attempting to create from training data")
        return self._create_scaler_from_training_data()
    
    def _create_scaler_from_training_data(self):
        """Crée un scaler à partir des données d'entraînement du timeframe."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Charger les données d'entraînement correspondantes
            processed_dir = project_root / "data" / "processed" / "unified"
            train_file = processed_dir / f"{self.training_timeframe}_train_merged.parquet"
            
            if not train_file.exists():
                # Fallback vers l'ancien format
                old_processed_dir = project_root / "data" / "processed" / "merged" / "unified"
                train_file = old_processed_dir / f"{self.training_timeframe}_train_merged.parquet"
            
            if not train_file.exists():
                logger.error(f"❌ No training data found for {self.training_timeframe}")
                return None
            
            logger.info(f"📊 Creating scaler from training data: {train_file}")
            train_df = pd.read_parquet(train_file)
            
            # Identifier les colonnes à normaliser (exclure OHLC)
            ohlc_patterns = ['open_', 'high_', 'low_', 'close_']
            cols_to_normalize = []
            
            for col in train_df.columns:
                should_normalize = True
                for pattern in ohlc_patterns:
                    if col.startswith(pattern):
                        should_normalize = False
                        break
                if should_normalize:
                    cols_to_normalize.append(col)
            
            if not cols_to_normalize:
                logger.warning("⚠️ No columns to normalize found")
                return None
            
            # Créer et ajuster le scaler
            scaler = StandardScaler()
            scaler.fit(train_df[cols_to_normalize])
            
            # Sauvegarder pour usage futur
            scaler_path = project_root / "data" / "scalers_encoders" / f"runtime_scaler_{self.training_timeframe}.joblib"
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"✅ Runtime scaler created and saved: {scaler_path}")
            logger.info(f"📊 Normalizable features: {len(cols_to_normalize)}")
            
            # Stocker les colonnes pour usage futur
            self.normalizable_columns = cols_to_normalize
            
            return scaler
            
        except Exception as e:
            logger.error(f"❌ Failed to create scaler from training data: {e}")
            return None
    
    def _initialize_order_manager(self):
        """Initialise le gestionnaire d'ordres."""
        self.order_manager = OrderManager(self.config, exchange_client=self.exchange)
        logger.info("✅ OrderManager initialized")
    
    def _initialize_state_builder(self):
        """Initialise le constructeur d'état."""
        # Pour le paper trading, nous utiliserons une version simplifiée
        # qui peut construire des états à partir de données live
        self.state_builder = None
        logger.info("⚠️ StateBuilder will be initialized with first market data")
    
    def get_live_market_data(self, symbol_ccxt, limit=50):
        """
        Récupère les données de marché en temps réel.
        
        Args:
            symbol_ccxt: Symbole au format CCXT (ex: "BTC/USDT")
            limit: Nombre de bougies à récupérer
            
        Returns:
            pd.DataFrame: Données OHLCV ou None si erreur
        """
        try:
            if not self.exchange:
                logger.warning("❌ No exchange connection - cannot fetch live data")
                return None
            
            # Récupérer les données OHLCV
            ohlcv = self.exchange.fetch_ohlcv(
                symbol_ccxt, 
                timeframe='1m',  # Toujours récupérer en 1m pour flexibilité
                limit=limit
            )
            
            if not ohlcv:
                logger.warning(f"❌ No OHLCV data received for {symbol_ccxt}")
                return None
            
            # Convertir en DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"📊 Fetched {len(df)} candles for {symbol_ccxt}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data for {symbol_ccxt}: {e}")
            return None
    
    def process_market_data_for_agent(self, market_data_dict):
        """
        Traite les données de marché pour créer une observation pour l'agent selon le training_timeframe.
        
        Args:
            market_data_dict: Dict avec asset_id -> DataFrame OHLCV 1m
            
        Returns:
            np.array: Observation normalisée pour l'agent ou None si erreur
        """
        try:
            logger.info(f"🔄 Processing market data for {self.training_timeframe} timeframe")
            
            # Étape 1: Créer les données au timeframe approprié
            processed_data = self._prepare_timeframe_data(market_data_dict)
            if processed_data is None:
                logger.error("❌ Failed to prepare timeframe data")
                return None
            
            # Étape 2: Calculer les features selon le timeframe
            features_data = self._calculate_features_for_timeframe(processed_data)
            if features_data is None:
                logger.error("❌ Failed to calculate features")
                return None
            
            # Étape 3: Normaliser les features
            normalized_data = self._normalize_features(features_data)
            if normalized_data is None:
                logger.error("❌ Failed to normalize features")
                return None
            
            # Étape 4: Construire l'observation finale
            observation = self._build_final_observation(normalized_data)
            
            if observation is not None:
                logger.debug(f"📊 Final observation shape: {observation.shape}")
                logger.debug(f"📊 Observation range: [{observation.min():.6f}, {observation.max():.6f}]")
            
            return observation
            
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")
            return None
    
    def _prepare_timeframe_data(self, market_data_dict):
        """Prépare les données au timeframe d'entraînement."""
        try:
            timeframe_data = {}
            
            for asset_id in self.assets:
                if asset_id not in market_data_dict:
                    logger.warning(f"⚠️ No market data for {asset_id}")
                    continue
                
                df_1m = market_data_dict[asset_id]
                
                if self.training_timeframe == "1m":
                    # Utiliser directement les données 1m
                    timeframe_data[asset_id] = df_1m.copy()
                    
                elif self.training_timeframe in ["1h", "1d"]:
                    # Ré-échantillonner vers le timeframe cible
                    freq = '1H' if self.training_timeframe == '1h' else '1D'
                    
                    # Assurer que l'index est datetime
                    if not isinstance(df_1m.index, pd.DatetimeIndex):
                        df_1m.index = pd.to_datetime(df_1m.index)
                    
                    # Ré-échantillonner OHLCV
                    resampled = df_1m.resample(freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    if len(resampled) < 2:
                        logger.warning(f"⚠️ Insufficient resampled data for {asset_id}: {len(resampled)}")
                        continue
                    
                    timeframe_data[asset_id] = resampled
                    logger.debug(f"📊 {asset_id}: {len(df_1m)} 1m bars → {len(resampled)} {self.training_timeframe} bars")
                
                else:
                    logger.error(f"❌ Unsupported training_timeframe: {self.training_timeframe}")
                    return None
            
            return timeframe_data if timeframe_data else None
            
        except Exception as e:
            logger.error(f"❌ Error preparing timeframe data: {e}")
            return None
    
    def _calculate_features_for_timeframe(self, timeframe_data):
        """Calcule les features selon le timeframe et data_source_type."""
        try:
            features_data = {}
            
            for asset_id, df in timeframe_data.items():
                if self.training_timeframe == "1m" and self.data_source_type in ["precomputed_features_1m_resample", "precomputed_features"]:
                    # Mode 1m avec features pré-calculées : simuler avec OHLCV + quelques indicateurs simples
                    logger.warning(f"⚠️ {asset_id}: Simulating precomputed features with basic indicators")
                    features_df = self._calculate_basic_indicators(df)
                    
                elif self.training_timeframe in ["1h", "1d"]:
                    # Mode 1h/1d : calculer les indicateurs selon la configuration
                    indicators_config = self.config.get('data', {}).get('indicators_by_timeframe', {}).get(self.training_timeframe, [])
                    features_df = self._calculate_timeframe_indicators(df, indicators_config)
                    
                else:
                    # Fallback : OHLCV uniquement
                    features_df = df.copy()
                
                features_data[asset_id] = features_df
                logger.debug(f"📊 {asset_id}: {features_df.shape[1]} features calculated")
            
            return features_data if features_data else None
            
        except Exception as e:
            logger.error(f"❌ Error calculating features: {e}")
            return None
    
    def _calculate_basic_indicators(self, df):
        """Calcule des indicateurs de base pour simuler les features pré-calculées."""
        try:
            import pandas_ta as ta
            
            features_df = df.copy()
            
            # Indicateurs simples calculables à la volée
            features_df['SMA_short'] = ta.sma(df['close'], length=10)
            features_df['SMA_long'] = ta.sma(df['close'], length=50)
            features_df['EMA_short'] = ta.ema(df['close'], length=12)
            features_df['EMA_long'] = ta.ema(df['close'], length=26)
            features_df['RSI'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_data is not None and len(macd_data.columns) >= 3:
                features_df['MACD'] = macd_data.iloc[:, 0]
                features_df['MACDs'] = macd_data.iloc[:, 1]
                features_df['MACDh'] = macd_data.iloc[:, 2]
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20, std=2)
            if bb_data is not None and len(bb_data.columns) >= 3:
                features_df['BBL'] = bb_data.iloc[:, 0]
                features_df['BBM'] = bb_data.iloc[:, 1]
                features_df['BBU'] = bb_data.iloc[:, 2]
            
            # ATR
            features_df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Nettoyer les NaN
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            logger.debug(f"📊 Basic indicators calculated: {features_df.shape[1]} features")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Error calculating basic indicators: {e}")
            return df  # Fallback to OHLCV only
    
    def _calculate_timeframe_indicators(self, df, indicators_config):
        """Calcule les indicateurs selon la configuration pour 1h/1d."""
        try:
            if not indicators_config:
                logger.warning(f"⚠️ No indicators configured for {self.training_timeframe}")
                return df
            
            # Import de la fonction add_technical_indicators si disponible
            try:
                from src.adan_trading_bot.data_processing.feature_engineering import add_technical_indicators
                features_df = add_technical_indicators(df, indicators_config, self.training_timeframe)
                logger.debug(f"📊 Timeframe indicators calculated: {features_df.shape[1]} features")
                return features_df
            except ImportError:
                logger.warning("⚠️ add_technical_indicators not available, using basic indicators")
                return self._calculate_basic_indicators(df)
            
        except Exception as e:
            logger.error(f"❌ Error calculating timeframe indicators: {e}")
            return df
    
    def _normalize_features(self, features_data):
        """Normalise les features avec le scaler approprié."""
        try:
            if not self.scaler:
                logger.warning("⚠️ No scaler available, using raw features")
                return features_data
            
            normalized_data = {}
            
            for asset_id, df in features_data.items():
                # Identifier les colonnes à normaliser
                normalizable_cols = getattr(self, 'normalizable_columns', None)
                
                if normalizable_cols is None:
                    # Auto-detect normalizable columns (exclude OHLC)
                    ohlc_patterns = ['open', 'high', 'low', 'close']
                    normalizable_cols = [col for col in df.columns 
                                       if not any(col.startswith(pattern) for pattern in ohlc_patterns)]
                
                # Filtrer les colonnes qui existent réellement
                available_cols = [col for col in normalizable_cols if col in df.columns]
                
                if not available_cols:
                    logger.warning(f"⚠️ {asset_id}: No normalizable columns found")
                    normalized_data[asset_id] = df
                    continue
                
                # Normaliser
                df_normalized = df.copy()
                try:
                    df_normalized[available_cols] = self.scaler.transform(df[available_cols])
                    logger.debug(f"📊 {asset_id}: {len(available_cols)} features normalized")
                except Exception as e:
                    logger.warning(f"⚠️ {asset_id}: Normalization failed: {e}")
                
                normalized_data[asset_id] = df_normalized
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"❌ Error normalizing features: {e}")
            return features_data  # Return unnormalized data as fallback
    
    def _build_final_observation(self, normalized_data):
        """Construit l'observation finale pour l'agent."""
        try:
            window_size = self.config.get('data', {}).get('cnn_input_window_size', 20)
            features = []
            
            for asset_id in self.assets:
                if asset_id not in normalized_data:
                    logger.warning(f"⚠️ No normalized data for {asset_id}")
                    # Ajouter des zéros pour maintenir la cohérence dimensionnelle
                    num_features = len(self.config.get('data', {}).get('base_market_features', [])) or 5
                    features.extend([0.0] * num_features * window_size)
                    continue
                
                df = normalized_data[asset_id]
                
                if len(df) < window_size:
                    logger.warning(f"⚠️ Insufficient data for {asset_id}: {len(df)} < {window_size}")
                    # Padding avec la dernière valeur disponible
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        padding_rows = []
                        for _ in range(window_size - len(df)):
                            padding_rows.append(last_row)
                        padding_df = pd.DataFrame(padding_rows, columns=df.columns)
                        df_windowed = pd.concat([df, padding_df], ignore_index=True)
                    else:
                        # Aucune donnée : utiliser des zéros
                        features.extend([0.0] * len(df.columns) * window_size)
                        continue
                else:
                    df_windowed = df.tail(window_size)
                
                # Aplatir les données de la fenêtre
                asset_features = df_windowed.values.flatten()
                features.extend(asset_features.tolist())
            
            if not features:
                logger.error("❌ No features generated for observation")
                return None
            
            observation = np.array(features, dtype=np.float32)
            
            # Validation de l'observation
            if np.any(np.isnan(observation)):
                logger.warning("⚠️ NaN values detected in observation, replacing with 0")
                observation = np.nan_to_num(observation, nan=0.0)
            
            if np.any(np.isinf(observation)):
                logger.warning("⚠️ Infinite values detected in observation, clipping")
                observation = np.clip(observation, -1e6, 1e6)
            
            return observation
            
        except Exception as e:
            logger.error(f"❌ Error building final observation: {e}")
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
            logger.warning(f"⚠️ Unknown quote currency for {asset_id}")
            return None
    
    def translate_action(self, action):
        """
        Traduit l'action numérique en asset_id et type de trade.
        
        Args:
            action: Action numérique de l'agent
            
        Returns:
            tuple: (asset_id, trade_type) ou (None, "HOLD")
        """
        try:
            if action == 0:  # HOLD
                return None, "HOLD"
            
            num_assets = len(self.assets)
            
            # Actions BUY: 1 à num_assets
            if 1 <= action <= num_assets:
                asset_index = action - 1
                asset_id = self.assets[asset_index]
                return asset_id, "BUY"
            
            # Actions SELL: num_assets+1 à 2*num_assets
            elif num_assets + 1 <= action <= 2 * num_assets:
                asset_index = action - num_assets - 1
                asset_id = self.assets[asset_index]
                return asset_id, "SELL"
            
            else:
                logger.warning(f"⚠️ Unknown action: {action}")
                return None, "HOLD"
                
        except Exception as e:
            logger.error(f"❌ Error translating action {action}: {e}")
            return None, "HOLD"
    
    def execute_trading_decision(self, asset_id, trade_type, current_prices):
        """
        Exécute une décision de trading.
        
        Args:
            asset_id: ID de l'actif à trader
            trade_type: Type de trade ("BUY" ou "SELL")
            current_prices: Dict des prix actuels
            
        Returns:
            dict: Résultat de l'exécution
        """
        try:
            if trade_type == "HOLD":
                return {"status": "HOLD", "message": "No action taken"}
            
            if asset_id not in current_prices:
                logger.error(f"❌ No current price for {asset_id}")
                return {"status": "ERROR", "message": f"No price for {asset_id}"}
            
            current_price = current_prices[asset_id]
            
            if trade_type == "BUY":
                # Allouer 20% du capital disponible pour cet achat
                allocation_percent = 0.2
                allocated_value = self.current_capital * allocation_percent
                
                logger.info(f"🔄 Executing BUY {asset_id}: ${allocated_value:.2f} at ${current_price:.6f}")
                
                reward_mod, status, info = self.order_manager.execute_order(
                    asset_id=asset_id,
                    action_type=1,  # BUY
                    current_price=current_price,
                    capital=self.current_capital,
                    positions=self.positions,
                    allocated_value_usdt=allocated_value
                )
                
                if info.get('new_capital') is not None:
                    self.current_capital = info['new_capital']
                
                return {
                    "status": status,
                    "message": f"BUY {asset_id}",
                    "info": info,
                    "reward_mod": reward_mod
                }
                
            elif trade_type == "SELL":
                if asset_id not in self.positions or self.positions[asset_id]["qty"] <= 0:
                    logger.warning(f"⚠️ Cannot SELL {asset_id}: No position")
                    return {"status": "NO_POSITION", "message": f"No position to sell for {asset_id}"}
                
                logger.info(f"🔄 Executing SELL {asset_id}: {self.positions[asset_id]['qty']:.6f} at ${current_price:.6f}")
                
                reward_mod, status, info = self.order_manager.execute_order(
                    asset_id=asset_id,
                    action_type=2,  # SELL
                    current_price=current_price,
                    capital=self.current_capital,
                    positions=self.positions
                )
                
                if info.get('new_capital') is not None:
                    self.current_capital = info['new_capital']
                
                return {
                    "status": status,
                    "message": f"SELL {asset_id}",
                    "info": info,
                    "reward_mod": reward_mod
                }
            
        except Exception as e:
            logger.error(f"❌ Error executing {trade_type} for {asset_id}: {e}")
            return {"status": "ERROR", "message": str(e)}
    
    def run_trading_loop(self, max_iterations=100, sleep_seconds=60):
        """
        Exécute la boucle principale de trading.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            sleep_seconds: Temps d'attente entre chaque décision
        """
        logger.info(f"🚀 Starting paper trading loop - Max iterations: {max_iterations}")
        logger.info(f"⏰ Decision frequency: Every {sleep_seconds} seconds")
        
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 ITERATION {iteration}/{max_iterations}")
                logger.info(f"💰 Current Capital: ${self.current_capital:.2f}")
                logger.info(f"📊 Open Positions: {len(self.positions)}")
                
                # 1. Récupérer les données de marché
                market_data_dict = {}
                current_prices = {}
                
                for asset_id in self.assets:
                    symbol_ccxt = self.convert_asset_to_ccxt_symbol(asset_id)
                    if symbol_ccxt:
                        market_data = self.get_live_market_data(symbol_ccxt, limit=30)
                        if market_data is not None and not market_data.empty:
                            market_data_dict[asset_id] = market_data
                            current_prices[asset_id] = float(market_data['close'].iloc[-1])
                            logger.debug(f"📈 {asset_id}: ${current_prices[asset_id]:.6f}")
                
                if not market_data_dict:
                    logger.warning("⚠️ No market data available - skipping iteration")
                    time.sleep(sleep_seconds)
                    continue
                
                # 2. Construire l'observation pour l'agent
                observation = self.process_market_data_for_agent(market_data_dict)
                if observation is None:
                    logger.warning("⚠️ Failed to build observation - skipping iteration")
                    time.sleep(sleep_seconds)
                    continue
                
                # 3. Obtenir la décision de l'agent
                try:
                    action, _ = self.agent.predict(observation, deterministic=True)
                    asset_id, trade_type = self.translate_action(action)
                    
                    logger.info(f"🤖 Agent Decision: Action={action} -> {trade_type} {asset_id or 'N/A'}")
                    
                except Exception as e:
                    logger.error(f"❌ Agent prediction failed: {e}")
                    asset_id, trade_type = None, "HOLD"
                
                # 4. Exécuter la décision
                execution_result = self.execute_trading_decision(asset_id, trade_type, current_prices)
                
                # 5. Enregistrer l'historique
                decision_record = {
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                    "action": action if 'action' in locals() else None,
                    "asset_id": asset_id,
                    "trade_type": trade_type,
                    "execution_result": execution_result,
                    "capital_before": self.current_capital,
                    "positions_count": len(self.positions)
                }
                
                self.decision_history.append(decision_record)
                
                # 6. Afficher le résumé
                pnl = self.current_capital - self.initial_capital
                pnl_pct = (pnl / self.initial_capital) * 100
                
                logger.info(f"💼 Portfolio Summary:")
                logger.info(f"   💰 Capital: ${self.current_capital:.2f}")
                logger.info(f"   📈 PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                logger.info(f"   🎯 Positions: {list(self.positions.keys())}")
                
                # 7. Attendre avant la prochaine décision
                if iteration < max_iterations:
                    logger.info(f"⏰ Sleeping {sleep_seconds} seconds...")
                    time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
        finally:
            self.save_trading_summary()
    
    def save_trading_summary(self):
        """Sauvegarde un résumé de la session de trading."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = project_root / f"paper_trading_summary_{timestamp}.json"
            
            # Calculer les statistiques finales
            final_pnl = self.current_capital - self.initial_capital
            final_pnl_pct = (final_pnl / self.initial_capital) * 100
            
            summary = {
                "session_info": {
                    "start_time": datetime.now().isoformat(),
                    "model_path": str(self.model_path),
                    "initial_capital": self.initial_capital,
                    "final_capital": self.current_capital,
                    "total_pnl": final_pnl,
                    "pnl_percentage": final_pnl_pct,
                    "total_decisions": len(self.decision_history),
                    "assets_traded": self.assets
                },
                "final_positions": self.positions,
                "decision_history": [
                    {k: v if k != "timestamp" else v.isoformat() for k, v in record.items()}
                    for record in self.decision_history
                ]
            }
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📊 Trading summary saved: {summary_file}")
            logger.info(f"🎯 Final Results:")
            logger.info(f"   💰 Capital: ${self.initial_capital:.2f} -> ${self.current_capital:.2f}")
            logger.info(f"   📈 PnL: ${final_pnl:.2f} ({final_pnl_pct:+.2f}%)")
            logger.info(f"   🔄 Decisions: {len(self.decision_history)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save trading summary: {e}")


def main():
    """Fonction principale du script de paper trading."""
    parser = argparse.ArgumentParser(description="ADAN Paper Trading Agent")
    parser.add_argument("--exec_profile", type=str, default="cpu", 
                       choices=["cpu", "gpu"], help="Profil d'exécution")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Chemin vers le modèle PPO pré-entraîné")
    parser.add_argument("--initial_capital", type=float, default=15000.0,
                       help="Capital initial pour le paper trading")
    parser.add_argument("--max_iterations", type=int, default=100,
                       help="Nombre maximum d'itérations de trading")
    parser.add_argument("--sleep_seconds", type=int, default=60,
                       help="Temps d'attente entre chaque décision (secondes)")
    
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
        
        # Initialiser l'agent de paper trading
        trading_agent = PaperTradingAgent(
            config=config,
            model_path=str(model_path),
            initial_capital=args.initial_capital
        )
        
        # Lancer la boucle de trading
        trading_agent.run_trading_loop(
            max_iterations=args.max_iterations,
            sleep_seconds=args.sleep_seconds
        )
        
    except Exception as e:
        logger.error(f"❌ Paper trading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()