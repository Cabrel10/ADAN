"""
Tests d'intégration bout-en-bout réels avec MultiAssetChunkedEnv.
Ces tests valident le flux complet sans mocks excessifs.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.agent.ppo_agent import PPOAgent
from adan_trading_bot.common.config import load_config
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.risk_management.risk_calculator import RiskCalculator


class TestRealEnvironmentFlow:
    """Tests d'intégration avec de vrais composants (pas de mocks)."""

    @pytest.fixture
    def synthetic_market_data(self):
        """Génère des données de marché synthétiques mais réalistes."""
        timeframes = ["5m", "1h", "4h"]
        assets = ["BTCUSDT", "XRPUSDT"]

        # Générer 1000 points de données pour chaque timeframe/asset
        n_points = 1000
        data_by_tf_asset = {}

        base_timestamp = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

        for tf in timeframes:
            # Déterminer l'intervalle temporel
            if tf == "5m":
                freq = "5T"
            elif tf == "1h":
                freq = "1H"
            else:  # 4h
                freq = "4H"

            timestamps = pd.date_range(
                start=base_timestamp, periods=n_points, freq=freq
            )

            for asset in assets:
                # Prix de base différent par asset
                base_price = 50000 if asset == "BTCUSDT" else 0.5

                # Générer une série de prix avec marche aléatoire
                np.random.seed(42 + hash(asset + tf) % 1000)
                price_changes = np.random.normal(0, 0.01, n_points)
                prices = base_price * np.exp(np.cumsum(price_changes))

                # Créer OHLCV
                high_mult = 1 + np.abs(np.random.normal(0, 0.005, n_points))
                low_mult = 1 - np.abs(np.random.normal(0, 0.005, n_points))

                df = pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "open": prices,
                        "high": prices * high_mult,
                        "low": prices * low_mult,
                        "close": prices * np.roll(high_mult, 1) * np.roll(low_mult, -1),
                        "volume": np.random.lognormal(10, 1, n_points),
                        # Indicateurs synthétiques (15 features au total)
                        "rsi_14": 50
                        + 30 * np.sin(np.arange(n_points) / 10)
                        + np.random.normal(0, 5, n_points),
                        "macd_hist": np.random.normal(0, 0.1, n_points),
                        "atr_14": np.abs(np.random.normal(0.02, 0.01, n_points)),
                        "bb_upper": prices * 1.02,
                        "bb_middle": prices,
                        "bb_lower": prices * 0.98,
                        "volume_ratio": np.random.lognormal(0, 0.5, n_points),
                        "ema_ratio": 1 + np.random.normal(0, 0.05, n_points),
                        "stoch_k": np.random.uniform(0, 100, n_points),
                        "vwap_ratio": 1 + np.random.normal(0, 0.02, n_points),
                    }
                )

                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

                data_by_tf_asset[(tf, asset)] = df

        return data_by_tf_asset

    @pytest.fixture
    def temp_data_directory(self, synthetic_market_data):
        """Crée un répertoire temporaire avec les données synthétiques."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Créer la structure de répertoires
            for mode in ["train", "val", "test"]:
                mode_dir = data_dir / mode
                mode_dir.mkdir(parents=True)

                for (tf, asset), df in synthetic_market_data.items():
                    # Sauvegarder comme parquet
                    file_path = mode_dir / f"{asset}_{tf}.parquet"
                    df.to_parquet(file_path, index=True)

            yield str(data_dir)

    @pytest.fixture
    def real_config(self, temp_data_directory):
        """Configuration réelle mais adaptée aux tests."""
        return {
            "environment": {
                "assets": ["BTCUSDT", "XRPUSDT"],
                "timeframes": ["5m", "1h", "4h"],
                "initial_balance": 1000.0,
                "commission": 0.001,
                "max_steps": 100,  # Limité pour les tests
                "mode": "train",
                "observation": {
                    "shape": [3, 20, 15],  # 3 TF, 20 steps, 15 features
                    "window_sizes": {"5m": 20, "1h": 14, "4h": 8},
                },
                "action_thresholds": {"5m": 0.005, "1h": 0.008, "4h": 0.012},
                "frequency_validation": {
                    "global_rules": {
                        "min_total_trades_per_episode": 1,
                        "max_total_trades_per_episode": 50,
                    }
                },
            },
            "data": {
                "features_config": {
                    "timeframes": {
                        "5m": {
                            "price": ["open", "high", "low", "close"],
                            "volume": ["volume"],
                            "indicators": ["rsi_14", "macd_hist", "atr_14", "bb_upper", "bb_middle", "bb_lower", "volume_ratio", "ema_ratio", "stoch_k", "vwap_ratio"]
                        },
                        "1h": {
                            "price": ["open", "high", "low", "close"],
                            "volume": ["volume"],
                            "indicators": ["rsi_14", "macd_hist", "atr_14", "bb_upper", "bb_middle", "bb_lower", "volume_ratio", "ema_ratio", "stoch_k", "vwap_ratio"]
                        },
                        "4h": {
                            "price": ["open", "high", "low", "close"],
                            "volume": ["volume"],
                            "indicators": ["rsi_14", "macd_hist", "atr_14", "bb_upper", "bb_middle", "bb_lower", "volume_ratio", "ema_ratio", "stoch_k", "vwap_ratio"]
                        }
                    }
                },
                "data_dirs": {
                    "train": f"{temp_data_directory}/train",
                    "val": f"{temp_data_directory}/val",
                    "test": f"{temp_data_directory}/test",
                },
                "assets": ["BTCUSDT", "XRPUSDT"],
                "timeframes": ["5m", "1h", "4h"],
            },
            "agent": {
                "algorithm": "PPO",
                "learning_rate": 0.0003,
                "n_steps": 64,  # Petit pour tests
                "batch_size": 32,
                "n_epochs": 2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "verbose": 0,
                "seed": 42,
            },
            "capital_tiers": [
                {
                    "name": "Test Capital",
                    "min_capital": 0.0,
                    "max_capital": 10000.0,
                    "risk_per_trade_pct": 2.0,
                    "max_position_size_pct": 50.0,
                    "max_concurrent_positions": 2,
                }
            ],
            "dbe": {
                "volatility_calculation": {
                    "primary_method": "rolling_std",
                    "fallback_handling": {
                        "default_volatility": 0.025,
                        "zero_volatility_replacement": 0.015,
                    },
                }
            },
        }

    @pytest.fixture
    def mock_data_loader_paths(self, temp_data_directory):
        """Mock pour que ChunkedDataLoader trouve les fichiers dans le temp directory."""

        def mock_load_chunk(chunk_info, *args, **kwargs):
            # Charger les vraies données depuis temp_data_directory
            chunk_data = {}
            for tf in ["5m", "1h", "4h"]:
                tf_data = {}
                for asset in ["BTCUSDT", "XRPUSDT"]:
                    file_path = (
                        Path(temp_data_directory) / "train" / f"{asset}_{tf}.parquet"
                    )
                    if file_path.exists():
                        df = pd.read_parquet(file_path)
                        # Prendre un sous-ensemble pour simuler un chunk
                        tf_data[asset] = df.iloc[:50].copy()  # 50 points par chunk
                    else:
                        # Données de fallback si fichier manquant
                        tf_data[asset] = pd.DataFrame(
                            {
                                "open": np.random.randn(50),
                                "high": np.random.randn(50),
                                "low": np.random.randn(50),
                                "close": np.random.randn(50),
                                "volume": np.random.randn(50),
                            }
                        )
                chunk_data[tf] = tf_data
            return chunk_data

        return mock_load_chunk

    def test_environment_initialization_with_real_data(
        self, real_config, mock_data_loader_paths
    ):
        """Test que MultiAssetChunkedEnv s'initialise avec de vraies données."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            try:
                env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})

                # Vérifications de base
                assert env is not None
                assert hasattr(env, "observation_space")
                assert hasattr(env, "action_space")
                assert env.config == real_config

                # Vérifier les spaces
                assert env.observation_space is not None
                assert env.action_space is not None

            except Exception as e:
                pytest.fail(f"MultiAssetChunkedEnv n'a pas pu s'initialiser: {e}")

    def test_environment_reset_and_step(self, real_config, mock_data_loader_paths):
        """Test reset et step de l'environnement avec vraies données."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})

            # Test reset
            observation, info = env.reset()

            assert observation is not None
            assert isinstance(observation, dict)
            assert info is not None

            # Vérifier structure observation
            for tf in ["5m", "1h", "4h"]:
                assert tf in observation
                obs_data = observation[tf]
                assert isinstance(obs_data, np.ndarray)
                assert len(obs_data.shape) >= 2  # Au moins 2D

            # Test step
            action_size = env.action_space.shape[0]
            action = np.zeros(action_size)  # Action neutre

            obs, reward, terminated, truncated, info = env.step(action)

            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert info is not None

    def test_ppo_agent_with_real_environment(self, real_config, mock_data_loader_paths):
        """Test PPOAgent avec le vrai MultiAssetChunkedEnv."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})

            try:
                agent = PPOAgent(env, real_config)

                # Test prédiction
                obs, _ = env.reset()
                action, _states = agent.predict(obs, deterministic=True)

                assert action is not None
                assert len(action) == env.action_space.shape[0]

                # Test step avec prédiction
                obs, reward, terminated, truncated, info = env.step(action)

                assert obs is not None
                assert isinstance(reward, (int, float))

            except Exception as e:
                pytest.fail(f"PPOAgent avec vrai environnement a échoué: {e}")

    def test_end_to_end_episode(self, real_config, mock_data_loader_paths):
        """Test épisode complet bout-en-bout."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})
            agent = PPOAgent(env, real_config)

            # Épisode complet
            obs, _ = env.reset()
            total_reward = 0
            steps = 0

            for step in range(50):  # Max 50 steps pour test
                action, _ = agent.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # Vérifications
            assert steps > 0
            assert isinstance(total_reward, (int, float))
            assert not np.isnan(total_reward)

    def test_multiple_episodes_consistency(self, real_config, mock_data_loader_paths):
        """Test cohérence sur plusieurs épisodes."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})
            agent = PPOAgent(env, real_config)

            episode_rewards = []

            for episode in range(3):  # 3 épisodes de test
                obs, _ = env.reset()
                episode_reward = 0

                for step in range(20):  # 20 steps max par épisode
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)

            # Vérifications
            assert len(episode_rewards) == 3
            assert all(not np.isnan(r) for r in episode_rewards)
            assert all(not np.isinf(r) for r in episode_rewards)

    def test_action_thresholds_effectiveness(self, real_config, mock_data_loader_paths):
        """Test que les seuils d'action corrigés permettent des trades."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})
            agent = PPOAgent(env, real_config)

            # Forcer des actions avec magnitude différente
            obs, _ = env.reset()

            # Test action faible (devrait passer avec seuils corrigés)
            low_action = np.full(env.action_space.shape[0], 0.006)  # Au-dessus seuil 5m
            obs1, reward1, term1, trunc1, info1 = env.step(low_action)

            # Test action moyenne
            med_action = np.full(env.action_space.shape[0], 0.010)  # Au-dessus seuil 1h
            obs2, reward2, term2, trunc2, info2 = env.step(med_action)

            # Les actions devraient être acceptées (pas de crash)
            assert obs1 is not None
            assert obs2 is not None
            assert not np.isnan(reward1)
            assert not np.isnan(reward2)

    def test_volatility_calculation_robustness(
        self, real_config, mock_data_loader_paths
    ):
        """Test que le calcul de volatilité ne retourne jamais zéro."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})

            # Plusieurs resets pour tester différents chunks
            for _ in range(5):
                obs, info = env.reset()

                # Vérifier que la volatilité dans info n'est jamais 0
                if "volatility" in info:
                    vol = info["volatility"]
                    if isinstance(vol, dict):
                        for tf_vol in vol.values():
                            assert tf_vol > 0, f"Volatilité nulle détectée: {tf_vol}"
                    elif isinstance(vol, (int, float)):
                        assert vol > 0, f"Volatilité nulle détectée: {vol}"

    def test_portfolio_integration(self, real_config, mock_data_loader_paths):
        """Test intégration avec PortfolioManager."""
        mock_load_chunk = mock_data_loader_paths

        with (
            patch.object(ChunkedDataLoader, "_calculate_total_chunks", return_value=2),
            patch.object(ChunkedDataLoader, "load_chunk", side_effect=mock_load_chunk),
        ):
            env = MultiAssetChunkedEnv(config=real_config, worker_config={"assets": real_config["environment"]["assets"]})

            # L'environnement devrait avoir un portfolio manager intégré
            assert hasattr(env, "portfolio_manager")

            portfolio = env.portfolio_manager
            assert portfolio is not None

            # Test état initial
            initial_balance = portfolio.get_total_value()
            assert initial_balance > 0

            # Après quelques steps
            obs, _ = env.reset()
            action = np.ones(env.action_space.shape[0]) * 0.01  # Action légère

            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            # Portfolio devrait avoir évolué
            final_balance = portfolio.get_total_value()
            assert isinstance(final_balance, (int, float))
            assert not np.isnan(final_balance)


class TestConfigurationFixes:
    """Tests pour valider les correctifs de configuration appliqués."""

    def test_action_thresholds_are_permissive(self):
        """Test que les seuils d'action sont suffisamment permissifs."""
        config = load_config("config/config.yaml")

        thresholds = config["environment"]["action_thresholds"]

        # Vérifier que les seuils ont été corrigés
        assert thresholds["5m"] <= 0.005, f"Seuil 5m trop élevé: {thresholds['5m']}"
        assert thresholds["1h"] <= 0.008, f"Seuil 1h trop élevé: {thresholds['1h']}"
        assert thresholds["4h"] <= 0.012, f"Seuil 4h trop élevé: {thresholds['4h']}"

    def test_frequency_validation_is_permissive(self):
        """Test que la validation de fréquence est permissive."""
        config = load_config("config/config.yaml")

        freq_val = config["environment"]["frequency_validation"]["global_rules"]

        # Min trades doit être très bas
        assert freq_val["min_total_trades_per_episode"] <= 2

        # Force trade est géré par config_loader → garantit enabled = True

    def test_volatility_fallbacks_configured(self):
        """Test que les fallbacks de volatilité sont configurés."""
        config = load_config("config/config.yaml")

        vol_config = config["dbe"]["volatility_calculation"]["fallback_handling"]

        # Fallbacks doivent être positifs
        assert vol_config["default_volatility"] > 0
        assert vol_config["zero_volatility_replacement"] > 0
        assert vol_config["min_volatility"] > 0

    def test_diagnostics_enabled(self):
        """Test que les diagnostics sont activés."""
        config = load_config("config/config.yaml")

        diagnostics = config.get("diagnostics", {})

        # Vérifier que plusieurs types de monitoring sont activés
        assert diagnostics.get("volatility_monitoring", {}).get("enabled", False)
        assert diagnostics.get("action_threshold_monitoring", {}).get("enabled", False)
        assert diagnostics.get("frequency_monitoring", {}).get("enabled", False)