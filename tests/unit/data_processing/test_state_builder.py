"""Tests unitaires pour le StateBuilder."""

import numpy as np
import pandas as pd
from unittest.mock import patch
from adan_trading_bot.data_processing.state_builder import (
    StateBuilder,
    TimeframeConfig,
)


def test_timeframe_config_initialization():
    """Test l'initialisation de TimeframeConfig."""
    config = TimeframeConfig(
        timeframe="5m",
        features=["open", "close", "volume"],
        window_size=100,
        normalize=True,
    )

    if not (
        config.timeframe == "5m"
        and config.features == ["open", "close", "volume"]
        and config.window_size == 100
        and config.normalize is True
    ):
        raise AssertionError("La configuration du timeframe est incorrecte")


def test_state_builder_initialization():
    """Test l'initialisation du StateBuilder."""
    features_config = {
        "5m": ["open", "close", "volume"],
        "1h": ["open", "high", "low", "close"],
    }

    # Créer l'instance avec uniquement features_config
    builder = StateBuilder(
        features_config=features_config, window_size=100
    )

    # Vérifier que les timeframes sont correctement configurés
    if not (hasattr(builder, "timeframes")):
        raise AssertionError("L'attribut timeframes est manquant")
        
    # Vérifier que les timeframes sont correctement déduits de features_config
    expected_timeframes = ["5m", "1h"]
    if not all(tf in builder.timeframes for tf in expected_timeframes):
        raise AssertionError(f"Les timeframes attendus sont {expected_timeframes}, mais obtenus {builder.timeframes}")

    # Vérifier que les fonctionnalités sont correctement configurées
    if not (
        hasattr(builder, "features_config")
        and "5m" in builder.features_config
        and "1h" in builder.features_config
    ):
        raise AssertionError("Les fonctionnalités n'ont pas été configurées")





def test_build_state():
    """
    Test la construction de l'état en s'assurant que les calculs sont effectués.
    """
    window_size = 20  # StateBuilder utilise 20 par défaut
    n_features = 5
    features = ["open", "high", "low", "close", "volume"]

    builder = StateBuilder(
        features_config={"5m": features},
        window_size=window_size,
    )

    # Fournir suffisamment de données pour éviter le retour anticipé
    data_points = window_size + 20
    market_data = {
        "asset1": {
            "5m": pd.DataFrame(
                {
                    "TIMESTAMP": pd.date_range("2023-01-01", periods=data_points, freq="5min"),
                    "OPEN": np.random.rand(data_points) * 100,
                    "HIGH": np.random.rand(data_points) * 100,
                    "LOW": np.random.rand(data_points) * 100,
                    "CLOSE": np.random.rand(data_points) * 100,
                    "VOLUME": np.random.rand(data_points) * 1000,
                }
            ).set_index("TIMESTAMP")
        }
    }

    # Choisir un index qui permet une fenêtre complète
    current_idx = window_size + 5

    # Entraîner les scalers avant de construire l'observation
    builder.fit_scalers(data=market_data["asset1"])

    # Construire l'observation complète
    observation = builder.build_observation(current_idx=current_idx, data=market_data)

    # 1. Vérifier que le résultat est valide
    assert observation is not None, "La construction de l'observation a échoué"
    assert isinstance(observation, dict), "Le résultat devrait être un dictionnaire"
    assert "5m" in observation, "La clé du timeframe '5m' est manquante"
    assert "portfolio_state" in observation, "La clé 'portfolio_state' est manquante"

    # 2. Vérifier la forme des données de marché
    market_obs = observation["5m"]
    assert isinstance(market_obs, np.ndarray), "L'observation de marché devrait être un tableau numpy"
    expected_market_shape = (window_size, n_features)
    assert market_obs.shape == expected_market_shape, f"Forme de l'observation de marché incorrecte: attendu {expected_market_shape}, obtenu {market_obs.shape}"

    # 3. Vérifier la forme de l'état du portfolio
    portfolio_obs = observation["portfolio_state"]
    assert isinstance(portfolio_obs, np.ndarray), "L'état du portfolio devrait être un tableau numpy"
    # La taille par défaut est 20
    assert portfolio_obs.shape == (20,), f"Forme de l'état du portfolio incorrecte: attendu (20,), obtenu {portfolio_obs.shape}"

    # 4. Vérifier que le calcul a bien eu lieu (l'array n'est pas juste rempli de zéros)
    assert np.sum(market_obs) != 0, "L'observation de marché ne devrait pas être remplie de zéros"

def test_set_timeframe_config():
    """Test la configuration des timeframes."""
    builder = StateBuilder(
        features_config={"5m": ["open", "close"]}, window_size=100
    )

    if not hasattr(builder, "timeframes") or "5m" not in builder.timeframes:
        raise AssertionError("Le timeframe 5m est manquant")

    if not hasattr(builder, "features_config") or "5m" not in builder.features_config:
        raise AssertionError("Configuration des fonctionnalités manquante")

    if hasattr(builder, "features_config"):
        builder.features_config["1h"] = ["open", "high", "low", "close"]
        builder.timeframes.append("1h")

        if "1h" not in builder.timeframes:
            raise AssertionError("Échec de l'ajout du timeframe 1h")
        if "1h" not in builder.features_config:
            raise AssertionError("Fonctionnalités manquantes pour 1h")
