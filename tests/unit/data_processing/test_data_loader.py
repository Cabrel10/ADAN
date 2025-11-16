import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader

@pytest.fixture
def dummy_data_path(tmp_path):
    data_dir = tmp_path / "processed" / "train"
    asset_dir = data_dir / "BTCUSDT"
    asset_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='5min'))
    df = pd.DataFrame({
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'volume': np.random.rand(100) * 100,
    }, index=dates)
    df.to_parquet(asset_dir / "5m.parquet")
    df.to_parquet(asset_dir / "1h.parquet")
    return tmp_path

def test_chunked_data_loader_initialization(dummy_data_path):
    """Test l'initialisation du ChunkedDataLoader."""
    config = {
        "paths": {
            "processed_data_dir": str(dummy_data_path / "processed")
        },
        "data": {
            "data_dirs": {
                "train": str(dummy_data_path / "processed" / "train")
            },
            "timeframes": ["5m", "1h"],
            "features_config": {
                "timeframes": {
                    "5m": {"price": ["open", "close"], "volume": ["volume"], "indicators": []},
                    "1h": {"price": ["open", "high", "low", "close"], "volume": [], "indicators": []}
                }
            }
        }
    }
    
    worker_config = {
        "assets": ["BTCUSDT"],
        "timeframes": ["5m", "1h"],
        "data_split": "train"
    }
    
    loader = ChunkedDataLoader(
        config=config,
        worker_config=worker_config,
        worker_id=0
    )
    
    assert loader.data_split == "train"
    assert "BTCUSDT" in loader.assets_list
    assert "5m" in loader.timeframes
    assert "1h" in loader.timeframes

def test_load_chunk(dummy_data_path):
    """Test le chargement d'un chunk de données."""
    config = {
        "paths": {
            "processed_data_dir": str(dummy_data_path / "processed")
        },
        "data": {
            "data_dirs": {
                "train": str(dummy_data_path / "processed" / "train")
            },
            "timeframes": ["5m"],
            "features_config": {
                "timeframes": {
                    "5m": {"price": ["open", "close"], "volume": ["volume"], "indicators": []}
                }
            }
        }
    }
    
    worker_config = {
        "assets": ["BTCUSDT"],
        "timeframes": ["5m"],
        "data_split": "train"
    }
    
    loader = ChunkedDataLoader(
        config=config,
        worker_config=worker_config,
        worker_id=0
    )
    
    chunk = loader.load_chunk(0)
    
    assert "BTCUSDT" in chunk
    assert "5m" in chunk["BTCUSDT"]
    assert isinstance(chunk["BTCUSDT"]["5m"], pd.DataFrame)
    assert not chunk["BTCUSDT"]["5m"].empty

def test_get_data_path(dummy_data_path):
    """Test la construction du chemin du fichier de données."""
    config = {
        "paths": {
            "processed_data_dir": str(dummy_data_path / "processed")
        },
        "data": {
            "data_dirs": {
                "train": str(dummy_data_path / "processed" / "train")
            },
            "timeframes": ["5m"],
            "features_config": {
                "timeframes": {
                    "5m": {"price": ["open", "close"], "volume": ["volume"], "indicators": []}
                }
            }
        }
    }
    
    worker_config = {
        "assets": ["BTCUSDT"],
        "timeframes": ["5m"],
        "data_split": "train"
    }
    
    loader = ChunkedDataLoader(
        config=config,
        worker_config=worker_config,
        worker_id=0
    )
    
    path = loader._get_data_path("BTCUSDT", "5m")
    expected_path = dummy_data_path / "processed" / "train" / "BTCUSDT" / "5m.parquet"
    assert path == expected_path