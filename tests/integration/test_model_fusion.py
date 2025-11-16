import pytest
import yaml
import os
import numpy as np
from datetime import datetime

# Assuming MultiAssetChunkedEnv is importable from src.adan_trading_bot.environment
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
# Assuming ModelFusion is importable from src.adan_trading_bot.training.model_fusion
from src.adan_trading_bot.training.model_fusion import ModelFusion

# Helper function to load YAML configuration
def load_config(config_path="/home/morningstar/Documents/trading/bot/config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="module")
def model_fusion_instance():
    # ModelFusion might need a config, or it might have defaults
    config = load_config()
    return ModelFusion(config=config)

def test_weighted_average_calculation(model_fusion_instance):
    """Vérifie le calcul de la moyenne pondérée entre workers"""
    fusion = model_fusion_instance
    
    # Simuler métriques de workers
    worker_outputs = {
        'w1': {'action': 0, 'confidence': 0.8, 'sharpe': 1.2, 'tier': 1},
        'w2': {'action': 1, 'confidence': 0.9, 'sharpe': 1.5, 'tier': 3},
        'w3': {'action': 1, 'confidence': 0.7, 'sharpe': 1.8, 'tier': 5},
        'w4': {'action': 0, 'confidence': 0.85, 'sharpe': 1.4, 'tier': 4}
    }
    
    result = fusion.aggregate(worker_outputs)
    
    # Vérifier que le résultat favorise les hauts Sharpe et la majorité
    # Here, action 1 has higher average sharpe and 2/4 votes.
    # The ModelFusion logic should weigh by performance.
    # Let's assume the aggregate method returns the action with the highest weighted score.
    # w1 (0): 0.8 * 1.2 = 0.96
    # w2 (1): 0.9 * 1.5 = 1.35
    # w3 (1): 0.7 * 1.8 = 1.26
    # w4 (0): 0.85 * 1.4 = 1.19
    # Total for action 0: 0.96 + 1.19 = 2.15
    # Total for action 1: 1.35 + 1.26 = 2.61
    # So, action 1 should be chosen.
    assert result['action'] == 1
    assert 0.7 <= result['confidence'] <= 0.9 # Confidence should be within the range of worker confidences

def test_consensus_threshold(model_fusion_instance):
    """Vérifie le seuil de consensus de 75%"""
    fusion = model_fusion_instance
    
    # Test 1: 100% consensus
    outputs_consensus = {f'w{i}': {'action': 1, 'confidence': 0.9, 'sharpe': 1.0, 'tier': 1} for i in range(4)}
    assert fusion.has_consensus(outputs_consensus) == True
    
    # Test 2: 75% consensus (3/4)
    outputs_75 = {
        'w1': {'action': 1, 'confidence': 0.9, 'sharpe': 1.0, 'tier': 1},
        'w2': {'action': 1, 'confidence': 0.9, 'sharpe': 1.0, 'tier': 1},
        'w3': {'action': 1, 'confidence': 0.9, 'sharpe': 1.0, 'tier': 1},
        'w4': {'action': 0, 'confidence': 0.1, 'sharpe': 0.5, 'tier': 1}
    }
    assert fusion.has_consensus(outputs_75) == True
    
    # Test 3: 50% consensus (2/4)
    outputs_50 = {
        'w1': {'action': 1, 'confidence': 0.9, 'sharpe': 1.0, 'tier': 1},
        'w2': {'action': 1, 'confidence': 0.9, 'sharpe': 1.0, 'tier': 1},
        'w3': {'action': 0, 'confidence': 0.1, 'sharpe': 0.5, 'tier': 1},
        'w4': {'action': 0, 'confidence': 0.1, 'sharpe': 0.5, 'tier': 1}
    }
    assert fusion.has_consensus(outputs_50) == False

def test_conflict_resolution(model_fusion_instance):
    """Vérifie la résolution de conflits (priorité tier + performance)"""
    fusion = model_fusion_instance
    
    # Conflit: 2 workers disent HOLD, 2 disent BUY
    conflicted_outputs = {
        'w1': {'action': 0, 'tier': 1, 'sharpe': 1.0, 'confidence': 0.8},  # Conservative HOLD
        'w2': {'action': 1, 'tier': 3, 'sharpe': 1.5, 'confidence': 0.9},  # Moderate BUY
        'w3': {'action': 1, 'tier': 5, 'sharpe': 1.8, 'confidence': 0.7},  # Aggressive BUY (gagne)
        'w4': {'action': 0, 'tier': 4, 'sharpe': 1.3, 'confidence': 0.85}   # Adaptive HOLD
    }
    
    result = fusion.resolve_conflict(conflicted_outputs)
    
    # Doit favoriser W3 (tier le plus élevé + meilleur Sharpe)
    assert result['action'] == 1
    assert result['source'] == 'w3'
