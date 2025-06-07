#!/usr/bin/env python3
"""
Script de validation finale simplifié pour ADAN.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from adan_trading_bot.environment.state_builder import StateBuilder
from adan_trading_bot.environment.order_manager import OrderManager
from adan_trading_bot.common.utils import get_logger, load_config

logger = get_logger()

def create_mock_dataframe(assets, base_features, num_rows=50):
    """Créer un DataFrame mock."""
    columns = []
    for asset in assets:
        for feature in base_features:
            columns.append(f"{feature}_{asset}")
    
    df = pd.DataFrame(np.random.randn(num_rows, len(columns)), columns=columns)
    df.index = pd.date_range('2024-01-01', periods=num_rows, freq='1h')
    
    # Prix positifs
    for asset in assets:
        for price_col in ['open', 'high', 'low', 'close']:
            col_name = f"{price_col}_{asset}"
            if col_name in df.columns:
                df[col_name] = np.abs(df[col_name]) + 50
    
    return df

def test_lot1():
    """Test Lot 1."""
    try:
        data_config = load_config('config/data_config_cpu_lot1.yaml')
        env_config = load_config('config/environment_config.yaml')
        env_config['export_history'] = False  # Désactiver affichage
        config = {'data': data_config, 'environment': env_config}
        
        assets = data_config['assets']
        training_tf = data_config.get('training_timeframe', '1h')
        
        # Construire features attendues
        expected_features = ["open", "high", "low", "close", "volume"]
        indicators_config = data_config.get('indicators_by_timeframe', {}).get(training_tf, [])
        
        for indicator in indicators_config:
            output_col_name = indicator.get('output_col_name')
            output_col_names = indicator.get('output_col_names')
            
            if output_col_name:
                expected_features.append(f"{output_col_name}_{training_tf}")
            elif output_col_names:
                for col_name in output_col_names:
                    expected_features.append(f"{col_name}_{training_tf}")
        
        df = create_mock_dataframe(assets, expected_features)
        env = MultiAssetEnv(df_received=df, config=config, max_episode_steps_override=5)
        
        # Tests
        assert len(env.base_feature_names) == len(expected_features)
        assert env.training_timeframe == training_tf
        
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(1)  # BUY
        
        expected_img_shape = (1, 20, len(expected_features) * len(assets))
        assert obs['image_features'].shape == expected_img_shape
        
        print(f"✅ Lot 1: {len(env.base_feature_names)} features, shape {obs['image_features'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ Lot 1: {e}")
        return False

def test_lot2():
    """Test Lot 2."""
    try:
        data_config = load_config('config/data_config_cpu_lot2.yaml')
        env_config = load_config('config/environment_config.yaml')
        env_config['export_history'] = False
        config = {'data': data_config, 'environment': env_config}
        
        assets = data_config['assets']
        base_features = data_config.get('base_market_features', [])
        
        df = create_mock_dataframe(assets, base_features)
        env = MultiAssetEnv(df_received=df, config=config, max_episode_steps_override=5)
        
        assert len(env.base_feature_names) == len(base_features)
        assert set(env.base_feature_names) == set(base_features)
        
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(3)  # SELL (rejeté)
        
        expected_img_shape = (1, 20, len(base_features) * len(assets))
        assert obs['image_features'].shape == expected_img_shape
        
        print(f"✅ Lot 2: {len(base_features)} features, shape {obs['image_features'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ Lot 2: {e}")
        return False

def test_order_manager():
    """Test OrderManager avec prix normalisés."""
    try:
        config = {
            'environment': {
                'penalties': {'invalid_order_base': -0.3, 'out_of_funds': -0.5},
                'order_rules': {'min_value_tolerable': 10.0, 'min_value_absolute': 9.0},
                'transaction': {'fee_percent': 0.001, 'fixed_fee': 0.0}
            }
        }
        
        order_manager = OrderManager(config)
        
        # Test BUY avec prix négatif
        capital = 100.0
        positions = {}
        current_price = -1.5
        
        reward, status, info = order_manager.execute_order(
            asset_id='BTCUSDT', action_type=1, current_price=current_price,
            capital=capital, positions=positions, allocated_value_usdt=50.0
        )
        
        assert status == "BUY_EXECUTED"
        assert 'BTCUSDT' in positions
        assert positions['BTCUSDT']['qty'] > 0
        assert positions['BTCUSDT']['price'] == current_price
        
        # Test SELL
        new_price = -1.2
        reward, status, info = order_manager.execute_order(
            asset_id='BTCUSDT', action_type=2, current_price=new_price,
            capital=info['new_capital'], positions=positions, quantity=None
        )
        
        assert status == "SELL_EXECUTED"
        assert 'BTCUSDT' not in positions
        
        print("✅ OrderManager: prix normalisés OK")
        return True
        
    except Exception as e:
        print(f"❌ OrderManager: {e}")
        return False

def test_state_builder():
    """Test StateBuilder avec features manquantes."""
    try:
        config = {
            'data': {'training_timeframe': '1h', 'cnn_input_window_size': 20},
            'environment': {'initial_capital': 100.0}
        }
        
        assets = ['BTCUSDT', 'ETHUSDT']
        base_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14_1h']
        
        # DataFrame avec features manquantes
        available_features = ['open', 'high', 'low', 'close', 'volume']
        columns = []
        for asset in assets:
            for feature in available_features:
                columns.append(f"{feature}_{asset}")
        
        df = pd.DataFrame(np.random.randn(30, len(columns)), columns=columns)
        
        state_builder = StateBuilder(config, assets, base_feature_names=base_features, cnn_input_window_size=20)
        
        window = df.tail(20)
        observation = state_builder.build_observation(window, 100.0, {})
        
        expected_shape = (1, 20, len(base_features) * len(assets))
        assert observation['image_features'].shape == expected_shape
        
        # Vérifier NaN pour features manquantes
        img_features = observation['image_features'][0]
        has_nan = np.isnan(img_features).any()
        assert has_nan
        
        print("✅ StateBuilder: gestion features manquantes OK")
        return True
        
    except Exception as e:
        print(f"❌ StateBuilder: {e}")
        return False

if __name__ == "__main__":
    print("🎯 VALIDATION FINALE SIMPLIFIÉE - ADAN")
    print("=" * 50)
    
    tests = [
        ("Lot 1", test_lot1),
        ("Lot 2", test_lot2),
        ("OrderManager", test_order_manager),
        ("StateBuilder", test_state_builder)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}: Erreur - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    success_count = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    overall_success = success_count == len(results)
    print(f"\n🎯 Résultat: {success_count}/{len(results)} tests réussis")
    
    if overall_success:
        print("🎉 VALIDATION COMPLÈTE - SYSTÈME PRÊT!")
    else:
        print("⚠️  Validation partielle - Vérifiez les erreurs")
    
    exit(0 if overall_success else 1)