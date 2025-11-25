import sys
import os
import numpy as np
import pandas as pd
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugPrediction")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.common.config_loader import ConfigLoader
from stable_baselines3 import PPO

def debug_prediction():
    print("="*60)
    print("🔍 DEEP DEBUG: ANALYSE DE L'ÉTAT ET DES PRÉDICTIONS")
    print("="*60)

    # 1. Charger la config
    config = ConfigLoader().load_config('config/config.yaml')
    print(f"DEBUG: Config keys: {config.keys()}")
    if 'data' in config:
        print(f"DEBUG: Data keys: {config['data'].keys()}")
        if 'features' in config['data']:
             print(f"DEBUG: Features keys: {config['data']['features'].keys()}")
    
    # Extraire la config des features du YAML
    try:
        data_conf = config.get('data', {})
        # Essayer 'features_config' (vu dans debug) ou 'features' (vu dans yaml)
        feat_conf = data_conf.get('features_config', data_conf.get('features', {}))
        print(f"DEBUG: feat_conf content: {feat_conf}")
        
        # Structure réelle vue dans debug:
        # {'timeframes': {'1h': {'indicators': [...], 'price': [...], 'volume': [...]}, ...}}
        
        timeframes_conf = feat_conf.get('timeframes', {})
        features_config = {}
        
        if timeframes_conf:
            for tf, conf in timeframes_conf.items():
                price_cols = conf.get('price', [])
                vol_cols = conf.get('volume', [])
                ind_cols = conf.get('indicators', [])
                
                # Combiner et mettre en majuscules (convention habituelle)
                all_feats = price_cols + vol_cols + ind_cols
                features_config[tf] = [f.upper() for f in all_feats]
        else:
            # Fallback ancienne logique si structure différente
            indicators = feat_conf.get('indicators', {})
            base_feats = feat_conf.get('base', ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
            for tf, feats in indicators.items():
                features_config[tf] = base_feats + feats
            
        print(f"✅ Config features extraite pour: {list(features_config.keys())}")
        # print(f"DEBUG: 5m features: {features_config.get('5m')}")
            
        print(f"✅ Config features extraite pour: {list(features_config.keys())}")
    except Exception as e:
        print(f"⚠️ Erreur extraction config, utilisation défaut: {e}")
        features_config = None

    # 2. Initialiser StateBuilder
    # Note: StateBuilder.__init__ prend features_config comme 1er argument
    builder = StateBuilder(features_config=features_config, config=config)
    
    # 3. Charger les scalers
    if hasattr(builder, '_load_training_scalers'):
        builder._load_training_scalers()
    
    # 4. Simuler des données (ou charger un parquet récent si possible)
    print("\n📂 Chargement des données d'entraînement (5m, 1h, 4h)...")
    try:
        # Chemins absolus ou relatifs corrects
        base_path = 'data/processed/indicators/train/BTCUSDT'
        df_5m = pd.read_parquet(f'{base_path}/5m.parquet').tail(1000)
        df_1h = pd.read_parquet(f'{base_path}/1h.parquet').tail(500)
        df_4h = pd.read_parquet(f'{base_path}/4h.parquet').tail(300)
        
        # Structure attendue par build_observation: {asset: {tf: df}}
        data = {
            'BTCUSDT': {
                '5m': df_5m,
                '1h': df_1h,
                '4h': df_4h
            }
        }
        print("✅ Données chargées.")
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        # Essayer de créer des données dummy si fichiers absents
        print("⚠️ Création de données dummy...")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='5min')
        df_dummy = pd.DataFrame(np.random.randn(1000, 20), index=dates, columns=[f'col_{i}' for i in range(20)])
        # Ajouter colonnes obligatoires
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_dummy[col] = np.random.rand(1000) * 10000
        
        data = {
            'BTCUSDT': {
                '5m': df_dummy,
                '1h': df_dummy,
                '4h': df_dummy
            }
        }

    # 5. Construire l'état
    print("\n🏗️ Construction de l'état...")
    try:
        # On doit mocker le portfolio manager
        class MockPortfolio:
            def get_state_vector(self): 
                return np.zeros(17)
            
        # build_observation prend (current_idx, data, portfolio_manager)
        # current_idx est l'index entier dans le dataframe (généralement le dernier)
        current_idx = len(df_5m) - 1
        
        # Note: build_observation retourne un DICT d'observations par timeframe
        # Mais le modèle attend un vecteur concaténé (flattened)
        # Il faut voir comment l'environnement fait.
        # StateBuilder a une méthode 'build_adaptive_observation' qui retourne peut-être le vecteur concaténé ?
        # Ou alors l'environnement concatène lui-même.
        
        # Vérifions ce que retourne build_observation
        current_idx = len(df_5m) - 1
        obs_dict = builder.build_observation(current_idx, data, portfolio_manager=MockPortfolio())
        print(f"✅ Observations construites (dict keys): {obs_dict.keys()}")
        print(f"   DEBUG shapes BEFORE expand_dims: {dict((k, v.shape) for k, v in obs_dict.items())}")
        
        
        # Le modèle attend un DICT (pas un vecteur), avec batch dimension
        # Ajouter la dimension batch (expand_dims) pour chaque clé
        obs_batch = {}
        for key, val in obs_dict.items():
            obs_batch[key] = np.expand_dims(val, axis=0)
        
        print(f"✅ obs_batch créé avec keys: {obs_batch.keys()}")
        print(f"   Shapes: {[{k: v.shape} for k, v in obs_batch.items()]}")
        
        # 6. Analyser l'état (prendre 5m pour stats car c'est le plus grand)
        state_5m = obs_dict['5m']
        print("\n📊 STATISTIQUES DE L'ÉTAT 5m (SAMPLE):")
        print(f"  Min: {state_5m.min():.6f}")
        print(f"  Max: {state_5m.max():.6f}")
        print(f"  Mean: {state_5m.mean():.6f}")
        print(f"  Std: {state_5m.std():.6f}")
        
        if state_5m.std() < 0.001:
            print("  ⚠️ ALERTE: Variance quasi-nulle ! L'état est constant.")
        
        # 7. Charger le modèle et prédire
        print("\n🤖 Test de prédiction avec le modèle w1...")
        model_path = "checkpoints_final/final/w1_final.zip"
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            action, _ = model.predict(obs_batch, deterministic=True)
            print(f"  👉 Action prédite: {action}")
            
            if abs(action - 1.0) < 0.001:
                print("  ❌ PRÉDICTION TOUJOURS 1.0")
            else:
                print("  ✅ PRÉDICTION VARIÉE (Succès sur données train)")
        else:
            print("❌ Modèle non trouvé.")

    except Exception as e:
        print(f"❌ Erreur pendant construction/prédiction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_prediction()
