#!/usr/bin/env python3
"""Vérification du CNN et PPO ADAN"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_cnn_ppo():
    """Vérifie le CNN et PPO"""
    print("🧠 VÉRIFICATION CNN & PPO")
    print("="*60)

    # 1. Vérifier PyTorch
    try:
        import torch
        print("🔥 PyTorch:")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA disponible: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch non disponible: {e}")
        return

    # 2. Vérifier la structure du CNN
    print("\n📐 STRUCTURE CNN:")
    input_shape = (1, 3, 20, 14)  # Batch=1, 3 timeframes, window=20, features=14
    print(f"  Shape d'entrée attendue: {input_shape}")
    
    # Créer un tenseur factice
    fake_input = torch.randn(input_shape)
    print(f"  Tenseur factice créé: {fake_input.shape}")
    print(f"  Valeurs: [{fake_input.min():.3f}, {fake_input.max():.3f}]")

    # 3. Vérifier les canaux (timeframes)
    print("\n🎯 CANAUX (TIMEFRAMES):")
    timeframes = ['5m', '1h', '4h']
    for i, tf in enumerate(timeframes):
        channel_data = fake_input[0, i, :, :]  # Canal i
        print(f"  {tf}: {channel_data.shape}, Mean: {channel_data.mean():.3f}")

    # 4. Vérifier la confusion potentielle
    print("\n🔍 DÉTECTION DE CONFUSION:")
    channel_similarities = []
    for i in range(3):
        for j in range(i+1, 3):
            sim = torch.cosine_similarity(
                fake_input[0, i].flatten(),
                fake_input[0, j].flatten(),
                dim=0
            )
            channel_similarities.append(sim.item())
    
    avg_similarity = np.mean(channel_similarities)
    print(f"  Similarité moyenne entre canaux: {avg_similarity:.3f}")
    
    if avg_similarity > 0.8:
        print("  ⚠️  RISQUE: Canaux trop similaires - CNN peut les confondre")
    elif avg_similarity > 0.5:
        print("  ℹ️  INFO: Canaux modérément similaires")
    else:
        print("  ✅ OK: Canaux suffisamment distincts")

    # 5. Vérifier PPO
    print("\n🤖 VÉRIFICATION PPO:")
    try:
        from stable_baselines3 import PPO
        
        model_paths = [
            "/mnt/new_data/t10_training/checkpoints/final/w1",
            "/mnt/new_data/t10_training/checkpoints/final/w2",
            "/mnt/new_data/t10_training/checkpoints/final/w3",
            "/mnt/new_data/t10_training/checkpoints/final/w4"
        ]
        
        for i, path in enumerate(model_paths, 1):
            if os.path.exists(path + ".zip"):
                print(f"  W{i}: ✅ Modèle trouvé")
                try:
                    model = PPO.load(path)
                    print(f"       Chargé avec succès")
                except Exception as e:
                    print(f"       ⚠️  Erreur chargement: {e}")
            else:
                print(f"  W{i}: ❌ Modèle non trouvé")
    except ImportError as e:
        print(f"  ❌ Stable-Baselines3 non disponible: {e}")

    # 6. Recommandations
    print("\n🎯 RECOMMANDATIONS:")
    if avg_similarity > 0.8:
        print("1. ⚠️  Normaliser les canaux séparément")
        print("2. ⚠️  Ajouter du dropout dans le CNN")
        print("3. ⚠️  Vérifier que les indicateurs sont distincts par timeframe")
    else:
        print("1. ✅ CNN devrait bien différencier les timeframes")
    print("4. Vérifier que chaque worker utilise son propre modèle")
    print("5. Vérifier la cohérence entre entraînement et inference")

if __name__ == "__main__":
    verify_cnn_ppo()
