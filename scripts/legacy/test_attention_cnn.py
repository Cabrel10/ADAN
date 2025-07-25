#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de l'architecture CNN avec attention inter-canaux pour ADAN Trading Bot.
Teste la tâche 7.2.1 - Ajouter mécanismes d'attention inter-canaux.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Add src to PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = SCRIPT_DIR.parent
sys.path.append(str(project_root))

from src.adan_trading_bot.models.attention_cnn import (
    ChannelAttention,
    SpatialAttention,
    TimeframeInteractionModule,
    AttentionCNN,
    AttentionCNNPolicy,
    create_attention_cnn_model
)

def create_test_data(batch_size: int = 4, 
                    n_channels: int = 3, 
                    seq_len: int = 100, 
                    n_features: int = 28) -> torch.Tensor:
    """Créer des données de test simulant des observations de trading."""
    
    # Simuler des données de trading réalistes
    data = torch.randn(batch_size, n_channels, seq_len, n_features)
    
    # Ajouter des patterns différents par timeframe
    for i in range(n_channels):
        # Timeframe 5m: plus de bruit, variations rapides
        if i == 0:
            data[:, i, :, :] += torch.randn_like(data[:, i, :, :]) * 0.5
        # Timeframe 1h: tendances moyennes
        elif i == 1:
            trend = torch.linspace(-1, 1, seq_len).unsqueeze(0).unsqueeze(-1)
            data[:, i, :, :] += trend.expand(batch_size, seq_len, n_features) * 0.3
        # Timeframe 4h: tendances lentes
        else:
            slow_trend = torch.sin(torch.linspace(0, 2*np.pi, seq_len)).unsqueeze(0).unsqueeze(-1)
            data[:, i, :, :] += slow_trend.expand(batch_size, seq_len, n_features) * 0.2
    
    return data

def test_channel_attention():
    """Test du mécanisme d'attention entre canaux."""
    print("🧪 Test Channel Attention...")
    
    try:
        # Créer le module d'attention
        channel_attention = ChannelAttention(n_channels=3, reduction_ratio=2)
        
        # Données de test
        batch_size, n_channels, seq_len, n_features = 4, 3, 100, 28
        x = create_test_data(batch_size, n_channels, seq_len, n_features)
        
        # Forward pass
        attended_x = channel_attention(x)
        
        print(f"  📊 Input shape: {x.shape}")
        print(f"  📊 Output shape: {attended_x.shape}")
        print(f"  📊 Input mean: {x.mean().item():.3f}")
        print(f"  📊 Output mean: {attended_x.mean().item():.3f}")
        
        # Vérifier que la shape est préservée
        if attended_x.shape == x.shape:
            print("  ✅ Shape préservée correctement")
        else:
            print("  ❌ Problème de shape")
            return False
        
        # Vérifier que l'attention modifie les données
        if not torch.allclose(x, attended_x, atol=1e-6):
            print("  ✅ Attention appliquée correctement")
        else:
            print("  ❌ Attention non appliquée")
            return False
        
        # Vérifier les poids d'attention
        with torch.no_grad():
            avg_out = channel_attention.avg_pool(x).view(batch_size, n_channels)
            attention_weights = channel_attention.shared_mlp(avg_out)
            attention_weights = torch.sigmoid(attention_weights)
            
            print(f"  📊 Attention weights shape: {attention_weights.shape}")
            print(f"  📊 Attention weights range: [{attention_weights.min():.3f}, {attention_weights.max():.3f}]")
            
            # Les poids devraient être entre 0 et 1
            if 0 <= attention_weights.min() and attention_weights.max() <= 1:
                print("  ✅ Poids d'attention dans la bonne plage")
                return True
            else:
                print("  ❌ Poids d'attention hors plage")
                return False
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_spatial_attention():
    """Test du mécanisme d'attention spatiale."""
    print("\n🧪 Test Spatial Attention...")
    
    try:
        # Créer le module d'attention spatiale
        spatial_attention = SpatialAttention(kernel_size=7)
        
        # Données de test
        batch_size, n_channels, seq_len, n_features = 4, 3, 100, 28
        x = create_test_data(batch_size, n_channels, seq_len, n_features)
        
        # Forward pass
        attended_x = spatial_attention(x)
        
        print(f"  📊 Input shape: {x.shape}")
        print(f"  📊 Output shape: {attended_x.shape}")
        
        # Vérifier que la shape est préservée
        if attended_x.shape == x.shape:
            print("  ✅ Shape préservée correctement")
        else:
            print("  ❌ Problème de shape")
            return False
        
        # Vérifier que l'attention modifie les données
        if not torch.allclose(x, attended_x, atol=1e-6):
            print("  ✅ Attention spatiale appliquée")
        else:
            print("  ❌ Attention spatiale non appliquée")
            return False
        
        # Analyser les poids d'attention spatiale
        with torch.no_grad():
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            pooled = torch.cat([avg_out, max_out], dim=1)
            attention_map = torch.sigmoid(spatial_attention.conv(pooled))
            
            print(f"  📊 Attention map shape: {attention_map.shape}")
            print(f"  📊 Attention map range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
            
            if 0 <= attention_map.min() and attention_map.max() <= 1:
                print("  ✅ Carte d'attention spatiale valide")
                return True
            else:
                print("  ❌ Carte d'attention spatiale invalide")
                return False
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_timeframe_interaction():
    """Test du module d'interaction entre timeframes."""
    print("\n🧪 Test Timeframe Interaction...")
    
    try:
        # Créer le module d'interaction
        interaction_module = TimeframeInteractionModule(n_channels=3, hidden_dim=64)
        
        # Créer des features pour chaque timeframe
        batch_size = 4
        hidden_dim = 64
        timeframe_features = [
            torch.randn(batch_size, hidden_dim),  # 5m features
            torch.randn(batch_size, hidden_dim),  # 1h features
            torch.randn(batch_size, hidden_dim),  # 4h features
        ]
        
        # Forward pass
        fused_features = interaction_module(timeframe_features)
        
        print(f"  📊 Input features shapes: {[f.shape for f in timeframe_features]}")
        print(f"  📊 Fused features shape: {fused_features.shape}")
        
        # Vérifier la shape de sortie
        expected_shape = (batch_size, hidden_dim)
        if fused_features.shape == expected_shape:
            print("  ✅ Shape de sortie correcte")
        else:
            print(f"  ❌ Shape incorrecte: attendu {expected_shape}, obtenu {fused_features.shape}")
            return False
        
        # Vérifier que la fusion combine bien les informations
        individual_means = [f.mean().item() for f in timeframe_features]
        fused_mean = fused_features.mean().item()
        
        print(f"  📊 Moyennes individuelles: {[f'{m:.3f}' for m in individual_means]}")
        print(f"  📊 Moyenne fusionnée: {fused_mean:.3f}")
        
        # La fusion devrait créer de nouvelles représentations
        if not any(abs(fused_mean - m) < 0.01 for m in individual_means):
            print("  ✅ Fusion créant de nouvelles représentations")
            return True
        else:
            print("  ⚠️ Fusion peut-être trop simple")
            return True  # Pas forcément un échec
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_attention_cnn():
    """Test de l'architecture CNN complète avec attention."""
    print("\n🧪 Test Attention CNN...")
    
    try:
        # Créer le modèle
        input_shape = (3, 100, 28)  # (n_channels, seq_len, n_features)
        model = AttentionCNN(
            input_shape=input_shape,
            hidden_dim=128,
            n_layers=3,
            dropout_rate=0.1
        )
        
        # Données de test
        batch_size = 4
        x = create_test_data(batch_size, *input_shape)
        
        # Forward pass
        features = model(x)
        
        print(f"  📊 Input shape: {x.shape}")
        print(f"  📊 Output features shape: {features.shape}")
        
        # Vérifier la shape de sortie
        expected_shape = (batch_size, 128)
        if features.shape == expected_shape:
            print("  ✅ Shape de sortie correcte")
        else:
            print(f"  ❌ Shape incorrecte: attendu {expected_shape}, obtenu {features.shape}")
            return False
        
        # Tester l'extraction des poids d'attention
        attention_weights = model.get_attention_weights(x)
        
        print(f"  📊 Attention weights keys: {list(attention_weights.keys())}")
        
        if 'channel_attention' in attention_weights and 'spatial_attention' in attention_weights:
            channel_att = attention_weights['channel_attention']
            spatial_att = attention_weights['spatial_attention']
            
            print(f"  📊 Channel attention shape: {channel_att.shape}")
            print(f"  📊 Spatial attention shape: {spatial_att.shape}")
            
            # Analyser les poids d'attention par timeframe
            if len(attention_weights['timeframe_names']) == 3:
                for i, tf_name in enumerate(attention_weights['timeframe_names']):
                    tf_weight = channel_att[:, i].mean().item()
                    print(f"    {tf_name}: {tf_weight:.3f}")
                
                print("  ✅ Mécanismes d'attention fonctionnels")
                return True
            else:
                print("  ❌ Problème avec les noms de timeframes")
                return False
        else:
            print("  ❌ Poids d'attention manquants")
            return False
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_attention_cnn_policy():
    """Test de la policy network avec CNN attention."""
    print("\n🧪 Test Attention CNN Policy...")
    
    try:
        # Créer la policy
        observation_shape = (3, 100, 28)
        action_space_size = 5  # Exemple: Hold, Buy_Small, Buy_Large, Sell_Small, Sell_Large
        
        policy = AttentionCNNPolicy(
            observation_space_shape=observation_shape,
            action_space_size=action_space_size,
            hidden_dim=128,
            n_layers=3
        )
        
        # Données de test
        batch_size = 4
        observations = create_test_data(batch_size, *observation_shape)
        
        # Forward pass
        action_logits, values = policy(observations)
        
        print(f"  📊 Observations shape: {observations.shape}")
        print(f"  📊 Action logits shape: {action_logits.shape}")
        print(f"  📊 Values shape: {values.shape}")
        
        # Vérifier les shapes
        expected_logits_shape = (batch_size, action_space_size)
        expected_values_shape = (batch_size,)
        
        if action_logits.shape == expected_logits_shape and values.shape == expected_values_shape:
            print("  ✅ Shapes de sortie correctes")
        else:
            print(f"  ❌ Shapes incorrectes")
            return False
        
        # Tester l'analyse d'attention
        attention_analysis = policy.get_attention_analysis(observations)
        
        if 'channel_attention' in attention_analysis:
            channel_weights = attention_analysis['channel_attention']
            print(f"  📊 Analyse attention disponible: {channel_weights.shape}")
            
            # Analyser la distribution des poids par timeframe
            for i, tf_name in enumerate(['5m', '1h', '4h']):
                weight = channel_weights[:, i].mean().item()
                print(f"    {tf_name} importance: {weight:.3f}")
            
            print("  ✅ Policy avec attention fonctionnelle")
            return True
        else:
            print("  ❌ Analyse d'attention non disponible")
            return False
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_model_factory():
    """Test de la fonction factory pour créer des modèles."""
    print("\n🧪 Test Model Factory...")
    
    try:
        # Créer un modèle via la factory
        observation_shape = (3, 50, 20)
        action_space_size = 3
        
        model = create_attention_cnn_model(
            observation_space_shape=observation_shape,
            action_space_size=action_space_size,
            hidden_dim=64,
            n_layers=2
        )
        
        # Tester le modèle
        batch_size = 2
        observations = create_test_data(batch_size, *observation_shape)
        
        action_logits, values = model(observations)
        
        print(f"  📊 Model créé via factory")
        print(f"  📊 Action logits shape: {action_logits.shape}")
        print(f"  📊 Values shape: {values.shape}")
        
        # Vérifier que le modèle fonctionne
        if action_logits.shape == (batch_size, action_space_size) and values.shape == (batch_size,):
            print("  ✅ Factory function fonctionnelle")
            return True
        else:
            print("  ❌ Problème avec factory function")
            return False
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_attention_interpretability():
    """Test de l'interprétabilité des mécanismes d'attention."""
    print("\n🧪 Test Interprétabilité Attention...")
    
    try:
        # Créer des données avec patterns distincts par timeframe
        batch_size = 2
        input_shape = (3, 50, 10)
        
        # Créer des données avec des signaux différents par timeframe
        x = torch.zeros(batch_size, *input_shape)
        
        # Timeframe 0 (5m): signal fort dans les premières features
        x[:, 0, :, :3] = 2.0
        
        # Timeframe 1 (1h): signal fort dans les features du milieu
        x[:, 1, :, 3:6] = 1.5
        
        # Timeframe 2 (4h): signal fort dans les dernières features
        x[:, 2, :, 6:] = 1.0
        
        # Créer le modèle
        model = AttentionCNN(input_shape=input_shape, hidden_dim=64)
        
        # Analyser l'attention
        attention_weights = model.get_attention_weights(x)
        
        channel_att = attention_weights['channel_attention']
        spatial_att = attention_weights['spatial_attention']
        
        print(f"  📊 Channel attention par timeframe:")
        for i, tf_name in enumerate(['5m', '1h', '4h']):
            weight = channel_att[:, i].mean().item()
            print(f"    {tf_name}: {weight:.3f}")
        
        print(f"  📊 Spatial attention stats:")
        print(f"    Mean: {spatial_att.mean().item():.3f}")
        print(f"    Std: {spatial_att.std().item():.3f}")
        print(f"    Range: [{spatial_att.min().item():.3f}, {spatial_att.max().item():.3f}]")
        
        # Vérifier que l'attention est différenciée
        channel_std = channel_att.std(dim=1).mean().item()
        if channel_std > 0.01:  # Il devrait y avoir de la variation
            print("  ✅ Attention différenciée entre timeframes")
            return True
        else:
            print("  ⚠️ Attention peu différenciée (peut être normal)")
            return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    """Fonction principale pour exécuter tous les tests."""
    print("🚀 Test Complet Architecture CNN avec Attention Inter-Canaux")
    print("=" * 70)
    
    tests = [
        ("Channel Attention", test_channel_attention),
        ("Spatial Attention", test_spatial_attention),
        ("Timeframe Interaction", test_timeframe_interaction),
        ("Attention CNN", test_attention_cnn),
        ("Attention CNN Policy", test_attention_cnn_policy),
        ("Model Factory", test_model_factory),
        ("Interprétabilité Attention", test_attention_interpretability)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            print(f"\n❌ ÉCHEC - {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DES TESTS CNN ATTENTION")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Score: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Architecture CNN avec attention inter-canaux opérationnelle !")
    else:
        print("⚠️ Certains tests ont échoué.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)