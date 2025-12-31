#!/usr/bin/python3
"""Exemple d'intégration du module d'optimisation dans le code ADAN existant."""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import torch
import numpy as np
from adan_trading_bot.optimization import (
    OptimizationConfig,
    OptimizedIntegratedCNNPPOModel,
    ExperimentTracker,
    run_load_test_suite,
    profile_model_inference
)


def example_1_basic_optimization():
    """Exemple 1: Optimisation basique du modèle."""
    print("🎯 Exemple 1: Optimisation basique")
    print("-" * 40)

    # Configuration simple
    config = OptimizationConfig(
        num_attention_heads=[4, 8],
        hidden_dims=[128, 256],
        learning_rates=[1e-4, 3e-4],
        n_epochs=5,  # Réduit pour l'exemple
        batch_size=32
    )

    # Simulation d'optimisation (version simplifiée)
    best_params = {
        'num_attention_heads': 8,
        'hidden_dim': 256,
        'learning_rate': 3e-4
    }

    print(f"Configuration testée: {config.num_attention_heads} heads, {config.hidden_dims} hidden dims")
    print(f"Meilleurs paramètres trouvés: {best_params}")

    return best_params


def example_2_model_creation():
    """Exemple 2: Création et test d'un modèle optimisé."""
    print("\n🏗️ Exemple 2: Création de modèle optimisé")
    print("-" * 40)

    # Créer un modèle optimisé
    model = OptimizedIntegratedCNNPPOModel(
        input_shape=(3, 50, 10),  # 3 timeframes, 50 time steps, 10 features
        action_dim=2,  # 2 actions (buy/sell)
        use_attention=True,
        use_multiscale=True
    )

    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")

    # Générer des données d'exemple
    batch_size = 4
    dummy_input = {
        '5m': torch.randn(batch_size, 50, 10),
        '1h': torch.randn(batch_size, 50, 10),
        '4h': torch.randn(batch_size, 50, 10),
        'portfolio_state': torch.randn(batch_size, 10)
    }

    # Test du modèle
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Sortie du modèle: {output.keys()}")
    print(f"Shape action_mean: {output['action_mean'].shape}")
    print(f"Shape action_std: {output['action_std'].shape}")
    print(f"Shape value: {output['value'].shape}")

    return model


def example_3_performance_monitoring():
    """Exemple 3: Monitoring des performances."""
    print("\n📊 Exemple 3: Monitoring des performances")
    print("-" * 40)

    # Créer un modèle
    model = OptimizedIntegratedCNNPPOModel(
        input_shape=(3, 50, 10),
        action_dim=2
    )

    # Profiler le modèle
    print("Profilage du modèle...")
    profile_results = profile_model_inference(
        model=model,
        input_shape=(3, 50, 10),
        device='cpu',  # Utiliser CPU pour l'exemple
        n_iter=10
    )

    print(f"Temps par itération: {profile_results['time_per_iter']*1000".2f"} ms")
    print(f"Débit: {profile_results['throughput_samples_per_sec']".2f"} échantillons/s")
    print(f"Mémoire utilisée: {profile_results['memory_used_mb']".2f"} Mo")

    return profile_results


def example_4_mlflow_integration():
    """Exemple 4: Intégration avec MLflow."""
    print("\n📈 Exemple 4: Intégration MLflow")
    print("-" * 40)

    # Configuration MLflow
    experiment_name = "adan_example_integration"

    with ExperimentTracker(experiment_name) as tracker:
        # Enregistrer des paramètres
        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'model_type': 'OptimizedIntegratedCNNPPOModel'
        }
        tracker.log_params(params)

        # Enregistrer des métriques
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.3,
            'accuracy': 0.85
        }
        tracker.log_metrics(metrics)

        # Créer un modèle simple pour l'exemple
        model = OptimizedIntegratedCNNPPOModel(
            input_shape=(3, 50, 10),
            action_dim=2
        )

        # Enregistrer le modèle
        tracker.log_model(model, "example_model")

        print(f"Expérience '{experiment_name}' créée avec succès")
        print(f"Run ID: {tracker.run_id}")

    return tracker.run_id


def example_5_custom_training_loop():
    """Exemple 5: Boucle d'entraînement personnalisée."""
    print("\n🏋️ Exemple 5: Entraînement personnalisé")
    print("-" * 40)

    # Créer le modèle
    model = OptimizedIntegratedCNNPPOModel(
        input_shape=(3, 50, 10),
        action_dim=2
    )

    # Configuration d'entraînement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Générer des données d'exemple
    n_samples = 100
    dummy_data = []

    for _ in range(n_samples):
        # Données d'entrée
        inputs = {
            '5m': torch.randn(50, 10),
            '1h': torch.randn(50, 10),
            '4h': torch.randn(50, 10),
            'portfolio_state': torch.randn(10)
        }

        # Cibles (actions)
        targets = torch.randn(4)  # 2 actions × 2 paramètres (mean, std)

        dummy_data.append((inputs, targets))

    # Boucle d'entraînement simple
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print("Entraînement sur 10 époques...")

    for epoch in range(10):
        epoch_loss = 0.0
        n_batches = 0

        for inputs, targets in dummy_data:
            # Forward pass
            inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
            targets = targets.unsqueeze(0).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs['action_mean'].squeeze(), targets[:2])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"Époque {epoch+1:2d}: loss = {avg_loss".4f"}")

    print("Entraînement terminé!")

    return model


def example_6_attention_analysis():
    """Exemple 6: Analyse des poids d'attention."""
    print("\n🔍 Exemple 6: Analyse d'attention")
    print("-" * 40)

    # Créer un modèle avec attention
    model = OptimizedIntegratedCNNPPOModel(
        input_shape=(3, 20, 5),  # Plus petit pour l'exemple
        action_dim=2,
        use_attention=True
    )

    # Générer des données et faire des prédictions
    dummy_input = {
        '5m': torch.randn(20, 5),
        '1h': torch.randn(20, 5),
        '4h': torch.randn(20, 5),
        'portfolio_state': torch.randn(10)
    }

    # Mode entraînement pour collecter les poids d'attention
    model.train()
    for _ in range(5):
        with torch.no_grad():
            _ = model({k: v.unsqueeze(0) for k, v in dummy_input.items()})

    # Récupérer les statistiques d'attention
    attention_stats = model.get_attention_stats()

    if attention_stats:
        print("Statistiques d'attention:")
        for key, value in attention_stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value".4f"}")
            else:
                print(f"  {key}: shape {np.array(value).shape}")
    else:
        print("Aucune statistique d'attention disponible")

    # Effacer les poids d'attention
    model.clear_attention_weights()
    print("Poids d'attention effacés")

    return attention_stats


def run_all_examples():
    """Exécute tous les exemples."""
    print("🚀 Exécution des exemples d'intégration")
    print("=" * 50)

    try:
        # Exemple 1
        best_params = example_1_basic_optimization()

        # Exemple 2
        model = example_2_model_creation()

        # Exemple 3
        profile_results = example_3_performance_monitoring()

        # Exemple 4
        run_id = example_4_mlflow_integration()

        # Exemple 5
        trained_model = example_5_custom_training_loop()

        # Exemple 6
        attention_stats = example_6_attention_analysis()

        print("\n🎉 Tous les exemples ont été exécutés avec succès!")
        print("\n📋 Résumé:")
        print(f"   - Meilleurs paramètres: {best_params}")
        print(f"   - Modèle créé: {type(model).__name__}")
        print(f"   - Run MLflow: {run_id}")
        print(f"   - Attention stats: {len(attention_stats)} métriques")

        return {
            'best_params': best_params,
            'model': model,
            'profile_results': profile_results,
            'mlflow_run_id': run_id,
            'attention_stats': attention_stats
        }

    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution des exemples: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Fonction principale."""
    print("🔬 Exemples d'intégration du module d'optimisation ADAN")
    print("=" * 60)

    # Vérifier les dépendances
    required_modules = [
        'torch', 'numpy', 'pandas', 'mlflow'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print(f"❌ Modules manquants: {missing_modules}")
        print("💡 Lancez: pip install -r requirements.txt")
        return

    print("✅ Toutes les dépendances sont disponibles")

    # Exécuter les exemples
    results = run_all_examples()

    if results:
        print("\n💾 Pour sauvegarder les résultats:")
        print("   import torch")
        print("   torch.save(results['model'].state_dict(), 'optimized_model.pth')")
        print("   print('Modèle sauvegardé dans optimized_model.pth')")


if __name__ == "__main__":
    main()
