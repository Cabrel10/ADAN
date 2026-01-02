#!/usr/bin/python3
"""Script principal pour l'optimisation du modèle d'attention ADAN."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from .config.experiment_config import OptimizationConfig, setup_mlflow
from .models.attention_model import OptimizedIntegratedCNNPPOModel, create_model_for_optuna
from .monitoring.experiment_tracker import ExperimentTracker, TrainingMonitor
from .utils.profiling import profile_model_inference, log_memory_usage, benchmark_model
from .tests.load_testing import run_load_test_suite, print_load_test_summary


def create_dummy_dataset(n_samples: int = 1000, input_shape: Tuple[int, ...] = (50, 10),
                        n_timeframes: int = 3) -> Tuple[DataLoader, DataLoader, Tuple, int]:
    """
    Crée un dataset factice pour les tests et l'optimisation.

    Args:
        n_samples: Nombre d'échantillons
        input_shape: Forme des données d'entrée (sans batch et timeframes)
        n_timeframes: Nombre de timeframes

    Returns:
        Tuple (train_loader, val_loader, input_shape, output_dim)
    """
    # Générer des données factices
    np.random.seed(42)

    # Données d'entrée: [n_samples, n_timeframes, window_size, n_features]
    X = np.random.randn(n_samples, n_timeframes, *input_shape).astype(np.float32)

    # État du portefeuille: [n_samples, portfolio_features]
    portfolio_state = np.random.randn(n_samples, 10).astype(np.float32)

    # Cibles (actions): [n_samples, 2] pour mean et std de chaque action
    y = np.random.randn(n_samples, 4).astype(np.float32)  # 2 actions × 2 paramètres

    # Séparer train/validation
    split_idx = int(n_samples * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    portfolio_train, portfolio_val = portfolio_state[:split_idx], portfolio_state[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Créer les datasets
    def create_timeframe_dict(X_batch, portfolio_batch):
        """Crée le dictionnaire d'observations pour le modèle"""
        return {
            '5m': torch.FloatTensor(X_batch[:, 0]),  # Premier timeframe
            '1h': torch.FloatTensor(X_batch[:, 1]),  # Deuxième timeframe
            '4h': torch.FloatTensor(X_batch[:, 2]),  # Troisième timeframe
            'portfolio_state': torch.FloatTensor(portfolio_batch)
        }

    # Datasets personnalisés
    class CustomDataset(TensorDataset):
        def __init__(self, X, portfolio, y):
            super().__init__(torch.FloatTensor(X), torch.FloatTensor(portfolio), torch.FloatTensor(y))

        def __getitem__(self, idx):
            X_batch, portfolio_batch, y_batch = super().__getitem__(idx)
            return create_timeframe_dict(X_batch.unsqueeze(0), portfolio_batch.unsqueeze(0)), y_batch

    train_dataset = CustomDataset(X_train, portfolio_train, y_train)
    val_dataset = CustomDataset(X_val, portfolio_val, y_val)

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, input_shape, 4  # 4 sorties (2 actions × 2 paramètres)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               config: OptimizationConfig, device: torch.device,
               tracker: ExperimentTracker) -> Dict[str, float]:
    """
    Entraîne le modèle avec suivi des métriques.

    Args:
        model: Le modèle à entraîner
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        config: Configuration d'optimisation
        device: Appareil à utiliser
        tracker: Tracker d'expériences

    Returns:
        Dictionnaire des métriques finales
    """
    model.to(device)

    # Fonction de perte
    def loss_fn(outputs, targets):
        """Fonction de perte pour les paramètres d'action"""
        return nn.MSELoss()(outputs, targets)

    # Optimiseur
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rates[0])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=config.early_stopping_patience//2, factor=0.5, verbose=True
    )

    # Historique des métriques
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("Démarrage de l'entraînement...")

    for epoch in range(config.n_epochs):
        # Mode entraînement
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_samples += targets.size(0)

        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)

        # Mode évaluation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                val_samples += targets.size(0)

        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)

        # Mise à jour du scheduler
        scheduler.step(avg_val_loss)

        # Enregistrer les métriques
        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        tracker.log_metrics(metrics, step=epoch)

        # Vérifier l'amélioration
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')

        print(f"Époque {epoch+1}/{config.n_epochs}: "
              f"Train Loss: {avg_train_loss".4f"}, "
              f"Val Loss: {avg_val_loss".4f"}, "
              f"LR: {optimizer.param_groups[0]['lr']".6f"}")

        # Early stopping
        if epoch > config.early_stopping_patience:
            recent_losses = val_losses[-config.early_stopping_patience:]
            if all(loss >= best_val_loss for loss in recent_losses):
                print(f"Early stopping à l'époque {epoch+1}")
                break

    # Métriques finales
    final_metrics = {
        'best_train_loss': min(train_losses),
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'epochs_trained': len(train_losses),
        'convergence_epoch': train_losses.index(min(train_losses)) if train_losses else 0
    }

    return final_metrics


def optimize_hyperparameters(config: OptimizationConfig) -> Dict[str, Any]:
    """
    Optimise les hyperparamètres du modèle d'attention.

    Args:
        config: Configuration d'optimisation

    Returns:
        Dictionnaire contenant les résultats d'optimisation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de l'appareil: {device}")

    # Créer les données
    print("Création du dataset...")
    train_loader, val_loader, input_shape, output_dim = create_dummy_dataset()

    # Configuration MLflow
    experiment_name = setup_mlflow()

    # Résultats d'optimisation
    optimization_results = {
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__ if hasattr(config, '__dict__') else config,
        'device': str(device),
        'input_shape': input_shape,
        'output_dim': output_dim,
        'results': []
    }

    # Tester toutes les combinaisons d'hyperparamètres
    total_combinations = (len(config.num_attention_heads) *
                         len(config.hidden_dims) *
                         len(config.learning_rates))

    print(f"Test de {total_combinations} combinaisons d'hyperparamètres...")

    best_val_loss = float('inf')
    best_params = None
    best_model = None

    combination_idx = 0

    for num_heads in config.num_attention_heads:
        for hidden_dim in config.hidden_dims:
            for lr in config.learning_rates:
                combination_idx += 1
                run_name = f"opt_h{num_heads}_d{hidden_dim}_lr{lr".0e"}"

                print(f"\n[{combination_idx}/{total_combinations}] Test: {run_name}")

                with ExperimentTracker(experiment_name).start_run(run_name) as tracker:
                    # Enregistrer les paramètres
                    params = {
                        'num_attention_heads': num_heads,
                        'hidden_dim': hidden_dim,
                        'learning_rate': lr,
                        'batch_size': config.batch_size,
                        'n_epochs': config.n_epochs
                    }
                    tracker.log_params(params)

                    # Créer le modèle
                    model = OptimizedIntegratedCNNPPOModel(
                        input_shape=(3, *input_shape),  # 3 timeframes
                        action_dim=2,
                        use_attention=True,
                        use_multiscale=True
                    )

                    # Mettre à jour les paramètres d'attention
                    model.ppo.temporal_attention = nn.Module()  # Placeholder - à implémenter
                    model.ppo.temporal_attention.num_heads = num_heads

                    # Entraîner le modèle
                    try:
                        final_metrics = train_model(
                            model, train_loader, val_loader, config, device, tracker
                        )

                        # Enregistrer les métriques finales
                        tracker.log_metrics(final_metrics)

                        # Sauvegarder le modèle
                        tracker.log_model(model, f"model_{run_name}")

                        # Mettre à jour les meilleurs résultats
                        if final_metrics['best_val_loss'] < best_val_loss:
                            best_val_loss = final_metrics['best_val_loss']
                            best_params = params
                            best_model = model.state_dict()

                        # Ajouter aux résultats
                        result = {
                            'combination': run_name,
                            'params': params,
                            'metrics': final_metrics,
                            'success': True
                        }

                    except Exception as e:
                        print(f"Erreur lors de l'entraînement: {str(e)}")
                        result = {
                            'combination': run_name,
                            'params': params,
                            'error': str(e),
                            'success': False
                        }

                    optimization_results['results'].append(result)

    # Sauvegarder le meilleur modèle
    if best_model is not None:
        torch.save({
            'model_state_dict': best_model,
            'best_params': best_params,
            'best_val_loss': best_val_loss
        }, 'best_optimized_model.pth')

        print("
🏆 Meilleurs résultats:"        print(f"   Validation Loss: {best_val_loss".4f"}")
        print(f"   Paramètres: {best_params}")

    # Enregistrer les résultats d'optimisation
    with open('optimization_results.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)

    return optimization_results


def run_benchmark(model_path: str = None, batch_sizes: list = None) -> Dict[str, Any]:
    """
    Exécute un benchmark des performances du modèle.

    Args:
        model_path: Chemin vers le modèle à benchmarker
        batch_sizes: Tailles de lot à tester

    Returns:
        Dictionnaire contenant les résultats du benchmark
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path and os.path.exists(model_path):
        # Charger le modèle optimisé
        checkpoint = torch.load(model_path, map_location=device)
        model = OptimizedIntegratedCNNPPOModel(
            input_shape=(3, 50, 10),  # À adapter selon les besoins
            action_dim=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle chargé depuis {model_path}")
    else:
        # Créer un modèle par défaut
        model = OptimizedIntegratedCNNPPOModel(
            input_shape=(3, 50, 10),
            action_dim=2,
            use_attention=True,
            use_multiscale=True
        )
        print("Modèle par défaut créé")

    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64, 128]

    # Exécuter le benchmark
    print("🚀 Exécution du benchmark...")
    benchmark_results = run_load_test_suite(
        model=model,
        input_shape=(50, 10),
        batch_sizes=batch_sizes,
        num_batches=25
    )

    # Afficher le résumé
    print_load_test_summary(benchmark_results)

    # Sauvegarder les résultats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    benchmark_file = f'benchmark_results_{timestamp}.json'
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"Résultats du benchmark sauvegardés dans {benchmark_file}")

    return benchmark_results


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Optimisation du modèle d\'attention ADAN')
    parser.add_argument('--mode', choices=['optimize', 'benchmark', 'train'],
                       default='optimize', help='Mode d\'exécution')
    parser.add_argument('--config', type=str, help='Chemin vers le fichier de configuration')
    parser.add_argument('--model_path', type=str, help='Chemin vers le modèle à charger')
    parser.add_argument('--batch_sizes', nargs='+', type=int,
                       default=[1, 8, 16, 32, 64], help='Tailles de lot pour le benchmark')

    args = parser.parse_args()

    # Configuration
    if args.config:
        config = OptimizationConfig.load_config_from_yaml(args.config)
    else:
        config = OptimizationConfig()

    print("🔧 Configuration:")
    print(f"   Mode: {args.mode}")
    print(f"   Têtes d'attention: {config.num_attention_heads}")
    print(f"   Dimensions cachées: {config.hidden_dims}")
    print(f"   Taux d'apprentissage: {config.learning_rates}")
    print(f"   Époques: {config.n_epochs}")
    print(f"   Batch size: {config.batch_size}")

    if args.mode == 'optimize':
        print("\n🎯 Démarrage de l'optimisation des hyperparamètres...")
        results = optimize_hyperparameters(config)

        print("\n✅ Optimisation terminée!")
        print(f"   Meilleure perte de validation: {results.get('best_val_loss', 'N/A')}")
        if 'best_params' in results:
            print(f"   Meilleurs paramètres: {results['best_params']}")

    elif args.mode == 'benchmark':
        print("\n📊 Démarrage du benchmark...")
        benchmark_results = run_benchmark(args.model_path, args.batch_sizes)
        print("\n✅ Benchmark terminé!")

    elif args.mode == 'train':
        print("\n🏋️ Démarrage de l'entraînement...")
        # Implémentation de l'entraînement standard
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Créer les données et le modèle
        train_loader, val_loader, input_shape, output_dim = create_dummy_dataset()

        model = OptimizedIntegratedCNNPPOModel(
            input_shape=(3, *input_shape),
            action_dim=2,
            use_attention=True,
            use_multiscale=True
        )

        # Configuration MLflow
        experiment_name = setup_mlflow("adan_training")

        # Entraînement avec monitoring
        with ExperimentTracker(experiment_name).start_run("training_run") as tracker:
            final_metrics = train_model(model, train_loader, val_loader, config, device, tracker)

            # Enregistrer le modèle final
            tracker.log_model(model, "final_trained_model")
            tracker.log_metrics(final_metrics)

            print("\n✅ Entraînement terminé!")
            print(f"   Meilleure perte de validation: {final_metrics['best_val_loss']".4f"}")

    else:
        print(f"❌ Mode non reconnu: {args.mode}")
        print("Modes disponibles: optimize, benchmark, train")


if __name__ == "__main__":
    main()
