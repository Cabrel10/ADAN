"""Suivi d'expériences et monitoring pour l'optimisation."""

import mlflow
import mlflow.pytorch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from pathlib import Path
import tempfile


class ExperimentTracker:
    """Classe pour suivre les expériences d'entraînement"""

    def __init__(self, experiment_name: str, tracking_uri: str = "http://localhost:5000"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id = None
        self.setup_mlflow()

    def setup_mlflow(self):
        """Configure MLflow pour le suivi des expériences"""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        """Démarre une nouvelle exécution MLflow"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        return self

    def log_params(self, params: Dict[str, Any]):
        """Enregistre les paramètres de l'expérience"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Enregistre les métriques de l'expérience"""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: torch.nn.Module, model_name: str):
        """Enregistre le modèle entraîné"""
        # Créer un dossier temporaire pour sauvegarder le modèle
        model_path = f"models/{model_name}_{self.run_id}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Sauvegarder l'état du modèle
        torch.save(model.state_dict(), model_path)

        # Enregistrer le modèle dans MLflow
        mlflow.pytorch.log_model(model, "model")

        # Enregistrer également l'architecture
        with open(f"{model_path}_arch.txt", "w") as f:
            f.write(str(model))

        mlflow.log_artifact(f"{model_path}_arch.txt")

    def log_attention_weights(self, attention_weights: List[np.ndarray],
                            prefix: str = "attention", save_plots: bool = True):
        """Enregistre les poids d'attention pour analyse"""
        if not attention_weights:
            return

        # Calculer les statistiques des poids d'attention
        weights = np.array(attention_weights)

        # Statistiques globales
        stats = {
            f'{prefix}_mean': np.mean(weights),
            f'{prefix}_std': np.std(weights),
            f'{prefix}_max': np.max(weights),
            f'{prefix}_min': np.min(weights),
            f'{prefix}_entropy': self._calculate_attention_entropy(weights)
        }

        self.log_metrics(stats)

        # Créer et sauvegarder des visualisations si demandé
        if save_plots:
            self._create_attention_plots(weights, prefix)

    def _calculate_attention_entropy(self, weights: np.ndarray) -> float:
        """Calcule l'entropie des poids d'attention"""
        # Aplatir et normaliser les poids
        flat_weights = weights.flatten()
        flat_weights = flat_weights / np.sum(flat_weights)

        # Calcul de l'entropie
        entropy = -np.sum(flat_weights * np.log2(flat_weights + 1e-10))
        return entropy

    def _create_attention_plots(self, weights: np.ndarray, prefix: str):
        """Crée des visualisations des poids d'attention"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Configurer le style
            plt.style.use('default')
            sns.set_palette("husl")

            # 1. Heatmap des poids moyens
            plt.figure(figsize=(12, 8))
            avg_weights = np.mean(weights, axis=(0, 1))  # Moyenne sur batch et heads

            if avg_weights.ndim == 2:
                sns.heatmap(avg_weights, cmap='viridis', annot=False)
                plt.title(f"Moyenne des Poids d'Attention - {prefix}")
                plt.xlabel("Position Cible")
                plt.ylabel("Position Source")

                # Sauvegarder
                plt.savefig(f'attention_heatmap_{prefix}.png', dpi=300, bbox_inches='tight')
                mlflow.log_artifact(f'attention_heatmap_{prefix}.png')
                plt.close()

            # 2. Distribution des poids
            plt.figure(figsize=(10, 6))
            flat_weights = weights.flatten()
            plt.hist(flat_weights, bins=50, alpha=0.7, density=True)
            plt.title(f"Distribution des Poids d'Attention - {prefix}")
            plt.xlabel("Poids")
            plt.ylabel("Densité")

            plt.savefig(f'attention_distribution_{prefix}.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact(f'attention_distribution_{prefix}.png')
            plt.close()

            # 3. Évolution temporelle des poids moyens
            if weights.shape[1] > 1:  # Si on a plusieurs étapes temporelles
                plt.figure(figsize=(10, 6))
                mean_over_batch = np.mean(weights, axis=1)  # Moyenne sur le batch
                mean_over_heads = np.mean(mean_over_batch, axis=1)  # Moyenne sur les heads

                plt.plot(mean_over_heads)
                plt.title(f"Évolution des Poids d'Attention - {prefix}")
                plt.xlabel("Étape")
                plt.ylabel("Poids Moyen")

                plt.savefig(f'attention_evolution_{prefix}.png', dpi=300, bbox_inches='tight')
                mlflow.log_artifact(f'attention_evolution_{prefix}.png')
                plt.close()

        except ImportError:
            print("Matplotlib et/ou Seaborn non disponibles pour les visualisations")
        except Exception as e:
            print(f"Erreur lors de la création des visualisations: {e}")

    def log_training_artifacts(self, model: nn.Module, config: Dict[str, Any],
                             metrics: Dict[str, float], save_plots: bool = True):
        """Enregistre tous les artefacts d'une session d'entraînement"""

        # Enregistrer le modèle
        self.log_model(model, "trained_model")

        # Enregistrer la configuration
        config_path = f"config_{self.run_id}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path)

        # Enregistrer les métriques finales
        self.log_metrics(metrics)

        # Créer et enregistrer un résumé des performances
        summary = self._create_performance_summary(model, config, metrics)
        summary_path = f"performance_summary_{self.run_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        # Visualisations si demandé
        if save_plots:
            self._create_training_plots(metrics)

    def _create_performance_summary(self, model: nn.Module, config: Dict[str, Any],
                                  metrics: Dict[str, float]) -> Dict[str, Any]:
        """Crée un résumé des performances"""
        from ..utils.profiling import log_memory_usage

        summary = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'model_architecture': str(type(model).__name__),
            'model_parameters': log_memory_usage(model),
            'training_config': config,
            'final_metrics': metrics,
            'performance_ratios': self._calculate_performance_ratios(metrics)
        }

        return summary

    def _calculate_performance_ratios(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calcule des ratios de performance"""
        ratios = {}

        # Ratios de performance basiques
        if 'train_loss' in metrics and 'val_loss' in metrics:
            ratios['overfitting_ratio'] = metrics['val_loss'] / metrics['train_loss']

        if 'training_time' in metrics and 'best_val_loss' in metrics:
            ratios['time_per_best_loss'] = metrics['training_time'] / metrics['best_val_loss']

        return ratios

    def _create_training_plots(self, metrics: Dict[str, float]):
        """Crée des visualisations de l'entraînement"""
        try:
            # Graphique des pertes d'entraînement et validation
            if 'train_loss_history' in metrics and 'val_loss_history' in metrics:
                plt.figure(figsize=(10, 6))

                train_history = metrics['train_loss_history']
                val_history = metrics['val_loss_history']

                plt.plot(train_history, label='Train Loss', alpha=0.7)
                plt.plot(val_history, label='Validation Loss', alpha=0.7)

                plt.title('Évolution des Pertes')
                plt.xlabel('Époque')
                plt.ylabel('Perte')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.savefig(f'training_losses_{self.run_id}.png', dpi=300, bbox_inches='tight')
                mlflow.log_artifact(f'training_losses_{self.run_id}.png')
                plt.close()

        except Exception as e:
            print(f"Erreur lors de la création des graphiques: {e}")

    def log_optimization_results(self, results: Dict[str, Any], output_file: str = None):
        """Enregistre les résultats d'optimisation"""
        if output_file is None:
            output_file = f"optimization_results_{self.run_id}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(output_file)

        # Enregistrer un résumé des meilleurs résultats
        if 'best_params' in results:
            self.log_params(results['best_params'])

        if 'best_val_loss' in results:
            self.log_metrics({'best_val_loss': results['best_val_loss']})

    def end_run(self):
        """Termine l'exécution en cours"""
        if mlflow.active_run():
            mlflow.end_run()

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()


class MetricsCollector:
    """Collecteur de métriques pour le monitoring"""

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def collect_metric(self, name: str, value: float, step: int = None):
        """Collecte une métrique"""
        if name not in self.metrics:
            self.metrics[name] = []
            self.history[name] = []

        self.metrics[name].append(value)
        if step is not None:
            self.history[name].append((step, value))

    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Retourne un résumé d'une métrique"""
        if name not in self.metrics or not self.metrics[name]:
            return {}

        values = np.array(self.metrics[name])
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'last': values[-1],
            'count': len(values)
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Retourne les résumés de toutes les métriques"""
        return {name: self.get_metric_summary(name) for name in self.metrics}

    def clear_metrics(self):
        """Efface toutes les métriques"""
        self.metrics = {}
        self.history = {}


class TrainingMonitor:
    """Moniteur d'entraînement intégré"""

    def __init__(self, experiment_name: str = "training_monitor"):
        self.tracker = ExperimentTracker(experiment_name)
        self.collector = MetricsCollector()
        self.start_time = None

    def start_training(self, model: nn.Module, config: Dict[str, Any]):
        """Démarre le monitoring d'une session d'entraînement"""
        self.start_time = datetime.now()

        with self.tracker.start_run() as run:
            # Enregistrer la configuration
            self.tracker.log_params(config)

            # Enregistrer les infos du modèle
            from ..utils.profiling import log_memory_usage
            model_info = log_memory_usage(model)
            self.tracker.log_params(model_info)

    def log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                         val_metrics: Dict[str, float]):
        """Enregistre les métriques d'une époque"""
        # Combiner les métriques
        all_metrics = {**train_metrics, **val_metrics}
        all_metrics['epoch'] = epoch

        # Enregistrer dans MLflow
        self.tracker.log_metrics(all_metrics, step=epoch)

        # Collecter pour l'analyse locale
        for name, value in all_metrics.items():
            self.collector.collect_metric(name, value, epoch)

    def log_batch_metrics(self, batch: int, metrics: Dict[str, float]):
        """Enregistre les métriques d'un batch"""
        for name, value in metrics.items():
            self.collector.collect_metric(f"batch_{name}", value, batch)

    def end_training(self, final_metrics: Dict[str, float], model: nn.Module = None):
        """Termine le monitoring"""
        # Enregistrer les métriques finales
        final_metrics['training_time'] = (datetime.now() - self.start_time).total_seconds()
        self.tracker.log_metrics(final_metrics)

        # Enregistrer le modèle si fourni
        if model is not None:
            self.tracker.log_model(model, "final_model")

        # Créer un résumé
        summary = self._create_training_summary(final_metrics)
        self.tracker.log_params(summary)

        self.tracker.end_run()

    def _create_training_summary(self, final_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Crée un résumé de l'entraînement"""
        return {
            'total_epochs': final_metrics.get('epoch', 0),
            'final_train_loss': final_metrics.get('train_loss', 0),
            'final_val_loss': final_metrics.get('val_loss', 0),
            'training_duration': final_metrics.get('training_time', 0),
            'convergence_epoch': self._find_convergence_epoch(),
            'best_val_loss': min(self.collector.metrics.get('val_loss', [float('inf')]))
        }

    def _find_convergence_epoch(self) -> Optional[int]:
        """Trouve l'époque de convergence"""
        if 'val_loss' not in self.collector.history:
            return None

        losses = [val for _, val in self.collector.history['val_loss']]
        if len(losses) < 10:
            return None

        # Trouver le minimum local après 10 époques
        min_loss = min(losses[10:])
        for i, loss in enumerate(losses[10:], 10):
            if loss == min_loss:
                return i

        return None


def log_system_metrics(tracker: ExperimentTracker, prefix: str = "system"):
    """Enregistre les métriques système"""
    try:
        import psutil

        # Métriques CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0

        # Métriques mémoire
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024 ** 3)  # Go

        # Métriques disque
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024 ** 3)  # Go

        system_metrics = {
            f'{prefix}_cpu_percent': cpu_percent,
            f'{prefix}_cpu_freq': cpu_freq,
            f'{prefix}_memory_percent': memory_percent,
            f'{prefix}_memory_used_gb': memory_used,
            f'{prefix}_disk_percent': disk_percent,
            f'{prefix}_disk_used_gb': disk_used
        }

        tracker.log_metrics(system_metrics)

    except Exception as e:
        print(f"Erreur lors de la collecte des métriques système: {e}")


def setup_tensorboard_logging(log_dir: str = "logs/tensorboard"):
    """Configure la journalisation TensorBoard"""
    from torch.utils.tensorboard import SummaryWriter

    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def log_to_tensorboard(writer: Any, metrics: Dict[str, float], step: int):
    """Enregistre les métriques dans TensorBoard"""
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)

    writer.flush()


def create_experiment_comparison(experiments: List[str],
                               metrics: List[str] = None) -> Dict[str, Any]:
    """
    Crée une comparaison entre plusieurs expériences.

    Args:
        experiments: Liste des noms d'expériences à comparer
        metrics: Liste des métriques à comparer

    Returns:
        Dictionnaire contenant la comparaison
    """
    if metrics is None:
        metrics = ['val_loss', 'train_loss', 'training_time']

    comparison = {
        'experiments': experiments,
        'metrics': {}
    }

    # Récupérer les runs de chaque expérience
    for exp_name in experiments:
        mlflow.set_experiment(exp_name)
        runs = mlflow.search_runs()

        if not runs.empty:
            # Meilleur run pour cette expérience
            best_run = runs.loc[runs['metrics.best_val_loss'].idxmin()]

            exp_metrics = {}
            for metric in metrics:
                if f'metrics.{metric}' in runs.columns:
                    exp_metrics[metric] = best_run[f'metrics.{metric}']

            comparison['metrics'][exp_name] = exp_metrics

    return comparison
