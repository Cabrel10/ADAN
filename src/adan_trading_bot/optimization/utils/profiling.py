"""Utilitaires de profilage et d'optimisation des performances."""

import cProfile
import pstats
import io
from pstats import SortKey
from typing import Callable, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import time
import numpy as np
from functools import wraps
import psutil
import GPUtil


def profile_function(func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Profile une fonction et retourne le résultat avec les métriques de performance.

    Args:
        func: Fonction à profiler
        *args: Arguments positionnels
        **kwargs: Arguments nommés

    Returns:
        Tuple (résultat_fonction, métriques_performance)
    """
    # Démarrer le profilage
    profiler = cProfile.Profile()
    profiler.enable()

    # Exécuter la fonction
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 2)  # Mo

    result = func(*args, **kwargs)

    end_time = time.time()
    end_memory = psutil.virtual_memory().used / (1024 ** 2)  # Mo

    profiler.disable()

    # Analyser les résultats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()

    execution_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Extraire les métriques clés
    lines = s.getvalue().split('\n')
    total_calls = 0
    total_time = 0.0

    for line in lines[:10]:  # Premières lignes contiennent les infos principales
        if line.strip() and not line.startswith(' '):
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                total_calls = int(parts[0])
                total_time = float(parts[2])
                break

    return result, {
        'execution_time': execution_time,
        'memory_used_mb': memory_used,
        'total_calls': total_calls,
        'total_time': total_time,
        'profile_output': s.getvalue()
    }


def profile_model_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'auto',
    n_iter: int = 100,
    batch_size: int = 1
) -> Dict[str, Any]:
    """
    Profile les performances d'inférence du modèle.

    Args:
        model: Le modèle à profiler
        input_shape: Forme du tenseur d'entrée (sans batch)
        device: Appareil à utiliser ('cpu', 'cuda', 'auto')
        n_iter: Nombre d'itérations pour le profilage
        batch_size: Taille du lot

    Returns:
        Dictionnaire contenant les statistiques de performance
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Déplacer le modèle sur le bon appareil
    model = model.to(device)
    model.eval()

    # Créer un tenseur d'entrée factice
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Fonction à profiler
    def run_inference():
        with torch.no_grad():
            for _ in range(n_iter):
                _ = model(dummy_input)

    # Profilage
    _, metrics = profile_function(run_inference)

    # Calculer les métriques par itération
    time_per_iter = metrics['execution_time'] / n_iter
    throughput = batch_size / time_per_iter if time_per_iter > 0 else float('inf')

    return {
        'total_time': metrics['execution_time'],
        'time_per_iter': time_per_iter,
        'throughput_samples_per_sec': throughput,
        'memory_used_mb': metrics['memory_used_mb'],
        'device': device,
        'batch_size': batch_size,
        'n_iterations': n_iter,
        'profile_output': metrics['profile_output']
    }


def log_memory_usage(model: nn.Module, name: str = "model") -> Dict[str, Any]:
    """Enregistre l'utilisation de la mémoire du modèle"""
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = mem_params + mem_bufs

    # Mémoire GPU si disponible
    gpu_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory[f'gpu_{i}_allocated'] = torch.cuda.memory_allocated(i) / (1024 ** 2)
            gpu_memory[f'gpu_{i}_reserved'] = torch.cuda.memory_reserved(i) / (1024 ** 2)

    return {
        'model_name': name,
        'parameters_mb': mem_params / (1024 ** 2),
        'buffers_mb': mem_bufs / (1024 ** 2),
        'total_mb': mem_total / (1024 ** 2),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_buffers': sum(p.numel() for p in model.buffers()),
        **gpu_memory
    }


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: Optional[list] = None,
    device: str = 'auto',
    n_warmup: int = 10,
    n_iter: int = 50
) -> Dict[str, Any]:
    """
    Benchmark complet du modèle avec différentes tailles de lot.

    Args:
        model: Le modèle à benchmarker
        input_shape: Forme du tenseur d'entrée (sans batch)
        batch_sizes: Liste des tailles de lot à tester
        device: Appareil à utiliser
        n_warmup: Nombre d'itérations d'échauffement
        n_iter: Nombre d'itérations de benchmark

    Returns:
        Dictionnaire contenant tous les résultats du benchmark
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64]

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    results = {
        'model_info': log_memory_usage(model),
        'device': device,
        'input_shape': input_shape,
        'batch_results': {}
    }

    for batch_size in batch_sizes:
        print(f"Benchmark avec batch_size={batch_size}...")

        # Créer l'entrée factice
        dummy_input = torch.randn(batch_size, *input_shape, device=device)

        # Échauffement
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy_input)

        # Mesure des performances
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

        with torch.no_grad():
            for _ in range(n_iter):
                _ = model(dummy_input)

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

        # Calcul des métriques
        total_time = end_time - start_time
        time_per_iter = total_time / n_iter
        throughput = batch_size / time_per_iter if time_per_iter > 0 else float('inf')
        memory_used = (end_memory - start_memory) / (1024 ** 2) if torch.cuda.is_available() else 0

        batch_results = {
            'batch_size': batch_size,
            'total_time': total_time,
            'time_per_iter': time_per_iter,
            'throughput_samples_per_sec': throughput,
            'memory_used_mb': memory_used,
            'n_iterations': n_iter
        }

        results['batch_results'][f'batch_{batch_size}'] = batch_results

        print(f"  Temps/itération: {time_per_iter*1000".2f"}ms")
        print(f"  Débit: {throughput".2f"} échantillons/s")
        if memory_used > 0:
            print(f"  Mémoire GPU: {memory_used".2f"} Mo")

    return results


def compare_models(models: Dict[str, nn.Module],
                  input_shape: Tuple[int, ...],
                  device: str = 'auto') -> Dict[str, Any]:
    """
    Compare les performances de plusieurs modèles.

    Args:
        models: Dictionnaire {nom: modèle} des modèles à comparer
        input_shape: Forme du tenseur d'entrée
        device: Appareil à utiliser

    Returns:
        Dictionnaire contenant la comparaison
    """
    results = {}

    for name, model in models.items():
        print(f"Profiling du modèle: {name}")
        try:
            # Benchmark
            benchmark_results = benchmark_model(model, input_shape, device=device)

            # Métriques mémoire
            memory_info = log_memory_usage(model, name)

            results[name] = {
                'benchmark': benchmark_results,
                'memory': memory_info
            }

        except Exception as e:
            print(f"Erreur lors du profiling de {name}: {str(e)}")
            results[name] = {'error': str(e)}

    return results


def profile_training_step(model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         loss_fn: Callable,
                         input_shape: Tuple[int, ...],
                         device: str = 'auto',
                         n_steps: int = 100) -> Dict[str, Any]:
    """
    Profile une étape d'entraînement complète.

    Args:
        model: Le modèle à profiler
        optimizer: L'optimiseur
        loss_fn: La fonction de perte
        input_shape: Forme du tenseur d'entrée
        device: Appareil à utiliser
        n_steps: Nombre d'étapes à profiler

    Returns:
        Dictionnaire contenant les métriques d'entraînement
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.train()

    # Créer des données factices
    dummy_input = torch.randn(*input_shape, device=device)
    dummy_target = torch.randn(input_shape[0], device=device)  # Supposant une sortie 1D

    def training_step():
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = loss_fn(output, dummy_target)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Profilage
    losses = []
    _, metrics = profile_function(lambda: [losses.append(training_step()) for _ in range(n_steps)])

    return {
        'total_time': metrics['execution_time'],
        'time_per_step': metrics['execution_time'] / n_steps,
        'memory_used_mb': metrics['memory_used_mb'],
        'avg_loss': np.mean(losses),
        'loss_std': np.std(losses),
        'profile_output': metrics['profile_output']
    }


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimise le modèle pour l'inférence (suppression des calculs inutiles).

    Args:
        model: Le modèle à optimiser

    Returns:
        Le modèle optimisé
    """
    model.eval()

    # Fusion des BatchNorm si possible
    if hasattr(model, 'fuse'):
        model = model.fuse()

    # Quantization si CUDA est disponible
    if torch.cuda.is_available():
        try:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
        except:
            pass  # La quantization n'est pas toujours possible

    return model


class PerformanceMonitor:
    """Moniteur de performance pour le suivi continu"""

    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, step: int = None):
        """Enregistre une métrique"""
        metric = {
            'timestamp': time.time() - self.start_time,
            'name': name,
            'value': value,
            'step': step
        }
        self.metrics_history.append(metric)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques"""
        if not self.metrics_history:
            return {}

        # Grouper par nom de métrique
        metrics_by_name = {}
        for metric in self.metrics_history:
            name = metric['name']
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(metric['value'])

        # Calculer les statistiques
        summary = {}
        for name, values in metrics_by_name.items():
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }

        return summary

    def clear_history(self):
        """Efface l'historique des métriques"""
        self.metrics_history = []
        self.start_time = time.time()


# Décorateur pour profiler automatiquement les fonctions
def auto_profile(func):
    """Décorateur pour profiler automatiquement une fonction"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result, metrics = profile_function(func, *args, **kwargs)

        # Afficher les métriques principales
        print(f"Profiling de {func.__name__}:")
        print(f"  Temps d'exécution: {metrics['execution_time']".4f"}s")
        print(f"  Mémoire utilisée: {metrics['memory_used_mb']".2f"} Mo")
        print(f"  Nombre d'appels: {metrics['total_calls']}")

        return result

    return wrapper


def get_system_info() -> Dict[str, Any]:
    """Récupère les informations système"""
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
        'memory_total': psutil.virtual_memory().total / (1024 ** 3),  # Go
        'memory_available': psutil.virtual_memory().available / (1024 ** 3),  # Go
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
    }

    # Informations GPU
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info['gpu_memory_total'] = gpu.memoryTotal
                info['gpu_memory_free'] = gpu.memoryFree
                info['gpu_memory_used'] = gpu.memoryUsed
                info['gpu_temperature'] = gpu.temperature
                info['gpu_utilization'] = gpu.load * 100
        except:
            pass
    else:
        info['cuda_available'] = False

    return info
