"""Tests de charge pour l'optimisation du modèle ADAN."""

import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import psutil
import json
from datetime import datetime

from ..models.attention_model import OptimizedIntegratedCNNPPOModel
from ..utils.profiling import log_memory_usage, profile_model_inference


class LoadTester:
    """Classe pour effectuer des tests de charge sur le modèle"""

    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        """
        Initialise le testeur de charge.

        Args:
            model: Le modèle à tester
            device: Appareil sur lequel effectuer les tests (CPU/GPU)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Statistiques de performance
        self.latencies = []
        self.memory_usage = []
        self.throughput_history = []

    def generate_dummy_batch(self, batch_size: int, input_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """Génère un lot de données factices pour les tests"""
        # Générer des données pour chaque timeframe
        dummy_data = {}
        for tf in ['5m', '1h', '4h']:
            dummy_data[tf] = torch.randn(batch_size, *input_shape, device=self.device)

        # Ajouter l'état du portefeuille
        dummy_data['portfolio_state'] = torch.randn(batch_size, 10, device=self.device)  # 10 features portfolio

        return dummy_data

    def measure_system_resources(self) -> Dict[str, float]:
        """Mesure les ressources système actuelles"""
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024 ** 3),
        }

        if torch.cuda.is_available():
            resources['gpu_memory_allocated'] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            resources['gpu_memory_reserved'] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            resources['gpu_utilization'] = torch.cuda.utilization(self.device)

        return resources

    def run_load_test(
        self,
        input_shape: Tuple[int, ...],
        num_batches: int = 100,
        batch_size: int = 32,
        warmup_batches: int = 10
    ) -> Dict[str, Any]:
        """
        Exécute un test de charge sur le modèle.

        Args:
            input_shape: Forme des données d'entrée (sans la dimension du lot)
            num_batches: Nombre de lots à traiter
            batch_size: Taille de chaque lot
            warmup_batches: Nombre de lots d'échauffement à ignorer

        Returns:
            Dictionnaire contenant les métriques de performance
        """
        self.latencies = []
        self.memory_usage = []
        self.throughput_history = []

        print(f"Démarrage du test de charge avec {num_batches} lots de taille {batch_size}...")
        print(f"Input shape: {input_shape}, Device: {self.device}")

        # Phase d'échauffement
        if warmup_batches > 0:
            print(f"Phase d'échauffement ({warmup_batches} lots)...")
            warmup_input = self.generate_dummy_batch(batch_size, input_shape)
            for _ in range(warmup_batches):
                with torch.no_grad():
                    _ = self.model(warm_input)

        # Test de charge
        print("Démarrage des mesures...")
        start_time = time.time()
        total_samples = 0

        for i in tqdm(range(num_batches), desc="Test de charge"):
            # Générer des données d'entrée factices
            inputs = self.generate_dummy_batch(batch_size, input_shape)

            # Mesurer les ressources avant
            resources_before = self.measure_system_resources()

            # Mesurer la latence
            batch_start_time = time.time()

            with torch.no_grad():
                outputs = self.model(inputs)

            # Synchroniser pour une mesure précise
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Mesurer les ressources après
            resources_after = self.measure_system_resources()

            # Enregistrer la latence
            latency = (time.time() - batch_start_time) * 1000  # en ms
            self.latencies.append(latency)

            # Enregistrer l'utilisation de la mémoire
            memory_delta = {
                k: resources_after.get(k, 0) - resources_before.get(k, 0)
                for k in resources_after.keys()
            }
            self.memory_usage.append(memory_delta)

            # Calculer le débit
            throughput = batch_size / (latency / 1000) if latency > 0 else float('inf')
            self.throughput_history.append(throughput)

            total_samples += batch_size

        total_time = time.time() - start_time

        # Calcul des métriques
        latencies = np.array(self.latencies)
        throughputs = np.array(self.throughput_history)

        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        avg_throughput = np.mean(throughputs)

        # Calcul des métriques mémoire
        memory_metrics = {}
        if self.memory_usage:
            for key in self.memory_usage[0].keys():
                values = [m[key] for m in self.memory_usage if key in m]
                if values:
                    memory_metrics[f'{key}_mean'] = np.mean(values)
                    memory_metrics[f'{key}_max'] = np.max(values)
                    memory_metrics[f'{key}_std'] = np.std(values)

        results = {
            'test_config': {
                'batch_size': batch_size,
                'num_batches': num_batches,
                'input_shape': input_shape,
                'device': str(self.device),
                'total_samples': total_samples,
                'test_duration': total_time
            },
            'latency_metrics': {
                'avg_latency_ms': avg_latency,
                'p50_latency_ms': p50,
                'p95_latency_ms': p95,
                'p99_latency_ms': p99,
                'latency_std_ms': np.std(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies)
            },
            'throughput_metrics': {
                'avg_throughput_samples_per_sec': avg_throughput,
                'total_samples_per_sec': total_samples / total_time,
                'throughput_std': np.std(throughputs)
            },
            'memory_metrics': memory_metrics,
            'efficiency_metrics': {
                'samples_per_ms': avg_throughput / 1000,
                'time_per_sample_ms': avg_latency / batch_size
            }
        }

        return results


def run_load_test_suite(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_sizes: Optional[List[int]] = None,
    num_batches: int = 50,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Exécute une série de tests de charge avec différentes tailles de lot.

    Args:
        model: Le modèle à tester
        input_shape: Forme des données d'entrée (sans la dimension du lot)
        batch_sizes: Liste des tailles de lot à tester
        num_batches: Nombre de lots par test
        device: Appareil à utiliser ('cpu', 'cuda', 'auto')

    Returns:
        Dictionnaire contenant les résultats de tous les tests
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64]

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tester = LoadTester(model, device)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info(),
        'model_info': log_memory_usage(model),
        'batch_results': {}
    }

    for bs in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Test avec taille de lot: {bs}")
        print(f"{'='*50}")

        try:
            results = tester.run_load_test(
                input_shape=input_shape,
                num_batches=num_batches,
                batch_size=bs
            )

            # Afficher les résultats
            print("\nRésultats:")
            print(f"  Latence moyenne: {results['latency_metrics']['avg_latency_ms']".2f"} ms")
            print(f"  P95 latence: {results['latency_metrics']['p95_latency_ms']".2f"} ms")
            print(f"  Débit moyen: {results['throughput_metrics']['avg_throughput_samples_per_sec']".2f"} échantillons/s")
            print(f"  Débit total: {results['throughput_metrics']['total_samples_per_sec']".2f"} échantillons/s")

            if results['memory_metrics']:
                gpu_mem = results['memory_metrics'].get('gpu_memory_allocated_mean', 0)
                if gpu_mem > 0:
                    print(f"  Mémoire GPU moyenne: {gpu_mem".2f"} Mo")

            all_results['batch_results'][f'batch_size_{bs}'] = results

        except RuntimeError as e:
            error_msg = f"Erreur avec batch_size={bs}: {str(e)}"
            print(error_msg)
            all_results['batch_results'][f'batch_size_{bs}'] = {"error": error_msg}
        except Exception as e:
            error_msg = f"Erreur inattendue avec batch_size={bs}: {str(e)}"
            print(error_msg)
            all_results['batch_results'][f'batch_size_{bs}'] = {"error": error_msg}

    return all_results


def run_stress_test(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    max_batch_size: int = 256,
    duration_minutes: int = 5,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Exécute un test de stress sur le modèle.

    Args:
        model: Le modèle à tester
        input_shape: Forme des données d'entrée
        max_batch_size: Taille de lot maximale à tester
        duration_minutes: Durée du test en minutes
        device: Appareil à utiliser

    Returns:
        Dictionnaire contenant les résultats du stress test
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tester = LoadTester(model, device)

    print(f"Test de stress - {duration_minutes} minutes avec batch_size croissant")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    results = {
        'start_time': datetime.now().isoformat(),
        'duration_minutes': duration_minutes,
        'batch_history': [],
        'errors': []
    }

    batch_size = 1
    batch_count = 0

    while time.time() < end_time:
        try:
            # Augmenter progressivement la taille du lot
            if batch_count % 10 == 0 and batch_size < max_batch_size:
                batch_size = min(batch_size * 2, max_batch_size)

            # Exécuter un batch
            inputs = tester.generate_dummy_batch(batch_size, input_shape)

            batch_start = time.time()
            with torch.no_grad():
                _ = tester.model(inputs)
            batch_time = time.time() - batch_start

            # Enregistrer les résultats
            batch_result = {
                'batch_id': batch_count,
                'batch_size': batch_size,
                'timestamp': time.time() - start_time,
                'execution_time': batch_time,
                'throughput': batch_size / batch_time if batch_time > 0 else float('inf'),
                'resources': tester.measure_system_resources()
            }

            results['batch_history'].append(batch_result)
            batch_count += 1

        except RuntimeError as e:
            # En cas d'erreur mémoire, réduire la taille du lot
            error_info = {
                'batch_id': batch_count,
                'batch_size': batch_size,
                'error': str(e),
                'timestamp': time.time() - start_time
            }
            results['errors'].append(error_info)

            batch_size = max(1, batch_size // 2)
            print(f"Erreur mémoire, réduction de batch_size à {batch_size}")

        except Exception as e:
            error_info = {
                'batch_id': batch_count,
                'batch_size': batch_size,
                'error': str(e),
                'timestamp': time.time() - start_time
            }
            results['errors'].append(error_info)

            print(f"Erreur inattendue: {str(e)}")
            break

    results['end_time'] = datetime.now().isoformat()
    results['total_batches'] = batch_count
    results['final_batch_size'] = batch_size

    # Calculer les métriques finales
    if results['batch_history']:
        throughputs = [b['throughput'] for b in results['batch_history']]
        results['avg_throughput'] = np.mean(throughputs)
        results['max_throughput'] = np.max(throughputs)
        results['total_samples'] = sum(b['batch_size'] for b in results['batch_history'])

    return results


def run_concurrent_test(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    num_threads: int = 4,
    num_requests_per_thread: int = 100,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Test de concurrence avec plusieurs threads.

    Args:
        model: Le modèle à tester
        input_shape: Forme des données d'entrée
        num_threads: Nombre de threads à utiliser
        num_requests_per_thread: Nombre de requêtes par thread
        device: Appareil à utiliser

    Returns:
        Dictionnaire contenant les résultats du test de concurrence
    """
    import threading
    import queue

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Préparer le modèle (un par thread pour éviter les conflits)
    models = [model.to(device) for _ in range(num_threads)]
    [m.eval() for m in models]

    # Queue pour collecter les résultats
    results_queue = queue.Queue()

    def worker_thread(thread_id: int, model: torch.nn.Module):
        """Fonction exécutée par chaque thread"""
        thread_results = []

        for i in range(num_requests_per_thread):
            try:
                # Générer des données
                batch_size = np.random.randint(1, 33)  # Taille aléatoire 1-32
                inputs = {
                    tf: torch.randn(batch_size, *input_shape, device=device)
                    for tf in ['5m', '1h', '4h']
                }
                inputs['portfolio_state'] = torch.randn(batch_size, 10, device=device)

                # Mesurer le temps
                start_time = time.time()

                with torch.no_grad():
                    _ = model(inputs)

                execution_time = time.time() - start_time
                throughput = batch_size / execution_time if execution_time > 0 else float('inf')

                thread_results.append({
                    'request_id': i,
                    'batch_size': batch_size,
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'success': True
                })

            except Exception as e:
                thread_results.append({
                    'request_id': i,
                    'error': str(e),
                    'success': False
                })

        results_queue.put(thread_results)

    # Démarrer les threads
    print(f"Démarrage du test de concurrence avec {num_threads} threads...")
    threads = []

    for i in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(i, models[i]))
        threads.append(thread)
        thread.start()

    # Attendre la fin de tous les threads
    for thread in threads:
        thread.join()

    # Collecter les résultats
    all_results = []
    for _ in range(num_threads):
        thread_results = results_queue.get()
        all_results.extend(thread_results)

    # Analyser les résultats
    successful_requests = [r for r in all_results if r['success']]
    failed_requests = [r for r in all_results if not r['success']]

    if successful_requests:
        throughputs = [r['throughput'] for r in successful_requests]
        execution_times = [r['execution_time'] for r in successful_requests]
        batch_sizes = [r['batch_size'] for r in successful_requests]

        concurrent_results = {
            'total_requests': len(all_results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(all_results),
            'avg_throughput': np.mean(throughputs),
            'total_throughput': sum(throughputs),
            'avg_execution_time': np.mean(execution_times),
            'avg_batch_size': np.mean(batch_sizes),
            'max_batch_size': np.max(batch_sizes),
            'min_batch_size': np.min(batch_sizes)
        }
    else:
        concurrent_results = {
            'total_requests': len(all_results),
            'successful_requests': 0,
            'failed_requests': len(failed_requests),
            'success_rate': 0,
            'error': "Tous les requêtes ont échoué"
        }

    return concurrent_results


def save_load_test_results(results: Dict[str, Any], filename: str = None):
    """Sauvegarde les résultats des tests de charge"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'load_test_results_{timestamp}.json'

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Résultats sauvegardés dans {filename}")
    return filename


def get_system_info() -> Dict[str, Any]:
    """Récupère les informations système"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
        'cpu_count': psutil.cpu_count(),
        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
        'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
        'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
    }

    # Informations GPU
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)

        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info['gpu_memory_total'] = gpu.memoryTotal
                info['gpu_memory_free'] = gpu.memoryFree
                info['gpu_memory_used'] = gpu.memoryUsed
                info['gpu_temperature'] = gpu.temperature
                info['gpu_utilization'] = gpu.load * 100
        except ImportError:
            pass
    else:
        info['cuda_available'] = False

    return info


def print_load_test_summary(results: Dict[str, Any]):
    """Affiche un résumé des tests de charge"""
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS DE CHARGE")
    print("="*60)

    if 'batch_results' in results:
        print("\n📊 RÉSULTATS PAR TAILLE DE LOT:")
        print("-" * 40)

        for batch_key, batch_result in results['batch_results'].items():
            if 'error' in batch_result:
                print(f"{batch_key}: ❌ ERREUR - {batch_result['error']}")
            else:
                config = batch_result['test_config']
                latency = batch_result['latency_metrics']
                throughput = batch_result['throughput_metrics']

                print(f"{batch_key}:")
                print(f"  Batch size: {config['batch_size']}")
                print(f"  Latence: {latency['avg_latency_ms']".2f"}ms (P95: {latency['p95_latency_ms']".2f"}ms)")
                print(f"  Débit: {throughput['avg_throughput_samples_per_sec']".2f"} échantillons/s")
                print(f"  Durée: {config['test_duration']".2f"}s")

    if 'system_info' in results:
        print("\n💻 INFORMATIONS SYSTÈME:")
        print("-" * 40)
        sys_info = results['system_info']
        print(f"CPU: {sys_info['cpu_count']} cœurs")
        if sys_info.get('cpu_freq'):
            print(f"Fréquence CPU: {sys_info['cpu_freq']".0f"} MHz")
        print(f"Mémoire: {sys_info['memory_total_gb']".1f"} Go")
        if sys_info.get('cuda_available'):
            print(f"GPU: {sys_info['cuda_device_name']}")
            if 'gpu_memory_total' in sys_info:
                print(f"Mémoire GPU: {sys_info['gpu_memory_total']".0f"} Mo")

    print("\n" + "="*60)


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un modèle de test
    input_shape = (50, 10)  # 50 time steps, 10 features
    model = OptimizedIntegratedCNNPPOModel(
        input_shape=(3, *input_shape),  # 3 timeframes
        action_dim=2,
        use_attention=True,
        use_multiscale=True
    )

    # Exécuter la suite de tests
    print("🚀 Démarrage des tests de charge...")
    results = run_load_test_suite(
        model=model,
        input_shape=input_shape,
        batch_sizes=[1, 8, 16, 32, 64],
        num_batches=25
    )

    # Afficher le résumé
    print_load_test_summary(results)

    # Sauvegarder les résultats
    save_load_test_results(results)

    # Test de stress
    print("\n🔥 Test de stress...")
    stress_results = run_stress_test(
        model=model,
        input_shape=input_shape,
        max_batch_size=128,
        duration_minutes=2
    )

    print(f"Test de stress terminé: {len(stress_results['batch_history'])} batches traités")
    print(f"Taille finale du lot: {stress_results['final_batch_size']}")
    print(f"Erreurs: {len(stress_results['errors'])}")

    # Sauvegarder les résultats du stress test
    save_load_test_results(stress_results, "stress_test_results.json")
