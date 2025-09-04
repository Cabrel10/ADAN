#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour le SharedExperienceBuffer avec monitoring.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(str(Path(__file__).parent.parent))

from src.adan_trading_bot.training.shared_experience_buffer import SharedExperienceBuffer

def generate_sample_experience(step: int) -> Dict[str, Any]:
    """Génère une expérience d'exemple pour les tests."""
    return {
        'state': np.random.rand(10).tolist(),
        'action': np.random.randint(0, 3),
        'reward': np.random.uniform(-1, 1),
        'next_state': np.random.rand(10).tolist(),
        'done': False,
        'step': step,
        'timestamp': time.time()
    }

def test_buffer_operations(buffer_size: int = 1000, num_workers: int = 4, steps: int = 1000):
    """Teste les opérations de base du buffer avec plusieurs workers."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # Créer un buffer partagé
    buffer = SharedExperienceBuffer(buffer_size=buffer_size)

    # Variable pour suivre la progression
    progress = {'added': 0, 'sampled': 0}
    progress_lock = threading.Lock()

    def worker_consumer(worker_id: int, steps: int):
        """Worker qui ajoute des expériences au buffer."""
        nonlocal progress
        for i in range(steps):
            # Générer une expérience
            exp = generate_sample_experience(i)

            # Ajouter au buffer avec une priorité aléatoire
            buffer.add(exp, priority=np.random.uniform(0.1, 1.0))

            # Mettre à jour la progression
            with progress_lock:
                progress['added'] += 1

            # Simuler un délai de traitement
            time.sleep(0.001)

    def trainer(steps: int, batch_size: int = 32):
        """Simule un entraîneur qui échantillonne des lots."""
        nonlocal progress
        for _ in range(steps):
            if len(buffer) >= batch_size:
                # Échantillonner un lot
                batch, indices, weights = buffer.sample(batch_size)

                # Simuler une mise à jour des priorités
                new_priorities = np.random.uniform(0.1, 1.0, size=len(indices))
                buffer.update_priorities(indices, new_priorities)

                # Mettre à jour la progression
                with progress_lock:
                    progress['sampled'] += batch_size

            # Attendre un peu
            time.sleep(0.01)

    # Démarrer les workers
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        # Démarrer les workers consommateurs
        for i in range(num_workers):
            executor.submit(worker_consumer, i, steps)

        # Démarrer le trainer
        executor.submit(trainer, steps * num_workers // 10)  # Moins d'échantillonnage

        # Afficher les statistiques périodiquement
        start_time = time.time()
        try:
            while True:
                time.sleep(5)
                stats = buffer.get_stats()
                elapsed = time.time() - start_time

                print("\n" + "="*50)
                print(f"Temps écoulé: {elapsed:.1f}s")
                print(f"Taille du buffer: {stats['size']}/{buffer_size} ({stats['utilization_percent']:.1f}%)")
                print(f"Ajouts: {progress['added']} ({stats['add_rate_per_second']:.1f}/s)")
                print(f"Échantillons: {progress['sampled']} ({stats['sample_rate_per_second']:.1f}/s)")
                print(f"Priorité max: {stats['priority_max']:.4f}")
                print(f"Beta actuel: {stats['beta']:.4f}")

                # Arrêter après un certain temps
                if elapsed > 30:  # 30 secondes de test
                    break

        except KeyboardInterrupt:
            print("\nArrêt demandé par l'utilisateur.")

        # Afficher les statistiques finales
        print("\n" + "="*50)
        print("RÉSUMÉ FINAL")
        print("="*50)
        for k, v in buffer.get_stats().items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    test_buffer_operations(buffer_size=10000, num_workers=4, steps=1000)
