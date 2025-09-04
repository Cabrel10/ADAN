#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour l'intégration du SharedExperienceBuffer avec le TrainingOrchestrator.

Ce script simule plusieurs workers qui ajoutent des expériences à un buffer partagé
et un processus d'entraînement qui consomme ces expériences.
"""

import os
import sys
import time
import random
import numpy as np
from multiprocessing import Process, Manager
from typing import Dict, List, Any, Tuple

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adan_trading_bot.training.shared_experience_buffer import SharedExperienceBuffer
from src.adan_trading_bot.training.training_orchestrator import TrainingOrchestrator

# Configuration
NUM_WORKERS = 4
STEPS_PER_WORKER = 2000
TRAINING_STEPS = 1000
BATCH_SIZE = 32
STATE_DIM = 10  # Dimension de l'état pour la simulation
BUFFER_SIZE = 10000


def worker_process(worker_id: int, buffer: SharedExperienceBuffer, num_steps: int) -> None:
    """Simule un worker qui ajoute des expériences au buffer."""
    print(f"Worker {worker_id} démarré pour {num_steps} étapes")

    for _ in range(num_steps):
        # Générer des données d'expérience aléatoires
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = np.random.randint(0, 3)  # 3 actions possibles
        reward = random.uniform(-1, 1)
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        done = random.random() < 0.01  # 1% de chances que l'épisode se termine

        # Créer un dictionnaire d'expérience
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        # Ajouter l'expérience au buffer avec une priorité de 1.0
        buffer.add(experience, priority=1.0)

        # Ralentir légèrement pour simuler un travail
        time.sleep(0.001)

    print(f"Worker {worker_id} terminé")


def train_process(buffer: SharedExperienceBuffer, num_steps: int) -> None:
    """Simule un processus d'entraînement qui consomme des expériences du buffer."""
    print(f"Processus d'entraînement démarré pour {num_steps} étapes")

    # Modèle dummy pour la simulation
    def model(x):
        return np.random.randn(len(x), 3)  # 3 actions possibles

    for step in range(num_steps):
        # Attendre qu'il y ait suffisamment d'expériences dans le buffer
        while len(buffer) < BATCH_SIZE:
            print(f"Attente d'expériences ({len(buffer)}/{BATCH_SIZE})...")
            time.sleep(0.5)

        # Échantillonner un lot d'expériences
        try:
            batch, indices, weights = buffer.sample(BATCH_SIZE)
        except ValueError as e:
            print(f"Erreur lors de l'échantillonnage: {e}")
            time.sleep(0.5)
            continue

        # Extraire les données du batch
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])

        # Simulation d'une étape d'entraînement
        q_values = model(states)
        next_q_values = model(next_states)

        # Mettre à jour les priorités (simulation)
        td_errors = np.abs(
            rewards + 0.99 * np.max(next_q_values, axis=1) * (1 - dones) -
            q_values[np.arange(len(actions)), actions]
        )
        buffer.update_priorities(indices, td_errors + 1e-5)  # Éviter les priorités nulles

        # Afficher des statistiques occasionnellement
        if step % 100 == 0:
            print(f"Étape {step}: taille du buffer = {len(buffer)}")

    print("Processus d'entraînement terminé")


def main() -> None:
    """Fonction principale du script de test."""
    print("Démarrage du test d'intégration...")

    # Créer le buffer partagé
    buffer = SharedExperienceBuffer(
        buffer_size=BUFFER_SIZE,
        alpha=0.6,  # Paramètre pour l'échantillonnage prioritaire
        beta=0.4,   # Paramètre d'importance sampling
        beta_increment=0.001  # Incrément progressif de beta
    )

    # Créer l'orchestrateur
    config = {
        'monitoring_interval': 5,  # secondes
        'training': {
            'batch_size': BATCH_SIZE,
            'gamma': 0.99,
            'lr': 1e-4,
            'target_update_freq': 100
        }
    }
    orchestrator = TrainingOrchestrator(
        config=config,
        shared_buffer=buffer
    )

    # Démarrer les workers
    workers = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(i, buffer, STEPS_PER_WORKER))
        p.start()
        workers.append(p)

    # Démarrer le processus d'entraînement
    trainer = Process(target=train_process, args=(buffer, TRAINING_STEPS))
    trainer.start()

    # Attendre que tous les workers aient terminé
    for p in workers:
        p.join()

    # Attendre que le processus d'entraînement ait terminé
    trainer.join()

    # Afficher les statistiques finales
    print("\n=== Statistiques finales ===")
    for k, v in buffer.get_stats().items():
        print(f"{k}: {v}")

    print("\nTest terminé avec succès!")


if __name__ == "__main__":
    main()
