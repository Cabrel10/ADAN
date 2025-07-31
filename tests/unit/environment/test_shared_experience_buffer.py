"""
Tests unitaires pour le SharedExperienceBuffer.
"""
import pytest
import numpy as np
from adan_trading_bot.training.shared_experience_buffer import SharedExperienceBuffer

def test_buffer_initialization():
    """Test l'initialisation du buffer."""
    buffer = SharedExperienceBuffer(buffer_size=1000)
    assert len(buffer) == 0
    assert buffer.buffer_size == 1000
    assert 0 <= buffer.beta <= 1.0

def test_add_experience():
    """Test l'ajout d'expériences au buffer."""
    buffer = SharedExperienceBuffer(buffer_size=10)
    
    # Ajout d'une expérience
    exp = {
        'state': np.random.rand(4),
        'action': 1,
        'reward': 1.0,
        'next_state': np.random.rand(4),
        'done': False
    }
    buffer.add(exp)
    assert len(buffer) == 1

def test_sample_experience():
    """Test l'échantillonnage d'expériences."""
    buffer = SharedExperienceBuffer(buffer_size=10)
    
    # Remplissage du buffer
    for i in range(5):
        exp = {
            'state': np.array([i] * 4, dtype=np.float32),
            'action': i % 2,
            'reward': float(i) / 10,
            'next_state': np.array([i+1] * 4, dtype=np.float32),
            'done': i == 4
        }
        buffer.add(exp)
    
    # Test d'échantillonnage
    batch, indices, weights = buffer.sample(batch_size=2)
    
    assert len(batch['state']) == 2
    assert len(indices) == 2
    assert len(weights) == 2
    assert all(0 <= i < 5 for i in indices)

def test_priority_updates():
    """Test la mise à jour des priorités."""
    buffer = SharedExperienceBuffer(buffer_size=10)
    
    # Ajout d'expériences
    for i in range(5):
        exp = {
            'state': np.array([i] * 4, dtype=np.float32),
            'action': i % 2,
            'reward': float(i) / 10,
            'next_state': np.array([i+1] * 4, dtype=np.float32),
            'done': i == 4
        }
        buffer.add(exp)
    
    # Échantillonnage et mise à jour des priorités
    _, indices, _ = buffer.sample(batch_size=2)
    buffer.update_priorities(indices, [10.0, 10.0])
    
    # Vérification que les priorités ont été mises à jour
    priorities = [buffer.priorities[i] for i in indices]
    assert all(p > 0 for p in priorities)

def test_buffer_wraparound():
    """Test le bouclage du buffer (FIFO)."""
    buffer_size = 5
    buffer = SharedExperienceBuffer(buffer_size=buffer_size)
    
    # Remplissage du buffer
    for i in range(buffer_size * 2):
        exp = {
            'state': np.array([i] * 4, dtype=np.float32),
            'action': i % 2,
            'reward': float(i) / 10,
            'next_state': np.array([i+1] * 4, dtype=np.float32),
            'done': False
        }
        buffer.add(exp)
        
        # Vérifie que la taille ne dépasse pas la capacité
        assert len(buffer) <= buffer_size
    
    # Vérifie que les anciennes expériences ont été écrasées
    assert len(buffer) == buffer_size

def test_serialization():
    """Test la sérialisation/désérialisation des expériences."""
    buffer = SharedExperienceBuffer(buffer_size=10)
    
    # Test avec différents types de données
    test_cases = [
        {'a': np.array([1, 2, 3]), 'b': 1, 'c': 'test'},
        {'d': {'nested': np.array([4, 5, 6])}, 'e': [1, 2, 3]},
        {'f': (1, 2, 3), 'g': True}
    ]
    
    for case in test_cases:
        serialized = buffer._make_serializable(case)
        # Vérifie qu'il n'y a plus de ndarray numpy
        assert not any(isinstance(v, np.ndarray) for v in serialized.values())

def test_concurrent_access():
    """Test l'accès concurrent au buffer."""
    import threading
    
    buffer = SharedExperienceBuffer(buffer_size=1000)
    num_threads = 10
    experiences_per_thread = 100
    
    def worker(worker_id):
        for i in range(experiences_per_thread):
            exp = {
                'state': np.array([worker_id, i]),
                'action': worker_id % 2,
                'reward': float(i) / 10,
                'next_state': np.array([worker_id, i+1]),
                'done': False,
                'worker_id': worker_id,
                'step': i
            }
            buffer.add(exp)
            
            # Essayer d'échantillonner
            if i % 10 == 0:
                batch, _, _ = buffer.sample(min(10, len(buffer)))
                assert len(batch['state']) > 0
    
    # Création et démarrage des threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Attente de la fin des threads
    for t in threads:
        t.join()
    
    # Vérifications finales
    assert len(buffer) <= buffer.buffer_size
    assert len(buffer) >= min(buffer.buffer_size, num_threads * experiences_per_thread)
