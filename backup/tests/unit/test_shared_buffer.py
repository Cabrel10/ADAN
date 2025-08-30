"""
Tests unitaires pour le SharedExperienceBuffer.
"""
import numpy as np
import pytest

from src.adan_trading_bot.training.shared_experience_buffer import SharedExperienceBuffer


def test_buffer_initialization(mock_config):
    """Teste l'initialisation du buffer."""
    buffer_config = mock_config["model"]["buffer"]
    buffer = SharedExperienceBuffer(
        buffer_size=buffer_config["buffer_size"],
        alpha=buffer_config["alpha"],
        beta=buffer_config["beta"],
        beta_increment=buffer_config["beta_increment"]
    )
    
    assert buffer.size == 0
    assert buffer.max_size == buffer_config["buffer_size"]
    assert buffer.alpha == buffer_config["alpha"]
    assert buffer.beta == buffer_config["beta"]
    assert buffer.beta_increment == buffer_config["beta_increment"]


def test_add_experience(mock_config, sample_experience):
    """Teste l'ajout d'expériences au buffer."""
    buffer_config = mock_config["model"]["buffer"]
    buffer = SharedExperienceBuffer(
        buffer_size=10,  # Taille réduite pour les tests
        alpha=buffer_config["alpha"]
    )
    
    # Ajouter plusieurs expériences
    for i in range(5):
        exp = sample_experience.copy()
        exp["reward"] = i * 0.1  # Récompense différente pour chaque expérience
        buffer.add(exp)
    
    assert buffer.size == 5
    
    # Tester le débordement du buffer
    for i in range(10):
        exp = sample_experience.copy()
        exp["reward"] = i * 0.2
        buffer.add(exp)
    
    assert buffer.size <= 10  # Ne doit pas dépasser la taille maximale


def test_sample_experience(mock_config, sample_experience):
    """Teste l'échantillonnage d'expériences."""
    buffer_config = mock_config["model"]["buffer"]
    batch_size = 4
    
    buffer = SharedExperienceBuffer(
        buffer_size=10,
        alpha=buffer_config["alpha"],
        beta=buffer_config["beta"]
    )
    
    # Remplir le buffer avec des expériences
    for i in range(8):
        exp = sample_experience.copy()
        exp["reward"] = i * 0.1
        buffer.add(exp)
    
    # Tester l'échantillonnage
    batch, indices, weights = buffer.sample(batch_size)
    
    assert len(batch) == batch_size
    assert len(indices) == batch_size
    assert len(weights) == batch_size
    assert all(isinstance(exp, dict) for exp in batch)
    assert all(isinstance(idx, int) for idx in indices)
    assert all(isinstance(w, float) for w in weights)


def test_update_priorities(mock_config, sample_experience):
    """Teste la mise à jour des priorités."""
    buffer = SharedExperienceBuffer(
        buffer_size=10,
        alpha=0.6
    )
    
    # Ajouter des expériences
    for i in range(5):
        exp = sample_experience.copy()
        exp["reward"] = i * 0.1
        buffer.add(exp)
    
    # Échantillonner pour obtenir des indices
    batch, indices, _ = buffer.sample(3)
    
    # Mettre à jour les priorités
    new_priorities = [0.8, 0.9, 1.0]
    buffer.update_priorities(indices, new_priorities)
    
    # Vérifier que les priorités ont été mises à jour
    for idx, priority in zip(indices, new_priorities):
        assert abs(buffer.priorities[idx] - priority ** buffer.alpha) < 1e-6


def test_beta_increment(mock_config):
    """Teste l'incrémentation automatique de beta."""
    initial_beta = 0.4
    beta_increment = 0.01
    
    buffer = SharedExperienceBuffer(
        buffer_size=10,
        beta=initial_beta,
        beta_increment=beta_increment
    )
    
    # Beta initial
    assert buffer.beta == initial_beta
    
    # Beta après un échantillonnage
    buffer.sample(1)
    assert abs(buffer.beta - (initial_beta + beta_increment)) < 1e-6
    
    # Beta après plusieurs échantillonnages
    for _ in range(10):
        buffer.sample(1)
    
    # Beta ne doit pas dépasser 1.0
    assert buffer.beta <= 1.0


def test_edge_cases(mock_config, sample_experience):
    """Teste les cas limites."""
    # Buffer vide
    buffer = SharedExperienceBuffer(buffer_size=10)
    
    with pytest.raises(ValueError):
        buffer.sample(1)  # Ne peut pas échantillonner d'un buffer vide
    
    # Ajouter une seule expérience
    buffer.add(sample_experience)
    
    # Échantillonnage avec une taille de lot plus grande que la taille du buffer
    batch, indices, weights = buffer.sample(5)
    assert len(batch) == 1
    assert len(indices) == 1
    assert len(weights) == 1
    
    # Mise à jour des priorités avec des indices invalides
    with pytest.raises(IndexError):
        buffer.update_priorities([100], [1.0])  # Index hors limites
