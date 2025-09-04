"""
Test d'intégration pour le SharedExperienceBuffer.

Ce test vérifie le bon fonctionnement du buffer d'expérience partagé avec plusieurs workers.
"""
import multiprocessing
import time
import numpy as np
import pytest
from adan_trading_bot.training.shared_experience_buffer import SharedExperienceBuffer

def worker_add_experiences(buffer, worker_id, num_experiences):
    """Worker qui ajoute des expériences au buffer."""
    for i in range(num_experiences):
        experience = {
            'state': np.random.rand(10),
            'action': np.random.randint(0, 3),
            'reward': float(np.random.rand()),
            'next_state': np.random.rand(10),
            'done': False,
            'worker_id': worker_id,
            'step': i
        }
        buffer.add(experience)
        time.sleep(0.001)  # Simule un léger délai

def worker_sample_experiences(buffer, num_batches, batch_size):
    """Worker qui échantillonne des expériences du buffer."""
    for _ in range(num_batches):
        batch, indices, weights = buffer.sample(batch_size)
        assert len(batch[0]['state']) == 10  # Assuming state is a numpy array of size 10
        assert len(indices) == batch_size
        assert len(weights) == batch_size
        time.sleep(0.001)

@pytest.mark.integration
def test_concurrent_access():
    """Test l'accès concurrent au buffer par plusieurs workers."""
    # Configuration du test
    buffer_size = 1000
    num_workers = 4
    experiences_per_worker = 100
    num_samplers = 2
    batch_size = 32
    num_batches = 10

    # Initialisation du buffer partagé
    buffer = SharedExperienceBuffer(buffer_size=buffer_size)

    # Création des workers d'ajout
    add_workers = []
    for i in range(num_workers):
        w = multiprocessing.Process(
            target=worker_add_experiences,
            args=(buffer, f'worker_{i}', experiences_per_worker)
        )
        add_workers.append(w)

    # Création des workers d'échantillonnage
    sample_workers = []
    for _ in range(num_samplers):
        w = multiprocessing.Process(
            target=worker_sample_experiences,
            args=(buffer, num_batches, batch_size)
        )
        sample_workers.append(w)

    # Démarrage des workers
    for w in add_workers + sample_workers:
        w.start()

    # Attente que tous les workers aient terminé
    for w in add_workers + sample_workers:
        w.join()

    # Vérifications finales
    assert len(buffer) <= buffer_size
    assert len(buffer) >= min(buffer_size, num_workers * experiences_per_worker)

    # Test d'échantillonnage final
    batch, indices, weights = buffer.sample(batch_size)
    assert len(batch[0]['state']) == 10
    assert len(indices) == batch_size
    assert len(weights) == batch_size

@pytest.mark.integration
def test_priority_updates():
    """Test la mise à jour des priorités."""
    buffer = SharedExperienceBuffer(buffer_size=100)

    # Ajout d'expériences
    for i in range(10):
        experience = {
            'state': np.random.rand(5),
            'action': i % 3,
            'reward': float(i / 10.0),
            'next_state': np.random.rand(5),
            'done': i == 9,
            'step': i
        }
        buffer.add(experience)

    # Échantillonnage initial
    batch, indices, _ = buffer.sample(5)

    # Mise à jour des priorités
    new_priorities = np.ones(len(indices)) * 10.0
    buffer.update_priorities(indices, new_priorities)

    # Vérification que les priorités ont été mises à jour
    new_priorities, new_weights = buffer._get_priority_weights(indices)
    # Calculer la valeur attendue après transformation
    expected_priority = (10.0 + buffer.epsilon) ** buffer.alpha
    # Vérifier que les priorités ont été mises à jour avec la formule attendue
    assert np.allclose(new_priorities, expected_priority)
    # Vérifier que les poids sont normalisés (entre 0 et 1)
    assert np.all(new_weights >= 0.0) and np.all(new_weights <= 1.0)
    assert np.isclose(np.max(new_weights), 1.0)  # Au moins un poids devrait être à 1.0

@pytest.mark.integration
def test_buffer_length_and_ready():
    """Test les méthodes __len__ et is_ready."""
    buffer = SharedExperienceBuffer(buffer_size=10)

    # Test avec buffer vide
    assert len(buffer) == 0
    assert not buffer.is_ready(1)

    # Ajout d'une expérience
    exp = {'state': np.zeros(5), 'action': 0, 'reward': 0.0, 'next_state': np.zeros(5), 'done': False}
    buffer.add(exp)

    # Test après ajout
    assert len(buffer) == 1
    assert buffer.is_ready(1)
    assert not buffer.is_ready(2)

    # Remplissage du buffer
    for _ in range(9):
        buffer.add(exp)

    # Test avec buffer plein
    assert len(buffer) == 10
    assert buffer.is_ready(10)
    assert not buffer.is_ready(11)

@pytest.mark.integration
def test_add_edge_cases():
    """Test les cas limites de la méthode add."""
    buffer = SharedExperienceBuffer(buffer_size=2)
    exp = {'state': np.zeros(5), 'action': 0, 'reward': 0.0, 'next_state': np.zeros(5), 'done': False}

    # Ajout normal
    buffer.add(exp)
    assert len(buffer) == 1

    # Ajout avec priorité personnalisée
    buffer.add(exp, priority=2.0)
    assert len(buffer) == 2

    # Dépassement de capacité (doit écraser la plus ancienne)
    buffer.add(exp)
    assert len(buffer) == 2

@pytest.mark.integration
def test_sample_edge_cases():
    """Test les cas limites de la méthode sample."""
    buffer = SharedExperienceBuffer(buffer_size=10)
    exp = {'state': np.zeros(5), 'action': 0, 'reward': 0.0, 'next_state': np.zeros(5), 'done': False}

    # Test avec buffer vide (devrait lever une exception)
    with pytest.raises(ValueError):
        buffer.sample(1)

    # Ajout d'expériences
    for i in range(5):
        buffer.add(exp)

    # Test avec batch_size > nombre d'expériences (devrait lever une exception)
    with pytest.raises(ValueError):
        buffer.sample(10)

@pytest.mark.integration
def test_get_stats():
    """Test la méthode get_stats."""
    buffer = SharedExperienceBuffer(buffer_size=10)
    exp = {'state': np.zeros(5), 'action': 0, 'reward': 0.0, 'next_state': np.zeros(5), 'done': False}

    # Stats avec buffer vide
    stats = buffer.get_stats()
    assert stats['max_size'] == 10
    assert stats['priority_max'] == 1.0

    # Ajout d'expériences
    for i in range(3):
        buffer.add(exp, priority=i+1)

    # Vérification des stats mises à jour
    stats = buffer.get_stats()
    assert stats['size'] == 3
    # La priorité maximale doit être la dernière priorité ajoutée (3.0)
    # transformée selon la formule PER: (abs(priority) + epsilon) ** alpha
    expected_priority = (3.0 + buffer.epsilon) ** buffer.alpha
    assert stats['priority_max'] == pytest.approx(expected_priority)

@pytest.mark.integration
def test_save_and_load(tmp_path):
    """Test la sauvegarde et le chargement du buffer."""
    # Création d'un buffer de test
    buffer1 = SharedExperienceBuffer(buffer_size=5)
    exp = {'state': np.ones(3), 'action': 1, 'reward': 1.0, 'next_state': np.ones(3), 'done': False}

    # Ajout de données
    for i in range(3):
        buffer1.add(exp, priority=i+1)

    # Sauvegarde
    save_path = tmp_path / "test_buffer.pkl"
    buffer1.save(str(save_path))

    # Chargement
    buffer2 = SharedExperienceBuffer.load(str(save_path))

    # Vérification
    assert len(buffer1) == len(buffer2)
    # La taille du buffer chargé est égale au nombre d'éléments chargés, pas à la taille d'origine
    assert buffer2.buffer_size == len(buffer1)
    assert buffer1.alpha == buffer2.alpha
    assert buffer1.beta == buffer2.beta

    # Vérification des priorités
    indices = list(range(len(buffer1)))
    p1, _ = buffer1._get_priority_weights(indices)
    p2, _ = buffer2._get_priority_weights(indices)
    assert np.allclose(p1, p2)

if __name__ == "__main__":
    test_concurrent_access()
    test_priority_updates()
    test_buffer_length_and_ready()
    test_add_edge_cases()
    test_sample_edge_cases()
    test_get_stats()
    test_save_and_load()
    print("Tous les tests d'intégration du SharedExperienceBuffer ont réussi !")
