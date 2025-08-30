import os
import sys
import unittest
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any

import yaml

# On se place toujours au root du projet
ROOT = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from adan_trading_bot.training.shared_experience_buffer import SharedExperienceBuffer
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader


def worker_run_env(worker_id: int, config: Dict[str, Any], worker_config: Dict[str, Any], shared_buffer: SharedExperienceBuffer, num_steps: int):
    """
    Fonction exécutée par chaque worker pour interagir avec l'environnement et ajouter des expériences au buffer partagé.
    """
    print(f"Worker {worker_id} started.")
    env = MultiAssetChunkedEnv(config=config, worker_config=worker_config, shared_buffer=shared_buffer)
    obs, info = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample() # Action aléatoire pour le test
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"Worker {worker_id} finished.")

class TestSharedExperienceBuffer(unittest.TestCase):
    """
    Tests d'intégration pour le SharedExperienceBuffer.
    """
    @classmethod
    def setUpClass(cls):
        cls.full_config = ConfigLoader.load_config()
        if 'workers' not in cls.full_config or not cls.full_config['workers']:
            raise unittest.SkipTest("Section 'workers' non trouvée ou vide dans config.yaml")

    def test_shared_buffer_with_parallel_workers(self):
        """
        Teste que le SharedExperienceBuffer collecte des expériences de plusieurs workers en parallèle.
        """
        buffer_size = 1000
        num_workers = 2
        steps_per_worker = 50

        shared_buffer = SharedExperienceBuffer(buffer_size=buffer_size)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                worker_id = f"w{i+1}"
                worker_config = self.full_config['workers'].get(worker_id)
                if not worker_config:
                    self.fail(f"Worker {worker_id} configuration not found.")
                
                # Assurez-vous que la configuration du worker est un dictionnaire Python standard
                # et non un OmegaConf DictConfig pour la sérialisation entre processus
                worker_config_dict = ConfigLoader.to_python_dict(worker_config)

                futures.append(executor.submit(worker_run_env, i + 1, self.full_config, worker_config_dict, shared_buffer, steps_per_worker))

            for future in futures:
                future.result() # Attendre que tous les workers aient terminé

        # Vérifier que le buffer contient des expériences
        self.assertGreater(len(shared_buffer), 0, "Le buffer partagé ne devrait pas être vide.")
        self.assertLessEqual(len(shared_buffer), buffer_size, "Le buffer ne doit pas dépasser sa taille maximale.")

        print(f"Buffer final size: {len(shared_buffer)}")
        print(f"Buffer stats: {shared_buffer.get_stats()}")

        # Vérifier qu'un échantillon peut être tiré
        if len(shared_buffer) > 0:
            batch_size = min(32, len(shared_buffer))
            batch, indices, weights = shared_buffer.sample(batch_size)
            self.assertEqual(len(batch['state']), batch_size, "La taille du batch échantillonné est incorrecte.")
            self.assertEqual(len(indices), batch_size, "Le nombre d'indices échantillonnés est incorrect.")
            self.assertEqual(len(weights), batch_size, "Le nombre de poids échantillonnés est incorrect.")

if __name__ == '__main__':
    unittest.main()