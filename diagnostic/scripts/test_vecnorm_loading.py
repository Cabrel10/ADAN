
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Définition de l'environnement factice, nécessaire pour le chargement
class DummyTradingEnv(gym.Env):
    """
    Environnement factice avec l'espace d'observation/action correct,
    nécessaire pour charger les statistiques de VecNormalize.
    """
    def __init__(self):
        super().__init__()
        # Doit correspondre EXACTEMENT à l'environnement d'entraînement
        # Note: Les shapes sont simplifiées pour le test, l'important est la structure du Dict
        self.observation_space = spaces.Dict({
            '5m': spaces.Box(low=-np.inf, high=np.inf, shape=(20, 14), dtype=np.float32),
            '1h': spaces.Box(low=-np.inf, high=np.inf, shape=(20, 14), dtype=np.float32),
            '4h': spaces.Box(low=-np.inf, high=np.inf, shape=(20, 14), dtype=np.float32),
            'portfolio_state': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Retourne une observation vide avec les bonnes clés et shapes
        obs = {
            '5m': np.zeros((20, 14), dtype=np.float32),
            '1h': np.zeros((20, 14), dtype=np.float32),
            '4h': np.zeros((20, 14), dtype=np.float32),
            'portfolio_state': np.zeros(20, dtype=np.float32)
        }
        return obs, {}

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

def test_vecnorm_loading():
    """
    Charge les fichiers vecnormalize.pkl pour chaque worker et extrait leurs statistiques.
    """
    print("--- Checkpoint 1.2: Test de Chargement des Statistiques VecNormalize ---")
    
    # Le script est exécuté depuis la racine du projet, donc le chemin relatif est correct
    # D'après le `ls -lR`, les modèles finaux sont dans `models/rl_agents/`
    base_path = Path("./models/rl_agents") 
    results_path = Path("./diagnostic/results")
    worker_ids = ["w1", "w2", "w3", "w4"]
    all_stats = {}
    success_count = 0

    if not base_path.exists():
        print(f"❌ ERREUR: Le dossier des modèles '{base_path}' est introuvable.")
        return

    for wid in worker_ids:
        print(f"\n[INFO] Traitement du worker: {wid}")
        # Le `ls -lR` montre que les `vecnormalize.pkl` ne sont pas dans les sous-dossiers w1,w2.. mais peut-être à la racine des checkpoints
        # Je vais chercher à l'emplacement le plus probable basé sur la structure vue
        vecnorm_path = base_path / "final" / f"{wid}_vecnormalize.pkl"
        
        # Si non trouvé, vérifier un autre emplacement possible (basé sur d'anciennes exécutions)
        if not vecnorm_path.exists():
             vecnorm_path = Path("./checkpoints/final/") / f"{wid}_vecnormalize.pkl"
             if not vecnorm_path.exists():
                print(f"  [WARN] Fichier non trouvé: {vecnorm_path} (et autres emplacements).")
                all_stats[wid] = {'path': str(vecnorm_path), 'status': 'NOT_FOUND'}
                continue

        try:
            dummy_env = DummyVecEnv([lambda: DummyTradingEnv()])
            vecnorm_env = VecNormalize.load(str(vecnorm_path), dummy_env)
            
            obs_rms = vecnorm_env.obs_rms
            if obs_rms and '5m' in obs_rms:
                mean = obs_rms['5m'].mean.tolist()
                var = obs_rms['5m'].var.tolist()
                
                all_stats[wid] = {
                    'path': str(vecnorm_path),
                    'status': 'SUCCESS',
                    'sample_mean_5m_shape': np.array(mean).shape,
                    'sample_var_5m_shape': np.array(var).shape
                }
                print(f"  [SUCCESS] Chargement et extraction des stats pour {wid} réussis.")
                success_count += 1
            else:
                all_stats[wid] = {'path': str(vecnorm_path), 'status': 'FAILURE', 'error': "Clé '5m' non trouvée dans les statistiques RMS."}
                print(f"  [ERROR] Clé '5m' non trouvée dans les stats de {wid}.")

        except Exception as e:
            all_stats[wid] = {'path': str(vecnorm_path), 'status': 'FAILURE', 'error': str(e)}
            print(f"  [ERROR] Échec du chargement pour {wid}: {e}")

    # Sauvegarder le rapport
    report_path = results_path / "vecnorm_loading_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=4)

    print("\n--- Résultat du Checkpoint 1.2 ---")
    print(f"Rapport de chargement sauvegardé dans : {report_path}")
    if success_count == len(worker_ids):
        print(f"✅ SUCCÈS: {success_count}/{len(worker_ids)} fichiers VecNormalize ont été chargés et analysés avec succès.")
    else:
        print(f"❌ ÉCHEC: {success_count}/{len(worker_ids)} fichiers seulement ont pu être traités. Veuillez consulter le rapport.")

if __name__ == "__main__":
    test_vecnorm_loading()
