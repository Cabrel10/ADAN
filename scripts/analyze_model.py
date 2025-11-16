#!/usr/bin/python3
"""
Analyse du modèle de trading pour diagnostiquer la passivité.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    """Classe pour analyser le comportement d'un modèle de trading"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise l'analyseur de modèle.
        
        Args:
            model_path: Chemin vers le modèle à analyser
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation du périphérique: {self.device}")
        
        # Dossiers de sortie
        self.output_dir = Path("model_analysis")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_path: Optional[str] = None):
        """Charge le modèle à partir du chemin spécifié"""
        if model_path is None:
            model_path = self.model_path
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Aucun modèle trouvé à l'emplacement: {model_path}")
        
        try:
            # Essayer de charger avec stable-baselines3
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            logger.info(f"Modèle chargé avec succès depuis {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def analyze_action_distribution(self, n_samples: int = 1000):
        """Analyse la distribution des actions du modèle"""
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Appelez d'abord load_model().")
        
        logger.info(f"Analyse de la distribution des actions sur {n_samples} échantillons...")
        
        # Générer des observations aléatoires (à adapter selon l'environnement)
        obs_shape = self.model.observation_space.shape
        obs = np.random.normal(0, 1, (n_samples, *obs_shape)).astype(np.float32)
        
        # Prédire les actions
        actions = []
        with torch.no_grad():
            for i in range(0, n_samples, 100):  # Par lots pour économiser de la mémoire
                batch = obs[i:i+100]
                if hasattr(self.model, 'predict'):
                    # Pour les modèles Stable Baselines
                    batch_actions, _ = self.model.predict(batch, deterministic=False)
                else:
                    # Pour les modèles PyTorch bruts
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    batch_actions = self.model(batch_tensor).cpu().numpy()
                actions.append(batch_actions)
        
        actions = np.concatenate(actions, axis=0)
        
        # Calculer les statistiques
        stats = {
            'mean': np.mean(actions, axis=0),
            'std': np.std(actions, axis=0),
            'min': np.min(actions, axis=0),
            'max': np.max(actions, axis=0),
            'pct_above_001': np.mean(np.abs(actions) > 0.01, axis=0),
            'pct_above_003': np.mean(np.abs(actions) > 0.03, axis=0),
            'pct_above_005': np.mean(np.abs(actions) > 0.05, axis=0),
        }
        
        # Afficher les statistiques
        logger.info("Statistiques des actions :")
        for i in range(actions.shape[1]):
            logger.info(f"Action {i+1}:")
            logger.info(f"  Moyenne: {stats['mean'][i]:.6f} ± {stats['std'][i]:.6f}")
            logger.info(f"  Plage: [{stats['min'][i]:.6f}, {stats['max'][i]:.6f}]")
            logger.info(f"  % > 0.01: {stats['pct_above_001'][i]:.1%}")
            logger.info(f"  % > 0.03: {stats['pct_above_003'][i]:.1%}")
            logger.info(f"  % > 0.05: {stats['pct_above_005'][i]:.1%}")
        
        # Tracer les distributions
        self._plot_action_distributions(actions, stats)
        
        return stats
    
    def _plot_action_distributions(self, actions: np.ndarray, stats: Dict):
        """Trace les distributions des actions"""
        n_actions = actions.shape[1]
        fig, axes = plt.subplots(n_actions, 2, figsize=(12, 3 * n_actions))
        
        if n_actions == 1:
            axes = np.array([axes])
        
        for i in range(n_actions):
            # Histogramme des actions
            ax = axes[i, 0] if n_actions > 1 else axes[0]
            ax.hist(actions[:, i], bins=50, alpha=0.7, color='blue')
            ax.axvline(x=0.03, color='red', linestyle='--', label='Seuil (0.03)')
            ax.axvline(x=-0.03, color='red', linestyle='--')
            ax.set_title(f'Distribution des actions {i+1}')
            ax.set_xlabel('Valeur de l\'action')
            ax.set_ylabel('Fréquence')
            ax.legend()
            ax.grid(True)
            
            # Graphique des valeurs absolues
            ax = axes[i, 1] if n_actions > 1 else axes[1]
            ax.hist(np.abs(actions[:, i]), bins=50, alpha=0.7, color='green')
            ax.axvline(x=0.03, color='red', linestyle='--', label='Seuil (0.03)')
            ax.set_title(f'Valeurs absolues des actions {i+1}')
            ax.set_xlabel('|Valeur de l\'action|')
            ax.set_ylabel('Fréquence')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'action_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Graphiques des distributions sauvegardés dans {plot_path}")
    
    def analyze_gradients(self, n_samples: int = 100):
        """Analyse les gradients du modèle (nécessite un modèle entraînable)"""
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Appelez d'abord load_model().")
        
        # Vérifier si le modèle est un modèle PyTorch brut ou un modèle SB3
        if hasattr(self.model, 'policy'):
            # Modèle Stable Baselines 3
            model = self.model.policy
            model.train()  # Passer en mode entraînement pour les gradients
        else:
            # Modèle PyTorch brut
            model = self.model
            model.train()
        
        # Générer des données d'entraînement factices
        obs_shape = self.model.observation_space.shape if hasattr(self.model, 'observation_space') else (1, 10)
        dummy_input = torch.randn(n_samples, *obs_shape[1:], device=self.device, requires_grad=True)
        
        # Forward pass
        output = model(dummy_input)
        
        # Calculer une perte factice (moyenne des sorties au carré)
        loss = (output ** 2).mean()
        
        # Backward pass
        loss.backward()
        
        # Analyser les gradients
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.cpu().numpy()
                grad_stats[name] = {
                    'mean': float(np.mean(grad)),
                    'std': float(np.std(grad)),
                    'min': float(np.min(grad)),
                    'max': float(np.max(grad)),
                    'abs_mean': float(np.mean(np.abs(grad))),
                }
        
        # Afficher les statistiques des gradients
        logger.info("Statistiques des gradients :")
        for name, stats in grad_stats.items():
            logger.info(f"{name}:")
            logger.info(f"  Moyenne: {stats['mean']:.2e} ± {stats['std']:.2e}")
            logger.info(f"  Plage: [{stats['min']:.2e}, {stats['max']:.2e}]")
            logger.info(f"  Moyenne des valeurs absolues: {stats['abs_mean']:.2e}")
        
        # Tracer les gradients
        self._plot_gradient_distributions(grad_stats)
        
        return grad_stats
    
    def _plot_gradient_distributions(self, grad_stats: Dict):
        """Trace les distributions des gradients"""
        if not grad_stats:
            return
        
        n_layers = len(grad_stats)
        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 2 * n_layers))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, (name, stats) in enumerate(grad_stats.items()):
            # Extraire le nom court pour l'affichage
            short_name = name.split('.')[-1] if '.' in name else name
            
            # Tracer l'histogramme des gradients
            ax = axes[i]
            ax.hist(stats['gradient'], bins=50, alpha=0.7, label=short_name)
            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_title(f'Distribution des gradients - {short_name}')
            ax.set_xlabel('Valeur du gradient')
            ax.set_ylabel('Fréquence')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'gradient_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Graphiques des gradients sauvegardés dans {plot_path}")

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse d'un modèle de trading")
    parser.add_argument('--model', type=str, help="Chemin vers le modèle à analyser")
    parser.add_argument('--n-samples', type=int, default=1000, 
                       help="Nombre d'échantillons pour l'analyse")
    
    args = parser.parse_args()
    
    try:
        # Initialiser l'analyseur
        analyzer = ModelAnalyzer(args.model)
        
        # Charger le modèle
        if args.model:
            analyzer.load_model(args.model)
        else:
            # Essayer de trouver automatiquement un modèle
            model_path = None
            for path in ["models/latest_model.zip", "models/best_model.zip"]:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "Aucun modèle spécifié et aucun modèle trouvé dans le dossier 'models/'"
                )
            
            analyzer.load_model(model_path)
        
        # Exécuter les analyses
        print("\n" + "="*50)
        print("ANALYSE DE LA DISTRIBUTION DES ACTIONS")
        print("="*50)
        action_stats = analyzer.analyze_action_distribution(n_samples=args.n_samples)
        
        print("\n" + "="*50)
        print("ANALYSE DES GRADIENTS")
        print("="*50)
        try:
            grad_stats = analyzer.analyze_gradients(n_samples=min(100, args.n_samples))
        except Exception as e:
            logger.warning(f"Impossible d'analyser les gradients: {e}")
        
        print("\n" + "="*50)
        print("RÉSUMÉ DES RÉSULTATS")
        print("="*50)
        
        # Vérifier si les actions sont trop faibles
        action_magnitudes = [np.mean(np.abs(action)) for action in action_stats['mean']]
        if all(mag < 0.01 for mag in action_magnitudes):
            print("\n⚠️  ATTENTION: Les actions sont très faibles (moyenne < 0.01)")
            print("   Cela pourrait expliquer pourquoi le modèle ne trade pas assez.")
            print("   Solutions possibles:")
            print("   - Augmenter le learning rate")
            print("   - Réduire les pénalités dans la fonction de récompense")
            print("   - Ajouter un bonus pour les actions plus fortes")
        else:
            print("\n✅ Les actions semblent avoir une amplitude raisonnable.")
        
        # Vérifier les gradients
        if 'grad_stats' in locals():
            vanishing_grads = any(
                stats['abs_mean'] < 1e-7 for stats in grad_stats.values()
            )
            
            if vanishing_grads:
                print("\n⚠️  ATTENTION: Certains gradients sont très proches de zéro (vanishing gradients)")
                print("   Cela peut empêcher l'apprentissage des premières couches du réseau.")
                print("   Solutions possibles:")
                print("   - Utiliser des fonctions d'activation différentes (ReLU, LeakyReLU)")
                print("   - Ajouter de la normalisation par lots (BatchNorm)")
                print("   - Réduire la profondeur du réseau")
            else:
                print("\n✅ Les gradients semblent sains (pas de vanishing gradients détecté).")
        
        print("\n📊 Les graphiques d'analyse ont été sauvegardés dans le dossier 'model_analysis/'")
        
        return 0
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du modèle: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
