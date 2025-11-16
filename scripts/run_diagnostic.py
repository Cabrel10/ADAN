#!/usr/bin/python3
"""
Script de diagnostic pour analyser la passivité du modèle de trading.
Exécute une série de tests et génère des rapports détaillés.
"""

import os
import sys
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

# Configuration du logging
from scripts.setup_diagnostic_logging import setup_diagnostic_logging
logger = logging.getLogger(__name__)

class DiagnosticTool:
    """Outil de diagnostic pour analyser le comportement du modèle de trading"""
    
    def __init__(self):
        self.log_file = setup_diagnostic_logging()
        logger.info("Initialisation de l'outil de diagnostic")
        
        # Créer les répertoires nécessaires
        self.diagnostic_dir = Path("diagnostics")
        self.diagnostic_dir.mkdir(exist_ok=True)
        
        # Configuration par défaut
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Charge la configuration par défaut pour le diagnostic"""
        return {
            "n_episodes": 5,  # Nombre d'épisodes pour le diagnostic
            "max_steps": 1000,  # Nombre maximal de pas par épisode
            "output_dir": str(self.diagnostic_dir),
            "log_interval": 10,  # Intervalle de logging
        }
    
    def run_diagnostic(self):
        """Exécute le diagnostic complet"""
        logger.info("Démarrage du diagnostic complet")
        
        # 1. Vérifier l'environnement
        self._check_environment()
        
        # 2. Analyser les données d'entrée
        self._analyze_input_data()
        
        # 3. Tester le modèle (si disponible)
        if self._check_model_exists():
            self._analyze_model()
        else:
            logger.warning("Aucun modèle trouvé. L'analyse du modèle sera ignorée.")
        
        # 4. Générer le rapport final
        self._generate_final_report()
        
        logger.info("Diagnostic terminé avec succès")
    
    def _check_environment(self):
        """Vérifie que l'environnement est correctement configuré"""
        logger.info("Vérification de l'environnement...")
        
        # Vérifier les dépendances
        self._check_dependencies()
        
        # Vérifier les dossiers nécessaires
        required_dirs = ["data", "models", "logs"]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"Dossier créé: {dir_name}")
    
    def _check_dependencies(self):
        """Vérifie que toutes les dépendances sont installées"""
        logger.info("Vérification des dépendances...")
        
        required_packages = [
            "numpy", "pandas", "matplotlib", "torch", "stable_baselines3",
            "gym", "pyyaml", "tqdm", "scikit-learn"
        ]
        
        missing = []
        for pkg in required_packages:
            try:
                __import__(pkg)
                logger.debug(f"✓ {pkg} est installé")
            except ImportError:
                missing.append(pkg)
        
        if missing:
            logger.warning(f"Packages manquants: {', '.join(missing)}")
            logger.info("Essayez d'installer les dépendances avec: pip install -r requirements.txt")
    
    def _check_model_exists(self) -> bool:
        """Vérifie si un modèle existe pour analyse"""
        models_dir = Path("models")
        return any(models_dir.glob("*.zip")) or any(models_dir.glob("*/*.zip"))
    
    def _analyze_input_data(self):
        """Analyse les données d'entrée du modèle"""
        logger.info("Analyse des données d'entrée...")
        
        # TODO: Implémenter l'analyse des données d'entrée
        # - Vérifier la présence des fichiers de données
        # - Analyser les statistiques descriptives
        # - Vérifier les valeurs manquantes
        # - Générer des visualisations
        
        logger.warning("L'analyse des données d'entrée n'est pas encore implémentée")
    
    def _analyze_model(self):
        """Analyse le comportement du modèle"""
        logger.info("Analyse du modèle...")
        
        # TODO: Implémenter l'analyse du modèle
        # - Charger le modèle
        # - Analyser les sorties du modèle
        # - Tester différentes entrées
        # - Générer des visualisations
        
        logger.warning("L'analyse du modèle n'est pas encore implémentée")
    
    def _generate_final_report(self):
        """Génère un rapport de diagnostic final"""
        logger.info("Génération du rapport final...")
        
        report_path = self.diagnostic_dir / "rapport_diagnostic.md"
        
        with open(report_path, "w") as f:
            f.write("# Rapport de Diagnostic du Modèle de Trading\n\n")
            f.write("## Résumé\n\n")
            f.write("Ce rapport présente les résultats du diagnostic du modèle de trading.\n\n")
            
            # Section Environnement
            f.write("## 1. Environnement\n\n")
            f.write("### Dépendances\n")
            f.write("- Python: " + sys.version.split()[0] + "\n")
            f.write("- Système d'exploitation: " + sys.platform + "\n\n")
            
            # Section Données
            f.write("## 2. Données\n\n")
            f.write("*Analyse des données non encore implémentée*\n\n")
            
            # Section Modèle
            f.write("## 3. Modèle\n\n")
            if self._check_model_exists():
                f.write("Un modèle a été détecté mais l'analyse n'est pas encore implémentée.\n\n")
            else:
                f.write("Aucun modèle trouvé pour l'analyse.\n\n")
            
            # Section Recommandations
            f.write("## 4. Recommandations\n\n")
            f.write("1. **Vérifier les données d'entrée**\n")
            f.write("   - S'assurer que les données sont correctement chargées et normalisées\n")
            f.write("   - Vérifier les valeurs manquantes ou aberrantes\n\n")
            f.write("2. **Analyser le modèle**\n")
            f.write("   - Vérifier l'architecture du modèle\n")
            f.write("   - Analyser les gradients pendant l'entraînement\n\n")
            f.write("3. **Ajuster les hyperparamètres**\n")
            f.write("   - Expérimenter avec différents taux d'apprentissage\n")
            f.write("   - Ajuster les paramètres d'exploration\n\n")
        
        logger.info(f"Rapport généré: {report_path}")
        
        # Générer un graphique exemple
        self._generate_example_plot()
    
    def _generate_example_plot(self):
        """Génère un graphique exemple pour le rapport"""
        plt.figure(figsize=(10, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        plt.plot(x, y, label='Données simulées')
        plt.title("Exemple de visualisation")
        plt.xlabel("Temps")
        plt.ylabel("Valeur")
        plt.legend()
        
        plot_path = self.diagnostic_dir / "exemple_visualisation.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique exemple généré: {plot_path}")

def main():
    """Fonction principale"""
    try:
        diagnostic = DiagnosticTool()
        diagnostic.run_diagnostic()
        print("\n✅ Diagnostic terminé avec succès!")
        print(f"Consultez les rapports dans le dossier: {diagnostic.diagnostic_dir}")
        return 0
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du diagnostic: {str(e)}", exc_info=True)
        print(f"\n❌ Erreur lors de l'exécution du diagnostic: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
