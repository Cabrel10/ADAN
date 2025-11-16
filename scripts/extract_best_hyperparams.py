#!/usr/bin/python3
"""
Extrait les 5 meilleurs hyperparamètres pour configurer les 4 workers spécialistes.
Utilise les résultats d'optimisation Optuna.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Add project-specific imports
from src.adan_trading_bot.common.config_loader import ConfigLoader
from src.adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from src.adan_trading_bot.evaluation.decision_quality_analyzer import DecisionQualityAnalyzer

# Couleurs
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class HyperparameterExtractor:
    """Extrait les meilleurs hyperparamètres pour les 4 workers."""

    def __init__(self, project_root: Path = None):
        """Initialise l'extracteur."""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = project_root
        self.results_dir = project_root / "results"
        self.configs_dir = project_root / "configs"
        
        # Charger la configuration et les données du marché pour l'analyse
        print("🔄 Chargement de la configuration et des données de marché pour l'analyse...")
        try:
            self.config = ConfigLoader.load_config(project_root / "config" / "config.yaml")
            self.market_data = self._load_market_data()
            print(f"{GREEN}✅ Données de marché chargées.{RESET}")
        except Exception as e:
            print(f"{RED}❌ Erreur lors du chargement de la configuration ou des données: {e}{RESET}")
            self.config = {}
            self.market_data = pd.DataFrame()

    def _load_market_data(self) -> pd.DataFrame:
        """Charge un échantillon de données de marché pour l'analyse."""
        try:
            # Utiliser la configuration du premier worker pour charger les données
            worker_config = self.config["workers"]["w1"]
            data_loader = ChunkedDataLoader(config=self.config, worker_config=worker_config, worker_id=0)
            # Charger un chunk de données représentatif
            data_chunk = data_loader.load_chunk(0) 
            # Aplatir les données en un seul DataFrame pour l'analyseur
            # Note: L'analyseur actuel n'utilise pas les données marché, mais c'est une bonne pratique de les fournir.
            # On prend les données du premier asset et timeframe comme référence.
            first_asset = list(data_chunk.keys())[0]
            first_tf = list(data_chunk[first_asset].keys())[0]
            return data_chunk[first_asset][first_tf]
        except Exception as e:
            print(f"{YELLOW}⚠️  Impossible de charger les données de marché pour l'analyse: {e}{RESET}")
            return pd.DataFrame()

    def extract_best_hyperparams(self) -> Dict:
        """Extrait les 5 meilleurs hyperparamètres."""
        print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
        print(f"{BOLD}{BLUE}EXTRACTION DES 5 MEILLEURS HYPERPARAMÈTRES{RESET}")
        print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

        # Charger les résultats d'optimisation
        trials_data = self._load_optimization_results()

        if trials_data is None or len(trials_data) == 0:
            print(f"{RED}❌ Aucun résultat d'optimisation trouvé{RESET}")
            return {}

        # Trier par score
        sorted_trials = self._sort_trials_by_score(trials_data)

        # Extraire les 5 meilleurs
        top_5 = sorted_trials[:5]

        # Afficher les résultats
        self._display_top_5(top_5)

        # Générer les configurations pour chaque worker
        worker_configs = self._generate_worker_configs(top_5)

        # Sauvegarder les configurations
        self._save_worker_configs(worker_configs)

        return worker_configs

    def _load_optimization_results(self) -> List[Dict]:
        """Charge les résultats d'optimisation."""
        trials_data = []

        # Chercher les fichiers de résultats
        result_files = [
            self.results_dir / "best_aggressive_pareto.json",
            self.results_dir / "best_position_pareto.json",
            self.results_dir / "best_scalper_pareto.json",
            self.results_dir / "best_swing_pareto.json",
        ]

        for result_file in result_files:
            if result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            trials_data.append(data)
                        elif isinstance(data, list):
                            trials_data.extend(data)
                except Exception as e:
                    print(f"{YELLOW}⚠️  Erreur lecture {result_file}: {e}{RESET}")

        # Chercher aussi les fichiers JSON dans results/optuna_trials
        optuna_dir = self.results_dir / "optuna_trials"
        if optuna_dir.exists():
            for trial_dir in optuna_dir.rglob("result.json"):
                try:
                    with open(trial_dir, "r") as f:
                        data = json.load(f)
                        trials_data.append(data)
                except Exception as e:
                    print(f"{YELLOW}⚠️  Erreur lecture {trial_dir}: {e}{RESET}")

        return trials_data

    def _sort_trials_by_score(self, trials_data: List[Dict]) -> List[Dict]:
        """Trie les trials par score (Sharpe Ratio ou reward)."""
        scored_trials = []

        for trial in trials_data:
            score = self._extract_score(trial)
            trial_with_score = {**trial, "_score": score}
            scored_trials.append(trial_with_score)

        # Trier par score décroissant
        sorted_trials = sorted(scored_trials, key=lambda x: x["_score"], reverse=True)

        return sorted_trials

    def _extract_score(self, trial: Dict) -> float:
        """Extrait le score d'un trial, en privilégiant le score de réflexion."""
        
        # Essayer d'utiliser l'analyseur de qualité de décision
        if "metrics" in trial and "trades" in trial["metrics"] and isinstance(trial["metrics"]["trades"], list) and len(trial["metrics"]["trades"]) > 0:
            try:
                trades_df = pd.DataFrame(trial["metrics"]["trades"])
                # L'analyseur a besoin de la colonne 'pnl'
                if "pnl" in trades_df.columns:
                    analyzer = DecisionQualityAnalyzer(trades_df, self.market_data)
                    quality_metrics = analyzer.analyze()
                    print(f"✅ Trial analysé avec un Score de Réflexion de: {quality_metrics.reflection_score:.2f}")
                    return quality_metrics.reflection_score
            except Exception as e:
                print(f"{YELLOW}⚠️  Impossible de calculer le score de réflexion pour le trial: {e}{RESET}")

        # Fallback sur les anciennes méthodes de score
        metrics = trial.get("metrics", trial) # Look in metrics dict first
        score = 0
        if "score" in metrics:
            score = metrics["score"]
        elif "sharpe_ratio" in metrics:
            score = metrics["sharpe_ratio"]
        elif "reward" in metrics:
            score = metrics["reward"]
        elif "pnl" in metrics:
            score = metrics["pnl"]
        elif "fitness" in metrics:
            score = metrics["fitness"]
        elif "value" in metrics:
            score = metrics["value"]
        
        print(f"⚠️  Utilisation du score de fallback: {score:.4f}")
        return score

    def _display_top_5(self, top_5: List[Dict]) -> None:
        """Affiche les 5 meilleurs trials."""
        print(f"{BOLD}🏆 TOP 5 MEILLEURS HYPERPARAMÈTRES{RESET}\n")

        for rank, trial in enumerate(top_5, 1):
            score = trial.get("_score", 0)
            print(f"{BOLD}{BLUE}Rang #{rank}{RESET} - Score: {GREEN}{score:.4f}{RESET}")

            # Afficher les hyperparamètres
            for key, value in trial.items():
                if not key.startswith("_") and key not in ["timestamp", "trial_id"]:
                    if isinstance(value, float):
                        print(f"  {key:30} : {value:.6f}")
                    elif isinstance(value, dict):
                        print(f"  {key:30} : {json.dumps(value, indent=2)}")
                    else:
                        print(f"  {key:30} : {value}")

            print()

    def _generate_worker_configs(self, top_5: List[Dict]) -> Dict:
        """Génère les configurations pour chaque worker."""
        worker_names = ["aggressive", "position", "scalper", "swing"]
        worker_configs = {}

        # Assigner les meilleurs hyperparamètres à chaque worker
        for i, worker_name in enumerate(worker_names):
            if i < len(top_5):
                trial = top_5[i]
                config = self._create_worker_config(worker_name, trial)
            else:
                # Utiliser le meilleur si pas assez de trials
                config = self._create_worker_config(worker_name, top_5[0])

            worker_configs[worker_name] = config

        return worker_configs

    def _create_worker_config(self, worker_name: str, trial: Dict) -> Dict:
        """Crée une configuration pour un worker."""
        config = {
            "name": worker_name,
            "type": "specialist",
            "hyperparameters": {},
            "score": trial.get("_score", 0),
            "created_at": datetime.now().isoformat(),
        }

        # Extraire les hyperparamètres pertinents
        relevant_keys = [
            "learning_rate",
            "batch_size",
            "n_steps",
            "n_epochs",
            "clip_range",
            "ent_coef",
            "gamma",
            "gae_lambda",
            "max_grad_norm",
            "vf_coef",
            "exploration_fraction",
            "initial_eps",
            "final_eps",
        ]

        for key in relevant_keys:
            if key in trial:
                config["hyperparameters"][key] = trial[key]

        # Ajouter les hyperparamètres spécifiques au worker
        config["hyperparameters"].update(
            self._get_worker_specific_params(worker_name)
        )

        return config

    def _get_worker_specific_params(self, worker_name: str) -> Dict:
        """Retourne les paramètres spécifiques à chaque worker."""
        params = {
            "aggressive": {
                "risk_level": "high",
                "position_size_multiplier": 1.5,
                "max_positions": 5,
                "take_profit_ratio": 0.05,
                "stop_loss_ratio": 0.02,
            },
            "position": {
                "risk_level": "medium",
                "position_size_multiplier": 1.0,
                "max_positions": 3,
                "take_profit_ratio": 0.03,
                "stop_loss_ratio": 0.015,
            },
            "scalper": {
                "risk_level": "low",
                "position_size_multiplier": 0.5,
                "max_positions": 10,
                "take_profit_ratio": 0.01,
                "stop_loss_ratio": 0.005,
            },
            "swing": {
                "risk_level": "medium-high",
                "position_size_multiplier": 1.2,
                "max_positions": 2,
                "take_profit_ratio": 0.08,
                "stop_loss_ratio": 0.03,
            },
        }

        return params.get(worker_name, {})

    def _save_worker_configs(self, worker_configs: Dict) -> None:
        """Sauvegarde les configurations des workers."""
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.configs_dir / "worker_configs_best_5.json"

        with open(output_file, "w") as f:
            json.dump(worker_configs, f, indent=2)

        print(f"{GREEN}✅ Configurations sauvegardées: {output_file}{RESET}")

        # Afficher un résumé
        print(f"\n{BOLD}📋 RÉSUMÉ DES CONFIGURATIONS{RESET}\n")
        for worker_name, config in worker_configs.items():
            print(f"{BOLD}{BLUE}{worker_name.upper()}{RESET}")
            print(f"  Score: {GREEN}{config['score']:.4f}{RESET}")
            print(f"  Hyperparamètres: {len(config['hyperparameters'])} paramètres")
            print()

    def validate_configs(self) -> bool:
        """Valide les configurations générées."""
        print(f"\n{BOLD}🔍 VALIDATION DES CONFIGURATIONS{RESET}\n")

        config_file = self.configs_dir / "worker_configs_best_5.json"

        if not config_file.exists():
            print(f"{RED}❌ Fichier de configuration non trouvé{RESET}")
            return False

        try:
            with open(config_file, "r") as f:
                configs = json.load(f)

            # Vérifier la structure
            required_workers = ["aggressive", "position", "scalper", "swing"]
            for worker in required_workers:
                if worker not in configs:
                    print(f"{RED}❌ Worker '{worker}' manquant{RESET}")
                    return False

                config = configs[worker]
                if "hyperparameters" not in config:
                    print(f"{RED}❌ Hyperparamètres manquants pour '{worker}'{RESET}")
                    return False

                print(f"{GREEN}✅ {worker.upper()}: OK{RESET}")

            print(f"\n{GREEN}✅ Toutes les configurations sont valides{RESET}")
            return True

        except Exception as e:
            print(f"{RED}❌ Erreur validation: {e}{RESET}")
            return False


def main():
    """Fonction principale."""
    extractor = HyperparameterExtractor()

    # Extraire les meilleurs hyperparamètres
    worker_configs = extractor.extract_best_hyperparams()

    # Valider les configurations
    if worker_configs:
        extractor.validate_configs()

        print(f"\n{BOLD}{GREEN}{'='*80}{RESET}")
        print(f"{BOLD}{GREEN}✅ EXTRACTION COMPLÈTE{RESET}")
        print(f"{BOLD}{GREEN}{'='*80}{RESET}\n")

        print(f"Les 5 meilleurs hyperparamètres ont été extraits et configurés pour:")
        print(f"  • Aggressive Worker")
        print(f"  • Position Worker")
        print(f"  • Scalper Worker")
        print(f"  • Swing Worker")
        print(f"\nFichier de configuration: {extractor.configs_dir}/worker_configs_best_5.json")

        return 0
    else:
        print(f"\n{RED}❌ Impossible d'extraire les hyperparamètres{RESET}\n")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
