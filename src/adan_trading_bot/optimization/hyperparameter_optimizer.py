"""
HyperparameterOptimizer - Optimisation automatisée des hyperparamètres avec Optuna.

Ce module implémente une interface pour optimiser les hyperparamètres des modèles
de trading en utilisant Optuna, avec support pour l'arrêt anticipé et l'élagage.
"""

from __future__ import annotations

# Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# Third-party imports
import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import Trial

# Configuration du logger
logger = logging.getLogger(__name__)

@dataclass
class HyperparameterOptimizer:
    """Classe pour l'optimisation des hyperparamètres avec Optuna.

    Attributes:
        study_name: Nom de l'étude Optuna
        storage_url: URL de stockage pour les études
        n_trials: Nombre d'essais d'optimisation
        timeout: Délai maximum en secondes (None pour illimité)
        direction: Direction d'optimisation ('minimize' ou 'maximize')
        sampler: Échantillonneur Optuna
        pruner: Élagueur Optuna
        n_jobs: Nombre de jobs parallèles (-1 = tous les cœurs)
    """

    study_name: str = "adan_hyperparameter_study"
    storage_url: str = "sqlite:///optuna_studies.db"
    n_trials: int = 100
    timeout: Optional[int] = 3600  # 1 heure par défaut
    direction: str = "maximize"
    sampler: Optional[BaseSampler] = None
    pruner: Optional[BasePruner] = None
    n_jobs: int = -1  # -1 = utiliser tous les cœurs

    def __post_init__(self):
        """Initialisation des composants Optuna."""
        if self.sampler is None:
            self.sampler = optuna.samplers.TPESampler()
        if self.pruner is None:
            self.pruner = optuna.pruners.MedianPruner()

    def create_study(self) -> Study:
        """Crée ou charge une étude Optuna."""
        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=self.direction,
            load_if_exists=True
        )

    def optimize(
        self,
        objective: Callable[..., float],
        param_distributions: Dict[str, Any],
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Study:
        """Exécute l'optimisation des hyperparamètres.

        Args:
            objective: Fonction objectif à optimiser
            param_distributions: Dictionnaire des distributions de paramètres
            n_trials: Nombre d'essais (remplace la valeur de l'instance)
            timeout: Délai maximum en secondes (remplace la valeur de l'instance)
            **kwargs: Arguments additionnels pour la fonction objectif

        Returns:
            L'étude Optuna complétée
        """
        study = self.create_study()
        
        def wrapped_objective(trial):
            """Enveloppe la fonction objectif pour passer les arguments."""
            params = self.suggest_hyperparameters(trial, param_distributions)
            return objective(trial, **params, **kwargs)

        study.optimize(
            wrapped_objective,
            n_trials=n_trials or self.n_trials,
            timeout=timeout or self.timeout,
            n_jobs=self.n_jobs
        )

        return study

    @staticmethod
    def suggest_hyperparameters(
        trial: Trial, 
        param_distributions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggère des valeurs d'hyperparamètres pour un essai.

        Args:
            trial: Essai Optuna
            param_distributions: Dictionnaire des distributions de paramètres

        Returns:
            Dictionnaire des paramètres suggérés
        """
        params = {}
        for name, distribution in param_distributions.items():
            dist_type = distribution.get('type', 'categorical')
            
            if dist_type == 'categorical':
                params[name] = trial.suggest_categorical(
                    name=name,
                    choices=distribution['choices']
                )
            elif dist_type == 'int':
                params[name] = trial.suggest_int(
                    name=name,
                    low=distribution['low'],
                    high=distribution['high'],
                    step=distribution.get('step', 1),
                    log=distribution.get('log', False)
                )
            elif dist_type == 'float':
                params[name] = trial.suggest_float(
                    name=name,
                    low=distribution['low'],
                    high=distribution['high'],
                    step=distribution.get('step'),
                    log=distribution.get('log', False)
                )
            elif dist_type == 'loguniform':
                params[name] = trial.suggest_float(
                    name,
                    low=distribution['low'],
                    high=distribution['high'],
                    log=True
                )
            else:
                raise ValueError(
                    f"Type de distribution non supporté: {dist_type}"
                )

        return params

    def get_best_params(self, study: Optional[Study] = None) -> Dict[str, Any]:
        """Récupère les meilleurs paramètres d'une étude.

        Args:
            study: Étude Optuna (si None, charge l'étude actuelle)

        Returns:
            Dictionnaire des meilleurs paramètres
        """
        study = study or self.create_study()
        return study.best_params

    def get_best_trial(self, study: Optional[Study] = None) -> optuna.trial.FrozenTrial:
        """Récupère le meilleur essai d'une étude.

        Args:
            study: Étude Optuna (si None, charge l'étude actuelle)

        Returns:
            Le meilleur essai
        """
        study = study or self.create_study()
        return study.best_trial
