"""Configuration pour l'optimisation des hyperparamètres du modèle d'attention."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import mlflow
from datetime import datetime


@dataclass
class OptimizationConfig:
    """Configuration pour l'optimisation des hyperparamètres"""

    # Paramètres du modèle d'attention
    num_attention_heads: List[int] = None
    hidden_dims: List[int] = None
    learning_rates: List[float] = None

    # Paramètres d'entraînement
    n_epochs: int = 50
    batch_size: int = 64
    early_stopping_patience: int = 10

    # Paramètres du modèle CNN-PPO
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Paramètres d'optimisation
    n_trials: int = 100
    timeout: int = 3600  # 1 heure
    n_jobs: int = 1

    def __post_init__(self):
        """Initialise les valeurs par défaut si non spécifiées"""
        if self.num_attention_heads is None:
            self.num_attention_heads = [4, 8, 16]
        if self.hidden_dims is None:
            self.hidden_dims = [128, 256, 512]
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 3e-4, 1e-3]


def setup_mlflow(experiment_name: str = None) -> str:
    """Configure MLflow pour le suivi des expériences"""

    if experiment_name is None:
        experiment_name = f"adan_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Configuration MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    return experiment_name


def get_default_config() -> OptimizationConfig:
    """Retourne la configuration par défaut pour l'optimisation"""
    return OptimizationConfig()


def load_config_from_yaml(config_path: str) -> OptimizationConfig:
    """Charge la configuration depuis un fichier YAML"""
    import yaml

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return OptimizationConfig(**config_dict)


def save_config_to_yaml(config: OptimizationConfig, config_path: str):
    """Sauvegarde la configuration dans un fichier YAML"""
    import yaml

    # Convertir en dictionnaire
    config_dict = {
        'num_attention_heads': config.num_attention_heads,
        'hidden_dims': config.hidden_dims,
        'learning_rates': config.learning_rates,
        'n_epochs': config.n_epochs,
        'batch_size': config.batch_size,
        'early_stopping_patience': config.early_stopping_patience,
        'n_steps': config.n_steps,
        'gamma': config.gamma,
        'gae_lambda': config.gae_lambda,
        'clip_range': config.clip_range,
        'ent_coef': config.ent_coef,
        'vf_coef': config.vf_coef,
        'max_grad_norm': config.max_grad_norm,
        'n_trials': config.n_trials,
        'timeout': config.timeout,
        'n_jobs': config.n_jobs
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
