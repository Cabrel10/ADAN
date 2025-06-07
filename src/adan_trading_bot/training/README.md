# Module `training`

Ce module contient les composants nécessaires pour orchestrer l'entraînement des agents d'apprentissage par renforcement dans le projet ADAN.

## Contenu

* `trainer.py`: Classe ou fonction principale pour orchestrer le processus d'entraînement RL.
* `callbacks.py`: Callbacks personnalisés pour Stable-Baselines3, permettant de suivre et contrôler l'entraînement.

## Fonctionnalités Principales

### Orchestration de l'Entraînement
- Configuration et initialisation de l'environnement et de l'agent
- Gestion du processus d'entraînement complet
- Intégration avec les systèmes de logging et de suivi des métriques
- Gestion des checkpoints et de la reprise d'entraînement

### Callbacks Personnalisés
- Sauvegarde périodique des modèles pendant l'entraînement
- Évaluation sur des données de validation
- Arrêt anticipé basé sur des critères de performance
- Logging des métriques d'entraînement (récompenses, pertes)
- Visualisation en temps réel de l'apprentissage

## Exemple d'Utilisation

```python
from adan_trading_bot.training.trainer import RLTrainer
from adan_trading_bot.training.callbacks import EvaluationCallback, SaveCheckpointCallback

# Créer le trainer
trainer = RLTrainer(
    env_config_path="config/environment_config.yaml",
    agent_config_path="config/agent_config.yaml",
    data_path="data/processed/training_features_v1_processed.parquet"
)

# Configurer les callbacks
callbacks = [
    EvaluationCallback(
        eval_env=validation_env,
        eval_freq=10000,
        n_eval_episodes=5,
        log_path="reports/metrics/validation_results.json"
    ),
    SaveCheckpointCallback(
        save_freq=50000,
        save_path="models/rl_agents/checkpoints",
        name_prefix="ppo_checkpoint"
    )
]

# Lancer l'entraînement
trained_agent = trainer.train(
    total_timesteps=1000000,
    callbacks=callbacks,
    log_interval=1000
)

# Sauvegarder l'agent final
trainer.save_agent("models/rl_agents/ppo_final_v1")
```

## Intégration avec le Monitoring

Ce module s'intègre avec des outils de monitoring comme TensorBoard ou Weights & Biases pour suivre l'évolution de l'entraînement:
- Courbes d'apprentissage
- Distribution des actions
- Évolution du portefeuille simulé
- Métriques de performance (Sharpe, PnL, etc.)

## Bonnes Pratiques

1. Sauvegarder régulièrement les checkpoints pendant l'entraînement
2. Évaluer périodiquement l'agent sur des données de validation
3. Utiliser des callbacks pour détecter et réagir aux problèmes d'entraînement
4. Documenter les hyperparamètres et les conditions d'entraînement
