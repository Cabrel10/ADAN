# Module `agent`

Ce module implémente la logique de l'agent d'apprentissage par renforcement pour le projet ADAN.

## Contenu

* `ppo_agent.py`: Wrapper ou implémentation de l'agent PPO (Proximal Policy Optimization) utilisant Stable-Baselines3.
* `policy_networks.py`: Architectures personnalisées des réseaux de neurones pour la politique et la fonction de valeur.

## Fonctionnalités Principales

### Agent PPO
- Intégration avec la bibliothèque Stable-Baselines3
- Configuration des hyperparamètres via le fichier `config/agent_config.yaml`
- Méthodes pour l'entraînement, l'inférence et l'évaluation
- Gestion du chargement et de la sauvegarde des modèles

### Réseaux de Politique
- Architectures de réseaux de neurones adaptées au trading
- Réseaux de politique (policy network) pour déterminer les actions
- Réseaux de valeur (value network) pour estimer les récompenses futures
- Possibilité d'utiliser des architectures partagées ou séparées

## Exemple d'Utilisation

```python
from adan_trading_bot.agent.ppo_agent import PPOTradingAgent
from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv

# Créer l'environnement
env = MultiAssetEnv(...)

# Créer l'agent
agent = PPOTradingAgent(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

# Entraîner l'agent
agent.train(total_timesteps=1000000)

# Sauvegarder l'agent
agent.save("models/rl_agents/ppo_agent_v1")

# Charger un agent existant
loaded_agent = PPOTradingAgent.load("models/rl_agents/ppo_agent_v1", env=env)

# Utiliser l'agent pour prédire des actions
observation = env.reset()
action, _states = loaded_agent.predict(observation, deterministic=True)
```

## Personnalisation

Ce module est conçu pour être facilement extensible:
- Ajout de nouveaux algorithmes RL (A2C, SAC, etc.)
- Création d'architectures de réseaux personnalisées
- Implémentation de stratégies d'exploration spécifiques
- Intégration de mécanismes d'attention ou de mémoire

## Dépendances

Ce module dépend principalement des bibliothèques suivantes:
- stable-baselines3 pour les implémentations des algorithmes RL
- torch (PyTorch) pour les réseaux de neurones
- gym pour l'interface avec l'environnement
