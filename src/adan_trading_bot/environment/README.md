# Module `environment`

Ce module implémente l'environnement d'apprentissage par renforcement personnalisé pour le trading d'actifs financiers dans le projet ADAN.

## Contenu

* `multi_asset_env.py`: Classe principale `MultiAssetEnv` qui implémente l'interface Gym pour l'apprentissage par renforcement.
* `order_manager.py`: Gestion des différents types d'ordres (MARKET, LIMIT, STOP) et de leur exécution.
* `reward_calculator.py`: Calcul des fonctions de récompense pour l'agent RL.
* `state_builder.py`: Construction du vecteur d'état/observation à partir des données de marché et du portefeuille.

## Fonctionnalités Principales

### Environnement Multi-Actifs
- Simulation de trading sur plusieurs actifs simultanément
- Gestion du portefeuille et de l'allocation d'actifs
- Calcul des frais de transaction réalistes
- Mécanismes de stop-loss et take-profit

### Gestion des Ordres
- Ordres au marché (exécution immédiate)
- Ordres limites (exécution à un prix spécifié)
- Ordres stop (déclenchement à un seuil de prix)
- Simulation de slippage et de latence

### Fonctions de Récompense
- Récompenses basées sur le PnL (Profit and Loss)
- Récompenses ajustées au risque (Sharpe, Sortino)
- Récompenses avec pénalités pour les drawdowns
- Récompenses personnalisables via la configuration

### Construction de l'État
- Intégration des features techniques
- Informations sur le portefeuille actuel
- Historique des actions précédentes
- Normalisation et mise à l'échelle des observations

## Interface Gym

L'environnement suit l'interface standard de Gym:

```python
from adan_trading_bot.environment.multi_asset_env import MultiAssetEnv

# Créer l'environnement
env = MultiAssetEnv(
    data=processed_data,
    window_size=20,
    commission=0.001,
    initial_balance=10000,
    assets=["BTC", "ETH"]
)

# Réinitialiser l'environnement
observation = env.reset()

# Boucle d'interaction
done = False
while not done:
    action = agent.predict(observation)  # L'agent prend une décision
    next_observation, reward, done, info = env.step(action)
    observation = next_observation
```

## Configuration

L'environnement est hautement configurable via le fichier `config/environment_config.yaml`, permettant d'ajuster:
- Les frais de transaction
- Les paramètres de slippage
- Les contraintes de trading (taille min/max des positions)
- Les paramètres des fonctions de récompense
- Les features à inclure dans l'observation
