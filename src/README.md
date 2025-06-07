# Code Source ADAN - Architecture et Modules

## 🏗️ Structure du Package `adan_trading_bot`

```
src/adan_trading_bot/
├── __init__.py
├── agents/                 # Agents RL et architectures CNN
│   ├── __init__.py
│   ├── custom_cnn_feature_extractor.py  # CNN pour données temporelles
│   └── ppo_agent.py                     # Agent PPO principal
├── common/                 # Utilitaires et helpers
│   ├── __init__.py
│   ├── callbacks.py        # Callbacks d'entraînement personnalisés
│   ├── constants.py        # Constantes globales
│   └── utils.py           # Fonctions utilitaires (chemins, configs)
├── data_processing/        # Pipeline de traitement des données
│   ├── __init__.py
│   ├── data_loader.py      # Chargement et fusion des données
│   └── feature_engineer.py # Calcul indicateurs techniques
└── environment/           # Environnement de trading Gymnasium
    ├── __init__.py
    ├── multi_asset_env.py   # Environnement multi-assets principal
    ├── order_manager.py     # Gestion des ordres de trading
    ├── reward_calculator.py # Calcul des récompenses
    └── state_builder.py     # Construction des observations
```

## 🧠 Modules Principaux

### `agents/` - Intelligence Artificielle

#### `custom_cnn_feature_extractor.py`
**Extracteur CNN pour données temporelles multi-features**

```python
class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN optimisé pour données de marché multi-assets
    - Input: (batch, 1, window_size, num_features)
    - Convolutions 1D pour patterns temporels
    - Output: features vectorielles pour PPO
    """
```

**Fonctionnalités :**
- Architecture CNN configurable via YAML
- Support fenêtres temporelles variables (15-30 steps)
- Dropout et normalisation pour régularisation
- Optimisé pour 95 features multi-assets

#### `ppo_agent.py`
**Agent PPO avec MultiInputPolicy**

```python
class ADanPPOAgent:
    """
    Agent principal utilisant Stable-Baselines3 PPO
    - Policy: MultiInputPolicy avec CNN+MLP
    - Support CPU/GPU avec configurations adaptées
    - Callbacks personnalisés intégrés
    """
```

### `environment/` - Simulation de Trading

#### `multi_asset_env.py`
**Environnement principal Gymnasium**

```python
class MultiAssetEnv(gym.Env):
    """
    Environnement multi-assets conforme Gymnasium
    - 5 cryptomonnaies simultanées
    - Actions: HOLD(0), BUY(1-5), SELL(6-10)
    - Observations: données CNN + vecteur état
    - Gestion paliers de capital adaptatifs
    """
```

**Fonctionnalités :**
- Space d'action: Discrete(11) - 1 HOLD + 5 BUY + 5 SELL
- Space d'observation: Dict avec CNN et données vectorielles
- Gestion positions multi-assets
- Calcul récompenses basé log-returns

#### `state_builder.py`
**Construction des observations**

```python
class StateBuilder:
    """
    Transforme les données brutes en observations RL
    - Fenêtre CNN: données normalisées (window_size, 95)
    - Vecteur état: capital, positions, prix
    - Validation features automatique
    """
```

**Validation critique :** Vérifie que toutes les 95 features attendues sont présentes

#### `order_manager.py`
**Gestion des ordres de trading**

```python
class OrderManager:
    """
    Exécution et suivi des ordres
    - Types: MARKET (implémenté), LIMIT/STOP (préparé)
    - Calcul quantités basé sur allocation
    - Gestion fees et slippage
    """
```

#### `reward_calculator.py`
**Système de récompenses**

```python
class RewardCalculator:
    """
    Calcul récompenses multi-niveaux
    - Paliers de capital avec multiplicateurs
    - Récompenses basées log-returns
    - Pénalités actions invalides
    """
```

### `data_processing/` - Pipeline de Données

#### `data_loader.py`
**Chargement et validation des données**

```python
class DataLoader:
    """
    Gestion des données fusionnées multi-assets
    - Chargement fichiers train/val/test
    - Validation structure (95 colonnes)
    - Support profils CPU/GPU
    """
```

#### `feature_engineer.py`
**Calcul des indicateurs techniques**

```python
class FeatureEngineer:
    """
    Pipeline indicateurs techniques par timeframe
    - Scalping (1m): EMAs, MACD, RSI, BBANDS, STOCH
    - Day Trading (1h): SMAs, EMA, MACD, RSI, BBANDS, STOCH
    - Swing Trading (1d): SMAs, MACD, RSI, BBANDS, OBV
    - Suffixes automatiques (_1m, _1h, _1d)
    """
```

**Indicateurs supportés :**
- pandas_ta: SMA, EMA, MACD, RSI, BBANDS, STOCH, OBV
- Normalisation StandardScaler (sauf OHLC)
- Validation cohérence noms

### `common/` - Utilitaires

#### `utils.py`
**Fonctions utilitaires système**

```python
def get_project_root() -> str:
    """Détection robuste racine projet ADAN"""

def get_path(key: str) -> str:
    """Résolution chemins via main_config.yaml"""

def load_config(path: str) -> dict:
    """Chargement configurations YAML"""
```

#### `callbacks.py`
**Callbacks d'entraînement personnalisés**

```python
class CustomTrainingInfoCallback(BaseCallback):
    """
    Monitoring avancé entraînement
    - Affichage périodique métriques
    - Sauvegarde checkpoints
    - Tables Rich (positions, récompenses)
    """
```

#### `constants.py`
**Constantes globales**

```python
# Actions possibles
ACTION_HOLD = 0
ACTION_BUY_START = 1
ACTION_SELL_START = 6

# Timeframes supportés
SUPPORTED_TIMEFRAMES = ["1m", "1h", "1d"]
```

## 🔄 Flux de Données

### 1. Pipeline de Traitement
```
Raw OHLCV → FeatureEngineer → Indicateurs + Normalisation → Fusion Multi-Assets → 95 colonnes finales
```

### 2. Entraînement RL
```
DataLoader → MultiAssetEnv → StateBuilder → CNN Observations → PPO Agent → Actions → OrderManager → Récompenses
```

### 3. Validation Critique
```
base_market_features (config) ↔ Colonnes DataFrames (reality) ↔ StateBuilder features (expected)
```

## ⚙️ Configuration et Intégration

### Profils d'Exécution
- **CPU** : Configurations légères pour tests
- **GPU** : Configurations maximisées pour production

### Points d'Intégration
- **Scripts** : Utilisent les modules via imports directs
- **Configs** : YAML centralisées dans `/config`
- **Données** : Flux standardisé `/data/raw` → `/data/processed` → `/data/processed/merged`

## 🔍 Points de Validation Critiques

### StateBuilder Feature Validation
```python
# Dans state_builder.py - vérification automatique
missing_features = []
for feature in self.base_feature_names:
    for asset in self.assets:
        column_name = f"{feature}_{asset}"
        if column_name not in market_data_window.columns:
            missing_features.append(column_name)
```

### CNN Input Shape Validation
```python
# Dans multi_asset_env.py
expected_shape = (self.cnn_input_window_size, len(self.base_feature_names) * len(self.assets))
# Doit être (15, 95) pour CPU ou (30, 95) pour GPU
```

## 🚀 Performance et Optimisations

### CPU Optimizations
- Fenêtre CNN : 15 steps
- Features : 32 dimensions
- Batch size : 64

### GPU Optimizations  
- Fenêtre CNN : 30 steps
- Features : 256 dimensions
- Batch size : 512
- Workers : 8 environnements parallèles

## 🔧 Extensibilité

### Ajout de Nouveaux Indicateurs
1. Modifier `config/data_config_*.yaml` section `indicators_by_timeframe`
2. Vérifier support dans `feature_engineer.py`
3. Mettre à jour `base_market_features` si nécessaire

### Nouveaux Assets
1. Ajouter dans `config/data_config_*.yaml` section `assets`
2. Régénérer pipeline complet
3. Les 95 colonnes deviennent N_assets × 19 features

### Nouveaux Timeframes
1. Ajouter configurations indicateurs
2. Étendre `data_split` et `timeframe_periods`
3. Mettre à jour `base_market_features` pour le `training_timeframe`

---

**🎯 Cette architecture modulaire garantit la maintenabilité, l'extensibilité et la robustesse du système ADAN.**