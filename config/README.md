# Configuration ADAN - Guide Complet

## 🎯 Vue d'Ensemble

Le dossier `config/` contient toutes les configurations YAML du projet ADAN, organisées par domaines fonctionnels et profils d'exécution (CPU/GPU).

## 📁 Structure des Fichiers

```
config/
├── main_config.yaml          # Configuration principale et chemins
├── data_config_cpu.yaml      # Données et indicateurs (CPU)
├── data_config_gpu.yaml      # Données et indicateurs (GPU)
├── agent_config_cpu.yaml     # Agent RL et CNN (CPU)
├── agent_config_gpu.yaml     # Agent RL et CNN (GPU)
├── environment_config.yaml   # Environnement de trading
└── logging_config.yaml       # Configuration logs
```

## 🔧 Configurations Principales

### `main_config.yaml`
**Configuration générale du projet**

```yaml
project:
  name: "ADAN"
  version: "2.0.0"
  description: "Agent de Décision Algorithmique Neuronal"

paths:
  # Noms des répertoires (utilisés par utils.get_path())
  data_dir_name: "data"
  models_dir_name: "models"
  config_dir_name: "config"
  reports_dir_name: "reports"
  src_dir_name: "src"
  scripts_dir_name: "scripts"
  docs_dir_name: "docs"
  tests_dir_name: "tests"
  notebooks_dir_name: "notebooks"
```

### `environment_config.yaml`
**Environnement de trading et règles métier**

```yaml
# Capital et allocation
initial_capital: 10000.0
min_trade_amount: 10.0
max_position_percentage: 0.95

# Système de paliers adaptatifs
capital_tiers:
  - threshold: 0      # Débutant
    max_positions: 1
    allocation_frac_per_pos: 0.95
    reward_pos_mult: 1.5
    reward_neg_mult: 0.8
  - threshold: 10     # Intermédiaire
    max_positions: 1
    allocation_frac_per_pos: 0.95
  - threshold: 30     # Avancé
    max_positions: 2
    allocation_frac_per_pos: 0.45
  # ... jusqu'à threshold: 500

# Récompenses et pénalités
reward_shaping:
  time_penalty: -0.0001
  invalid_action_penalty: -0.01
  large_loss_penalty_threshold: -0.05
  large_loss_penalty: -0.02

# Frais de trading
fees:
  maker_fee: 0.001
  taker_fee: 0.001
  withdrawal_fee: 0.0005
```

**Points clés :**
- **Paliers adaptatifs** : Plus de capital = plus de positions simultanées
- **Allocation dynamique** : Pourcentage par position varie selon le palier
- **Récompenses graduées** : Multiplicateurs selon le niveau de capital

## 📊 Configurations de Données

### `data_config_cpu.yaml` vs `data_config_gpu.yaml`

#### Différences Principales

| Paramètre | CPU | GPU | Impact |
|-----------|-----|-----|--------|
| `cnn_input_window_size` | 15 | 30 | Contexte temporel CNN |
| `features_dim` | 32 | 256 | Capacité extracteur |
| `total_timesteps` | 200k | 2M | Durée entraînement |
| Période données | Réduite | Maximisée | Volume historique |

#### Structure Commune

```yaml
# Assets et timeframes
assets: ["DOGEUSDT", "XRPUSDT", "LTCUSDT", "SOLUSDT", "ADAUSDT"]
timeframes: ["1m", "1h", "1d"]
training_timeframe: "1h"

# Features de base (19 au total pour timeframe 1h)
base_market_features: [
  "open", "high", "low", "close", "volume",          # 5 OHLCV
  "sma_50_1h", "sma_200_1h", "ema_20_1h",           # 3 MAs
  "macd_1h", "macd_signal_1h", "macd_hist_1h",      # 3 MACD
  "rsi_14_1h",                                       # 1 RSI
  "bbl_20_2.0_1h", "bbm_20_2.0_1h", "bbu_20_2.0_1h", # 5 Bollinger
  "bbb_20_2.0_1h", "bbp_20_2.0_1h",
  "stoch_k_1h", "stoch_d_1h"                         # 2 Stochastic
]

# Indicateurs par timeframe (spécialisation par style de trading)
indicators_by_timeframe:
  "1m":    # Scalping - EMAs rapides
    - {name: "EMA 5", function: "ema", params: {length: 5}, output_col_name: "ema_5_1m"}
    - {name: "EMA 8", function: "ema", params: {length: 8}, output_col_name: "ema_8_1m"}
    # ... + MACD, RSI, BBANDS, STOCH (14 indicateurs total)
  
  "1h":    # Day Trading - SMAs + EMA
    - {name: "SMA 50", function: "sma", params: {length: 50}, output_col_name: "sma_50_1h"}
    - {name: "SMA 200", function: "sma", params: {length: 200}, output_col_name: "sma_200_1h"}
    # ... + EMA, MACD, RSI, BBANDS, STOCH (14 indicateurs total)
  
  "1d":    # Swing Trading - SMAs long terme
    - {name: "SMA 50", function: "sma", params: {length: 50}, output_col_name: "sma_50_1d"}
    - {name: "SMA 100", function: "sma", params: {length: 100}, output_col_name: "sma_100_1d"}
    # ... + MACD, RSI, BBANDS, OBV (13 indicateurs total)
```

**🎯 Point Critique :** `base_market_features` DOIT correspondre exactement aux indicateurs générés pour le `training_timeframe`

## 🧠 Configurations Agent RL

### `agent_config_cpu.yaml` vs `agent_config_gpu.yaml`

#### Architecture CNN

| Composant | CPU | GPU |
|-----------|-----|-----|
| `features_dim` | 32 | 256 |
| Couches conv | 2 (16→32) | 3 (32→64→128) |
| FC layers | [64] | [512, 256] |
| Net arch (pi/vf) | [64, 32] | [512, 256, 128] |

#### Hyperparamètres PPO

| Paramètre | CPU | GPU | Description |
|-----------|-----|-----|-------------|
| `n_envs` | 1 | 8 | Environnements parallèles |
| `n_steps` | 1024 | 4096 | Steps par rollout |
| `batch_size` | 64 | 512 | Taille batch |
| `total_timesteps` | 200k | 2M | Steps total |

#### Structure Commune

```yaml
agent:
  algorithm: "ppo"
  policy_type: "MultiInputPolicy"
  seed: 42
  deterministic_inference: true

policy:
  activation_fn: "tanh"
  gamma: 0.99
  learning_rate: 0.0003
  lr_schedule: "constant"

ppo:
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  n_epochs: 10
```

## 📝 Configuration des Logs

### `logging_config.yaml`

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: training_log.txt
    mode: a

loggers:
  adan_trading_bot:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

## 🔧 Utilisation des Profils

### Sélection Automatique

```python
# Dans les scripts
parser.add_argument('--exec_profile', choices=['cpu', 'gpu'], default='cpu')

# Chargement conditionnel
if args.exec_profile == 'cpu':
    data_config = load_config('config/data_config_cpu.yaml')
    agent_config = load_config('config/agent_config_cpu.yaml')
elif args.exec_profile == 'gpu':
    data_config = load_config('config/data_config_gpu.yaml')
    agent_config = load_config('config/agent_config_gpu.yaml')
```

### Validation des Configurations

```bash
# Test des profils
python scripts/test_exec_profiles.py --exec_profile cpu
python scripts/test_exec_profiles.py --exec_profile gpu
```

## ⚠️ Points Critiques

### 1. Cohérence des Features
**`base_market_features` doit être synchronisé avec `indicators_by_timeframe[training_timeframe]`**

```yaml
# Si training_timeframe: "1h"
# Alors base_market_features doit contenir :
# - OHLCV (5) + tous les output_col_name des indicateurs "1h" (14) = 19 total
```

### 2. Suffixes de Timeframe
**Tous les indicateurs ont des suffixes automatiques**

```yaml
# Configuration
output_col_name: "sma_50_1h"  # ✅ Correct
# PAS output_col_name: "sma_50"  # ❌ Incorrect
```

### 3. Calcul des Colonnes Finales
**95 colonnes = 19 features × 5 assets**

```
Structure finale des données fusionnées :
- open_ADAUSDT, sma_50_1h_ADAUSDT, ..., stoch_d_1h_ADAUSDT (19 colonnes)
- open_DOGEUSDT, sma_50_1h_DOGEUSDT, ..., stoch_d_1h_DOGEUSDT (19 colonnes)
- ... (×5 assets)
= 95 colonnes total
```

## 🚀 Personnalisation Avancée

### Ajout d'Indicateurs

1. **Modifier `indicators_by_timeframe`**
```yaml
"1h":
  # Indicateurs existants...
  - name: "VWAP"
    function: "vwap"
    params: {length: 20}
    output_col_name: "vwap_20_1h"
```

2. **Mettre à jour `base_market_features`**
```yaml
base_market_features: [
  # Features existantes...
  "vwap_20_1h"  # Nouvelle feature
]
```

3. **Régénérer les données**
```bash
python scripts/process_data.py --exec_profile cpu
python scripts/merge_processed_data.py --exec_profile cpu
```

### Optimisation GPU

Pour machines avec plus de VRAM :
```yaml
# Dans agent_config_gpu.yaml
ppo:
  batch_size: 1024  # Augmenter si >8GB VRAM
  n_steps: 8192     # Plus de steps par rollout

features_extractor_kwargs:
  features_dim: 512  # Extracteur plus puissant
```

## 📊 Templates de Configuration

### Nouveau Timeframe
```yaml
# Dans data_config_*.yaml
"4h":  # Nouveau timeframe
  - name: "SMA 20"
    function: "sma"
    params: {length: 20}
    output_col_name: "sma_20_4h"
  # ... autres indicateurs
```

### Nouveau Asset
```yaml
# Ajouter dans assets
assets: ["DOGEUSDT", "XRPUSDT", "LTCUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT"]
# Résultat : 6 assets × 19 features = 114 colonnes
```

---

**🎯 Ces configurations permettent une personnalisation complète d'ADAN tout en maintenant la cohérence du système.**