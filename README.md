# ADAN – Adaptive Dynamic Agent Network

**Version** : 0.1.0 | **Statut** : Production-Ready (avec corrections critiques en cours)

## 1. Vue d'ensemble

ADAN est un système de trading algorithmique multi-agents basé sur **PPO (Proximal Policy Optimization)** de Stable-Baselines3. Le projet implémente :

- **4 workers spécialisés** (W1, W2, W3, W4) avec des profils de risque distincts
- **Optimisation Optuna** pour les hyperparamètres (unique source valide)
- **Gestion des paliers de capital** (Micro → Enterprise) définis dans `config/trading.yaml`
- **Pipeline d'observation unifié** (entraînement ↔ inférence) avec normalisation VecNormalize
- **Paper trading** avec suivi en temps réel et validation des décisions

## 2. Structure du projet

```
ADAN/
├── config/                    # Configurations centrales
│   ├── config.yaml           # Config principale (agent, features, training)
│   ├── trading.yaml          # Paliers de capital et règles de trading
│   ├── training.yaml         # Paramètres d'entraînement
│   └── workers.yaml          # Configuration des workers
├── configs/                   # Templates de configuration (swing, scalper, etc.)
├── src/adan_trading_bot/      # Code source
│   ├── environment/          # Environnements d'entraînement (MultiAssetChunkedEnv)
│   ├── normalization/        # Pipeline de normalisation (VecNormalize)
│   ├── observation/          # Construction des observations
│   ├── agent/                # Logique des agents
│   ├── training/             # Boucles d'entraînement
│   ├── live_trading/         # Exécution live
│   ├── portfolio/            # Gestion du portefeuille
│   ├── risk_management/      # DBE (Dynamic Boundary Enforcement)
│   └── [autres modules]      # Indicateurs, validation, monitoring, etc.
├── scripts/                   # Scripts d'orchestration
│   ├── train_parallel_agents.py      # Entraînement multi-workers
│   ├── paper_trading_monitor.py      # Paper trading (à corriger)
│   ├── optimize_hyperparams.py       # Optuna (unique source valide)
│   └── [autres scripts]              # Monitoring, diagnostic, etc.
├── tests/                     # Batteries de tests (pytest)
├── models/                    # Modèles sauvegardés (8 workers)
│   └── worker_*/
│       ├── model.zip         # Modèle PPO
│       └── vecnormalize.pkl  # Statistiques de normalisation (CRITIQUE)
├── optuna_results/           # Études Optuna et résultats
├── historical_data/          # Données historiques préchargées
├── data/                     # Datasets additionnels
├── results/                  # Résultats d'expériences
├── api/                      # API REST (optionnel)
├── bot_pres/                 # Packaging et présentation
├── del/                      # Archives du nettoyage stratégique
├── README.md                 # Ce fichier
├── README_MODIFICATIONS.md   # Synthèse du nettoyage
├── README_CORRECTIONS.md     # Correctifs critiques à implémenter
├── requirements.txt          # Dépendances Python
├── pyproject.toml            # Configuration du projet
└── setup.py                  # Installation du package
```

## 3. Concepts clés

### 3.1 Workers et profils de risque

| Worker | Timeframe | Profil | Objectif |
|--------|-----------|--------|----------|
| **W1** | 4h | Ultra-stable | Sharpe > 0.5, DD < 15% |
| **W2** | 1h | Modéré | Profit factor > 1.0, Sharpe > 0.3 |
| **W3** | 5m | Agressif | Trades fréquents, DD < 25% |
| **W4** | Multi | Optimisé Sharpe | Sharpe > 1.0, DD < 10% |

### 3.2 Normalisation et covariate shift

**Problème** : Si les observations ne sont pas normalisées de manière cohérente entre l'entraînement et l'inférence, le modèle reçoit des distributions différentes et ses performances se dégradent.

**Solution** : 
1. Pendant l'entraînement : `VecNormalize` accumule les statistiques (mean, var) et les sauvegarde dans `models/worker_*/vecnormalize.pkl`
2. Pendant l'inférence : Charger ces statistiques via `VecNormalize.load(..., training=False)` et les appliquer aux observations

**État actuel** : 
- ✅ Entraînement : VecNormalize est utilisé correctement
- ⚠️ Paper trading : Normalisation manuelle (fenêtre glissante) – **À corriger** (voir `README_CORRECTIONS.md`)

### 3.3 Paliers de capital

Définis dans `config/trading.yaml`, les paliers contrôlent la taille des positions et le risque :

```yaml
capital_tiers:
  - name: Micro Capital
    min_capital: 11.0
    max_capital: 30.0
    max_position_size_pct: 90
    max_drawdown_pct: 50.0
    
  - name: Small Capital
    min_capital: 30.0
    max_capital: 100.0
    max_position_size_pct: 70
    max_drawdown_pct: 4.0
    
  # ... (Medium, High, Enterprise)
```

**Règle** : Ne jamais modifier ces paliers manuellement. Ils sont la source de vérité pour la gestion du risque.

### 3.4 Optimisation Optuna

**Unique source valide** : `scripts/optimize_hyperparams.py`

- Optimise les hyperparamètres PPO pour chaque worker
- Valide les trials avec des seuils stricts (Sharpe, DD, trades)
- Sauvegarde les résultats dans `optuna_results/`
- Les hyperparamètres validés sont injectés dans `config/config.yaml` avant l'entraînement

**Règle** : Seuls les trials `COMPLETE` avec métriques valides doivent être utilisés.

## 4. Workflows principaux

### 4.1 Entraînement

```bash
# 1. Optimiser les hyperparamètres (Optuna)
python scripts/optimize_hyperparams.py --worker W1 --trials 50

# 2. Injecter les meilleurs hyperparamètres dans config.yaml
# (fait automatiquement par optimize_hyperparams.py)

# 3. Entraîner les workers
python scripts/train_parallel_agents.py --config config/config.yaml --steps 1000000

# Résultat : models/worker_*/model.zip + models/worker_*/vecnormalize.pkl
```

### 4.2 Paper trading

```bash
# Lancer le monitor (à corriger pour utiliser VecNormalize)
python scripts/paper_trading_monitor.py --config config/config.yaml

# Résultat : Décisions d'achat/vente en temps réel (simulation)
```

### 4.3 Validation

```bash
# Tests unitaires
pytest tests/

# Validation de cohérence (entraînement ↔ inférence)
python scripts/validate_observation_spaces.py
```

## 5. Points de vigilance

### 5.1 Fichiers critiques à ne pas supprimer

- `models/worker_*/model.zip` : Modèles PPO entraînés (8 workers)
- `models/worker_*/vecnormalize.pkl` : Statistiques de normalisation (CRITIQUE pour l'inférence)
- `optuna_results/` : Historique des optimisations
- `config/trading.yaml` : Paliers de capital (source de vérité)

### 5.2 Dépendances compilées localement

- **ta-lib** : Compilée localement, ne pas supprimer
- Vérifier que `requirements.txt` et `setup.py` sont à jour

### 5.3 Règles de gouvernance

1. **Optuna** : Seul `scripts/optimize_hyperparams.py` peut modifier les hyperparamètres
2. **Paliers** : Ne jamais modifier `config/trading.yaml` manuellement
3. **Normalisation** : Toujours utiliser `VecNormalize.load()` en inférence
4. **Tests** : Valider tout changement avec `pytest tests/`

## 6. Documentation complémentaire

- **`README_MODIFICATIONS.md`** : Synthèse du nettoyage stratégique (fichiers archivés, structure finale)
- **`README_CORRECTIONS.md`** : Correctifs critiques à implémenter (normalisation paper trading, pipeline unifié, validations out-of-sample)
- **`del/README.md`** : Inventaire des artefacts archivés (216 fichiers, 782 MB)

## 7. Commandes rapides

```bash
# Installation
pip install -r requirements.txt
python setup.py develop

# Entraînement
python scripts/train_parallel_agents.py --config config/config.yaml --steps 1000000

# Paper trading
python scripts/paper_trading_monitor.py --config config/config.yaml

# Tests
pytest tests/ -v

# Monitoring
tensorboard --logdir=logs/
```

## 8. État du projet

| Aspect | Statut | Notes |
|--------|--------|-------|
| **Entraînement** | ✅ Fonctionnel | VecNormalize utilisé correctement |
| **Paper trading** | ⚠️ À corriger | Normalisation manuelle – voir `README_CORRECTIONS.md` |
| **Optuna** | ✅ Fonctionnel | Validation stricte des trials |
| **Paliers de capital** | ✅ Définis | `config/trading.yaml` |
| **Tests** | ✅ Présents | Batteries complètes dans `tests/` |
| **Documentation** | ✅ À jour | README, corrections, modifications |

---

**Dernière mise à jour** : 24 décembre 2025  
**Nettoyage stratégique** : Terminé (216 fichiers archivés dans `del/`)  
**Prochaines étapes** : Implémenter les correctifs de `README_CORRECTIONS.md`
