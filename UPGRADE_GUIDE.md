# Guide de Mise à Jour et d'Exécution d'ADAN sur Machine Cible

Ce guide s'adresse à l'utilisateur ayant reçu une version précédente du projet et souhaitant mettre à jour vers la version stable actuelle et lancer un entraînement.

## Prérequis

- Le code source complet du projet ADAN (dernière version stable) a été transféré sur votre machine
- Vos données de marché (nouvelles données 1m dans `data/new/` ou données legacy dans `data/backup/`) sont présentes sur votre machine
- Un environnement Conda nommé `trading_env` avec Python 3.11 et toutes les dépendances est créé et activé
- Pour l'entraînement GPU : Drivers NVIDIA, CUDA Toolkit, et PyTorch avec support CUDA installés

## Étapes de Mise à Jour et d'Exécution

### 1. Remplacer le Code Existant

- Sauvegardez vos configurations personnalisées si vous en avez
- Remplacez l'intégralité du code de votre version "compromise" par les fichiers de cette nouvelle version stable (dossiers `config`, `scripts`, `src`)

### 2. Vérifier/Adapter les Configurations (TRÈS IMPORTANT)

#### Configuration principale
- Ouvrez `config/main_config.yaml`
- Adaptez `paths.base_project_dir_local` pour qu'il pointe vers la racine de VOTRE copie du projet ADAN sur votre machine

#### Configuration des données
- Ouvrez `config/data_config_cpu.yaml` (ou `_gpu.yaml` si vous ciblez le GPU)

**data_sources** : Vérifiez que cette section pointe vers les bons répertoires où se trouvent vos fichiers Parquet sources :
```yaml
data_sources:
  - group_name: "new_1m_features"  # Pour les nouvelles données 1m riches
    directory: "new"
    assets: ["ADAUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"]
    timeframes: ["1m"]
    filename_pattern: "{ASSET}_features.parquet"
    features_ready: true
  - group_name: "legacy_backup"    # Pour les données legacy si disponibles
    directory: "backup"
    assets: ["ADAUSDT", "DOGEUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"]
    timeframes: ["1m", "1h", "1d"]
    filename_pattern: "{ASSET}/{ASSET}_{TIMEFRAME}_raw.parquet"
```

**assets** : Assurez-vous que cette liste correspond aux actifs que vous voulez utiliser et pour lesquels vous avez des données.

**training_timeframe** : Choisissez "1m", "1h", ou "1d" selon vos données disponibles.

**base_market_features** : 
- **CETTE LISTE EST AUTOMATIQUEMENT MISE À JOUR** selon le `training_timeframe`
- Pour les nouvelles données 1m : 47 features riches sont automatiquement utilisées
- Pour les données legacy 1h/1d : 18 features standard sont utilisées

#### Configuration de l'agent
- Ouvrez `config/agent_config_cpu.yaml` (ou `_gpu.yaml`)
- Adaptez `total_timesteps`, `batch_size`, `n_steps`, et les architectures réseau selon vos ressources
- Pour GPU : Augmentez les paramètres selon votre VRAM disponible

### 3. Installer les Dépendances Manquantes

```bash
# Activer l'environnement conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trading_env

# Installer pandas_ta si manquant
pip install pandas_ta

# Configurer les locales pour l'affichage rich
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

### 4. Exécuter le Pipeline de Données Complet

#### Nettoyer les anciennes données traitées
```bash
cd /chemin/vers/votre/ADAN
rm -rf data/processed/* data/scalers_encoders/*
```

#### Traiter les données
```bash
# Pour les nouvelles données 1m (recommandé)
python scripts/process_data.py --exec_profile cpu --data_config config/data_config_cpu.yaml --main_config config/main_config.yaml

# Fusionner les données traitées
python scripts/merge_processed_data.py --exec_profile cpu --timeframes 1m --splits train val test --training-timeframe 1m
```

#### Vérifier que les fichiers ont été créés
```bash
# Vérifier les données fusionnées
ls -la data/processed/merged/
# Doit afficher : 1m_train_merged.parquet, 1m_val_merged.parquet, 1m_test_merged.parquet
```

### 5. Lancer l'Entraînement

#### Test rapide (validation du système)
```bash
# Test avec 10 timesteps et épisodes courts
python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 10 --max_episode_steps 5
```

#### Entraînement court (CPU)
```bash
# Entraînement de développement avec les nouvelles données 1m
python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 5000 --max_episode_steps 1000
```

#### Entraînement long (CPU)
```bash
# Entraînement sérieux sur CPU
python scripts/train_rl_agent.py --exec_profile cpu --device cpu --initial_capital 15 --total_timesteps 500000
```

#### Entraînement GPU (si disponible)
```bash
# Vérifier d'abord que training_timeframe dans data_config_gpu.yaml est configuré
# et que agent_config_gpu.yaml a des paramètres adaptés à votre VRAM
python scripts/train_rl_agent.py --exec_profile gpu --device cuda --initial_capital 15 --total_timesteps 1000000
```

### 6. Surveiller l'Entraînement

#### Console
- Regardez la sortie console qui devrait être formatée par rich
- Si l'affichage est cassé, utilisez : `TERM=dumb python scripts/train_rl_agent.py ...`

#### TensorBoard
```bash
# Dans un autre terminal
tensorboard --logdir reports/tensorboard_logs/
# Puis ouvrir http://localhost:6006
```

#### Fichiers de sortie
- Modèles sauvegardés : `models/`
- Logs TensorBoard : `reports/tensorboard_logs/`
- Exports d'épisodes : `reports/` (si activé)

## Résolution de Problèmes Courants

### Erreur "No module named 'pandas_ta'"
```bash
pip install pandas_ta
```

### Erreur "Aucune source de données trouvée"
- Vérifiez que vos fichiers de données sont dans les bons répertoires
- Adaptez la section `data_sources` dans `data_config_*.yaml`

### Erreur de colonnes manquantes
- Pour les nouvelles données 1m : assurez-vous que `training_timeframe: "1m"` dans la config
- Pour les données legacy : assurez-vous que `training_timeframe: "1h"` ou `"1d"`

### Performance lente
- Réduisez `total_timesteps` pour les tests
- Réduisez `max_episode_steps` pour des épisodes plus courts
- Utilisez GPU si disponible

### Erreur de mémoire
- Réduisez `batch_size` et `n_steps` dans `agent_config_*.yaml`
- Réduisez le nombre d'actifs dans `data_config_*.yaml`

## Configuration des Nouvelles Données 1m

Si vous avez les fichiers `*_features.parquet` dans `data/new/` :

1. Assurez-vous que `training_timeframe: "1m"` dans `data_config_*.yaml`
2. Les 47 features riches seront automatiquement utilisées
3. 5 actifs supportés : ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, XRPUSDT
4. Période : 2024-01-01 à 2025-02-01 (données minutaires)
5. Split automatique : Train (11 mois) / Val (1 mois) / Test (1 mois)

## Configuration des Données Legacy

Si vous avez les données dans `data/backup/` :

1. Configurez `training_timeframe: "1h"` ou `"1d"`
2. 18 features standard seront utilisées
3. Actifs legacy : ADAUSDT, DOGEUSDT, LTCUSDT, SOLUSDT, XRPUSDT

## Support

En cas de problème :
1. Vérifiez les logs dans la console
2. Vérifiez que les fichiers de données sont présents et accessibles
3. Testez avec `--total_timesteps 10` pour validation rapide
4. Consultez les fichiers de configuration pour les chemins et paramètres

Ce guide devrait vous permettre de mettre à jour votre version et de lancer des entraînements avec la base stabilisée et la flexibilité pour vos données.