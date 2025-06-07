# Guide des commandes du projet ADAN

Ce document présente les commandes à utiliser pour chaque opération du projet ADAN, avec des exemples d'utilisation pour les profils d'exécution CPU et GPU.

## Table des matières

1. [Profils d'exécution](#profils-dexécution)
2. [Téléchargement des données](#téléchargement-des-données)
3. [Traitement des données](#traitement-des-données)
4. [Fusion des données](#fusion-des-données)
5. [Test de l'environnement](#test-de-lenvironnement)
6. [Entraînement de l'agent](#entraînement-de-lagent)
7. [Vérification des configurations](#vérification-des-configurations)

## Profils d'exécution

Tous les scripts du projet ADAN supportent désormais les profils d'exécution via l'argument `--exec_profile`. Deux profils sont disponibles :

- `cpu` : Utilise les configurations optimisées pour CPU (par défaut)
- `gpu` : Utilise les configurations optimisées pour GPU

Ces profils déterminent quels fichiers de configuration sont chargés :
- `config/data_config_cpu.yaml` ou `config/data_config_gpu.yaml`
- `config/agent_config_cpu.yaml` ou `config/agent_config_gpu.yaml`

## Téléchargement des données

Pour télécharger les données historiques depuis Binance :

```bash
# Avec le profil CPU
python scripts/fetch_data_ccxt.py --exec_profile cpu

# Avec le profil GPU
python scripts/fetch_data_ccxt.py --exec_profile gpu
```

Options supplémentaires :
- `--main_config` : Chemin vers le fichier de configuration principal (défaut : `config/main_config.yaml`)
- `--data_config` : Chemin vers un fichier de configuration des données spécifique (défaut : basé sur le profil)

## Traitement des données

Pour traiter les données brutes et générer les indicateurs techniques :

```bash
# Avec le profil CPU
python scripts/process_data.py --exec_profile cpu

# Avec le profil GPU
python scripts/process_data.py --exec_profile gpu
```

Options supplémentaires :
- `--main_config` : Chemin vers le fichier de configuration principal (défaut : `config/main_config.yaml`)
- `--data_config` : Chemin vers un fichier de configuration des données spécifique (défaut : basé sur le profil)

## Fusion des données

Pour fusionner les données traitées par actif en un seul DataFrame par timeframe et split :

```bash
# Avec le profil CPU
python scripts/merge_processed_data.py --exec_profile cpu

# Avec le profil GPU
python scripts/merge_processed_data.py --exec_profile gpu
```

Options supplémentaires :
- `--timeframes` : Liste des timeframes à traiter (défaut : tous)
- `--splits` : Liste des splits à traiter (défaut : train, val, test)
- `--training-timeframe` : Timeframe principal pour l'entraînement

## Test de l'environnement

Pour tester l'environnement MultiAssetEnv avec les données fusionnées :

```bash
# Avec le profil CPU
python scripts/test_environment_with_merged_data.py --exec_profile cpu

# Avec le profil GPU
python scripts/test_environment_with_merged_data.py --exec_profile gpu
```

Options supplémentaires :
- `--initial_capital` : Capital initial pour le test (outrepasse la config)
- `--data_file` : Chemin vers le fichier de données fusionné à utiliser
- `--num_rows` : Nombre de lignes du dataset à utiliser pour le test
- `--timeframe` : Timeframe à utiliser pour le test (1m, 1h, 1d)
- `--split` : Split à utiliser pour le test (train, val, test)

## Entraînement de l'agent

Pour entraîner l'agent ADAN :

```bash
# Avec le profil CPU
python scripts/train_rl_agent.py --exec_profile cpu --device cpu

# Avec le profil GPU
python scripts/train_rl_agent.py --exec_profile gpu --device cuda
```

Options supplémentaires :
- `--device` : Appareil à utiliser pour l'entraînement (auto, cpu, cuda)
- `--main_config` : Chemin vers le fichier de configuration principal
- `--data_config` : Chemin vers le fichier de configuration des données
- `--env_config` : Chemin vers le fichier de configuration de l'environnement
- `--agent_config` : Chemin vers le fichier de configuration de l'agent
- `--logging_config` : Chemin vers le fichier de configuration du logging
- `--initial_capital` : Capital initial pour l'entraînement
- `--training_data_file` : Chemin vers le fichier de données d'entraînement
- `--validation_data_file` : Chemin vers le fichier de données de validation
- `--total_timesteps` : Nombre total de timesteps pour l'entraînement
- `--learning_rate` : Taux d'apprentissage pour l'agent
- `--batch_size` : Taille du batch pour l'entraînement

## Vérification des configurations

Pour vérifier que les fichiers de configuration nécessaires existent et peuvent être chargés correctement :

```bash
# Vérifier le profil CPU
python scripts/test_exec_profiles.py --exec_profile cpu

# Vérifier le profil GPU
python scripts/test_exec_profiles.py --exec_profile gpu
```

Ce script affiche également des informations sur les configurations chargées pour aider à diagnostiquer les problèmes potentiels.
