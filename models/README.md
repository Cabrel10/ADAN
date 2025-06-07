# Modèles Entraînés

Ce répertoire contient les modèles entraînés et les artefacts associés pour le projet ADAN.

## Structure

* `rl_agents/`: Modèles d'agents d'apprentissage par renforcement (PPO, etc.)
* `encoders/`: Auto-encodeurs ou autres modèles de réduction de dimensionnalité
* `baselines/`: Modèles de référence pour la comparaison des performances

## Conventions de Nommage

Les modèles suivent généralement la convention de nommage:
`{MODEL_TYPE}_{DESCRIPTION}_{VERSION}_{DATE}.{FORMAT}`

Exemples:
* `ppo_adan_v1_20231026.zip`
* `autoencoder_market_features_v2_20231015.h5`
* `baseline_buy_and_hold_v1_20231020.pkl`

## Gestion des Versions

Il est recommandé de conserver les différentes versions des modèles pour pouvoir suivre l'évolution des performances et revenir à des versions antérieures si nécessaire. Chaque modèle doit être accompagné d'un fichier de métadonnées ou d'un enregistrement dans les rapports décrivant:

* Les hyperparamètres utilisés
* Les données d'entraînement (version, période)
* Les performances obtenues
* Les particularités ou améliorations par rapport aux versions précédentes
