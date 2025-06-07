# Modèles d'Agents RL

Ce répertoire contient les modèles d'agents d'apprentissage par renforcement entraînés pour le projet ADAN.

## Contenu Typique

* Modèles PPO (Proximal Policy Optimization)
* Modèles A2C (Advantage Actor-Critic)
* Modèles SAC (Soft Actor-Critic)
* Autres architectures d'agents RL

## Conventions de Nommage

Les fichiers suivent généralement la convention de nommage:
`{ALGORITHM}_{DESCRIPTION}_{VERSION}_{DATE}.{FORMAT}`

Exemples:
* `ppo_multi_asset_v1_20231026.zip`
* `sac_btc_eth_v2_20231115.zip`
* `a2c_custom_reward_v3_20231201.zip`

## Remarque

Ces modèles sont généralement sauvegardés au format compatible avec la bibliothèque utilisée (par exemple, Stable-Baselines3). Ils contiennent à la fois la politique (policy) et la fonction de valeur (value function), ainsi que les paramètres d'entraînement.
