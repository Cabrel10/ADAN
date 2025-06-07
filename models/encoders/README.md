# Encodeurs

Ce répertoire contient les modèles d'encodeurs entraînés pour la réduction de dimensionnalité ou l'extraction de caractéristiques.

## Contenu Typique

* Auto-encodeurs pour la compression des features de marché
* Encodeurs variationnels (VAE)
* Réseaux de neurones pour l'extraction de caractéristiques
* Autres modèles de représentation d'apprentissage

## Conventions de Nommage

Les fichiers suivent généralement la convention de nommage:
`{ENCODER_TYPE}_{DESCRIPTION}_{VERSION}_{DATE}.{FORMAT}`

Exemples:
* `autoencoder_market_features_v1_20231015.h5`
* `vae_technical_indicators_v2_20231120.h5`
* `feature_extractor_price_patterns_v1_20231205.keras`

## Utilisation

Ces encodeurs sont généralement utilisés pour:
1. Réduire la dimensionnalité des données d'entrée
2. Extraire des caractéristiques latentes significatives
3. Améliorer la généralisation de l'agent RL
4. Faciliter la visualisation des données

Ils sont intégrés dans le pipeline de prétraitement des données et/ou dans l'architecture de l'agent RL.
