# Gestion des Données pour ADAN

Ce répertoire centralise toutes les données utilisées et générées par le projet ADAN.

## Arborescence

*   `raw/`: Contient les données brutes telles qu'elles ont été collectées (ex: historiques de prix OHLCV depuis un exchange). Ces données ne doivent pas être modifiées directement.
*   `interim/`: Données intermédiaires résultant d'un premier nettoyage ou d'une transformation légère (ex: gestion des valeurs manquantes, alignement temporel).
*   `processed/`: Données finales, nettoyées et enrichies de features (indicateurs techniques), prêtes à être utilisées pour l'entraînement et l'évaluation de l'agent.
*   `scalers_encoders/`: Artefacts de transformation des données sauvegardés, tels que les scalers (ex: `StandardScaler` de scikit-learn) ou les auto-encodeurs entraînés, pour assurer une transformation cohérente entre l'entraînement et l'inférence.

## Workflow Typique

1.  Les données brutes sont placées dans `raw/` (souvent via un script de `scripts/fetch_data.py`).
2.  Des scripts de prétraitement (ex: `scripts/preprocess_data.py` utilisant les modules de `src/adan_trading_bot/data_processing/`) lisent les données de `raw/`, effectuent des transformations et sauvegardent les résultats dans `interim/` et/ou `processed/`.
3.  Les scalers et encoders ajustés sur les données d'entraînement sont sauvegardés dans `scalers_encoders/`.
