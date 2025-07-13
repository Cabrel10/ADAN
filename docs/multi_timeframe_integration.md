# Intégration Multi-Timeframe dans ADAN

Ce document décrit les modifications apportées pour supporter l'intégration de données multi-timeframe dans le projet ADAN.

## Modifications effectuées

### 1. Gestion de la fraîcheur des données

Le fichier `merge_processed_data.py` a été mis à jour pour inclure une information de "fraîcheur" pour chaque timeframe :

- Ajout de la colonne `{tf}_minutes_since_update` pour chaque timeframe (1m, 1h, 3h)
- Cette colonne indique le nombre de minutes écoulées depuis la dernière mise à jour des données pour ce timeframe
- Pour le timeframe 1m, cette valeur est toujours 0 (mise à jour à chaque minute)
- Pour les timeframes supérieurs, cette valeur est incrémentée jusqu'à la prochaine bougie

### 2. Mise à jour de l'extracteur de caractéristiques

Le `CustomCNNFeatureExtractor` a été mis à jour pour :

- Supporter 3 canaux d'entrée (1m, 1h, 3h)
- Inclure des couches de normalisation par lots (batch normalization)
- Améliorer la gestion des valeurs aberrantes (NaN, Inf)
- Ajouter des logs de débogage pour le suivi des dimensions et des plages de valeurs

### 3. Script de test

Un script de test `test_observation_builder.py` a été ajouté pour vérifier :

- Le chargement correct des données multi-timeframe
- La construction des observations avec la bonne structure
- Le bon fonctionnement de l'extracteur de caractéristiques

## Prochaines étapes

1. **Valider la représentation des données**
   - Vérifier que les données sont correctement alignées entre les différents timeframes
   - S'assurer que les métadonnées de fraîcheur sont correctement calculées

2. **Entraîner un modèle de base**
   - Lancer un entraînement court pour valider que le pipeline complet fonctionne
   - Vérifier que la perte diminue et que les prédictions sont stables

3. **Optimiser l'architecture**
   - Ajuster la profondeur et la largeur du CNN en fonction des performances
   - Expérimenter avec différentes configurations de couches de pooling et de dropout

4. **Évaluer les performances**
   - Comparer les performances avec et sans les données multi-timeframe
   - Analyser l'impact de la fraîcheur des données sur les prédictions

## Exécution des tests

Pour exécuter le script de test :

```bash
conda activate trading_env
python scripts/test_observation_builder.py
```

## Résolution des problèmes

### Problèmes courants

1. **Dimensions incorrectes**
   - Vérifier que les données d'entrée ont la forme attendue : [batch, channels, height, width]
   - S'assurer que le nombre de canaux d'entrée correspond au nombre de timeframes

2. **Valeurs aberrantes**
   - Vérifier les logs pour détecter des valeurs NaN ou infinies
   - Ajouter des vérifications de plage pour les entrées et sorties

3. **Performances**
   - Surveiller l'utilisation de la mémoire GPU/CPU
   - Ajuster la taille des lots (batch size) si nécessaire
