# Contexte du Projet ADAN

## 🎯 Objectifs Techniques
- Résoudre l'erreur de dimension des observations (Shape Mismatch)
- Consolider la configuration dans un fichier unique `config.yaml`
- Améliorer la documentation technique

## 🧩 Composants Clés

### 1. StateBuilder
- **Fichier** : `src/adan_trading_bot/environment/state_builder.py`
- **Responsabilité** : Transforme les données brutes en observations pour le modèle
- **Problème actuel** : Incohérence entre la forme de sortie et l'espace d'observation défini

### 2. MultiAssetChunkedEnv
- **Fichier** : `src/adan_trading_bot/environment/multi_asset_chunked_env.py`
- **Responsabilité** : Gère l'environnement de trading avec chargement par chunks
- **Problème actuel** : `_setup_spaces()` ne correspond pas à la sortie du StateBuilder

## 🔍 Problèmes Connus
1. **Shape Mismatch** : L'observation renvoyée n'a pas la forme attendue par l'environnement
2. **Configuration fragmentée** : Plusieurs fichiers de configuration qui doivent être unifiés
3. **Documentation incomplète** : Besoin de mieux documenter les formats de données

## 🛠 Prochaines Étapes
1. [ ] Déboguer `state_builder.py` pour comprendre la forme exacte des sorties
2. [ ] Aligner `_setup_spaces()` dans `multi_asset_chunked_env.py`
3. [ ] Mettre à jour la documentation dans `SPEC.md`"
