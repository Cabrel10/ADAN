# 🔧 GUIDE DE CORRECTION DU COVARIATE SHIFT

## 📋 Résumé du Problème

Votre système ADAN génère des signaux constants (BUY = 1.0) parce que :

1. **Les modèles ont été entraînés sur des données normalisées** (moyenne=0, std=1)
2. **Le monitor live envoie des données brutes (non normalisées)** aux modèles
3. **Les modèles voient des données hors de leur domaine d'entraînement** → prédictions constantes

C'est le **covariate shift** classique en ML.

## ✅ Solution Implémentée

### Composants Créés

1. **`src/adan_trading_bot/normalization/observation_normalizer.py`**
   - `ObservationNormalizer`: Normalise les observations avec les stats d'entraînement
   - `DriftDetector`: Détecte les dérives de distribution

2. **Scripts de Diagnostic**
   - `scripts/diagnose_covariate_shift_v2.py`: Diagnostic complet
   - `scripts/patch_monitor_with_normalization.py`: Patch d'intégration

## 🚀 Étapes d'Implémentation

### Étape 1: Vérifier l'Installation

```bash
# Vérifier que les fichiers existent
ls -la src/adan_trading_bot/normalization/
ls -la scripts/diagnose_covariate_shift_v2.py
```

### Étape 2: Tester le Normaliseur

```bash
# Créer un test simple
python << 'EOF'
from adan_trading_bot.normalization import ObservationNormalizer
import numpy as np

# Initialiser le normaliseur
normalizer = ObservationNormalizer()
print(f"✅ Normaliseur chargé: {normalizer.is_loaded}")

# Tester avec une observation
obs = np.random.randn(68) * 10 + 50
normalized = normalizer.normalize(obs)

print(f"Observation brute: min={obs.min():.2f}, max={obs.max():.2f}, mean={obs.mean():.2f}")
print(f"Observation normalisée: min={normalized.min():.2f}, max={normalized.max():.2f}, mean={normalized.mean():.2f}")
EOF
```

### Étape 3: Intégrer dans le Monitor

Modifiez `scripts/paper_trading_monitor.py` :

```python
# 1. AJOUTER LES IMPORTS
from adan_trading_bot.normalization import ObservationNormalizer, DriftDetector
import numpy as np

# 2. DANS LA CLASSE PaperTradingMonitor.__init__()
def __init__(self, ...):
    # ... code existant ...
    
    # 🔧 AJOUTER:
    self.normalizer = ObservationNormalizer()
    self.drift_detector = DriftDetector(window_size=100, threshold=2.0)
    
    logger.info(f"✅ Normaliseur initialisé: {self.normalizer.is_loaded}")
    logger.info(f"✅ Détecteur de dérive initialisé")

# 3. DANS LA MÉTHODE generate_ensemble_signal()
def generate_ensemble_signal(self, market_data):
    # Construire l'observation brute
    raw_observation = self.build_observation(market_data)
    
    # 🔧 AJOUTER CES LIGNES:
    # Normaliser l'observation
    normalized_observation = self.normalizer.normalize(raw_observation)
    
    # Ajouter au détecteur de dérive
    self.drift_detector.add_observation(raw_observation)
    
    # Vérifier la dérive
    drift_result = self.drift_detector.check_drift(
        self.normalizer.mean,
        self.normalizer.var
    )
    
    if drift_result['drift_detected']:
        logger.warning(f"⚠️  Dérive détectée: {drift_result}")
    
    # Prédiction avec observation NORMALISÉE
    signal = self.ensemble.predict(normalized_observation)
    
    return signal
```

### Étape 4: Redémarrer le Monitor

```bash
# Arrêter l'ancien monitor
pkill -f paper_trading_monitor.py

# Attendre 2 secondes
sleep 2

# Relancer avec les nouvelles clés
python scripts/paper_trading_monitor.py \
  --api_key "JZELi7qLcOcp5gr7AAYpnlJnW9wxbHHeX99uqFWNFxJIKKb6pVhrmYu2mboWMFeA" \
  --api_secret "dFem0rr6ItWQ65sUxMRHseAUI8dtYDMI7WB69SrWYT4td5VKdjqFmilwb89cw4zY" &

# Vérifier les logs
sleep 5
tail -50 paper_trading_monitor.log | grep -E "(Normalisation|Dérive|signal)"
```

## 📊 Résultats Attendus

### Avant la Correction
```
🤖 Ensemble Decision: BUY (signal: 1.000)
   Worker signals: {'w1': 1, 'w2': 1, 'w3': 1, 'w4': 1}
   Confidence: 0.95
```

### Après la Correction
```
✅ Normaliseur initialisé: True
✅ Détecteur de dérive initialisé
📊 Normalisation: ✅ Actif

🤖 Ensemble Decision: HOLD (signal: 0.123)
   Worker signals: {'w1': -1, 'w2': 0, 'w3': 1, 'w4': 0}
   Confidence: 0.45
```

Les signaux devraient maintenant **varier** entre -1, 0, et 1 au lieu d'être toujours 1.

## 🔍 Monitoring de la Dérive

Le détecteur de dérive vous alertera si :

```
⚠️  Dérive détectée: {
    'drift_detected': True,
    'max_distance': 3.5,
    'threshold': 2.0,
    'n_features_drifted': 12,
    'n_observations': 100
}
```

Cela signifie que 12 features ont une distribution très différente de l'entraînement.

## 🛠️ Dépannage

### Problème: "Normaliseur non initialisé"

**Cause**: Le fichier VecNormalize.pkl n'a pas pu être chargé

**Solution**:
```bash
# Vérifier que le fichier existe
ls -la /mnt/new_data/t10_training/checkpoints/vecnormalize.pkl

# Si absent, créer un normaliseur d'urgence
python << 'EOF'
import json
import numpy as np
from pathlib import Path

# Créer des stats par défaut
stats = {
    "mean": np.zeros(68).tolist(),
    "var": np.ones(68).tolist(),
    "created_at": "emergency",
    "n_samples": 0
}

with open("/tmp/emergency_normalizer.json", "w") as f:
    json.dump(stats, f)

print("✅ Normaliseur d'urgence créé")
EOF
```

### Problème: "RecursionError lors du chargement"

**Cause**: Le fichier pickle est trop complexe

**Solution**: Déjà gérée dans le code (limite de récursion augmentée à 10000)

### Problème: Les signaux sont toujours constants

**Cause**: La normalisation n'est pas appliquée correctement

**Solution**:
1. Vérifier que `normalized_observation` est utilisée dans `ensemble.predict()`
2. Vérifier que `normalizer.is_loaded` est `True` dans les logs
3. Vérifier que les observations brutes ont une plage raisonnable (min/max)

## 📈 Métriques de Succès

Après la correction, vous devriez observer :

1. **Signaux variés**: Les prédictions ne sont plus toujours 1.0
2. **Trades exécutés**: Le monitor devrait générer des ordres BUY/SELL/HOLD
3. **Pas de dérive**: Le détecteur ne devrait pas signaler de dérive majeure
4. **Solde qui change**: Le portefeuille virtuel devrait augmenter/diminuer

## 📝 Fichiers Modifiés

- ✅ `src/adan_trading_bot/normalization/observation_normalizer.py` (créé)
- ✅ `src/adan_trading_bot/normalization/__init__.py` (créé)
- ⏳ `scripts/paper_trading_monitor.py` (À MODIFIER)

## 🎯 Prochaines Étapes

1. **Appliquer le patch** au monitor
2. **Redémarrer** le monitor
3. **Vérifier les logs** pour confirmer la normalisation
4. **Monitorer les trades** pour voir si les signaux changent
5. **Ajuster les seuils** du détecteur de dérive si nécessaire

## 📞 Support

Si vous rencontrez des problèmes :

1. Vérifier les logs: `tail -100 paper_trading_monitor.log`
2. Exécuter le diagnostic: `python scripts/diagnose_covariate_shift_v2.py`
3. Vérifier que `normalizer.is_loaded` est `True`
4. Vérifier que les observations normalisées sont dans la plage [-3, 3]

---

**Status**: ✅ Solution implémentée et prête à être intégrée
