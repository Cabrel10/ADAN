# INVESTIGATION CRITIQUE: Ensemble Action Saturation

## 🚨 PROBLÈME IDENTIFIÉ

Tous les workers retournent **exactement 1.0000** (BUY avec confiance 1.0):
```
w1: raw=1.0000 → BUY, conf=1.000
w2: raw=1.0000 → BUY, conf=1.000
w3: raw=1.0000 → BUY, conf=1.000
w4: raw=1.0000 → BUY, conf=1.000
```

## 🔍 CAUSES POTENTIELLES

### 1. **Saturation de l'action PPO**
- Les modèles PPO retournent des actions continues dans [-1, 1]
- Si TOUS les modèles retournent exactement 1.0, c'est une **saturation**
- Cela indique que les modèles sont **trop confiants** ou **overfittés**

### 2. **Configuration d'inférence**
```yaml
agent:
  deterministic_inference: false  # ← PROBLÈME!
```
- `deterministic_inference: false` = utilise stochastique (ajoute du bruit)
- Mais les valeurs sont EXACTEMENT 1.0 → pas de bruit appliqué
- Devrait être `true` pour inférence déterministe

### 3. **Action Scaling**
```yaml
action:
  action_scale: 1.0
  min_action_threshold: 0.01
```
- `action_scale: 1.0` = pas de scaling
- `min_action_threshold: 0.01` = très bas

### 4. **Mapping d'action**
```python
# Dans get_ensemble_action():
action_value = float(action[0])  # Prend seulement le premier élément!

# Thresholds:
if action_value < -0.33:
    discrete_action = 2  # SELL
elif action_value > 0.33:
    discrete_action = 1  # BUY
else:
    discrete_action = 0  # HOLD
```

**PROBLÈME**: Si `action[0]` est toujours 1.0, alors TOUJOURS `> 0.33` → TOUJOURS BUY

## 📊 DIAGNOSTIC DÉTAILLÉ

### Hypothèse 1: Modèles Overfittés
- Les modèles ont appris que "BUY" est toujours la meilleure action
- Pendant l'entraînement, le marché était en tendance haussière
- Les modèles ont saturé à action=1.0

### Hypothèse 2: Problème de Normalisation
- L'observation n'est pas correctement normalisée
- Les modèles reçoivent des valeurs extrêmes
- Ils répondent avec saturation (±1.0)

### Hypothèse 3: Problème d'Architecture
- La couche de sortie n'a pas de contrainte
- Les poids de la dernière couche sont trop grands
- Résultat: saturation à ±1.0

### Hypothèse 4: Problème de Données
- Les données d'entraînement étaient biaisées vers BUY
- Les modèles ont appris ce biais
- Pendant l'inférence, ils reproduisent ce biais

## 🔧 SOLUTIONS À TESTER

### Solution 1: Activer Deterministic Inference
```yaml
agent:
  deterministic_inference: true  # Utiliser mode déterministe
```

### Solution 2: Ajouter Exploration Noise
```yaml
agent:
  exploration:
    noise_std: 0.1  # Ajouter du bruit gaussien
    noise_sigma: 0.2
```

### Solution 3: Vérifier la Sortie Brute
Avant le mapping, logger la sortie brute:
```python
logger.info(f"  {wid}: raw_action={action}, action[0]={action_value}")
```

### Solution 4: Vérifier la Normalisation
```python
logger.info(f"  Observation stats: min={obs.min()}, max={obs.max()}, mean={obs.mean()}")
```

### Solution 5: Appliquer Clipping
```python
# Ajouter du clipping pour éviter la saturation
action_value = np.clip(action_value, -0.95, 0.95)
```

## 📋 CHECKLIST D'INVESTIGATION

- [ ] Vérifier la sortie brute des modèles PPO
- [ ] Vérifier la distribution des actions pendant l'entraînement
- [ ] Vérifier la normalisation de l'observation
- [ ] Vérifier les poids de la dernière couche
- [ ] Tester avec `deterministic_inference: true`
- [ ] Tester avec exploration noise
- [ ] Comparer avec les modèles d'entraînement

## 🎯 PROCHAINES ÉTAPES

1. **Immédiat**: Activer `deterministic_inference: true`
2. **Court terme**: Ajouter logging détaillé de la sortie brute
3. **Moyen terme**: Réentraîner les modèles avec meilleure régularisation
4. **Long terme**: Implémenter ensemble diversity check

## ⚠️ RISQUES

- **Saturation**: Tous les signaux sont BUY → pas de diversité
- **Overfitting**: Modèles trop spécialisés pour les données d'entraînement
- **Perte de granularité**: Pas de nuances dans les décisions
- **Faux positifs**: Confiance 100% dans des décisions biaisées

## 📝 NOTES

- ADAN est une **fusion pondérée** de 4 modèles, pas 4 portefeuilles
- Les poids sont: w1=0.25, w2=0.27, w3=0.30, w4=0.18
- Si tous les modèles retournent 1.0, la fusion retourne aussi 1.0
- Le problème est dans les modèles individuels, pas dans la fusion
