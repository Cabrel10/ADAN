# 🚨 ALERTE CRITIQUE - DIVERGENCE PORTFOLIO DIMENSIONS

**Date**: 2 Janvier 2026  
**Gravité**: 🔴 **CRITIQUE - BLOCAGE IMMÉDIAT**

---

## ⚠️ DÉCOUVERTE MAJEURE

**Les modèles entraînés attendent `portfolio_state: (20,)` mais nous avons patché pour 17 !**

### Observation Space des Modèles

```
w1, w2, w3, w4:
  '5m': Box(-inf, inf, (20, 14), float32)      ← 20 fenêtres × 14 features
  '1h': Box(-inf, inf, (20, 14), float32)      ← 20 fenêtres × 14 features
  '4h': Box(-inf, inf, (20, 14), float32)      ← 20 fenêtres × 14 features
  'portfolio_state': Box(-inf, inf, (20,), float32)  ← 20 dimensions ⚠️
```

### Configuration Live (Après Patch)

```
StateBuilder:
  5m: 19 fenêtres × 15 features = 285
  1h: 10 fenêtres × 16 features = 160
  4h:  5 fenêtres × 16 features =  80
  Portfolio: 17 dimensions ← DIVERGENCE!
  ─────────────────────────────────────
  Total: 525 + 17 = 542
```

---

## 🔴 PROBLÈME

**Les modèles reçoivent**:
- `portfolio_state` de dimension 20 (entraînement)

**Mais nous envoyons**:
- `portfolio_state` de dimension 17 (live)

**Résultat**: Les modèles reçoivent un vecteur de taille incorrecte → **CRASH GARANTI**

---

## 🎯 SOLUTION IMMÉDIATE

### Option 1: Corriger StateBuilder pour 20 dimensions (RECOMMANDÉ)

Modifier `src/adan_trading_bot/data_processing/state_builder.py` pour que `portfolio_state` retourne 20 dimensions au lieu de 17.

### Option 2: Réentraîner les modèles avec 17 dimensions

Réentraîner w1, w2, w3, w4 avec la nouvelle configuration (17 dimensions portfolio).

### Option 3: Patcher les modèles (IMPOSSIBLE)

Les modèles sont des fichiers binaires - impossible de les modifier directement.

---

## 📊 ANALYSE DÉTAILLÉE

### Observation Space Entraînement

```python
# Ce que les modèles attendent:
observation_space = Dict({
    '5m': Box(shape=(20, 14)),      # 20 fenêtres × 14 features
    '1h': Box(shape=(20, 14)),      # 20 fenêtres × 14 features
    '4h': Box(shape=(20, 14)),      # 20 fenêtres × 14 features
    'portfolio_state': Box(shape=(20,))  # 20 dimensions
})
```

### Observation Space Live (Après Patch)

```python
# Ce que nous envoyons:
observation_space = Dict({
    '5m': Box(shape=(19, 15)),      # 19 fenêtres × 15 features
    '1h': Box(shape=(10, 16)),      # 10 fenêtres × 16 features
    '4h': Box(shape=(5, 16)),       # 5 fenêtres × 16 features
    'portfolio_state': Box(shape=(17,))  # 17 dimensions ← DIVERGENCE!
})
```

---

## 🚨 IMPACT

**Scénario de déploiement**:

```
1. Bot démarre
2. Récupère observation (5m, 1h, 4h, portfolio_state)
3. Envoie à w1.predict(observation)
4. w1 attend portfolio_state de dimension 20
5. Reçoit portfolio_state de dimension 17
6. ❌ CRASH: Shape mismatch error
```

---

## 🔧 CORRECTION REQUISE

### Étape 1: Déterminer les 20 dimensions du portfolio

Chercher dans le code d'entraînement quelles sont les 20 dimensions du portfolio_state.

Fichiers à vérifier:
- `src/adan_trading_bot/environment/multi_asset_chunked_env.py` (ligne ~590)
- `src/adan_trading_bot/agent/trading_agent.py`
- `src/adan_trading_bot/portfolio/portfolio_manager.py`

### Étape 2: Modifier StateBuilder

Modifier `src/adan_trading_bot/data_processing/state_builder.py` pour retourner 20 dimensions au lieu de 17.

### Étape 3: Valider les dimensions

Exécuter `verify_dims.py` pour confirmer que le total est maintenant:
- 5m: 20 × 15 = 300
- 1h: 10 × 16 = 160
- 4h: 5 × 16 = 80
- Portfolio: 20
- **Total: 560** (au lieu de 542)

---

## 📋 CHECKLIST

- [ ] Identifier les 20 dimensions du portfolio_state
- [ ] Modifier StateBuilder pour retourner 20 dimensions
- [ ] Valider que les dimensions matchent exactement
- [ ] Relancer l'audit
- [ ] Confirmer que tous les ✅ sont présents
- [ ] Déployer

---

## ⚠️ RECOMMANDATION

**NE PAS DÉPLOYER** tant que cette divergence n'est pas résolue.

Les modèles crasheront immédiatement avec une erreur de shape mismatch.

---

**Status**: 🔴 **BLOQUANT - CORRECTION REQUISE IMMÉDIATEMENT**
