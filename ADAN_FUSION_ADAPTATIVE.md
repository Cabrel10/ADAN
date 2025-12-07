# 🎯 ADAN - Fusion Adaptative par Pallier de Capital

## Vue d'ensemble

Le modèle ADAN utilise une **fusion adaptative** des 4 workers (W1, W2, W3, W4) en fonction du pallier de capital. Chaque pallier a une stratégie optimale.

## 📊 Stratégie par Pallier

### 🟢 Micro (0-100$)
**Poids:** W1: 50%, W2: 20%, W3: 10%, W4: 20%

**Stratégie:** Conservative dominant
- Capital très faible = risque minimal
- W1 (Ultra-Conservative) prend les décisions principales
- Autres workers = diversification légère
- Objectif: Préserver le capital et croître lentement

### 🔵 Small (100-1000$)
**Poids:** W1: 20%, W2: 50%, W3: 10%, W4: 20%

**Stratégie:** Balanced dominant
- Capital petit = croissance stable
- W2 (Balanced) prend les décisions principales
- Équilibre risque/rendement
- Objectif: Croissance régulière vers Medium

### 🟡 Medium (1000-10k$)
**Poids:** W1: 15%, W2: 20%, W3: 15%, W4: 50%

**Stratégie:** Hybrid dominant
- Capital moyen = flexibilité
- W4 (Hybrid) prend les décisions principales
- Adapte risque selon conditions
- Objectif: Atteindre High tier

### 🔴 High (10k-100k$)
**Poids:** W1: 10%, W2: 15%, W3: 50%, W4: 25%

**Stratégie:** Aggressive dominant
- Capital important = rendement maximal
- W3 (Aggressive) prend les décisions principales
- Accepte plus de volatilité
- Objectif: Maximiser rendement

### ⭐ Enterprise (100k+$)
**Poids:** W1: 25%, W2: 25%, W3: 25%, W4: 25%

**Stratégie:** Ensemble équilibré
- Capital très important = diversification complète
- Tous les workers contribuent équalement
- Réduction du risque par diversification
- Objectif: Stabilité et rendement durable

## 🔧 Implémentation

### Utilisation en Production

```python
from adan_trading_bot.model.model_ensemble import ModelEnsemble

# Charger l'ensemble ADAN
ensemble = ModelEnsemble.load("checkpoints/final/ADAN_ensemble.pkl")

# Obtenir les poids pour un pallier
weights = ensemble.get_fusion_weights(capital_tier="Small")
# {'w1': 0.20, 'w2': 0.50, 'w3': 0.10, 'w4': 0.20}

# Faire une prédiction avec poids adaptatifs
prediction = ensemble.predict_with_tier(observation, capital_tier="Small")
```

### Adaptation Dynamique

Les poids sont aussi ajustés selon les **performances réelles** de chaque worker:

```
poids_final = poids_pallier × facteur_performance
```

Où `facteur_performance` est basé sur la précision historique du worker.

## 📈 Avantages

1. **Optimisation par étape:** Chaque pallier utilise la meilleure stratégie
2. **Gestion du risque:** Capital faible = risque minimal
3. **Croissance progressive:** Transition naturelle entre palliers
4. **Diversification:** Tous les workers contribuent
5. **Adaptabilité:** Ajustement selon performances réelles

## 🎯 Résultats Attendus

| Pallier | Stratégie | Sharpe Attendu | Drawdown Max |
|---------|-----------|----------------|--------------|
| Micro | Conservative | 3.4+ | <7% |
| Small | Balanced | 3.5+ | <16% |
| Medium | Hybrid | 3.2+ | <18% |
| High | Aggressive | 2.0+ | <22% |
| Enterprise | Équilibré | 3.0+ | <15% |

## 🚀 Lancement

```bash
# Entraîner les 4 workers et créer ADAN
python scripts/train_parallel_agents.py --config config/config.yaml

# Résultat: ADAN_ensemble.pkl avec fusion adaptative
```

## 📝 Notes

- Les poids sont normalisés (somme = 1.0)
- Chaque pallier a une stratégie cohérente
- Les performances réelles ajustent les poids
- Transition fluide entre palliers
