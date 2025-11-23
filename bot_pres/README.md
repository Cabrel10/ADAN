# ADAN Bot - Production Snapshot
**Date de Sauvegarde**: 2025-11-23 16:58 UTC

## 📊 Performance Validée

### Entraînement
- **Asset**: BTCUSDT
- **Steps**: 640,000
- **Durée**: Formation complète avec 4 workers (w1-w4)

### Évaluation (Cross-Asset)
- **Asset**: XRPUSDT (généralisation testée)
- **Capital Initial**: $20.50 USDT
- **Capital Final**: $70.45 USDT
- **Rendement Total**: +243.64%
- **Win Rate**: 52.31%
- **Profit Factor**: 1.06
- **Max Drawdown**: -16.72%
- **Trades**: 260

## 📁 Contenu

```
bot_pres/
├── model/
│   └── adan_model_checkpoint_640000_steps.zip  # Modèle PPO complet
├── config/
│   ├── config_snapshot.yaml                     # Configuration optimisée
│   └── optuna_snapshot.db                       # Historique d'optimisation
└── README.md                                    # Ce fichier
```

## 🎯 Objectif

Ce snapshot préserve le modèle avant la phase de **backtest approfondi**.

### Caractéristiques Clés
- ✅ Généralisation cross-asset validée (BTC → XRP)
- ✅ Paramètres de risque optimisés via Optuna
- ✅ Capital tier management fonctionnel
- ✅ Dashboard de monitoring opérationnel

## 🚀 Utilisation

### Charger le Modèle
```python
from stable_baselines3 import PPO
model = PPO.load("bot_pres/model/adan_model_checkpoint_640000_steps.zip")
```

### Restaurer la Configuration
```bash
cp bot_pres/config/config_snapshot.yaml config/config.yaml
```

## 📝 Notes

- Ce modèle a été entraîné avec un paradigme **single-model multi-environment**
- Les 4 workers (w1-w4) ont contribué à l'expérience, mais partagent un cerveau unique
- Performance robuste démontrée sur actif non-vu (XRPUSDT)

---
**Status**: ✅ Prêt pour Backtest
**Next Phase**: Validation sur données historiques étendues
