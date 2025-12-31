# Phase 2 - Instructions Finales pour Checkpoint 2.5

## 🎯 Objectif
Remplacer la normalisation manuelle par VecNormalize dans `build_observation()` pour éliminer le covariate shift (divergence 72.76%).

## 📍 Localisation Exacte du Problème

**Fichier**: `scripts/paper_trading_monitor.py`
**Lignes**: ~1040-1043

### Code Actuel (À REMPLACER)
```python
mean = window.mean(axis=0)
std = window.std(axis=0)
std = np.where(std == 0, 1, std)  # Avoid div by zero
window_normalized = (window - mean) / std
window_normalized = np.nan_to_num(window_normalized, 0)
window_normalized = np.clip(window_normalized, -10, 10)  # Clip outliers
```

## ✅ Solution Complète

### Étape 1: Modifier la Signature
```python
# AVANT:
def build_observation(self, df_5m, df_1h, df_4h):

# APRÈS:
def build_observation(self, worker_id: str, df_5m, df_1h, df_4h):
```

### Étape 2: Remplacer la Normalisation
```python
# REMPLACER le bloc de normalisation manuelle par:

# CORRECTION CRITIQUE: Utiliser VecNormalize au lieu de normalisation manuelle
# Référence diagnostic: divergence de 72.76% avant correction
try:
    env = self.worker_envs[worker_id]
    
    # Construire l'observation brute (Dict)
    obs_dict = {
        '5m': normalized_data_5m,      # Shape: (20, 15)
        '1h': normalized_data_1h,      # Shape: (10, 15)
        '4h': normalized_data_4h,      # Shape: (5, 15)
        'portfolio_state': portfolio_state  # Shape: (20,)
    }
    
    # VecNormalize attend un batch, donc ajouter dimension
    obs_batch = {k: np.expand_dims(v, axis=0) for k, v in obs_dict.items()}
    
    # Normaliser avec VecNormalize (statistiques d'entraînement figées)
    normalized_batch = env.normalize_obs(obs_batch)
    
    # Retirer la dimension de batch
    normalized_obs = {k: v[0] for k, v in normalized_batch.items()}
    
    logger.debug(f"✅ Observation normalisée via VecNormalize pour {worker_id}")
    return normalized_obs
    
except Exception as e:
    logger.error(f"❌ Erreur normalisation VecNormalize pour {worker_id}: {e}")
    raise
```

### Étape 3: Mettre à Jour les Appels

Chercher tous les appels à `build_observation()`:
```bash
grep -n "build_observation(" scripts/paper_trading_monitor.py
```

Pour chaque appel trouvé, ajouter le paramètre `worker_id`:
```python
# AVANT:
obs = self.build_observation(df_5m, df_1h, df_4h)

# APRÈS:
obs = self.build_observation("w1", df_5m, df_1h, df_4h)
# ou
obs = self.build_observation(worker_id, df_5m, df_1h, df_4h)
```

## 🧪 Validation

### Test 1: Vérifier la Signature
```bash
python scripts/test_checkpoint_2_5.py
```

### Test 2: Vérifier la Divergence
```bash
python scripts/validate_normalization_coherence.py
```

### Test 3: Compilation
```bash
python -m py_compile scripts/paper_trading_monitor.py
```

## 📊 Critères de Succès

- ✅ Compilation réussie (aucune erreur de syntaxe)
- ✅ Tous les appels à `build_observation()` ont `worker_id`
- ✅ Normalisation manuelle remplacée par VecNormalize
- ✅ Divergence post-correction < 0.001 (ou < 0.01 acceptable)
- ✅ Amélioration > 99% par rapport au diagnostic initial (64.19 → <0.001)

## 🚀 Prochaines Étapes

Une fois Checkpoint 2.5 validé:

1. **Checkpoint 2.6**: Valider divergence < 0.001
2. **Phase 3**: Tests d'inférence basiques
3. **Phase 4**: Entraînement MVP

## 📝 Notes Importantes

- Ne pas supprimer le code de normalisation manuelle, le remplacer
- Garder la gestion d'erreur (try/except)
- Garder les logs pour le debugging
- Compiler après chaque modification
- Tester avec `python -m py_compile` avant d'exécuter

## ⚠️ Attention

Si vous rencontrez une erreur `KeyError: 'w1'` lors de l'appel à `self.worker_envs[worker_id]`:
- Vérifier que `initialize_worker_environments()` a été appelée dans `__init__()`
- Vérifier que les fichiers `vecnormalize.pkl` existent
- Vérifier que `self.worker_ids` est défini

## 📞 Support

Si vous êtes bloqué:
1. Vérifier la compilation: `python -m py_compile scripts/paper_trading_monitor.py`
2. Vérifier les logs: `tail -f logs/paper_trading_monitor.log`
3. Consulter le diagnostic: `diagnostic/PHASE2_COMPLETION_SUMMARY.md`
