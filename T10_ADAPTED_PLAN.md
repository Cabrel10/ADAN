# T10 : PLAN ADAPTÉ - CONTRAINTES SÉVÈRES

## ⚠️ SITUATION CRITIQUE

**Ressources Disponibles** :
- Espace disque : 3.7 GB (besoin ~50 GB)
- RAM disponible : 6.5 GB (besoin ~12 GB)

**Solution** : Mode ultra-séquentiel avec nettoyage agressif

## 🎯 STRATÉGIE ADAPTÉE

### Option 1 : Entraînement Ultra-Séquentiel (RECOMMANDÉ)
- Lancer 1 worker à la fois
- Attendre la fin complète avant le suivant
- Nettoyer les logs/checkpoints après chaque worker
- Durée : ~12-15h (4 workers × 3-4h)

### Option 2 : Réduire les Steps
- Au lieu de 250k steps par worker → 100k steps
- Durée : ~5-6h total
- Risque : Moins d'entraînement

### Option 3 : Nettoyer d'Abord
- Supprimer les anciens logs Optuna
- Supprimer les checkpoints inutiles
- Libérer de l'espace disque

## 📋 ACTIONS IMMÉDIATES

1. Nettoyer les anciens fichiers
2. Vérifier l'espace libéré
3. Lancer T10 avec mode ultra-séquentiel
4. Monitoring strict de l'espace disque

## ✅ CRITÈRES DE SUCCÈS ADAPTÉS

**Minimums** :
- Au moins 2/4 workers complétés
- Sharpe moyen ≥ 1.0
- Pas de crash OOM

**Optimaux** :
- Tous les 4 workers complétés
- Sharpe moyen ≥ 5.0
- RAM stable < 70%
