# Guide du Dynamic Behavior Engine (DBE) et ReplayLogger

## Vue d'ensemble

Le **Dynamic Behavior Engine (DBE)** est un composant clé du système de trading ADAN qui ajuste dynamiquement les paramètres de trading en fonction des conditions du marché et des performances du portefeuille. Le **ReplayLogger** est un outil de journalisation qui enregistre toutes les décisions prises par le DBE pour analyse ultérieure.

## Fonctionnalités du DBE

### 1. Ajustement dynamique des paramètres
- **Stop-Loss (SL) et Take-Profit (TP)**: Ajustement en fonction de la volatilité et des performances récentes
- **Taille de position**: Modulation en fonction du risque et de la confiance du modèle
- **Récompenses**: Ajustement des récompenses pour influencer le comportement de l'agent

### 2. Lissage des paramètres
- Utilisation d'un facteur de lissage (par défaut 0.7) pour éviter les changements brusques
- Adaptation dynamique à la volatilité du marché

### 3. Gestion du risque
- Protection contre les pertes excessives
- Ajustement automatique des paramètres en fonction du drawdown

## Utilisation du ReplayLogger

Le ReplayLogger enregistre toutes les décisions du DBE dans des fichiers JSONL pour analyse ultérieure.

### Configuration

Le ReplayLogger est configuré dans le fichier `config/logging_config.yaml` :

```yaml
log_dirs:
  base: "logs"
  dbe_replay: "logs/dbe_replay"  # Répertoire pour les logs du ReplayLogger
```

### Format des logs

Chaque entrée de log contient :
- `timestamp`: Horodatage de la décision
- `type`: Type de log (décision, erreur, etc.)
- `step_index`: Numéro de l'étape
- `modulation`: Paramètres de modulation calculés par le DBE
- `context`: Contexte de la décision (valeur du portefeuille, drawdown, etc.)

## Analyse des logs

Utilisez le script `analyze_dbe_logs.py` pour analyser les décisions du DBE :

```bash
# Afficher un résumé des décisions
python scripts/analyze_dbe_logs.py logs/dbe_replay/dbe_*.jsonl

# Générer des graphiques
python scripts/analyze_dbe_logs.py logs/dbe_replay/dbe_*.jsonl --plot --output dbe_analysis.png
```

## Test d'intégration

Pour tester l'intégration du DBE avec le ReplayLogger :

```bash
python scripts/test_dbe_integration.py
```

## Bonnes pratiques

1. **Surveillance** : Vérifiez régulièrement les logs pour détecter tout comportement inattendu
2. **Rétro-analyse** : Utilisez les logs pour comprendre les performances passées
3. **Ajustement** : Modifiez les paramètres du DBE dans `config/environment_config.yaml` selon les besoins

## Dépannage

### Problèmes courants

1. **Aucun fichier de log généré**
   - Vérifiez les permissions d'écriture dans le répertoire de logs
   - Assurez-vous que le DBE est correctement initialisé

2. **Données manquantes dans les logs**
   - Vérifiez que tous les champs obligatoires sont fournis lors de l'appel à `log_decision()`
   - Assurez-vous que le format des données est correct

3. **Performances**
   - Pour les backtests longs, envisagez de réduire la fréquence de journalisation
   - Utilisez des chemins de stockage rapides pour les fichiers de logs

## Contributeurs

- [Votre nom]
- [Autres contributeurs]

---

*Dernière mise à jour : 11 juillet 2024*
