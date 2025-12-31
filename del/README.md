# Archive `del/` – Artefacts du Nettoyage Stratégique

**Date** : 24 décembre 2025  
**Taille** : 782 MB | **Fichiers** : 216

## 1. Objectif

Ce dossier centralise tous les fichiers et documents déplacés durant le nettoyage stratégique du projet ADAN. L'objectif est de libérer la racine du dépôt principal tout en conservant l'historique complet et les artefacts.

**Principe** : Rien n'a été supprimé, uniquement déplacé.

## 2. Contenu

### 2.1 Rapports d'analyse (100+ fichiers)

```
├── DIAGNOSTIC_*.md              # Diagnostics techniques
├── ANALYSIS_*.md                # Analyses détaillées
├── BREAKTHROUGH_*.md            # Découvertes majeures
├── CRITICAL_*.md                # Alertes critiques
├── FINAL_*.md                   # Rapports finaux
├── PHASE*_*.md                  # Rapports par phase
├── COVARIATE_SHIFT_*.md         # Analyses du covariate shift
├── DASHBOARD_*.md               # Rapports dashboard
├── WORKER_*.md                  # Analyses par worker
└── [autres rapports]
```

### 2.2 Logs d'exécution (50+ fichiers)

```
├── *.log                        # Logs d'exécution
├── monitoring_*.log             # Logs de monitoring
├── training_*.log               # Logs d'entraînement
├── paper_trading.log            # Logs paper trading
└── [autres logs]
```

### 2.3 Scripts de diagnostic (30+ fichiers)

```
├── diagnose_*.py                # Scripts de diagnostic
├── check_*.py                   # Scripts de vérification
├── analyze_*.py                 # Scripts d'analyse
├── monitor_*.py                 # Scripts de monitoring
├── test_*.py                    # Scripts de test ponctuels
└── [autres scripts]
```

### 2.4 Dossiers archivés

```
├── checkpoints/                 # Checkpoints intermédiaires
├── ARCHIVES_BEFORE_NAN_FIX/     # Archives avant corrections
├── investigation_results/       # Résultats d'investigations
├── logs_archive/                # Archive de logs
└── [autres dossiers]
```

### 2.5 Fichiers temporaires

```
├── *.autosave~                  # Fichiers autosave
├── test_*.db                    # Bases de données de test
├── dbe_state_*.pkl              # États DBE intermédiaires
├── *.tar.gz                     # Archives compressées
└── [autres fichiers]
```

### 2.6 Dossiers cachés

```
├── .hypothesis/                 # Cache Hypothesis
├── .kiro/                       # Configuration Kiro
└── [autres dossiers cachés]
```

## 3. Statistiques

| Catégorie | Nombre | Taille |
|-----------|--------|--------|
| Rapports | 100+ | 50 MB |
| Logs | 50+ | 100 MB |
| Scripts | 30+ | 5 MB |
| Dossiers | 10+ | 600 MB |
| Fichiers temporaires | 26+ | 27 MB |
| **Total** | **216** | **782 MB** |

## 4. Restauration

### 4.1 Restaurer un fichier spécifique

```bash
# Exemple : restaurer un rapport
mv del/DIAGNOSTIC_ADAN_CRITICAL_FINDINGS.md ./

# Exemple : restaurer un dossier
mv del/checkpoints/ ./

# Exemple : restaurer un log
mv del/paper_trading.log ./
```

### 4.2 Restaurer tout

```bash
# Restaurer tous les fichiers (attention : remplit la racine)
mv del/* ./
```

### 4.3 Rechercher un fichier

```bash
# Chercher par nom
find del/ -name "*diagnostic*"

# Chercher par type
find del/ -name "*.log"
find del/ -name "*.md"
find del/ -name "*.py"
```

## 5. Fichiers importants à connaître

### 5.1 Diagnostics critiques

- `DIAGNOSTIC_ADAN_CRITICAL_FINDINGS.md` : Analyse complète des problèmes
- `COVARIATE_SHIFT_FIX_GUIDE.md` : Guide de correction du covariate shift
- `ROOT_CAUSE_ANALYSIS.md` : Analyse des causes racines

### 5.2 Rapports de performance

- `FINAL_PERFORMANCE_REPORT.md` : Rapport de performance final
- `WORKER_PERFORMANCE_ANALYSIS.md` : Analyse par worker
- `TENSORBOARD_ANALYSIS_REPORT.md` : Analyse TensorBoard

### 5.3 Logs critiques

- `paper_trading.log` : Logs du paper trading
- `training_log.txt` : Logs d'entraînement
- `monitoring_console.log` : Logs de monitoring

## 6. Quand restaurer

### 6.1 Restaurer si...

- Vous avez besoin de l'historique d'une analyse
- Vous voulez rejouer un diagnostic
- Vous cherchez des données de performance historiques
- Vous avez besoin de scripts de monitoring

### 6.2 Ne pas restaurer si...

- Vous voulez garder la racine propre
- Les fichiers sont obsolètes (vérifier la date)
- Vous avez une version plus récente ailleurs

## 7. Maintenance

### 7.1 Nettoyer l'archive

```bash
# Supprimer les fichiers obsolètes (attention : irréversible)
rm del/OBSOLETE_FILE.md

# Compresser l'archive
tar -czf del_archive_backup.tar.gz del/
```

### 7.2 Documenter les restaurations

Si vous restaurez un fichier, documentez pourquoi :

```bash
# Exemple
mv del/DIAGNOSTIC_ADAN_CRITICAL_FINDINGS.md ./
echo "Restauré pour analyse du covariate shift" >> RESTORATION_LOG.txt
```

## 8. Références

- **README.md** : Documentation principale du projet
- **README_MODIFICATIONS.md** : Synthèse du nettoyage
- **README_CORRECTIONS.md** : Correctifs critiques à implémenter

---

**Archive créée** : 24 décembre 2025  
**Taille totale** : 782 MB  
**Fichiers** : 216  
**Statut** : Archivé (rien n'a été supprimé)
