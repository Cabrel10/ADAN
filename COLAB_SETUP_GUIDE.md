# 🚀 Guide d'Installation ADAN Trading Bot sur Google Colab

## 📋 Prérequis

1. Un compte Google Drive avec au moins 500 MB d'espace libre
2. Un compte Google Colab (gratuit)
3. Le fichier `bot_fixed_v2.tar.gz` (246 MB) téléchargé sur votre Drive

## 🔧 Étape 1 : Télécharger le projet sur Google Drive

```bash
# Sur votre machine locale
# Le fichier bot_fixed_v2.tar.gz se trouve à:
# /home/morningstar/Documents/trading/bot_fixed_v2.tar.gz

# Téléversez-le dans votre Google Drive (dossier racine ou MyDrive)
```

## ⚡ Étape 2 : Lancer Colab et exécuter le setup

1. Allez sur [Google Colab](https://colab.research.google.com)
2. Créez un nouveau notebook
3. Dans la première cellule, collez ce code :

```python
# Montage de Drive et extraction
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content')

# Extraction du projet
!tar -xzf drive/MyDrive/bot_fixed_v2.tar.gz
os.chdir('/content/bot')

# Installation complète
!bash scripts/setup_colab.sh
```

4. Exécutez la cellule (Runtime → Run cell)
5. Attendez ~5-10 minutes (selon votre connexion)

## 🎯 Étape 3 : Lancer l'entraînement

Une fois le setup terminé, dans une nouvelle cellule :

```python
import os
os.chdir('/content/bot')

# Lancer l'entraînement
!python scripts/train_parallel_agents.py --config-path config/config.yaml --checkpoint-dir checkpoints --resume
```

## 📊 Résultats

Les logs et checkpoints seront sauvegardés dans :
- `/content/bot/logs/` - Fichiers de logs
- `/content/bot/checkpoints/` - Modèles sauvegardés

## ⚠️ Troubleshooting

### Erreur : "No module named 'adan_trading_bot'"
→ Relancez le setup : `!bash scripts/setup_colab.sh`

### Erreur : "EOFError" pendant l'entraînement
→ C'est normal sur Colab. Le script utilise 1 seul worker pour éviter cela.

### Erreur : "Out of memory"
→ Réduisez `batch_size` dans `config/config_colab.yaml`

## 💾 Sauvegarder les résultats

Après l'entraînement, téléchargez les résultats :

```python
# Télécharger les logs
!zip -r /content/drive/MyDrive/adan_logs.zip /content/bot/logs/

# Télécharger les checkpoints
!zip -r /content/drive/MyDrive/adan_checkpoints.zip /content/bot/checkpoints/
```

## 🔄 Prochaines sessions

Pour les sessions suivantes :
1. Ouvrez un nouveau notebook Colab
2. Exécutez le code de l'Étape 2 et 3
3. Les checkpoints seront automatiquement chargés (mode `--resume`)

## 📞 Support

Si vous rencontrez des problèmes :
1. Vérifiez que `bot_fixed_v2.tar.gz` est dans votre Drive
2. Assurez-vous que vous avez assez d'espace disque
3. Redémarrez le runtime Colab (Runtime → Restart runtime)
