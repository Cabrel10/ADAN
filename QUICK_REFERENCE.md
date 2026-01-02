# Référence Rapide - ADAN Trading Bot

## 🚀 Commandes Essentielles

### Vérification et Test

```bash
# Vérifier les fichiers critiques
python3 check_deployment.py

# Test complet du déploiement local
bash test_local_deployment.sh

# Test sans accès à /mnt/new_data
bash test_without_external_disk.sh
```

### Démarrage et Arrêt

```bash
# Démarrer le bot
./start.sh

# Arrêter le bot
pkill -f paper_trading_monitor

# Redémarrer le bot
pkill -f paper_trading_monitor && sleep 2 && ./start.sh
```

### Logs

```bash
# Voir les logs en temps réel
tail -f logs/bot.log

# Voir les 100 dernières lignes
tail -100 logs/bot.log

# Chercher une erreur
grep ERROR logs/bot.log

# Chercher un worker spécifique
grep w1 logs/bot.log
```

### Monitoring

```bash
# Vérifier que le bot tourne
ps aux | grep paper_trading_monitor

# Utilisation CPU/Mémoire
top -p $(pgrep -f paper_trading_monitor)

# Espace disque
df -h

# Connexion réseau
netstat -an | grep ESTABLISHED
```

---

## 📦 Structure des Fichiers

```
.
├── models/                          # Modèles et normalisateurs
│   ├── w1/
│   │   ├── w1_model_final.zip      # Modèle w1
│   │   └── vecnormalize.pkl        # Normalisateur w1
│   ├── w2/
│   ├── w3/
│   ├── w4/
│   └── ensemble/
│       └── adan_ensemble_config.json # Configuration ADAN
├── config/
│   └── config.yaml                  # Configuration principale
├── scripts/
│   └── paper_trading_monitor.py     # Bot principal
├── logs/                            # Répertoire des logs
├── check_deployment.py              # Vérification pré-déploiement
├── start.sh                         # Script de démarrage
├── .env.example                     # Template de configuration
├── DEPLOYMENT_GUIDE.md              # Guide complet
├── MIGRATION_SUMMARY.md             # Résumé de la migration
└── SERVER_DEPLOYMENT_CHECKLIST.md   # Checklist serveur
```

---

## 🔧 Configuration

### Variables d'Environnement (.env)

```bash
# Clés API
API_KEY=your_api_key
SECRET_KEY=your_secret_key

# Telegram (optionnel)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Configuration
LOG_LEVEL=INFO
TRADING_MODE=paper
MONITOR_INTERVAL=60
```

### Configuration YAML (config/config.yaml)

```yaml
paths:
  trained_models_dir: ./models
  data_dir: ./data
  logs_dir: ./logs

trading:
  mode: paper
  exchange: binance
  
workers:
  - w1
  - w2
  - w3
  - w4
```

---

## 📊 Poids ADAN Ensemble

```
w1: 24.9%
w2: 25.0%
w3: 25.1%
w4: 25.0%
```

---

## 🔍 Vérification Rapide

### Avant de Démarrer

```bash
# 1. Vérifier les fichiers
python3 check_deployment.py

# 2. Vérifier l'espace disque
df -h

# 3. Vérifier la configuration
cat .env | grep -v "^#"

# 4. Vérifier les modèles
ls -lh models/*/
```

### Après le Démarrage

```bash
# 1. Vérifier que le bot tourne
ps aux | grep paper_trading_monitor

# 2. Vérifier les logs
tail -20 logs/bot.log

# 3. Chercher les erreurs
grep ERROR logs/bot.log

# 4. Vérifier le chargement des modèles
grep "loaded" logs/bot.log
```

---

## 🚨 Troubleshooting Rapide

### Le bot ne démarre pas

```bash
# Vérifier les fichiers
python3 check_deployment.py

# Vérifier les logs
tail -50 logs/bot.log

# Vérifier les permissions
ls -la models/
```

### Erreur "modèle non trouvé"

```bash
# Vérifier que les fichiers existent
ls -lh models/w1/w1_model_final.zip

# Vérifier l'intégrité du ZIP
unzip -t models/w1/w1_model_final.zip
```

### Erreur "pickle corrompu"

```bash
# Vérifier le fichier pickle
python3 -c "import pickle; pickle.load(open('models/w1/vecnormalize.pkl', 'rb'))"
```

### Erreur API

```bash
# Vérifier les clés API
grep API_KEY .env

# Tester la connectivité
ping api.binance.com

# Vérifier les logs d'erreur
grep -i "api\|error" logs/bot.log
```

---

## 📈 Déploiement Serveur

### Préparation

```bash
# 1. Créer l'archive
tar -czf adan_bot_deploy.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='logs/*' \
  --exclude='.env' \
  .

# 2. Transférer
scp adan_bot_deploy.tar.gz user@serveur:/home/user/

# 3. Extraire
ssh user@serveur "cd /home/user && tar -xzf adan_bot_deploy.tar.gz"
```

### Installation

```bash
# 1. Créer l'environnement virtuel
python3 -m venv venv

# 2. Installer les dépendances
source venv/bin/activate
pip install -r requirements.txt

# 3. Configurer
cp .env.example .env
nano .env  # Ajouter les clés API

# 4. Vérifier
python3 check_deployment.py

# 5. Démarrer
./start.sh
```

### Service Systemd

```bash
# Créer le service
sudo nano /etc/systemd/system/adan-bot.service

# Recharger
sudo systemctl daemon-reload

# Activer
sudo systemctl enable adan-bot

# Démarrer
sudo systemctl start adan-bot

# Vérifier
sudo systemctl status adan-bot

# Logs
sudo journalctl -u adan-bot -f
```

---

## 📞 Fichiers de Référence

| Fichier | Objectif |
|---------|----------|
| `DEPLOYMENT_GUIDE.md` | Guide complet de déploiement |
| `MIGRATION_SUMMARY.md` | Résumé de la migration |
| `SERVER_DEPLOYMENT_CHECKLIST.md` | Checklist serveur |
| `check_deployment.py` | Vérification des fichiers |
| `start.sh` | Script de démarrage |
| `.env.example` | Template de configuration |

---

## ⏱️ Chronologie Typique

```
Jour 1: Migration des modèles (✅ FAIT)
Jour 2: Test local complet
Jour 3: Préparation serveur
Jour 4: Déploiement serveur
Jour 5: Tests post-déploiement
Jour 6-7: Monitoring et ajustements
```

---

## 🎯 Objectifs

- ✅ Modèles migrés localement
- ✅ Scripts de déploiement créés
- ⏳ Test local sans `/mnt/new_data`
- ⏳ Déploiement serveur
- ⏳ Monitoring 24/7
- ⏳ Rapports de performance

---

**Dernière mise à jour:** 2 janvier 2026
**Version:** 1.0
