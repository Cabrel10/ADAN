# Checklist de Déploiement Serveur - ADAN Trading Bot

## 📋 Avant le Déploiement

### Préparation Locale

- [ ] Tous les tests locaux réussis
  ```bash
  python3 check_deployment.py
  bash test_local_deployment.sh
  ```

- [ ] Bot testé sans `/mnt/new_data`
  ```bash
  bash test_without_external_disk.sh
  ```

- [ ] Logs générés correctement
  ```bash
  tail -f logs/bot.log
  ```

- [ ] Espace disque suffisant (> 1 GB)
  ```bash
  df -h
  ```

- [ ] Archive de déploiement créée
  ```bash
  tar -czf adan_bot_deploy.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='logs/*' --exclude='.env' .
  ```

### Préparation Serveur

- [ ] Serveur Linux 24/7 accessible
- [ ] Python 3.9+ installé
  ```bash
  python3 --version
  ```

- [ ] pip/venv disponible
  ```bash
  python3 -m venv --help
  ```

- [ ] Utilisateur dédié créé
  ```bash
  sudo useradd -m -s /bin/bash trading_user
  ```

- [ ] Répertoire de déploiement préparé
  ```bash
  sudo mkdir -p /home/trading_user/adan_bot
  sudo chown trading_user:trading_user /home/trading_user/adan_bot
  ```

- [ ] SSH configuré (clés publiques)
- [ ] Firewall configuré (ports nécessaires)
- [ ] Connexion réseau stable testée

---

## 🚀 Déploiement

### Transfert des Fichiers

- [ ] Archive transférée au serveur
  ```bash
  scp adan_bot_deploy.tar.gz user@serveur:/home/trading_user/
  ```

- [ ] Intégrité de l'archive vérifiée
  ```bash
  ssh user@serveur "tar -tzf adan_bot_deploy.tar.gz | head -20"
  ```

### Installation

- [ ] Archive extraite
  ```bash
  ssh user@serveur "cd /home/trading_user && tar -xzf adan_bot_deploy.tar.gz"
  ```

- [ ] Environnement virtuel créé
  ```bash
  ssh user@serveur "cd /home/trading_user/adan_bot && python3 -m venv venv"
  ```

- [ ] Dépendances installées
  ```bash
  ssh user@serveur "cd /home/trading_user/adan_bot && source venv/bin/activate && pip install -r requirements.txt"
  ```

- [ ] Permissions configurées
  ```bash
  ssh user@serveur "chmod +x /home/trading_user/adan_bot/start.sh /home/trading_user/adan_bot/check_deployment.py"
  ```

### Configuration

- [ ] Fichier `.env` créé avec les vraies clés
  ```bash
  ssh user@serveur "cp /home/trading_user/adan_bot/.env.example /home/trading_user/adan_bot/.env"
  # Éditer le fichier avec les clés API réelles
  ```

- [ ] Vérification des fichiers critiques
  ```bash
  ssh user@serveur "cd /home/trading_user/adan_bot && python3 check_deployment.py"
  ```

- [ ] Répertoires de logs créés
  ```bash
  ssh user@serveur "mkdir -p /home/trading_user/adan_bot/logs"
  ```

---

## 🔧 Configuration Systemd

### Créer le Service

- [ ] Fichier service créé
  ```bash
  sudo nano /etc/systemd/system/adan-bot.service
  ```

  Contenu :
  ```ini
  [Unit]
  Description=ADAN Trading Bot
  After=network.target

  [Service]
  Type=simple
  User=trading_user
  WorkingDirectory=/home/trading_user/adan_bot
  ExecStart=/home/trading_user/adan_bot/start.sh
  Restart=always
  RestartSec=10
  StandardOutput=journal
  StandardError=journal

  [Install]
  WantedBy=multi-user.target
  ```

- [ ] Service rechargé
  ```bash
  sudo systemctl daemon-reload
  ```

- [ ] Service activé au démarrage
  ```bash
  sudo systemctl enable adan-bot
  ```

- [ ] Service démarré
  ```bash
  sudo systemctl start adan-bot
  ```

- [ ] Statut vérifié
  ```bash
  sudo systemctl status adan-bot
  ```

---

## 📊 Monitoring et Logs

### Configuration des Logs

- [ ] Logrotate configuré
  ```bash
  sudo nano /etc/logrotate.d/adan-bot
  ```

  Contenu :
  ```
  /home/trading_user/adan_bot/logs/*.log {
      daily
      rotate 7
      compress
      missingok
      notifempty
      create 0640 trading_user trading_user
  }
  ```

- [ ] Logs testés
  ```bash
  sudo journalctl -u adan-bot -f
  ```

### Monitoring

- [ ] Uptime du bot vérifié
  ```bash
  ps aux | grep paper_trading_monitor
  ```

- [ ] Utilisation CPU/Mémoire vérifiée
  ```bash
  top -p $(pgrep -f paper_trading_monitor)
  ```

- [ ] Espace disque vérifié
  ```bash
  df -h
  ```

- [ ] Connexion réseau vérifiée
  ```bash
  netstat -an | grep ESTABLISHED
  ```

---

## 🔔 Alertes et Notifications

### Configuration Telegram (Optionnel)

- [ ] Token Telegram obtenu
- [ ] Chat ID Telegram obtenu
- [ ] Variables d'environnement configurées dans `.env`
  ```
  TELEGRAM_BOT_TOKEN=xxx
  TELEGRAM_CHAT_ID=xxx
  ```

- [ ] Test d'alerte effectué
  ```bash
  # Vérifier dans les logs que les alertes s'envoient
  sudo journalctl -u adan-bot -f
  ```

### Alertes Critiques

- [ ] Erreur de chargement des modèles
- [ ] Erreur de connexion API
- [ ] Redémarrage du bot
- [ ] Rapport de performance quotidien

---

## ✅ Tests Post-Déploiement

### Test de Démarrage

- [ ] Bot démarre correctement
  ```bash
  sudo systemctl restart adan-bot
  sleep 5
  sudo systemctl status adan-bot
  ```

- [ ] Modèles se chargent
  ```bash
  sudo journalctl -u adan-bot -n 50
  ```

- [ ] Aucune erreur critique
  ```bash
  sudo journalctl -u adan-bot | grep ERROR
  ```

### Test de Stabilité

- [ ] Bot tourne pendant 1 heure sans erreur
- [ ] Logs générés correctement
- [ ] Espace disque stable
- [ ] CPU/Mémoire stables

### Test de Redémarrage

- [ ] Arrêter le bot
  ```bash
  sudo systemctl stop adan-bot
  ```

- [ ] Vérifier qu'il est arrêté
  ```bash
  ps aux | grep paper_trading_monitor
  ```

- [ ] Redémarrer le bot
  ```bash
  sudo systemctl start adan-bot
  ```

- [ ] Vérifier qu'il redémarre correctement
  ```bash
  sudo systemctl status adan-bot
  ```

### Test de Récupération

- [ ] Tuer le processus
  ```bash
  pkill -f paper_trading_monitor
  ```

- [ ] Vérifier que systemd le redémarre
  ```bash
  sleep 15
  ps aux | grep paper_trading_monitor
  ```

---

## 🛡️ Sécurité

### Permissions

- [ ] Fichiers `.env` protégés
  ```bash
  chmod 600 /home/trading_user/adan_bot/.env
  ```

- [ ] Clés API non en clair dans les logs
- [ ] Accès SSH limité aux administrateurs
- [ ] Firewall configuré (ports minimaux)

### Backup

- [ ] Configuration sauvegardée
  ```bash
  tar -czf adan_bot_config_backup.tar.gz /home/trading_user/adan_bot/config /home/trading_user/adan_bot/.env
  ```

- [ ] Logs archivés régulièrement
- [ ] Plan de rollback documenté

---

## 📈 Performance

### Métriques à Suivre

- [ ] Uptime du bot (objectif: 99.9%)
- [ ] Latence API (< 1 seconde)
- [ ] Nombre de trades par jour
- [ ] Profit/Perte quotidien
- [ ] Utilisation CPU (< 50%)
- [ ] Utilisation Mémoire (< 500 MB)
- [ ] Espace disque utilisé (< 50%)

### Rapports

- [ ] Rapport quotidien généré
- [ ] Rapport hebdomadaire généré
- [ ] Rapport mensuel généré

---

## 🚨 Troubleshooting

### Le bot ne démarre pas

- [ ] Vérifier les logs
  ```bash
  sudo journalctl -u adan-bot -n 100
  ```

- [ ] Vérifier les fichiers critiques
  ```bash
  python3 check_deployment.py
  ```

- [ ] Vérifier les permissions
  ```bash
  ls -la /home/trading_user/adan_bot/models/
  ```

### Erreur de chargement des modèles

- [ ] Vérifier l'intégrité des archives
  ```bash
  unzip -t /home/trading_user/adan_bot/models/w1/w1_model_final.zip
  ```

- [ ] Vérifier les fichiers pickle
  ```bash
  python3 -c "import pickle; pickle.load(open('/home/trading_user/adan_bot/models/w1/vecnormalize.pkl', 'rb'))"
  ```

### Erreur de connexion API

- [ ] Vérifier les clés API
  ```bash
  grep -E "API_KEY|SECRET_KEY" /home/trading_user/adan_bot/.env
  ```

- [ ] Tester la connectivité
  ```bash
  ping api.binance.com
  ```

- [ ] Vérifier les logs d'erreur
  ```bash
  sudo journalctl -u adan-bot | grep -i "api\|error"
  ```

---

## 📞 Support et Documentation

### Fichiers de Référence

- [ ] `DEPLOYMENT_GUIDE.md` - Guide complet
- [ ] `MIGRATION_SUMMARY.md` - Résumé de la migration
- [ ] `check_deployment.py` - Vérification des fichiers
- [ ] `start.sh` - Script de démarrage
- [ ] `.env.example` - Template de configuration

### Contacts

- [ ] Administrateur système
- [ ] Responsable du trading
- [ ] Support technique

---

## ✨ Finalisation

- [ ] Tous les tests réussis
- [ ] Documentation mise à jour
- [ ] Équipe informée du déploiement
- [ ] Monitoring configuré
- [ ] Alertes testées
- [ ] Plan de rollback documenté
- [ ] Backup effectué

---

**Statut:** À compléter avant le déploiement
**Date de Déploiement Prévue:** [À définir]
**Responsable:** [À définir]
**Dernière Mise à Jour:** 2 janvier 2026
