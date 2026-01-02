ement

```bash
# Depuis la racine du projet
tar -czf adan_bot_deploy.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='logs/*' \
  --exclude='.env' \
  .

# Vérifier la taille
ls -lh adan_bot_deploy.tar.gz
```

### 2. Transférer vers le serveur

```bash
scp adan_bot_deploy.tar.gz user@serveur:/home/user/
```

### 3. Sur le serveur distant

```bash
# Se connecter au serveur
ssh user@serveur

# Extraire l'archive
cd /home/user
tar -xzf adan_bot_deploy.tar.gz
cd adan_bot_deploy

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
nano .env  # Ajouter vos clés API

# Vérifier les fichiers
python3 check_deployment.py

# Démarrer le bot
chmod +x start.sh
./start.sh
```

## 🔄 Gestion du Service (Systemd)

### Créer un service systemd

Créer `/etc/systemd/system/adan-bot.service` :

```ini
[Unit]
Description=ADAN Trading Bot
After=network.target

[Service]
Type=simple
User=trading_user
WorkingDirectory=/home/trading_user/adan_bot_deploy
ExecStart=/home/trading_user/adan_bot_deploy/start.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Commandes de gestion

```bash
# Activer le service au démarrage
sudo systemctl enable adan-bot

# Démarrer le service
sudo systemctl start adan-bot

# Vérifier le statut
sudo systemctl status adan-bot

# Voir les logs
sudo journalctl -u adan-bot -f

# Arrêter le service
sudo systemctl stop adan-bot

# Redémarrer le service
sudo systemctl restart adan-bot
```

## 📊 Monitoring

### Vérifier que le bot tourne

```bash
ps aux | grep paper_trading_monitor
```

### Voir les logs en temps réel

```bash
tail -f logs/bot.log
```

### Vérifier les performances

```bash
# Utilisation CPU/Mémoire
top -p $(pgrep -f paper_trading_monitor)

# Espace disque
df -h

# Connexion réseau
netstat -an | grep ESTABLISHED
```

## 🛠️ Troubleshooting

### Le bot ne démarre pas

1. Vérifier les fichiers critiques :
   ```bash
   python3 check_deployment.py
   ```

2. Vérifier les logs :
   ```bash
   tail -100 logs/bot.log
   ```

3. Vérifier les permissions :
   ```bash
   ls -la models/
   ls -la config/
   ```

### Erreur de chargement des modèles

```
❌ Impossible de charger w1_model_final.zip
```

Solution :
```bash
# Vérifier que les fichiers existent
ls -lh models/w1/

# Vérifier l'intégrité du zip
unzip -t models/w1/w1_model_final.zip
```

### Erreur de normalisation

```
❌ Impossible de charger vecnormalize.pkl
```

Solution :
```bash
# Vérifier que le fichier existe
ls -lh models/w1/vecnormalize.pkl

# Vérifier qu'il n'est pas corrompu
python3 -c "import pickle; pickle.load(open('models/w1/vecnormalize.pkl', 'rb'))"
```

### Problème de connexion API

1. Vérifier les clés API dans `.env`
2. Vérifier la connectivité réseau :
   ```bash
   ping api.binance.com
   ```
3. Vérifier les logs pour les erreurs spécifiques

## 📈 Optimisations pour Production

### 1. Rotation des logs

Créer `/etc/logrotate.d/adan-bot` :

```
/home/trading_user/adan_bot_deploy/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0640 trading_user trading_user
}
```

### 2. Monitoring avec Prometheus (optionnel)

Ajouter des métriques dans `paper_trading_monitor.py` :
- Nombre de trades exécutés
- Profit/Perte
- Uptime du bot
- Latence API

### 3. Alertes Telegram

Configurer les alertes pour :
- Erreurs critiques
- Redémarrages du bot
- Changements de stratégie
- Rapports de performance quotidiens

## ✅ Checklist Pré-Déploiement

- [ ] Tous les fichiers critiques présents (`check_deployment.py` réussi)
- [ ] Variables d'environnement configurées (`.env` rempli)
- [ ] Test local réussi (bot démarre et charge les modèles)
- [ ] Logs générés correctement
- [ ] Clés API valides et testées
- [ ] Espace disque suffisant (> 1 GB)
- [ ] Connexion réseau stable
- [ ] Backup des configurations
- [ ] Plan de rollback en cas de problème
- [ ] Monitoring configuré

## 📞 Support

En cas de problème :

1. Vérifier les logs : `tail -f logs/bot.log`
2. Exécuter la vérification : `python3 check_deployment.py`
3. Consulter les fichiers de diagnostic
4. Vérifier la connectivité réseau et API

---

**Dernière mise à jour:** 2 janvier 2026
**Version:** 1.0
