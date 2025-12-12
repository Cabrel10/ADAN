#!/bin/bash

# T10 : Lancement Entraînement Final - Mode Séquentiel
# Utilise /mnt/new_data pour les logs et checkpoints

set -e

WORK_DIR="/mnt/new_data/t10_training"
LOG_DIR="$WORK_DIR/logs"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
RESULTS_DIR="$WORK_DIR/results"

# Créer répertoires
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"

echo "================================================================================"
echo "T10 : LANCEMENT ENTRAÎNEMENT FINAL - MODE SÉQUENTIEL"
echo "================================================================================"
echo ""
echo "📁 Répertoires:"
echo "   Logs: $LOG_DIR"
echo "   Checkpoints: $CHECKPOINT_DIR"
echo "   Résultats: $RESULTS_DIR"
echo ""

# Fonction de lancement avec retry
launch_worker() {
    local WORKER_ID=$1
    local STEPS=$2
    local MAX_RETRIES=3
    local RETRY=0
    
    while [ $RETRY -lt $MAX_RETRIES ]; do
        echo "🚀 Lancement $WORKER_ID (tentative $((RETRY+1))/$MAX_RETRIES)..."
        
        # Lancer avec timeout de 6h
        timeout 6h python3 << EOF > "$LOG_DIR/training_${WORKER_ID}.log" 2>&1 &
import sys
import os
sys.path.insert(0, 'src')

from adan_trading_bot.training.ppo_trainer import PPOTrainer
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
import yaml
import time
from datetime import datetime

print(f"[{datetime.now()}] Démarrage $WORKER_ID")

# Charger config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

worker_key = '${WORKER_ID}'.lower()
worker_config = config['workers'][worker_key]

# Créer environnement
env = MultiAssetChunkedEnv(
    config=config,
    worker_id=int('${WORKER_ID}'[1:]),
    max_positions=worker_config['trading_parameters'].get('max_concurrent_positions', 3)
)

# Créer trainer
ppo_params = worker_config['agent_config']['ppo_hyperparams']
trainer = PPOTrainer(
    env=env,
    total_timesteps=$STEPS,
    learning_rate=ppo_params.get('learning_rate', 3e-4),
    n_steps=ppo_params.get('n_steps', 2048),
    batch_size=ppo_params.get('batch_size', 64),
    n_epochs=ppo_params.get('n_epochs', 10),
    gamma=ppo_params.get('gamma', 0.99),
    gae_lambda=ppo_params.get('gae_lambda', 0.95),
    clip_range=ppo_params.get('clip_range', 0.2),
    ent_coef=ppo_params.get('ent_coef', 0.01),
    vf_coef=ppo_params.get('vf_coef', 0.5),
    max_grad_norm=ppo_params.get('max_grad_norm', 0.5),
    verbose=1
)

# Lancer entraînement
print(f"[{datetime.now()}] Entraînement en cours...")
trainer.learn(total_timesteps=$STEPS)

# Sauvegarder
os.makedirs('$RESULTS_DIR', exist_ok=True)
model_path = '$RESULTS_DIR/${WORKER_ID}_final_model.zip'
trainer.model.save(model_path)
print(f"[{datetime.now()}] Modèle sauvegardé: {model_path}")

print(f"[{datetime.now()}] $WORKER_ID COMPLÉTÉ")
EOF
        
        local PID=$!
        echo "   PID: $PID"
        
        # Attendre 30s pour vérifier que ça démarre bien
        sleep 30
        
        if ps -p $PID > /dev/null 2>&1; then
            echo "   ✅ $WORKER_ID démarré avec succès"
            echo "$PID" > "$LOG_DIR/${WORKER_ID}.pid"
            
            # Attendre la fin
            echo "   ⏳ Attente de la fin de $WORKER_ID..."
            wait $PID
            
            if [ $? -eq 0 ]; then
                echo "   ✅ $WORKER_ID COMPLÉTÉ"
                rm -f "$LOG_DIR/${WORKER_ID}.pid"
                return 0
            else
                echo "   ❌ $WORKER_ID a échoué"
                RETRY=$((RETRY+1))
            fi
        else
            echo "   ❌ $WORKER_ID a crashé au démarrage"
            RETRY=$((RETRY+1))
            
            if [ $RETRY -lt $MAX_RETRIES ]; then
                echo "   ⏳ Attente 10s avant retry..."
                sleep 10
            fi
        fi
    done
    
    echo "❌ ÉCHEC: Impossible de lancer $WORKER_ID après $MAX_RETRIES tentatives"
    return 1
}

# Lancer les 4 workers en séquence
STEPS_PER_WORKER=250000

for WORKER in W1 W2 W3 W4; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "WORKER: $WORKER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if ! launch_worker $WORKER $STEPS_PER_WORKER; then
        echo ""
        echo "❌ ARRÊT: Échec du lancement de $WORKER"
        exit 1
    fi
    
    echo "   ⏳ Nettoyage mémoire avant le prochain worker..."
    sleep 30
done

echo ""
echo "================================================================================"
echo "✅ TOUS LES WORKERS COMPLÉTÉS AVEC SUCCÈS"
echo "================================================================================"
echo ""
echo "📊 Résultats disponibles dans: $RESULTS_DIR"
echo ""
