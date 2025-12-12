#!/bin/bash

# 🚀 LAUNCH TENSORBOARD FOR W1 METRICS VISUALIZATION

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🚀 LAUNCHING TENSORBOARD - W1 METRICS                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
LOGDIR="/home/morningstar/Documents/trading/bot/ARCHIVES_BEFORE_NAN_FIX/logs_before_nan_fix/tensorboard_w1"
PORT=6006
HOST="0.0.0.0"

# Vérifier que le répertoire existe
if [ ! -d "$LOGDIR" ]; then
    echo "❌ Log directory not found: $LOGDIR"
    exit 1
fi

echo "📊 TensorBoard Configuration"
echo "═══════════════════════════════════════════════════════════════"
echo "Log Directory: $LOGDIR"
echo "Port: $PORT"
echo "Host: $HOST"
echo ""

# Compter les événements
event_count=$(find "$LOGDIR" -name "events.out.tfevents*" | wc -l)
echo "📈 Found $event_count event files"
echo ""

echo "🚀 Starting TensorBoard..."
echo ""
echo "Access TensorBoard at:"
echo "  🌐 http://localhost:$PORT"
echo "  🌐 http://127.0.0.1:$PORT"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Lancer TensorBoard
tensorboard --logdir="$LOGDIR" --port=$PORT --host=$HOST
