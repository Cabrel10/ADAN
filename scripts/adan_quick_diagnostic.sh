#!/bin/bash

echo "🚀 DIAGNOSTIC RAPIDE ADAN"
echo "========================"

echo -e "\n1. 📁 Espace disque:"
df -h /home/morningstar 2>/dev/null | tail -1 || df -h | tail -1

echo -e "\n2. 🐍 Processus Python:"
ps aux | grep -E "python.*(monitor|adan|trading)" | grep -v grep || echo "Aucun processus trouvé"

echo -e "\n3. 📊 Fichiers logs:"
ls -la paper_trading*.log 2>/dev/null || echo "Pas de logs"
if [ -f paper_trading.log ]; then
    echo "Dernières lignes:"
    tail -5 paper_trading.log
fi

echo -e "\n4. 🧠 Vérification Python:"
python3 << 'PYEOF'
import sys
print(f'Python {sys.version.split()[0]}')
try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
except:
    print('❌ PyTorch manquant')
try:
    import pandas_ta
    print('✅ pandas_ta installé')
except:
    print('❌ pandas_ta manquant')
try:
    import yaml
    print('✅ PyYAML installé')
except:
    print('❌ PyYAML manquant')
PYEOF

echo -e "\n5. 🔗 Vérification API:"
python3 << 'PYEOF'
try:
    from binance.client import Client
    print('✅ Binance API installé')
except ImportError as e:
    print(f'❌ Binance API: {e}')
PYEOF

echo -e "\n6. 📁 Vérification des chemins critiques:"
[ -d "/mnt/new_data/t10_training" ] && echo "✅ Dossier training trouvé" || echo "❌ Dossier training manquant"
[ -f "/mnt/new_data/t10_training/config.yaml" ] && echo "✅ Config YAML trouvée" || echo "❌ Config YAML manquante"
[ -d "/mnt/new_data/t10_training/checkpoints/final" ] && echo "✅ Checkpoints trouvés" || echo "❌ Checkpoints manquants"

echo -e "\n🎯 Status:"
echo "Si espace disque > 90%: ❌ Problème critique"
echo "Si pas de processus: ❌ Monitor non démarré"
echo "Si pas de logs: ❌ Problème d'écriture"
