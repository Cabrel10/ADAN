#!/bin/bash
# ============================================================================
# COMMANDES DE MISE À JOUR POUR COLAB
# À exécuter dans le terminal Colab après le setup initial
# ============================================================================

echo "🔧 MISE À JOUR INCRÉMENTALE POUR COLAB"
echo "========================================"

# ============== SECTION 1: TÉLÉCHARGER LE PATCH ==============
echo ""
echo "📥 Section 1: Téléchargement du patch..."
cd /content/bot

# Créer le script de correction
cat > fix_all_bugs.py << 'FIXSCRIPT'
#!/usr/bin/env python3
import os
import re
from pathlib import Path

def fix_datetime_timezone_comprehensive(root_dir='src'):
    """Corrige TOUTES les occurrences de datetime.timezone"""
    print("🔧 Correction de datetime.timezone...")
    
    fixes = []
    for path in Path(root_dir).rglob('*.py'):
        with open(path, 'r') as f:
            content = f.read()
        
        original = content
        
        # Pattern 1: datetime.now(datetime.timezone.utc)
        content = re.sub(
            r'datetime\.now\(datetime\.timezone\.utc\)',
            'datetime.now(timezone.utc)',
            content
        )
        
        # Pattern 2: datetime.timezone.utc seul
        content = re.sub(
            r'datetime\.timezone\.utc',
            'timezone.utc',
            content
        )
        
        # Ajouter l'import si nécessaire
        if 'timezone.utc' in content:
            has_timezone_import = 'from datetime import' in content and 'timezone' in content
            
            if not has_timezone_import:
                if 'from datetime import datetime' in content:
                    content = content.replace(
                        'from datetime import datetime',
                        'from datetime import datetime, timezone',
                        1
                    )
                elif 'import datetime' in content:
                    lines = content.split('\n')
                    import_idx = next((i for i, line in enumerate(lines) if 'import datetime' in line), -1)
                    if import_idx >= 0:
                        lines.insert(import_idx + 1, 'from datetime import timezone')
                        content = '\n'.join(lines)
        
        if content != original:
            with open(path, 'w') as f:
                f.write(content)
            fixes.append(str(path))
    
    return fixes

if __name__ == '__main__':
    fixes = fix_datetime_timezone_comprehensive()
    print(f"✅ {len(fixes)} fichiers corrigés")
    for f in fixes[:5]:
        print(f"  - {f}")
    if len(fixes) > 5:
        print(f"  ... et {len(fixes) - 5} autres")
FIXSCRIPT

# ============== SECTION 2: APPLIQUER LE PATCH ==============
echo ""
echo "🔧 Section 2: Application du patch..."
python fix_all_bugs.py

# ============== SECTION 3: VÉRIFICATION ==============
echo ""
echo "🔍 Section 3: Vérification des corrections..."
python << 'EOF'
import os
import subprocess

# Vérifier qu'il ne reste plus de datetime.timezone
result = subprocess.run(
    ['grep', '-rn', 'datetime.timezone', 'src/', '--include=*.py'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("❌ Il reste des occurrences:")
    print(result.stdout[:500])
else:
    print("✅ Toutes les occurrences corrigées!")

# Test d'import
try:
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer
    print("✅ Tous les imports OK!")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
EOF

# ============== SECTION 4: MISE À JOUR DES DÉPENDANCES ==============
echo ""
echo "📚 Section 4: Mise à jour des dépendances critiques..."
pip install --upgrade --no-cache-dir \
  numpy==1.26.4 \
  pandas==2.2.2 \
  ta-lib \
  2>&1 | tail -5

# ============== SECTION 5: LANCEMENT ==============
echo ""
echo "🚀 Section 5: Prêt pour l'entraînement!"
echo ""
echo "Pour lancer l'entraînement, exécutez:"
echo "python scripts/train_parallel_agents.py --config-path config/config_colab.yaml --checkpoint-dir checkpoints --resume"
echo ""
echo "========================================"
echo "✅ Mise à jour terminée avec succès!"
echo "========================================"
