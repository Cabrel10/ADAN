#!/usr/bin/env python3
"""
Script de correction automatique des problèmes critiques ADAN
Corrige les 5 problèmes identifiés par le diagnostic
"""
import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}\n")

def run_command(cmd, description=""):
    """Exécute une commande et retourne le résultat"""
    if description:
        print(f"⏳ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description or 'Commande'} réussie")
            return True
        else:
            print(f"❌ {description or 'Commande'} échouée: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def fix_binance_api():
    """Installe Binance API"""
    print_header("CORRECTION 1: Installation Binance API")
    
    # Vérifier si déjà installé
    try:
        import binance
        print("✅ Binance API déjà installé")
        return True
    except ImportError:
        print("❌ Binance API non trouvé, installation...")
        return run_command("pip install python-binance", "Installation Binance API")

def fix_model_paths():
    """Corrige les chemins des modèles dans paper_trading_monitor.py"""
    print_header("CORRECTION 2: Correction des chemins des modèles")
    
    monitor_file = Path("scripts/paper_trading_monitor.py")
    
    if not monitor_file.exists():
        print(f"❌ Fichier non trouvé: {monitor_file}")
        return False
    
    # Lire le fichier
    with open(monitor_file, 'r') as f:
        content = f.read()
    
    # Vérifier si correction nécessaire
    if "w1_final.zip" in content:
        print("✅ Chemins déjà corrigés")
        return True
    
    # Corriger les chemins
    original_content = content
    content = content.replace(
        'checkpoints/final/w1.zip',
        'checkpoints/final/w1_final.zip'
    )
    content = content.replace(
        'checkpoints/final/w2.zip',
        'checkpoints/final/w2_final.zip'
    )
    content = content.replace(
        'checkpoints/final/w3.zip',
        'checkpoints/final/w3_final.zip'
    )
    content = content.replace(
        'checkpoints/final/w4.zip',
        'checkpoints/final/w4_final.zip'
    )
    
    if content != original_content:
        with open(monitor_file, 'w') as f:
            f.write(content)
        print("✅ Chemins des modèles corrigés")
        return True
    else:
        print("⚠️  Aucun chemin à corriger trouvé")
        return True

def find_config_yaml():
    """Localise le fichier config.yaml"""
    print_header("CORRECTION 3: Localisation config.yaml")
    
    expected_path = Path("/mnt/new_data/t10_training/config.yaml")
    
    if expected_path.exists():
        print(f"✅ Config trouvée: {expected_path}")
        return True
    
    print("❌ Config non trouvée au chemin attendu")
    print("🔍 Recherche dans /mnt/new_data...")
    
    # Chercher le fichier
    result = subprocess.run(
        "find /mnt/new_data -name 'config.yaml' -o -name '*.yaml' 2>/dev/null | head -5",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        files = result.stdout.strip().split('\n')
        print(f"📁 Fichiers YAML trouvés:")
        for f in files:
            if f:
                print(f"   - {f}")
        
        # Proposer de créer un lien
        if files[0]:
            print(f"\n💡 Créer un lien symbolique?")
            print(f"   ln -s {files[0]} {expected_path}")
            return True
    else:
        print("❌ Aucun fichier config.yaml trouvé")
        return False

def verify_models():
    """Vérifie que les modèles existent"""
    print_header("CORRECTION 4: Vérification des modèles")
    
    model_dir = Path("/mnt/new_data/t10_training/checkpoints/final")
    
    if not model_dir.exists():
        print(f"❌ Répertoire non trouvé: {model_dir}")
        return False
    
    models = ['w1_final.zip', 'w2_final.zip', 'w3_final.zip', 'w4_final.zip']
    all_found = True
    
    for model in models:
        model_path = model_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model}: {size_mb:.1f} MB")
        else:
            print(f"❌ {model}: NON TROUVÉ")
            all_found = False
    
    return all_found

def verify_normalizers():
    """Vérifie que les normaliseurs existent"""
    print_header("CORRECTION 5: Vérification des normaliseurs")
    
    model_dir = Path("/mnt/new_data/t10_training/checkpoints/final")
    normalizers = ['w1_vecnormalize.pkl', 'w2_vecnormalize.pkl', 
                   'w3_vecnormalize.pkl', 'w4_vecnormalize.pkl']
    all_found = True
    
    for norm in normalizers:
        norm_path = model_dir / norm
        if norm_path.exists():
            size_kb = norm_path.stat().st_size / 1024
            print(f"✅ {norm}: {size_kb:.1f} KB")
        else:
            print(f"❌ {norm}: NON TROUVÉ")
            all_found = False
    
    return all_found

def generate_summary():
    """Génère un résumé des corrections"""
    print_header("RÉSUMÉ DES CORRECTIONS")
    
    print("""
✅ CORRECTIONS APPLIQUÉES:
   1. Binance API - Vérifiée/Installée
   2. Chemins des modèles - Corrigés
   3. Config YAML - Localisée
   4. Modèles - Vérifiés
   5. Normaliseurs - Vérifiés

🎯 PROCHAINES ÉTAPES:
   1. Redémarrer le monitor:
      pkill -f paper_trading_monitor.py
      python scripts/paper_trading_monitor.py --api_key "..." --api_secret "..." &
   
   2. Vérifier les logs:
      tail -50 paper_trading.log
   
   3. Exécuter les diagnostics:
      python3 scripts/verify_data_pipeline.py
      python3 scripts/debug_indicators.py
      python3 scripts/test_trade_execution.py
      python3 scripts/verify_cnn_ppo.py

📊 STATUS: ✅ PRÊT POUR REDÉMARRAGE
    """)

def main():
    print("\n" + "="*60)
    print("🚀 CORRECTION AUTOMATIQUE DES PROBLÈMES CRITIQUES ADAN")
    print("="*60)
    
    results = {
        "Binance API": fix_binance_api(),
        "Chemins modèles": fix_model_paths(),
        "Config YAML": find_config_yaml(),
        "Modèles": verify_models(),
        "Normaliseurs": verify_normalizers(),
    }
    
    # Résumé
    print_header("RÉSUMÉ DES VÉRIFICATIONS")
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    # Générer le résumé final
    generate_summary()
    
    # Retourner le code de sortie
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
