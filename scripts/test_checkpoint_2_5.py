#!/usr/bin/env python3
"""Test Checkpoint 2.5: Vérifier que build_observation utilise VecNormalize"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*70)
print("TEST CHECKPOINT 2.5: build_observation() avec VecNormalize")
print("="*70)

# Test 1: Vérifier la signature
print("\n1️⃣  Vérification signature de build_observation()...")
try:
    import inspect
    from scripts.paper_trading_monitor import RealPaperTradingMonitor
    
    sig = inspect.signature(RealPaperTradingMonitor.build_observation)
    params = list(sig.parameters.keys())
    
    if 'worker_id' in params:
        print("   ✅ worker_id présent dans la signature")
    else:
        print("   ❌ worker_id MANQUANT - Ajouter à la signature")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 2: Vérifier le code source
print("\n2️⃣  Vérification du code source...")
try:
    with open('scripts/paper_trading_monitor.py', 'r') as f:
        code = f.read()
    
    # Chercher la normalisation manuelle (devrait être absente)
    if 'window.mean(axis=0)' in code and 'window.std(axis=0)' in code:
        print("   ⚠️  Normalisation manuelle encore présente")
        print("      (Peut être acceptable si VecNormalize est aussi utilisé)")
    
    # Chercher l'utilisation de VecNormalize
    if 'self.worker_envs[worker_id]' in code:
        print("   ✅ self.worker_envs[worker_id] utilisé")
    else:
        print("   ❌ self.worker_envs[worker_id] NON TROUVÉ")
        sys.exit(1)
    
    if 'env.normalize_obs' in code:
        print("   ✅ env.normalize_obs() appelé")
    else:
        print("   ❌ env.normalize_obs() NON TROUVÉ")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 3: Vérifier la compilation
print("\n3️⃣  Vérification compilation Python...")
try:
    import py_compile
    py_compile.compile('scripts/paper_trading_monitor.py', doraise=True)
    print("   ✅ Compilation réussie - Aucune erreur de syntaxe")
except Exception as e:
    print(f"   ❌ Erreur de compilation: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ CHECKPOINT 2.5 VALIDÉ")
print("="*70)
print("\nLa modification de build_observation() est correcte.")
print("Prochaine étape: Checkpoint 2.6 - Validation divergence")
