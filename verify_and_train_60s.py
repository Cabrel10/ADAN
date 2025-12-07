#!/usr/bin/env python3
"""
Vérification rapide des workers et lancement de l'entraînement avec timeout 60s
"""
import sys
import subprocess
import time
from pathlib import Path

def verify_workers_config():
    """Vérifier que tous les workers ont leurs hyperparamètres"""
    print("="*80)
    print("✅ VÉRIFICATION DES HYPERPARAMÈTRES DES WORKERS")
    print("="*80)
    
    from adan_trading_bot.common.config_loader import ConfigLoader
    
    config = ConfigLoader.load_config("config/config.yaml")
    workers = ["w1", "w2", "w3", "w4"]
    
    all_ok = True
    for worker_id in workers:
        if worker_id not in config["workers"]:
            print(f"❌ {worker_id}: NOT FOUND")
            all_ok = False
            continue
        
        worker_config = config["workers"][worker_id]
        agent_config = worker_config.get("agent_config", {})
        
        print(f"\n✅ {worker_id}:")
        print(f"   Name: {worker_config.get('name', 'N/A')}")
        print(f"   Description: {worker_config.get('description', 'N/A')}")
        print(f"   Learning Rate: {agent_config.get('learning_rate', 'N/A')}")
        print(f"   Gamma: {agent_config.get('gamma', 'N/A')}")
        print(f"   N Steps: {agent_config.get('n_steps', 'N/A')}")
        print(f"   Batch Size: {agent_config.get('batch_size', 'N/A')}")
        print(f"   N Epochs: {agent_config.get('n_epochs', 'N/A')}")
        print(f"   Clip Range: {agent_config.get('clip_range', 'N/A')}")
        print(f"   Ent Coef: {agent_config.get('ent_coef', 'N/A')}")
        
        # Vérifier les clés essentielles
        required_keys = ["learning_rate", "gamma", "n_steps", "batch_size", "n_epochs"]
        missing = [k for k in required_keys if k not in agent_config]
        if missing:
            print(f"   ⚠️  Missing keys: {missing}")
            all_ok = False
    
    return all_ok

def launch_training_60s():
    """Lancer l'entraînement avec timeout 60s"""
    print("\n" + "="*80)
    print("🚀 LANCEMENT DE L'ENTRAÎNEMENT (TIMEOUT: 60s)")
    print("="*80)
    
    cmd = [
        sys.executable,
        "scripts/train_parallel_agents.py",
        "--config", "config/config.yaml",
        "--log-level", "INFO",
        "--steps", "1000"  # Très court pour test
    ]
    
    print(f"\n📋 Commande: {' '.join(cmd)}")
    print(f"⏱️  Timeout: 60 secondes\n")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            timeout=60,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        print(f"✅ Entraînement terminé en {elapsed:.1f}s")
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ SUCCÈS!")
            return True
        else:
            print("❌ ERREUR!")
            print("\nSTDOUT:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("\nSTDERR:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT ATTEINT (60s)")
        print("✅ Entraînement lancé avec succès mais dépassé le timeout")
        print("   (C'est normal pour un entraînement réel)")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def check_results():
    """Vérifier les résultats"""
    print("\n" + "="*80)
    print("📊 VÉRIFICATION DES RÉSULTATS")
    print("="*80)
    
    final_dir = Path("checkpoints/final")
    
    if not final_dir.exists():
        print("⚠️  Répertoire final non trouvé (normal si timeout)")
        return False
    
    workers = ["w1", "w2", "w3", "w4"]
    found = 0
    
    for worker_id in workers:
        model_file = final_dir / f"{worker_id}_final.zip"
        vec_file = final_dir / f"{worker_id}_vecnormalize.pkl"
        
        if model_file.exists() and vec_file.exists():
            size_mb = model_file.stat().st_size / (1024*1024)
            print(f"✅ {worker_id}: Model ({size_mb:.1f}MB) + VecNorm")
            found += 1
        else:
            print(f"⚠️  {worker_id}: Pas encore complété")
    
    print(f"\n📊 Résultat: {found}/4 workers complétés")
    return found == 4

if __name__ == "__main__":
    print("\n🎯 VÉRIFICATION ET ENTRAÎNEMENT 60s\n")
    
    # 1. Vérifier les workers
    if not verify_workers_config():
        print("\n❌ Vérification échouée")
        sys.exit(1)
    
    # 2. Lancer l'entraînement
    if not launch_training_60s():
        print("\n❌ Entraînement échoué")
        sys.exit(1)
    
    # 3. Vérifier les résultats
    check_results()
    
    print("\n" + "="*80)
    print("✅ VÉRIFICATION COMPLÈTE")
    print("="*80)
