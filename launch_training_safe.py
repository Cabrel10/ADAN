#!/usr/bin/env python3
"""
Lancement sécurisé de l'entraînement ADAN
Avec tests préalables et monitoring
"""
import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

def run_pre_tests():
    """Exécuter les tests préalables"""
    print("🧪 Exécution des tests préalables...")
    try:
        result = subprocess.run([
            sys.executable, "test_train_parallel_agents.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Tous les tests préalables passent")
            return True
        else:
            print("❌ Tests préalables échoués:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Tests préalables timeout (5min)")
        return False
    except Exception as e:
        print(f"❌ Erreur tests préalables: {e}")
        return False

def create_training_session():
    """Créer une session d'entraînement"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"training_sessions/session_{timestamp}"
    os.makedirs(session_dir, exist_ok=True)
    
    session_info = {
        "session_id": timestamp,
        "start_time": datetime.now().isoformat(),
        "status": "STARTING",
        "workers": ["w1", "w2", "w3", "w4"],
        "session_dir": session_dir
    }
    
    with open(f"{session_dir}/session_info.json", 'w') as f:
        json.dump(session_info, f, indent=2)
    
    return session_dir, session_info

def launch_training():
    """Lancer l'entraînement principal"""
    print("🚀 Lancement de l'entraînement...")
    cmd = [
        sys.executable, 
        "scripts/train_parallel_agents.py",
        "--config", "config/config.yaml",
        "--log-level", "INFO"
    ]
    
    try:
        # Lancer en arrière-plan mais avec monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        print(f"📋 Processus lancé (PID: {process.pid})")
        
        # Lire la sortie en temps réel
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        if return_code == 0:
            print("✅ Entraînement terminé avec succès")
            return True
        else:
            print(f"❌ Entraînement échoué (code: {return_code})")
            return False
    except KeyboardInterrupt:
        print("\n⚠️  Interruption utilisateur")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"❌ Erreur lancement: {e}")
        return False

def generate_final_report(session_dir):
    """Générer le rapport final"""
    print("📊 Génération du rapport final...")
    final_dir = Path("checkpoints/final")
    
    if not final_dir.exists():
        print("⚠️  Répertoire final non trouvé")
        return None
    
    models = list(final_dir.glob("w*_final.zip"))
    
    report = {
        "session_completed": datetime.now().isoformat(),
        "models_created": len(models),
        "models_list": [str(m) for m in models],
        "ready_for_fusion": len(models) == 4,
        "next_steps": [
            "1. Analyser les performances de chaque worker",
            "2. Comparer les métriques (Sharpe, Drawdown, Win Rate)",
            "3. Décider des poids de fusion basés sur les résultats",
            "4. Créer l'ensemble ADAN avec fusion adaptative"
        ]
    }
    
    report_path = f"{session_dir}/final_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Rapport final: {report_path}")
    return report

def main():
    """Fonction principale"""
    print("="*80)
    print("🎯 LANCEMENT SÉCURISÉ - ENTRAÎNEMENT ADAN")
    print("="*80)
    
    # 1. Tests préalables
    if not run_pre_tests():
        print("❌ Tests préalables échoués - Arrêt")
        return False
    
    # 2. Créer session
    session_dir, session_info = create_training_session()
    print(f"📁 Session créée: {session_dir}")
    
    # 3. Lancer entraînement
    success = launch_training()
    
    # 4. Rapport final
    if success:
        report = generate_final_report(session_dir)
        if report:
            print("\n" + "="*80)
            print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            print("="*80)
            print(f"📊 Modèles créés: {report['models_created']}/4")
            print(f"🔄 Prêt pour fusion: {'✅ OUI' if report['ready_for_fusion'] else '❌ NON'}")
            print("\n📋 PROCHAINES ÉTAPES:")
            for step in report['next_steps']:
                print(f"   {step}")
        return True
    else:
        print("\n❌ ENTRAÎNEMENT ÉCHOUÉ")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
