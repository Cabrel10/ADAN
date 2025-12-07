#!/usr/bin/env python3
"""
Script d'analyse détaillée des résultats de chaque worker
Extrait les métriques de performance et les compare
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

def analyze_worker_results():
    """Analyser les résultats détaillés de chaque worker"""
    print("="*80)
    print("📊 ANALYSE DÉTAILLÉE DES RÉSULTATS PAR WORKER")
    print("="*80)
    
    final_dir = Path("checkpoints/final")
    
    if not final_dir.exists():
        print("❌ Répertoire final non trouvé - Entraînement non lancé")
        return False
    
    # 1. Charger le rapport de performance
    report_path = final_dir / "training_performance_report.json"
    if not report_path.exists():
        print("❌ Rapport de performance non trouvé")
        return False
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("\n📋 RAPPORT D'ENTRAÎNEMENT")
    print("-" * 80)
    print(f"Timestamp: {report.get('timestamp', 'N/A')}")
    print(f"Entraînement complété: {'✅ OUI' if report.get('training_completed') else '❌ NON'}")
    print(f"Workers entraînés: {report.get('workers_trained', 0)}/4")
    print(f"Prêt pour fusion: {'✅ OUI' if report.get('fusion_ready') else '❌ NON'}")
    
    # 2. Analyser chaque worker
    print("\n📈 RÉSULTATS PAR WORKER")
    print("-" * 80)
    
    worker_results = report.get('worker_results', {})
    successful_workers = []
    failed_workers = []
    
    for worker_id in ["w1", "w2", "w3", "w4"]:
        if worker_id not in worker_results:
            print(f"\n❌ {worker_id}: Pas de données")
            failed_workers.append(worker_id)
            continue
        
        result = worker_results[worker_id]
        status = result.get('status', 'UNKNOWN')
        
        if "SUCCESS" in status:
            successful_workers.append(worker_id)
            print(f"\n✅ {worker_id}: {status}")
        else:
            failed_workers.append(worker_id)
            print(f"\n❌ {worker_id}: {status}")
        
        print(f"   Model Size: {result.get('model_size_mb', 0):.2f} MB")
        print(f"   Model: {Path(result.get('model_path', '')).name}")
        print(f"   VecNorm: {Path(result.get('vec_path', '')).name}")
    
    # 3. Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ")
    print("="*80)
    
    print(f"\n✅ Workers réussis: {len(successful_workers)}/4")
    for w in successful_workers:
        print(f"   • {w}")
    
    if failed_workers:
        print(f"\n❌ Workers échoués: {len(failed_workers)}/4")
        for w in failed_workers:
            print(f"   • {w}")
    
    # 4. Prochaines étapes
    print("\n" + "="*80)
    print("🎯 PROCHAINES ÉTAPES")
    print("="*80)
    
    if report.get('fusion_ready'):
        print("\n✅ TOUS LES WORKERS SONT PRÊTS POUR LA FUSION!")
        print("\nÉtapes recommandées:")
        for i, step in enumerate(report.get('next_steps', []), 1):
            print(f"   {step}")
        
        print("\n💡 CONSEILS POUR LA FUSION:")
        print("   1. Analyser les logs de chaque worker")
        print("   2. Comparer les métriques (Sharpe, Drawdown, Win Rate)")
        print("   3. Identifier le worker dominant par pallier de capital")
        print("   4. Définir les poids de fusion adaptatifs")
        print("   5. Créer l'ensemble ADAN avec vos poids optimaux")
        
        return True
    else:
        print("\n⚠️  CERTAINS WORKERS NE SONT PAS PRÊTS")
        print("   Relancer l'entraînement ou corriger les erreurs")
        return False

def extract_worker_metrics_from_logs():
    """Extraire les métriques de chaque worker depuis les logs"""
    print("\n" + "="*80)
    print("📝 EXTRACTION DES MÉTRIQUES DEPUIS LES LOGS")
    print("="*80)
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("⚠️  Répertoire logs non trouvé")
        return
    
    workers = ["w1", "w2", "w3", "w4"]
    
    for worker_id in workers:
        print(f"\n🔍 {worker_id}:")
        
        # Chercher les fichiers de log du worker
        worker_logs = list(logs_dir.glob(f"{worker_id}*.log"))
        
        if not worker_logs:
            print(f"   ⚠️  Pas de logs trouvés")
            continue
        
        print(f"   ✅ {len(worker_logs)} fichiers de log trouvés")
        
        # Analyser le dernier log
        latest_log = max(worker_logs, key=lambda x: x.stat().st_mtime)
        print(f"   📄 Dernier log: {latest_log.name}")
        print(f"   📅 Modifié: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")
        print(f"   📊 Taille: {latest_log.stat().st_size / 1024:.1f} KB")

def create_comparison_report():
    """Créer un rapport de comparaison entre les workers"""
    print("\n" + "="*80)
    print("📊 RAPPORT DE COMPARAISON ENTRE WORKERS")
    print("="*80)
    
    final_dir = Path("checkpoints/final")
    report_path = final_dir / "training_performance_report.json"
    
    if not report_path.exists():
        print("⚠️  Rapport de performance non trouvé")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    worker_results = report.get('worker_results', {})
    
    print("\n📋 TABLEAU COMPARATIF")
    print("-" * 80)
    print(f"{'Worker':<10} {'Status':<15} {'Model Size (MB)':<20} {'Ready':<10}")
    print("-" * 80)
    
    for worker_id in ["w1", "w2", "w3", "w4"]:
        if worker_id not in worker_results:
            print(f"{worker_id:<10} {'N/A':<15} {'N/A':<20} {'❌':<10}")
            continue
        
        result = worker_results[worker_id]
        status = "✅ SUCCESS" if "SUCCESS" in result.get('status', '') else "❌ FAILED"
        size = result.get('model_size_mb', 0)
        ready = "✅" if "SUCCESS" in result.get('status', '') else "❌"
        
        print(f"{worker_id:<10} {status:<15} {size:<20.2f} {ready:<10}")
    
    print("-" * 80)
    
    # Créer un fichier de rapport
    comparison_report = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "worker_comparison",
        "workers": {}
    }
    
    for worker_id in ["w1", "w2", "w3", "w4"]:
        if worker_id in worker_results:
            result = worker_results[worker_id]
            comparison_report["workers"][worker_id] = {
                "status": result.get('status', 'UNKNOWN'),
                "model_size_mb": result.get('model_size_mb', 0),
                "model_exists": result.get('status', '').count('SUCCESS') > 0,
                "vecnorm_exists": result.get('status', '').count('SUCCESS') > 0,
            }
    
    # Sauvegarder le rapport
    comparison_path = final_dir / "worker_comparison_report.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print(f"\n✅ Rapport de comparaison sauvegardé: {comparison_path.name}")

if __name__ == "__main__":
    print("\n🚀 ANALYSE COMPLÈTE DES RÉSULTATS D'ENTRAÎNEMENT\n")
    
    success = analyze_worker_results()
    extract_worker_metrics_from_logs()
    create_comparison_report()
    
    if success:
        print("\n" + "="*80)
        print("✅ ANALYSE COMPLÈTE - PRÊT POUR LA FUSION!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  ANALYSE INCOMPLÈTE - VÉRIFIER LES ERREURS")
        print("="*80)
    
    sys.exit(0 if success else 1)
