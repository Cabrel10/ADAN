#!/usr/bin/env python3
"""
Script de vérification pré-déploiement
Vérifie que tous les fichiers critiques sont présents et accessibles
"""
import os
import sys
import json

def check_requirements():
    """Vérifie que tous les fichiers critiques sont présents"""
    
    required_files = {
        'w1': [
            'models/w1/w1_model_final.zip',
            'models/w1/vecnormalize.pkl',
        ],
        'w2': [
            'models/w2/w2_model_final.zip',
            'models/w2/vecnormalize.pkl',
        ],
        'w3': [
            'models/w3/w3_model_final.zip',
            'models/w3/vecnormalize.pkl',
        ],
        'w4': [
            'models/w4/w4_model_final.zip',
            'models/w4/vecnormalize.pkl',
        ],
        'ensemble': [
            'models/ensemble/adan_ensemble_config.json',
        ],
        'config': [
            'config/config.yaml',
        ],
    }
    
    print("🔍 Vérification des fichiers critiques...\n")
    
    all_present = True
    missing_files = []
    
    for category, files in required_files.items():
        print(f"📦 {category.upper()}:")
        for f in files:
            if os.path.exists(f):
                size = os.path.getsize(f)
                size_mb = size / (1024 * 1024)
                print(f"  ✅ {f} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ {f} - MANQUANT")
                all_present = False
                missing_files.append(f)
        print()
    
    # Vérifier la configuration ensemble
    if os.path.exists('models/ensemble/adan_ensemble_config.json'):
        try:
            with open('models/ensemble/adan_ensemble_config.json', 'r') as f:
                config = json.load(f)
                print("📋 Configuration ADAN Ensemble:")
                print(f"  Poids w1: {config.get('weights', {}).get('w1', 'N/A')}")
                print(f"  Poids w2: {config.get('weights', {}).get('w2', 'N/A')}")
                print(f"  Poids w3: {config.get('weights', {}).get('w3', 'N/A')}")
                print(f"  Poids w4: {config.get('weights', {}).get('w4', 'N/A')}")
                print()
        except Exception as e:
            print(f"  ⚠️  Erreur lecture config: {e}\n")
    
    if all_present:
        print("✅ TOUS LES FICHIERS REQUIS SONT PRÉSENTS")
        print("🚀 Le bot est prêt pour le déploiement\n")
        return True
    else:
        print("❌ FICHIERS MANQUANTS:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n⚠️  Le déploiement ne peut pas procéder\n")
        return False

def check_disk_space():
    """Vérifie l'espace disque disponible"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"💾 Espace disque disponible: {free_gb:.2f} GB")
    if free_gb < 1:
        print("  ⚠️  Espace disque faible!")
        return False
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("VÉRIFICATION PRÉ-DÉPLOIEMENT - ADAN TRADING BOT")
    print("=" * 60 + "\n")
    
    files_ok = check_requirements()
    disk_ok = check_disk_space()
    
    if files_ok and disk_ok:
        print("\n" + "=" * 60)
        print("✅ VÉRIFICATION RÉUSSIE - PRÊT POUR DÉPLOIEMENT")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ VÉRIFICATION ÉCHOUÉE - CORRECTIONS NÉCESSAIRES")
        print("=" * 60)
        sys.exit(1)
