#!/usr/bin/env python3
"""
🔧 PATCH POUR INTÉGRER LA NORMALISATION DANS LE MONITOR
Ajoute la normalisation des observations pour éviter le covariate shift
"""

import sys
from pathlib import Path

def create_patch():
    """Crée le patch de normalisation"""
    
    patch_code = '''
# ============================================================================
# 🔧 PATCH DE NORMALISATION - À AJOUTER DANS paper_trading_monitor.py
# ============================================================================

# 1. IMPORTS (ajouter à la section imports)
from adan_trading_bot.normalization import ObservationNormalizer, DriftDetector
import numpy as np

# 2. INITIALISATION (ajouter dans __init__ ou au démarrage du monitor)
class PaperTradingMonitorPatched:
    def __init__(self, ...):
        # ... code existant ...
        
        # 🔧 AJOUTER CES LIGNES:
        self.normalizer = ObservationNormalizer()
        self.drift_detector = DriftDetector(window_size=100, threshold=2.0)
        
        logger.info(f"✅ Normaliseur initialisé: {self.normalizer.is_loaded}")
        logger.info(f"✅ Détecteur de dérive initialisé")

# 3. MODIFICATION DE LA BOUCLE DE TRADING (chercher où vous construisez l'observation)
# AVANT (problème):
def generate_ensemble_signal(self, market_data):
    raw_observation = self.build_observation(market_data)  # Données brutes
    signal = self.ensemble.predict(raw_observation)  # Prédiction sur données brutes
    return signal

# APRÈS (solution):
def generate_ensemble_signal(self, market_data):
    raw_observation = self.build_observation(market_data)  # Données brutes
    
    # 🔧 AJOUTER CES LIGNES:
    # Normaliser l'observation
    normalized_observation = self.normalizer.normalize(raw_observation)
    
    # Ajouter au détecteur de dérive
    self.drift_detector.add_observation(raw_observation)
    
    # Vérifier la dérive
    drift_result = self.drift_detector.check_drift(
        self.normalizer.mean,
        self.normalizer.var
    )
    
    if drift_result['drift_detected']:
        logger.warning(f"⚠️  Dérive détectée: {drift_result}")
    
    # Prédiction avec observation normalisée
    signal = self.ensemble.predict(normalized_observation)
    
    return signal

# 4. LOGGING (ajouter dans la section de logging)
def log_monitoring_status(self):
    # ... code existant ...
    
    # 🔧 AJOUTER CES LIGNES:
    logger.info(f"📊 Normalisation: {'✅ Actif' if self.normalizer.is_loaded else '❌ Inactif'}")
    
    drift_summary = self.drift_detector.get_drift_summary()
    if drift_summary['total_drifts'] > 0:
        logger.warning(f"⚠️  Dérives détectées: {drift_summary['total_drifts']}")

# ============================================================================
# 🎯 RÉSUMÉ DES CHANGEMENTS
# ============================================================================
# 1. Importer ObservationNormalizer et DriftDetector
# 2. Initialiser le normaliseur au démarrage
# 3. Normaliser les observations AVANT la prédiction
# 4. Ajouter les observations au détecteur de dérive
# 5. Logger les dérives détectées
# ============================================================================
'''
    
    return patch_code

def main():
    patch = create_patch()
    
    # Sauvegarder le patch
    patch_path = Path("/tmp/normalization_patch.py")
    with open(patch_path, 'w') as f:
        f.write(patch)
    
    print("✅ Patch créé: /tmp/normalization_patch.py")
    print("\n" + "="*70)
    print(patch)
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
