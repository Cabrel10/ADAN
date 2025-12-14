#!/usr/bin/env python3
"""
🔍 DIAGNOSTIC COMPLET DU COVARIATE SHIFT
Analyse la distribution des données live vs entraînement
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CovariateShiftDiagnostic:
    """Diagnostic du covariate shift"""
    
    def __init__(self):
        self.vecnorm_path = Path("/mnt/new_data/t10_training/checkpoints/vecnormalize.pkl")
        self.vecnorm = None
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'findings': [],
            'recommendations': []
        }
    
    def load_vecnormalize(self):
        """Charge le VecNormalize (normalisation d'entraînement)"""
        try:
            if not self.vecnorm_path.exists():
                logger.error(f"❌ Fichier non trouvé: {self.vecnorm_path}")
                return False
            
            with open(self.vecnorm_path, 'rb') as f:
                self.vecnorm = pickle.load(f)
            
            logger.info(f"✅ VecNormalize chargé")
            
            # Extraire les stats
            if hasattr(self.vecnorm, 'mean'):
                logger.info(f"   Mean shape: {self.vecnorm.mean.shape}")
                logger.info(f"   Var shape: {self.vecnorm.var.shape}")
                logger.info(f"   Mean (premiers 5): {self.vecnorm.mean[:5]}")
                logger.info(f"   Var (premiers 5): {self.vecnorm.var[:5]}")
                
                self.report['findings'].append({
                    'type': 'vecnormalize_loaded',
                    'status': 'success',
                    'mean_shape': str(self.vecnorm.mean.shape),
                    'var_shape': str(self.vecnorm.var.shape),
                    'mean_sample': self.vecnorm.mean[:5].tolist(),
                    'var_sample': self.vecnorm.var[:5].tolist()
                })
                return True
            else:
                logger.error("❌ VecNormalize n'a pas d'attribut 'mean'")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            self.report['findings'].append({
                'type': 'vecnormalize_load_error',
                'error': str(e)
            })
            return False
    
    def generate_test_observation(self):
        """Génère une observation de test (simulation)"""
        logger.info("\n📊 Génération d'une observation de test...")
        
        # Créer une observation aléatoire
        n_features = self.vecnorm.mean.shape[0] if self.vecnorm else 68
        
        # Observation brute (non normalisée) - simule des données live
        raw_obs = np.random.randn(n_features) * 10 + 50
        
        logger.info(f"   Shape: {raw_obs.shape}")
        logger.info(f"   Min: {raw_obs.min():.4f}, Max: {raw_obs.max():.4f}")
        logger.info(f"   Mean: {raw_obs.mean():.4f}, Std: {raw_obs.std():.4f}")
        
        self.report['findings'].append({
            'type': 'raw_observation',
            'shape': str(raw_obs.shape),
            'min': float(raw_obs.min()),
            'max': float(raw_obs.max()),
            'mean': float(raw_obs.mean()),
            'std': float(raw_obs.std())
        })
        
        return raw_obs
    
    def normalize_observation(self, obs):
        """Normalise une observation avec VecNormalize"""
        if not self.vecnorm:
            logger.error("❌ VecNormalize non chargé")
            return None
        
        try:
            # Utiliser la méthode normalize_obs du VecNormalize
            if hasattr(self.vecnorm, 'normalize_obs'):
                normalized = self.vecnorm.normalize_obs(obs)
            else:
                # Fallback: normalisation manuelle
                normalized = (obs - self.vecnorm.mean) / np.sqrt(self.vecnorm.var + 1e-8)
            
            logger.info(f"\n✅ Observation normalisée")
            logger.info(f"   Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
            logger.info(f"   Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
            
            self.report['findings'].append({
                'type': 'normalized_observation',
                'min': float(normalized.min()),
                'max': float(normalized.max()),
                'mean': float(normalized.mean()),
                'std': float(normalized.std())
            })
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la normalisation: {e}")
            self.report['findings'].append({
                'type': 'normalization_error',
                'error': str(e)
            })
            return None
    
    def check_distribution_shift(self, raw_obs, normalized_obs):
        """Vérifie s'il y a un shift de distribution"""
        logger.info("\n🔍 Analyse du shift de distribution...")
        
        # Vérifier si les données normalisées sont dans la plage attendue
        expected_range = (-3, 3)  # Pour des données normalisées
        
        out_of_range = np.sum((normalized_obs < expected_range[0]) | (normalized_obs > expected_range[1]))
        pct_out_of_range = (out_of_range / len(normalized_obs)) * 100
        
        logger.info(f"   Plage attendue: {expected_range}")
        logger.info(f"   Valeurs hors plage: {out_of_range}/{len(normalized_obs)} ({pct_out_of_range:.1f}%)")
        
        if pct_out_of_range > 5:
            logger.warning(f"⚠️  ALERTE: {pct_out_of_range:.1f}% des valeurs sont hors de la plage attendue!")
            self.report['recommendations'].append(
                "Les données live ont une distribution très différente des données d'entraînement. "
                "Cela peut indiquer un covariate shift significatif."
            )
        else:
            logger.info(f"✅ Distribution OK: {pct_out_of_range:.1f}% hors plage (acceptable)")
        
        self.report['findings'].append({
            'type': 'distribution_check',
            'out_of_range_count': int(out_of_range),
            'out_of_range_pct': float(pct_out_of_range),
            'status': 'warning' if pct_out_of_range > 5 else 'ok'
        })
    
    def generate_report(self):
        """Génère le rapport final"""
        logger.info("\n" + "="*70)
        logger.info("📋 RAPPORT DE DIAGNOSTIC")
        logger.info("="*70)
        
        # Sauvegarder le rapport
        report_path = Path("/tmp/covariate_shift_diagnostic.json")
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        logger.info(f"\n✅ Rapport sauvegardé: {report_path}")
        
        # Afficher les recommandations
        if self.report['recommendations']:
            logger.info("\n🎯 RECOMMANDATIONS:")
            for i, rec in enumerate(self.report['recommendations'], 1):
                logger.info(f"   {i}. {rec}")
        
        logger.info("\n" + "="*70)
        
        return self.report
    
    def run(self):
        """Exécute le diagnostic complet"""
        logger.info("🚀 Démarrage du diagnostic du covariate shift...")
        logger.info("="*70)
        
        # Étape 1: Charger VecNormalize
        if not self.load_vecnormalize():
            logger.error("❌ Impossible de charger VecNormalize")
            return False
        
        # Étape 2: Générer une observation de test
        raw_obs = self.generate_test_observation()
        
        # Étape 3: Normaliser l'observation
        normalized_obs = self.normalize_observation(raw_obs)
        if normalized_obs is None:
            logger.error("❌ Impossible de normaliser l'observation")
            return False
        
        # Étape 4: Vérifier le shift
        self.check_distribution_shift(raw_obs, normalized_obs)
        
        # Étape 5: Générer le rapport
        self.generate_report()
        
        return True

def main():
    diagnostic = CovariateShiftDiagnostic()
    success = diagnostic.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
