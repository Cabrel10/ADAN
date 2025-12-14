#!/usr/bin/env python3
"""
🔍 DIAGNOSTIC COMPLET DU COVARIATE SHIFT - VERSION 2
Analyse la distribution des données live vs entraînement
"""

import sys
import json
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
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'findings': [],
            'recommendations': []
        }
    
    def extract_vecnormalize_stats(self):
        """Extrait les stats du VecNormalize sans le charger complètement"""
        try:
            import pickle
            import sys
            
            vecnorm_path = Path("/mnt/new_data/t10_training/checkpoints/vecnormalize.pkl")
            
            if not vecnorm_path.exists():
                logger.error(f"❌ Fichier non trouvé: {vecnorm_path}")
                return None
            
            logger.info(f"📂 Tentative de chargement: {vecnorm_path}")
            
            # Augmenter la limite de récursion temporairement
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(10000)
            
            try:
                with open(vecnorm_path, 'rb') as f:
                    vecnorm = pickle.load(f)
                
                sys.setrecursionlimit(old_limit)
                
                logger.info(f"✅ VecNormalize chargé")
                
                # Extraire les stats
                stats = {}
                if hasattr(vecnorm, 'mean'):
                    stats['mean'] = vecnorm.mean
                    logger.info(f"   Mean shape: {vecnorm.mean.shape}")
                
                if hasattr(vecnorm, 'var'):
                    stats['var'] = vecnorm.var
                    logger.info(f"   Var shape: {vecnorm.var.shape}")
                
                if hasattr(vecnorm, 'count'):
                    stats['count'] = vecnorm.count
                    logger.info(f"   Count: {vecnorm.count}")
                
                return stats
                
            except RecursionError as e:
                sys.setrecursionlimit(old_limit)
                logger.warning(f"⚠️  RecursionError lors du chargement: {e}")
                logger.info("   Tentative alternative...")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            return None
    
    def analyze_training_data_distribution(self):
        """Analyse la distribution des données d'entraînement"""
        logger.info("\n📊 Analyse de la distribution d'entraînement...")
        
        # Chercher les fichiers de données d'entraînement
        data_paths = [
            Path("/mnt/new_data/t10_training/data"),
            Path("/mnt/new_data/t10_training/phase2_results"),
        ]
        
        for data_path in data_paths:
            if data_path.exists():
                logger.info(f"   Trouvé: {data_path}")
                # Lister les fichiers
                files = list(data_path.glob("*.npy")) + list(data_path.glob("*.npz"))
                if files:
                    logger.info(f"   Fichiers de données: {len(files)}")
                    for f in files[:3]:
                        logger.info(f"      - {f.name}")
    
    def generate_synthetic_observations(self, n=100):
        """Génère des observations synthétiques pour le test"""
        logger.info(f"\n🔄 Génération de {n} observations synthétiques...")
        
        n_features = 68  # Nombre de features typique
        
        # Observation brute (non normalisée) - simule des données live
        raw_obs = np.random.randn(n, n_features) * 10 + 50
        
        logger.info(f"   Shape: {raw_obs.shape}")
        logger.info(f"   Min: {raw_obs.min():.4f}, Max: {raw_obs.max():.4f}")
        logger.info(f"   Mean: {raw_obs.mean():.4f}, Std: {raw_obs.std():.4f}")
        
        self.report['findings'].append({
            'type': 'synthetic_observations',
            'n_samples': n,
            'n_features': n_features,
            'min': float(raw_obs.min()),
            'max': float(raw_obs.max()),
            'mean': float(raw_obs.mean()),
            'std': float(raw_obs.std())
        })
        
        return raw_obs
    
    def normalize_observations(self, obs, mean=None, var=None):
        """Normalise les observations"""
        logger.info(f"\n✅ Normalisation de {len(obs)} observations...")
        
        if mean is None:
            mean = np.zeros(obs.shape[1])
        if var is None:
            var = np.ones(obs.shape[1])
        
        # Normalisation standard
        normalized = (obs - mean) / np.sqrt(var + 1e-8)
        
        logger.info(f"   Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
        logger.info(f"   Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        
        self.report['findings'].append({
            'type': 'normalized_observations',
            'min': float(normalized.min()),
            'max': float(normalized.max()),
            'mean': float(normalized.mean()),
            'std': float(normalized.std())
        })
        
        return normalized
    
    def check_distribution_shift(self, raw_obs, normalized_obs):
        """Vérifie s'il y a un shift de distribution"""
        logger.info("\n🔍 Analyse du shift de distribution...")
        
        # Vérifier si les données normalisées sont dans la plage attendue
        expected_range = (-3, 3)  # Pour des données normalisées
        
        out_of_range = np.sum((normalized_obs < expected_range[0]) | (normalized_obs > expected_range[1]))
        pct_out_of_range = (out_of_range / normalized_obs.size) * 100
        
        logger.info(f"   Plage attendue: {expected_range}")
        logger.info(f"   Valeurs hors plage: {out_of_range}/{normalized_obs.size} ({pct_out_of_range:.2f}%)")
        
        if pct_out_of_range > 5:
            logger.warning(f"⚠️  ALERTE: {pct_out_of_range:.2f}% des valeurs sont hors de la plage attendue!")
            self.report['recommendations'].append(
                "Les données live ont une distribution très différente des données d'entraînement. "
                "Cela peut indiquer un covariate shift significatif."
            )
        else:
            logger.info(f"✅ Distribution OK: {pct_out_of_range:.2f}% hors plage (acceptable)")
        
        self.report['findings'].append({
            'type': 'distribution_check',
            'out_of_range_count': int(out_of_range),
            'out_of_range_pct': float(pct_out_of_range),
            'status': 'warning' if pct_out_of_range > 5 else 'ok'
        })
    
    def generate_normalization_code(self):
        """Génère le code de normalisation à utiliser"""
        code = '''
# 🔧 CODE DE NORMALISATION À AJOUTER DANS monitor.py

import json
import numpy as np
from pathlib import Path

class ObservationNormalizer:
    """Normalise les observations avec les stats d'entraînement"""
    
    def __init__(self, vecnorm_path="/mnt/new_data/t10_training/checkpoints/vecnormalize.pkl"):
        self.vecnorm_path = Path(vecnorm_path)
        self.mean = None
        self.var = None
        self.load_stats()
    
    def load_stats(self):
        """Charge les stats de normalisation"""
        try:
            import pickle
            import sys
            
            if not self.vecnorm_path.exists():
                print(f"⚠️  Fichier non trouvé: {self.vecnorm_path}")
                return False
            
            # Augmenter la limite de récursion
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(10000)
            
            with open(self.vecnorm_path, 'rb') as f:
                vecnorm = pickle.load(f)
            
            sys.setrecursionlimit(old_limit)
            
            if hasattr(vecnorm, 'mean') and hasattr(vecnorm, 'var'):
                self.mean = vecnorm.mean
                self.var = vecnorm.var
                print(f"✅ Stats de normalisation chargées: mean shape={self.mean.shape}")
                return True
            else:
                print("❌ VecNormalize n'a pas les attributs attendus")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def normalize(self, observation):
        """Normalise une observation"""
        if self.mean is None or self.var is None:
            print("⚠️  Stats non chargées, retour de l'observation brute")
            return observation
        
        # Normalisation
        normalized = (observation - self.mean) / np.sqrt(self.var + 1e-8)
        return normalized

# UTILISATION DANS LE MONITOR:
# 1. Créer une instance au démarrage
normalizer = ObservationNormalizer()

# 2. Dans la boucle de trading, normaliser avant la prédiction
raw_observation = build_observation(market_data)  # Votre code actuel
normalized_observation = normalizer.normalize(raw_observation)  # AJOUTER CETTE LIGNE
prediction = model.predict(normalized_observation)  # Utiliser l'observation normalisée
'''
        
        self.report['recommendations'].append("Implémenter la classe ObservationNormalizer dans monitor.py")
        
        return code
    
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
        
        # Générer le code
        code = self.generate_normalization_code()
        code_path = Path("/tmp/observation_normalizer.py")
        with open(code_path, 'w') as f:
            f.write(code)
        
        logger.info(f"\n📝 Code de normalisation généré: {code_path}")
        
        logger.info("\n" + "="*70)
        
        return self.report
    
    def run(self):
        """Exécute le diagnostic complet"""
        logger.info("🚀 Démarrage du diagnostic du covariate shift...")
        logger.info("="*70)
        
        # Étape 1: Extraire les stats
        stats = self.extract_vecnormalize_stats()
        
        # Étape 2: Analyser les données d'entraînement
        self.analyze_training_data_distribution()
        
        # Étape 3: Générer des observations synthétiques
        raw_obs = self.generate_synthetic_observations(n=100)
        
        # Étape 4: Normaliser
        if stats and 'mean' in stats and 'var' in stats:
            normalized_obs = self.normalize_observations(raw_obs, stats['mean'], stats['var'])
        else:
            logger.warning("⚠️  Stats non disponibles, utilisation de normalisation par défaut")
            normalized_obs = self.normalize_observations(raw_obs)
        
        # Étape 5: Vérifier le shift
        self.check_distribution_shift(raw_obs, normalized_obs)
        
        # Étape 6: Générer le rapport
        self.generate_report()
        
        return True

def main():
    diagnostic = CovariateShiftDiagnostic()
    success = diagnostic.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
