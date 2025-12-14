#!/usr/bin/env python3
"""
ADAN Health Diagnostic - Vérifie l'état complet du système
- Stress des modèles (GPU/CPU/Memory)
- Ouverture/Fermeture des positions
- Transmission des indicateurs
- Confusion CNN/PPO
"""

import os
import sys
import json
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ADANHealthDiagnostic:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_stress': {},
            'position_management': {},
            'indicator_transmission': {},
            'model_confusion': {},
            'overall_status': 'UNKNOWN'
        }
    
    # ============================================================================
    # 1. VÉRIFIER LE STRESS DES MODÈLES
    # ============================================================================
    
    def check_model_stress(self):
        """Vérifie si les modèles sont stressés"""
        logger.info("\n🔍 VÉRIFICATION 1: STRESS DES MODÈLES")
        logger.info("=" * 60)
        
        stress_data = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids()),
            'status': 'OK'
        }
        
        # Vérifier GPU si disponible
        try:
            import torch
            if torch.cuda.is_available():
                stress_data['gpu_available'] = True
                stress_data['gpu_count'] = torch.cuda.device_count()
                stress_data['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                stress_data['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"✅ GPU Disponible: {torch.cuda.device_count()} GPU(s)")
                logger.info(f"   Mémoire allouée: {stress_data['gpu_memory_allocated_gb']:.2f} GB")
                logger.info(f"   Mémoire réservée: {stress_data['gpu_memory_reserved_gb']:.2f} GB")
            else:
                stress_data['gpu_available'] = False
                logger.warning("⚠️  GPU non disponible (CPU only)")
        except Exception as e:
            logger.warning(f"⚠️  Impossible de vérifier GPU: {e}")
            stress_data['gpu_available'] = False
        
        # Vérifier les seuils
        if stress_data['cpu_percent'] > 80:
            stress_data['status'] = 'HIGH_CPU'
            logger.warning(f"⚠️  CPU élevé: {stress_data['cpu_percent']:.1f}%")
        
        if stress_data['memory_percent'] > 80:
            stress_data['status'] = 'HIGH_MEMORY'
            logger.warning(f"⚠️  Mémoire élevée: {stress_data['memory_percent']:.1f}%")
        
        if stress_data['disk_percent'] > 90:
            stress_data['status'] = 'DISK_FULL'
            logger.warning(f"⚠️  Disque presque plein: {stress_data['disk_percent']:.1f}%")
        
        logger.info(f"📊 CPU: {stress_data['cpu_percent']:.1f}%")
        logger.info(f"📊 Mémoire: {stress_data['memory_percent']:.1f}% ({stress_data['memory_used_gb']:.1f}/{psutil.virtual_memory().total/(1024**3):.1f} GB)")
        logger.info(f"📊 Disque: {stress_data['disk_percent']:.1f}%")
        logger.info(f"📊 Processus: {stress_data['process_count']}")
        logger.info(f"✅ Status: {stress_data['status']}")
        
        self.results['model_stress'] = stress_data
        return stress_data['status'] == 'OK'
    
    # ============================================================================
    # 2. VÉRIFIER OUVERTURE/FERMETURE DES POSITIONS
    # ============================================================================
    
    def check_position_management(self):
        """Vérifie si les positions s'ouvrent et se ferment correctement"""
        logger.info("\n🔍 VÉRIFICATION 2: GESTION DES POSITIONS")
        logger.info("=" * 60)
        
        position_data = {
            'trades_opened': 0,
            'trades_closed': 0,
            'trades_with_tp_sl': 0,
            'average_hold_time': 0,
            'status': 'OK'
        }
        
        try:
            # Vérifier les logs du monitor
            log_file = Path("paper_trading.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = f.readlines()
                
                # Compter les trades ouverts/fermés
                for line in logs[-1000:]:  # Derniers 1000 logs
                    if "Trade Exécuté" in line:
                        position_data['trades_opened'] += 1
                    if "Position fermée" in line:
                        position_data['trades_closed'] += 1
                    if "TP:" in line and "SL:" in line:
                        position_data['trades_with_tp_sl'] += 1
                
                logger.info(f"✅ Trades ouverts: {position_data['trades_opened']}")
                logger.info(f"✅ Trades fermés: {position_data['trades_closed']}")
                logger.info(f"✅ Trades avec TP/SL: {position_data['trades_with_tp_sl']}")
                
                # Vérifier le ratio
                if position_data['trades_opened'] > 0:
                    close_ratio = position_data['trades_closed'] / position_data['trades_opened']
                    logger.info(f"📊 Ratio fermeture: {close_ratio:.2%}")
                    
                    if close_ratio < 0.5:
                        position_data['status'] = 'POSITIONS_NOT_CLOSING'
                        logger.warning("⚠️  Beaucoup de positions ouvertes non fermées!")
                
                # Vérifier que TP/SL sont définis
                if position_data['trades_opened'] > 0 and position_data['trades_with_tp_sl'] == 0:
                    position_data['status'] = 'NO_TP_SL'
                    logger.warning("⚠️  Aucun TP/SL détecté!")
            else:
                logger.warning("⚠️  Fichier log non trouvé")
                position_data['status'] = 'NO_LOG'
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification: {e}")
            position_data['status'] = 'ERROR'
        
        self.results['position_management'] = position_data
        return position_data['status'] == 'OK'
    
    # ============================================================================
    # 3. VÉRIFIER TRANSMISSION DES INDICATEURS
    # ============================================================================
    
    def check_indicator_transmission(self):
        """Vérifie que les indicateurs sont transmis correctement"""
        logger.info("\n🔍 VÉRIFICATION 3: TRANSMISSION DES INDICATEURS")
        logger.info("=" * 60)
        
        indicator_data = {
            'indicators_detected': [],
            'timeframes_detected': [],
            'observation_shape_valid': False,
            'status': 'OK'
        }
        
        try:
            # Vérifier les indicateurs dans les logs
            log_file = Path("paper_trading.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = f.readlines()
                
                # Chercher les indicateurs
                indicators = ['rsi', 'macd', 'bb', 'atr', 'adx', 'stoch', 'ichimoku']
                timeframes = ['5m', '1h', '4h']
                
                for line in logs[-500:]:
                    for indicator in indicators:
                        if indicator.lower() in line.lower():
                            if indicator not in indicator_data['indicators_detected']:
                                indicator_data['indicators_detected'].append(indicator)
                    
                    for tf in timeframes:
                        if tf in line:
                            if tf not in indicator_data['timeframes_detected']:
                                indicator_data['timeframes_detected'].append(tf)
                
                logger.info(f"✅ Indicateurs détectés: {indicator_data['indicators_detected']}")
                logger.info(f"✅ Timeframes détectés: {indicator_data['timeframes_detected']}")
                
                # Vérifier la forme de l'observation
                for line in logs[-100:]:
                    if "Built observation" in line:
                        # Chercher les dimensions
                        if "(20, 14)" in line:
                            indicator_data['observation_shape_valid'] = True
                            logger.info("✅ Forme d'observation valide: (20, 14)")
                        else:
                            logger.warning("⚠️  Forme d'observation invalide!")
                            indicator_data['status'] = 'INVALID_SHAPE'
                
                # Vérifier que tous les indicateurs sont présents
                if len(indicator_data['indicators_detected']) < 3:
                    indicator_data['status'] = 'MISSING_INDICATORS'
                    logger.warning(f"⚠️  Seulement {len(indicator_data['indicators_detected'])} indicateurs détectés")
                
                if len(indicator_data['timeframes_detected']) < 3:
                    indicator_data['status'] = 'MISSING_TIMEFRAMES'
                    logger.warning(f"⚠️  Seulement {len(indicator_data['timeframes_detected'])} timeframes détectés")
            
            else:
                logger.warning("⚠️  Fichier log non trouvé")
                indicator_data['status'] = 'NO_LOG'
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification: {e}")
            indicator_data['status'] = 'ERROR'
        
        self.results['indicator_transmission'] = indicator_data
        return indicator_data['status'] == 'OK'
    
    # ============================================================================
    # 4. VÉRIFIER CONFUSION CNN/PPO
    # ============================================================================
    
    def check_model_confusion(self):
        """Vérifie que CNN et PPO ne confondent pas les indicateurs"""
        logger.info("\n🔍 VÉRIFICATION 4: CONFUSION CNN/PPO")
        logger.info("=" * 60)
        
        confusion_data = {
            'worker_signals_consistent': True,
            'signal_variance': 0,
            'signal_distribution': {},
            'status': 'OK'
        }
        
        try:
            # Vérifier la cohérence des signaux
            log_file = Path("paper_trading.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = f.readlines()
                
                signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                worker_signals = {}
                
                for line in logs[-500:]:
                    if "Ensemble:" in line:
                        if "BUY" in line:
                            signals['BUY'] += 1
                        elif "SELL" in line:
                            signals['SELL'] += 1
                        else:
                            signals['HOLD'] += 1
                    
                    # Vérifier les signaux des workers
                    for worker in ['w1', 'w2', 'w3', 'w4']:
                        if f"{worker}:" in line:
                            if worker not in worker_signals:
                                worker_signals[worker] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                            
                            if "BUY" in line:
                                worker_signals[worker]['BUY'] += 1
                            elif "SELL" in line:
                                worker_signals[worker]['SELL'] += 1
                            else:
                                worker_signals[worker]['HOLD'] += 1
                
                confusion_data['signal_distribution'] = signals
                
                logger.info(f"✅ Distribution des signaux:")
                logger.info(f"   BUY: {signals['BUY']}")
                logger.info(f"   SELL: {signals['SELL']}")
                logger.info(f"   HOLD: {signals['HOLD']}")
                
                # Vérifier la variance
                total_signals = sum(signals.values())
                if total_signals > 0:
                    variance = np.var([signals['BUY'], signals['SELL'], signals['HOLD']])
                    confusion_data['signal_variance'] = float(variance)
                    logger.info(f"📊 Variance des signaux: {variance:.2f}")
                    
                    # Si variance très basse, les modèles confondent peut-être
                    if variance < 1 and total_signals > 10:
                        confusion_data['status'] = 'LOW_VARIANCE'
                        logger.warning("⚠️  Variance très basse - possible confusion CNN/PPO!")
                
                # Vérifier la cohérence entre workers
                if worker_signals:
                    logger.info(f"✅ Signaux par worker:")
                    for worker, sigs in worker_signals.items():
                        logger.info(f"   {worker}: BUY={sigs['BUY']}, SELL={sigs['SELL']}, HOLD={sigs['HOLD']}")
                        
                        # Vérifier que chaque worker a une distribution variée
                        worker_variance = np.var([sigs['BUY'], sigs['SELL'], sigs['HOLD']])
                        if worker_variance < 0.5 and sum(sigs.values()) > 5:
                            confusion_data['worker_signals_consistent'] = False
                            logger.warning(f"⚠️  {worker} a une variance très basse!")
            
            else:
                logger.warning("⚠️  Fichier log non trouvé")
                confusion_data['status'] = 'NO_LOG'
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification: {e}")
            confusion_data['status'] = 'ERROR'
        
        self.results['model_confusion'] = confusion_data
        return confusion_data['status'] == 'OK'
    
    # ============================================================================
    # RAPPORT FINAL
    # ============================================================================
    
    def generate_report(self):
        """Génère un rapport final"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 RAPPORT FINAL - ADAN HEALTH DIAGNOSTIC")
        logger.info("=" * 60)
        
        # Résumé
        checks = [
            ("Stress des modèles", self.results['model_stress'].get('status', 'UNKNOWN')),
            ("Gestion des positions", self.results['position_management'].get('status', 'UNKNOWN')),
            ("Transmission des indicateurs", self.results['indicator_transmission'].get('status', 'UNKNOWN')),
            ("Confusion CNN/PPO", self.results['model_confusion'].get('status', 'UNKNOWN'))
        ]
        
        all_ok = True
        for check_name, status in checks:
            icon = "✅" if status == 'OK' else "⚠️ " if status != 'ERROR' else "❌"
            logger.info(f"{icon} {check_name}: {status}")
            if status != 'OK':
                all_ok = False
        
        # Status global
        self.results['overall_status'] = 'HEALTHY' if all_ok else 'NEEDS_ATTENTION'
        logger.info(f"\n🎯 Status Global: {self.results['overall_status']}")
        
        # Sauvegarder le rapport
        report_file = Path("diagnostic_report.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n📁 Rapport sauvegardé: {report_file}")
        
        return self.results
    
    def run_full_diagnostic(self):
        """Exécute le diagnostic complet"""
        logger.info("\n🚀 DÉMARRAGE DU DIAGNOSTIC ADAN")
        logger.info("=" * 60)
        
        self.check_model_stress()
        self.check_position_management()
        self.check_indicator_transmission()
        self.check_model_confusion()
        
        return self.generate_report()

if __name__ == "__main__":
    diagnostic = ADANHealthDiagnostic()
    results = diagnostic.run_full_diagnostic()
    
    # Exit code basé sur le status
    sys.exit(0 if results['overall_status'] == 'HEALTHY' else 1)
