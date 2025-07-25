#!/usr/bin/env python3
"""
Script de test pour valider le reload dynamique des configurations YAML.

Ce script lance un entraînement court et modifie les configurations pendant
l'exécution pour vérifier que les changements sont appliqués à chaud.
"""

import os
import sys
import time
import yaml
import threading
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.training.training_orchestrator import TrainingOrchestrator
from adan_trading_bot.common.utils import get_logger, load_config
from stable_baselines3 import PPO

logger = get_logger(__name__)

class ConfigModifier:
    """Classe pour modifier les configurations pendant l'entraînement."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.original_configs = {}
        self.modifications_applied = []
        
    def backup_configs(self):
        """Sauvegarde les configurations originales."""
        config_files = [
            'train_config.yaml',
            'dbe_config.yaml',
            'environment_config.yaml'
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.original_configs[config_file] = yaml.safe_load(f)
                logger.info(f"📋 Backed up {config_file}")
    
    def restore_configs(self):
        """Restaure les configurations originales."""
        for config_file, original_config in self.original_configs.items():
            config_path = self.config_dir / config_file
            with open(config_path, 'w') as f:
                yaml.dump(original_config, f, default_flow_style=False)
            logger.info(f"🔄 Restored {config_file}")
    
    def modify_training_config(self, delay: float = 5.0):
        """Modifie la configuration d'entraînement après un délai."""
        def _modify():
            time.sleep(delay)
            logger.info("🔧 Modifying training config...")
            
            config_path = self.config_dir / 'train_config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Modifier le learning rate
                old_lr = config.get('learning_rate', 3e-4)
                new_lr = old_lr * 0.5  # Réduire de moitié
                config['learning_rate'] = new_lr
                
                # Modifier l'entropy coefficient
                old_ent_coef = config.get('ent_coef', 0.0)
                new_ent_coef = 0.01  # Augmenter l'exploration
                config['ent_coef'] = new_ent_coef
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.modifications_applied.append({
                    'file': 'train_config.yaml',
                    'changes': {
                        'learning_rate': {'old': old_lr, 'new': new_lr},
                        'ent_coef': {'old': old_ent_coef, 'new': new_ent_coef}
                    },
                    'timestamp': time.time()
                })
                
                logger.info(f"✅ Training config modified: LR {old_lr} → {new_lr}, EntCoef {old_ent_coef} → {new_ent_coef}")
        
        thread = threading.Thread(target=_modify)
        thread.daemon = True
        thread.start()
        return thread
    
    def modify_dbe_config(self, delay: float = 10.0):
        """Modifie la configuration DBE après un délai."""
        def _modify():
            time.sleep(delay)
            logger.info("🔧 Modifying DBE config...")
            
            config_path = self.config_dir / 'dbe_config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Modifier les paramètres de risque
                risk_params = config.get('risk_parameters', {})
                old_sl = risk_params.get('base_sl_pct', 0.02)
                new_sl = 0.015  # SL plus serré
                risk_params['base_sl_pct'] = new_sl
                
                old_tp = risk_params.get('base_tp_pct', 0.04)
                new_tp = 0.06  # TP plus large
                risk_params['base_tp_pct'] = new_tp
                
                config['risk_parameters'] = risk_params
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.modifications_applied.append({
                    'file': 'dbe_config.yaml',
                    'changes': {
                        'risk_parameters.base_sl_pct': {'old': old_sl, 'new': new_sl},
                        'risk_parameters.base_tp_pct': {'old': old_tp, 'new': new_tp}
                    },
                    'timestamp': time.time()
                })
                
                logger.info(f"✅ DBE config modified: SL {old_sl} → {new_sl}, TP {old_tp} → {new_tp}")
        
        thread = threading.Thread(target=_modify)
        thread.daemon = True
        thread.start()
        return thread
    
    def modify_environment_config(self, delay: float = 15.0):
        """Modifie la configuration d'environnement après un délai."""
        def _modify():
            time.sleep(delay)
            logger.info("🔧 Modifying environment config...")
            
            config_path = self.config_dir / 'environment_config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Modifier les paramètres de modulation du risque
                risk_mod = config.get('risk_modulation', {})
                old_threshold = risk_mod.get('drawdown_threshold_defensif', 15.0)
                new_threshold = 12.0  # Seuil plus strict
                risk_mod['drawdown_threshold_defensif'] = new_threshold
                
                config['risk_modulation'] = risk_mod
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.modifications_applied.append({
                    'file': 'environment_config.yaml',
                    'changes': {
                        'risk_modulation.drawdown_threshold_defensif': {'old': old_threshold, 'new': new_threshold}
                    },
                    'timestamp': time.time()
                })
                
                logger.info(f"✅ Environment config modified: Drawdown threshold {old_threshold} → {new_threshold}")
        
        thread = threading.Thread(target=_modify)
        thread.daemon = True
        thread.start()
        return thread

def create_test_config():
    """Crée une configuration de test pour l'orchestrateur."""
    return {
        "num_environments": 1,
        "curriculum_learning": False,
        "shared_experience_buffer": False,
        "dynamic_adaptation": {"enabled": True},
        "config_dir": "config",
        "environment_config": {
            "data": {
                "data_dir": "data/final",
                "chunk_size": 1000,
                "assets": ["BTC"]
            },
            "environment": {"initial_capital": 100.0},
            "portfolio": {},
            "trading": {},
            "state": {
                "window_size": 10,
                "timeframes": ["5m"],
                "features_per_timeframe": {"5m": ["open", "high", "low", "close", "volume"]}
            }
        }
    }

def create_test_agent_config():
    """Crée une configuration d'agent de test."""
    return {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 100,  # Petit pour test rapide
        "batch_size": 32,
        "n_epochs": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "verbose": 1,
        "device": "cpu"
    }

def main():
    """Fonction principale du test."""
    logger.info("🚀 Starting dynamic config reload test...")
    
    # Créer le modificateur de config
    config_modifier = ConfigModifier()
    
    try:
        # Sauvegarder les configs originales
        config_modifier.backup_configs()
        
        # Créer les configurations de test
        orchestrator_config = create_test_config()
        agent_config = create_test_agent_config()
        
        # Créer l'orchestrateur avec ConfigWatcher
        logger.info("📊 Creating TrainingOrchestrator with ConfigWatcher...")
        orchestrator = TrainingOrchestrator(
            config=orchestrator_config,
            agent_class=PPO,
            agent_config=agent_config
        )
        
        # Programmer les modifications de config
        logger.info("⏰ Scheduling config modifications...")
        thread1 = config_modifier.modify_training_config(delay=3.0)
        thread2 = config_modifier.modify_dbe_config(delay=6.0)
        thread3 = config_modifier.modify_environment_config(delay=9.0)
        
        # Lancer l'entraînement court
        logger.info("🎯 Starting short training session...")
        total_timesteps = 500  # Entraînement très court pour le test
        orchestrator.train_agent(total_timesteps=total_timesteps)
        
        # Attendre que toutes les modifications soient appliquées
        logger.info("⏳ Waiting for all modifications to complete...")
        thread1.join()
        thread2.join()
        thread3.join()
        
        # Afficher le résumé des modifications
        logger.info("📋 Configuration modifications summary:")
        for i, mod in enumerate(config_modifier.modifications_applied, 1):
            logger.info(f"  {i}. {mod['file']}:")
            for change_key, change_data in mod['changes'].items():
                logger.info(f"     - {change_key}: {change_data['old']} → {change_data['new']}")
        
        # Fermer l'orchestrateur
        orchestrator.close()
        
        logger.info("✅ Dynamic config reload test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        
    finally:
        # Restaurer les configurations originales
        logger.info("🔄 Restoring original configurations...")
        config_modifier.restore_configs()
        logger.info("✅ Original configurations restored")

if __name__ == "__main__":
    main()