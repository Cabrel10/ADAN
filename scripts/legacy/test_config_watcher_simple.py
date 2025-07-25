#!/usr/bin/env python3
"""
Test simple du ConfigWatcher pour vérifier le fonctionnement de base.
"""

import os
import sys
import time
import yaml
import threading
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.common.config_watcher import ConfigWatcher
from adan_trading_bot.common.utils import get_logger

logger = get_logger(__name__)

def test_callback(config_type: str, new_config: dict, changes: dict):
    """Callback de test pour les changements de configuration."""
    logger.info(f"🔄 Config changed: {config_type}")
    logger.info(f"Changes: {list(changes.keys())}")
    for key, change in changes.items():
        if 'old_value' in change and 'new_value' in change:
            logger.info(f"  {key}: {change['old_value']} → {change['new_value']}")

def modify_config_after_delay():
    """Modifie une configuration après un délai."""
    time.sleep(3)
    
    config_path = Path("config/dbe_config.yaml")
    if config_path.exists():
        logger.info("🔧 Modifying DBE config...")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Modifier un paramètre
        old_value = config['risk_parameters']['base_sl_pct']
        new_value = 0.025
        config['risk_parameters']['base_sl_pct'] = new_value
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Modified base_sl_pct: {old_value} → {new_value}")
        logger.info(f"✅ Modified base_sl_pct: {old_value} → {new_value}")
        
        # Restaurer après 2 secondes
        time.sleep(2)
        config['risk_parameters']['base_sl_pct'] = old_value
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"🔄 Restored base_sl_pct to {old_value}")

def main():
    """Test principal du ConfigWatcher."""
    print("🚀 Testing ConfigWatcher...")
    logger.info("🚀 Testing ConfigWatcher...")
    
    try:
        # Créer le ConfigWatcher
        print("📁 Creating ConfigWatcher...")
        watcher = ConfigWatcher("config", enabled=True)
        print(f"✅ ConfigWatcher created, status: {watcher.get_reload_status()}")
        
        # Enregistrer un callback
        print("📝 Registering callback...")
        watcher.register_callback('dbe', test_callback)
        
        # Lancer la modification en arrière-plan
        print("🔧 Starting config modifier thread...")
        modifier_thread = threading.Thread(target=modify_config_after_delay)
        modifier_thread.daemon = True
        modifier_thread.start()
        
        # Attendre et observer
        print("⏳ Waiting for config changes...")
        logger.info("⏳ Waiting for config changes...")
        time.sleep(8)
        
        # Arrêter le watcher
        print("🛑 Stopping watcher...")
        watcher.stop()
        
        print("✅ ConfigWatcher test completed!")
        logger.info("✅ ConfigWatcher test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"❌ Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()