"""
ConfigWatcher - Système de surveillance et rechargement dynamique des configurations YAML.

Ce module permet de surveiller les fichiers de configuration et d'appliquer les changements
en temps réel pendant l'entraînement, sans redémarrer le processus.
"""
import os
import time
import threading
import yaml
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from datetime import datetime

from .utils import get_logger

logger = get_logger(__name__)

class ConfigFileHandler(FileSystemEventHandler):
    """Handler pour les événements de modification de fichiers de configuration."""
    
    def __init__(self, config_watcher: 'ConfigWatcher'):
        self.config_watcher = config_watcher
        
    def on_modified(self, event):
        """Appelé quand un fichier est modifié."""
        if not event.is_directory and event.src_path.endswith('.yaml'):
            logger.info(f"Configuration file modified: {event.src_path}")
            self.config_watcher._handle_config_change(event.src_path)

class ConfigWatcher:
    """
    Surveillant de configuration qui recharge automatiquement les fichiers YAML
    et applique les changements aux composants en cours d'exécution.
    """
    
    def __init__(self, config_dir: str = "config", enabled: bool = True):
        """
        Initialise le ConfigWatcher.
        
        Args:
            config_dir: Répertoire contenant les fichiers de configuration
            enabled: Active/désactive la surveillance
        """
        self.config_dir = Path(config_dir)
        self.enabled = enabled
        self.observer = None
        self.callbacks: Dict[str, List[Callable]] = {}
        self.current_configs: Dict[str, Dict[str, Any]] = {}
        self.last_reload_times: Dict[str, datetime] = {}
        
        # Fichiers de configuration surveillés
        self.watched_files = {
            'train_config.yaml': 'training',
            'environment_config.yaml': 'environment', 
            'agent_config.yaml': 'agent',
            'dbe_config.yaml': 'dbe',
            'reward_config.yaml': 'reward',
            'risk_config.yaml': 'risk'
        }
        
        # Chargement initial des configurations
        self._load_initial_configs()
        
        if self.enabled:
            self._start_watching()
            logger.info(f"🔍 ConfigWatcher started - monitoring {self.config_dir}")
        else:
            logger.info("ConfigWatcher disabled")
    
    def _load_initial_configs(self):
        """Charge toutes les configurations initiales."""
        for filename, config_type in self.watched_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        self.current_configs[config_type] = config
                        self.last_reload_times[config_type] = datetime.now()
                        logger.info(f"Loaded initial config: {filename}")
                except Exception as e:
                    logger.error(f"Failed to load initial config {filename}: {e}")
    
    def _start_watching(self):
        """Démarre la surveillance des fichiers."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory {self.config_dir} does not exist")
            return
            
        self.observer = Observer()
        event_handler = ConfigFileHandler(self)
        self.observer.schedule(event_handler, str(self.config_dir), recursive=False)
        self.observer.start()
        logger.info("File watcher started")
    
    def _handle_config_change(self, file_path: str):
        """Traite un changement de fichier de configuration."""
        try:
            file_path = Path(file_path)
            filename = file_path.name
            
            if filename not in self.watched_files:
                return
                
            config_type = self.watched_files[filename]
            
            # Petit délai pour éviter les lectures partielles
            time.sleep(0.1)
            
            # Rechargement de la configuration
            with open(file_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            old_config = self.current_configs.get(config_type, {})
            
            # Détection des changements
            changes = self._detect_changes(old_config, new_config)
            
            if changes:
                self.current_configs[config_type] = new_config
                self.last_reload_times[config_type] = datetime.now()
                
                logger.info(f"🔄 Config reloaded: {filename}")
                logger.info(f"Changes detected: {list(changes.keys())}")
                
                # Notification des callbacks
                self._notify_callbacks(config_type, new_config, changes)
            else:
                logger.debug(f"No changes detected in {filename}")
                
        except Exception as e:
            logger.error(f"Error handling config change for {file_path}: {e}")
    
    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte les changements entre deux configurations."""
        changes = {}
        
        def compare_dicts(old_dict, new_dict, path=""):
            for key, new_value in new_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in old_dict:
                    changes[current_path] = {'action': 'added', 'new_value': new_value}
                elif isinstance(new_value, dict) and isinstance(old_dict[key], dict):
                    compare_dicts(old_dict[key], new_value, current_path)
                elif old_dict[key] != new_value:
                    changes[current_path] = {
                        'action': 'modified',
                        'old_value': old_dict[key],
                        'new_value': new_value
                    }
            
            # Vérifier les clés supprimées
            for key in old_dict:
                if key not in new_dict:
                    current_path = f"{path}.{key}" if path else key
                    changes[current_path] = {'action': 'removed', 'old_value': old_dict[key]}
        
        compare_dicts(old_config, new_config)
        return changes
    
    def _notify_callbacks(self, config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]):
        """Notifie tous les callbacks enregistrés pour un type de configuration."""
        callbacks = self.callbacks.get(config_type, [])
        
        for callback in callbacks:
            try:
                callback(config_type, new_config, changes)
            except Exception as e:
                logger.error(f"Error in config callback: {e}")
    
    def register_callback(self, config_type: str, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], None]):
        """
        Enregistre un callback pour un type de configuration.
        
        Args:
            config_type: Type de configuration ('training', 'environment', etc.)
            callback: Fonction appelée lors des changements (config_type, new_config, changes)
        """
        if config_type not in self.callbacks:
            self.callbacks[config_type] = []
        
        self.callbacks[config_type].append(callback)
        logger.info(f"Callback registered for {config_type} config changes")
    
    def get_config(self, config_type: str) -> Optional[Dict[str, Any]]:
        """Récupère la configuration actuelle d'un type donné."""
        return self.current_configs.get(config_type)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Récupère toutes les configurations actuelles."""
        return self.current_configs.copy()
    
    def force_reload(self, config_type: Optional[str] = None):
        """Force le rechargement d'une configuration ou de toutes."""
        if config_type:
            # Rechargement d'un type spécifique
            for filename, file_config_type in self.watched_files.items():
                if file_config_type == config_type:
                    config_path = self.config_dir / filename
                    if config_path.exists():
                        self._handle_config_change(str(config_path))
                    break
        else:
            # Rechargement de toutes les configurations
            for filename in self.watched_files.keys():
                config_path = self.config_dir / filename
                if config_path.exists():
                    self._handle_config_change(str(config_path))
    
    def get_reload_status(self) -> Dict[str, Any]:
        """Retourne le statut des rechargements."""
        return {
            'enabled': self.enabled,
            'watched_files': list(self.watched_files.keys()),
            'loaded_configs': list(self.current_configs.keys()),
            'last_reload_times': {
                config_type: timestamp.isoformat() 
                for config_type, timestamp in self.last_reload_times.items()
            }
        }
    
    def stop(self):
        """Arrête la surveillance des fichiers."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("ConfigWatcher stopped")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Décorateur pour les méthodes qui doivent être appelées lors des changements de config
def config_reactive(config_type: str):
    """
    Décorateur pour marquer les méthodes qui réagissent aux changements de configuration.
    
    Args:
        config_type: Type de configuration à surveiller
    """
    def decorator(func):
        func._config_reactive = True
        func._config_type = config_type
        return func
    return decorator

# Exemple d'utilisation
if __name__ == "__main__":
    def on_training_config_change(config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]):
        print(f"Training config changed: {changes}")
        # Ici on pourrait ajuster les paramètres de l'agent
        if 'learning_rate' in changes:
            print(f"Learning rate changed to: {new_config.get('learning_rate')}")
    
    def on_environment_config_change(config_type: str, new_config: Dict[str, Any], changes: Dict[str, Any]):
        print(f"Environment config changed: {changes}")
        # Ici on pourrait ajuster les paramètres de l'environnement
    
    # Test du ConfigWatcher
    with ConfigWatcher("config") as watcher:
        watcher.register_callback('training', on_training_config_change)
        watcher.register_callback('environment', on_environment_config_change)
        
        print("ConfigWatcher running... Modify config files to see changes")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping ConfigWatcher...")