"""
Utility functions for the ADAN trading bot.
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

def _is_running_in_colab():
    """Check if the code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
        str: Absolute path to the project root directory.
    """
    # utils.py est dans src/adan_trading_bot/common/
    # Remonter de 3 niveaux pour atteindre la racine ADAN/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    # Vérification que nous sommes bien dans le bon répertoire
    # en cherchant des fichiers caractéristiques du projet ADAN
    expected_files = ["config", "src", "scripts", "data"]
    if all(os.path.exists(os.path.join(project_root, f)) for f in expected_files):
        return project_root
    
    # Fallback 1: detection Colab
    if _is_running_in_colab():
        colab_path = "/content/ADAN"
        if os.path.exists(colab_path):
            return colab_path
    
    # Fallback 2: chercher ADAN dans le chemin courant ou parent
    current_path = os.getcwd()
    path_parts = current_path.split(os.sep)
    
    # Chercher "ADAN" dans le chemin actuel
    for i, part in enumerate(path_parts):
        if part == "ADAN":
            potential_root = os.sep.join(path_parts[:i + 1])
            if all(os.path.exists(os.path.join(potential_root, f)) for f in expected_files):
                return potential_root
    
    # Fallback 3: remonter depuis le répertoire courant jusqu'à trouver un répertoire ADAN valide
    search_path = current_path
    for _ in range(10):  # Limiter la recherche à 10 niveaux
        if os.path.basename(search_path) == "ADAN":
            if all(os.path.exists(os.path.join(search_path, f)) for f in expected_files):
                return search_path
        parent = os.path.dirname(search_path)
        if parent == search_path:  # Arrivé à la racine du système
            break
        search_path = parent
    
    # Fallback 4: chercher un répertoire ADAN dans les répertoires parents
    search_path = current_path
    for _ in range(10):
        adan_candidate = os.path.join(search_path, "ADAN")
        if os.path.exists(adan_candidate) and all(os.path.exists(os.path.join(adan_candidate, f)) for f in expected_files):
            return adan_candidate
        parent = os.path.dirname(search_path)
        if parent == search_path:
            break
        search_path = parent
    
    # Si tout échoue, retourner le calcul initial et espérer que ça marche
    return project_root

def get_path(path_key, main_config_path_relative_to_root="config/main_config.yaml"):
    """
    Get an absolute path based on the project root and the path key.
    
    Args:
        path_key: Key for the path (e.g., 'data', 'models', etc.)
        main_config_path_relative_to_root: Path to main config relative to project root.
        
    Returns:
        str: Absolute path for the specified key.
    """
    project_root = get_project_root()
    
    # Construire le chemin absolu vers main_config.yaml
    full_main_config_path = os.path.join(project_root, main_config_path_relative_to_root)
    
    # Essayer de lire le nom du répertoire depuis la configuration
    try:
        with open(full_main_config_path, 'r') as f:
            main_cfg = yaml.safe_load(f)
        
        # Chercher la clé correspondante dans la section 'paths'
        dir_name_key = f"{path_key}_dir_name"  # ex: data_dir_name
        dir_name = main_cfg.get('paths', {}).get(dir_name_key, path_key)  # Fallback sur la clé
        return os.path.join(project_root, dir_name)
        
    except Exception as e:
        # Fallback: utiliser directement la clé comme nom de répertoire
        logger = logging.getLogger("adan_trading_bot")
        logger.warning(f"Could not load directory name from config for '{path_key}': {e}")
        return os.path.join(project_root, path_key)

def load_config(config_path):
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        dict: Configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_to_pickle(obj, file_path):
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save.
        file_path: Path where to save the object.
    """
    joblib.dump(obj, file_path)

def load_from_pickle(file_path):
    """
    Load an object from a pickle file.
    
    Args:
        file_path: Path to the pickle file.
        
    Returns:
        Object loaded from the pickle file.
    """
    return joblib.load(file_path)

def create_directories(*dir_paths):
    """
    Create multiple directories if they don't exist.
    
    Args:
        *dir_paths: Variable number of directory paths to create.
    """
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


def ensure_dir_exists(dir_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def timestamp_to_datetime(timestamp):
    """
    Convert a timestamp to a datetime object.
    
    Args:
        timestamp: Timestamp as an integer (milliseconds since epoch) or string.
        
    Returns:
        datetime: Datetime object.
    """
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            # Try to parse as integer
            timestamp = int(timestamp)
    
    # Assume milliseconds if timestamp is large
    if timestamp > 1e10:
        return datetime.fromtimestamp(timestamp / 1000.0)
    else:
        return datetime.fromtimestamp(timestamp)

def format_currency(value, precision=2):
    """
    Format a value as currency.
    
    Args:
        value: Value to format.
        precision: Number of decimal places.
        
    Returns:
        str: Formatted currency string.
    """
    return f"${value:.{precision}f}"

def calculate_pnl(entry_price, exit_price, quantity, is_long=True):
    """
    Calculate the PnL for a trade.
    
    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Quantity traded.
        is_long: Whether the position is long (True) or short (False).
        
    Returns:
        float: PnL amount.
    """
    if is_long:
        return (exit_price - entry_price) * quantity
    else:
        return (entry_price - exit_price) * quantity

def calculate_return_pct(entry_price, exit_price, is_long=True):
    """
    Calculate the percentage return for a trade.
    
    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        is_long: Whether the position is long (True) or short (False).
        
    Returns:
        float: Percentage return.
    """
    if is_long:
        return (exit_price - entry_price) / entry_price * 100
    else:
        return (entry_price - exit_price) / entry_price * 100

def calculate_log_return(old_value, new_value):
    """
    Calculate the logarithmic return between two values.
    
    Args:
        old_value: Old value.
        new_value: New value.
        
    Returns:
        float: Logarithmic return.
    """
    return np.log(new_value / old_value)

def get_dataframe_numeric_columns(df):
    """
    Get the numeric columns of a DataFrame.
    
    Args:
        df: Pandas DataFrame.
        
    Returns:
        list: List of numeric column names.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_logger(name=None):
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger. If None, returns the root logger.
        
    Returns:
        logging.Logger: Logger object.
    """
    if name is None:
        return logging.getLogger()
    else:
        return logging.getLogger(name)
