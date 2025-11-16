import logging
import os
from datetime import datetime

def setup_diagnostic_logging():
    """Configure le logging pour la session de diagnostic"""
    log_dir = "logs/diagnostics"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"diagnostic_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Niveau plus élevé pour les librairies externes
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('stable_baselines3').setLevel(logging.INFO)
    
    return log_file

if __name__ == "__main__":
    log_file = setup_diagnostic_logging()
    print(f"Logging configuré avec succès. Fichier: {log_file}")
