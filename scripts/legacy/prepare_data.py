import os
import shutil
from pathlib import Path

def prepare_data():
    # Dossier source et destination
    src_dir = Path("data/final")
    dst_dir = Path("data/new")
    
    # Créer le dossier destination s'il n'existe pas
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copier les fichiers pour chaque actif
    assets = ["BTC", "ETH", "SOL", "XRP", "ADA"]
    for asset in assets:
        src_file = src_dir / f"{asset}" / "train.parquet"
        dst_file = dst_dir / f"{asset}USDT_features.parquet"
        
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"Copié {src_file} vers {dst_file}")
        else:
            print(f"Attention: {src_file} n'existe pas")

if __name__ == "__main__":
    prepare_data()
