import pandas as pd
from pathlib import Path
import logging

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemin absolu vers le répertoire du script
SCRIPT_DIR = Path(__file__).parent
# Chemin absolu vers les données traitées
PROCESSED_DIR = SCRIPT_DIR.parent / 'data' / 'processed'
# Chemin absolu vers le répertoire de sortie
FINAL_DIR = SCRIPT_DIR.parent / 'data' / 'final'
ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
TIMEFRAMES = ['5m', '1h', '4h']
SPLITS = ['train', 'val', 'test']

FINAL_DIR.mkdir(parents=True, exist_ok=True)

def merge_timeframes(asset_data: dict):
    """Fusionne les données de différents timeframes en un seul DataFrame."""
    try:
        logging.info("Starting merge_timeframes")
        # Trier les timeframes du plus fréquent au moins fréquent
        logging.info(f"  Debug - Asset data keys: {asset_data.keys()}")
        
        # Vérifier que tous les DataFrames ont des données
        for tf, df in list(asset_data.items()):
            if df is None or df.empty:
                logging.warning(f"  Warning: No data for timeframe {tf}, skipping")
                del asset_data[tf]
        
        if not asset_data:
            logging.error("  Error: No valid data to merge")
            return None
            
        for tf, df in asset_data.items():
            logging.info(f"  Debug - {tf} data shape: {df.shape}")
            logging.info(f"  Debug - {tf} index type: {type(df.index)}")
            logging.info(f"  Debug - {tf} index name: {df.index.name}")
            logging.info(f"  Debug - {tf} index values[:5]: {df.index[:5] if len(df) > 0 else 'empty'}")
        
        timeframes = list(asset_data.keys())
        logging.info(f"  Debug - Timeframes to process: {timeframes}")
        
        # Trier les timeframes par fréquence (du plus fréquent au moins fréquent)
        base_tf = sorted(timeframes, key=lambda x: pd.to_timedelta(x.replace('m', 'min').replace('h', 'hour')))[0]
        logging.info(f"  Debug - Selected base timeframe: {base_tf}")
        
        if base_tf not in asset_data:
            logging.error(f"  Error: Base timeframe {base_tf} not found in asset data")
            return None
            
        merged_df = asset_data[base_tf].copy()
        
    except Exception as e:
        logging.error(f"  Error in merge_timeframes: {str(e)}", exc_info=True)
        return None
    
    for tf, df in asset_data.items():
        if tf == base_tf:
            continue
        merged_df = pd.merge_asof(merged_df, df, on='timestamp', direction='backward')
    
    return merged_df

for asset in ASSETS:
    # Extraire le symbole de l'actif (par exemple, 'BTC' à partir de 'BTC/USDT')
    asset_symbol = asset.split('/')[0]
    asset_dir = asset_symbol  # Utiliser uniquement le symbole pour le dossier
    logging.info(f'Processing asset: {asset} (directory: {asset_dir})')
    (FINAL_DIR / asset_dir).mkdir(exist_ok=True, parents=True)
    
    for split in SPLITS:
        logging.info(f'  Processing split: {split}')
        
        asset_data_for_split = {}
        for tf in TIMEFRAMES:
            # Utiliser uniquement le symbole pour le chemin du fichier
            file_path = PROCESSED_DIR / asset_symbol / f'{asset_symbol}_{tf}_{split}.parquet'
            if file_path.exists():
                logging.info(f'    Loading {file_path}')
                asset_data_for_split[tf] = pd.read_parquet(file_path)
            else:
                logging.warning(f'    File not found: {file_path}')
        
        if not asset_data_for_split:
            logging.warning(f'  No data to merge for {asset} {split}')
            continue
            
        merged_df = merge_timeframes(asset_data_for_split)
        # Créer le dossier de sortie s'il n'existe pas
        output_dir = FINAL_DIR / asset.split('/')[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemin de sortie pour le fichier fusionné
        output_path = output_dir / f'{asset.split("/")[0]}_{split}.parquet'
        merged_df.to_parquet(output_path)
        logging.info(f'  ==> Saved merged file to {output_path}')

logging.info('All assets processed.')
