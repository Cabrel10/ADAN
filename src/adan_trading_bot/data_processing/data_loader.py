"""
Data loading utilities for the ADAN trading bot.
"""
import os
import pandas as pd
import numpy as np
from ..common.utils import get_path, get_logger

logger = get_logger()

def load_data_from_parquet(file_path, start_date=None, end_date=None):
    """
    Load data from a Parquet file.
    
    Args:
        file_path: Path to the Parquet file.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Apply date filters if provided
        if start_date or end_date:
            df = filter_by_date(df, start_date, end_date)
            
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def load_data_from_csv(file_path, start_date=None, end_date=None, **kwargs):
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        **kwargs: Additional arguments to pass to pd.read_csv.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Apply date filters if provided
        if start_date or end_date:
            df = filter_by_date(df, start_date, end_date)
            
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def filter_by_date(df, start_date=None, end_date=None, timestamp_col='timestamp'):
    """
    Filter a DataFrame by date range.
    
    Args:
        df: DataFrame to filter.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        timestamp_col: Name of the timestamp column.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Make a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Ensure timestamp column is datetime
    if timestamp_col in filtered_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[timestamp_col]):
            filtered_df[timestamp_col] = pd.to_datetime(filtered_df[timestamp_col])
    
        # Apply filters
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df[timestamp_col] >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df[timestamp_col] <= end_date]
    
    return filtered_df

def load_data_from_config(config):
    """
    Load data based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    data_config = config.get('data', {})
    data_source = data_config.get('source', {})
    source_type = data_source.get('type', 'parquet')
    file_path = data_source.get('path', '')
    
    # Get absolute path
    if not os.path.isabs(file_path):
        file_path = os.path.join(get_path('data'), file_path)
    
    # Get date filters
    start_date = data_config.get('start_date', None)
    end_date = data_config.get('end_date', None)
    
    # Load data based on source type
    if source_type.lower() == 'parquet':
        return load_data_from_parquet(file_path, start_date, end_date)
    elif source_type.lower() == 'csv':
        return load_data_from_csv(file_path, start_date, end_date)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")

def load_multiple_assets(config):
    """
    Load data for multiple assets based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        dict: Dictionary mapping asset names to DataFrames.
    """
    assets_config = config.get('assets', {})
    symbols = assets_config.get('symbols', [])
    data_dir = get_path('data')
    
    assets_data = {}
    
    for symbol in symbols:
        # Construct file path based on symbol
        file_name = f"{symbol}USDT_{assets_config.get('timeframe', '1h')}.parquet"
        file_path = os.path.join(data_dir, 'processed', file_name)
        
        try:
            assets_data[symbol] = load_data_from_parquet(
                file_path,
                assets_config.get('start_date'),
                assets_config.get('end_date')
            )
        except Exception as e:
            logger.warning(f"Could not load data for {symbol}: {e}")
    
    return assets_data

def merge_assets_data(assets_data, timestamp_col='timestamp'):
    """
    Merge data for multiple assets into a single DataFrame.
    
    Args:
        assets_data: Dictionary mapping asset names to DataFrames.
        timestamp_col: Name of the timestamp column.
        
    Returns:
        pd.DataFrame: Merged DataFrame with an additional 'pair' column.
    """
    merged_data = []
    
    for asset, df in assets_data.items():
        # Add pair column
        df_copy = df.copy()
        df_copy['pair'] = f"{asset}USDT"
        merged_data.append(df_copy)
    
    # Concatenate all DataFrames
    if merged_data:
        return pd.concat(merged_data, ignore_index=True)
    else:
        return pd.DataFrame()

def transform_to_multi_asset_format(df, assets):
    """
    Transforme un DataFrame avec des colonnes génériques (open, high, low, close, etc.)
    en un DataFrame avec des colonnes au format {feature}_{asset} (open_ADAUSDT, close_ADAUSDT, etc.)
    
    Args:
        df: DataFrame à transformer
        assets: Liste des actifs à inclure
        
    Returns:
        pd.DataFrame: DataFrame transformé avec colonnes au format {feature}_{asset}
    """
    if df.empty:
        logger.warning("DataFrame vide, impossible de transformer au format multi-actif")
        return df
    
    # Vérifier si le DataFrame est déjà au format multi-actif
    # en cherchant des colonnes comme 'open_ADAUSDT'
    asset_columns = [col for col in df.columns if any(f"_{asset}" in col for asset in assets)]
    if asset_columns:
        logger.info(f"DataFrame déjà au format multi-actif avec {len(asset_columns)} colonnes d'actifs")
        return df
    
    # Vérifier si le DataFrame a une colonne 'asset' qui indique l'actif pour chaque ligne
    if 'asset' in df.columns:
        logger.info("Transformation du DataFrame au format multi-actif en utilisant la colonne 'asset'")
        
        # Créer un DataFrame vide pour stocker les données transformées
        transformed_dfs = []
        
        # Pour chaque actif, filtrer les lignes correspondantes et renommer les colonnes
        for asset in assets:
            # Filtrer les lignes pour cet actif
            asset_df = df[df['asset'] == asset].copy()
            if asset_df.empty:
                logger.warning(f"Aucune donnée trouvée pour l'actif {asset}")
                continue
            
            # Renommer les colonnes pour inclure le nom de l'actif
            renamed_columns = {}
            for col in asset_df.columns:
                if col != 'asset' and col != 'timestamp':
                    renamed_columns[col] = f"{col}_{asset}"
            
            asset_df = asset_df.rename(columns=renamed_columns)
            transformed_dfs.append(asset_df)
        
        if not transformed_dfs:
            logger.error("Aucun DataFrame transformé n'a été créé")
            return df
        
        # Fusionner tous les DataFrames transformés
        result_df = pd.concat(transformed_dfs, axis=1)
        logger.info(f"DataFrame transformé avec succès. Nouvelles dimensions: {result_df.shape}")
        logger.info(f"Nouvelles colonnes (premières 20): {result_df.columns.tolist()[:20]}")
        return result_df
    
    # Si pas de colonne 'asset', mais qu'on a des colonnes de base comme 'open', 'close', etc.
    # Nous supposons que le DataFrame contient des données pour un seul actif ou des données génériques
    base_columns = ['open', 'high', 'low', 'close', 'volume']
    if any(col in df.columns for col in base_columns):
        logger.info("Transformation du DataFrame générique au format multi-actif pour tous les actifs")
        
        # Créer un nouveau DataFrame avec les colonnes renommées pour chaque actif
        new_df = pd.DataFrame(index=df.index)
        
        # Conserver les colonnes de temps/index si elles existent
        timestamp_cols = ['timestamp', 'date', 'datetime']
        for col in timestamp_cols:
            if col in df.columns:
                new_df[col] = df[col]
        
        # Pour chaque actif, dupliquer toutes les colonnes avec le suffixe de l'actif
        for asset in assets:
            for col in df.columns:
                if col not in timestamp_cols:
                    new_df[f"{col}_{asset}"] = df[col]
        
        logger.info(f"DataFrame générique transformé en format multi-actif. Nouvelles dimensions: {new_df.shape}")
        logger.info(f"Nouvelles colonnes (premières 20): {new_df.columns.tolist()[:20]}")
        return new_df
    
    logger.warning("Impossible de transformer le DataFrame au format multi-actif: format non reconnu")
    return df

def load_merged_data(config, split_type='train'):
    """
    Load pre-merged data for a specific split type (train, val, test).
    
    Args:
        config: Configuration dictionary.
        split_type: Type of split to load ('train', 'val', 'test').
        
    Returns:
        pd.DataFrame: Loaded merged data.
    """
    import os  # Import explicite pour éviter l'erreur 'cannot access local variable'
    
    logger.info(f"load_merged_data: Début pour split_type: {split_type}")

    # 1. Accéder à la configuration des chemins via config['paths'] (structure combinée de trainer.py)
    try:
        project_root = config['paths']['base_project_dir_local']
        data_dir_name = config['paths']['data_dir_name']
    except KeyError as e:
        logger.critical(f"load_merged_data: Configuration manquante dans config['paths']: {e}")
        raise ValueError(f"Configuration 'paths' manquante ou incomplète: {e}")

    if not project_root or not os.path.isdir(project_root):
        logger.critical(f"load_merged_data: Chemin 'base_project_dir_local' invalide ou non trouvé: '{project_root}'.")
        raise ValueError(f"Chemin 'base_project_dir_local' ({project_root}) est manquant ou invalide. Vérifiez config/main_config.yaml.")

    # 2. Accéder à la configuration spécifique aux données (data_config_cpu.yaml ou _gpu.yaml)
    data_cfg = config.get('data', {})
    processed_dir_name = data_cfg.get('processed_data_dir', 'processed')
    training_timeframe = data_cfg.get('training_timeframe', '1h') # Lire depuis la config data spécifique au profil
    
    # Nouveau: Support des lots de données
    lot_id = data_cfg.get('lot_id', None)

    # 3. Construire le chemin vers le répertoire 'merged/unified' avec support des lots
    unified_segment = 'unified'
    if lot_id:
        merged_dir = os.path.join(project_root, data_dir_name, processed_dir_name, 'merged', lot_id, unified_segment)
    else:
        merged_dir = os.path.join(project_root, data_dir_name, processed_dir_name, 'merged', unified_segment)
    logger.info(f"load_merged_data: Chemin construit pour le répertoire merged/unified: {merged_dir}")

    if not os.path.isdir(merged_dir):
        logger.error(f"load_merged_data: ERREUR CRITIQUE - Répertoire des données fusionnées introuvable : {merged_dir}")
        # List files in parent to help debug
        parent_of_merged = os.path.dirname(merged_dir)
        if os.path.isdir(parent_of_merged):
             logger.info(f"Contenu de {parent_of_merged}: {os.listdir(parent_of_merged)}")
        raise FileNotFoundError(f"Répertoire des données fusionnées introuvable: {merged_dir}. Avez-vous exécuté merge_processed_data.py ?")

    # 4. Construire le chemin complet vers le fichier
    file_name = f"{training_timeframe}_{split_type}_merged.parquet"
    file_path = os.path.join(merged_dir, file_name)
    logger.info(f"load_merged_data: Tentative de chargement du fichier fusionné : {file_path}")

    # 5. Vérifier l'existence du fichier et charger
    if not os.path.exists(file_path):
        logger.error(f"load_merged_data: ERREUR CRITIQUE - Fichier introuvable : {file_path}")
        available_files = os.listdir(merged_dir)
        logger.info(f"load_merged_data: Fichiers disponibles dans {merged_dir} : {available_files}")
        raise FileNotFoundError(f"Fichier de données fusionnées introuvable: {file_path}.")

    df = pd.read_parquet(file_path) # try-except est déjà autour de l'appel à load_merged_data
    logger.info(f"load_merged_data: SUCCÈS - Fichier {file_path} chargé. Shape: {df.shape}. Colonnes (extrait): {df.columns.tolist()[:10]}")
    return df


def prepare_data_for_training(config):
    """
    Prepare data for training based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames préparées pour l'entraînement.
    """
    logger.info("Preparing data for training")
    
    # Charger directement les données fusionnées (seule méthode supportée)
    try:
        # Charger les données fusionnées pour chaque split
        logger.info("Chargement des données fusionnées pour l'entraînement...")
        train_df = load_merged_data(config, 'train')
        
        logger.info("Chargement des données fusionnées pour la validation...")
        val_df = load_merged_data(config, 'val')
        
        logger.info("Chargement des données fusionnées pour le test...")
        test_df = load_merged_data(config, 'test')
        
        # Vérifier que les données ont été chargées correctement
        if train_df.empty:
            raise ValueError("Aucune donnée d'entraînement n'a été chargée. Assurez-vous que les fichiers fusionnés existent.")
        
        logger.info(f"Successfully loaded pre-merged data with {len(train_df)} training rows")
        logger.info(f"Validation data: {len(val_df)} rows")
        logger.info(f"Test data: {len(test_df)} rows")
        
        # Vérifier que les DataFrames ont bien les colonnes attendues
        assets = config.get('assets', config.get('data', {}).get('assets', []))
        if not assets:
            logger.warning("No assets specified in configuration. Using default assets.")
            assets = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "XRPUSDT", "LTCUSDT", "SOLUSDT", "ADAUSDT"]
        
        # Vérifier la présence des colonnes de prix pour chaque actif
        missing_columns = []
        for asset in assets:
            price_col = f"close_{asset}"
            if price_col not in train_df.columns:
                missing_columns.append(price_col)
        
        if missing_columns:
            logger.warning(f"Colonnes manquantes dans les données fusionnées: {missing_columns}")
            logger.warning("Vérifiez que les scripts/process_data.py et scripts/merge_processed_data.py ont été exécutés correctement.")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"ERREUR CRITIQUE lors de la préparation des données: {e}")
        logger.error("Assurez-vous que les étapes suivantes ont été effectuées:")
        logger.error("1. Exécuter scripts/process_data.py --exec_profile cpu")
        logger.error("2. Exécuter scripts/merge_processed_data.py --exec_profile cpu")
        raise
    
    # Si le code atteint ce point, c'est une erreur car une exception aurait dû être levée plus haut
    logger.error("ERREUR INATTENDUE: Le code ne devrait jamais atteindre ce point.")
    logger.error("La méthode legacy pour charger les données individuelles a été supprimée.")
    logger.error("Veuillez vous assurer que les fichiers fusionnés existent avant l'entraînement.")
    raise RuntimeError("Méthode de préparation des données non valide.")


def generate_synthetic_data(assets, num_points_per_asset=1000):
    """
    Générer des données synthétiques pour le test.
    
    Args:
        assets: Liste des actifs pour lesquels générer des données.
        num_points_per_asset: Nombre de points de données à générer par actif.
        
    Returns:
        pd.DataFrame: Données synthétiques.
    """
    logger.info(f"Generating synthetic data for {len(assets)} assets, {num_points_per_asset} points each")
    
    all_dfs = []
    
    for asset in assets:
        # Générer des timestamps équidistants
        start_date = pd.Timestamp('2023-01-01')
        timestamps = [start_date + pd.Timedelta(hours=i) for i in range(num_points_per_asset)]
        
        # Générer des prix qui suivent un mouvement brownien géométrique
        np.random.seed(42)  # Pour la reproductibilité
        price = 100.0  # Prix initial
        prices = [price]
        for _ in range(1, num_points_per_asset):
            change_percent = np.random.normal(0, 0.01)  # Moyenne 0, écart-type 1%
            price = price * (1 + change_percent)
            prices.append(price)
        
        # Générer des volumes
        volumes = np.random.lognormal(10, 1, num_points_per_asset)
        
        # Créer un DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            f'open_{asset}': prices,
            f'high_{asset}': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],  # High légèrement supérieur
            f'low_{asset}': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],   # Low légèrement inférieur
            f'close_{asset}': prices,
            f'volume_{asset}': volumes,
            'asset': asset
        })
        
        # Ajouter quelques indicateurs techniques synthétiques
        df[f'rsi_14_{asset}'] = np.random.uniform(0, 100, num_points_per_asset)
        df[f'ema_20_{asset}'] = pd.Series(prices).ewm(span=20).mean().values
        df[f'sma_50_{asset}'] = pd.Series(prices).rolling(window=min(50, num_points_per_asset)).mean().fillna(prices[0]).values
        df[f'macd_{asset}'] = np.random.normal(0, 1, num_points_per_asset)
        df[f'macd_signal_{asset}'] = np.random.normal(0, 1, num_points_per_asset)
        df[f'macd_hist_{asset}'] = df[f'macd_{asset}'] - df[f'macd_signal_{asset}']
        
        all_dfs.append(df)
    
    # Fusionner tous les DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    logger.info(f"Generated synthetic data with {len(merged_df)} rows and {len(merged_df.columns)} columns")
    
    return merged_df


def load_raw_data_for_asset_timeframe(asset, timeframe, raw_data_dir):
    """
    Load raw data for a specific asset and timeframe from the raw data directory.
    
    Args:
        asset (str): Asset symbol (e.g., 'BTCUSDT')
        timeframe (str): Timeframe (e.g., '1h', '1d')
        raw_data_dir (str): Path to the raw data directory
        
    Returns:
        pd.DataFrame: Raw data for the specified asset and timeframe
    """
    logger.info(f"Loading raw data for {asset} ({timeframe})")
    
    # Construct the expected filename
    filename = f"{asset}_{timeframe}_raw.parquet"
    file_path = os.path.join(raw_data_dir, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"Raw data file not found: {file_path}")
        return pd.DataFrame()  # Return empty DataFrame if file not found
    
    try:
        # Load data from Parquet file
        df = pd.read_parquet(file_path)
        
        # Ensure the DataFrame has the expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in {file_path}: {missing_columns}")
            
            # Try to adapt column names if they are capitalized
            for col in missing_columns:
                capitalized = col.capitalize()
                if capitalized in df.columns:
                    df[col] = df[capitalized]
                    df.drop(capitalized, axis=1, inplace=True)
        
        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            elif 'Timestamp' in df.columns:
                df.set_index('Timestamp', inplace=True)
            else:
                logger.warning(f"No timestamp column found in {file_path}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
