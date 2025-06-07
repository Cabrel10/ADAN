"""
Feature engineering utilities for the ADAN trading bot.
"""
import pandas as pd
import numpy as np
import pandas_ta as pta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from ..common.utils import get_path, ensure_dir_exists, get_logger
from ..common.constants import (
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME,
    COL_RSI_14, COL_EMA_10, COL_EMA_20, COL_SMA_50,
    COL_MACD, COL_MACD_SIGNAL, COL_MACD_HIST,
    COL_BB_UPPER, COL_BB_MIDDLE, COL_BB_LOWER,
    COL_ATR_14, COL_ADX_14
)

logger = get_logger()

def add_technical_indicators(df, indicators_config_list_for_current_tf, timeframe_str):
    """
    Add technical indicators to a DataFrame using pandas_ta.
    
    Args:
        df: DataFrame with OHLCV data.
        indicators_config_list_for_current_tf: List of indicator configurations for current timeframe.
        timeframe_str: Current timeframe string (e.g., '1m', '1h', '1d').
            
    Returns:
        pd.DataFrame: DataFrame with added indicators.
        list: List of feature names that were added (for normalization).
    """
    timeframe_suffix = f"_{timeframe_str}"
    logger.info(f"🔧 Calcul des indicateurs pour timeframe: {timeframe_str}")
    logger.info(f"📊 Nombre d'indicateurs à calculer: {len(indicators_config_list_for_current_tf)}")
    
    df_with_ta = df.copy()
    added_features = []
    
    # Ensure df has lowercase column names for consistency
    df_with_ta.columns = [col.lower() for col in df_with_ta.columns]
    
    # Mapping pandas-ta column names to expected names
    pandas_ta_mappings = {
        'macd': {
            'pattern': lambda fast, slow, signal: f'MACD_{fast}_{slow}_{signal}',
            'mapping': lambda fast, slow, signal: {
                f'MACD_{fast}_{slow}_{signal}': 'MACD',
                f'MACDh_{fast}_{slow}_{signal}': 'MACDh', 
                f'MACDs_{fast}_{slow}_{signal}': 'MACDs'
            }
        },
        'bbands': {
            'pattern': lambda length, std: f'_{length}_{std}',
            'mapping': lambda length, std: {
                f'BBL_{length}_{std}': 'BBL',
                f'BBM_{length}_{std}': 'BBM', 
                f'BBU_{length}_{std}': 'BBU'
            }
        },
        'stoch': {
            'pattern': lambda k, d, smooth_k: f'_{k}_{d}_{smooth_k}',
            'mapping': lambda k, d, smooth_k: {
                f'STOCHk_{k}_{d}_{smooth_k}': 'STOCHk',
                f'STOCHd_{k}_{d}_{smooth_k}': 'STOCHd'
            }
        },
        'psar': {
            'mapping': lambda af0, af, max_af: {
                f'PSARl_{af0}_{max_af}': 'Parabolic_SAR',
                f'PSARs_{af0}_{max_af}': 'PSAR_signal'
            }
        },
        'ichimoku': {
            'mapping': lambda tenkan, kijun, senkou: {
                f'ITS_{tenkan}': 'Ichimoku_Tenkan',
                f'IKS_{kijun}': 'Ichimoku_Kijun',
                f'ISA_{tenkan}_{kijun}_{senkou}': 'Ichimoku_SenkouA',
                f'ISB_{kijun}_{senkou}': 'Ichimoku_SenkouB',
                f'ICS_{kijun}': 'Ichimoku_Chikou'
            }
        },
        'stochrsi': {
            'pattern': lambda length, rsi_length, k, d: f'_{length}_{rsi_length}_{k}_{d}',
            'mapping': lambda length, rsi_length, k, d: {
                f'STOCHRSIk_{length}_{rsi_length}_{k}_{d}': 'STOCHRSIk',
                f'STOCHRSId_{length}_{rsi_length}_{k}_{d}': 'STOCHRSId'
            }
        },
        'ppo': {
            'pattern': lambda fast, slow, signal: f'_{fast}_{slow}_{signal}',
            'mapping': lambda fast, slow, signal: {
                f'PPO_{fast}_{slow}_{signal}': 'PPO',
                f'PPOh_{fast}_{slow}_{signal}': 'PPOh',
                f'PPOs_{fast}_{slow}_{signal}': 'PPOs'
            }
        },
        'fisher': {
            'pattern': lambda length: f'_{length}_1',
            'mapping': lambda length: {
                f'FISHERT_{length}_1': 'FISHERt',
                f'FISHERTs_{length}_1': 'FISHERts'
            }
        }
    }
    
    # If no indicators_config is provided, skip
    if indicators_config_list_for_current_tf is None or len(indicators_config_list_for_current_tf) == 0:
        logger.warning(f"❌ Aucune configuration d'indicateurs fournie pour {timeframe_str}")
        return df_with_ta, added_features
    
    # Process each indicator
    for indicator_conf in indicators_config_list_for_current_tf:
        name = indicator_conf.get("name", "")
        function_name = indicator_conf.get("function", "").lower()
        params = indicator_conf.get("params", {})
        
        # Get expected output column names
        output_col_name = indicator_conf.get("output_col_name", None)
        output_col_names = indicator_conf.get("output_col_names", None)
        
        # Convert single output_col_name to list for uniform processing
        expected_names = []
        if output_col_names:
            expected_names = output_col_names if isinstance(output_col_names, list) else [output_col_names]
        elif output_col_name:
            expected_names = [output_col_name] if isinstance(output_col_name, str) else output_col_name
        
        if not function_name:
            logger.warning(f"❌ Aucune fonction spécifiée pour {name}. Ignoré.")
            continue
        
        if not expected_names:
            logger.warning(f"❌ Aucun nom de colonne attendu pour {name}. Ignoré.")
            continue
        
        try:
            # Vérifier si la fonction existe dans pandas_ta
            if not hasattr(df_with_ta.ta, function_name):
                logger.warning(f"❌ Fonction {function_name} introuvable dans pandas_ta. Ignoré.")
                continue
            
            logger.debug(f"🔧 Calcul de {name} ({function_name}) avec params: {params}")
            
            # Appeler la fonction avec les paramètres
            indicator_func = getattr(df_with_ta.ta, function_name)
            result = indicator_func(**params, append=False)
            
            if result is None or (isinstance(result, pd.DataFrame) and result.empty) or (isinstance(result, pd.Series) and result.empty):
                logger.warning(f"⚠️  {name} a retourné un résultat vide. Ignoré.")
                continue
            
            logger.debug(f"📊 {name} retourné: type={type(result)}, colonnes={result.columns.tolist() if isinstance(result, pd.DataFrame) else 'N/A'}")
            
            # Gérer les différents types de résultats
            if isinstance(result, pd.DataFrame):
                # Get pandas-ta mapping for this function
                mapping_info = pandas_ta_mappings.get(function_name, {})
                if 'mapping' in mapping_info:
                    # Use specific mapping for this function
                    pandas_ta_to_expected = mapping_info['mapping'](**params)
                else:
                    # Default mapping: use expected names directly
                    pandas_ta_to_expected = {col: expected_names[i] if i < len(expected_names) else col 
                                           for i, col in enumerate(result.columns)}
                
                logger.debug(f"🔄 Mapping pour {name}: {pandas_ta_to_expected}")
                
                # Map each column from pandas-ta result to expected name
                for pandas_ta_col in result.columns:
                    if pandas_ta_col in pandas_ta_to_expected:
                        expected_name = pandas_ta_to_expected[pandas_ta_col]
                        final_col_name = f"{expected_name}{timeframe_suffix}"
                        
                        # Add the column to dataframe
                        if len(result[pandas_ta_col]) != len(df_with_ta):
                            padded_series = pd.Series(index=df_with_ta.index, dtype=float)
                            padded_series.iloc[-len(result):] = result[pandas_ta_col].values
                            df_with_ta[final_col_name] = padded_series
                        else:
                            df_with_ta[final_col_name] = result[pandas_ta_col].values
                        
                        added_features.append(final_col_name)
                        logger.debug(f"✅ {pandas_ta_col} → {final_col_name}")
                    else:
                        logger.warning(f"⚠️  Colonne inattendue de pandas-ta: {pandas_ta_col} pour {name}")
                        
            elif isinstance(result, pd.Series):
                # Single column indicator
                if len(expected_names) > 0:
                    expected_name = expected_names[0]
                    final_col_name = f"{expected_name}{timeframe_suffix}"
                    
                    # Add the series to dataframe
                    if len(result) != len(df_with_ta):
                        padded_series = pd.Series(index=df_with_ta.index, dtype=float)
                        padded_series.iloc[-len(result):] = result.values
                        df_with_ta[final_col_name] = padded_series
                    else:
                        df_with_ta[final_col_name] = result.values
                    
                    added_features.append(final_col_name)
                    logger.debug(f"✅ {name} → {final_col_name}")
                else:
                    logger.warning(f"❌ Aucun nom attendu pour {name}")
            else:
                logger.warning(f"❌ Type inattendu pour {name}: {type(result)}")
                
            logger.info(f"✅ {name} calculé avec succès ({len([f for f in added_features if timeframe_suffix in f and any(e in f for e in expected_names)])} colonnes)")
                
        except Exception as e:
            logger.error(f"❌ Erreur pour {name} ({function_name}): {e}")
    
    # Validation finale
    if added_features:
        logger.info(f"🎯 SUCCÈS: {len(added_features)} features ajoutées pour {timeframe_str}")
        logger.debug(f"📝 Features: {added_features}")
        
        # Vérifier que les noms des features ont bien le suffixe de timeframe
        if timeframe_suffix:
            features_with_suffix = [f for f in added_features if timeframe_suffix in f]
            if len(features_with_suffix) != len(added_features):
                missing_suffix = [f for f in added_features if timeframe_suffix not in f]
                logger.warning(f"ATTENTION: {len(missing_suffix)} features n'ont pas le suffixe {timeframe_suffix}:")
                logger.warning(f"  - {missing_suffix}")
            else:
                logger.info(f"VALIDATION OK: Toutes les features ont bien le suffixe {timeframe_suffix}")
    else:
        logger.warning(f"⚠️  Aucune feature ajoutée pour {timeframe_str}")
    
    # Supprimer les colonnes dupliquées (sans suffixe) si des colonnes avec suffixe existent
    if timeframe_suffix and added_features:
        columns_to_remove = []
        for feature in added_features:
            if timeframe_suffix in feature:
                # Extraire le nom de base sans suffixe
                base_name = feature.replace(timeframe_suffix, "")
                if base_name in df_with_ta.columns and base_name != feature:
                    columns_to_remove.append(base_name)
        
        if columns_to_remove:
            logger.info(f"NETTOYAGE: Suppression des colonnes dupliquées (sans suffixe): {columns_to_remove}")
            df_with_ta = df_with_ta.drop(columns=columns_to_remove)
        else:
            logger.info("NETTOYAGE: Aucune colonne dupliquée à supprimer")
    
    # Afficher les colonnes finales du DataFrame pour vérification
    logger.info(f"VALIDATION FINALE: Colonnes du DataFrame après ajout des indicateurs pour {timeframe_str}:")
    for i in range(0, len(df_with_ta.columns), 5):
        logger.info(f"  - {df_with_ta.columns.tolist()[i:i+5]}")
    
    return df_with_ta, added_features


# Alias pour la compatibilité avec le code existant
def add_indicators(df, indicators_config=None):
    """Alias pour add_technical_indicators pour la compatibilité avec le code existant."""
    result, _ = add_technical_indicators(df, indicators_config)
    return result

def normalize_features(df, scaler=None, scaler_path=None, fit=True, numeric_cols=None, scaler_type='standard'):
    """
    Normalize numeric features in a DataFrame.
    
    Args:
        df: DataFrame with features.
        scaler: Pre-fitted scaler (optional).
        scaler_path: Path to save/load the scaler (optional).
        fit: Whether to fit the scaler on the data.
        numeric_cols: List of numeric columns to normalize.
            If None, all numeric columns will be used.
        scaler_type: Type of scaler to use ('standard', 'minmax', or 'robust').
            
    Returns:
        tuple: (DataFrame with normalized features, fitted scaler)
    """
    # Note: La normalisation est actuellement appliquée sur l'ensemble complet des données.
    # Une amélioration future pourrait consister à splitter avant normalisation.
    
    df_norm = df.copy()
    
    # Get numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude timestamp if present
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
    
    # Vérifier si des colonnes numériques sont disponibles
    if not numeric_cols:
        logger.warning("Aucune colonne numérique à normaliser.")
        return df_norm, None
    
    # Créer le bon type de scaler si aucun n'est fourni
    if scaler is None:
        if scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type.lower() == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:  # default to standard
            scaler = StandardScaler()
    
    # Load existing scaler if path is provided and scaler is None
    if scaler is None and scaler_path and os.path.exists(scaler_path) and not fit:
        try:
            logger.info(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
        except Exception as e:
            logger.error(f"Error loading scaler from {scaler_path}: {e}")
            # Créer un nouveau scaler si le chargement échoue
            if scaler_type.lower() == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type.lower() == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:  # default to standard
                scaler = StandardScaler()
            fit = True
    
    # Fit or transform
    if fit:
        # Handle NaN values before fitting
        df_numeric = df_norm[numeric_cols].ffill().bfill()
        scaler.fit(df_numeric)
        
        # Save scaler if path is provided
        if scaler_path:
            ensure_dir_exists(os.path.dirname(scaler_path))
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
    
    # Transform the data
    try:
        # Handle NaN values before transforming
        df_numeric = df_norm[numeric_cols].ffill().bfill()
        
        # Vérification de cohérence avant transformation
        if hasattr(scaler, 'n_features_in_'):
            if len(numeric_cols) != scaler.n_features_in_:
                logger.error(f"Nombre de features incompatible: {len(numeric_cols)} vs {scaler.n_features_in_}")
                logger.error(f"Features attendues: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'inconnues'}")
                logger.error(f"Features actuelles: {numeric_cols}")
                raise ValueError(f"Incompatibilité du nombre de features: {len(numeric_cols)} vs {scaler.n_features_in_}")
        
        # Transformation
        df_norm[numeric_cols] = scaler.transform(df_numeric)
        logger.info(f"Normalized {len(numeric_cols)} features using {scaler_type} scaler")
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        # Si la transformation échoue, essayer d'identifier le problème
        if hasattr(scaler, 'n_features_in_'):
            if len(numeric_cols) != scaler.n_features_in_:
                logger.error(f"Number of features mismatch: {len(numeric_cols)} vs {scaler.n_features_in_}")
                logger.error(f"Expected features: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'unknown'}")
                logger.error(f"Actual features: {numeric_cols}")
        raise
    
    return df_norm, scaler

def prepare_features(df, config, is_training=True):
    """
    Prepare features for the trading environment.
    
    Args:
        df: DataFrame with raw data.
        config: Configuration dictionary.
        is_training: Whether this is for training (True) or inference (False).
        
    Returns:
        pd.DataFrame: DataFrame with prepared features.
    """
    # Add technical indicators
    indicators_config = config.get('data', {}).get('indicators', None)
    df_with_indicators = add_indicators(df, indicators_config)
    
    # Handle missing values
    df_clean = df_with_indicators.fillna(method='ffill').fillna(method='bfill')
    
    # Normalize features if specified
    if config.get('data', {}).get('normalize', True):
        scaler_path = None
        
        # Determine scaler path
        if 'scaler_path' in config.get('data', {}):
            scaler_path = config['data']['scaler_path']
            if not os.path.isabs(scaler_path):
                scaler_path = os.path.join(get_path('data'), 'scalers_encoders', scaler_path)
        
        # Get numeric columns
        numeric_cols = config.get('data', {}).get('numeric_cols', None)
        if numeric_cols is None:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if 'timestamp' in numeric_cols:
                numeric_cols.remove('timestamp')
        
        # Normalize
        df_norm, _ = normalize_features(
            df_clean,
            scaler_path=scaler_path,
            fit=is_training,
            numeric_cols=numeric_cols
        )
        
        return df_norm
    
    return df_clean

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=False, timestamp_col='timestamp'):
    """
    Split data into training, validation, and test sets.
    
    Args:
        df: DataFrame to split.
        train_ratio: Ratio of training data.
        val_ratio: Ratio of validation data.
        test_ratio: Ratio of test data.
        shuffle: Whether to shuffle the data before splitting.
        timestamp_col: Name of the timestamp column for chronological splitting.
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio != 1.0:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Sort by timestamp if available and not shuffling
    if timestamp_col in df.columns and not shuffle:
        df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split the data
    if shuffle:
        indices = np.random.permutation(n)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
    else:
        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)
    
    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df

def prepare_data_pipeline(config, is_training=True):
    """
    Run the full data preparation pipeline.
    
    Args:
        config: Configuration dictionary.
        is_training: Whether this is for training (True) or inference (False).
        
    Returns:
        tuple: Prepared data splits (train_df, val_df, test_df) if is_training=True,
               otherwise a single DataFrame.
    """
    from .data_loader import prepare_data_for_training
    
    # Load data - la nouvelle version de prepare_data_for_training retourne directement les splits
    if is_training:
        train_df, val_df, test_df = prepare_data_for_training(config)
        
        if train_df.empty:
            logger.error("No training data loaded")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Training data loaded: {train_df.shape}")
        if not val_df.empty:
            logger.info(f"Validation data loaded: {val_df.shape}")
        if not test_df.empty:
            logger.info(f"Test data loaded: {test_df.shape}")
        
        # Les données sont déjà préparées et divisées, pas besoin de traitement supplémentaire
        return train_df, val_df, test_df
    else:
        # Pour l'inférence, on charge seulement les données de test
        from .data_loader import load_merged_data
        
        test_df = load_merged_data(config, 'test')
        
        if test_df.empty:
            logger.error("No test data loaded for inference")
            return pd.DataFrame()
        
        logger.info(f"Test data loaded for inference: {test_df.shape}")
        return test_df
