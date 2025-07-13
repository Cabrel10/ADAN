#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature engineering module for the ADAN trading bot.

This module takes merged, multi-timeframe data and enriches it with technical
indicators and other features as specified in the data configuration.
"""

import pandas as pd
import pandas_ta as pta
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Applies feature engineering to a DataFrame of market data based on a config.

    - Calculates technical indicators for multiple timeframes.
    - Handles missing values.
    - Normalizes specified data columns for the RL agent.
    - Manages the state of the scaler (saving/loading).
    """
    def __init__(self, data_config: Dict[str, Any], models_dir: str = 'models'):
        """
        Initializes the FeatureEngineer.

        Args:
            data_config: A dictionary, typically from data_config.yaml.
            models_dir: Directory to save or load the scaler object.
        """
        self.config = data_config['feature_engineering']
        self.timeframes = self.config['timeframes']
        self.indicators_config = self.config['indicators']
        self.columns_to_normalize = self.config['columns_to_normalize']
        
        self.scaler_path = Path(models_dir) / 'feature_scaler.joblib'
        self.scaler = StandardScaler()
        self.fitted = False

        logger.info("FeatureEngineer initialized.")

    def process_data(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Main processing pipeline for feature engineering.

        Args:
            data: The merged DataFrame from the DataLoader.
            fit_scaler: If True, fits the scaler to the data. Should only be
                        done on the training dataset.

        Returns:
            The processed DataFrame with all features and normalizations applied.
        """
        # 1. Calculate indicators for each timeframe
        data_with_indicators = self._calculate_all_indicators(data)

        # 2. Handle any missing values that might have been generated
        data_filled = self._handle_missing_values(data_with_indicators)

        # 3. Normalize the data
        data_normalized = self._normalize_features(data_filled, fit=fit_scaler)
        
        logger.info(f"Feature engineering complete. DataFrame shape: {data_normalized.shape}")
        return data_normalized

    def _calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through each timeframe and calculates the configured indicators.
        """
        df = data.copy()
        for tf in self.timeframes:
            logger.debug(f"Calculating indicators for timeframe: {tf}")
            # Construct a temporary DataFrame with standard column names for pandas_ta
            temp_df = pd.DataFrame({
                'open': df[f'open_{tf}'],
                'high': df[f'high_{tf}'],
                'low': df[f'low_{tf}'],
                'close': df[f'close_{tf}'],
                'volume': df[f'volume_{tf}']
            })

            # Use pandas_ta to calculate indicators based on the config
            temp_df.ta.strategy(self._get_ta_strategy(), append=True)

            # Merge the new indicator columns back into the main DataFrame
            # with the correct timeframe suffix.
            indicator_cols = [col for col in temp_df.columns if col not in temp_df.columns[:5]]
            for col in indicator_cols:
                df[f"{col}_{tf}"] = temp_df[col]
        
        return df

    def _get_ta_strategy(self) -> List[Dict]:
        """
        Converts the YAML indicator config into a list of dictionaries that
        pandas_ta can understand.
        """
        strategy = []
        for indicator, params in self.indicators_config.items():
            if isinstance(params, list): # For indicators with multiple configs (e.g., SMA)
                for p in params:
                    strategy.append({"kind": indicator, **p})
            else:
                strategy.append({"kind": indicator, **params})
        logger.debug(f"Generated pandas_ta strategy: {strategy}")
        return strategy

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using forward-fill and then back-fill.
        """
        # Using ffill and bfill is generally safe for financial time series.
        return data.ffill().bfill()

    def _normalize_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Applies StandardScaler to the features specified in the config.
        """
        df = data.copy()
        # Dynamically build the list of columns to scale based on config and available data
        cols_to_scale = []
        for base_col_name in self.columns_to_normalize:
            for tf in self.timeframes:
                # Check for base columns (e.g., 'close_1m')
                tf_col_name = f"{base_col_name}_{tf}"
                if tf_col_name in df.columns:
                    cols_to_scale.append(tf_col_name)
                # Check for indicator columns (e.g., 'SMA_20_1m')
                # This requires a more robust check
                for indicator_col in df.columns:
                    if indicator_col.startswith(base_col_name) and indicator_col.endswith(tf):
                        if indicator_col not in cols_to_scale:
                            cols_to_scale.append(indicator_col)
        
        # Ensure we only use columns that actually exist
        final_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
        logger.info(f"Found {len(final_cols_to_scale)} columns to normalize.")

        if not final_cols_to_scale:
            logger.warning("No columns found to normalize. Skipping scaling.")
            return df

        if fit:
            logger.info("Fitting scaler and transforming data.")
            df[final_cols_to_scale] = self.scaler.fit_transform(df[final_cols_to_scale])
            self.save_scaler()
            self.fitted = True
        else:
            if not self.fitted:
                self.load_scaler()
            logger.info("Transforming data using existing scaler.")
            df[final_cols_to_scale] = self.scaler.transform(df[final_cols_to_scale])
            
        return df

    def save_scaler(self) -> None:
        """Saves the fitted scaler object to a file."""
        logger.info(f"Saving scaler to {self.scaler_path}")
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)

    def load_scaler(self) -> None:
        """Loads a pre-fitted scaler object from a file."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. "
                                    "Please fit the scaler on training data first.")
        logger.info(f"Loading scaler from {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        self.fitted = True


def add_technical_indicators(df: pd.DataFrame, indicators_config: List[Dict], timeframe: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ajoute des indicateurs techniques à un DataFrame de données de marché.
    
    Args:
        df: DataFrame contenant les données OHLCV
        indicators_config: Liste des indicateurs à ajouter
        timeframe: Période de temps (ex: '1h', '4h')
        
    Returns:
        Tuple contenant le DataFrame avec les indicateurs ajoutés et la liste des noms des indicateurs ajoutés
    """
    from typing import List, Dict, Tuple
    import pandas_ta as ta
    
    if df.empty:
        return df, []
    
    # Faire une copie pour éviter les modifications inattendues
    df = df.copy()
    
    # Liste pour stocker les noms des indicateurs ajoutés
    added_features = []
    
    # Dictionnaire pour mapper les types d'indicateurs aux fonctions pandas_ta
    indicator_functions = {
        'sma': lambda p: ta.sma(df['close'], length=p['length']),
        'ema': lambda p: ta.ema(df['close'], length=p['length']),
        'rsi': lambda p: ta.rsi(df['close'], length=p['length']),
        'macd': lambda p: ta.macd(df['close'], fast=p.get('fast', 12), slow=p.get('slow', 26), signal=p.get('signal', 9)),
        'bbands': lambda p: ta.bbands(df['close'], length=p.get('length', 20), std=p.get('std', 2.0)),
        'atr': lambda p: ta.atr(df['high'], df['low'], df['close'], length=p.get('length', 14)),
        'obv': lambda p: ta.obv(df['close'], df['volume']),
        'vwap': lambda p: ta.vwap(df['high'], df['low'], df['close'], df['volume']),
        'mfi': lambda p: ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=p.get('length', 14)),
        'adx': lambda p: ta.adx(df['high'], df['low'], df['close'], length=p.get('length', 14)),
        'cci': lambda p: ta.cci(df['high'], df['low'], df['close'], length=p.get('length', 20)),
        'willr': lambda p: ta.willr(df['high'], df['low'], df['close'], length=p.get('length', 14)),
        'stoch': lambda p: ta.stoch(df['high'], df['low'], df['close'], 
                                   k=p.get('k', 14), d=p.get('d', 3), smooth_k=p.get('smooth_k', 3))
    }
    
    # Parcourir la configuration des indicateurs
    for indicator_config in indicators_config:
        if isinstance(indicator_config, str):
            # Si c'est une chaîne, c'est une référence à un indicateur prédéfini
            indicator_name = indicator_config
            indicator_type = indicator_name.split('_')[0]  # Ex: 'sma_20' -> 'sma'
            
            # Vérifier si l'indicateur est pris en charge
            if indicator_type not in indicator_functions:
                logger.warning(f"Indicateur non pris en charge: {indicator_name}")
                continue
                
            # Générer un nom de colonne unique
            col_name = f"{indicator_name}"
            
            # Récupérer les paramètres par défaut
            params = {}
            if indicator_type in ['sma', 'ema', 'rsi', 'atr', 'mfi', 'adx', 'cci', 'willr']:
                # Extraire la longueur du nom (ex: 'sma_20' -> 20)
                try:
                    params['length'] = int(indicator_name.split('_')[1])
                except (IndexError, ValueError):
                    # Utiliser la valeur par défaut
                    params['length'] = 14
            
            # Calculer l'indicateur
            result = indicator_functions[indicator_type](params)
            
            # Gérer les résultats (certains indicateurs retournent plusieurs colonnes)
            if isinstance(result, pd.DataFrame):
                # Renommer les colonnes pour éviter les conflits
                for col in result.columns:
                    new_col = f"{col}_{timeframe}" if not col.endswith(timeframe) else col
                    df[new_col] = result[col]
                    added_features.append(new_col)
            else:
                # C'est une seule série
                new_col = f"{indicator_name}_{timeframe}"
                df[new_col] = result
                added_features.append(new_col)
    
    # Nettoyer les valeurs NaN qui peuvent résulter du calcul des indicateurs
    df = df.ffill().bfill()
    
    # Supprimer les lignes restantes avec des valeurs manquantes
    df.dropna(inplace=True)
    
    logger.info(f"Ajout de {len(added_features)} indicateurs techniques pour le timeframe {timeframe}")
    
    return df, added_features
