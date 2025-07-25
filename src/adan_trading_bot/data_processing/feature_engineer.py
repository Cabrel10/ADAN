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

        # Vérifier la configuration
        self._validate_config()
        
        logger.info("FeatureEngineer initialized.")
    
    def _validate_config(self) -> None:
        """
        Vérifie la validité de la configuration.
        """
        try:
            # Vérifier les timeframes
            if not isinstance(self.timeframes, list) or not all(isinstance(tf, str) for tf in self.timeframes):
                raise ValueError("Timeframes must be a list of strings")
            
            # Vérifier les indicateurs
            if not isinstance(self.indicators_config, dict):
                raise ValueError("Indicators config must be a dictionary")
            
            # Vérifier les colonnes à normaliser
            if not isinstance(self.columns_to_normalize, list) or not all(isinstance(col, str) for col in self.columns_to_normalize):
                raise ValueError("Columns to normalize must be a list of strings")
            
            # Vérifier que tous les indicateurs existent
            for indicator in self.indicators_config:
                if indicator not in pta.indicators:
                    raise ValueError(f"Invalid indicator: {indicator}")
            
            logger.info("Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation error: {str(e)}")
            raise
    
    def process_data(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Pipeline principal pour le feature engineering.
        
        Args:
            data: DataFrame contenant les données de marché
            fit_scaler: Si True, ajuste le scaler aux données
            
        Returns:
            DataFrame avec les features ingénierisées

        Raises:
            ValueError: Si les colonnes à normaliser ne sont pas présentes dans le DataFrame
        """
        try:
            # Vérifier la cohérence des données d'entrée
            self._validate_input_data(data)

            # Ajouter les indicateurs techniques
            data = self.add_technical_indicators(data)

            # Vérifier que les colonnes à normaliser existent
            missing_cols = [col for col in self.columns_to_normalize if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes pour la normalisation: {missing_cols}")

            # Normaliser les données
            if fit_scaler:
                try:
                    self.scaler.fit(data[self.columns_to_normalize])
                    joblib.dump(self.scaler, self.scaler_path)
                    logger.info(f"Scaler fitted and saved to {self.scaler_path}")
                    self.fitted = True
                except Exception as e:
                    logger.error(f"Error fitting scaler: {str(e)}")
                    raise

            # Transformer les données
            data[self.columns_to_normalize] = self.scaler.transform(data[self.columns_to_normalize])

            # Nettoyer les valeurs NaN
            data = self._clean_dataframe(data)

            logger.info(f"Feature engineering complete. DataFrame shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise
    
    def _validate_input_data(self, data: pd.DataFrame):
        """
        Vérifie la cohérence des données d'entrée selon les spécifications ADAN.
        
        Args:
            data: DataFrame contenant les données de marché
            
        Raises:
            ValueError: Si les données ne sont pas conformes au format attendu
            TypeError: Si les types de données ne sont pas corrects
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Les données d'entrée doivent être un DataFrame pandas, "
                              f"type reçu: {type(data)}")
                
            # Vérifier l'existence des colonnes requises
            required_columns = {
                '5m': ['5m_open', '5m_high', '5m_low', '5m_close', '5m_volume'],
                '1h': ['1h_open', '1h_high', '1h_low', '1h_close', '1h_volume'],
                '4h': ['4h_open', '4h_high', '4h_low', '4h_close', '4h_volume']
            }
            
            # Validation des colonnes par timeframe
            for tf, cols in required_columns.items():
                # Vérifier l'existence des colonnes
                missing_cols = [c for c in cols if c not in data.columns]
                if missing_cols:
                    raise ValueError(f"Colonnes manquantes pour le timeframe {tf}: {missing_cols}")
                    
                # Vérifier les types de données
                for col in cols:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise TypeError(f"La colonne {col} doit être numérique")
                        
                # Vérifier la présence de NaN
                if data[cols].isna().any().any():
                    raise ValueError(f"Valeurs NaN détectées dans les colonnes {cols}")
                    
                # Vérifier les valeurs extrêmes
                if (data[cols] <= 0).any().any():
                    raise ValueError(f"Valeurs négatives ou nulles détectées dans les colonnes {cols}")
                    
            # Vérifier l'ordre chronologique
            if not data.index.is_monotonic_increasing:
                raise ValueError("Les données doivent être triées par ordre chronologique croissant")
                
            # Vérifier la continuité temporelle
            if len(data) > 1:
                time_diff = data.index.to_series().diff().mode()[0]
                if pd.isna(time_diff) or time_diff == 0:
                    raise ValueError("Les données doivent avoir un intervalle temporel constant")
                    
            logger.info("Validation des données d'entrée réussie")
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation des données: {str(e)}")
            raise
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les indicateurs techniques configurés pour chaque timeframe.
        """
        for tf in self.timeframes:
            logger.debug(f"Calculating 22+ indicators for timeframe: {tf}")
            
            temp_df = pd.DataFrame({
                'open': df[f'{tf}_open'],
                'high': df[f'{tf}_high'],
                'low': df[f'{tf}_low'],
                'close': df[f'{tf}_close'],
                'volume': df[f'{tf}_volume']
            })
            
            # EMA
            temp_df[f'EMA_5_{tf}'] = pta.ema(temp_df['close'], length=5)
            temp_df[f'EMA_10_{tf}'] = pta.ema(temp_df['close'], length=10)
            temp_df[f'EMA_20_{tf}'] = pta.ema(temp_df['close'], length=20)
            temp_df[f'EMA_50_{tf}'] = pta.ema(temp_df['close'], length=50)
            
            # SMA
            temp_df[f'SMA_10_{tf}'] = pta.sma(temp_df['close'], length=10)
            temp_df[f'SMA_20_{tf}'] = pta.sma(temp_df['close'], length=20)
            temp_df[f'SMA_50_{tf}'] = pta.sma(temp_df['close'], length=50)
            
            # RSI
            temp_df[f'RSI_14_{tf}'] = pta.rsi(temp_df['close'], length=14)
            
            # MACD
            macd = pta.macd(temp_df['close'])
            temp_df[f'MACD_{tf}'] = macd['MACD_12_26_9']
            temp_df[f'MACD_Signal_{tf}'] = macd['MACDs_12_26_9']
            temp_df[f'MACD_Hist_{tf}'] = macd['MACDh_12_26_9']
            
            # Ichimoku
            ichimoku = pta.ichimoku(temp_df['high'], temp_df['low'], temp_df['close'])
            temp_df[f'Ichimoku_Conversion_{tf}'] = ichimoku['ITS_9_26_52']
            temp_df[f'Ichimoku_Base_{tf}'] = ichimoku['IKS_9_26_52']
            temp_df[f'Ichimoku_SpanA_{tf}'] = ichimoku['ISA_9_26_52']
            temp_df[f'Ichimoku_SpanB_{tf}'] = ichimoku['ISB_9_26_52']
            
            # SuperTrend
            supertrend = pta.supertrend(temp_df['high'], temp_df['low'], temp_df['close'])
            temp_df[f'SuperTrend_{tf}'] = supertrend['SUPERT_10_3.0']
            temp_df[f'SuperTrend_Direction_{tf}'] = supertrend['SUPERTd_10_3.0']
            
            # Parabolic SAR
            psar = pta.psar(temp_df['high'], temp_df['low'], temp_df['close'])
            temp_df[f'PSAR_{tf}'] = psar['PSARs_0.02_0.2']
            
            # Bollinger Bands
            bbands = pta.bbands(temp_df['close'])
            temp_df[f'BB_Middle_{tf}'] = bbands['BBM_20_2.0']
            temp_df[f'BB_Upper_{tf}'] = bbands['BBU_20_2.0']
            temp_df[f'BB_Lower_{tf}'] = bbands['BBL_20_2.0']
            
            # Fusionner les nouveaux indicateurs dans le DataFrame principal
            for col in temp_df.columns:
                if col not in df.columns:
                    df[col] = temp_df[col]
            
            logger.info(f"Added technical indicators for timeframe {tf}")
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame, tf: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calcule les 5 indicateurs de volume.
        """
        indicators = []
        
        # OBV (On Balance Volume)
        col_name = f"OBV_{tf}"
        df[col_name] = pta.obv(df[f'{tf}_close'], df[f'{tf}_volume'])
        indicators.append(col_name)
        
        # Volume Weighted Average Price (VWAP)
        col_name = f"VWAP_{tf}"
        df[col_name] = pta.vwap(df[f'{tf}_high'], df[f'{tf}_low'], df[f'{tf}_close'], df[f'{tf}_volume'])
        indicators.append(col_name)
        
        # Money Flow Index (MFI)
        col_name = f"MFI_{tf}"
        df[col_name] = pta.mfi(df[f'{tf}_high'], df[f'{tf}_low'], df[f'{tf}_close'], df[f'{tf}_volume'])
        indicators.append(col_name)
        
        # Chaikin Money Flow (CMF)
        col_name = f"CMF_{tf}"
        df[col_name] = pta.cmf(df[f'{tf}_high'], df[f'{tf}_low'], df[f'{tf}_close'], df[f'{tf}_volume'])
        indicators.append(col_name)
        
        # Force Index
        col_name = f"Force_{tf}"
        df[col_name] = pta.force(df[f'{tf}_close'], df[f'{tf}_volume'])
        indicators.append(col_name)
        
        return df, indicators

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame des valeurs manquantes et infinies selon les spécifications ADAN.
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame nettoyé sans valeurs NaN ni infinies
            
        Raises:
            ValueError: Si le nettoyage ne peut pas être effectué correctement
        """
        try:
            # Validation pré-nettoyage
            self._validate_input_data(df)
            
            # 1. Gestion des valeurs infinies
            if np.isinf(df.values).any():
                logger.warning("Valeurs infinies détectées - remplacement par NaN")
                df = df.replace([np.inf, -np.inf], np.nan)
            
            # 2. Gestion des valeurs NaN
            if df.isna().any().any():
                logger.warning("Valeurs NaN détectées - début du processus de nettoyage")
                
                # a) Interpolation linéaire pour les données temporelles
                for tf in self.timeframes:
                    timeframe_cols = [col for col in df.columns if col.startswith(f'{tf}_')]
                    if timeframe_cols:
                        df[timeframe_cols] = df[timeframe_cols].interpolate(method='linear')
                
                # b) Remplissage avec la moyenne des 5 dernières valeurs
                for col in df.columns:
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].rolling(window=5, min_periods=1).mean())
                
                # c) Remplissage final avec la moyenne globale
                df = df.fillna(df.mean())
            
            # 3. Validation post-nettoyage
            if df.isna().any().any():
                raise ValueError("NaN values still present after cleaning")
                
            if np.isinf(df.values).any():
                raise ValueError("Infinite values still present after cleaning")
                
            # 4. Normalisation des valeurs extrêmes
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min <= 0 or col_max > 1e6:
                    logger.warning(f"Valeurs extrêmes détectées dans {col}: min={col_min}, max={col_max}")
                    df[col] = df[col].clip(lower=0, upper=1e6)
            
            logger.info("Nettoyage du DataFrame terminé avec succès")
            return df
            
            
        except Exception as e:
            logger.error(f"Error during dataframe cleaning: {str(e)}")
            raise
