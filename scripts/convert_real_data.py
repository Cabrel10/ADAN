#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour convertir les données réelles de data/new vers le format pipeline ADAN multi-timeframe.
Supporte la génération de données pour 1m (pré-calculées), 1h et 1d (calculées dynamiquement).
Convertit du format "long" vers le format "wide" avec gestion multi-timeframe.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.data_processing.feature_engineer import add_technical_indicators

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resample_data(df, target_timeframe):
    """
    Ré-échantillonne les données OHLCV de 1m vers 1h ou 1d.
    
    Args:
        df: DataFrame avec colonnes OHLCV et index timestamp
        target_timeframe: '1h' ou '1d'
        
    Returns:
        pd.DataFrame: Données ré-échantillonnées
    """
    logger.info(f"🔄 Ré-échantillonnage vers {target_timeframe}")
    
    # Assurer que l'index est datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Définir la fréquence de ré-échantillonnage
    freq = '1h' if target_timeframe == '1h' else '1D'
    
    # Ré-échantillonner les données OHLCV
    ohlcv_resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    logger.info(f"✅ Ré-échantillonnage: {len(df)} → {len(ohlcv_resampled)} barres")
    return ohlcv_resampled

def process_asset_for_timeframe(asset, timeframe, config):
    """
    Traite un actif pour un timeframe spécifique.
    
    Args:
        asset: Nom de l'actif (ex: 'ADAUSDT')
        timeframe: '1m', '1h', ou '1d'
        config: Configuration complète
        
    Returns:
        pd.DataFrame: Données traitées pour cet actif
    """
    logger.info(f"📊 Traitement {asset} pour {timeframe}")
    
    # Charger les données 1m de base
    source_file = f"data/new/{asset}_features.parquet"
    if not os.path.exists(source_file):
        logger.error(f"❌ Fichier source manquant: {source_file}")
        return None
    
    df = pd.read_parquet(source_file)
    logger.info(f"📈 Données chargées: {df.shape}")
    
    # Assurer que timestamp est l'index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Supprimer les colonnes non numériques
    columns_to_remove = ['symbol']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"🗑️ Colonne {col} supprimée")
    
    if timeframe == '1m':
        # Pour 1m, utiliser les données avec toutes les features pré-calculées
        logger.info(f"✅ Utilisation des features pré-calculées (1m)")
        return df
    
    else:
        # Pour 1h/1d, ré-échantillonner et recalculer les indicateurs
        logger.info(f"🔄 Ré-échantillonnage et recalcul des indicateurs ({timeframe})")
        
        # Extraire seulement OHLCV pour le ré-échantillonnage
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        available_ohlcv = [col for col in ohlcv_cols if col in df.columns]
        
        if not available_ohlcv:
            logger.error(f"❌ Colonnes OHLCV manquantes pour {asset}")
            return None
        
        df_ohlcv = df[available_ohlcv].copy()
        
        # Ré-échantillonner
        df_resampled = resample_data(df_ohlcv, timeframe)
        
        # Ajouter les indicateurs techniques pour ce timeframe
        indicators_config = config.get('indicators_by_timeframe', {}).get(timeframe, [])
        
        if indicators_config:
            logger.info(f"📊 Calcul de {len(indicators_config)} indicateurs pour {timeframe}")
            df_with_indicators, added_features = add_technical_indicators(df_resampled, indicators_config, timeframe)
            return df_with_indicators
        else:
            logger.warning(f"⚠️ Aucun indicateur configuré pour {timeframe}")
            return df_resampled

def convert_to_wide_format(processed_data_by_asset, assets, timeframe):
    """
    Convertit les données traitées au format wide.
    
    Args:
        processed_data_by_asset: Dict {asset: DataFrame} 
        assets: Liste des actifs
        timeframe: Timeframe traité
        
    Returns:
        pd.DataFrame: DataFrame au format wide
    """
    logger.info(f"🔗 Conversion au format wide pour {timeframe}")
    
    # Obtenir l'index de référence (du premier actif disponible)
    reference_asset = None
    for asset in assets:
        if asset in processed_data_by_asset and processed_data_by_asset[asset] is not None:
            reference_asset = asset
            break
    
    if not reference_asset:
        logger.error("❌ Aucun actif disponible pour la conversion")
        return None
    
    base_index = processed_data_by_asset[reference_asset].index
    merged_data = {}
    
    for asset in assets:
        if asset not in processed_data_by_asset or processed_data_by_asset[asset] is None:
            logger.warning(f"⚠️ Données manquantes pour {asset}, remplissage par zéros")
            continue
        
        asset_df = processed_data_by_asset[asset]
        
        # Ajouter le suffixe d'actif à chaque colonne
        for col in asset_df.columns:
            # Pour les colonnes OHLCV, pas de suffixe timeframe
            if col in ['open', 'high', 'low', 'close', 'volume']:
                new_col_name = f"{col}_{asset}"
            else:
                # Pour les indicateurs, ajouter le suffixe timeframe si pas déjà présent
                if f"_{timeframe}" not in col:
                    new_col_name = f"{col}_{timeframe}_{asset}"
                else:
                    new_col_name = f"{col}_{asset}"
            merged_data[new_col_name] = asset_df[col].reindex(base_index, fill_value=0.0)
    
    if not merged_data:
        logger.error("❌ Aucune donnée à fusionner")
        return None
    
    # Créer le DataFrame final
    result_df = pd.DataFrame(merged_data, index=base_index)
    logger.info(f"✅ Format wide créé: {result_df.shape}")
    
    return result_df

def split_data_by_timeframe(df, config, timeframe):
    """
    Divise les données selon les configurations de split pour le timeframe.
    
    Args:
        df: DataFrame à diviser
        config: Configuration complète
        timeframe: Timeframe concerné
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info(f"✂️ Division des données pour {timeframe}")
    
    data_split = config.get('data', {}).get('data_split', {})
    timeframe_split = data_split.get(timeframe, {})
    
    if not timeframe_split:
        logger.warning(f"⚠️ Pas de split configuré pour {timeframe}, utilisation des pourcentages par défaut")
        # Split par pourcentages
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.2)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
    else:
        # Split par dates
        train_start = pd.to_datetime(timeframe_split.get('train_start_date'))
        train_end = pd.to_datetime(timeframe_split.get('train_end_date'))
        val_start = pd.to_datetime(timeframe_split.get('validation_start_date'))
        val_end = pd.to_datetime(timeframe_split.get('validation_end_date'))
        test_start = pd.to_datetime(timeframe_split.get('test_start_date'))
        test_end = pd.to_datetime(timeframe_split.get('test_end_date'))
        
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        val_df = df[(df.index >= val_start) & (df.index <= val_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
    
    logger.info(f"📊 Split résultats: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def normalize_features(train_df, val_df, test_df, config, timeframe, assets):
    """
    Normalise les features (sauf OHLC) avec StandardScaler.
    
    Args:
        train_df, val_df, test_df: DataFrames de données
        config: Configuration
        timeframe: Timeframe traité
        assets: Liste des actifs
        
    Returns:
        tuple: (train_norm, val_norm, test_norm, scaler)
    """
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"🔧 Normalisation des features pour {timeframe}")
    
    # Identifier les colonnes à normaliser (exclure OHLC et colonnes non numériques)
    ohlc_patterns = ['open_', 'high_', 'low_', 'close_']
    cols_to_normalize = []
    
    for col in train_df.columns:
        # Vérifier que la colonne est numérique
        if train_df[col].dtype in ['object', 'string']:
            continue
        
        should_normalize = True
        for pattern in ohlc_patterns:
            if col.startswith(pattern):
                should_normalize = False
                break
        if should_normalize:
            cols_to_normalize.append(col)
    
    logger.info(f"📊 Colonnes à normaliser: {len(cols_to_normalize)}/{len(train_df.columns)}")
    
    if not cols_to_normalize:
        logger.warning("⚠️ Aucune colonne à normaliser")
        return train_df, val_df, test_df, None
    
    # Créer et ajuster le scaler sur les données d'entraînement
    scaler = StandardScaler()
    scaler.fit(train_df[cols_to_normalize])
    
    # Appliquer la normalisation
    train_norm = train_df.copy()
    val_norm = val_df.copy()
    test_norm = test_df.copy()
    
    train_norm[cols_to_normalize] = scaler.transform(train_df[cols_to_normalize])
    val_norm[cols_to_normalize] = scaler.transform(val_df[cols_to_normalize])
    test_norm[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize])
    
    logger.info("✅ Normalisation terminée")
    return train_norm, val_norm, test_norm, scaler

def save_processed_data(train_df, val_df, test_df, scaler, timeframe, assets):
    """
    Sauvegarde les données traitées et le scaler.
    
    Args:
        train_df, val_df, test_df: DataFrames normalisés
        scaler: Scaler ajusté
        timeframe: Timeframe traité
        assets: Liste des actifs
    """
    logger.info(f"💾 Sauvegarde des données pour {timeframe}")
    
    # Créer les répertoires
    processed_dir = Path("data/processed/unified")
    scalers_dir = Path("data/scalers_encoders")
    
    for directory in [processed_dir, scalers_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les splits de données
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        output_file = processed_dir / f"{timeframe}_{split_name}_merged.parquet"
        df.to_parquet(output_file)
        logger.info(f"✅ Sauvegardé: {output_file} ({df.shape})")
    
    # Sauvegarder le scaler
    if scaler is not None:
        scaler_file = scalers_dir / f"scaler_{timeframe}.joblib"
        import joblib
        joblib.dump(scaler, scaler_file)
        logger.info(f"✅ Scaler sauvegardé: {scaler_file}")

def process_unified_pipeline(config, exec_profile):
    """
    Pipeline unifié pour traiter tous les timeframes.
    
    Args:
        config: Configuration complète
        exec_profile: Profil d'exécution (cpu/gpu)
    """
    logger.info("🚀 Démarrage du pipeline unifié multi-timeframe")
    
    # Récupérer la configuration
    assets = config.get('assets', [])
    timeframes_to_process = config.get('timeframes_to_process', ['1m'])
    
    logger.info(f"📊 Assets: {assets}")
    logger.info(f"⏰ Timeframes: {timeframes_to_process}")
    
    for timeframe in timeframes_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Traitement du timeframe: {timeframe}")
        logger.info(f"{'='*60}")
        
        # Traiter chaque actif pour ce timeframe
        processed_data_by_asset = {}
        
        for asset in assets:
            try:
                processed_df = process_asset_for_timeframe(asset, timeframe, config)
                if processed_df is not None:
                    processed_data_by_asset[asset] = processed_df
                    logger.info(f"✅ {asset} traité: {processed_df.shape}")
                else:
                    logger.error(f"❌ Échec du traitement de {asset}")
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {asset}: {e}")
        
        if not processed_data_by_asset:
            logger.error(f"❌ Aucun actif traité pour {timeframe}")
            continue
        
        # Convertir au format wide
        wide_df = convert_to_wide_format(processed_data_by_asset, assets, timeframe)
        if wide_df is None:
            logger.error(f"❌ Échec de la conversion wide pour {timeframe}")
            continue
        
        # Diviser les données
        train_df, val_df, test_df = split_data_by_timeframe(wide_df, config, timeframe)
        
        # Normaliser les features
        train_norm, val_norm, test_norm, scaler = normalize_features(
            train_df, val_df, test_df, config, timeframe, assets
        )
        
        # Sauvegarder
        save_processed_data(train_norm, val_norm, test_norm, scaler, timeframe, assets)
        
        logger.info(f"✅ Timeframe {timeframe} traité avec succès!")

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(description="Pipeline unifié de traitement des données ADAN")
    parser.add_argument("--exec_profile", type=str, default="cpu", 
                       choices=["cpu", "gpu"], help="Profil d'exécution")
    
    args = parser.parse_args()
    
    try:
        # Charger la configuration
        config_file = f"config/data_config_{args.exec_profile}.yaml"
        config = load_config(config_file)
        logger.info(f"✅ Configuration chargée pour le profil: {args.exec_profile}")
        
        # Exécuter le pipeline unifié
        process_unified_pipeline(config, args.exec_profile)
        
        logger.info("🎉 Pipeline unifié terminé avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {e}")
        raise

if __name__ == "__main__":
    main()