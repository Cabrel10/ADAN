#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.data_processing.feature_engineer import add_technical_indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resample_data(df, target_timeframe):
    logger.info(f"🔄 Ré-échantillonnage vers {target_timeframe}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    freq = '1h' if target_timeframe == '1h' else '1D'
    ohlcv_resampled = df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    logger.info(f"✅ Ré-échantillonnage: {len(df)} → {len(ohlcv_resampled)} barres")
    return ohlcv_resampled

def remove_timeframe_suffix_from_indicators(df, timeframe_suffix):
    cols_to_rename = {}
    ohlcv_base = ['open', 'high', 'low', 'close', 'volume']
    if not timeframe_suffix or not isinstance(timeframe_suffix, str):
        return df
    for col in df.columns:
        if col not in ohlcv_base and col.endswith(timeframe_suffix):
            cols_to_rename[col] = col[:-len(timeframe_suffix)]
    if cols_to_rename:
        df = df.rename(columns=cols_to_rename)
        logger.info(f"Renamed {len(cols_to_rename)} columns to remove '{timeframe_suffix}' suffix.")
    return df

def process_asset_for_timeframe(asset, timeframe, config):
    logger.info(f"📊 Traitement {asset} pour {timeframe}")
    processed_df = None
    data_config = config.get('data', {})

    if timeframe == '1m':
        source_file = f"data/raw/{asset}_1m_raw.parquet"
        if not os.path.exists(source_file):
            logger.error(f"❌ Fichier source OHLCV brut manquant: {source_file}")
            return None
        df_ohlcv_raw = pd.read_parquet(source_file)
        logger.info(f"📈 Données OHLCV brutes chargées: {df_ohlcv_raw.shape} de {source_file}")
        if 'timestamp' in df_ohlcv_raw.columns:
            df_ohlcv_raw['timestamp'] = pd.to_datetime(df_ohlcv_raw['timestamp'])
            df_ohlcv_raw = df_ohlcv_raw.set_index('timestamp')
        elif not isinstance(df_ohlcv_raw.index, pd.DatetimeIndex):
            df_ohlcv_raw.index = pd.to_datetime(df_ohlcv_raw.index)
        df_ohlcv_raw = df_ohlcv_raw.tz_convert('UTC') if df_ohlcv_raw.index.tzinfo else df_ohlcv_raw.tz_localize('UTC')
        indicators_config_1m = data_config.get('indicators_by_timeframe', {}).get('1m', [])
        df_with_indicators = df_ohlcv_raw.copy()
        if indicators_config_1m:
            logger.info(f"📊 Calcul de {len(indicators_config_1m)} indicateurs pour 1m (à partir des données brutes)")
            df_with_indicators, _ = add_technical_indicators(df_with_indicators, indicators_config_1m, '1m')
        else:
            logger.warning("⚠️ Aucun indicateur configuré pour 1m dans 'indicators_by_timeframe'.")
        base_features_names = data_config.get('base_market_features', [])
        if not base_features_names:
            logger.warning("⚠️ 'base_market_features' non défini.")
            processed_df = df_with_indicators
        else:
            final_selection = []
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for feature_name in base_features_names:
                if feature_name in ohlcv_cols:
                    if feature_name in df_with_indicators.columns:
                        final_selection.append(feature_name)
                else:
                    suffixed_name = f"{feature_name}_1m"
                    if suffixed_name in df_with_indicators.columns:
                        final_selection.append(suffixed_name)
                    elif feature_name in df_with_indicators.columns:
                         final_selection.append(feature_name)
            missing_cols = [bf for bf in base_features_names if not ((bf in ohlcv_cols and bf in final_selection) or (f"{bf}_1m" in final_selection))]
            if missing_cols:
                 logger.warning(f"⚠️ Following base_market_features not found after 1m processing: {missing_cols}")
            processed_df = df_with_indicators[[col for col in final_selection if col in df_with_indicators.columns]].copy()
    else:
        source_file_1m_raw = f"data/raw/{asset}_1m_raw.parquet"
        if not os.path.exists(source_file_1m_raw):
            logger.error(f"❌ Fichier source OHLCV brut 1m manquant pour ré-échantillonnage: {source_file_1m_raw}")
            return None
        df_1m_ohlcv_raw = pd.read_parquet(source_file_1m_raw)
        if 'timestamp' in df_1m_ohlcv_raw.columns:
             df_1m_ohlcv_raw = df_1m_ohlcv_raw.set_index(pd.to_datetime(df_1m_ohlcv_raw['timestamp']))
        elif not isinstance(df_1m_ohlcv_raw.index, pd.DatetimeIndex):
            df_1m_ohlcv_raw.index = pd.to_datetime(df_1m_ohlcv_raw.index)
        df_1m_ohlcv_raw = df_1m_ohlcv_raw.tz_convert('UTC') if df_1m_ohlcv_raw.index.tzinfo else df_1m_ohlcv_raw.tz_localize('UTC')
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df_1m_ohlcv_for_resample = df_1m_ohlcv_raw[ohlcv_cols].copy()
        df_resampled = resample_data(df_1m_ohlcv_for_resample, timeframe)
        indicators_config_higher_tf = data_config.get('indicators_by_timeframe', {}).get(timeframe, [])
        if indicators_config_higher_tf:
            logger.info(f"📊 Calcul de {len(indicators_config_higher_tf)} indicateurs pour {timeframe}")
            processed_df, _ = add_technical_indicators(df_resampled.copy(), indicators_config_higher_tf, timeframe)
        else:
            logger.warning(f"⚠️ Aucun indicateur configuré pour {timeframe}")
            processed_df = df_resampled
    if processed_df is not None:
        timeframe_suffix_to_remove = f"_{timeframe}"
        processed_df = remove_timeframe_suffix_from_indicators(processed_df, timeframe_suffix_to_remove)
        logger.info(f"✅ Final columns for {asset} {timeframe} (after suffix removal): {processed_df.columns.tolist()}")
    return processed_df

def split_data_by_timeframe(df, config, timeframe, data_type=""):
    logger.info(f"✂️ Division des données pour {timeframe} ({len(df)} lignes)")
    if df.empty:
        logger.warning(f"⚠️ DataFrame vide pour {timeframe}, impossible de diviser.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = df.sort_index() # Ensure data is sorted by time
    min_data_date = df.index.min()
    max_data_date = df.index.max()
    data_split_config = config.get('data', {}).get('data_split', {})
    timeframe_split_params = data_split_config.get(timeframe, {})
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    use_proportional_split = True
    if timeframe_split_params:
        train_start = pd.to_datetime(timeframe_split_params.get('train_start_date'), errors='coerce', utc=True)
        train_end = pd.to_datetime(timeframe_split_params.get('train_end_date'), errors='coerce', utc=True)
        val_start = pd.to_datetime(timeframe_split_params.get('validation_start_date'), errors='coerce', utc=True)
        val_end = pd.to_datetime(timeframe_split_params.get('validation_end_date'), errors='coerce', utc=True)
        test_start = pd.to_datetime(timeframe_split_params.get('test_start_date'), errors='coerce', utc=True)
        test_end = pd.to_datetime(timeframe_split_params.get('test_end_date'), errors='coerce', utc=True)
        if not (train_start and train_end and val_start and val_end and test_start and test_end):
            logger.warning(f"⚠️ Dates de split invalides ou manquantes pour {timeframe}. Passage au split proportionnel.")
        elif max(train_start, val_start, test_start) > max_data_date or min(train_end, val_end, test_end) < min_data_date:
             logger.warning(f"⚠️ Dates de split configurées pour {timeframe} sont hors de la plage de données ({min_data_date} - {max_data_date}). Passage au split proportionnel.")
        else:
            train_df = df[(df.index >= train_start) & (df.index <= train_end) & (df.index >= min_data_date) & (df.index <= max_data_date)]
            val_df = df[(df.index >= val_start) & (df.index <= val_end) & (df.index >= min_data_date) & (df.index <= max_data_date)]
            test_df = df[(df.index >= test_start) & (df.index <= test_end) & (df.index >= min_data_date) & (df.index <= max_data_date)]
            if not train_df.empty and not val_df.empty and not test_df.empty:
                use_proportional_split = False
                logger.info(f"Utilisation des dates de split configurées pour {timeframe}.")
            else:
                logger.warning(f"⚠️ Split par dates pour {timeframe} a produit >=1 dataframe vide. Tentative de split proportionnel.")
                train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                use_proportional_split = True
    if use_proportional_split:
        logger.info(f"Utilisation du split proportionnel (70/15/15) pour {timeframe} sur {len(df)} lignes.")
        train_ratio = 0.70
        val_ratio = 0.15
        n = len(df)
        train_end_idx = int(n * train_ratio)
        val_end_idx = int(n * (train_ratio + val_ratio))
        train_df = df.iloc[:train_end_idx]
        val_df = df.iloc[train_end_idx:val_end_idx]
        test_df = df.iloc[val_end_idx:]
    logger.info(f"📊 Split résultats pour {timeframe}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def save_asset_data_split(df_split, split_name, asset, timeframe, asset_data_dir):
    if df_split.empty:
        logger.warning(f"Le DataFrame pour {asset} {timeframe} {split_name} est vide. Sauvegarde ignorée.")
        return
    output_file = asset_data_dir / f"{asset}_{timeframe}_{split_name}.parquet"
    try:
        df_split.to_parquet(output_file)
        logger.info(f"✅ Data split saved: {output_file} ({df_split.shape})")
    except Exception as e:
        logger.error(f"❌ Failed to save data split {output_file}: {e}")

def process_unified_pipeline(config, exec_profile):
    logger.info("🚀 Démarrage du pipeline unifié multi-timeframe")
    data_cfg = config.get('data', {})
    assets = data_cfg.get('assets', [])
    timeframes_to_process = data_cfg.get('timeframes_to_process', ['1m'])
    logger.info(f"📊 Assets: {assets}")
    logger.info(f"⏰ Timeframes to process: {timeframes_to_process}")
    for timeframe in timeframes_to_process:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Timeframe: {timeframe}")
        logger.info(f"{'='*70}")
        timeframe_unified_data_dir = Path(data_cfg.get('processed_data_dir', 'data/processed')) / "unified" / timeframe
        timeframe_scalers_dir = Path(data_cfg.get('scalers_encoders_dir', 'data/scalers_encoders')) / timeframe
        timeframe_unified_data_dir.mkdir(parents=True, exist_ok=True)
        timeframe_scalers_dir.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            logger.info(f"\n--- Processing Asset: {asset} for Timeframe: {timeframe} ---")
            asset_specific_data_dir = timeframe_unified_data_dir / asset
            asset_specific_data_dir.mkdir(parents=True, exist_ok=True)
            try:
                asset_df = process_asset_for_timeframe(asset, timeframe, config)
                if asset_df is None or asset_df.empty:
                    logger.error(f"❌ No data processed for {asset} at {timeframe}. Skipping.")
                    continue
                train_df, val_df, test_df = split_data_by_timeframe(asset_df, config, timeframe)
                if train_df.empty :
                     logger.error(f"Train split for {asset} - {timeframe} is empty. Cannot save.")
                     continue
                logger.info(f"Skipping normalization for {asset} - {timeframe}. Saving data as is.")
                save_asset_data_split(train_df, "train", asset, timeframe, asset_specific_data_dir)
                save_asset_data_split(val_df, "val", asset, timeframe, asset_specific_data_dir)
                save_asset_data_split(test_df, "test", asset, timeframe, asset_specific_data_dir)
                logger.info(f"✅ Successfully processed and saved (unnormalized) {asset} for {timeframe}")
            except Exception as e:
                logger.error(f"❌ Error processing {asset} for {timeframe}: {e}", exc_info=True)
        logger.info(f"✅ Timeframe {timeframe} processed successfully for all assets!")

def main():
    parser = argparse.ArgumentParser(description="Pipeline unifié de traitement des données ADAN")
    parser.add_argument("--exec_profile", type=str, default="cpu", choices=["cpu", "gpu", "smoke_cpu"], help="Profil d'exécution")
    parser.add_argument('--data_config', type=str, default=None, help='Path to a specific data_config YAML file.')
    args = parser.parse_args()
    try:
        data_config_path = args.data_config if args.data_config else f"config/data_config_{args.exec_profile}.yaml"
        logger.info(f"Loading data configuration from: {data_config_path}")
        config_dict = load_config(data_config_path)
        if config_dict is None:
            raise FileNotFoundError(f"Configuration file {data_config_path} not found or is empty.")
        logger.info(f"Data configuration loaded successfully from {data_config_path}")
        process_unified_pipeline(config_dict, args.exec_profile)
        logger.info("🎉 Pipeline unifié terminé avec succès!")
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {e}", exc_info=True)
if __name__ == "__main__":
    main()
