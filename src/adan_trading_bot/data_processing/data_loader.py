#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading, processing, and chunking utilities for the ADAN trading bot.

This module provides a robust data loader capable of handling multiple timeframes,
merging them, and serving them in chunks for efficient memory usage during training
and backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
import warnings
import gc
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os

# Configure logging
logger = logging.getLogger(__name__)

class ComprehensiveDataLoader:
    """
    Handles loading, merging, and chunking of multi-timeframe market data based
    on a configuration dictionary.

    OPTIMIZED FOR 10+ TRADING PAIRS:
    - Intelligent caching system for frequently accessed chunks
    - Parallel loading of multiple assets
    - Memory-efficient parquet handling
    - Automatic memory monitoring and cleanup
    - Optimized data structures for large datasets
    """
    def __init__(self, data_config: Dict[str, Any], processed_data_dir: str = 'data/processed'):
        """
        Initializes the data loader with a configuration dictionary.

        Args:
            data_config: A dictionary, typically loaded from data_config.yaml.
            processed_data_dir: The base directory where processed Parquet data is stored.
        """
        self.config = data_config
        self.processed_data_dir = Path(processed_data_dir)
        self.timeframes = self.config['feature_engineering']['timeframes']
        self.assets = sorted(list(set([asset for source in self.config['data_sources'] for asset in source['assets']])))
        self.chunk_size = self.config.get('chunk_size', 2000)  # Default chunk size
        self.price_column = self.config.get('price_column', 'close')
        
        self.asset_data_paths: Dict[str, Dict[str, Path]] = {} # Stores paths to processed parquet files for each asset and timeframe
        self.asset_total_rows: Dict[str, int] = {} # Stores total rows for each asset
        self.current_asset_index: int = 0
        self.current_asset: Optional[str] = None
        
        self.merged_data: Optional[pd.DataFrame] = None # This will now hold only the current chunk
        self.current_chunk_start_index: int = 0
        self.total_rows_current_asset: int = 0
        self.current_chunk_id: int = 0
        self.chunk_pnl: Dict[int, Dict[str, float]] = {}  # Track PnL per chunk
        self.current_chunk: Optional[pd.DataFrame] = None
        self.current_chunk_index: int = 0  # Track current position within chunk

        # OPTIMIZATIONS FOR 10+ PAIRS
        # Cache intelligent pour les chunks fréquemment accédés
        self.enable_cache = data_config.get('enable_cache', True)
        self.cache_size = data_config.get('cache_size', 50)  # Nombre de chunks en cache
        self.chunk_cache: Dict[str, pd.DataFrame] = {}
        self.cache_access_count: Dict[str, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Gestion mémoire avancée
        self.memory_threshold_mb = data_config.get('memory_threshold_mb', 4000)  # 4GB par défaut
        self.enable_parallel_loading = data_config.get('enable_parallel_loading', True)
        self.max_workers = data_config.get('max_workers', min(4, len(self.assets)))
        
        # Préchargement intelligent
        self.preload_next_chunks = data_config.get('preload_next_chunks', 2)
        self.preloaded_chunks: Dict[str, pd.DataFrame] = {}
        
        # Statistiques de performance
        self.load_times: List[float] = []
        self.memory_usage_history: List[float] = []
        
        logger.info(f"ComprehensiveDataLoader initialized for {len(self.assets)} assets: {self.assets} "
                    f"with timeframes: {self.timeframes}")
        logger.info(f"Optimizations: Cache={self.enable_cache} (size={self.cache_size}), "
                    f"Parallel={self.enable_parallel_loading} (workers={self.max_workers}), "
                    f"Memory threshold={self.memory_threshold_mb}MB")
        
    def calculate_optimal_pnl_for_chunk(self, chunk_data: pd.DataFrame) -> float:
        """
        Calculate the optimal possible PnL for a given chunk of data.
        
        This simulates a perfect trader that buys at the absolute low and sells at the absolute high.
        
        Args:
            chunk_data: DataFrame containing the chunk of market data
            
        Returns:
            float: The optimal PnL as a percentage of the initial capital
        """
        if self.price_column not in chunk_data.columns:
            logger.warning(f"Price column '{self.price_column}' not found in chunk data. Using 'close' as fallback.")
            price_series = chunk_data.get('close', chunk_data.iloc[:, 0])
        else:
            price_series = chunk_data[self.price_column]
        
        if len(price_series) < 2:
            return 0.0
            
        # Find the minimum and maximum prices in the chunk
        min_price = price_series.min()
        max_price = price_series.max()
        
        # Calculate optimal PnL (buy at absolute low, sell at absolute high)
        optimal_pnl_pct = ((max_price - min_price) / min_price) * 100
        
        # Store the chunk's optimal PnL
        self.chunk_pnl[self.current_chunk_id] = {
            'optimal_pnl': optimal_pnl_pct,
            'min_price': min_price,
            'max_price': max_price,
            'start_price': price_series.iloc[0],
            'end_price': price_series.iloc[-1]
        }
        
        logger.debug(f"Chunk {self.current_chunk_id} - Optimal PnL: {optimal_pnl_pct:.2f}% "
                    f"(Price range: {min_price:.2f} - {max_price:.2f})")
        
        return optimal_pnl_pct

    def load_asset_paths(self) -> None:
        """
        Identifies and stores paths to processed Parquet files for each asset and timeframe.
        Also determines the total number of rows for each asset.
        """
        logger.info("Identifying processed asset data paths...")
        for asset in self.assets:
            asset_file = asset.replace('/', '')
            self.asset_data_paths[asset] = {}
            first_timeframe_rows = 0
            
            for tf in self.timeframes:
                file_path = self.processed_data_dir / tf / f"{asset_file}.parquet"
                if not file_path.exists():
                    logger.error(f"Processed data file not found: {file_path}")
                    raise FileNotFoundError(f"Required processed data file not found: {file_path}")
                self.asset_data_paths[asset][tf] = file_path
                
                # For total rows, we can read the first timeframe's parquet metadata
                if tf == self.timeframes[0]:
                    try:
                        # Read parquet file to get number of rows
                        temp_df = pd.read_parquet(file_path)
                        first_timeframe_rows = len(temp_df)
                        del temp_df  # Libérer la mémoire immédiatement
                        gc.collect()
                    except Exception as e:
                        logger.error(f"Error reading parquet metadata for {file_path}: {e}")
                        raise
            self.asset_total_rows[asset] = first_timeframe_rows
            logger.info(f"Found {first_timeframe_rows} rows for asset {asset}")

        if not self.asset_data_paths:
            logger.error("No processed data found for any asset. Aborting.")
            raise ValueError("No processed data found.")

        self.reset() # Reset to start from the first asset
        logger.info("Asset data paths identified and total rows calculated.")

    def _get_chunk_cache_key(self, asset: str, start_index: int, end_index: int) -> str:
        """Génère une clé de cache pour un chunk spécifique."""
        return f"{asset}_{start_index}_{end_index}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Récupère un chunk du cache."""
        if not self.enable_cache or cache_key not in self.chunk_cache:
            self.cache_misses += 1
            return None
        
        self.cache_hits += 1
        self.cache_access_count[cache_key] = self.cache_access_count.get(cache_key, 0) + 1
        return self.chunk_cache[cache_key].copy()
    
    def _store_in_cache(self, cache_key: str, chunk: pd.DataFrame) -> None:
        """Stocke un chunk dans le cache avec gestion de la taille."""
        if not self.enable_cache:
            return
        
        # Vérifier la mémoire disponible
        if self._check_memory_usage():
            logger.warning("Memory threshold exceeded, skipping cache storage")
            return
        
        # Gérer la taille du cache
        if len(self.chunk_cache) >= self.cache_size:
            # Supprimer le chunk le moins utilisé
            least_used_key = min(self.cache_access_count.keys(), 
                                key=lambda k: self.cache_access_count.get(k, 0))
            del self.chunk_cache[least_used_key]
            del self.cache_access_count[least_used_key]
        
        self.chunk_cache[cache_key] = chunk.copy()
        self.cache_access_count[cache_key] = 1
    
    def _check_memory_usage(self) -> bool:
        """Vérifie si l'utilisation mémoire dépasse le seuil."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage_history.append(memory_mb)
            
            # Garder seulement les 100 dernières mesures
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-100:]
            
            return memory_mb > self.memory_threshold_mb
        except Exception as e:
            logger.warning(f"Error checking memory usage: {e}")
            return False
    
    def _load_timeframe_chunk(self, asset: str, tf: str, start_index: int, end_index: int) -> pd.DataFrame:
        """Charge un chunk pour un timeframe spécifique."""
        file_path = self.asset_data_paths[asset][tf]
        
        try:
            # Lecture optimisée du parquet
            df_tf = pd.read_parquet(
                file_path, 
                engine='fastparquet',
                columns=None,
                filters=[('__index_level_0__', '>=', start_index), 
                        ('__index_level_0__', '<', end_index)]
            )
            
            # Traitement des timestamps
            if 'timestamp' in df_tf.columns:
                df_tf['timestamp'] = pd.to_datetime(df_tf['timestamp'])
                df_tf = df_tf.set_index('timestamp')
            
            # Ajouter suffixe timeframe
            df_tf = df_tf.add_suffix(f'_{tf}')
            return df_tf
            
        except Exception as e:
            logger.error(f"Error loading chunk for {asset} {tf}: {e}")
            raise
    
    def _load_chunk_parallel(self, asset: str, start_index: int, end_index: int) -> Dict[str, pd.DataFrame]:
        """Charge les chunks de tous les timeframes en parallèle."""
        chunk_dfs = {}
        
        if self.enable_parallel_loading and len(self.timeframes) > 1:
            # Chargement parallèle
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.timeframes))) as executor:
                future_to_tf = {
                    executor.submit(self._load_timeframe_chunk, asset, tf, start_index, end_index): tf 
                    for tf in self.timeframes
                }
                
                for future in as_completed(future_to_tf):
                    tf = future_to_tf[future]
                    try:
                        chunk_dfs[tf] = future.result()
                    except Exception as e:
                        logger.error(f"Error loading {tf} chunk: {e}")
                        raise
        else:
            # Chargement séquentiel
            for tf in self.timeframes:
                chunk_dfs[tf] = self._load_timeframe_chunk(asset, tf, start_index, end_index)
        
        return chunk_dfs
    
    def get_next_chunk(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the next chunk of data for the current asset.
        OPTIMIZED VERSION with caching and parallel loading.

        Returns:
            A pandas DataFrame containing the next data chunk, or None if the
            dataset has been fully consumed for all assets.
        """
        start_time = time.time()
        
        if not self.asset_data_paths:
            self.load_asset_paths()

        if self.current_asset is None or self.current_chunk_start_index >= self.total_rows_current_asset:
            # Move to the next asset if current one is exhausted or not set
            if self.current_asset_index >= len(self.assets):
                logger.info("End of all assets reached.")
                return None

            self.current_asset = self.assets[self.current_asset_index]
            self.total_rows_current_asset = self.asset_total_rows[self.current_asset]
            self.current_chunk_start_index = 0
            self.current_chunk_id = 0
            self.current_asset_index += 1
            logger.info(f"Switching to asset: {self.current_asset} with {self.total_rows_current_asset} rows.")

        end_index = min(self.current_chunk_start_index + self.chunk_size, self.total_rows_current_asset)
        
        if self.current_chunk_start_index >= end_index:
            return None

        # Vérifier le cache intelligent
        cache_key = self._get_chunk_cache_key(self.current_asset, self.current_chunk_start_index, end_index)
        cached_chunk = self._get_from_cache(cache_key)
        
        if cached_chunk is not None:
            self.current_chunk_start_index = end_index
            self.current_chunk_id += 1
            
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            
            logger.debug(f"Cache HIT - Chunk {self.current_chunk_id} for {self.current_asset}: "
                        f"{len(cached_chunk)} rows, load time: {load_time:.3f}s")
            return cached_chunk

        # Chargement avec optimisations
        try:
            # Chargement parallèle des timeframes
            chunk_dfs = self._load_chunk_parallel(self.current_asset, self.current_chunk_start_index, end_index)
            
            # Fusion des timeframes
            base_tf = self.timeframes[0]
            merged_chunk = chunk_dfs[base_tf].copy()

            for tf in self.timeframes[1:]:
                merged_chunk = pd.merge_asof(
                    merged_chunk, chunk_dfs[tf], 
                    left_index=True, right_index=True, 
                    direction='forward', tolerance=pd.Timedelta('10 days')
                )

            # Nettoyage des données
            merged_chunk.ffill(inplace=True)
            merged_chunk.dropna(inplace=True)

            # Stocker dans le cache
            self._store_in_cache(cache_key, merged_chunk)
            
            self.current_chunk_start_index = end_index
            self.current_chunk_id += 1
            
            # Nettoyage mémoire agressif
            del chunk_dfs
            gc.collect()
            
            # Vérifier la mémoire
            if self._check_memory_usage():
                logger.warning(f"Memory usage high: {self.memory_usage_history[-1]:.1f}MB")
            
            load_time = time.time() - start_time
            self.load_times.append(load_time)

            logger.debug(f"Cache MISS - Chunk {self.current_chunk_id} for {self.current_asset}: "
                        f"{len(merged_chunk)} rows, load time: {load_time:.3f}s, "
                        f"progress: {self.get_progress():.2f}%")
            
            return merged_chunk
            
        except Exception as e:
            logger.error(f"Error loading chunk for {self.current_asset}: {e}")
            raise

    def reset(self) -> None:
        """Resets the data loader to the beginning of the first asset."""
        self.current_asset_index = 0
        self.current_asset = None
        self.current_chunk_start_index = 0
        self.total_rows_current_asset = 0
        self.current_chunk_id = 0
        self.current_chunk_index = 0
        self.current_chunk = None
        logger.info("DataLoader reset to the beginning of the dataset.")

    def get_progress(self) -> float:
        """
        Returns the current data consumption progress as a percentage for the current asset.
        If no asset is being processed, returns 0.0.
        """
        if self.total_rows_current_asset == 0:
            return 0.0
        return (self.current_chunk_start_index / self.total_rows_current_asset) * 100.0

    def get_overall_progress(self) -> float:
        """
        Returns the overall data consumption progress across all assets as a percentage.
        """
        total_processed_rows = 0
        total_possible_rows = 0
        
        for i, asset in enumerate(self.assets):
            total_possible_rows += self.asset_total_rows[asset]
            if i < self.current_asset_index - 1: # Assets already fully processed
                total_processed_rows += self.asset_total_rows[asset]
            elif i == self.current_asset_index - 1: # Current asset being processed
                total_processed_rows += self.current_chunk_start_index
        
        if total_possible_rows == 0:
            return 0.0
        return (total_processed_rows / total_possible_rows) * 100.0

    def get_next_observation(self) -> pd.Series:
        """
        Returns the next observation (row) from the current chunk.
        If the current chunk is exhausted, loads the next chunk.
        
        Returns:
            A pandas Series containing the current observation, or None if no more data.
        """
        if self.current_chunk is None or self.current_chunk_index >= len(self.current_chunk):
            self.current_chunk = self.get_next_chunk()
            self.current_chunk_index = 0
            
            if self.current_chunk is None:
                return None
                
        observation = self.current_chunk.iloc[self.current_chunk_index]
        self.current_chunk_index += 1
        
        return observation

    def get_current_chunk(self) -> Optional[pd.DataFrame]:
        """
        Returns the current chunk of data being processed.
        
        Returns:
            A pandas DataFrame containing the current data chunk, or None if no chunk is loaded.
        """
        return self.current_chunk
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache intelligent.
        
        Returns:
            Dictionnaire contenant les statistiques du cache
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.enable_cache,
            'cache_size': len(self.chunk_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de performance du DataLoader.
        
        Returns:
            Dictionnaire contenant les statistiques de performance
        """
        stats = {
            'total_assets': len(self.assets),
            'current_asset': self.current_asset,
            'current_asset_index': self.current_asset_index,
            'chunks_loaded': len(self.load_times),
            'cache_stats': self.get_cache_statistics()
        }
        
        if self.load_times:
            stats.update({
                'avg_load_time': np.mean(self.load_times),
                'min_load_time': np.min(self.load_times),
                'max_load_time': np.max(self.load_times),
                'total_load_time': np.sum(self.load_times)
            })
        
        if self.memory_usage_history:
            stats.update({
                'current_memory_mb': self.memory_usage_history[-1],
                'avg_memory_mb': np.mean(self.memory_usage_history),
                'max_memory_mb': np.max(self.memory_usage_history),
                'memory_threshold_mb': self.memory_threshold_mb
            })
        
        return stats
    
    def optimize_cache_size(self) -> None:
        """
        Optimise automatiquement la taille du cache basée sur l'utilisation mémoire.
        """
        if not self.enable_cache or not self.memory_usage_history:
            return
        
        current_memory = self.memory_usage_history[-1]
        memory_usage_ratio = current_memory / self.memory_threshold_mb
        
        if memory_usage_ratio > 0.8:  # Si on utilise plus de 80% du seuil
            # Réduire la taille du cache
            new_cache_size = max(10, int(self.cache_size * 0.8))
            if new_cache_size < self.cache_size:
                self.cache_size = new_cache_size
                # Nettoyer le cache si nécessaire
                while len(self.chunk_cache) > self.cache_size:
                    least_used_key = min(self.cache_access_count.keys(), 
                                        key=lambda k: self.cache_access_count.get(k, 0))
                    del self.chunk_cache[least_used_key]
                    del self.cache_access_count[least_used_key]
                
                logger.info(f"Cache size optimized to {self.cache_size} due to memory pressure")
        
        elif memory_usage_ratio < 0.5:  # Si on utilise moins de 50% du seuil
            # Augmenter la taille du cache
            new_cache_size = min(100, int(self.cache_size * 1.2))
            if new_cache_size > self.cache_size:
                self.cache_size = new_cache_size
                logger.info(f"Cache size increased to {self.cache_size}")
    
    def preload_chunks(self, num_chunks: int = None) -> None:
        """
        Précharge les prochains chunks pour améliorer les performances.
        
        Args:
            num_chunks: Nombre de chunks à précharger (utilise self.preload_next_chunks par défaut)
        """
        if not self.enable_cache:
            return
        
        if num_chunks is None:
            num_chunks = self.preload_next_chunks
        
        if num_chunks <= 0:
            return
        
        # Sauvegarder l'état actuel
        saved_asset_index = self.current_asset_index
        saved_asset = self.current_asset
        saved_chunk_start = self.current_chunk_start_index
        saved_chunk_id = self.current_chunk_id
        
        try:
            # Précharger les prochains chunks
            for _ in range(num_chunks):
                if self._check_memory_usage():
                    logger.warning("Memory threshold reached, stopping preload")
                    break
                
                # Calculer les indices du prochain chunk
                next_start = self.current_chunk_start_index
                next_end = min(next_start + self.chunk_size, self.total_rows_current_asset)
                
                if next_start >= next_end:
                    # Passer au prochain asset
                    if self.current_asset_index >= len(self.assets):
                        break
                    
                    self.current_asset = self.assets[self.current_asset_index]
                    self.total_rows_current_asset = self.asset_total_rows[self.current_asset]
                    self.current_chunk_start_index = 0
                    self.current_asset_index += 1
                    continue
                
                # Vérifier si le chunk est déjà en cache
                cache_key = self._get_chunk_cache_key(self.current_asset, next_start, next_end)
                if cache_key in self.chunk_cache:
                    self.current_chunk_start_index = next_end
                    continue
                
                # Charger le chunk
                try:
                    chunk_dfs = self._load_chunk_parallel(self.current_asset, next_start, next_end)
                    
                    # Fusion des timeframes
                    base_tf = self.timeframes[0]
                    merged_chunk = chunk_dfs[base_tf].copy()

                    for tf in self.timeframes[1:]:
                        merged_chunk = pd.merge_asof(
                            merged_chunk, chunk_dfs[tf], 
                            left_index=True, right_index=True, 
                            direction='forward', tolerance=pd.Timedelta('10 days')
                        )

                    # Nettoyage des données
                    merged_chunk.ffill(inplace=True)
                    merged_chunk.dropna(inplace=True)

                    # Stocker dans le cache
                    self._store_in_cache(cache_key, merged_chunk)
                    
                    self.current_chunk_start_index = next_end
                    
                    # Nettoyage mémoire
                    del chunk_dfs
                    gc.collect()
                    
                    logger.debug(f"Preloaded chunk for {self.current_asset}: {len(merged_chunk)} rows")
                    
                except Exception as e:
                    logger.warning(f"Error preloading chunk: {e}")
                    break
        
        finally:
            # Restaurer l'état original
            self.current_asset_index = saved_asset_index
            self.current_asset = saved_asset
            self.current_chunk_start_index = saved_chunk_start
            self.current_chunk_id = saved_chunk_id
    
    def clear_cache(self) -> None:
        """Vide le cache pour libérer la mémoire."""
        self.chunk_cache.clear()
        self.cache_access_count.clear()
        gc.collect()
        logger.info("Cache cleared")
    
    def get_memory_usage_mb(self) -> float:
        """
        Retourne l'utilisation mémoire actuelle en MB.
        
        Returns:
            Utilisation mémoire en MB
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

# ==============================================================================
# DEPRECATED CLASSES
# ==============================================================================

class ChunkedDataLoader:
    """
    [DEPRECATED] Use ComprehensiveDataLoader instead.
    Efficiently loads and processes large datasets in chunks to manage memory usage.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ChunkedDataLoader is deprecated and will be removed in a future version. "
            "Please switch to ComprehensiveDataLoader.",
            DeprecationWarning
        )
        # You might want to delegate to the new class or just leave it as a stub
        pass

def load_multi_timeframe_data(*args, **kwargs):
    warnings.warn(
        "load_multi_timeframe_data is deprecated and will be removed in a future version. "
        "Functionality is now part of ComprehensiveDataLoader.",
        DeprecationWarning
    )
    return {}