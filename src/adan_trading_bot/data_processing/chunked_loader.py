#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chunked data loading and processing for the ADAN trading bot.

This module provides efficient loading of large datasets in chunks to manage memory usage
during training and backtesting. It's designed to work with the multi-timeframe data
structure produced by the merge_processed_data.py script.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import warnings
import pyarrow.parquet as pq
import gc
import psutil
import os

# Configure logging
logger = logging.getLogger(__name__)

class ChunkedDataLoader:
    """
    Efficiently loads and processes large datasets in chunks to manage memory usage.
    
    This loader is optimized for multi-timeframe data and includes utilities for
    calculating optimal PnL on a per-chunk basis.
    
    Memory Optimizations:
    - Single chunk strategy: Only one chunk in memory at a time
    - Aggressive cleanup: Explicit memory cleanup after each chunk
    - Garbage collection: Forced GC after chunk operations
    - Memory monitoring: Track and log memory usage
    """
    
    def __init__(self, 
                 config: Dict[str, Any], # Pass the full config
                 chunk_size: Optional[int] = None,
                 assets_list: Optional[List[str]] = None,
                 features_by_timeframe: Optional[Dict[str, List[str]]] = None,
                 split: str = 'train',
                 timeframes: Optional[List[str]] = None,
                 memory_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ChunkedDataLoader with memory optimizations.
        
        Args:
            config: The full configuration dictionary.
            chunk_size: Number of rows to load per chunk (reduced default for memory optimization)
            assets_list: List of asset symbols to load (e.g., ['BTCUSDT', 'ETHUSDT'])
            features_by_timeframe: Dictionary mapping timeframes to their expected features
            split: Which split to load ('train', 'val', or 'test')
            timeframes: List of timeframes to include (e.g., ['5m', '1h', '4h'])
            memory_config: Memory optimization configuration
        """
        self.config = config
        self.data_dir = Path(self.config['paths']['indicators_data_dir']) / 'data' / 'processed' / 'indicators' # Use indicators_data_dir with correct subpath
        self.chunk_size = chunk_size if chunk_size is not None else self.config['data']['chunked_loader']['chunk_size']
        self.assets_list = assets_list or self.config['data']['assets']
        self.features_by_timeframe = features_by_timeframe or self.config['environment']['state']['features_per_timeframe']
        self.split = split
        self.timeframes = timeframes or self.config['data']['timeframes']
        
        # Memory optimization configuration
        self.memory_config = memory_config or self.config['data']['memory_optimizations']
        self.aggressive_cleanup = self.memory_config.get('aggressive_cleanup', True)
        self.force_gc = self.memory_config.get('force_gc', True)
        self.memory_monitoring = self.memory_config.get('memory_monitoring', True)
        self.memory_warning_threshold_mb = self.memory_config.get('memory_warning_threshold_mb', 5600)
        self.memory_critical_threshold_mb = self.memory_config.get('memory_critical_threshold_mb', 6300)
        
        # Cache configuration (disabled for memory optimization)
        self.disable_caching = self.memory_config.get('disable_caching', True)
        self.cache_data = {} if not self.disable_caching else None
        
        # State tracking
        self.current_chunk_index = 0
        self.total_chunks = 0
        self.asset_files = {}
        self.asset_parquet_files = {}
        self.asset_row_counts = {}
        self.current_chunk_data = None  # Single chunk in memory
        
        # Memory tracking
        self.initial_memory_mb = self._get_memory_usage_mb()
        
        # Performance monitoring
        self.performance_metrics = {
            'chunks_loaded': 0,
            'total_load_time': 0.0,
            'average_load_time': 0.0,
            'memory_peak_mb': self.initial_memory_mb,
            'gc_collections': 0,
            'errors_count': 0,
            'warnings_count': 0
        }
        
        # Validate inputs (split is still relevant for data range, but not file path)
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
        # Initialize the loader
        self._initialize_loader()
        
        logger.info(f"ChunkedDataLoader initialized with memory optimizations: "
                   f"aggressive_cleanup={self.aggressive_cleanup}, force_gc={self.force_gc}, "
                   f"memory_monitoring={self.memory_monitoring}")
        logger.info(f"Initial memory usage: {self.initial_memory_mb:.1f} MB")
    
    def _initialize_loader(self) -> None:
        """
        Initialize the loader by scanning the data directory and setting up file handles.
        
        This method looks for data files in the following locations:
        data_dir/ASSET/TIMEframe.parquet (e.g., data/processed/indicators/BTC/5m.parquet)
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"Initializing loader with data directory: {self.data_dir}")
        
        # If assets_list is not provided, discover all available assets
        if not self.assets_list:
            logger.info("No assets list provided, discovering available assets...")
            self._discover_assets()
        
        logger.info(f"Processing {len(self.assets_list)} assets: {self.assets_list}")
        
        # Initialize parquet file handles and row counts for each asset and timeframe
        for asset in self.assets_list:
            asset_name = self._get_asset_filename(asset)
            asset_dir = self.data_dir / asset_name
            
            if not asset_dir.exists():
                logger.warning(f"Asset directory not found: {asset_dir}. Skipping asset {asset}.")
                continue
            
            asset_files_for_timeframes = {}
            min_rows_for_asset = float('inf')
            
            for tf in self.timeframes:
                asset_file = asset_dir / f"{tf}.parquet"
                
                logger.info(f"Looking for data file: {asset_file}")
                
                if not asset_file.exists():
                    logger.warning(f"Data file not found for {asset} - {tf} in {asset_file}. Skipping timeframe.")
                    continue
                    
                try:
                    parquet_file = pq.ParquetFile(asset_file)
                    num_rows = parquet_file.metadata.num_rows
                    
                    asset_files_for_timeframes[tf] = {
                        'file': parquet_file,
                        'rows': num_rows,
                        'path': str(asset_file)
                    }
                    min_rows_for_asset = min(min_rows_for_asset, num_rows)
                    
                    # Log available columns for the first asset/timeframe (for debugging)
                    if asset == self.assets_list[0] and tf == self.timeframes[0]:
                        schema = parquet_file.schema_arrow
                        logger.info(f"Loaded {asset} - {tf} from {asset_file}")
                        logger.info(f"Available columns: {schema.names[:10]}...")
                        
                except Exception as e:
                    logger.error(f"Error loading {asset_file}: {e}")
                    continue
            
            if asset_files_for_timeframes:
                self.asset_files[asset] = asset_files_for_timeframes
                self.asset_row_counts[asset] = min_rows_for_asset # Store min rows across timeframes for this asset
                # Store the first timeframe's file for backward compatibility (if needed, though not ideal)
                first_tf = next(iter(asset_files_for_timeframes))
                self.asset_parquet_files[asset] = asset_files_for_timeframes[first_tf]['file']
            else:
                logger.warning(f"No valid timeframe files found for asset {asset}. Skipping asset.")
        
        if not self.asset_parquet_files:
            raise ValueError(f"No valid asset files found in {self.data_dir} for any timeframe.")
        
        # Calculate total chunks based on the asset with the fewest rows across all its timeframes
        # And then across all assets
        overall_min_rows = min(self.asset_row_counts.values()) if self.asset_row_counts else 0
        self.total_chunks = (overall_min_rows + self.chunk_size - 1) // self.chunk_size if overall_min_rows > 0 else 0
        
        logger.info(f"Initialized ChunkedDataLoader with {len(self.asset_files)} assets and {self.total_chunks} chunks of size {self.chunk_size}")
    
    def _discover_assets(self) -> None:
        """Discover all available assets in the data directory."""
        logger.info("Starting asset discovery...")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Timeframes: {self.timeframes}")
        
        # If assets are specified in config, use them
        if self.assets_list:
            logger.info(f"Using specified assets from config: {', '.join(self.assets_list)}")
            return
            
        # List to store discovered assets
        discovered_assets = set()
        
        # Check for the new structure: data_dir/ASSET/TIMEframe.parquet
        logger.info("Checking for new directory structure (data_dir/ASSET/TIMEframe.parquet)...")
        
        for asset_dir in self.data_dir.iterdir():
            if not asset_dir.is_dir():
                logger.debug(f"Skipping non-directory: {asset_dir}")
                continue
                
            asset_name = asset_dir.name
            logger.debug(f"Checking directory: {asset_dir}")
            
            # Check if there are data files for this asset across any timeframe
            found_data_for_asset = False
            for tf in self.timeframes:
                parquet_file = asset_dir / f"{tf}.parquet"
                if parquet_file.exists():
                    symbol = f"{asset_name}/USDT" # Assuming USDT as base currency for now
                    discovered_assets.add(symbol)
                    logger.info(f"Found asset data: {symbol} for {tf} in {parquet_file}")
                    found_data_for_asset = True
                    break # Only need to find one timeframe file to confirm asset exists
            
            if not found_data_for_asset:
                logger.warning(f"No data files found for asset {asset_name} across specified timeframes.")
        
        # Convert the set to a sorted list
        self.assets_list = sorted(list(discovered_assets))
            
        if not self.assets_list:
            available_files = list(self.data_dir.glob("**/*.parquet"))
            logger.error(f"No valid asset files found in {self.data_dir} with timeframes {self.timeframes}")
            logger.error(f"Available parquet files: {[str(f) for f in available_files]}")
            raise ValueError(f"No valid asset files found in {self.data_dir} with timeframes {self.timeframes}")
            
        logger.info(f"Discovered {len(self.assets_list)} assets: {', '.join(self.assets_list)}")
        
    def _get_asset_filename(self, symbol: str) -> str:
        """
        Convert symbol to filename format.
        Examples:
            - ARB/USDT -> ARB
            - BTC/USDT -> BTC
            - BTC -> BTC
        """
        # Si le symbole contient un slash, on prend la partie avant
        if '/' in symbol:
            return symbol.split('/')[0]
        # Si le symbole se termine par USDT, on le retire
        if symbol.endswith('USDT'):
            return symbol[:-4]
        # Si le symbole est déjà dans le bon format (sans /USDT)
        return symbol
    
    def load_chunk(self, chunk_index: int) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load a specific chunk of data for all assets and timeframes.
        
        Args:
            chunk_index: Index of the chunk to load
            
        Returns:
            Nested dictionary mapping asset symbols to timeframes to DataFrames
            Example: {'ARB/USDT': {'5m': df_5m, '1h': df_1h}, ...}
        """
        if not self.asset_files:
            raise RuntimeError("No asset files loaded. Call _initialize_loader() first.")
            
        if chunk_index >= self.total_chunks:
            raise IndexError(f"Chunk index {chunk_index} out of range (0-{self.total_chunks-1})")
            
        chunk_data = {asset: {} for asset in self.assets_list}
        start_row = chunk_index * self.chunk_size
        end_row = min(start_row + self.chunk_size, min(self.asset_row_counts.values()))
        
        for asset, tf_files_info in self.asset_files.items():
            # For each timeframe, load its specific parquet file
            for tf, file_info in tf_files_info.items():
                parquet_file = file_info['file']
                num_rows = file_info['rows']
                
                try:
                    # Read the chunk directly from the specific timeframe parquet file
                    start_row = chunk_index * self.chunk_size
                    end_row = min(start_row + self.chunk_size, num_rows)
                    
                    # Read the entire file and then slice, or use read_row_group if applicable
                    # For simplicity and robustness, we'll read the whole file and slice for now.
                    # Optimized reading by row group or specific row ranges can be added later if performance is an issue.
                    table = parquet_file.read(columns=None)
                    df = table.to_pandas().iloc[start_row:end_row]
                    
                    # Validate data integrity
                    if df is None or df.empty:
                        logger.warning(f"Empty data chunk for {asset} - {tf} at index {chunk_index}")
                        chunk_data[asset][tf] = pd.DataFrame()
                        continue
                    logger.info(f"Loaded DataFrame shape for {asset} - {tf} in chunk {chunk_index}: {df.shape}")
                    
                    # Ensure 'timestamp' is the index
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                    elif df.index.name != 'timestamp':
                        logger.warning(f"Timestamp column/index not found for {asset} - {tf}. Assuming index is time-based.")
                        
                    # Filter features if specified in config.environment.state.features_per_timeframe
                    if self.features_by_timeframe and tf in self.features_by_timeframe:
                        expected_features = self.features_by_timeframe[tf]
                        # Ensure all expected features are present, if not, log warning and drop missing ones
                        actual_features = [f for f in expected_features if f in df.columns]
                        missing_features = [f for f in expected_features if f not in df.columns]
                        if missing_features:
                            logger.warning(f"Missing features for {asset} - {tf}: {missing_features}. Dropping them.")
                        df = df[actual_features]
                    
                    # Add to chunk data
                    chunk_data[asset][tf] = df
                    
                    # Log the first few rows for debugging
                    if not df.empty:
                        logger.debug(f"First few rows for {asset} {tf}:\n{df.head()}")
                
                except Exception as e:
                    logger.error(f"Error loading chunk {chunk_index} for {asset} - {tf}: {e}", exc_info=True)
                    chunk_data[asset][tf] = pd.DataFrame() # Return empty DataFrame on error
                
        self.current_chunk_index = chunk_index
        loaded_count = sum(1 for asset_data in chunk_data.values() for df in asset_data.values() if not df.empty)
        logger.debug(f"Loaded chunk {chunk_index} with {loaded_count} non-empty asset-timeframe combinations")
        
        return chunk_data
    
    def _filter_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Filter DataFrame columns based on features_by_timeframe configuration.
        
        Args:
            df: Input DataFrame to filter
            timeframe: The timeframe for which to filter features
            
        Returns:
            Filtered DataFrame with only the requested features for the given timeframe
        """
        if df.empty:
            return df
            
        # Get the list of requested features for this timeframe
        if not self.features_by_timeframe or timeframe not in self.features_by_timeframe:
            # If no specific features requested, return all columns
            return df
            
        requested_features = self.features_by_timeframe[timeframe]
        
        # Always include OHLCV columns if they exist
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Find all available features that match the requested ones
        available_features = []
        for feature in requested_features + base_columns:
            if feature in df.columns and feature not in available_features:
                available_features.append(feature)
        
        # Add any additional indicator columns that match the pattern
        for col in df.columns:
            # Check for indicator columns (e.g., RSI_14, BB_UPPER_20_2.0)
            if col not in available_features and any(
                col.startswith(ind.split('_')[0]) 
                for ind in requested_features 
                if isinstance(ind, str)
            ):
                available_features.append(col)
        
        if not available_features:
            logger.warning(f"No matching features found for timeframe {timeframe}")
            return pd.DataFrame()
            
        logger.debug(f"Selected {len(available_features)} features for {timeframe}: {available_features}")
        return df[available_features]
    
    def calculate_optimal_pnl_for_chunk(self, chunk_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate the optimal possible PnL for each asset in the chunk.
        
        This simulates a perfect trader that buys at the absolute low and sells at the absolute high.
        
        Args:
            chunk_data: Dictionary of DataFrames, one per asset
            
        Returns:
            Dictionary mapping asset symbols to their optimal PnL percentages
        """
        optimal_pnl = {}
        
        for asset, df in chunk_data.items():
            if df is None or df.empty:
                logger.warning(f"No data for {asset} in chunk {self.current_chunk_index}")
                optimal_pnl[asset] = 0.0
                continue
            
            try:
                # Find the price column (prioritize close, then open, then first numeric column)
                price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
                if not price_cols:
                    price_cols = [col for col in df.columns if 'open' in col.lower()]
                if not price_cols:
                    # Fall back to first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    price_cols = [numeric_cols[0]] if len(numeric_cols) > 0 else []
                
                if not price_cols:
                    logger.warning(f"No price column found for {asset}")
                    optimal_pnl[asset] = 0.0
                    continue
                
                price_series = df[price_cols[0]]
                
                if len(price_series) < 2:
                    optimal_pnl[asset] = 0.0
                    continue
                
                # Find the minimum and maximum prices in the chunk
                min_price = price_series.min()
                max_price = price_series.max()
                start_price = price_series.iloc[0]
                
                # Calculate optimal PnL (buy at absolute low, sell at absolute high)
                if min_price > 0:
                    optimal_pnl[asset] = ((max_price - min_price) / min_price) * 100
                else:
                    optimal_pnl[asset] = 0.0
                
                logger.debug(f"Chunk {self.current_chunk_index} - {asset} - "
                           f"Optimal PnL: {optimal_pnl[asset]:.2f}% "
                           f"(Price range: {min_price:.8f} - {max_price:.8f})")
                
            except Exception as e:
                logger.error(f"Error calculating PnL for {asset} in chunk {self.current_chunk_index}: {e}")
                optimal_pnl[asset] = 0.0
        
        return optimal_pnl
    
    def __iter__(self):
        """Make the loader iterable over chunks."""
        for i in range(self.total_chunks):
            yield self.load_chunk(i)
    
    def __len__(self) -> int:
        """Return the total number of chunks."""
        return self.total_chunks
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and log warnings if thresholds are exceeded."""
        if not self.memory_monitoring:
            return
            
        current_memory_mb = self._get_memory_usage_mb()
        memory_increase_mb = current_memory_mb - self.initial_memory_mb
        
        if current_memory_mb > self.memory_critical_threshold_mb:
            logger.critical(f"CRITICAL: Memory usage {current_memory_mb:.1f} MB exceeds critical threshold "
                          f"{self.memory_critical_threshold_mb} MB! Increase: +{memory_increase_mb:.1f} MB")
            # Force aggressive cleanup
            self._aggressive_cleanup()
        elif current_memory_mb > self.memory_warning_threshold_mb:
            logger.warning(f"WARNING: Memory usage {current_memory_mb:.1f} MB exceeds warning threshold "
                         f"{self.memory_warning_threshold_mb} MB. Increase: +{memory_increase_mb:.1f} MB")
        else:
            logger.debug(f"Memory usage: {current_memory_mb:.1f} MB (increase: +{memory_increase_mb:.1f} MB)")
    
    def _aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        if not self.aggressive_cleanup:
            return
            
        logger.debug("Performing aggressive memory cleanup...")
        
        # Clear current chunk data recursively
        if self.current_chunk_data is not None:
            # Clear nested dictionaries and DataFrames
            for asset_data in self.current_chunk_data.values():
                if isinstance(asset_data, dict):
                    for df in asset_data.values():
                        if hasattr(df, 'memory_usage'):
                            del df
            
            # Clear the main dictionary
            self.current_chunk_data.clear()
            del self.current_chunk_data
            self.current_chunk_data = None
        
        # Force multiple garbage collection passes
        if self.force_gc:
            collected_total = 0
            for generation in range(3):  # Collect all generations
                collected = gc.collect(generation)
                collected_total += collected
            logger.debug(f"Aggressive garbage collection freed {collected_total} objects")
        
        # Log memory usage after cleanup
        if self.memory_monitoring:
            current_memory_mb = self._get_memory_usage_mb()
            logger.debug(f"Memory usage after cleanup: {current_memory_mb:.1f} MB")
    
    def load_chunk_optimized(self, chunk_index: int) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load a specific chunk with memory optimizations.
        
        This method implements the single chunk strategy with aggressive cleanup.
        
        Args:
            chunk_index: Index of the chunk to load
            
        Returns:
            Nested dictionary mapping asset symbols to timeframes to DataFrames
        """
        import time
        start_time = time.time()
        
        # Check memory before loading
        self._check_memory_usage()
        
        # Clean up previous chunk if it exists
        if self.current_chunk_data is not None:
            logger.debug(f"Cleaning up previous chunk data...")
            self._aggressive_cleanup()
        
        try:
            # Load the new chunk
            logger.debug(f"Loading chunk {chunk_index} with memory optimizations...")
            chunk_data = self.load_chunk(chunk_index)
            
            # Validate the structure of the loaded data
            if not isinstance(chunk_data, dict):
                error_msg = f"Loaded data is not a dictionary but {type(chunk_data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            for asset, timeframe_data in chunk_data.items():
                if not isinstance(timeframe_data, dict):
                    error_msg = f"Timeframe data for {asset} is not a dictionary but {type(timeframe_data)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                for timeframe, df in timeframe_data.items():
                    if not isinstance(df, pd.DataFrame):
                        error_msg = f"Data for {asset} - {timeframe} is not a DataFrame but {type(df)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            
            # Store as current chunk (single chunk strategy)
            self.current_chunk_data = chunk_data
            
            # Update performance metrics
            load_time = time.time() - start_time
            self.performance_metrics['chunks_loaded'] += 1
            self.performance_metrics['total_load_time'] += load_time
            self.performance_metrics['average_load_time'] = (
                self.performance_metrics['total_load_time'] / 
                self.performance_metrics['chunks_loaded']
            )
            
            # Update memory peak
            current_memory = self._get_memory_usage_mb()
            if current_memory > self.performance_metrics['memory_peak_mb']:
                self.performance_metrics['memory_peak_mb'] = current_memory
            
            # Check memory after loading
            self._check_memory_usage()
            
            # Force garbage collection if enabled
            if self.force_gc:
                collected = gc.collect()
                self.performance_metrics['gc_collections'] += 1
                logger.debug(f"Post-load garbage collection freed {collected} objects")
            
            logger.debug(f"Chunk {chunk_index} loaded in {load_time:.3f}s")
            return chunk_data
            
        except Exception as e:
            self.performance_metrics['errors_count'] += 1
            logger.error(f"Error in optimized chunk loading: {e}")
            raise
    
    def get_current_chunk(self) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
        """Get the currently loaded chunk (single chunk strategy)."""
        return self.current_chunk_data
    
    def clear_current_chunk(self) -> None:
        """Clear the current chunk from memory."""
        if self.current_chunk_data is not None:
            logger.debug("Clearing current chunk from memory...")
            self._aggressive_cleanup()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        current_memory_mb = self._get_memory_usage_mb()
        return {
            'current_memory_mb': current_memory_mb,
            'initial_memory_mb': self.initial_memory_mb,
            'memory_increase_mb': current_memory_mb - self.initial_memory_mb,
            'warning_threshold_mb': self.memory_warning_threshold_mb,
            'critical_threshold_mb': self.memory_critical_threshold_mb
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring.
        
        Returns:
            Dictionary containing performance statistics
        """
        current_memory = self._get_memory_usage_mb()
        
        return {
            'chunks_loaded': self.performance_metrics['chunks_loaded'],
            'total_load_time': self.performance_metrics['total_load_time'],
            'average_load_time': self.performance_metrics['average_load_time'],
            'memory_peak_mb': self.performance_metrics['memory_peak_mb'],
            'current_memory_mb': current_memory,
            'memory_increase_mb': current_memory - self.initial_memory_mb,
            'gc_collections': self.performance_metrics['gc_collections'],
            'errors_count': self.performance_metrics['errors_count'],
            'warnings_count': self.performance_metrics['warnings_count'],
            'chunks_per_second': (
                self.performance_metrics['chunks_loaded'] / self.performance_metrics['total_load_time']
                if self.performance_metrics['total_load_time'] > 0 else 0
            ),
            'memory_efficiency': (
                self.performance_metrics['chunks_loaded'] / 
                (current_memory - self.initial_memory_mb)
                if (current_memory - self.initial_memory_mb) > 0 else 0
            )
        }
    
    def log_performance_summary(self) -> None:
        """Log a comprehensive performance summary."""
        metrics = self.get_performance_metrics()
        
        logger.info("=== ChunkedDataLoader Performance Summary ===")
        logger.info(f"Chunks loaded: {metrics['chunks_loaded']}")
        logger.info(f"Total load time: {metrics['total_load_time']:.2f}s")
        logger.info(f"Average load time: {metrics['average_load_time']:.3f}s")
        logger.info(f"Chunks per second: {metrics['chunks_per_second']:.2f}")
        logger.info(f"Memory peak: {metrics['memory_peak_mb']:.1f} MB")
        logger.info(f"Memory increase: +{metrics['memory_increase_mb']:.1f} MB")
        logger.info(f"GC collections: {metrics['gc_collections']}")
        logger.info(f"Errors: {metrics['errors_count']}")
        logger.info(f"Warnings: {metrics['warnings_count']}")
        
        if metrics['memory_efficiency'] > 0:
            logger.info(f"Memory efficiency: {metrics['memory_efficiency']:.2f} chunks/MB")
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics counters."""
        self.performance_metrics = {
            'chunks_loaded': 0,
            'total_load_time': 0.0,
            'average_load_time': 0.0,
            'memory_peak_mb': self._get_memory_usage_mb(),
            'gc_collections': 0,
            'errors_count': 0,
            'warnings_count': 0
        }
        logger.info("Performance metrics reset")
    
    def is_caching_disabled(self) -> bool:
        """Check if caching is disabled for memory optimization."""
        return self.disable_caching
    
    def clear_cache(self) -> None:
        """Clear any cached data (if caching was enabled)."""
        if self.cache_data is not None:
            logger.debug("Clearing cache data...")
            self.cache_data.clear()
            if self.force_gc:
                collected = gc.collect()
                logger.debug(f"Cache cleanup garbage collection freed {collected} objects")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache status and usage."""
        return {
            'caching_disabled': self.disable_caching,
            'cache_size': len(self.cache_data) if self.cache_data is not None else 0,
            'cache_enabled': not self.disable_caching
        }
    
    def get_current_chunk(self) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
        """Get the currently loaded chunk (single chunk strategy)."""
        return self.current_chunk_data
    
    def clear_current_chunk(self) -> None:
        """Clear the current chunk from memory."""
        if self.current_chunk_data is not None:
            logger.debug("Clearing current chunk from memory...")
            self._aggressive_cleanup()
    
    def validate_data_quality(self, chunk_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Validate data quality for a chunk according to design specifications.
        
        This method checks for:
        - Data completeness
        - Missing data percentage
        - Timeframe synchronization
        - Data consistency
        - Outlier detection
        
        Args:
            chunk_data: Chunk data dictionary (asset -> timeframe -> DataFrame)
            
        Returns:
            Dictionary containing quality metrics and validation results
        """
        quality_report = {
            'overall_quality': 'GOOD',
            'issues': [],
            'warnings': [],
            'metrics': {},
            'asset_quality': {}
        }
        
        total_assets = len(chunk_data)
        valid_assets = 0
        
        for asset, timeframe_data in chunk_data.items():
            asset_quality = {
                'status': 'GOOD',
                'missing_data_pct': {},
                'timeframe_sync': True,
                'outliers_detected': False,
                'data_completeness': {}
            }
            
            try:
                # Check data completeness for each timeframe
                for tf, df in timeframe_data.items():
                    if df is None or df.empty:
                        asset_quality['status'] = 'POOR'
                        quality_report['issues'].append(f"{asset} {tf}: No data available")
                        asset_quality['data_completeness'][tf] = 0.0
                        continue
                    
                    # Calculate missing data percentage
                    total_cells = df.size
                    missing_cells = df.isna().sum().sum()
                    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 100
                    
                    asset_quality['missing_data_pct'][tf] = missing_pct
                    asset_quality['data_completeness'][tf] = 100 - missing_pct
                    
                    # Check missing data threshold (design spec: monitor missing data percentage)
                    if missing_pct > 10:  # More than 10% missing
                        asset_quality['status'] = 'POOR'
                        quality_report['issues'].append(f"{asset} {tf}: High missing data ({missing_pct:.1f}%)")
                    elif missing_pct > 5:  # More than 5% missing
                        asset_quality['status'] = 'FAIR'
                        quality_report['warnings'].append(f"{asset} {tf}: Moderate missing data ({missing_pct:.1f}%)")
                    
                    # Check for outliers in price data
                    price_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close']]
                    for price_col in price_cols:
                        if price_col in df.columns:
                            prices = df[price_col].dropna()
                            if len(prices) > 0:
                                # Use IQR method for outlier detection
                                Q1 = prices.quantile(0.25)
                                Q3 = prices.quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                
                                outliers = ((prices < lower_bound) | (prices > upper_bound)).sum()
                                outlier_pct = (outliers / len(prices)) * 100
                                
                                if outlier_pct > 5:  # More than 5% outliers
                                    asset_quality['outliers_detected'] = True
                                    quality_report['warnings'].append(f"{asset} {tf} {price_col}: High outliers ({outlier_pct:.1f}%)")
                
                # Check timeframe synchronization
                timeframe_lengths = {tf: len(df) for tf, df in timeframe_data.items() if df is not None and not df.empty}
                if len(set(timeframe_lengths.values())) > 1:
                    asset_quality['timeframe_sync'] = False
                    quality_report['warnings'].append(f"{asset}: Timeframe length mismatch {timeframe_lengths}")
                
                # Determine overall asset quality
                if asset_quality['status'] == 'GOOD':
                    valid_assets += 1
                
            except Exception as e:
                asset_quality['status'] = 'ERROR'
                quality_report['issues'].append(f"{asset}: Validation error - {str(e)}")
                logger.error(f"Error validating data quality for {asset}: {e}")
            
            quality_report['asset_quality'][asset] = asset_quality
        
        # Calculate overall metrics
        quality_report['metrics'] = {
            'total_assets': total_assets,
            'valid_assets': valid_assets,
            'asset_success_rate': (valid_assets / total_assets) * 100 if total_assets > 0 else 0,
            'total_issues': len(quality_report['issues']),
            'total_warnings': len(quality_report['warnings'])
        }
        
        # Determine overall quality
        success_rate = quality_report['metrics']['asset_success_rate']
        if success_rate < 50 or len(quality_report['issues']) > 5:
            quality_report['overall_quality'] = 'POOR'
        elif success_rate < 80 or len(quality_report['issues']) > 0:
            quality_report['overall_quality'] = 'FAIR'
        elif len(quality_report['warnings']) > 3:
            quality_report['overall_quality'] = 'FAIR'
        
        return quality_report
    
    def get_data_quality_summary(self, chunk_index: int) -> Dict[str, Any]:
        """
        Get data quality summary for a specific chunk.
        
        Args:
            chunk_index: Index of the chunk to validate
            
        Returns:
            Data quality summary dictionary
        """
        if chunk_index >= self.total_chunks:
            raise IndexError(f"Chunk index {chunk_index} out of range")
        
        # Load chunk if not already loaded
        if self.current_chunk_data is None or self.current_chunk_index != chunk_index:
            chunk_data = self.load_chunk_optimized(chunk_index)
        else:
            chunk_data = self.current_chunk_data
        
        return self.validate_data_quality(chunk_data)
    
    def log_data_quality_report(self, quality_report: Dict[str, Any]) -> None:
        """
        Log data quality report according to design specifications.
        
        Args:
            quality_report: Quality report from validate_data_quality()
        """
        logger.info(f"Data Quality Report - Overall: {quality_report['overall_quality']}")
        logger.info(f"Asset Success Rate: {quality_report['metrics']['asset_success_rate']:.1f}%")
        
        if quality_report['issues']:
            logger.warning(f"Found {len(quality_report['issues'])} data quality issues:")
            for issue in quality_report['issues']:
                logger.warning(f"  - {issue}")
        
        if quality_report['warnings']:
            logger.info(f"Found {len(quality_report['warnings'])} data quality warnings:")
            for warning in quality_report['warnings']:
                logger.info(f"  - {warning}")
        
        # Log detailed metrics for each asset
        for asset, asset_quality in quality_report['asset_quality'].items():
            if asset_quality['status'] != 'GOOD':
                logger.debug(f"{asset} quality: {asset_quality['status']}")
                for tf, completeness in asset_quality['data_completeness'].items():
                    logger.debug(f"  {tf}: {completeness:.1f}% complete")
    
    def get_error_recovery_suggestions(self, quality_report: Dict[str, Any]) -> List[str]:
        """
        Generate error recovery suggestions based on data quality issues.
        
        Args:
            quality_report: Quality report from validate_data_quality()
            
        Returns:
            List of suggested recovery actions
        """
        suggestions = []
        
        if quality_report['overall_quality'] == 'POOR':
            suggestions.append("Consider using a different data split or chunk")
            suggestions.append("Check data preprocessing pipeline for errors")
        
        # Analyze specific issues
        for issue in quality_report['issues']:
            if 'No data available' in issue:
                suggestions.append(f"Verify data files exist for {issue.split(':')[0]}")
            elif 'High missing data' in issue:
                suggestions.append(f"Apply forward-fill or interpolation for {issue.split(':')[0]}")
            elif 'Timeframe length mismatch' in issue:
                suggestions.append("Re-run data synchronization process")
        
        # Remove duplicates
        return list(set(suggestions))
    
    def handle_missing_data(self, chunk_data: Dict[str, Dict[str, pd.DataFrame]], 
                           strategy: str = 'forward_fill') -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Handle missing data according to design specifications.
        
        This method implements various strategies for handling missing data:
        - Forward fill with validity mask
        - Backward fill for remaining gaps
        - Interpolation for numeric data
        - Dropping rows/columns with excessive missing data
        
        Args:
            chunk_data: Chunk data dictionary (asset -> timeframe -> DataFrame)
            strategy: Strategy to use ('forward_fill', 'interpolate', 'drop', 'mixed')
            
        Returns:
            Processed chunk data with missing data handled
        """
        processed_data = {}
        
        for asset, timeframe_data in chunk_data.items():
            processed_data[asset] = {}
            
            for tf, df in timeframe_data.items():
                if df is None or df.empty:
                    processed_data[asset][tf] = df
                    continue
                
                try:
                    processed_df = df.copy()
                    initial_na_count = processed_df.isna().sum().sum()
                    
                    if initial_na_count == 0:
                        processed_data[asset][tf] = processed_df
                        continue
                    
                    logger.debug(f"Handling {initial_na_count} missing values in {asset} {tf}")
                    
                    if strategy == 'forward_fill' or strategy == 'mixed':
                        # Apply forward fill
                        processed_df = processed_df.ffill()
                        
                        # Apply backward fill for remaining NAs
                        processed_df = processed_df.bfill()
                        
                        # Add validity mask for forward-filled data
                        validity_mask = ~df.isna()  # True where original data exists
                        
                        # Add validity columns for each feature
                        for col in df.columns:
                            if col not in ['timestamp']:
                                validity_col = f"{col}_valid"
                                processed_df[validity_col] = validity_mask[col].astype(int)
                    
                    elif strategy == 'interpolate':
                        # Use interpolation for numeric columns
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        processed_df[numeric_cols] = processed_df[numeric_cols].interpolate(method='linear')
                        
                        # Forward fill for remaining NAs
                        processed_df = processed_df.ffill().bfill()
                    
                    elif strategy == 'drop':
                        # Drop rows with too many missing values
                        threshold = len(processed_df.columns) * 0.5  # Keep rows with at least 50% data
                        processed_df = processed_df.dropna(thresh=threshold)
                        
                        # Drop columns with too many missing values
                        threshold = len(processed_df) * 0.9  # Keep columns with at least 90% data
                        processed_df = processed_df.dropna(axis=1, thresh=threshold)
                    
                    final_na_count = processed_df.isna().sum().sum()
                    improvement = ((initial_na_count - final_na_count) / initial_na_count * 100) if initial_na_count > 0 else 0
                    
                    logger.debug(f"{asset} {tf}: Missing data handling {improvement:.1f}% effective "
                               f"({initial_na_count} -> {final_na_count} NAs)")
                    
                    processed_data[asset][tf] = processed_df
                    
                except Exception as e:
                    logger.error(f"Error handling missing data for {asset} {tf}: {e}")
                    processed_data[asset][tf] = df  # Return original data on error
        
        return processed_data
    
    def get_missing_data_report(self, chunk_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Generate a comprehensive missing data report.
        
        Args:
            chunk_data: Chunk data dictionary (asset -> timeframe -> DataFrame)
            
        Returns:
            Dictionary containing missing data statistics and recommendations
        """
        report = {
            'overall_missing_pct': 0.0,
            'asset_reports': {},
            'recommendations': [],
            'critical_issues': []
        }
        
        total_cells = 0
        total_missing = 0
        
        for asset, timeframe_data in chunk_data.items():
            asset_report = {
                'timeframe_missing': {},
                'worst_columns': {},
                'status': 'GOOD'
            }
            
            for tf, df in timeframe_data.items():
                if df is None or df.empty:
                    asset_report['timeframe_missing'][tf] = 100.0
                    asset_report['status'] = 'CRITICAL'
                    continue
                
                # Calculate missing data statistics
                tf_total_cells = df.size
                tf_missing_cells = df.isna().sum().sum()
                tf_missing_pct = (tf_missing_cells / tf_total_cells * 100) if tf_total_cells > 0 else 0
                
                asset_report['timeframe_missing'][tf] = tf_missing_pct
                
                # Find worst columns
                column_missing = df.isna().sum()
                worst_cols = column_missing.nlargest(3)
                asset_report['worst_columns'][tf] = {
                    col: (count / len(df) * 100) for col, count in worst_cols.items() if count > 0
                }
                
                # Update totals
                total_cells += tf_total_cells
                total_missing += tf_missing_cells
                
                # Determine status
                if tf_missing_pct > 20:
                    asset_report['status'] = 'CRITICAL'
                elif tf_missing_pct > 10:
                    asset_report['status'] = 'WARNING'
            
            report['asset_reports'][asset] = asset_report
            
            # Generate recommendations
            if asset_report['status'] == 'CRITICAL':
                report['critical_issues'].append(f"{asset}: Critical missing data levels")
                report['recommendations'].append(f"Consider excluding {asset} or using alternative data source")
            elif asset_report['status'] == 'WARNING':
                report['recommendations'].append(f"Apply aggressive forward-fill for {asset}")
        
        # Calculate overall statistics
        report['overall_missing_pct'] = (total_missing / total_cells * 100) if total_cells > 0 else 0
        
        # Generate general recommendations
        if report['overall_missing_pct'] > 15:
            report['recommendations'].append("Overall missing data is high - review data collection process")
        elif report['overall_missing_pct'] > 5:
            report['recommendations'].append("Apply mixed strategy (forward-fill + interpolation)")
        
        return report
        
    def clear_current_chunk(self) -> None:
        """Clear the current chunk from memory."""
        if self.current_chunk_data is not None:
            logger.debug("Clearing current chunk from memory...")
            self._aggressive_cleanup()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        current_memory_mb = self._get_memory_usage_mb()
        return {
            'current_memory_mb': current_memory_mb,
            'initial_memory_mb': self.initial_memory_mb,
            'memory_increase_mb': current_memory_mb - self.initial_memory_mb,
            'warning_threshold_mb': self.memory_warning_threshold_mb,
            'critical_threshold_mb': self.memory_critical_threshold_mb
        }
    
    def is_caching_disabled(self) -> bool:
        """Check if caching is disabled for memory optimization."""
        return self.disable_caching
    
    def clear_cache(self) -> None:
        """Clear any cached data (if caching was enabled)."""
        if self.cache_data is not None:
            logger.debug("Clearing cache data...")
            self.cache_data.clear()
            if self.force_gc:
                collected = gc.collect()
                logger.debug(f"Cache cleanup garbage collection freed {collected} objects")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache status and usage."""
        return {
            'caching_disabled': self.disable_caching,
            'cache_size': len(self.cache_data) if self.cache_data is not None else 0,
            'cache_enabled': not self.disable_caching
        }
    
    def calculate_optimal_pnl(self, chunk_data: Dict[str, Dict[str, pd.DataFrame]], 
                             price_column: str = 'close') -> Dict[str, float]:
        """
        Calculate optimal PnL for a chunk of data.
        
        This method calculates the theoretical maximum profit that could be achieved
        by perfect market timing (buying at lows, selling at highs) for each asset.
        This is used for reward shaping and performance benchmarking.
        
        Args:
            chunk_data: Chunk data dictionary (asset -> timeframe -> DataFrame)
            price_column: Column name to use for price data (default: 'close')
            
        Returns:
            Dictionary mapping asset symbols to optimal PnL percentages
        """
        optimal_pnl = {}
        
        for asset, timeframe_data in chunk_data.items():
            try:
                # Use the shortest timeframe (5m) for optimal PnL calculation
                if '5m' in timeframe_data and timeframe_data['5m'] is not None:
                    df = timeframe_data['5m']
                    
                    # Get price column with timeframe prefix
                    price_col = f"5m_{price_column}"
                    if price_col not in df.columns:
                        logger.warning(f"Price column {price_col} not found for {asset}")
                        optimal_pnl[asset] = 0.0
                        continue
                    
                    prices = df[price_col].dropna()
                    if len(prices) < 2:
                        optimal_pnl[asset] = 0.0
                        continue
                    
                    # Calculate optimal PnL using perfect market timing
                    # This assumes we can buy at every local minimum and sell at every local maximum
                    optimal_return = self._calculate_perfect_timing_return(prices)
                    optimal_pnl[asset] = optimal_return
                    
                    logger.debug(f"Optimal PnL for {asset}: {optimal_return:.4f} ({optimal_return*100:.2f}%)")
                    
                else:
                    logger.warning(f"No 5m data available for {asset}")
                    optimal_pnl[asset] = 0.0
                    
            except Exception as e:
                logger.error(f"Error calculating optimal PnL for {asset}: {e}")
                optimal_pnl[asset] = 0.0
        
        return optimal_pnl
    
    def _calculate_perfect_timing_return(self, prices: pd.Series) -> float:
        """
        Calculate the theoretical maximum return using perfect market timing.
        
        This method identifies all local minima and maxima and calculates
        the cumulative return from buying at lows and selling at highs.
        
        Args:
            prices: Series of price data
            
        Returns:
            Cumulative return from perfect timing (as decimal, e.g., 0.1 = 10%)
        """
        if len(prices) < 3:
            return 0.0
        
        try:
            # Convert to numpy array for faster processing
            price_array = prices.values
            
            # Find local minima and maxima
            # A local minimum is where price[i-1] > price[i] < price[i+1]
            # A local maximum is where price[i-1] < price[i] > price[i+1]
            
            buy_points = []  # Local minima
            sell_points = []  # Local maxima
            
            for i in range(1, len(price_array) - 1):
                # Local minimum (buy point)
                if price_array[i-1] > price_array[i] < price_array[i+1]:
                    buy_points.append((i, price_array[i]))
                
                # Local maximum (sell point)
                elif price_array[i-1] < price_array[i] > price_array[i+1]:
                    sell_points.append((i, price_array[i]))
            
            if not buy_points or not sell_points:
                # If no clear local extrema, use simple buy-low-sell-high
                min_price = np.min(price_array)
                max_price = np.max(price_array)
                return (max_price - min_price) / min_price if min_price > 0 else 0.0
            
            # Calculate optimal trading strategy
            cumulative_return = 0.0
            position = 0.0  # 0 = no position, 1 = long position
            
            # Combine and sort all trading points by time
            all_points = []
            for idx, price in buy_points:
                all_points.append((idx, price, 'buy'))
            for idx, price in sell_points:
                all_points.append((idx, price, 'sell'))
            
            all_points.sort(key=lambda x: x[0])  # Sort by time index
            
            entry_price = None
            
            for idx, price, action in all_points:
                if action == 'buy' and position == 0.0:
                    # Enter long position
                    position = 1.0
                    entry_price = price
                    
                elif action == 'sell' and position == 1.0 and entry_price is not None:
                    # Exit long position
                    trade_return = (price - entry_price) / entry_price
                    cumulative_return += trade_return
                    position = 0.0
                    entry_price = None
            
            return cumulative_return
            
        except Exception as e:
            logger.error(f"Error in perfect timing calculation: {e}")
            return 0.0
    
    def get_chunk_optimal_pnl(self, chunk_index: int) -> Dict[str, float]:
        """
        Get optimal PnL for a specific chunk.
        
        Args:
            chunk_index: Index of the chunk
            
        Returns:
            Dictionary mapping asset symbols to optimal PnL percentages
        """
        if chunk_index >= self.total_chunks:
            raise IndexError(f"Chunk index {chunk_index} out of range")
        
        # Load chunk if not already loaded
        if self.current_chunk_data is None or self.current_chunk_index != chunk_index:
            chunk_data = self.load_chunk_optimized(chunk_index)
            self.current_chunk_index = chunk_index
        else:
            chunk_data = self.current_chunk_data
        
        return self.calculate_optimal_pnl(chunk_data)
