#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the ChunkedDataLoader with memory optimizations.

This script tests the ChunkedDataLoader to ensure it works correctly
with the existing data and memory optimization features.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_chunked_loader():
    """Test the ChunkedDataLoader functionality."""
    try:
        from adan_trading_bot.data_processing.chunked_loader import ChunkedDataLoader
        
        # Configuration for memory optimization
        memory_config = {
            'aggressive_cleanup': True,
            'force_gc': True,
            'memory_monitoring': True,
            'disable_caching': True,
            'memory_warning_threshold_mb': 5600,
            'memory_critical_threshold_mb': 6300
        }
        
        # Test with existing data
        data_dir = Path(__file__).parent.parent / 'data' / 'final'
        assets_list = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']
        timeframes = ['5m', '1h', '4h']
        
        print(f"Testing ChunkedDataLoader with data from: {data_dir}")
        print(f"Assets: {assets_list}")
        print(f"Timeframes: {timeframes}")
        print("=" * 60)
        
        # Create config dictionary
        config = {
            'paths': {
                'indicators_data_dir': str(Path(__file__).parent.parent / 'data' / 'processed' / 'indicators'),
            },
            'data': {
                'chunked_loader': {
                    'chunk_size': 100,  # Small chunk size for testing
                },
                'assets': assets_list,
                'timeframes': timeframes,
                'memory_optimizations': memory_config
            },
            'environment': {
                'state': {
                    'features_per_timeframe': {
                        '5m': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'RSI_14'],
                        '1h': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'MACD_12_26_9'],
                        '4h': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'BBL_20_2.0']
                    }
                }
            }
        }
        
        # Initialize the loader with config
        loader = ChunkedDataLoader(config=config)
        
        print(f"✓ ChunkedDataLoader initialized successfully")
        print(f"  - Total chunks: {len(loader)}")
        print(f"  - Assets loaded: {len(loader.assets_list)}")
        print(f"  - Caching disabled: {loader.is_caching_disabled()}")
        
        # Test memory stats
        memory_stats = loader.get_memory_stats()
        print(f"  - Initial memory: {memory_stats['initial_memory_mb']:.1f} MB")
        
        # Test cache info
        cache_info = loader.get_cache_info()
        print(f"  - Cache status: {cache_info}")
        
        # Test loading a chunk with optimizations
        if len(loader) > 0:
            print(f"\nTesting optimized chunk loading...")
            chunk_data = loader.load_chunk_optimized(0)
            
            print(f"✓ Chunk 0 loaded successfully")
            print(f"  - Assets in chunk: {list(chunk_data.keys())}")
            
            # Check data structure
            for asset, timeframe_data in chunk_data.items():
                print(f"  - {asset}: {list(timeframe_data.keys())} timeframes")
                for tf, df in timeframe_data.items():
                    if df is not None and not df.empty:
                        print(f"    - {tf}: {len(df)} rows, {len(df.columns)} columns")
                    else:
                        print(f"    - {tf}: No data")
                break  # Just show first asset for brevity
            
            # Test memory stats after loading
            memory_stats_after = loader.get_memory_stats()
            memory_increase = memory_stats_after['memory_increase_mb']
            print(f"  - Memory increase: +{memory_increase:.1f} MB")
            
            # Test cleanup
            loader.clear_current_chunk()
            print(f"✓ Chunk cleanup completed")
            
            # Test memory stats after cleanup
            memory_stats_final = loader.get_memory_stats()
            memory_final = memory_stats_final['memory_increase_mb']
            print(f"  - Memory after cleanup: +{memory_final:.1f} MB")
        
        print("\n" + "=" * 60)
        print("✅ ChunkedDataLoader test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This might be due to missing dependencies.")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Testing ChunkedDataLoader with Memory Optimizations")
    print("=" * 60)
    
    success = test_chunked_loader()
    
    if success:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())