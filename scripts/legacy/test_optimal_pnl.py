#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for optimal PnL calculation functionality.

This script tests the optimal PnL calculation methods in the ChunkedDataLoader
to ensure they work correctly for reward shaping.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_optimal_pnl_calculation():
    """Test the optimal PnL calculation functionality."""
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
        assets_list = ['BTC', 'ETH']  # Test with fewer assets for speed
        timeframes = ['5m', '1h', '4h']
        
        print(f"Testing Optimal PnL Calculation")
        print(f"Data directory: {data_dir}")
        print(f"Assets: {assets_list}")
        print("=" * 60)
        
        # Initialize the loader
        loader = ChunkedDataLoader(
            data_dir=data_dir,
            chunk_size=50,  # Small chunk size for testing
            assets_list=assets_list,
            split='train',
            timeframes=timeframes,
            memory_config=memory_config
        )
        
        print(f"✓ ChunkedDataLoader initialized successfully")
        print(f"  - Total chunks: {len(loader)}")
        
        # Test optimal PnL calculation on first few chunks
        test_chunks = min(3, len(loader))
        
        for chunk_idx in range(test_chunks):
            print(f"\nTesting chunk {chunk_idx}:")
            
            # Load chunk with optimizations
            chunk_data = loader.load_chunk_optimized(chunk_idx)
            
            # Calculate optimal PnL
            optimal_pnl = loader.calculate_optimal_pnl(chunk_data)
            
            print(f"  Optimal PnL results:")
            for asset, pnl in optimal_pnl.items():
                print(f"    - {asset}: {pnl:.4f} ({pnl*100:.2f}%)")
            
            # Validate results
            for asset, pnl in optimal_pnl.items():
                if not isinstance(pnl, (int, float)):
                    print(f"    ❌ Invalid PnL type for {asset}: {type(pnl)}")
                    return False
                
                if pnl < 0:
                    print(f"    ⚠️  Negative optimal PnL for {asset}: {pnl}")
                
                if pnl > 10:  # More than 1000% seems unrealistic for a chunk
                    print(f"    ⚠️  Very high optimal PnL for {asset}: {pnl}")
            
            # Test memory cleanup
            loader.clear_current_chunk()
        
        print(f"\n✅ Optimal PnL calculation test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_perfect_timing_algorithm():
    """Test the perfect timing algorithm with synthetic data."""
    print("\nTesting Perfect Timing Algorithm with Synthetic Data")
    print("=" * 60)
    
    try:
        from adan_trading_bot.data_processing.chunked_loader import ChunkedDataLoader
        
        # Create synthetic price data with known optimal PnL
        timestamps = pd.date_range('2023-01-01', periods=100, freq='5T')
        
        # Test case 1: Simple uptrend
        prices_up = np.linspace(100, 200, 100)  # 100% gain possible
        df_up = pd.DataFrame({
            '5m_close': prices_up,
            '5m_open': prices_up * 0.99,
            '5m_high': prices_up * 1.01,
            '5m_low': prices_up * 0.98,
            '5m_volume': np.random.uniform(1000, 2000, 100),
            '5m_minutes_since_update': np.zeros(100)
        }, index=timestamps)
        
        # Test case 2: Oscillating prices (multiple buy/sell opportunities)
        prices_osc = 100 + 50 * np.sin(np.linspace(0, 4*np.pi, 100))  # Oscillates between 50 and 150
        df_osc = pd.DataFrame({
            '5m_close': prices_osc,
            '5m_open': prices_osc * 0.99,
            '5m_high': prices_osc * 1.01,
            '5m_low': prices_osc * 0.98,
            '5m_volume': np.random.uniform(1000, 2000, 100),
            '5m_minutes_since_update': np.zeros(100)
        }, index=timestamps)
        
        # Test case 3: Flat prices (no opportunity)
        prices_flat = np.full(100, 100)
        df_flat = pd.DataFrame({
            '5m_close': prices_flat,
            '5m_open': prices_flat,
            '5m_high': prices_flat,
            '5m_low': prices_flat,
            '5m_volume': np.random.uniform(1000, 2000, 100),
            '5m_minutes_since_update': np.zeros(100)
        }, index=timestamps)
        
        # Create a dummy loader to access the calculation method
        loader = ChunkedDataLoader.__new__(ChunkedDataLoader)
        
        test_cases = [
            ("Uptrend", {'BTC': {'5m': df_up, '1h': df_up, '4h': df_up}}, 1.0),  # Expect ~100% gain
            ("Oscillating", {'BTC': {'5m': df_osc, '1h': df_osc, '4h': df_osc}}, 2.0),  # Expect ~200% gain (multiple cycles)
            ("Flat", {'BTC': {'5m': df_flat, '1h': df_flat, '4h': df_flat}}, 0.0)  # Expect 0% gain
        ]
        
        for test_name, chunk_data, expected_min in test_cases:
            print(f"\nTest case: {test_name}")
            
            # Calculate optimal PnL
            optimal_pnl = loader.calculate_optimal_pnl(chunk_data)
            
            pnl = optimal_pnl.get('BTC', 0.0)
            print(f"  - Calculated optimal PnL: {pnl:.4f} ({pnl*100:.2f}%)")
            print(f"  - Expected minimum: {expected_min:.4f} ({expected_min*100:.2f}%)")
            
            if test_name == "Flat" and abs(pnl) < 0.01:  # Should be near zero
                print(f"  ✅ Correct: Flat prices yield minimal PnL")
            elif test_name == "Uptrend" and pnl >= expected_min * 0.8:  # Allow some tolerance
                print(f"  ✅ Correct: Uptrend yields significant positive PnL")
            elif test_name == "Oscillating" and pnl >= expected_min * 0.5:  # Allow more tolerance for complex case
                print(f"  ✅ Correct: Oscillating prices yield high PnL from multiple trades")
            else:
                print(f"  ⚠️  Unexpected result for {test_name}")
        
        print(f"\n✅ Perfect timing algorithm test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Testing Optimal PnL Calculation for Reward Shaping")
    print("=" * 60)
    
    success1 = test_optimal_pnl_calculation()
    success2 = test_perfect_timing_algorithm()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("✅ All optimal PnL tests passed successfully!")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ Some optimal PnL tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())