#!/usr/bin/env python
"""
Profile script to identify bottleneck in StateBuilder.build_observation()
Uses line_profiler to get line-by-line timing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from line_profiler import LineProfiler
from adan_trading_bot.data_processing.state_builder import StateBuilder
from scripts.run_paper_trading import BinancePaperTrader

# Initialize bot
trader = BinancePaperTrader(
    config_path="config/config.yaml",
    models_dir="models/ppo_ensemble",
    strategy="median"
)

# Prepare data once
data = trader.prepare_data()

# Profile build_observation
lp = LineProfiler()
lp.add_function(trader.state_builder.build_observation)
lp.add_function(trader.state_builder.build_state)

# Wrap and run
lp_wrapper = lp(trader.state_builder.build_observation)

# Mock portfolio manager
class MockPM:
    def get_state_vector(self, *args, **kwargs):
        return trader.get_portfolio_state()

result = lp_wrapper(
    current_idx=len(data['BTCUSDT']['5m']) - 1,
    data=data,
    portfolio_manager=MockPM()
)

# Print stats
lp.print_stats()
