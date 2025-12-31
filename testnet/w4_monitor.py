#!/usr/bin/env python3
"""
W4 Monitor - Analyse chronologique step-by-step
Capture les données réelles et les analyse
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

os.environ["BINANCE_TESTNET_API_KEY"] = "OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW"
os.environ["BINANCE_TESTNET_API_SECRET"] = "wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
logging.basicConfig(level=logging.WARNING)

from adan_trading_bot.exchange_api.connector import get_exchange_client
from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.indicators.calculator import IndicatorCalculator
from adan_trading_bot.observation.builder import ObservationBuilder
from adan_trading_bot.validation.data_validator import DataValidator
from stable_baselines3 import PPO
import pandas as pd
import numpy as np

class W4Monitor:
    def __init__(self):
        self.worker_id = "w4"
        self.model_path = Path(__file__).parent.parent / "models" / "w4_final.zip"
        self.steps = []
        self.errors = []
        
    def setup(self):
        """Setup components"""
        try:
            # Exchange
            config = {'paper_trading': {'api_key': os.environ["BINANCE_TESTNET_API_KEY"], 'api_secret': os.environ["BINANCE_TESTNET_API_SECRET"]}}
            self.exchange = get_exchange_client(config)
            
            # Model
            self.model = PPO.load(str(self.model_path))
            
            # Components
            self.state_builder = StateBuilder()
            self.indicator_calc = IndicatorCalculator()
            self.obs_builder = ObservationBuilder()
            self.data_validator = DataValidator()
            
            return True
        except Exception as e:
            self.errors.append(f"Setup failed: {e}")
            return False
    
    def run_step(self, step_num):
        """Execute one step and capture data"""
        step_data = {
            "step": step_num,
            "timestamp": datetime.now().isoformat(),
            "status": "UNKNOWN",
            "data": {},
            "errors": []
        }
        
        try:
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '5m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            step_data["data"]["price"] = float(df['close'].iloc[-1])
            step_data["data"]["volume"] = float(df['volume'].iloc[-1])
            
            # Validate
            if not self.data_validator.validate(df):
                step_data["errors"].append("Data validation failed")
                step_data["status"] = "VALIDATION_FAILED"
                return step_data
            
            # Build observation
            indicators = self.indicator_calc.calculate(df)
            obs = self.obs_builder.build(df, indicators)
            
            if obs is None:
                step_data["errors"].append("Observation building failed")
                step_data["status"] = "OBS_FAILED"
                return step_data
            
            # Inference
            action, _ = self.model.predict(obs, deterministic=False)
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            
            step_data["data"]["action"] = action_value
            step_data["data"]["action_type"] = "BUY" if action_value > 0.5 else ("SELL" if action_value < -0.5 else "HOLD")
            step_data["status"] = "SUCCESS"
            
        except Exception as e:
            step_data["errors"].append(str(e))
            step_data["status"] = "ERROR"
        
        return step_data
    
    def run(self, num_steps=12):
        """Run monitoring for N steps"""
        print(f"\n{'='*80}")
        print(f"W4 CHRONOLOGICAL ANALYSIS - {num_steps} STEPS")
        print(f"{'='*80}\n")
        
        if not self.setup():
            print("❌ Setup failed")
            return False
        
        print("✅ Setup complete\n")
        
        for i in range(num_steps):
            step_data = self.run_step(i)
            self.steps.append(step_data)
            
            # Print step result
            status_icon = "✅" if step_data["status"] == "SUCCESS" else "❌"
            print(f"{status_icon} Step {i+1:2d} | {step_data['timestamp']} | {step_data['status']}")
            
            if step_data["status"] == "SUCCESS":
                print(f"         Price: ${step_data['data']['price']:.2f} | Action: {step_data['data']['action_type']} ({step_data['data']['action']:.4f})")
            
            if step_data["errors"]:
                for err in step_data["errors"]:
                    print(f"         ⚠️  {err}")
            
            time.sleep(5)  # 5 min interval
        
        self.analyze()
        return True
    
    def analyze(self):
        """Analyze results"""
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}\n")
        
        success_count = sum(1 for s in self.steps if s["status"] == "SUCCESS")
        error_count = sum(1 for s in self.steps if s["status"] == "ERROR")
        
        print(f"Total steps: {len(self.steps)}")
        print(f"Success: {success_count}/{len(self.steps)}")
        print(f"Errors: {error_count}/{len(self.steps)}")
        
        if success_count > 0:
            actions = [s["data"]["action"] for s in self.steps if s["status"] == "SUCCESS"]
            print(f"\nAction statistics:")
            print(f"  Mean: {np.mean(actions):.4f}")
            print(f"  Std: {np.std(actions):.4f}")
            print(f"  Min: {np.min(actions):.4f}")
            print(f"  Max: {np.max(actions):.4f}")
            
            action_types = defaultdict(int)
            for s in self.steps:
                if s["status"] == "SUCCESS":
                    action_types[s["data"]["action_type"]] += 1
            
            print(f"\nAction distribution:")
            for action_type, count in action_types.items():
                print(f"  {action_type}: {count}")
        
        # Save results
        results = {
            "worker": "w4",
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(self.steps),
            "success": success_count,
            "errors": error_count,
            "steps": self.steps
        }
        
        results_path = Path(__file__).parent / "w4_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to {results_path}")

if __name__ == "__main__":
    monitor = W4Monitor()
    monitor.run(num_steps=12)
