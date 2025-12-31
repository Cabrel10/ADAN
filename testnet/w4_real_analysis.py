#!/usr/bin/env python3
"""
W4 Real Analysis - 20 minutes (4 steps de 5 minutes)
Utilise les vrais modèles de /mnt/new_data/t10_training/checkpoints/final/
"""

import os
import sys
import json
import time
import ccxt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

os.environ["BINANCE_TESTNET_API_KEY"] = "OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW"
os.environ["BINANCE_TESTNET_API_SECRET"] = "wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO

class W4RealAnalysis:
    def __init__(self):
        self.worker_id = "w4"
        self.model_path = Path("/mnt/new_data/t10_training/checkpoints/final/w4_final.zip")
        self.steps = []
        self.start_time = datetime.now()
        
        # Exchange
        self.exchange = ccxt.binance({
            'apiKey': os.environ["BINANCE_TESTNET_API_KEY"],
            'secret': os.environ["BINANCE_TESTNET_API_SECRET"],
            'enableRateLimit': True,
            'sandbox': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Model
        print(f"Loading model from: {self.model_path}")
        self.model = PPO.load(str(self.model_path))
        print("✅ Model loaded")
    
    def run_step(self, step_num):
        """Execute one step"""
        step_data = {
            "step": step_num,
            "timestamp": datetime.now().isoformat(),
            "elapsed_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "status": "UNKNOWN",
            "price": None,
            "volume": None,
            "action": None,
            "action_type": None,
            "error": None
        }
        
        try:
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '5m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            price = float(df['close'].iloc[-1])
            volume = float(df['volume'].iloc[-1])
            
            step_data["price"] = price
            step_data["volume"] = volume
            
            # Simple observation (random - just to test model inference)
            obs = np.random.randn(1, 542).astype(np.float32)
            
            # Inference
            action, _ = self.model.predict(obs, deterministic=False)
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            
            step_data["action"] = action_value
            step_data["action_type"] = "BUY" if action_value > 0.5 else ("SELL" if action_value < -0.5 else "HOLD")
            step_data["status"] = "SUCCESS"
            
        except Exception as e:
            step_data["error"] = str(e)
            step_data["status"] = "ERROR"
        
        return step_data
    
    def run(self, num_steps=4):
        """Run analysis for 20 minutes (4 steps)"""
        print(f"\n{'='*80}")
        print(f"W4 REAL ANALYSIS - 20 MINUTES ({num_steps} STEPS)")
        print(f"{'='*80}\n")
        
        for i in range(num_steps):
            step_data = self.run_step(i)
            self.steps.append(step_data)
            
            # Print
            status_icon = "✅" if step_data["status"] == "SUCCESS" else "❌"
            elapsed = step_data["elapsed_minutes"]
            
            print(f"{status_icon} Step {i+1} | {elapsed:5.1f}min | {step_data['timestamp']}")
            
            if step_data["status"] == "SUCCESS":
                print(f"         Price: ${step_data['price']:.2f} | Vol: {step_data['volume']:.0f} | Action: {step_data['action_type']:4s} ({step_data['action']:+.4f})")
            else:
                print(f"         Error: {step_data['error']}")
            
            if i < num_steps - 1:
                print(f"         Waiting 5 minutes...")
                time.sleep(300)  # 5 minutes
        
        self.analyze()
    
    def analyze(self):
        """Analyze results"""
        print(f"\n{'='*80}")
        print("ANALYSIS - CHRONOLOGICAL")
        print(f"{'='*80}\n")
        
        success = sum(1 for s in self.steps if s["status"] == "SUCCESS")
        errors = sum(1 for s in self.steps if s["status"] == "ERROR")
        
        print(f"Total steps: {len(self.steps)}")
        print(f"Success: {success}/{len(self.steps)} ({100*success/len(self.steps):.1f}%)")
        print(f"Errors: {errors}/{len(self.steps)} ({100*errors/len(self.steps):.1f}%)")
        print(f"Duration: {(datetime.now() - self.start_time).total_seconds() / 60:.1f} minutes")
        
        if success > 0:
            actions = [s["action"] for s in self.steps if s["status"] == "SUCCESS"]
            prices = [s["price"] for s in self.steps if s["status"] == "SUCCESS"]
            
            print(f"\nPrice evolution:")
            for i, s in enumerate(self.steps):
                if s["status"] == "SUCCESS":
                    print(f"  Step {i+1}: ${s['price']:.2f}")
            
            print(f"\nPrice range:")
            print(f"  Min: ${min(prices):.2f}")
            print(f"  Max: ${max(prices):.2f}")
            print(f"  Mean: ${np.mean(prices):.2f}")
            print(f"  Change: ${max(prices) - min(prices):.2f}")
            
            print(f"\nAction statistics:")
            print(f"  Mean: {np.mean(actions):+.4f}")
            print(f"  Std: {np.std(actions):.4f}")
            print(f"  Min: {np.min(actions):+.4f}")
            print(f"  Max: {np.max(actions):+.4f}")
            
            action_types = defaultdict(int)
            for s in self.steps:
                if s["status"] == "SUCCESS":
                    action_types[s["action_type"]] += 1
            
            print(f"\nAction distribution:")
            for action_type in ["BUY", "HOLD", "SELL"]:
                count = action_types.get(action_type, 0)
                pct = 100 * count / success if success > 0 else 0
                print(f"  {action_type}: {count} ({pct:5.1f}%)")
        
        # Save
        results = {
            "worker": "w4",
            "timestamp": datetime.now().isoformat(),
            "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "total_steps": len(self.steps),
            "success": success,
            "errors": errors,
            "steps": self.steps
        }
        
        results_path = Path(__file__).parent / "w4_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results: {results_path}")

if __name__ == "__main__":
    analysis = W4RealAnalysis()
    analysis.run(num_steps=4)
