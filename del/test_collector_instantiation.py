import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

try:
    from adan_trading_bot.dashboard.real_collector import RealDataCollector
    collector = RealDataCollector()
    print("✅ RealDataCollector instantiated successfully")
    
    if collector.connect():
        print("✅ RealDataCollector connected successfully")
    else:
        print("❌ RealDataCollector failed to connect")
        
except TypeError as e:
    print(f"❌ TypeError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
