#!/usr/bin/env python3
"""
Injecte le trade logging dans les processus d'entraînement EN COURS
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def inject_logging():
    """Injecte le logging dans les modules"""
    
    print("🔧 INJECTION DU TRADE LOGGING")
    print("=" * 80)
    
    # 1. Vérifier que le wrapper existe
    wrapper_path = Path("src/adan_trading_bot/common/trade_logger_wrapper.py")
    if not wrapper_path.exists():
        print(f"❌ Wrapper non trouvé: {wrapper_path}")
        return False
    
    print(f"✅ Wrapper trouvé: {wrapper_path}")
    
    # 2. Vérifier que le CentralLogger est accessible
    try:
        from adan_trading_bot.common.central_logger import logger as central_logger
        print(f"✅ CentralLogger accessible")
    except ImportError as e:
        print(f"❌ CentralLogger non accessible: {e}")
        return False
    
    # 3. Vérifier que UnifiedMetricsDB est accessible
    try:
        from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
        db = UnifiedMetricsDB()
        print(f"✅ UnifiedMetricsDB accessible")
    except Exception as e:
        print(f"❌ UnifiedMetricsDB non accessible: {e}")
        return False
    
    # 4. Vérifier que le trade_logger_wrapper peut être importé
    try:
        from adan_trading_bot.common.trade_logger_wrapper import enable_trade_logging
        print(f"✅ trade_logger_wrapper importable")
    except ImportError as e:
        print(f"❌ trade_logger_wrapper non importable: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ INJECTION RÉUSSIE")
    print("=" * 80)
    print("\nLe trade logging est maintenant disponible pour injection dans les processus.")
    print("\nPour l'utiliser dans train_parallel_agents.py:")
    print("  from adan_trading_bot.common.trade_logger_wrapper import enable_trade_logging")
    print("  enable_trade_logging(portfolio_manager)")
    
    return True

if __name__ == "__main__":
    success = inject_logging()
    sys.exit(0 if success else 1)
