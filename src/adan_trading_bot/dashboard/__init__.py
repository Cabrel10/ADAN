"""
ADAN BTC/USDT Terminal Dashboard Package

Professional real-time monitoring interface for ADAN trading bot.
"""

__version__ = "1.0.0"
__author__ = "ADAN Team"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "AdanBtcDashboard":
        from .app import AdanBtcDashboard
        return AdanBtcDashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AdanBtcDashboard"]
