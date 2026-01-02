"""
Professional trading color palette for ADAN Dashboard

Defines color schemes for P&L, confidence, risk, and other metrics.
Uses professional trading colors (not neon).
"""

from enum import Enum
from typing import Tuple


class TradingColor(Enum):
    """Professional trading color palette"""
    
    # Profit colors (green spectrum)
    PROFIT_LARGE = "#00d084"      # Bright green for >2% profit
    PROFIT_SMALL = "#00b86f"      # Light green for 0-2% profit
    
    # Loss colors (red spectrum)
    LOSS_SMALL = "#ff6b6b"        # Light red for 0-2% loss
    LOSS_LARGE = "#cc0000"        # Dark red for >2% loss
    
    # Neutral colors
    BREAKEVEN = "#ffd700"         # Gold for breakeven (-0.5% to +0.5%)
    NEUTRAL = "#808080"           # Gray for neutral
    
    # Confidence colors (blue spectrum)
    CONFIDENCE_VERY_HIGH = "#0033cc"  # Dark blue for 0.9-1.0
    CONFIDENCE_HIGH = "#0066ff"       # Blue for 0.8-0.9
    CONFIDENCE_MODERATE = "#00cc00"   # Green for 0.7-0.8
    CONFIDENCE_LOW = "#ffcc00"        # Yellow for 0.6-0.7
    CONFIDENCE_VERY_LOW = "#ff3333"   # Red for 0.0-0.6
    
    # Risk colors
    RISK_LOW = "#00cc00"          # Green for <1%
    RISK_MEDIUM = "#ffcc00"       # Yellow for 1-2%
    RISK_HIGH = "#ff9900"         # Orange for 2-3%
    RISK_CRITICAL = "#ff0000"     # Red for >3%
    
    # Signal colors
    SIGNAL_BUY = "#00cc00"        # Green for BUY
    SIGNAL_SELL = "#ff0000"       # Red for SELL
    SIGNAL_HOLD = "#ffcc00"       # Yellow for HOLD
    
    # Status colors
    STATUS_OK = "#00cc00"         # Green for OK
    STATUS_WARNING = "#ffcc00"    # Yellow for warning
    STATUS_ERROR = "#ff0000"      # Red for error
    
    # Background colors (subtle)
    BG_PROFIT = "#001a00"         # Very dark green
    BG_LOSS = "#1a0000"           # Very dark red
    BG_NEUTRAL = "#0a0a0a"        # Very dark gray
    BG_SIGNAL = "#001a1a"         # Very dark cyan


def get_pnl_color(pnl_percent: float) -> str:
    """
    Get color for P&L percentage.
    
    Args:
        pnl_percent: P&L as percentage
    
    Returns:
        Color code (hex string)
    """
    if pnl_percent > 2.0:
        return TradingColor.PROFIT_LARGE.value
    elif pnl_percent >= 0.5:
        return TradingColor.PROFIT_SMALL.value
    elif pnl_percent >= -0.5:
        return TradingColor.BREAKEVEN.value
    elif pnl_percent >= -2.0:
        return TradingColor.LOSS_SMALL.value
    else:
        return TradingColor.LOSS_LARGE.value


def get_confidence_color(confidence: float) -> str:
    """
    Get color for confidence score.
    
    Args:
        confidence: Confidence score (0.0-1.0)
    
    Returns:
        Color code (hex string)
    """
    if confidence >= 0.9:
        return TradingColor.CONFIDENCE_VERY_HIGH.value
    elif confidence >= 0.8:
        return TradingColor.CONFIDENCE_HIGH.value
    elif confidence >= 0.7:
        return TradingColor.CONFIDENCE_MODERATE.value
    elif confidence > 0.6:
        return TradingColor.CONFIDENCE_LOW.value
    else:
        return TradingColor.CONFIDENCE_VERY_LOW.value


def get_risk_color(risk_percent: float) -> str:
    """
    Get color for risk percentage.
    
    Args:
        risk_percent: Risk as percentage
    
    Returns:
        Color code (hex string)
    """
    if risk_percent < 1.0:
        return TradingColor.RISK_LOW.value
    elif risk_percent < 2.0:
        return TradingColor.RISK_MEDIUM.value
    elif risk_percent < 3.0:
        return TradingColor.RISK_HIGH.value
    else:
        return TradingColor.RISK_CRITICAL.value


def get_signal_color(signal_direction: str) -> str:
    """
    Get color for trading signal.
    
    Args:
        signal_direction: Signal direction (BUY/SELL/HOLD)
    
    Returns:
        Color code (hex string)
    """
    if signal_direction == "BUY":
        return TradingColor.SIGNAL_BUY.value
    elif signal_direction == "SELL":
        return TradingColor.SIGNAL_SELL.value
    else:  # HOLD
        return TradingColor.SIGNAL_HOLD.value


def get_status_color(status: bool, warning: bool = False) -> str:
    """
    Get color for status indicator.
    
    Args:
        status: True for OK, False for error
        warning: True for warning state
    
    Returns:
        Color code (hex string)
    """
    if warning:
        return TradingColor.STATUS_WARNING.value
    elif status:
        return TradingColor.STATUS_OK.value
    else:
        return TradingColor.STATUS_ERROR.value


def get_pnl_background(pnl_percent: float) -> str:
    """
    Get background color for P&L.
    
    Args:
        pnl_percent: P&L as percentage
    
    Returns:
        Background color code (hex string)
    """
    if pnl_percent > 0:
        return TradingColor.BG_PROFIT.value
    elif pnl_percent < 0:
        return TradingColor.BG_LOSS.value
    else:
        return TradingColor.BG_NEUTRAL.value


# Rich style definitions for easy use
RICH_STYLES = {
    "profit_large": f"bold {TradingColor.PROFIT_LARGE.value}",
    "profit_small": f"bold {TradingColor.PROFIT_SMALL.value}",
    "loss_small": f"bold {TradingColor.LOSS_SMALL.value}",
    "loss_large": f"bold {TradingColor.LOSS_LARGE.value}",
    "breakeven": f"bold {TradingColor.BREAKEVEN.value}",
    "neutral": f"dim {TradingColor.NEUTRAL.value}",
    "confidence_very_high": f"bold {TradingColor.CONFIDENCE_VERY_HIGH.value}",
    "confidence_high": f"bold {TradingColor.CONFIDENCE_HIGH.value}",
    "confidence_moderate": f"bold {TradingColor.CONFIDENCE_MODERATE.value}",
    "confidence_low": f"bold {TradingColor.CONFIDENCE_LOW.value}",
    "confidence_very_low": f"bold {TradingColor.CONFIDENCE_VERY_LOW.value}",
    "risk_low": f"bold {TradingColor.RISK_LOW.value}",
    "risk_medium": f"bold {TradingColor.RISK_MEDIUM.value}",
    "risk_high": f"bold {TradingColor.RISK_HIGH.value}",
    "risk_critical": f"bold {TradingColor.RISK_CRITICAL.value}",
    "signal_buy": f"bold {TradingColor.SIGNAL_BUY.value}",
    "signal_sell": f"bold {TradingColor.SIGNAL_SELL.value}",
    "signal_hold": f"bold {TradingColor.SIGNAL_HOLD.value}",
    "status_ok": f"bold {TradingColor.STATUS_OK.value}",
    "status_warning": f"bold {TradingColor.STATUS_WARNING.value}",
    "status_error": f"bold {TradingColor.STATUS_ERROR.value}",
}
