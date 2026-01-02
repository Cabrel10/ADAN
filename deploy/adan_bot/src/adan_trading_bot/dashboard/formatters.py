"""
Formatting utilities for ADAN Dashboard

Provides consistent formatting for numbers, currencies, times, and other values.
"""

from datetime import timedelta
from typing import Union


def format_usd(value: float, decimals: int = 2) -> str:
    """
    Format value as USD currency.
    
    Args:
        value: Value to format
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted string (e.g., "$1,234.56")
    """
    if value < 0:
        return f"-${abs(value):,.{decimals}f}"
    return f"${value:,.{decimals}f}"


def format_btc(value: float, decimals: int = 4) -> str:
    """
    Format value as BTC.
    
    Args:
        value: Value in BTC
        decimals: Number of decimal places (default: 4)
    
    Returns:
        Formatted string (e.g., "0.0245")
    """
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 2, include_sign: bool = True) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value as percentage (e.g., 2.5 for 2.5%)
        decimals: Number of decimal places (default: 2)
        include_sign: Include + sign for positive values (default: True)
    
    Returns:
        Formatted string (e.g., "+2.50%" or "2.50%")
    """
    if include_sign:
        if value > 0:
            return f"+{value:.{decimals}f}%"
        elif value < 0:
            return f"{value:.{decimals}f}%"
        else:
            return f"{0:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}%"


def format_time(duration: Union[timedelta, float]) -> str:
    """
    Format duration as HH:MM:SS or HH:MM.
    
    Args:
        duration: timedelta object or seconds as float
    
    Returns:
        Formatted string (e.g., "2h14m" or "02:14:30")
    """
    if isinstance(duration, timedelta):
        total_seconds = int(duration.total_seconds())
    else:
        total_seconds = int(duration)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}h{minutes}m"
    elif minutes > 0:
        return f"{minutes}m{seconds}s"
    else:
        return f"{seconds}s"


def format_time_hms(duration: Union[timedelta, float]) -> str:
    """
    Format duration as HH:MM:SS.
    
    Args:
        duration: timedelta object or seconds as float
    
    Returns:
        Formatted string (e.g., "02:14:30")
    """
    if isinstance(duration, timedelta):
        total_seconds = int(duration.total_seconds())
    else:
        total_seconds = int(duration)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_confidence(value: float, decimals: int = 2) -> str:
    """
    Format confidence score (0.0-1.0).
    
    Args:
        value: Confidence value (0.0-1.0)
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted string (e.g., "0.87")
    """
    return f"{value:.{decimals}f}"


def format_price(value: float, decimals: int = 2) -> str:
    """
    Format price value.
    
    Args:
        value: Price value
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted string (e.g., "43,217.50")
    """
    return f"{value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format generic number with thousands separator.
    
    Args:
        value: Number to format
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted string (e.g., "1,234.56")
    """
    return f"{value:,.{decimals}f}"


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format ratio value (e.g., profit factor).
    
    Args:
        value: Ratio value
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted string (e.g., "1.63")
    """
    return f"{value:.{decimals}f}"


def format_rsi_level(rsi: int) -> str:
    """
    Get RSI level classification.
    
    Args:
        rsi: RSI value (0-100)
    
    Returns:
        Level string (Oversold/Neutral/Overbought)
    """
    if rsi < 30:
        return "Oversold"
    elif rsi > 70:
        return "Overbought"
    else:
        return "Neutral"


def format_adx_strength(adx: int) -> str:
    """
    Get ADX trend strength classification.
    
    Args:
        adx: ADX value (0-100)
    
    Returns:
        Strength string (Weak/Moderate/Strong)
    """
    if adx < 25:
        return "Weak"
    elif adx < 50:
        return "Moderate"
    else:
        return "Strong"


def format_outcome_symbol(is_win: bool, is_breakeven: bool = False) -> str:
    """
    Get outcome symbol for trade.
    
    Args:
        is_win: True if trade was profitable
        is_breakeven: True if trade was breakeven
    
    Returns:
        Symbol string (✅/❌/⚠️)
    """
    if is_breakeven:
        return "⚠️"
    elif is_win:
        return "✅"
    else:
        return "❌"


def format_status_symbol(status: bool, warning: bool = False) -> str:
    """
    Get status symbol.
    
    Args:
        status: True for OK, False for error
        warning: True for warning state
    
    Returns:
        Symbol string (✅/⚠️/❌)
    """
    if warning:
        return "⚠️"
    elif status:
        return "✅"
    else:
        return "❌"


def format_signal_symbol(direction: str) -> str:
    """
    Get signal symbol.
    
    Args:
        direction: Signal direction (BUY/SELL/HOLD)
    
    Returns:
        Symbol string (🟢/🔴/🟡)
    """
    if direction == "BUY":
        return "🟢"
    elif direction == "SELL":
        return "🔴"
    else:  # HOLD
        return "🟡"


# Utility functions for common formatting patterns

def format_pnl_display(pnl_usd: float, pnl_pct: float) -> str:
    """
    Format P&L for display.
    
    Args:
        pnl_usd: P&L in USD
        pnl_pct: P&L as percentage
    
    Returns:
        Formatted string (e.g., "+$192 (+1.44%)")
    """
    return f"{format_usd(pnl_usd)} ({format_percentage(pnl_pct)})"


def format_position_size_display(size_btc: float, price: float) -> str:
    """
    Format position size for display.
    
    Args:
        size_btc: Size in BTC
        price: Current price
    
    Returns:
        Formatted string (e.g., "0.0245 BTC / $1,074.33")
    """
    value_usd = size_btc * price
    return f"{format_btc(size_btc)} BTC / {format_usd(value_usd)}"


def format_price_range(entry: float, current: float, sl: float, tp: float) -> str:
    """
    Format price range for display.
    
    Args:
        entry: Entry price
        current: Current price
        sl: Stop loss price
        tp: Take profit price
    
    Returns:
        Formatted string showing the range
    """
    return f"SL: {format_price(sl)} | Entry: {format_price(entry)} | Current: {format_price(current)} | TP: {format_price(tp)}"
