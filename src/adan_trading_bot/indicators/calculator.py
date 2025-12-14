"""
Indicator Calculator - Correct RSI, ADX, ATR formulas using standard technical analysis methods.

This module implements mathematically correct indicator calculations using Wilder's smoothing
for RSI and ADX, and standard true range smoothing for ATR. All indicators use 14-period
calculations as the standard.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """Calculate technical indicators using correct formulas."""
    
    DEFAULT_PERIOD = 14
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = DEFAULT_PERIOD) -> float:
        """
        Calculate RSI (Relative Strength Index) using Wilder's smoothing method.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Args:
            prices: Array of closing prices
            period: Lookback period (default 14)
            
        Returns:
            RSI value between 0 and 100
            
        Raises:
            ValueError: If insufficient data or invalid input
        """
        if len(prices) < period + 1:
            raise ValueError(f"Need at least {period + 1} prices, got {len(prices)}")
        
        prices = np.asarray(prices, dtype=np.float64)
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initial average gain and loss (first period)
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        
        # Wilder's smoothing for subsequent periods
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Calculate RS and RSI
        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        logger.debug(f"RSI calculated: {rsi:.2f} (avg_gain={avg_gain:.6f}, avg_loss={avg_loss:.6f})")
        return float(rsi)
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = DEFAULT_PERIOD) -> float:
        """
        Calculate ATR (Average True Range) using standard smoothing.
        
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = SMA of True Range
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: Lookback period (default 14)
            
        Returns:
            ATR value in price units
            
        Raises:
            ValueError: If insufficient data or invalid input
        """
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            raise ValueError(f"Need at least {period + 1} prices, got {len(high)}")
        
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        
        # Calculate true range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using Wilder's smoothing
        atr = tr[:period].mean()
        
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + tr[i]) / period
        
        logger.debug(f"ATR calculated: {atr:.6f}")
        return float(atr)
    
    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int = DEFAULT_PERIOD) -> float:
        """
        Calculate ADX (Average Directional Index) using directional movement method.
        
        ADX measures trend strength from 0 to 100.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: Lookback period (default 14)
            
        Returns:
            ADX value between 0 and 100
            
        Raises:
            ValueError: If insufficient data or invalid input
        """
        if len(high) < 2 * period + 1 or len(low) < 2 * period + 1 or len(close) < 2 * period + 1:
            raise ValueError(f"Need at least {2 * period + 1} prices for ADX, got {len(high)}")
        
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        
        # Calculate directional movements
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        # Determine +DM and -DM
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate true range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Smooth using Wilder's method
        def wilder_smooth(values, period):
            smoothed = np.zeros(len(values))
            smoothed[period - 1] = values[:period].sum()
            
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / period) + values[i]
            
            return smoothed / period
        
        # Calculate smoothed values
        plus_di_values = wilder_smooth(plus_dm, period)
        minus_di_values = wilder_smooth(minus_dm, period)
        tr_values = wilder_smooth(tr, period)
        
        # Calculate DI values
        plus_di = 100 * plus_di_values / (tr_values + 1e-10)
        minus_di = 100 * minus_di_values / (tr_values + 1e-10)
        
        # Calculate DX
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
        
        # Calculate ADX (smooth DX)
        adx = wilder_smooth(dx, period)[-1]
        
        logger.debug(f"ADX calculated: {adx:.2f}")
        return float(np.clip(adx, 0, 100))
    
    @staticmethod
    def calculate_all(ohlcv: pd.DataFrame, period: int = DEFAULT_PERIOD) -> Dict[str, float]:
        """
        Calculate all indicators from OHLCV data.
        
        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume
            period: Lookback period (default 14)
            
        Returns:
            Dictionary with keys: rsi, adx, atr, atr_percent
            
        Raises:
            ValueError: If insufficient data or missing columns
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in ohlcv.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if len(ohlcv) < 2 * period + 1:
            raise ValueError(f"Need at least {2 * period + 1} rows, got {len(ohlcv)}")
        
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        
        try:
            rsi = IndicatorCalculator.calculate_rsi(close, period)
            atr = IndicatorCalculator.calculate_atr(high, low, close, period)
            adx = IndicatorCalculator.calculate_adx(high, low, close, period)
            
            # Calculate ATR as percentage of current price
            current_price = close[-1]
            atr_percent = (atr / current_price * 100) if current_price != 0 else 0
            
            result = {
                'rsi': rsi,
                'adx': adx,
                'atr': atr,
                'atr_percent': atr_percent
            }
            
            logger.info(f"Indicators calculated: RSI={rsi:.2f}, ADX={adx:.2f}, ATR={atr:.6f}, ATR%={atr_percent:.4f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
