#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import pandas as pd
from src.adan_trading_bot.data_processing.feature_engineer import calculate_technical_indicators, handle_missing_values

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.dummy_df = pd.DataFrame({
            'open': [10, 12, 15, 13, 11],
            'high': [16, 17, 18, 14, 12],
            'low': [9, 11, 13, 10, 9],
            'close': [12, 15, 13, 11, 10],
            'volume': [100, 120, 110, 130, 105]
        })

    def test_calculate_technical_indicators(self):
        indicators = ['rsi', 'macd']
        df_with_indicators = calculate_technical_indicators(self.dummy_df.copy(), indicators)
        self.assertIn('RSI_14', df_with_indicators.columns)
        self.assertIn('MACD_12_26_9', df_with_indicators.columns)

    def test_handle_missing_values(self):
        df_with_nan = self.dummy_df.copy()
        df_with_nan.loc[2, 'close'] = None
        df_handled = handle_missing_values(df_with_nan)
        self.assertFalse(df_handled.isnull().any().any())

if __name__ == '__main__':
    unittest.main()
