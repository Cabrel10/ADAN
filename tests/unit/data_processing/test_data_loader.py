#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import os
from pathlib import Path
from src.adan_trading_bot.data_processing.data_loader import load_data, normalize_data

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path('./test_data')
        self.test_dir.mkdir(exist_ok=True)
        self.merged_data_dir = self.test_dir / 'merged'
        self.merged_data_dir.mkdir(exist_ok=True)

        # Create dummy data files
        self.dummy_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        self.dummy_df.to_parquet(self.merged_data_dir / 'train_merged.parquet')
        self.dummy_df.to_parquet(self.merged_data_dir / 'validation_merged.parquet')

        self.scaler_path = self.test_dir / 'scalers' / 'test_scaler.joblib'

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_load_data(self):
        df = load_data('train', str(self.merged_data_dir))
        self.assertFalse(df.empty)
        pd.testing.assert_frame_equal(df, self.dummy_df)

    def test_normalize_data(self):
        df_normalized, scaler = normalize_data(self.dummy_df, str(self.scaler_path))
        self.assertFalse(df_normalized.empty)
        self.assertIsNotNone(scaler)
        self.assertTrue(self.scaler_path.exists())

if __name__ == '__main__':
    unittest.main()
