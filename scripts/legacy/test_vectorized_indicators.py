#!/usr/bin/env python3
"""
Test et validation des indicateurs techniques vectorisés optimisés.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add src to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

from adan_trading_bot.utils.vectorized_indicators import (
    vectorized_sma, vectorized_ema, vectorized_rsi, vectorized_macd,
    vectorized_bollinger_bands, vectorized_atr, vectorized_stochastic,
    vectorized_williams_r, vectorized_cci, vectorized_rolling_correlation,
    vectorized_rolling_zscore, vectorized_portfolio_metrics
)


class VectorizedIndicatorTester:
    """Testeur pour les indicateurs vectorisés"""
    
    def __init__(self):
        self.test_results = {}
        self.sample_data = self._generate_test_data()
    
    def _generate_test_data(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Génère des données de test réalistes"""
        np.random.seed(42)
        
        # Prix de base avec tendance
        base_price = 50000
        price_changes = np.random.randn(n_samples) * 0.02
        prices = base_price * np.cumprod(1 + price_changes)
        
        # OHLC data
        volatility = 0.01
        high = prices * (1 + np.random.rand(n_samples) * volatility)
        low = prices * (1 - np.random.rand(n_samples) * volatility)
        open_prices = prices + np.random.randn(n_samples) * volatility * prices * 0.5
        
        # Volume
        volume = np.random.exponential(1000000, n_samples)
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])  # Add first return as 0
        
        return {
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume,
            'returns': returns
        }
    
    def test_sma(self) -> Dict[str, Any]:
        """Test SMA vectorisé"""
        print("📊 Test SMA vectorisé...")
        
        prices = self.sample_data['close']
        window = 20
        
        # Test vectorisé
        start_time = time.time()
        vectorized_result = vectorized_sma(prices, window)
        vectorized_time = time.time() - start_time
        
        # Test pandas pour comparaison
        start_time = time.time()
        pandas_result = pd.Series(prices).rolling(window).mean().values
        pandas_time = time.time() - start_time
        
        # Validation
        valid_mask = ~np.isnan(pandas_result)
        accuracy = np.allclose(
            vectorized_result[valid_mask], 
            pandas_result[valid_mask], 
            rtol=1e-10
        )
        
        return {
            'name': 'SMA',
            'vectorized_time': vectorized_time,
            'pandas_time': pandas_time,
            'speedup': pandas_time / vectorized_time if vectorized_time > 0 else 0,
            'accuracy': accuracy,
            'max_diff': np.max(np.abs(vectorized_result[valid_mask] - pandas_result[valid_mask])) if accuracy else float('inf')
        }
    
    def test_ema(self) -> Dict[str, Any]:
        """Test EMA vectorisé"""
        print("📈 Test EMA vectorisé...")
        
        prices = self.sample_data['close']
        span = 12
        
        # Test vectorisé
        start_time = time.time()
        vectorized_result = vectorized_ema(prices, span)
        vectorized_time = time.time() - start_time
        
        # Test pandas pour comparaison
        start_time = time.time()
        pandas_result = pd.Series(prices).ewm(span=span, adjust=False).mean().values
        pandas_time = time.time() - start_time
        
        # Validation
        accuracy = np.allclose(vectorized_result, pandas_result, rtol=1e-10)
        
        return {
            'name': 'EMA',
            'vectorized_time': vectorized_time,
            'pandas_time': pandas_time,
            'speedup': pandas_time / vectorized_time if vectorized_time > 0 else 0,
            'accuracy': accuracy,
            'max_diff': np.max(np.abs(vectorized_result - pandas_result)) if accuracy else float('inf')
        }
    
    def test_rsi(self) -> Dict[str, Any]:
        """Test RSI vectorisé"""
        print("🔥 Test RSI vectorisé...")
        
        prices = self.sample_data['close']
        window = 14
        
        # Test vectorisé
        start_time = time.time()
        vectorized_result = vectorized_rsi(prices, window)
        vectorized_time = time.time() - start_time
        
        # Test pandas pour comparaison (implémentation simplifiée)
        start_time = time.time()
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        pandas_result = (100 - (100 / (1 + rs))).values
        pandas_time = time.time() - start_time
        
        # Validation (RSI peut avoir de légères différences selon l'implémentation)
        valid_mask = ~np.isnan(pandas_result) & ~np.isnan(vectorized_result)
        if np.any(valid_mask):
            max_diff = np.max(np.abs(vectorized_result[valid_mask] - pandas_result[valid_mask]))
            accuracy = max_diff < 1.0  # Tolérance de 1 point RSI
        else:
            accuracy = False
            max_diff = float('inf')
        
        return {
            'name': 'RSI',
            'vectorized_time': vectorized_time,
            'pandas_time': pandas_time,
            'speedup': pandas_time / vectorized_time if vectorized_time > 0 else 0,
            'accuracy': accuracy,
            'max_diff': max_diff
        }
    
    def test_macd(self) -> Dict[str, Any]:
        """Test MACD vectorisé"""
        print("📊 Test MACD vectorisé...")
        
        prices = self.sample_data['close']
        
        # Test vectorisé
        start_time = time.time()
        macd_line, signal_line, histogram = vectorized_macd(prices, 12, 26, 9)
        vectorized_time = time.time() - start_time
        
        # Test pandas pour comparaison
        start_time = time.time()
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        pandas_macd = ema_12 - ema_26
        pandas_signal = pandas_macd.ewm(span=9, adjust=False).mean()
        pandas_histogram = pandas_macd - pandas_signal
        pandas_time = time.time() - start_time
        
        # Validation
        macd_accuracy = np.allclose(macd_line, pandas_macd.values, rtol=1e-10)
        signal_accuracy = np.allclose(signal_line, pandas_signal.values, rtol=1e-10)
        hist_accuracy = np.allclose(histogram, pandas_histogram.values, rtol=1e-10)
        
        accuracy = macd_accuracy and signal_accuracy and hist_accuracy
        
        return {
            'name': 'MACD',
            'vectorized_time': vectorized_time,
            'pandas_time': pandas_time,
            'speedup': pandas_time / vectorized_time if vectorized_time > 0 else 0,
            'accuracy': accuracy,
            'components': {
                'macd_line': macd_accuracy,
                'signal_line': signal_accuracy,
                'histogram': hist_accuracy
            }
        }
    
    def test_bollinger_bands(self) -> Dict[str, Any]:
        """Test Bollinger Bands vectorisées"""
        print("📊 Test Bollinger Bands vectorisées...")
        
        prices = self.sample_data['close']
        window = 20
        std_dev = 2.0
        
        # Test vectorisé
        start_time = time.time()
        sma, upper, lower = vectorized_bollinger_bands(prices, window, std_dev)
        vectorized_time = time.time() - start_time
        
        # Test pandas pour comparaison
        start_time = time.time()
        pandas_sma = pd.Series(prices).rolling(window).mean()
        pandas_std = pd.Series(prices).rolling(window).std()
        pandas_upper = pandas_sma + (pandas_std * std_dev)
        pandas_lower = pandas_sma - (pandas_std * std_dev)
        pandas_time = time.time() - start_time
        
        # Validation
        valid_mask = ~np.isnan(pandas_sma.values)
        sma_accuracy = np.allclose(sma[valid_mask], pandas_sma.values[valid_mask], rtol=1e-10)
        upper_accuracy = np.allclose(upper[valid_mask], pandas_upper.values[valid_mask], rtol=1e-8)
        lower_accuracy = np.allclose(lower[valid_mask], pandas_lower.values[valid_mask], rtol=1e-8)
        
        accuracy = sma_accuracy and upper_accuracy and lower_accuracy
        
        return {
            'name': 'Bollinger Bands',
            'vectorized_time': vectorized_time,
            'pandas_time': pandas_time,
            'speedup': pandas_time / vectorized_time if vectorized_time > 0 else 0,
            'accuracy': accuracy,
            'components': {
                'sma': sma_accuracy,
                'upper': upper_accuracy,
                'lower': lower_accuracy
            }
        }
    
    def test_atr(self) -> Dict[str, Any]:
        """Test ATR vectorisé"""
        print("📊 Test ATR vectorisé...")
        
        high = self.sample_data['high']
        low = self.sample_data['low']
        close = self.sample_data['close']
        window = 14
        
        # Test vectorisé
        start_time = time.time()
        vectorized_result = vectorized_atr(high, low, close, window)
        vectorized_time = time.time() - start_time
        
        # Validation basique (ATR doit être positif et non-NaN après la période initiale)
        valid_values = vectorized_result[window:]
        accuracy = np.all(valid_values > 0) and not np.any(np.isnan(valid_values))
        
        return {
            'name': 'ATR',
            'vectorized_time': vectorized_time,
            'pandas_time': 0,  # Pas de comparaison pandas directe
            'speedup': 0,
            'accuracy': accuracy,
            'min_value': np.min(valid_values) if len(valid_values) > 0 else 0,
            'max_value': np.max(valid_values) if len(valid_values) > 0 else 0
        }
    
    def test_stochastic(self) -> Dict[str, Any]:
        """Test Stochastic vectorisé"""
        print("📊 Test Stochastic vectorisé...")
        
        high = self.sample_data['high']
        low = self.sample_data['low']
        close = self.sample_data['close']
        
        # Test vectorisé
        start_time = time.time()
        k_percent, d_percent = vectorized_stochastic(high, low, close, 14, 3)
        vectorized_time = time.time() - start_time
        
        # Validation basique (Stochastic doit être entre 0 et 100)
        valid_k = k_percent[~np.isnan(k_percent)]
        valid_d = d_percent[~np.isnan(d_percent)]
        
        k_accuracy = np.all((valid_k >= 0) & (valid_k <= 100)) if len(valid_k) > 0 else False
        d_accuracy = np.all((valid_d >= 0) & (valid_d <= 100)) if len(valid_d) > 0 else False
        
        accuracy = k_accuracy and d_accuracy
        
        return {
            'name': 'Stochastic',
            'vectorized_time': vectorized_time,
            'pandas_time': 0,
            'speedup': 0,
            'accuracy': accuracy,
            'k_range': (np.min(valid_k), np.max(valid_k)) if len(valid_k) > 0 else (0, 0),
            'd_range': (np.min(valid_d), np.max(valid_d)) if len(valid_d) > 0 else (0, 0)
        }
    
    def test_portfolio_metrics(self) -> Dict[str, Any]:
        """Test métriques de portfolio vectorisées"""
        print("💼 Test Portfolio Metrics vectorisées...")
        
        returns = self.sample_data['returns']
        window = 252  # 1 an de données
        
        # Test vectorisé
        start_time = time.time()
        roll_ret, roll_vol, roll_sharpe = vectorized_portfolio_metrics(returns, window)
        vectorized_time = time.time() - start_time
        
        # Test pandas pour comparaison
        start_time = time.time()
        pandas_ret = pd.Series(returns).rolling(window).mean().values
        pandas_vol = pd.Series(returns).rolling(window).std().values
        pandas_sharpe = pandas_ret / pandas_vol
        pandas_time = time.time() - start_time
        
        # Validation
        valid_mask = ~np.isnan(pandas_ret)
        ret_accuracy = np.allclose(roll_ret[valid_mask], pandas_ret[valid_mask], rtol=1e-10)
        vol_accuracy = np.allclose(roll_vol[valid_mask], pandas_vol[valid_mask], rtol=1e-10)
        
        accuracy = ret_accuracy and vol_accuracy
        
        return {
            'name': 'Portfolio Metrics',
            'vectorized_time': vectorized_time,
            'pandas_time': pandas_time,
            'speedup': pandas_time / vectorized_time if vectorized_time > 0 else 0,
            'accuracy': accuracy,
            'components': {
                'returns': ret_accuracy,
                'volatility': vol_accuracy
            }
        }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests de validation"""
        print("🚀 Tests Complets des Indicateurs Vectorisés")
        print("=" * 60)
        
        test_functions = [
            self.test_sma,
            self.test_ema,
            self.test_rsi,
            self.test_macd,
            self.test_bollinger_bands,
            self.test_atr,
            self.test_stochastic,
            self.test_portfolio_metrics
        ]
        
        results = {}
        total_speedup = 0
        speedup_count = 0
        
        for test_func in test_functions:
            try:
                result = test_func()
                results[result['name']] = result
                
                # Affichage des résultats
                status = "✅" if result['accuracy'] else "❌"
                speedup_text = f"{result['speedup']:.1f}x" if result['speedup'] > 0 else "N/A"
                
                print(f"{status} {result['name']}: {speedup_text} speedup, "
                      f"{result['vectorized_time']:.4f}s")
                
                if result['speedup'] > 0:
                    total_speedup += result['speedup']
                    speedup_count += 1
                
            except Exception as e:
                print(f"❌ Erreur dans {test_func.__name__}: {e}")
                results[test_func.__name__] = {'error': str(e)}
        
        # Résumé
        avg_speedup = total_speedup / speedup_count if speedup_count > 0 else 0
        passed_tests = sum(1 for r in results.values() if r.get('accuracy', False))
        total_tests = len([r for r in results.values() if 'accuracy' in r])
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"  Tests réussis: {passed_tests}/{total_tests}")
        print(f"  Accélération moyenne: {avg_speedup:.1f}x")
        print(f"  Gain de temps moyen: {((avg_speedup - 1) / avg_speedup * 100):.1f}%")
        
        # Sauvegarde des résultats
        self._save_test_results(results, avg_speedup, passed_tests, total_tests)
        
        return results
    
    def _save_test_results(self, results: Dict[str, Any], avg_speedup: float, passed: int, total: int):
        """Sauvegarde les résultats des tests"""
        from datetime import datetime
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'timestamp': timestamp,
            'tests_passed': passed,
            'tests_total': total,
            'success_rate': passed / total if total > 0 else 0,
            'average_speedup': avg_speedup,
            'time_saved_percentage': ((avg_speedup - 1) / avg_speedup * 100) if avg_speedup > 0 else 0,
            'detailed_results': results
        }
        
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/vectorized_indicators_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📁 Résultats sauvegardés: {filename}")


def main():
    """Fonction principale"""
    tester = VectorizedIndicatorTester()
    results = tester.run_comprehensive_tests()
    
    return results


if __name__ == "__main__":
    main()