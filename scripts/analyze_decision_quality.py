# analysis/decision_quality_analyzer.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass

@dataclass
class Trade:
    entry_time: str
    exit_time: str  
    pnl: float
    fees: float
    asset: str
    position_size: float
    signal_strength: float
    timeframe: str

class DecisionQualityAnalyzer:
    def __init__(self, trades_data: pd.DataFrame, equity_curve: pd.DataFrame = None):
        self.trades = trades_data
        self.equity = equity_curve
        
    def comprehensive_analysis(self) -> Dict:
        """Analyse complète qualité décisions"""
        return {
            "decision_quality": self._analyze_decision_quality(),
            "pattern_effectiveness": self._analyze_pattern_effectiveness(), 
            "risk_intelligence": self._analyze_risk_intelligence(),
            "statistical_significance": self._run_statistical_tests(),
            "robustness_metrics": self._assess_robustness()
        }
    
    def _analyze_decision_quality(self) -> Dict:
        """Analyse qualité décisions trade par trade"""
        wins = self.trades[self.trades['pnl'] > 0]
        losses = self.trades[self.trades['pnl'] <= 0]
        
        win_rate = len(wins) / len(self.trades) if len(self.trades) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # Consistance des décisions
        signal_quality = self._analyze_signal_quality()
        action_consistency = self._calculate_action_consistency()
        
        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss, 
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "signal_quality_score": signal_quality,
            "action_consistency": action_consistency,
            "gambling_indicators": self._detect_gambling_behavior()
        }
    
    def _analyze_pattern_effectiveness(self) -> Dict:
        """Analyse efficacité des patterns identifiés"""
        # Regroupement par conditions similaires (simplifié)
        pattern_metrics = {}
        
        # Analyse par timeframe
        for tf in self.trades['timeframe'].unique():
            tf_trades = self.trades[self.trades['timeframe'] == tf]
            if len(tf_trades) > 0:
                win_rate = len(tf_trades[tf_trades['pnl'] > 0]) / len(tf_trades)
                pattern_metrics[f'timeframe_{tf}'] = {
                    'win_rate': win_rate,
                    'avg_pnl': tf_trades['pnl'].mean(),
                    'trade_count': len(tf_trades)
                }
        
        # Analyse par force de signal
        self.trades['signal_strength_bin'] = pd.cut(self.trades['signal_strength'], bins=5)
        signal_analysis = self.trades.groupby('signal_strength_bin').agg({
            'pnl': ['mean', 'count'],
            'signal_strength': 'mean'
        }).round(4)
        
        return {
            "timeframe_performance": pattern_metrics,
            "signal_strength_analysis": signal_analysis.to_dict(),
            "pattern_repetition": self._calculate_pattern_repetition(),
            "edge_consistency": self._calculate_edge_consistency()
        }
    
    def _analyze_risk_intelligence(self) -> Dict:
        """Analyse intelligence risque"""
        if self.equity is not None:
            returns = self.equity['equity'].pct_change().dropna()
            sharpe = self._annualized_sharpe(returns)
            sortino = self._sortino_ratio(returns)
            max_dd = self._max_drawdown(self.equity['equity'].values)
        else:
            sharpe = sortino = max_dd = np.nan
            
        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino, 
            "max_drawdown": max_dd,
            "risk_adjustment_quality": self._assess_risk_adjustment(),
            "position_sizing_efficiency": self._assess_position_sizing(),
            "drawdown_recovery": self._analyze_drawdown_recovery()
        }
    
    def _run_statistical_tests(self) -> Dict:
        """Tests statistiques de significativité"""
        if len(self.trades) == 0:
            return {"error": "No trades available for statistical tests"}
            
        # Bootstrap test
        obs_mean, boot_means, pval_bootstrap = self._bootstrap_test(self.trades['pnl'].values)
        
        # Permutation test
        if 'signal_strength' in self.trades.columns and 'future_return' in self.trades.columns:
            ic, pval_perm, _ = self._permutation_test(
                self.trades['signal_strength'].values, 
                self.trades['future_return'].values
            )
        else:
            ic = pval_perm = np.nan
            
        # Runs test for randomness
        runs_test = self._runs_test(self.trades['pnl'] > 0)
        
        return {
            "bootstrap_p_value": pval_bootstrap,
            "observed_mean_return": obs_mean,
            "information_coefficient": ic,
            "permutation_p_value": pval_perm,
            "runs_test_p_value": runs_test,
            "significance_summary": self._interpret_significance(pval_bootstrap, pval_perm)
        }
    
    def _assess_robustness(self) -> Dict:
        """Évaluation robustesse"""
        return {
            "out_of_sample_stability": self._assess_oos_stability(),
            "parameter_sensitivity": self._assess_parameter_sensitivity(),
            "regime_performance": self._analyze_regime_performance(),
            "temporal_consistency": self._analyze_temporal_consistency()
        }
    
    # === MÉTHODES D'ANALYSE SPÉCIFIQUES ===
    
    def _analyze_signal_quality(self) -> float:
        """Qualité des signaux (corrélation signal→performance)"""
        if 'signal_strength' not in self.trades.columns or len(self.trades) < 5:
            return 0.0
            
        try:
            correlation = np.corrcoef(self.trades['signal_strength'], self.trades['pnl'])[0,1]
            return abs(correlation)  # Valeur absolue car direction moins importante que relation
        except:
            return 0.0
    
    def _calculate_action_consistency(self) -> Dict:
        """Consistance des actions"""
        if 'signal_strength' not in self.trades.columns:
            return {"score": 0, "volatility": 1}
            
        signal_volatility = self.trades['signal_strength'].std()
        action_stability = 1 / (1 + signal_volatility)  # Plus stable = meilleur
        
        return {
            "score": min(1.0, action_stability),
            "volatility": signal_volatility,
            "consistency_rating": "HIGH" if action_stability > 0.8 else "MEDIUM" if action_stability > 0.5 else "LOW"
        }
    
    def _detect_gambling_behavior(self) -> Dict:
        """Détection comportement hasardeux"""
        if len(self.trades) < 10:
            return {"score": 0, "indicators": []}
            
        indicators = []
        
        # 1. Overtrading
        avg_trades_per_day = len(self.trades) / (self.trades['entry_time'].nunique() if self.trades['entry_time'].nunique() > 0 else 1)
        if avg_trades_per_day > 10:
            indicators.append("OVERTRADING")
            
        # 2. Position sizing erratic
        position_volatility = self.trades['position_size'].std() / self.trades['position_size'].mean()
        if position_volatility > 2.0:
            indicators.append("ERRATIC_POSITION_SIZING")
            
        # 3. Win rate très bas avec forte variance
        if len(self.trades) > 20:
            win_rates = [self.trades.iloc[i:i+10]['pnl'].gt(0).mean() for i in range(0, len(self.trades)-9, 5)]
            if np.std(win_rates) > 0.3:
                indicators.append("INCONSISTENT_PERFORMANCE")
                
        # 4. Recovery trading (trade après grosse perte)
        big_losses = self.trades[self.trades['pnl'] < self.trades['pnl'].quantile(0.1)]
        if len(big_losses) > 0:
            recovery_pattern = self._detect_recovery_trading(big_losses)
            if recovery_pattern:
                indicators.append("REVENGE_TRADING")
        
        gambling_score = len(indicators) / 4  # Normalisé 0-1
        
        return {
            "score": gambling_score,
            "indicators": indicators,
            "risk_level": "HIGH" if gambling_score > 0.6 else "MEDIUM" if gambling_score > 0.3 else "LOW"
        }
    
    # === MÉTHODES STATISTIQUES ===
    
    def _bootstrap_test(self, pnls, n_boot=1000):
        """Test bootstrap pour significativité des returns"""
        obs_mean = np.mean(pnls)
        boot_means = []
        
        for _ in range(n_boot):
            sample = resample(pnls, replace=True, n_samples=len(pnls))
            boot_means.append(np.mean(sample))
            
        boot_means = np.array(boot_means)
        
        # p-value: probabilité que le vrai mean soit <= 0
        if obs_mean > 0:
            pval = np.mean(boot_means <= 0)
        else:
            pval = np.mean(boot_means >= 0)
            
        return obs_mean, boot_means, pval
    
    def _permutation_test(self, signals, returns, n_perm=1000):
        """Test de permutation pour corrélation signal→return"""
        if len(signals) != len(returns) or len(signals) < 10:
            return 0, 1, []
            
        obs_ic = np.corrcoef(signals, returns)[0,1]
        perm_ics = []
        
        for _ in range(n_perm):
            perm_returns = np.random.permutation(returns)
            ic = np.corrcoef(signals, perm_returns)[0,1]
            perm_ics.append(ic)
            
        perm_ics = np.array(perm_ics)
        pval = np.mean(np.abs(perm_ics) >= abs(obs_ic))
        
        return obs_ic, pval, perm_ics
    
    def _runs_test(self, binary_series):
        """Test de runs pour randomité"""
        if len(binary_series) < 2:
            return 1.0
            
        # Compter les runs (séquences de valeurs identiques)
        runs = 0
        prev = None
        for val in binary_series:
            if val != prev:
                runs += 1
            prev = val
            
        n1 = binary_series.sum()  # Nombre de succès
        n2 = len(binary_series) - n1  # Nombre d'échecs
        
        # Statistique attendue sous hypothèse nulle (random)
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        std_runs = ((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                   ((n1 + n2)**2 * (n1 + n2 - 1)))**0.5
        
        if std_runs == 0:
            return 1.0
            
        z = (runs - expected_runs) / std_runs
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return pval
    
    # === MÉTHODES DE CALCUL PERFORMANCE ===
    
    def _annualized_sharpe(self, returns, periods=252):
        """Ratio de Sharpe annualisé"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(periods)
    
    def _sortino_ratio(self, returns, periods=252, mar=0.0):
        """Ratio de Sortino (downside risk only)"""
        if len(returns) < 2:
            return 0.0
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return (returns.mean() - mar) / downside_returns.std() * np.sqrt(periods)
    
    def _max_drawdown(self, equity_curve):
        """Calcul du drawdown maximum"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    # === MÉTHODES SPÉCIFIQUES ADAN ===
    
    def _detect_recovery_trading(self, big_losses):
        """Détection revenge trading après grosses pertes"""
        if len(big_losses) < 2:
            return False
            
        # Vérifie si trades rapprochés après grosses pertes
        big_losses = big_losses.sort_values('exit_time')
        time_diffs = big_losses['exit_time'].diff().dt.total_seconds().dropna()
        
        # Si trades dans l'heure après grosse perte
        rapid_recovery = (time_diffs < 3600).any()
        return rapid_recovery
    
    def _calculate_pattern_repetition(self):
        """Calcul répétition patterns profitables"""
        # Simplifié: regarde la variance des performances sur fenêtres glissantes
        if len(self.trades) < 20:
            return 0.0
            
        window_performance = []
        window_size = min(10, len(self.trades) // 2)
        
        for i in range(0, len(self.trades) - window_size + 1, window_size):
            window = self.trades.iloc[i:i+window_size]
            win_rate = (window['pnl'] > 0).mean()
            window_performance.append(win_rate)
            
        consistency = 1 - np.std(window_performance)  # Plus proche de 1 = plus consistant
        return max(0, consistency)
    
    def _calculate_edge_consistency(self):
        """Consistance de l'edge sur différentes conditions"""
        if len(self.trades) < 10:
            return 0.0
            
        # Analyse performance par taille de position
        if 'position_size' in self.trades.columns:
            size_bins = pd.cut(self.trades['position_size'], bins=3)
            size_performance = self.trades.groupby(size_bins)['pnl'].mean()
            size_consistency = 1 - (size_performance.std() / abs(size_performance.mean())) if size_performance.mean() != 0 else 0
        else:
            size_consistency = 0
            
        return size_consistency
    
    def _interpret_significance(self, pval_bootstrap, pval_perm):
        """Interprétation résultats tests statistiques"""
        interpretations = []
        
        if pval_bootstrap < 0.05:
            interpretations.append("RETURNS_STATISTICALLY_SIGNIFICANT")
        else:
            interpretations.append("RETURNS_MAY_BE_RANDOM")
            
        if pval_perm < 0.05:
            interpretations.append("SIGNALS_PREDICTIVE") 
        elif not np.isnan(pval_perm):
            interpretations.append("SIGNALS_NOT_PREDICTIVE")
            
        return interpretations
    
    # === MÉTHODES ROBUSTESSE (SIMPLIFIÉES) ===
    
    def _assess_oos_stability(self):
        """Stabilité out-of-sample (simplifié)"""
        if len(self.trades) < 20:
            return "INSUFFICIENT_DATA"
            
        # Split temporel
        split_idx = len(self.trades) // 2
        first_half = self.trades.iloc[:split_idx]
        second_half = self.trades.iloc[split_idx:]
        
        perf_first = (first_half['pnl'] > 0).mean() if len(first_half) > 0 else 0
        perf_second = (second_half['pnl'] > 0).mean() if len(second_half) > 0 else 0
        
        performance_drop = abs(perf_first - perf_second)
        
        if performance_drop < 0.1:
            return "STABLE"
        elif performance_drop < 0.2:
            return "MODERATELY_STABLE"
        else:
            return "UNSTABLE"
    
    def _assess_risk_adjustment(self):
        """Qualité ajustement risque"""
        # Vérifie si le modèle réduit position sizing en haute volatilité
        # Simplifié pour l'exemple
        return "NEEDS_DEEPER_ANALYSIS"
    
    def _assess_position_sizing(self):
        """Efficacité position sizing"""
        if 'position_size' not in self.trades.columns:
            return "NO_DATA"
            
        sizing_efficiency = self.trades['position_size'].mean() / self.trades['position_size'].max()
        return f"EFFICIENCY_{sizing_efficiency:.2f}"
    
    def generate_report(self) -> str:
        """Génère un rapport complet lisible"""
        analysis = self.comprehensive_analysis()
        
        report = []
        report.append("=" * 60)
        report.append("📊 RAPPORT QUALITÉ DÉCISIONS ADAN")
        report.append("=" * 60)
        
        # Résumé exécutif
        dq = analysis['decision_quality']
        report.append(f"\n🎯 QUALITÉ DÉCISIONS:")
        report.append(f"   Win Rate: {dq['win_rate']:.1%}")
        report.append(f"   Profit Factor: {dq['profit_factor']:.2f}")
        report.append(f"   Expectancy: ${dq['expectancy']:.4f}")
        report.append(f"   Score Gambling: {dq['gambling_indicators']['score']:.1%} ({dq['gambling_indicators']['risk_level']})")
        
        # Significativité statistique
        stats = analysis['statistical_significance']
        report.append(f"\n📈 SIGNIFICATIVITÉ STATISTIQUE:")
        report.append(f"   Bootstrap p-value: {stats['bootstrap_p_value']:.4f}")
        if not np.isnan(stats['information_coefficient']):
            report.append(f"   Information Coefficient: {stats['information_coefficient']:.4f}")
        report.append(f"   Interprétation: {', '.join(stats['significance_summary'])}")
        
        # Patterns
        patterns = analysis['pattern_effectiveness']
        report.append(f"\n🔍 EFFICACITÉ PATTERNS:")
        report.append(f"   Consistance Patterns: {patterns['pattern_repetition']:.1%}")
        report.append(f"   Consistance Edge: {patterns['edge_consistency']:.1%}")
        
        # Risque
        risk = analysis['risk_intelligence']
        report.append(f"\n🛡️ INTELLIGENCE RISQUE:")
        report.append(f"   Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        report.append(f"   Max Drawdown: {risk['max_drawdown']:.1%}")
        
        # Robustesse
        robust = analysis['robustness_metrics']
        report.append(f"\n🏗️ ROBUSTESSE:")
        report.append(f"   Stabilité OOS: {robust['out_of_sample_stability']}")
        
        # Recommandations
        report.append(f"\n💡 RECOMMANDATIONS:")
        recommendations = self._generate_recommendations(analysis)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"   {i}. {rec}")
            
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recs = []
        dq = analysis['decision_quality']
        stats = analysis['statistical_significance']
        risk = analysis['risk_intelligence']
        
        # Décisions
        if dq['gambling_indicators']['score'] > 0.5:
            recs.append("Réduire l'agressivité du trading - comportement hasardeux détecté")
            
        if dq['profit_factor'] < 1.2:
            recs.append("Améliorer la sélectivité des trades - profit factor trop bas")
            
        if dq['expectancy'] < 0:
            recs.append("Revoir la stratégie - espérance négative")
            
        # Statistiques
        if stats['bootstrap_p_value'] > 0.05:
            recs.append("Performance peut être due au hasard - besoin de plus de données")
            
        # Risque
        if risk['max_drawdown'] < -0.2:
            recs.append("Drawdown trop élevé - renforcer la gestion risque")
            
        if len(recs) == 0:
            recs.append("Continuer monitoring - performances dans les normes")
            
        return recs

# === UTILITAIRE POUR CHARGER LES DONNÉES ADAN ===
def load_adan_data_from_logs(log_file_path: str) -> pd.DataFrame:
    """
    Charge les données de trades depuis les logs ADAN
    À adapter selon le format de vos logs
    """
    # Cette fonction doit être adaptée à votre format de logs
    trades = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                if '[POSITION OUVERTE]' in line:
                    # Exemple de parsing - à adapter
                    parts = line.split('|')
                    if len(parts) >= 4:
                        asset = parts[0].split()[-1]
                        size = float(parts[1].split(' @')[1].strip())
                        fees = float(parts[3].split('$')[1].strip())
                        
                        # Estimation PnL (à remplacer par données réelles)
                        pnl = np.random.normal(0, 0.1)  # Remplacer
                        
                        trades.append({
                            'asset': asset,
                            'position_size': size,
                            'fees': fees,
                            'pnl': pnl,
                            'signal_strength': np.random.random(),  # Remplacer
                            'timeframe': '5m'  # Remplacer
                        })
    except Exception as e:
        print(f"Erreur chargement logs: {e}")
        
    return pd.DataFrame(trades)

# === EXEMPLE D'UTILISATION ===
if __name__ == "__main__":
    # Exemple avec données simulées
    np.random.seed(42)
    n_trades = 100
    
    sample_trades = pd.DataFrame({
        'entry_time': pd.date_range('2024-01-01', periods=n_trades, freq='H'),
        'exit_time': pd.date_range('2024-01-01 01:00:00', periods=n_trades, freq='H'),
        'pnl': np.random.normal(0.001, 0.02, n_trades),  # Légère edge positive
        'fees': np.full(n_trades, 0.001),
        'asset': ['BTCUSDT'] * n_trades,
        'position_size': np.random.uniform(10, 100, n_trades),
        'signal_strength': np.random.uniform(0, 1, n_trades),
        'timeframe': np.random.choice(['5m', '1h', '4h'], n_trades),
        'future_return': np.random.normal(0.0005, 0.01, n_trades)
    })
    
    # Courbe equity simulée
    equity_curve = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'equity': 1000 * (1 + np.random.normal(0.0001, 0.01, 1000)).cumprod()
    })
    
    # Analyse
    analyzer = DecisionQualityAnalyzer(sample_trades, equity_curve)
    report = analyzer.generate_report()
    print(report)
    
    # Sauvegarde analyse détaillée
    analysis = analyzer.comprehensive_analysis()
    with open('decision_quality_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
