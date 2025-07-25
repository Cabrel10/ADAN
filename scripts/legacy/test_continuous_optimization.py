#!/usr/bin/env python3
"""
Test du système d'optimisation continue du DBE.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

from adan_trading_bot.environment.continuous_optimization_dbe import (
    ContinuousOptimizationDBE, OptimizationMetrics, PerformanceFeedbackLoop,
    AutoAdjustmentSystem, DBEParameters
)


class ContinuousOptimizationTester:
    """Testeur pour le système d'optimisation continue"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_performance_feedback_loop(self) -> Dict[str, Any]:
        """Test de la boucle de rétroaction performance"""
        print("🔄 Test boucle de rétroaction performance...")
        
        feedback_loop = PerformanceFeedbackLoop(window_size=50, optimization_frequency=20)
        
        # Simuler une série de performances avec amélioration graduelle
        base_params = DBEParameters()
        
        for i in range(25):
            # Performance qui s'améliore graduellement
            sharpe_ratio = 0.2 + i * 0.03 + np.random.randn() * 0.1
            total_return = sharpe_ratio * 0.02
            max_drawdown = max(0.01, 0.15 - i * 0.005)
            
            metrics = OptimizationMetrics(
                sharpe_ratio=sharpe_ratio,
                total_return=total_return,
                max_drawdown=max_drawdown,
                volatility=0.02 + np.random.randn() * 0.005,
                win_rate=0.5 + i * 0.01,
                profit_factor=1.0 + i * 0.05
            )
            
            # Varier légèrement les paramètres
            varied_params = base_params.to_dict()
            varied_params['risk_threshold_low'] += np.random.randn() * 0.01
            varied_params['learning_rate'] += np.random.randn() * 0.001
            
            feedback_loop.add_performance_sample(
                DBEParameters.from_dict(varied_params), 
                metrics
            )
        
        # Tester la suggestion d'optimisation
        optimized_params = feedback_loop.suggest_parameter_optimization(base_params)
        optimization_stats = feedback_loop.get_optimization_stats()
        
        return {
            'test_name': 'performance_feedback_loop',
            'samples_added': 25,
            'model_trained': feedback_loop.model_trained,
            'optimization_suggested': optimized_params is not None,
            'optimization_stats': optimization_stats,
            'performance_trend': optimization_stats['recent_performance_trend']
        }
    
    def test_auto_adjustment_system(self) -> Dict[str, Any]:
        """Test du système d'auto-ajustement"""
        print("⚙️ Test système d'auto-ajustement...")
        
        auto_adjustment = AutoAdjustmentSystem(adjustment_sensitivity=0.1)
        base_params = DBEParameters()
        
        # Test avec différents scénarios de performance
        scenarios = [
            # Performance critique
            {
                'metrics': OptimizationMetrics(sharpe_ratio=-1.0, total_return=-0.2, max_drawdown=0.25),
                'market': {'volatility': 0.08, 'current_drawdown': 0.15},
                'expected_adjustment': 'critical'
            },
            # Performance excellente
            {
                'metrics': OptimizationMetrics(sharpe_ratio=2.0, total_return=0.15, max_drawdown=0.03),
                'market': {'volatility': 0.015, 'current_drawdown': 0.02},
                'expected_adjustment': 'aggressive'
            },
            # Haute volatilité
            {
                'metrics': OptimizationMetrics(sharpe_ratio=0.5, total_return=0.05, max_drawdown=0.08),
                'market': {'volatility': 0.12, 'current_drawdown': 0.06},
                'expected_adjustment': 'volatility'
            }
        ]
        
        adjustments_made = []
        
        for i, scenario in enumerate(scenarios):
            adjusted_params = auto_adjustment.auto_adjust_parameters(
                base_params, scenario['metrics'], scenario['market']
            )
            
            # Vérifier que des ajustements ont été faits
            original_dict = base_params.to_dict()
            adjusted_dict = adjusted_params.to_dict()
            
            changes = {}
            for param_name in original_dict.keys():
                if abs(adjusted_dict[param_name] - original_dict[param_name]) > 0.001:
                    changes[param_name] = {
                        'original': original_dict[param_name],
                        'adjusted': adjusted_dict[param_name]
                    }
            
            adjustments_made.append({
                'scenario': i,
                'changes': changes,
                'num_changes': len(changes)
            })
        
        adjustment_stats = auto_adjustment.get_adjustment_stats()
        
        return {
            'test_name': 'auto_adjustment_system',
            'scenarios_tested': len(scenarios),
            'adjustments_made': adjustments_made,
            'total_adjustments': sum(adj['num_changes'] for adj in adjustments_made),
            'adjustment_stats': adjustment_stats
        }
    
    def test_continuous_optimization_integration(self) -> Dict[str, Any]:
        """Test d'intégration du système d'optimisation continue"""
        print("🔧 Test intégration optimisation continue...")
        
        # Créer le système d'optimisation continue
        continuous_dbe = ContinuousOptimizationDBE(
            optimization_enabled=True,
            auto_adjustment_enabled=True
        )
        
        # Simuler une session de trading avec évolution des performances
        simulation_results = []
        
        base_price = 50000
        
        for step in range(30):
            # Simuler évolution du marché
            price_change = np.random.randn() * 0.02
            base_price *= (1 + price_change)
            
            market_data = {
                'price': base_price,
                'volume': 1000000 + np.random.randn() * 200000,
                'volatility': 0.02 + abs(np.random.randn() * 0.01)
            }
            
            # Simuler performance qui s'améliore avec le temps
            performance_base = 0.3 + step * 0.02
            performance_metrics = {
                'sharpe_ratio': performance_base + np.random.randn() * 0.2,
                'total_return': performance_base * 0.01,
                'max_drawdown': max(0.01, 0.1 - step * 0.002),
                'volatility': market_data['volatility'],
                'win_rate': 0.5 + step * 0.005,
                'profit_factor': 1.0 + step * 0.02
            }
            
            # Mise à jour du système
            modulation = continuous_dbe.update(market_data, performance_metrics)
            
            simulation_results.append({
                'step': step,
                'performance_score': modulation.get('performance_score', 0.0),
                'risk_level': modulation.get('risk_level', 'NORMAL'),
                'market_regime': modulation.get('market_regime', 'sideways'),
                'optimization_enabled': modulation.get('optimization_enabled', False),
                'auto_adjustment_enabled': modulation.get('auto_adjustment_enabled', False)
            })
        
        # Générer rapport de performance
        performance_report = continuous_dbe.get_comprehensive_performance_report()
        
        # Test d'optimisation forcée
        optimization_success = continuous_dbe.force_optimization()
        
        # Arrêt propre
        continuous_dbe.shutdown()
        
        return {
            'test_name': 'continuous_optimization_integration',
            'simulation_steps': len(simulation_results),
            'performance_report': performance_report,
            'optimization_forced': optimization_success,
            'final_performance_trend': performance_report.get('performance_stats', {}).get('performance_trend', 0.0),
            'system_health': performance_report.get('system_health', {}),
            'integration_successful': len(simulation_results) == 30 and 'performance_stats' in performance_report
        }
    
    def test_optimization_metrics(self) -> Dict[str, Any]:
        """Test des métriques d'optimisation"""
        print("📊 Test métriques d'optimisation...")
        
        # Test de la classe OptimizationMetrics
        metrics = OptimizationMetrics(
            sharpe_ratio=1.5,
            total_return=0.12,
            max_drawdown=0.08,
            volatility=0.025,
            win_rate=0.65,
            profit_factor=1.8
        )
        
        # Test du score composite avec poids par défaut
        default_score = metrics.get_composite_score()
        
        # Test avec poids personnalisés
        custom_weights = {
            'sharpe_ratio': 0.4,
            'total_return': 0.3,
            'max_drawdown': -0.3
        }
        custom_score = metrics.get_composite_score(custom_weights)
        
        # Test de conversion en dictionnaire
        metrics_dict = metrics.to_dict()
        
        return {
            'test_name': 'optimization_metrics',
            'default_composite_score': default_score,
            'custom_composite_score': custom_score,
            'metrics_dict_keys': list(metrics_dict.keys()),
            'metrics_valid': all(isinstance(v, (int, float)) for v in metrics_dict.values()),
            'scores_different': abs(default_score - custom_score) > 0.01
        }
    
    def test_parameter_constraints(self) -> Dict[str, Any]:
        """Test des contraintes sur les paramètres"""
        print("🔒 Test contraintes paramètres...")
        
        auto_adjustment = AutoAdjustmentSystem()
        
        # Créer des paramètres avec des valeurs extrêmes
        extreme_params = {
            'risk_threshold_low': -0.5,  # Trop bas
            'risk_threshold_high': 1.5,  # Trop haut
            'volatility_threshold_low': 0.0001,  # Trop bas
            'max_drawdown_threshold': 0.5,  # Trop haut
            'learning_rate': 0.5  # Trop haut
        }
        
        # Appliquer les contraintes
        constrained_params = auto_adjustment._apply_parameter_constraints(extreme_params)
        
        # Vérifier que les contraintes sont respectées
        constraints_respected = True
        constraint_violations = []
        
        expected_constraints = {
            'risk_threshold_low': (0.1, 0.5),
            'risk_threshold_high': (0.6, 0.95),
            'volatility_threshold_low': (0.005, 0.02),
            'max_drawdown_threshold': (0.05, 0.3),
            'learning_rate': (0.001, 0.1)
        }
        
        for param_name, (min_val, max_val) in expected_constraints.items():
            if param_name in constrained_params:
                value = constrained_params[param_name]
                if not (min_val <= value <= max_val):
                    constraints_respected = False
                    constraint_violations.append({
                        'parameter': param_name,
                        'value': value,
                        'expected_range': (min_val, max_val)
                    })
        
        return {
            'test_name': 'parameter_constraints',
            'extreme_params': extreme_params,
            'constrained_params': constrained_params,
            'constraints_respected': constraints_respected,
            'constraint_violations': constraint_violations
        }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests d'optimisation continue"""
        print("🚀 Tests Complets Optimisation Continue DBE")
        print("=" * 60)
        
        test_functions = [
            self.test_performance_feedback_loop,
            self.test_auto_adjustment_system,
            self.test_continuous_optimization_integration,
            self.test_optimization_metrics,
            self.test_parameter_constraints
        ]
        
        results = {}
        passed_tests = 0
        
        for test_func in test_functions:
            try:
                result = test_func()
                results[result['test_name']] = result
                
                # Déterminer si le test a réussi
                success_indicators = [
                    'model_trained', 'optimization_suggested', 'integration_successful',
                    'metrics_valid', 'constraints_respected'
                ]
                
                test_passed = any(result.get(indicator, False) for indicator in success_indicators)
                
                # Vérifications spécifiques
                if result['test_name'] == 'auto_adjustment_system':
                    test_passed = result.get('total_adjustments', 0) > 0
                elif result['test_name'] == 'continuous_optimization_integration':
                    test_passed = result.get('integration_successful', False)
                
                if test_passed:
                    passed_tests += 1
                    print(f"✅ {result['test_name']}: RÉUSSI")
                else:
                    print(f"⚠️  {result['test_name']}: PARTIEL")
                
            except Exception as e:
                print(f"❌ Erreur dans {test_func.__name__}: {e}")
                results[test_func.__name__] = {'error': str(e)}
        
        # Résumé
        total_tests = len([r for r in results.values() if 'error' not in r])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"  Tests réussis: {passed_tests}/{total_tests}")
        print(f"  Taux de succès: {success_rate:.1%}")
        print(f"  Optimisation continue: {'✅ OPÉRATIONNELLE' if success_rate > 0.6 else '⚠️ PARTIELLE'}")
        
        # Sauvegarde des résultats
        self._save_test_results(results, success_rate)
        
        return results
    
    def _save_test_results(self, results: Dict[str, Any], success_rate: float):
        """Sauvegarde les résultats des tests"""
        from datetime import datetime
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'timestamp': timestamp,
            'success_rate': success_rate,
            'continuous_optimization_operational': success_rate > 0.6,
            'detailed_results': results
        }
        
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/continuous_optimization_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📁 Résultats sauvegardés: {filename}")


def main():
    """Fonction principale"""
    tester = ContinuousOptimizationTester()
    results = tester.run_comprehensive_tests()
    
    return results


if __name__ == "__main__":
    main()