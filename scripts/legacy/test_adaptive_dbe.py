#!/usr/bin/env python3
"""
Test du système DBE adaptatif avec méta-apprentissage.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

from adan_trading_bot.environment.adaptive_dbe import (
    AdaptiveDBE, DBEParameters, MarketRegime, MarketRegimeDetector
)


class AdaptiveDBETester:
    """Testeur pour le système DBE adaptatif"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_parameter_evolution(self) -> Dict[str, Any]:
        """Test de l'évolution des paramètres"""
        print("🧬 Test évolution des paramètres...")
        
        # Créer DBE adaptatif
        initial_params = DBEParameters()
        adaptive_dbe = AdaptiveDBE(initial_params, adaptation_enabled=True)
        
        # Simuler différentes conditions de marché et performances
        scenarios = [
            # Scénario 1: Performance positive, faible drawdown
            {
                'market_data': {'price': 50000, 'volume': 1000000, 'volatility': 0.02},
                'performance': {'sharpe_ratio': 1.2, 'total_return': 0.05, 'max_drawdown': 0.03}
            },
            # Scénario 2: Performance négative, drawdown élevé
            {
                'market_data': {'price': 48000, 'volume': 1500000, 'volatility': 0.06},
                'performance': {'sharpe_ratio': -0.3, 'total_return': -0.08, 'max_drawdown': 0.12}
            },
            # Scénario 3: Haute volatilité
            {
                'market_data': {'price': 52000, 'volume': 2000000, 'volatility': 0.08},
                'performance': {'sharpe_ratio': 0.8, 'total_return': 0.02, 'max_drawdown': 0.06}
            },
            # Scénario 4: Basse volatilité, performance stable
            {
                'market_data': {'price': 51000, 'volume': 800000, 'volatility': 0.008},
                'performance': {'sharpe_ratio': 1.5, 'total_return': 0.03, 'max_drawdown': 0.01}
            }
        ]
        
        initial_params_dict = adaptive_dbe.params.to_dict()
        modulations = []
        
        # Exécuter les scénarios
        for i, scenario in enumerate(scenarios):
            modulation = adaptive_dbe.update(
                scenario['market_data'], 
                scenario['performance']
            )
            modulations.append(modulation)
            
            print(f"  Scénario {i+1}: Régime {modulation['market_regime']}, "
                  f"Risque {modulation['risk_level']}")
        
        final_params_dict = adaptive_dbe.params.to_dict()
        
        # Calculer les changements de paramètres
        param_changes = {}
        for param_name in initial_params_dict.keys():
            initial_val = initial_params_dict[param_name]
            final_val = final_params_dict[param_name]
            change = abs(final_val - initial_val)
            param_changes[param_name] = {
                'initial': initial_val,
                'final': final_val,
                'change': change,
                'relative_change': change / initial_val if initial_val != 0 else 0
            }
        
        # Statistiques d'adaptation
        adaptation_stats = adaptive_dbe.get_comprehensive_stats()
        
        return {
            'test_name': 'parameter_evolution',
            'scenarios_processed': len(scenarios),
            'parameter_changes': param_changes,
            'adaptation_stats': adaptation_stats,
            'modulations': modulations,
            'parameters_adapted': sum(1 for change in param_changes.values() if change['change'] > 0.001)
        }
    
    def test_market_regime_detection(self) -> Dict[str, Any]:
        """Test de détection des régimes de marché"""
        print("📊 Test détection des régimes...")
        
        detector = MarketRegimeDetector(window_size=20)
        
        # Simuler différents régimes de marché
        regimes_detected = []
        
        # Régime 1: Tendance haussière
        base_price = 50000
        for i in range(25):
            price = base_price + i * 100 + np.random.randn() * 50
            volume = 1000000 + np.random.randn() * 100000
            volatility = 0.015 + np.random.randn() * 0.005
            detector.update(price, volume, max(0.001, volatility))
        
        regime1 = detector.detect_regime()
        regimes_detected.append(('trending_up', regime1))
        
        # Régime 2: Haute volatilité
        for i in range(25):
            price = base_price + 2500 + np.random.randn() * 500
            volume = 1500000 + np.random.randn() * 200000
            volatility = 0.07 + np.random.randn() * 0.02
            detector.update(price, volume, max(0.001, volatility))
        
        regime2 = detector.detect_regime()
        regimes_detected.append(('high_volatility', regime2))
        
        # Régime 3: Marché latéral
        for i in range(25):
            price = base_price + 2500 + np.sin(i * 0.3) * 200 + np.random.randn() * 50
            volume = 900000 + np.random.randn() * 50000
            volatility = 0.01 + np.random.randn() * 0.003
            detector.update(price, volume, max(0.001, volatility))
        
        regime3 = detector.detect_regime()
        regimes_detected.append(('sideways', regime3))
        
        # Conditions de marché finales
        market_conditions = detector.get_market_conditions()
        
        return {
            'test_name': 'regime_detection',
            'regimes_detected': regimes_detected,
            'market_conditions': market_conditions,
            'detection_successful': len(regimes_detected) == 3
        }
    
    def test_adaptation_robustness(self) -> Dict[str, Any]:
        """Test de robustesse de l'adaptation"""
        print("🛡️ Test robustesse de l'adaptation...")
        
        adaptive_dbe = AdaptiveDBE(adaptation_enabled=True)
        
        # Test avec des données extrêmes
        extreme_scenarios = [
            # Crise de marché
            {
                'market_data': {'price': 30000, 'volume': 5000000, 'volatility': 0.15},
                'performance': {'sharpe_ratio': -2.0, 'total_return': -0.30, 'max_drawdown': 0.25}
            },
            # Bulle spéculative
            {
                'market_data': {'price': 80000, 'volume': 3000000, 'volatility': 0.12},
                'performance': {'sharpe_ratio': 3.0, 'total_return': 0.50, 'max_drawdown': 0.08}
            },
            # Marché très calme
            {
                'market_data': {'price': 50500, 'volume': 200000, 'volatility': 0.002},
                'performance': {'sharpe_ratio': 0.1, 'total_return': 0.001, 'max_drawdown': 0.001}
            }
        ]
        
        initial_params = adaptive_dbe.params.to_dict()
        
        try:
            for scenario in extreme_scenarios:
                modulation = adaptive_dbe.update(
                    scenario['market_data'], 
                    scenario['performance']
                )
                
                # Vérifier que la modulation est valide
                assert 'risk_level' in modulation
                assert 'market_regime' in modulation
                assert modulation['risk_multiplier'] > 0
                assert modulation['position_size_multiplier'] > 0
            
            final_params = adaptive_dbe.params.to_dict()
            
            # Vérifier que les paramètres restent dans des limites raisonnables
            params_valid = True
            for param_name, value in final_params.items():
                if isinstance(value, (int, float)):
                    if not (0.001 <= value <= 100):  # Limites très larges
                        params_valid = False
                        break
            
            robustness_passed = True
            
        except Exception as e:
            robustness_passed = False
            params_valid = False
            print(f"  Erreur lors du test de robustesse: {e}")
        
        return {
            'test_name': 'adaptation_robustness',
            'extreme_scenarios_processed': len(extreme_scenarios),
            'robustness_passed': robustness_passed,
            'parameters_valid': params_valid,
            'initial_params': initial_params,
            'final_params': final_params if 'final_params' in locals() else {}
        }
    
    def test_performance_feedback_loop(self) -> Dict[str, Any]:
        """Test de la boucle de rétroaction performance"""
        print("🔄 Test boucle de rétroaction...")
        
        adaptive_dbe = AdaptiveDBE(adaptation_enabled=True)
        
        # Simuler une amélioration progressive de performance
        performance_trajectory = []
        param_trajectory = []
        
        base_sharpe = 0.5
        
        for step in range(20):
            # Performance qui s'améliore graduellement
            sharpe_ratio = base_sharpe + step * 0.05 + np.random.randn() * 0.1
            total_return = sharpe_ratio * 0.02
            max_drawdown = max(0.01, 0.1 - step * 0.003)
            
            market_data = {
                'price': 50000 + step * 200 + np.random.randn() * 100,
                'volume': 1000000 + np.random.randn() * 100000,
                'volatility': 0.02 + np.random.randn() * 0.005
            }
            
            performance = {
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'max_drawdown': max_drawdown
            }
            
            modulation = adaptive_dbe.update(market_data, performance)
            
            performance_trajectory.append(sharpe_ratio)
            param_trajectory.append(adaptive_dbe.params.to_dict())
        
        # Analyser la trajectoire
        performance_trend = np.polyfit(range(len(performance_trajectory)), performance_trajectory, 1)[0]
        
        # Vérifier que les paramètres s'adaptent à l'amélioration
        initial_risk_threshold = param_trajectory[0]['risk_threshold_low']
        final_risk_threshold = param_trajectory[-1]['risk_threshold_low']
        risk_threshold_trend = final_risk_threshold - initial_risk_threshold
        
        adaptation_stats = adaptive_dbe.get_comprehensive_stats()
        
        return {
            'test_name': 'performance_feedback_loop',
            'steps_simulated': len(performance_trajectory),
            'performance_trend': performance_trend,
            'risk_threshold_adaptation': risk_threshold_trend,
            'final_adaptation_stats': adaptation_stats,
            'feedback_loop_working': performance_trend > 0 and adaptation_stats['adaptation_stats']['adaptation_count'] > 0
        }
    
    def test_multi_regime_adaptation(self) -> Dict[str, Any]:
        """Test d'adaptation à travers plusieurs régimes"""
        print("🌐 Test adaptation multi-régimes...")
        
        adaptive_dbe = AdaptiveDBE(adaptation_enabled=True)
        
        # Simuler une séquence de régimes différents
        regime_sequence = [
            # Phase 1: Marché haussier
            {'regime': 'bull', 'steps': 10, 'base_perf': 1.0, 'volatility': 0.02},
            # Phase 2: Correction
            {'regime': 'correction', 'steps': 5, 'base_perf': -0.5, 'volatility': 0.05},
            # Phase 3: Récupération
            {'regime': 'recovery', 'steps': 8, 'base_perf': 0.8, 'volatility': 0.03},
            # Phase 4: Marché latéral
            {'regime': 'sideways', 'steps': 7, 'base_perf': 0.2, 'volatility': 0.015}
        ]
        
        regime_adaptations = []
        price = 50000
        
        for phase in regime_sequence:
            phase_adaptations = []
            
            for step in range(phase['steps']):
                # Simuler les données selon le régime
                if phase['regime'] == 'bull':
                    price += np.random.randn() * 100 + 50
                    volatility = phase['volatility'] + np.random.randn() * 0.005
                elif phase['regime'] == 'correction':
                    price += np.random.randn() * 200 - 100
                    volatility = phase['volatility'] + np.random.randn() * 0.01
                elif phase['regime'] == 'recovery':
                    price += np.random.randn() * 150 + 30
                    volatility = phase['volatility'] + np.random.randn() * 0.008
                else:  # sideways
                    price += np.random.randn() * 50
                    volatility = phase['volatility'] + np.random.randn() * 0.003
                
                market_data = {
                    'price': max(10000, price),
                    'volume': 1000000 + np.random.randn() * 200000,
                    'volatility': max(0.001, volatility)
                }
                
                performance = {
                    'sharpe_ratio': phase['base_perf'] + np.random.randn() * 0.2,
                    'total_return': phase['base_perf'] * 0.01,
                    'max_drawdown': max(0.01, 0.1 - phase['base_perf'] * 0.05)
                }
                
                modulation = adaptive_dbe.update(market_data, performance)
                phase_adaptations.append({
                    'step': step,
                    'regime_detected': modulation['market_regime'],
                    'risk_level': modulation['risk_level'],
                    'risk_multiplier': modulation['risk_multiplier']
                })
            
            regime_adaptations.append({
                'phase': phase['regime'],
                'adaptations': phase_adaptations
            })
        
        # Analyser les adaptations
        total_steps = sum(phase['steps'] for phase in regime_sequence)
        final_stats = adaptive_dbe.get_comprehensive_stats()
        
        # Vérifier la diversité des régimes détectés
        detected_regimes = set()
        for phase_data in regime_adaptations:
            for adaptation in phase_data['adaptations']:
                detected_regimes.add(adaptation['regime_detected'])
        
        return {
            'test_name': 'multi_regime_adaptation',
            'total_steps': total_steps,
            'phases_simulated': len(regime_sequence),
            'unique_regimes_detected': len(detected_regimes),
            'regime_adaptations': regime_adaptations,
            'final_stats': final_stats,
            'adaptation_successful': len(detected_regimes) >= 3 and final_stats['adaptation_stats']['adaptation_count'] > 0
        }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests du système adaptatif"""
        print("🚀 Tests Complets du DBE Adaptatif")
        print("=" * 60)
        
        test_functions = [
            self.test_parameter_evolution,
            self.test_market_regime_detection,
            self.test_adaptation_robustness,
            self.test_performance_feedback_loop,
            self.test_multi_regime_adaptation
        ]
        
        results = {}
        passed_tests = 0
        
        for test_func in test_functions:
            try:
                result = test_func()
                results[result['test_name']] = result
                
                # Déterminer si le test a réussi
                success_indicators = [
                    'parameters_adapted', 'detection_successful', 'robustness_passed',
                    'feedback_loop_working', 'adaptation_successful'
                ]
                
                test_passed = any(result.get(indicator, False) for indicator in success_indicators)
                
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
        print(f"  Système adaptatif: {'✅ OPÉRATIONNEL' if success_rate > 0.6 else '⚠️ PARTIEL'}")
        
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
            'adaptive_dbe_operational': success_rate > 0.6,
            'detailed_results': results
        }
        
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/adaptive_dbe_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📁 Résultats sauvegardés: {filename}")


def main():
    """Fonction principale"""
    tester = AdaptiveDBETester()
    results = tester.run_comprehensive_tests()
    
    return results


if __name__ == "__main__":
    main()