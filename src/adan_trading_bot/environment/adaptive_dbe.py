"""
Dynamic Behavior Engine avec méta-apprentissage adaptatif pour ADAN Trading Bot.
Implémente la tâche 9.2.1 - Évolution paramètres DBE.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Types de régimes de marché"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


@dataclass
class DBEParameters:
    """Paramètres du DBE avec valeurs par défaut"""
    # Seuils de risque
    risk_threshold_low: float = 0.3
    risk_threshold_medium: float = 0.6
    risk_threshold_high: float = 0.8
    
    # Paramètres de volatilité
    volatility_window: int = 20
    volatility_threshold_low: float = 0.01
    volatility_threshold_high: float = 0.05
    
    # Paramètres de drawdown
    max_drawdown_threshold: float = 0.15
    drawdown_recovery_factor: float = 0.5
    
    # Paramètres de performance
    min_sharpe_ratio: float = 0.5
    performance_window: int = 100
    
    # Paramètres d'adaptation
    adaptation_rate: float = 0.1
    learning_rate: float = 0.01
    momentum: float = 0.9
    
    # Paramètres de régime
    regime_detection_window: int = 50
    trend_threshold: float = 0.02
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DBEParameters':
        """Crée depuis un dictionnaire"""
        return cls(**data)


class ParameterEvolution:
    """Système d'évolution des paramètres DBE"""
    
    def __init__(self, initial_params: DBEParameters):
        self.current_params = initial_params
        self.param_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.gradient_estimates = {}
        self.momentum_terms = {}
        
        # Initialize gradient estimates and momentum
        for param_name in asdict(initial_params).keys():
            self.gradient_estimates[param_name] = 0.0
            self.momentum_terms[param_name] = 0.0
    
    def update_parameters(self, performance_metrics: Dict[str, float], 
                         market_conditions: Dict[str, float]) -> DBEParameters:
        """
        Met à jour les paramètres basé sur la performance et les conditions de marché.
        
        Args:
            performance_metrics: Métriques de performance récentes
            market_conditions: Conditions actuelles du marché
            
        Returns:
            Nouveaux paramètres DBE
        """
        # Enregistrer l'état actuel
        self.param_history.append(self.current_params.to_dict())
        self.performance_history.append(performance_metrics.copy())
        
        # Calculer les gradients estimés
        gradients = self._estimate_gradients(performance_metrics, market_conditions)
        
        # Mettre à jour les paramètres avec momentum
        new_params_dict = self.current_params.to_dict()
        
        for param_name, gradient in gradients.items():
            if param_name in new_params_dict:
                # Mise à jour du momentum
                self.momentum_terms[param_name] = (
                    self.current_params.momentum * self.momentum_terms[param_name] +
                    self.current_params.learning_rate * gradient
                )
                
                # Mise à jour du paramètre
                old_value = new_params_dict[param_name]
                new_value = old_value + self.momentum_terms[param_name]
                
                # Contraintes sur les valeurs
                new_value = self._apply_constraints(param_name, new_value)
                new_params_dict[param_name] = new_value
                
                logger.debug(f"Parameter {param_name}: {old_value:.4f} -> {new_value:.4f} "
                           f"(gradient: {gradient:.6f})")
        
        # Créer nouveaux paramètres
        self.current_params = DBEParameters.from_dict(new_params_dict)
        
        return self.current_params
    
    def _estimate_gradients(self, performance_metrics: Dict[str, float], 
                          market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Estime les gradients pour chaque paramètre"""
        gradients = {}
        
        if len(self.performance_history) < 2:
            return {param: 0.0 for param in asdict(self.current_params).keys()}
        
        # Performance actuelle vs précédente
        current_perf = performance_metrics.get('sharpe_ratio', 0.0)
        previous_perf = self.performance_history[-2].get('sharpe_ratio', 0.0)
        perf_delta = current_perf - previous_perf
        
        # Gradients basés sur la performance et les conditions de marché
        volatility = market_conditions.get('volatility', 0.02)
        drawdown = market_conditions.get('current_drawdown', 0.0)
        trend_strength = market_conditions.get('trend_strength', 0.0)
        
        # Adaptation des seuils de risque
        if perf_delta > 0 and drawdown < 0.05:
            # Performance positive, on peut être plus agressif
            gradients['risk_threshold_low'] = 0.01
            gradients['risk_threshold_medium'] = 0.01
            gradients['risk_threshold_high'] = 0.01
        elif perf_delta < 0 or drawdown > 0.1:
            # Performance négative ou drawdown élevé, être plus conservateur
            gradients['risk_threshold_low'] = -0.01
            gradients['risk_threshold_medium'] = -0.01
            gradients['risk_threshold_high'] = -0.01
        else:
            gradients['risk_threshold_low'] = 0.0
            gradients['risk_threshold_medium'] = 0.0
            gradients['risk_threshold_high'] = 0.0
        
        # Adaptation des seuils de volatilité
        if volatility > 0.04:  # Haute volatilité
            gradients['volatility_threshold_low'] = 0.001
            gradients['volatility_threshold_high'] = 0.002
        elif volatility < 0.01:  # Basse volatilité
            gradients['volatility_threshold_low'] = -0.001
            gradients['volatility_threshold_high'] = -0.001
        else:
            gradients['volatility_threshold_low'] = 0.0
            gradients['volatility_threshold_high'] = 0.0
        
        # Adaptation du seuil de drawdown
        if drawdown > 0.1:
            gradients['max_drawdown_threshold'] = -0.005  # Plus strict
        elif drawdown < 0.02 and perf_delta > 0:
            gradients['max_drawdown_threshold'] = 0.002   # Plus permissif
        else:
            gradients['max_drawdown_threshold'] = 0.0
        
        # Adaptation du Sharpe ratio minimum
        if current_perf > 1.0:
            gradients['min_sharpe_ratio'] = 0.01  # Augmenter les standards
        elif current_perf < 0.2:
            gradients['min_sharpe_ratio'] = -0.01  # Réduire les standards
        else:
            gradients['min_sharpe_ratio'] = 0.0
        
        # Adaptation du taux d'apprentissage
        if abs(perf_delta) > 0.1:  # Changements importants
            gradients['learning_rate'] = -0.001  # Réduire le learning rate
        elif abs(perf_delta) < 0.01:  # Changements faibles
            gradients['learning_rate'] = 0.0005  # Augmenter légèrement
        else:
            gradients['learning_rate'] = 0.0
        
        # Paramètres par défaut pour les autres
        for param_name in asdict(self.current_params).keys():
            if param_name not in gradients:
                gradients[param_name] = 0.0
        
        return gradients
    
    def _apply_constraints(self, param_name: str, value: float) -> float:
        """Applique les contraintes sur les valeurs des paramètres"""
        constraints = {
            'risk_threshold_low': (0.1, 0.5),
            'risk_threshold_medium': (0.3, 0.8),
            'risk_threshold_high': (0.6, 0.95),
            'volatility_threshold_low': (0.005, 0.02),
            'volatility_threshold_high': (0.02, 0.1),
            'max_drawdown_threshold': (0.05, 0.3),
            'drawdown_recovery_factor': (0.1, 0.9),
            'min_sharpe_ratio': (0.1, 2.0),
            'adaptation_rate': (0.01, 0.5),
            'learning_rate': (0.001, 0.1),
            'momentum': (0.5, 0.99),
            'trend_threshold': (0.005, 0.05),
            'volatility_window': (10, 100),
            'performance_window': (50, 500),
            'regime_detection_window': (20, 200)
        }
        
        if param_name in constraints:
            min_val, max_val = constraints[param_name]
            return np.clip(value, min_val, max_val)
        
        return value
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'adaptation"""
        if len(self.param_history) < 2:
            return {'adaptation_count': 0}
        
        # Calculer les changements de paramètres
        current = self.current_params.to_dict()
        initial = self.param_history[0]
        
        changes = {}
        for param_name in current.keys():
            if param_name in initial:
                change = abs(current[param_name] - initial[param_name])
                changes[param_name] = change
        
        return {
            'adaptation_count': len(self.param_history),
            'parameter_changes': changes,
            'total_adaptation': sum(changes.values()),
            'most_adapted_param': max(changes.keys(), key=lambda k: changes[k]) if changes else None,
            'current_learning_rate': self.current_params.learning_rate,
            'current_momentum': self.current_params.momentum
        }


class MarketRegimeDetector:
    """Détecteur de régimes de marché"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.volatility_history = deque(maxlen=window_size)
        
    def update(self, price: float, volume: float, volatility: float) -> None:
        """Met à jour l'historique des données"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.volatility_history.append(volatility)
    
    def detect_regime(self) -> MarketRegime:
        """Détecte le régime de marché actuel"""
        if len(self.price_history) < self.window_size:
            return MarketRegime.SIDEWAYS
        
        prices = np.array(self.price_history)
        volatilities = np.array(self.volatility_history)
        
        # Calcul de la tendance
        returns = np.diff(prices) / prices[:-1]
        trend_strength = np.mean(returns)
        volatility_level = np.mean(volatilities)
        
        # Détection de crise (volatilité extrême + rendements négatifs)
        if volatility_level > 0.08 and trend_strength < -0.02:
            return MarketRegime.CRISIS
        
        # Détection de volatilité
        if volatility_level > 0.05:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility_level < 0.01:
            return MarketRegime.LOW_VOLATILITY
        
        # Détection de tendance
        if trend_strength > 0.015:
            return MarketRegime.TRENDING_UP
        elif trend_strength < -0.015:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.SIDEWAYS
    
    def get_market_conditions(self) -> Dict[str, float]:
        """Retourne les conditions actuelles du marché"""
        if len(self.price_history) < 2:
            return {
                'volatility': 0.02,
                'trend_strength': 0.0,
                'current_drawdown': 0.0,
                'regime_stability': 1.0
            }
        
        prices = np.array(self.price_history)
        volatilities = np.array(self.volatility_history)
        
        # Calculs
        returns = np.diff(prices) / prices[:-1]
        trend_strength = np.mean(returns)
        volatility = np.mean(volatilities)
        
        # Drawdown
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        current_drawdown = drawdown[-1]
        
        # Stabilité du régime (variance des rendements)
        regime_stability = 1.0 / (1.0 + np.std(returns))
        
        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'current_drawdown': current_drawdown,
            'regime_stability': regime_stability
        }


class AdaptiveDBE:
    """Dynamic Behavior Engine avec adaptation automatique des paramètres"""
    
    def __init__(self, initial_params: Optional[DBEParameters] = None,
                 adaptation_enabled: bool = True,
                 save_path: str = "logs/adaptive_dbe"):
        
        self.params = initial_params or DBEParameters()
        self.adaptation_enabled = adaptation_enabled
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Composants d'adaptation
        self.param_evolution = ParameterEvolution(self.params)
        self.regime_detector = MarketRegimeDetector()
        
        # Historique et métriques
        self.performance_history = deque(maxlen=1000)
        self.regime_history = deque(maxlen=100)
        self.adaptation_log = []
        
        # Threading pour sauvegarde asynchrone
        self.save_lock = threading.Lock()
        
        logger.info("AdaptiveDBE initialized with parameter evolution enabled")
    
    def update(self, market_data: Dict[str, Any], 
               performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Met à jour le DBE avec adaptation des paramètres.
        
        Args:
            market_data: Données de marché actuelles
            performance_metrics: Métriques de performance
            
        Returns:
            Modulation DBE mise à jour
        """
        # Mise à jour du détecteur de régime
        price = market_data.get('price', 50000)
        volume = market_data.get('volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        
        self.regime_detector.update(price, volume, volatility)
        current_regime = self.regime_detector.detect_regime()
        market_conditions = self.regime_detector.get_market_conditions()
        
        # Enregistrer l'historique
        self.performance_history.append(performance_metrics.copy())
        self.regime_history.append(current_regime)
        
        # Adaptation des paramètres si activée
        if self.adaptation_enabled and len(self.performance_history) > 1:
            old_params = self.params.to_dict()
            self.params = self.param_evolution.update_parameters(
                performance_metrics, market_conditions
            )
            new_params = self.params.to_dict()
            
            # Log des changements significatifs
            significant_changes = []
            for param_name in old_params.keys():
                if param_name in new_params:
                    change = abs(new_params[param_name] - old_params[param_name])
                    if change > 0.001:  # Seuil de changement significatif
                        significant_changes.append({
                            'parameter': param_name,
                            'old_value': old_params[param_name],
                            'new_value': new_params[param_name],
                            'change': change
                        })
            
            if significant_changes:
                self.adaptation_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'regime': current_regime.value,
                    'performance': performance_metrics.copy(),
                    'changes': significant_changes
                })
                
                logger.info(f"DBE parameters adapted for regime {current_regime.value}: "
                           f"{len(significant_changes)} parameters changed")
        
        # Calcul de la modulation basée sur les paramètres actuels
        modulation = self._calculate_modulation(market_conditions, current_regime)
        
        # Sauvegarde périodique
        if len(self.adaptation_log) % 10 == 0:
            self._save_adaptation_state()
        
        return modulation
    
    def _calculate_modulation(self, market_conditions: Dict[str, float], 
                            regime: MarketRegime) -> Dict[str, Any]:
        """Calcule la modulation DBE basée sur les paramètres actuels"""
        volatility = market_conditions['volatility']
        drawdown = market_conditions['current_drawdown']
        trend_strength = market_conditions['trend_strength']
        
        # Niveau de risque basé sur les seuils adaptatifs
        if drawdown > self.params.max_drawdown_threshold:
            risk_level = "HIGH"
            risk_multiplier = 0.5
        elif volatility > self.params.volatility_threshold_high:
            risk_level = "MEDIUM"
            risk_multiplier = 0.7
        elif volatility < self.params.volatility_threshold_low:
            risk_level = "LOW"
            risk_multiplier = 1.2
        else:
            risk_level = "NORMAL"
            risk_multiplier = 1.0
        
        # Ajustements spécifiques au régime
        regime_adjustments = {
            MarketRegime.CRISIS: {'risk_multiplier': 0.3, 'position_size': 0.5},
            MarketRegime.HIGH_VOLATILITY: {'risk_multiplier': 0.6, 'position_size': 0.7},
            MarketRegime.LOW_VOLATILITY: {'risk_multiplier': 1.1, 'position_size': 1.0},
            MarketRegime.TRENDING_UP: {'risk_multiplier': 1.2, 'position_size': 1.1},
            MarketRegime.TRENDING_DOWN: {'risk_multiplier': 0.8, 'position_size': 0.8},
            MarketRegime.SIDEWAYS: {'risk_multiplier': 1.0, 'position_size': 0.9}
        }
        
        regime_adj = regime_adjustments.get(regime, {'risk_multiplier': 1.0, 'position_size': 1.0})
        
        return {
            'risk_level': risk_level,
            'risk_multiplier': risk_multiplier * regime_adj['risk_multiplier'],
            'position_size_multiplier': regime_adj['position_size'],
            'market_regime': regime.value,
            'volatility_level': volatility,
            'drawdown_level': drawdown,
            'trend_strength': trend_strength,
            'parameters_snapshot': self.params.to_dict(),
            'adaptation_enabled': self.adaptation_enabled
        }
    
    def _save_adaptation_state(self) -> None:
        """Sauvegarde l'état d'adaptation"""
        with self.save_lock:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                state = {
                    'timestamp': timestamp,
                    'current_parameters': self.params.to_dict(),
                    'adaptation_stats': self.param_evolution.get_adaptation_stats(),
                    'recent_adaptations': self.adaptation_log[-10:],  # 10 dernières adaptations
                    'regime_distribution': self._get_regime_distribution(),
                    'performance_summary': self._get_performance_summary()
                }
                
                filepath = self.save_path / f"adaptive_dbe_state_{timestamp}.json"
                with open(filepath, 'w') as f:
                    json.dump(state, f, indent=2)
                
                logger.debug(f"Adaptive DBE state saved to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save adaptive DBE state: {e}")
    
    def _get_regime_distribution(self) -> Dict[str, float]:
        """Calcule la distribution des régimes récents"""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for regime in self.regime_history:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        total = len(self.regime_history)
        return {regime: count / total for regime, count in regime_counts.items()}
    
    def _get_performance_summary(self) -> Dict[str, float]:
        """Calcule un résumé des performances récentes"""
        if not self.performance_history:
            return {}
        
        recent_perfs = list(self.performance_history)[-50:]  # 50 dernières
        
        sharpe_ratios = [p.get('sharpe_ratio', 0.0) for p in recent_perfs]
        returns = [p.get('total_return', 0.0) for p in recent_perfs]
        
        return {
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_return': np.mean(returns),
            'sharpe_std': np.std(sharpe_ratios),
            'return_std': np.std(returns),
            'performance_trend': np.polyfit(range(len(sharpe_ratios)), sharpe_ratios, 1)[0] if len(sharpe_ratios) > 1 else 0.0
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques complètes du système adaptatif"""
        return {
            'current_parameters': self.params.to_dict(),
            'adaptation_stats': self.param_evolution.get_adaptation_stats(),
            'regime_distribution': self._get_regime_distribution(),
            'performance_summary': self._get_performance_summary(),
            'adaptation_log_size': len(self.adaptation_log),
            'total_updates': len(self.performance_history),
            'adaptation_enabled': self.adaptation_enabled
        }
    
    def reset_adaptation(self) -> None:
        """Remet à zéro le système d'adaptation"""
        self.params = DBEParameters()  # Paramètres par défaut
        self.param_evolution = ParameterEvolution(self.params)
        self.performance_history.clear()
        self.regime_history.clear()
        self.adaptation_log.clear()
        
        logger.info("Adaptive DBE system reset to default parameters")