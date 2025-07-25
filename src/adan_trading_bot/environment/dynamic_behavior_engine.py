"""
Dynamic Behavior Engine (DBE) - Module de contrôle adaptatif pour l'agent de trading.

Ce module implémente un système de modulation dynamique des paramètres de trading
en fonction des performances et des conditions de marché en temps réel.
"""
from typing import Dict, Any, Optional, List, Union
import numpy as np
import json
import os
from datetime import datetime # Import datetime
from dataclasses import dataclass, field
import logging
import pickle
from pathlib import Path

import yaml
from pathlib import Path

from ..common.utils import get_logger
from ..common.replay_logger import ReplayLogger

logger = get_logger(__name__)

DBE_LOG_FILE = os.getenv("DBE_LOG_FILE", "dbe_replay.jsonl")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class DBESnapshot:
    """Snapshot de l'état du DBE à un instant T."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step: int = 0
    market_regime: str = "NEUTRAL"
    risk_level: float = 1.0
    sl_pct: float = 0.02
    tp_pct: float = 0.04
    position_size_pct: float = 0.1
    reward_boost: float = 1.0
    penalty_inaction: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

DEFAULT_CONFIG = {
    'risk_parameters': {
        'base_sl_pct': 0.02,
        'base_tp_pct': 0.04,
        'max_sl_pct': 0.10,
        'min_sl_pct': 0.005,
        'drawdown_risk_multiplier': 2.0,
        'volatility_impact': 1.5,
    },
    'position_sizing': {
        'base_position_size': 0.1,
        'max_position_size': 0.3,
        'min_position_size': 0.01,
    },
    'modes': {
        'volatile': {'sl_multiplier': 1.3, 'tp_multiplier': 0.8, 'position_size_multiplier': 0.7},
        'sideways': {'sl_multiplier': 0.8, 'tp_multiplier': 0.8, 'position_size_multiplier': 0.9},
        'bull': {'sl_multiplier': 0.9, 'tp_multiplier': 1.2, 'position_size_multiplier': 1.1},
        'bear': {'sl_multiplier': 1.1, 'tp_multiplier': 0.9, 'position_size_multiplier': 0.8},
    },
    'learning': {
        'learning_rate_range': [1e-5, 1e-3],
        'ent_coef_range': [0.001, 0.1],
        'gamma_range': [0.9, 0.999]
    }
}

class DynamicBehaviorEngine:
    """
    Moteur de comportement dynamique avancé qui ajuste les paramètres de trading
    en fonction des conditions de marché, de la performance du portefeuille
    et de l'état interne de l'agent.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 finance_manager: Optional[Any] = None):
        """
        Initialise le DBE avec la configuration fournie.
        
        Args:
            config: Dictionnaire de configuration (optionnel)
            finance_manager: Instance de FinanceManager (optionnel)
        """
        # Fusion de la configuration par défaut avec celle fournie
        # Start with the default configuration
        self.config = DEFAULT_CONFIG.copy()

        # Load external config from dbe_config.yaml if it exists
        dbe_config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'dbe_config.yaml'
        if dbe_config_path.exists():
            with open(dbe_config_path, 'r') as f:
                dbe_config = yaml.safe_load(f)
                if dbe_config:
                    # Deep merge the loaded config into our existing config
                    def deep_update(d, u):
                        for k, v in u.items():
                            if isinstance(v, dict):
                                d[k] = deep_update(d.get(k, {}), v)
                            else:
                                d[k] = v
                        return d
                    deep_update(self.config, dbe_config)

        # Merge the provided config argument, which takes the highest precedence
        if config:
            self.config.update(config)
        
        # Initialisation du gestionnaire financier
        self.finance_manager = finance_manager
        
        # État interne
        self.state = {
            'current_step': 0,
            'drawdown': 0.0,
            'winrate': 0.0,
            'volatility': 0.0,
            'market_regime': 'NEUTRAL',
            'last_trade_pnl': 0.0,
            'consecutive_losses': 0,
            'position_duration': 0,
            'current_risk_level': 1.0,
            'max_risk_level': 2.0,
            'min_risk_level': 0.5,
            'last_modulation': {},
            'performance_metrics': {}
        }
        
        # Historique des trades et des décisions
        self.trade_history: List[Dict[str, Any]] = []
        self.decision_history: List[DBESnapshot] = []
        self.win_rates = []
        self.drawdowns = []
        self.position_durations = []
        self.pnl_history = []
        self.trade_results = []
        
        # Initialisation du logger de relecture
        log_config = self.config.get('logging', {})
        self.logger = ReplayLogger(
            log_dir=log_config.get('log_dir', 'logs/dbe'),
            compression=log_config.get('compression', 'gzip')
        )
        
        # Configuration du niveau de log
        log_level = log_config.get('log_level', 'INFO').upper()
        logging.getLogger().setLevel(getattr(logging, log_level))
        
        # Initialisation des paramètres de lissage
        self.smoothing_factor = self.config.get('smoothing_factor', 0.1)  # Facteur de lissage exponentiel
        self.smoothed_params = {
            'sl_pct': self.config.get('risk_parameters', {}).get('base_sl_pct', 0.02),
            'tp_pct': self.config.get('risk_parameters', {}).get('base_tp_pct', 0.04)
        }
        
        # Configuration de la persistance d'état
        self.state_persistence_enabled = config.get('state_persistence', {}).get('enabled', True) if config else True
        self.state_save_path = config.get('state_persistence', {}).get('save_path', 'logs/dbe/state') if config else 'logs/dbe/state'
        self.state_save_interval = config.get('state_persistence', {}).get('save_interval', 100) if config else 100
        
        # Créer le répertoire de sauvegarde si nécessaire
        if self.state_persistence_enabled:
            Path(self.state_save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 Dynamic Behavior Engine initialisé (version avancée)")
        logger.info(f"Configuration: {json.dumps(self._serialize_config(), indent=2)}")
        logger.info(f"Persistance d'état: {'Activée' if self.state_persistence_enabled else 'Désactivée'}")

    def _serialize_config(self) -> Dict[str, Any]:
        """Sérialise la configuration pour le logging."""
        # Crée une copie profonde pour éviter de modifier la configuration originale
        config = self.config.copy()
        
        # Éviter de logger des informations sensibles
        if 'api_keys' in config:
            config['api_keys'] = {k: '***' for k in config['api_keys']}
            
        return config
    
    def update_state(self, live_metrics: Dict[str, Any]) -> None:
        """
        Met à jour l'état interne du DBE avec les dernières métriques.
        
        Args:
            live_metrics: Dictionnaire des métriques en temps réel
        """
        try:
            # Mise à jour des métriques de base
            self.state['current_step'] = live_metrics.get('step', self.state['current_step'] + 1)
            
            # Mise à jour des métriques du gestionnaire financier si disponible
            if self.finance_manager and 'current_prices' in live_metrics:
                self.finance_manager.update_market_value(live_metrics['current_prices'])
                portfolio_metrics = self.finance_manager.get_performance_metrics()
                
                # Mise à jour de l'état avec les métriques financières
                self.state.update({
                    'drawdown': portfolio_metrics.get('current_drawdown', 0.0) * 100,  # en pourcentage
                    'winrate': portfolio_metrics.get('win_rate', 0.0),
                    'total_trades': portfolio_metrics.get('trade_count', 0)
                })
            
            # Détection du régime de marché
            self.state['market_regime'] = self._detect_market_regime(live_metrics)
            
            # Mise à jour des métriques de trading si disponibles
            if 'trade_result' in live_metrics:
                trade_result = live_metrics['trade_result']
                self._process_trade_result(trade_result)
            
            # Ajustement du niveau de risque
            self._adjust_risk_level()

            # Adaptation du facteur de lissage
            self._adapt_smoothing_factor()
            
            logger.debug(f"État DBE mis à jour - Step: {self.state['current_step']} | "
                       f"Régime: {self.state['market_regime']} | "
                       f"Winrate: {self.state['winrate']*100:.1f}% | "
                       f"Drawdown: {self.state['drawdown']:.2f}%")
                       
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'état du DBE: {e}", exc_info=True)
            raise
    
    def _process_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """Traite le résultat d'un trade et met à jour les métriques."""
        self.state['last_trade_pnl'] = trade_result.get('pnl_pct', 0.0)
        
        # Mise à jour du nombre de pertes consécutives
        if trade_result.get('pnl_pct', 0) <= 0:
            self.state['consecutive_losses'] += 1
        else:
            self.state['consecutive_losses'] = 0
            
        # Mise à jour de la durée de position
        if 'position_duration' in trade_result:
            self.position_durations.append(trade_result['position_duration'])
            self.state['position_duration'] = np.mean(self.position_durations[-100:]) if self.position_durations else 0
        
        # Mise à jour du taux de réussite
        if 'is_win' in trade_result:
            self.win_rates.append(1 if trade_result['is_win'] else 0)
            self.state['winrate'] = np.mean(self.win_rates[-100:]) if self.win_rates else 0.0
        
        # Mise à jour du drawdown
        if 'drawdown' in trade_result:
            self.drawdowns.append(trade_result['drawdown'])
            self.state['drawdown'] = np.mean(self.drawdowns[-100:]) if self.drawdowns else 0.0
        
        # Ajout à l'historique des trades
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'pnl_pct': trade_result.get('pnl_pct', 0.0),
            'is_win': trade_result.get('is_win', False),
            'position_duration': trade_result.get('position_duration', 0),
            'drawdown': trade_result.get('drawdown', 0.0),
            'market_regime': self.state['market_regime']
        })

    @property
    def market_regime(self) -> str:
        """Get current market regime."""
        return self.state.get('market_regime', 'NEUTRAL')

    @property
    def current_step(self) -> int:
        """Get current step."""
        return self.state.get('current_step', 0)

    @property
    def risk_level(self) -> float:
        """Get current risk level."""
        return self.state.get('current_risk_level', 1.0)

    def save_state(self, filepath: str) -> None:
        """Save the current state to a file."""
        state_data = {
            'state': self.state,
            'trade_history': self.trade_history,
            'decision_history': [vars(snapshot) for snapshot in self.decision_history],
            'win_rates': self.win_rates,
            'drawdowns': self.drawdowns,
            'position_durations': self.position_durations
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
        
        logger.info(f"DBE state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load state from a file."""
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        
        self.state = state_data['state']
        self.trade_history = state_data['trade_history']
        self.decision_history = [DBESnapshot(**snapshot) for snapshot in state_data['decision_history']]
        self.win_rates = state_data['win_rates']
        self.drawdowns = state_data['drawdowns']
        self.position_durations = state_data['position_durations']
        
        logger.info(f"DBE state loaded from {filepath}")

    def get_status(self) -> str:
        """Get current status as a formatted string."""
        return f"DBE Status - Step: {self.current_step}, Regime: {self.market_regime}, Risk: {self.risk_level:.2f}"

    def on_trade_closed(self, trade_result: Dict[str, Any]) -> None:
        """Process a closed trade result."""
        self._process_trade_result(trade_result)

    def compute_dynamic_modulation(self) -> Dict[str, Any]:
        """
        Calcule la modulation dynamique des paramètres de trading.
        
        Returns:
            Dictionnaire contenant les paramètres modulés
        """
        try:
            # Initialisation des paramètres de base
            mod = {
                'sl_pct': self.config.get('risk_parameters', {}).get('base_sl_pct', 0.02),
                'tp_pct': self.config.get('risk_parameters', {}).get('base_tp_pct', 0.04),
                'reward_boost': 1.0,
                'penalty_inaction': 0.0,
                'position_size_pct': self.config.get('position_sizing', {}).get('base_position_size', 0.1),
                'leverage': self.config.get('position_sizing', {}).get('base_leverage', 1.0),
                'risk_mode': 'NORMAL',  # 'DEFENSIVE', 'NORMAL', 'AGGRESSIVE'
                'learning_rate': None,
                'ent_coef': None,
                'gamma': None
            }
            
            # Calcul des paramètres de risque
            self._compute_risk_parameters(self.state, mod)
            
            # Application des modulations spécifiques au régime de marché
            self._apply_market_regime_modulation(mod)
            
            # Ajustement des paramètres d'apprentissage
            self._adjust_learning_parameters(mod)
            
            # Validation et ajustement final des paramètres
            self._validate_parameters(mod)
            
            # Création d'un snapshot de la décision
            snapshot = DBESnapshot(
                step=self.state['current_step'],
                market_regime=self.state['market_regime'],
                risk_level=self.state['current_risk_level'],
                sl_pct=mod['sl_pct'],
                tp_pct=mod['tp_pct'],
                position_size_pct=mod['position_size_pct'],
                reward_boost=mod['reward_boost'],
                penalty_inaction=mod['penalty_inaction'],
                metrics=self.state['performance_metrics'].copy()
            )
            
            # Ajout à l'historique des décisions
            self.decision_history.append(snapshot)
            self.state['last_modulation'] = mod.copy()
            
            # Journalisation de la décision
            self._log_decision(snapshot, mod)
            
            return mod
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la modulation: {e}", exc_info=True)
            # Retourner une modulation sécurisée en cas d'erreur
            return {
                'sl_pct': 0.02,
                'tp_pct': 0.04,
                'reward_boost': 1.0,
                'penalty_inaction': 0.0,
                'position_size_pct': 0.1,
                'risk_mode': 'NORMAL',
                'error': str(e)
            }
    

    
    def _apply_market_regime_modulation(self, mod: Dict[str, Any]) -> None:
        """Applique les modulations spécifiques au régime de marché."""
        regime = self.state['market_regime'].upper()
        mode_config = self.config.get('modes', {}).get(regime.lower(), {})
        
        if not mode_config:
            return
            
        # Application des multiplicateurs
        mod['sl_pct'] *= mode_config.get('sl_multiplier', 1.0)
        mod['tp_pct'] *= mode_config.get('tp_multiplier', 1.0)
        mod['position_size_pct'] *= mode_config.get('position_size_multiplier', 1.0)
        mod['risk_mode'] = regime
    
    def _adjust_learning_parameters(self, mod: Dict[str, Any]) -> None:
        """Ajuste les paramètres d'apprentissage en fonction du risque."""
        learning_config = self.config.get('learning', {})
        lr_range = [float(x) for x in learning_config.get('learning_rate_range', [1e-5, 1e-3])]
        ent_coef_range = [float(x) for x in learning_config.get('ent_coef_range', [0.001, 0.1])]
        gamma_range = [float(x) for x in learning_config.get('gamma_range', [0.9, 0.999])]
        
        # Ajustement basé sur le niveau de risque
        risk_factor = self.state['current_risk_level']
        
        # Plus de risque = learning rate plus élevé, plus d'exploration
        mod['learning_rate'] = lr_range[0] + (lr_range[1] - lr_range[0]) * (risk_factor - 1.0)
        mod['ent_coef'] = ent_coef_range[0] + (ent_coef_range[1] - ent_coef_range[0]) * (1.0 / risk_factor)
        mod['gamma'] = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * (risk_factor - 1.0)
    
    def _validate_parameters(self, mod: Dict[str, Any]) -> None:
        """Valide et contraint les paramètres dans des limites acceptables."""
        risk_params = self.config.get('risk_parameters', {})
        pos_params = self.config.get('position_sizing', {})
        
        # Contraintes sur les SL/TP
        mod['sl_pct'] = np.clip(
            mod['sl_pct'],
            risk_params.get('min_sl_pct', 0.005),
            risk_params.get('max_sl_pct', 0.10)
        )
        
        # TP minimum = 1.5x SL pour assurer un ratio risque/rendement positif
        min_tp_ratio = 1.5
        mod['tp_pct'] = max(mod['tp_pct'], mod['sl_pct'] * min_tp_ratio)
        
        # Contraintes sur la taille de position
        mod['position_size_pct'] = np.clip(
            mod['position_size_pct'],
            pos_params.get('min_position_size', 0.01),
            pos_params.get('max_position_size', 0.30)
        )
    
    def _log_decision(self, snapshot: DBESnapshot, mod: Dict[str, Any]) -> None:
        """Journalise la décision prise par le DBE."""
        decision_data = {
            'step': snapshot.step,
            'market_regime': snapshot.market_regime,
            'risk_level': snapshot.risk_level,
            'modulation': {
                'sl_pct': snapshot.sl_pct,
                'tp_pct': snapshot.tp_pct,
                'position_size_pct': snapshot.position_size_pct,
                'reward_boost': snapshot.reward_boost,
                'penalty_inaction': snapshot.penalty_inaction,
                'learning_rate': mod.get('learning_rate'),
                'ent_coef': mod.get('ent_coef'),
                'gamma': mod.get('gamma')
            },
            'performance_metrics': snapshot.metrics,
            'timestamp': snapshot.timestamp.isoformat()
        }
        
        # Utilisation du ReplayLogger pour enregistrer la décision
        self.logger.log_decision(
            step_index=snapshot.step,
            modulation_dict=decision_data['modulation'],
            context_metrics={
                'market_regime': snapshot.market_regime,
                'risk_level': snapshot.risk_level,
                'drawdown': self.state['drawdown'],
                'winrate': self.state['winrate'],
                'volatility': self.state['volatility']
            },
            performance_metrics=snapshot.metrics,
            additional_info={
                'consecutive_losses': self.state['consecutive_losses'],
                'position_duration': self.state['position_duration']
            }
        )
        
        logger.info(
            f"DBE Decision - Step: {snapshot.step} | "
            f"Regime: {snapshot.market_regime} | "
            f"SL: {snapshot.sl_pct*100:.2f}% | "
            f"TP: {snapshot.tp_pct*100:.2f}% | "
            f"PosSize: {snapshot.position_size_pct*100:.1f}% | "
            f"Winrate: {self.state['winrate']*100:.1f}%"
        )
    
    def _detect_market_regime(self, live_metrics: Dict[str, Any]) -> str:
        """
        Détecte le régime de marché actuel.
        
        Args:
            live_metrics: Métriques en temps réel du marché
            
        Returns:
            Chaîne identifiant le régime de marché
        """
        try:
            # Récupération des indicateurs techniques
            rsi = live_metrics.get('rsi', 50)
            adx = live_metrics.get('adx', 20)
            ema_ratio = live_metrics.get('ema_ratio', 1.0)  # Ratio EMA rapide / lente
            atr = live_metrics.get('atr', 0.0)
            atr_pct = live_metrics.get('atr_pct', 0.0)  # ATR en pourcentage du prix
            
            # Détection du régime de marché
            if adx > 25:  # Marché avec tendance
                if ema_ratio > 1.005:  # Tendance haussière
                    return 'BULL'
                elif ema_ratio < 0.995:  # Tendance baissière
                    return 'BEAR'
            
            # Marché sans tendance
            if atr_pct > 0.02:  # Volatilité élevée
                return 'VOLATILE'
            else:
                return 'SIDEWAYS'
                
        except Exception as e:
            logger.error(f"Erreur lors de la détection du régime de marché: {e}")
            return 'UNKNOWN'
    
    def _adjust_risk_level(self) -> None:
        """Ajuste le niveau de risque en fonction des performances récentes."""
        try:
            # Facteurs d'ajustement
            winrate_factor = self.state['winrate'] / 0.6  # Normalisé par rapport à un winrate cible de 60%
            drawdown_factor = 1.0 - (self.state['drawdown'] / 100.0)  # Réduction du risque avec le drawdown
            loss_streak_factor = 1.0 / (1.0 + self.state['consecutive_losses'] * 0.2)  # Réduction après des pertes consécutives
            
            # Calcul du nouveau niveau de risque
            new_risk = self.state['current_risk_level'] * winrate_factor * drawdown_factor * loss_streak_factor
            
            # Lissage pour éviter les changements trop brutaux
            alpha = 0.2  # Facteur de lissage
            smoothed_risk = (1 - alpha) * self.state['current_risk_level'] + alpha * new_risk
            
            # Application des limites
            self.state['current_risk_level'] = np.clip(
                smoothed_risk,
                self.state['min_risk_level'],
                self.state['max_risk_level']
            )
            
            logger.debug(
                f"Ajustement du risque - Niveau: {self.state['current_risk_level']:.2f} | "
                f"Winrate: {self.state['winrate']*100:.1f}% | "
                f"Drawdown: {self.state['drawdown']:.2f}% | "
                f"Pertes consécutives: {self.state['consecutive_losses']}"
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajustement du niveau de risque: {e}")
            # En cas d'erreur, on reste sur le niveau de risque actuel
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de performance actuelles.
        
        Returns:
            Dictionnaire des métriques de performance
        """
        if not self.finance_manager:
            return {}
            
        # Récupération des métriques du gestionnaire financier
        portfolio_metrics = self.finance_manager.get_performance_metrics()
        
        # Calcul des métriques avancées
        if self.trade_history:
            recent_trades = self.trade_history[-100:]  # 100 derniers trades
            pnls = [t['pnl_pct'] for t in recent_trades if 'pnl_pct' in t]
            wins = [t for t in recent_trades if t.get('is_win', False)]
            losses = [t for t in recent_trades if not t.get('is_win', True)]
            
            avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0.0
            avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 0.0
            win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
            
            sharpe_ratio = self._calculate_sharpe_ratio(pnls) if pnls else 0.0
            sortino_ratio = self._calculate_sortino_ratio(pnls) if pnls else 0.0
        else:
            avg_win = avg_loss = win_loss_ratio = sharpe_ratio = sortino_ratio = 0.0
        
        # Construction du dictionnaire de résultats
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'step': self.state['current_step'],
            'portfolio': {
                'total_value': portfolio_metrics.get('total_capital', 0.0),
                'free_cash': portfolio_metrics.get('free_capital', 0.0),
                'invested': portfolio_metrics.get('invested_capital', 0.0),
                'total_return': portfolio_metrics.get('total_return', 0.0),
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0.0),
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            },
            'trading': {
                'total_trades': portfolio_metrics.get('trade_count', 0),
                'win_rate': portfolio_metrics.get('win_rate', 0.0) * 100,  # en pourcentage
                'avg_win_pct': avg_win * 100,  # en pourcentage
                'avg_loss_pct': avg_loss * 100,  # en pourcentage
                'win_loss_ratio': win_loss_ratio,
                'consecutive_losses': self.state['consecutive_losses'],
                'avg_trade_duration': self.state.get('position_duration', 0)
            },
            'risk': {
                'current_risk_level': self.state['current_risk_level'],
                'market_regime': self.state['market_regime'],
                'current_volatility': self.state.get('volatility', 0.0),
                'current_drawdown': self.state.get('drawdown', 0.0)
            }
        }
        
        # Mise à jour des métriques de performance dans l'état
        self.state['performance_metrics'] = metrics
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sharpe annualisé."""
        if not returns:
            return 0.0
            
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate / 252  # Taux sans risque journalier
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sortino annualisé."""
        if not returns:
            return 0.0
            
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate / 252  # Taux sans risque journalier
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / (downside_std + 1e-9) * np.sqrt(252)
        return float(sortino)
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retourne l'historique des décisions prises par le DBE.
        
        Args:
            limit: Nombre maximum de décisions à retourner
            
        Returns:
            Liste des décisions au format dictionnaire
        """
        # Sélection des décisions les plus récentes
        recent_decisions = self.decision_history[-limit:] if self.decision_history else []
        
        # Conversion des snapshots en dictionnaires
        return [{
            'timestamp': d.timestamp.isoformat(),
            'step': d.step,
            'market_regime': d.market_regime,
            'risk_level': d.risk_level,
            'sl_pct': d.sl_pct,
            'tp_pct': d.tp_pct,
            'position_size_pct': d.position_size_pct,
            'reward_boost': d.reward_boost,
            'penalty_inaction': d.penalty_inaction,
            'metrics': d.metrics
        } for d in recent_decisions]
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retourne l'historique des trades effectués.
        
        Args:
            limit: Nombre maximum de trades à retourner
            
        Returns:
            Liste des trades au format dictionnaire
        """
        # Sélection des trades les plus récents
        recent_trades = self.trade_history[-limit:] if self.trade_history else []
        
        # Conversion des timestamps en chaînes
        return [{
            'timestamp': t['timestamp'].isoformat() if hasattr(t['timestamp'], 'isoformat') else str(t['timestamp']),
            'pnl_pct': t.get('pnl_pct', 0.0),
            'is_win': t.get('is_win', False),
            'position_duration': t.get('position_duration', 0),
            'drawdown': t.get('drawdown', 0.0),
            'market_regime': t.get('market_regime', 'UNKNOWN')
        } for t in recent_trades]
    
    def save_state(self, filepath: Union[str, Path]) -> bool:
        """
        Sauvegarde l'état actuel du DBE dans un fichier.
        
        Args:
            filepath: Chemin vers le fichier de sauvegarde
            
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            state = {
                'state': self.state,
                'trade_history': self.trade_history,
                'decision_history': [d.__dict__ for d in self.decision_history],
                'win_rates': self.win_rates,
                'drawdowns': self.drawdowns,
                'position_durations': self.position_durations,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"État du DBE sauvegardé dans {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état du DBE: {e}")
            return False
    
    @classmethod
    def load_state(cls, filepath: Union[str, Path], finance_manager: Optional[Any] = None) -> Optional['DynamicBehaviorEngine']:
        """
        Charge un état précédemment sauvegardé.
        
        Args:
            filepath: Chemin vers le fichier de sauvegarde
            finance_manager: Instance de FinanceManager (optionnel)
            
        Returns:
            Une instance de DynamicBehaviorEngine avec l'état chargé, ou None en cas d'erreur
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Création d'une nouvelle instance avec la configuration sauvegardée
            dbe = cls(config=state.get('config', {}), finance_manager=finance_manager)
            
            # Restauration de l'état
            dbe.state = state.get('state', {})
            dbe.trade_history = state.get('trade_history', [])
            dbe.decision_history = [DBESnapshot(**d) for d in state.get('decision_history', [])]
            dbe.win_rates = state.get('win_rates', [])
            dbe.drawdowns = state.get('drawdowns', [])
            dbe.position_durations = state.get('position_durations', [])
            
            logger.info(f"État du DBE chargé depuis {filepath}")
            return dbe
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état du DBE: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retourne un résumé de l'état actuel du DBE.
        
        Returns:
            Dictionnaire contenant les informations de statut
        """
        if not self.finance_manager:
            portfolio_value = 0.0
            free_cash = 0.0
        else:
            metrics = self.finance_manager.get_performance_metrics()
            portfolio_value = metrics.get('total_capital', 0.0)
            free_cash = metrics.get('free_capital', 0.0)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'step': self.state['current_step'],
            'market_regime': self.state['market_regime'],
            'risk_level': self.state['current_risk_level'],
            'portfolio_value': portfolio_value,
            'free_cash': free_cash,
            'drawdown': self.state['drawdown'],
            'winrate': self.state['winrate'],
            'consecutive_losses': self.state['consecutive_losses'],
            'last_modulation': self.state.get('last_modulation', {}),
            'total_decisions': len(self.decision_history),
            'total_trades': len(self.trade_history)
        }
    
    def reset(self) -> None:
        """Réinitialise l'état interne du DBE."""
        # Réinitialisation de l'état
        self.state = {
            'current_step': 0,
            'drawdown': 0.0,
            'winrate': 0.0,
            'volatility': 0.0,
            'market_regime': 'NEUTRAL',
            'last_trade_pnl': 0.0,
            'consecutive_losses': 0,
            'position_duration': 0,
            'current_risk_level': 1.0,
            'max_risk_level': 2.0,
            'min_risk_level': 0.5,
            'last_modulation': {},
            'performance_metrics': {}
        }
        
        # Réinitialisation des historiques
        self.trade_history = []
        self.decision_history = []
        self.win_rates = []
        self.drawdowns = []
        self.position_durations = []
        self.pnl_history = []
        self.trade_results = []

        # Réinitialisation des paramètres lissés aux valeurs de base
        self.smoothed_params = {
            'sl_pct': self.config.get('risk_parameters', {}).get('base_sl_pct', 0.02),
            'tp_pct': self.config.get('risk_parameters', {}).get('base_tp_pct', 0.04)
        }
        
        # Réinitialisation du gestionnaire financier si disponible
        if self.finance_manager:
            self.finance_manager.reset()
        
        logger.info("DBE réinitialisé")
    
    def _reset_for_new_chunk(self) -> None:
        """
        Réinitialise les métriques spécifiques au chunk, mais conserve l'historique
        et les paramètres lissés pour la continuité entre les chunks.
        """
        self.state['current_step'] = 0
        self.state['last_trade_pnl'] = 0.0
        self.state['consecutive_losses'] = 0
        self.state['position_duration'] = 0
        self.state['volatility'] = 0.0
        self.state['market_regime'] = 'NEUTRAL'
        self.state['trend_strength'] = 0.0
        
        # Les historiques (trade_history, decision_history, win_rates, drawdowns, pnl_history, trade_results)
        # et les smoothed_params, current_risk_level sont conservés pour la continuité.
        
        logger.info("🔄 DBE: Réinitialisation pour un nouveau chunk (continuité préservée)")

    def _adapt_smoothing_factor(self) -> None:
        """
        Adapte le facteur de lissage (smoothing_factor) en fonction des performances récentes.
        - Réduit le lissage (augmente smoothing_factor) si les performances sont bonnes (winrate élevé, faible drawdown).
        - Augmente le lissage (diminue smoothing_factor) si les performances sont mauvaises (winrate faible, drawdown élevé).
        """
        current_winrate = self.state.get('winrate', 0.0)
        current_drawdown = self.state.get('drawdown', 0.0)

        # Paramètres de configuration pour l'adaptation du lissage
        adapt_config = self.config.get('smoothing_adaptation', {
            'min_smoothing': 0.01,
            'max_smoothing': 0.5,
            'winrate_threshold_good': 0.6,
            'winrate_threshold_bad': 0.4,
            'drawdown_threshold_good': 5.0, # in percent
            'drawdown_threshold_bad': 15.0, # in percent
            'adaptation_rate': 0.01
        })

        min_smoothing = adapt_config['min_smoothing']
        max_smoothing = adapt_config['max_smoothing']
        winrate_threshold_good = adapt_config['winrate_threshold_good']
        winrate_threshold_bad = adapt_config['winrate_threshold_bad']
        drawdown_threshold_good = adapt_config['drawdown_threshold_good']
        drawdown_threshold_bad = adapt_config['drawdown_threshold_bad']
        adaptation_rate = adapt_config['adaptation_rate']

        new_smoothing_factor = self.smoothing_factor

        # Ajustement basé sur le winrate
        if current_winrate > winrate_threshold_good:
            new_smoothing_factor += adaptation_rate # Reduce smoothing (faster adaptation)
        elif current_winrate < winrate_threshold_bad:
            new_smoothing_factor -= adaptation_rate # Increase smoothing (slower adaptation)

        # Ajustement basé sur le drawdown
        if current_drawdown < drawdown_threshold_good: # Lower drawdown is good
            new_smoothing_factor += adaptation_rate
        elif current_drawdown > drawdown_threshold_bad: # Higher drawdown is bad
            new_smoothing_factor -= adaptation_rate

        # Clip the smoothing factor to stay within bounds
        self.smoothing_factor = np.clip(new_smoothing_factor, min_smoothing, max_smoothing)
        logger.debug(f"Smoothing factor adapted to: {self.smoothing_factor:.3f} (Winrate: {current_winrate:.2f}, Drawdown: {current_drawdown:.2f})")

    def __str__(self) -> str:
        """Représentation textuelle de l'état du DBE."""
        status = self.get_status()
        return (
            f"DBE Status (Step: {status['step']})\n"
            f"Portfolio: ${status['portfolio']['total_value']:,.2f} "
            f"(Return: {status['portfolio']['total_return_pct']:.2f}%)\n"
            f"Trades: {status['trading']['total_trades']} "
            f"(Win Rate: {status['trading']['win_rate']:.1f}%)\n"
            f"Risk: {status['risk']['current_risk_level']:.2f} "
            f"(Regime: {status['risk']['market_regime']})\n"
            f"Drawdown: {status['risk']['current_drawdown']:.2f}% | "
            f"Volatility: {status['risk']['volatility']:.4f}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration actuelle du DBE.
        
        Returns:
            Dictionnaire de configuration
        """
        # Retourne une copie pour éviter les modifications accidentelles
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Met à jour la configuration du DBE.
        
        Args:
            new_config: Dictionnaire contenant les nouvelles valeurs de configuration
        """
        # Mise à jour récursive de la configuration
        def deep_update(current: Dict[str, Any], new: Dict[str, Any]) -> None:
            for key, value in new.items():
                if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                    deep_update(current[key], value);
                else:
                    current[key] = value
        
        # Application de la mise à jour
        deep_update(self.config, new_config)
        logger.info("Configuration du DBE mise à jour")
        
        # Mise à jour du niveau de log si nécessaire
        if 'logging' in new_config and 'log_level' in new_config['logging']:
            log_level = new_config['logging']['log_level'].upper()
            logging.getLogger().setLevel(getattr(logging, log_level))

    def __del__(self):
        if hasattr(self, 'dbe_log_file') and not self.dbe_log_file.closed:
            self.dbe_log_file.close()
    
    def _compute_risk_parameters(self, state: Dict[str, Any], mod: Dict[str, Any]) -> None:
        """Calcule les paramètres de risque dynamiques (SL/TP)."""
        # Valeurs de base de la configuration
        risk_cfg = self.config.get('risk', {})
        base_sl = risk_cfg.get('base_sl_pct', 0.02)
        base_tp = risk_cfg.get('base_tp_pct', 0.04)
        
        # Ajustement basé sur le drawdown
        drawdown_factor = state['drawdown'] * risk_cfg.get('drawdown_sl_factor', 0.5)
        
        # Ajustement basé sur la volatilité
        vol_factor = state['volatility'] * risk_cfg.get('volatility_sl_factor', 0.1)
        
        # Ajustement basé sur le winrate
        winrate_factor = (0.5 - state['winrate']) * risk_cfg.get('winrate_tp_factor', 0.05)
        
        # Calcul des nouvelles valeurs brutes
        new_sl = max(
            risk_cfg.get('min_sl_pct', 0.01),
            min(
                risk_cfg.get('max_sl_pct', 0.1),
                base_sl + drawdown_factor + vol_factor
            )
        )
        
        new_tp = max(
            risk_cfg.get('min_tp_pct', 0.01),
            min(
                risk_cfg.get('max_tp_pct', 0.2),
                base_tp + winrate_factor - (vol_factor * 0.5)  # TP moins sensible à la volatilité
            )
        )
        
        # Application du lissage exponentiel
        self.smoothed_params['sl_pct'] = (
            self.smoothing_factor * self.smoothed_params['sl_pct'] + 
            (1 - self.smoothing_factor) * new_sl
        )
        
        self.smoothed_params['tp_pct'] = (
            self.smoothing_factor * self.smoothed_params['tp_pct'] + 
            (1 - self.smoothing_factor) * new_tp
        )
        
        # Assignation des valeurs lissées
        mod['sl_pct'] = self.smoothed_params['sl_pct']
        mod['tp_pct'] = self.smoothed_params['tp_pct']
        
        # Journalisation des changements importants
        if state['current_step'] % 100 == 0:
            logger.info(
                f"🔧 Paramètres de risque - "
                f"Nouveau SL: {new_sl:.2%} (lissé: {mod['sl_pct']:.2%}), "
                f"Nouveau TP: {new_tp:.2%} (lissé: {mod['tp_pct']:.2%})"
            )
    
    def _compute_reward_modulation(self, mod: Dict[str, Any]) -> None:
        """Calcule la modulation des récompenses."""
        # Paramètres configurables
        reward_config = self.config.get('reward', {})
        winrate_threshold = reward_config.get('winrate_threshold', 0.55)
        max_boost = reward_config.get('max_boost', 2.0)
        
        # Reward boost basé sur le winrate
        if self.state.get('winrate', 0.0) > winrate_threshold:
            boost_factor = min(
                max_boost,
                1.0 + (self.state['winrate'] - winrate_threshold) * 5.0
            )
            mod['reward_boost'] = boost_factor
        else:
            mod['reward_boost'] = 1.0

        # Pénalité d'inaction progressive
        inaction_factor = reward_config.get('inaction_factor', 0.1)
        action_freq = self.state.get('action_frequency', 1.0) # Default to 1 to avoid penalty if not present
        min_action_freq = reward_config.get('min_action_frequency', 0.1)
        
        if action_freq < min_action_freq and self.state.get('market_regime') in ['BULL', 'BEAR']:
            # Pénalité progressive basée sur la fréquence d'action
            mod['penalty_inaction'] = -inaction_factor * (min_action_freq - action_freq) * 10
        else:
            mod['penalty_inaction'] = 0.0

    def _compute_position_sizing(self, mod: Dict[str, Any]) -> None:
        """
        Calcule la taille de position dynamique.
        
        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        sizing_cfg = self.config.get('position_sizing', {})
        base_size = sizing_cfg.get('base_position_size', 0.1)  # 10% par défaut
        
        # Ajustement basé sur la confiance (winrate récent)
        confidence_factor = min(2.0, max(0.5, self.state['winrate'] / 0.5))  # 0.5-2.0x
        
        # Ajustement basé sur le drawdown
        drawdown_factor = 1.0 - (self.state['drawdown'] / 100.0 * 2)  # Réduit la taille avec le drawdown
        
        # Ajustement basé sur la volatilité
        vol_factor = 1.0 / (1.0 + self.state['volatility'] * 10)  # Réduit la taille avec la volatilité
        
        # Calcul final avec limites
        mod['position_size_pct'] = max(
            sizing_cfg.get('min_position_size', 0.01),
            min(
                sizing_cfg.get('max_position_size', 0.3),
                base_size * confidence_factor * drawdown_factor * vol_factor
            )
        )

    def _compute_risk_mode(self, mod: Dict[str, Any]) -> None:
        """
        Détermine le mode de risque global (DEFENSIVE, NORMAL, AGGRESSIVE).
        
        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        # Mode défensif si drawdown élevé ou pertes consécutives
        if self.state['drawdown'] > 10.0 or self.state['consecutive_losses'] >= 3:
            mod['risk_mode'] = "DEFENSIVE"
            mod['position_size_pct'] *= 0.5 # Reduce position size
            mod['sl_pct'] *= 1.2 # Tighten stop loss
        # Mode agressif si bonnes performances et faible drawdown
        elif self.state['winrate'] > 0.7 and self.state['drawdown'] < 2.0:
            mod['risk_mode'] = "AGGRESSIVE"
            mod['position_size_pct'] *= 1.2 # Increase position size
            mod['tp_pct'] *= 1.2 # Loosen take profit
        else:
            mod['risk_mode'] = "NORMAL"

    def _apply_market_regime_modifiers(self, mod: Dict[str, Any]) -> None:
        """
        Applique des ajustements spécifiques au régime de marché.
        
        Args:
            mod: Dictionnaire des paramètres modulés à mettre à jour
        """
        regime = self.state.get('market_regime', 'NORMAL')
        regime_cfg = self.config.get('modes', {}).get(regime.lower(), {})
        
        if not regime_cfg:
            return
            
        mod['position_size_pct'] *= regime_cfg.get('position_size_multiplier', 1.0)
        mod['sl_pct'] *= regime_cfg.get('sl_multiplier', 1.0)
        mod['tp_pct'] *= regime_cfg.get('tp_multiplier', 1.0)
        
        # Specific adjustments for trending markets
        if regime == "BULL" or regime == "BEAR":
            mod['trailing_stop'] = True # Activate trailing stop in trending markets
    
    def reset_chunk(self) -> None:
        """Réinitialise les métriques au début d'un nouveau chunk."""
        # Conserver certaines métriques (comme le winrate) mais réinitialiser les autres
        self.state.update({
            'current_step': 0,
            'chunk_optimal_pnl': 0.0,
            'position_size_pct': self.config.get('position_sizing', {}).get('base_position_size', 0.1)
        })
        logger.info("🔄 DBE: Nouveau chunk - réinitialisation des métriques")

    def _log_dbe_state(self, modulation: Dict[str, Any]) -> None:
        """
        Logs the current state and modulation of the DBE to a JSONL file.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.state.get('current_step', 0),
            "drawdown": self.state.get('drawdown', 0.0),
            "winrate": self.state.get('winrate', 0.0),
            "volatility": self.state.get('volatility', 0.0),
            "market_regime": self.state.get('market_regime', 'NORMAL'),
            "sl_pct": modulation.get('sl_pct', 0.0),
            "tp_pct": modulation.get('tp_pct', 0.0),
            "reward_boost": modulation.get('reward_boost', 0.0),
            "penalty_inaction": modulation.get('penalty_inaction', 0.0),
            "position_size_pct": modulation.get('position_size_pct', 0.0),
            "risk_mode": modulation.get('risk_mode', 'NORMAL')
        }
        try:
            self.dbe_log_file.write(json.dumps(log_entry, cls=NpEncoder) + '\n')
            self.dbe_log_file.flush() # Ensure data is written to disk immediately
        except Exception as e:
            logger.error(f"Error writing to DBE log file: {e}")
