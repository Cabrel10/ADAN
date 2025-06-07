"""
Reward calculator for the ADAN trading environment.
"""
import numpy as np
from ..common.utils import get_logger, calculate_log_return
from ..common.constants import (
    REWARD_MIN, REWARD_MAX,
    PENALTY_TIME
)

logger = get_logger()

class RewardCalculator:
    """
    Calculates rewards for the RL agent based on portfolio performance and trading actions.
    """
    
    def __init__(self, config):
        """
        Initialize the reward calculator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        env_config = config.get('environment', {})
        
        # Charger les configurations depuis le fichier environment_config.yaml
        self.penalties_config = env_config.get('penalties', {})
        self.reward_shaping_config = env_config.get('reward_shaping', {})
        
        # Pénalité de temps (petite pénalité à chaque pas pour encourager l'action)
        self.time_penalty = self.penalties_config.get('time_step', -0.001)
        
        # Multiplicateur de log-return et clipping
        self.log_return_multiplier = self.reward_shaping_config.get('log_return_multiplier', 100.0)
        self.reward_clip_min = self.reward_shaping_config.get('clip_min', -5.0)
        self.reward_clip_max = self.reward_shaping_config.get('clip_max', 5.0)
        
        # Multiplicateurs de palier par défaut
        self.default_reward_pos_mult = 1.0
        self.default_reward_neg_mult = 1.0
        
        # Configuration des paliers
        self.tiers = env_config.get('tiers', [
            {
                "threshold": 0,
                "max_positions": 1,
                "allocation_frac_per_pos": 0.20,
                "reward_pos_mult": 1.0,
                "reward_neg_mult": 1.0
            },
            {
                "threshold": 15000,
                "max_positions": 2,
                "allocation_frac_per_pos": 0.25,
                "reward_pos_mult": 1.1,
                "reward_neg_mult": 1.2
            }
        ])
        
        logger.info(f"RewardCalculator initialized with {len(self.tiers)} tiers, log_return_multiplier={self.log_return_multiplier}")
    
    def calculate_reward(self, old_portfolio_value, new_portfolio_value, penalties=0.0, tier=None):
        """
        Calculate the reward based on portfolio performance.
        
        Args:
            old_portfolio_value: Portfolio value before action.
            new_portfolio_value: Portfolio value after action.
            penalties: Additional penalties to apply.
            tier: Current tier (optional).
            
        Returns:
            float: Calculated reward.
        """
        # Plafonner les valeurs de portefeuille pour éviter les overflows
        MAX_PORTFOLIO_VALUE = 1e6  # 1 million USD maximum
        capped_old_value = min(old_portfolio_value, MAX_PORTFOLIO_VALUE)
        capped_new_value = min(new_portfolio_value, MAX_PORTFOLIO_VALUE)
        
        # Vérifier que les valeurs sont positives
        capped_old_value = max(capped_old_value, 1e-6)  # Éviter la division par zéro
        capped_new_value = max(capped_new_value, 1e-6)
        
        # Calculate log return avec les valeurs plafonnées
        log_return = calculate_log_return(capped_old_value, capped_new_value)
        
        # Réduire le multiplicateur de log-return pour plus de stabilité
        # Passer de 100.0 à 10.0 comme suggéré dans l'analyse
        adjusted_multiplier = 10.0  # Valeur plus stable
        log_return = log_return * adjusted_multiplier
        
        # Appliquer les multiplicateurs de palier si disponibles
        if tier is not None:
            # Vérifier que les multiplicateurs sont dans des limites raisonnables
            if log_return >= 0:
                reward_pos_mult = tier.get("reward_pos_mult", self.default_reward_pos_mult)
                # Limiter le multiplicateur à une valeur raisonnable (entre 0.5 et 2.0)
                reward_pos_mult = max(0.5, min(reward_pos_mult, 2.0))
                shaped_return = log_return * reward_pos_mult
            else:
                reward_neg_mult = tier.get("reward_neg_mult", self.default_reward_neg_mult)
                # Limiter le multiplicateur à une valeur raisonnable (entre 0.5 et 2.0)
                reward_neg_mult = max(0.5, min(reward_neg_mult, 2.0))
                shaped_return = log_return * reward_neg_mult
        else:
            # Utiliser les multiplicateurs par défaut
            if log_return >= 0:
                shaped_return = log_return * self.default_reward_pos_mult
            else:
                shaped_return = log_return * self.default_reward_neg_mult
        
        # Limiter les pénalités pour éviter les valeurs extrêmes
        capped_penalties = np.clip(penalties, -2.0, 2.0)
        
        # Appliquer les pénalités et la pénalité de temps
        reward = shaped_return - capped_penalties - self.time_penalty
        
        # Clip reward selon les limites configurables
        # Utiliser des limites plus strictes (-2.0, 2.0) au lieu de (-5.0, 5.0)
        reward = np.clip(reward, -2.0, 2.0)
        
        # Vérifier que la récompense est une valeur finie
        if not np.isfinite(reward):
            logger.warning(f"Récompense non finie détectée: {reward}. Utilisation de 0.0 comme valeur par défaut.")
            reward = 0.0
        
        return reward
    
    def get_current_tier(self, capital):
        """
        Get the current tier based on capital.
        
        Args:
            capital: Current capital.
            
        Returns:
            dict: Current tier configuration.
        """
        # Plafonner le capital à une valeur raisonnable pour éviter les overflows
        MAX_CAPITAL = 1e6  # 1 million USD maximum pour la sélection du palier
        capped_capital = min(capital, MAX_CAPITAL)
        
        # Log the capital and tiers structure for debugging
        logger.debug(f"Capital: ${capital:.2f}, capital plafonné: ${capped_capital:.2f}")
        logger.debug(f"Tiers structure: {self.tiers}")
        
        # Définir le palier par défaut (premier palier)
        current_tier = self.tiers[0]
        logger.debug(f"Initial tier: {current_tier}")
        
        # Parcourir les paliers pour trouver celui qui correspond au capital plafonné
        for tier in self.tiers:
            if capped_capital >= tier["threshold"]:
                current_tier = tier
                logger.info(f"Selected tier with threshold ${tier['threshold']:.2f}")
            else:
                logger.info(f"Skipping tier with threshold ${tier['threshold']:.2f} (capital < threshold)")
                break
        
        # Vérifier que le palier contient toutes les clés nécessaires
        if 'allocation_frac_per_pos' not in current_tier:
            logger.warning(f"Palier sans allocation_frac_per_pos: {current_tier}")
            current_tier['allocation_frac_per_pos'] = 0.95  # Valeur par défaut
        
        # Limiter l'allocation à une valeur raisonnable (entre 0.05 et 0.95)
        current_tier['allocation_frac_per_pos'] = max(0.05, min(current_tier['allocation_frac_per_pos'], 0.95))
        
        logger.info(f"Final tier: {current_tier}")
        logger.info(f"Allocation fraction: {current_tier.get('allocation_frac_per_pos', -1.0):.4f}")
        
        return current_tier
    
    def calculate_pnl_bonus(self, pnl):
        """
        Calculate bonus reward for positive PnL.
        
        Args:
            pnl: Profit and Loss amount.
            
        Returns:
            float: Bonus reward.
        """
        if pnl <= 0:
            return 0.0
        
        # Bonus of 1% of PnL, capped at 1.0
        return min(pnl * 0.01, 1.0)
