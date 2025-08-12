#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Portfolio management module for the ADAN trading bot.

This module is responsible for tracking the agent's financial status, including
capital, positions, and performance metrics.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


logger = logging.getLogger(__name__)


class Position:
    """Represents a single, simple trading position (long or short)."""

    def __init__(self):
        self.is_open = False
        self.entry_price = 0.0
        self.size = 0.0  # Number of units
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0

    def open(
        self,
        entry_price: float,
        size: float,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> None:
        self.is_open = True
        self.entry_price = entry_price
        self.size = size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def close(self):
        self.is_open = False
        self.entry_price = 0.0
        self.size = 0.0
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0

    def get_status(self) -> str:
        """Get the status of the position.

        Returns:
            str: A string describing the position status.
        """
        if self.is_open:
            return f"Open ({self.size} units @ {self.entry_price:.2f})"
        return "Closed"


class PortfolioManager:
    """Manages the trading portfolio for a single asset.

    Handles capital allocation, tracks PnL, and enforces risk rules defined
    in the environment configuration. Also tracks performance per data chunk
    for reward shaping and learning purposes.
    """

    def __init__(self, env_config: Dict[str, Any], assets: List[str]) -> None:
        """
        Initialize the PortfolioManager with environment configuration and assets.

        Args:
            env_config: Dictionary containing environment configuration
            assets: List of asset symbols to be managed in the portfolio
        """
        self.config = env_config

        # Initialize configuration sections
        portfolio_config = self.config.get("portfolio", {})
        environment_config = self.config.get("environment", {})
        trading_rules_config = self.config.get("trading_rules", {})

        # Initialize equity and capital
        self.initial_equity = portfolio_config.get(
            "initial_balance", environment_config.get("initial_balance", 20.0)
        )
        self.initial_capital = self.initial_equity  # For backward compatibility
        self.current_equity = self.initial_equity

        # Initialize portfolio state (will be fully set in reset())
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.total_capital = 0.0

        # Initialize position tracking
        self.positions: Dict[str, Position] = {asset: Position() for asset in assets}
        self.trade_history: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []  # Initialize trade log

        # Initialize chunk-based tracking
        self.chunk_pnl: Dict[int, Dict[str, float]] = {}
        self.current_chunk_id = 0
        self.chunk_start_equity = self.initial_equity

        # Trading rules configuration
        self.futures_enabled = trading_rules_config.get("futures_enabled", False)
        self.leverage = trading_rules_config.get("leverage", 1)

        # Commission and fees
        if self.futures_enabled:
            self.commission_pct = trading_rules_config.get(
                "futures_commission_pct", 0.0
            )
        else:
            self.commission_pct = trading_rules_config.get("commission_pct", 0.0)

        # Position sizing rules
        self.min_trade_size = trading_rules_config.get("min_trade_size", 0.0001)
        self.min_notional_value = trading_rules_config.get("min_notional_value", 10.0)
        self.max_notional_value = trading_rules_config.get(
            "max_notional_value", 100000.0
        )

        # Configuration de la gestion des risques
        risk_management = self.config.get("risk_management", {})

        # 1. Charger les capital_tiers depuis la configuration
        self.capital_tiers = self.config.get("capital_tiers", [])

        # Si vide, essayer de charger depuis risk_management
        if not self.capital_tiers and "capital_tiers" in risk_management:
            self.capital_tiers = risk_management["capital_tiers"]
            logger.info("Chargement des capital_tiers depuis risk_management")

        # 2. Valider le format et le contenu des tiers
        if not isinstance(self.capital_tiers, list):
            logger.error(
                "capital_tiers doit être une liste, mais a reçu: %s",
                type(self.capital_tiers),
            )
            self.capital_tiers = []

        # 3. Définir les clés requises pour chaque tier
        REQUIRED_KEYS = {
            "name",
            "min_capital",
            "max_position_size_pct",
            "risk_per_trade_pct",
            "max_drawdown_pct",
            "leverage",
        }

        # 4. Valider chaque tier
        valid_tiers = []
        for i, tier in enumerate(self.capital_tiers):
            if not isinstance(tier, dict):
                logger.warning(
                    "Le tier %d n'est pas un dictionnaire et sera ignoré: %s", i, tier
                )
                continue

            # Vérifier les clés requises
            missing_keys = REQUIRED_KEYS - tier.keys()
            if missing_keys:
                logger.warning(
                    "Le tier %d manque des clés requises %s et sera ignoré: %s",
                    i,
                    missing_keys,
                    tier,
                )
                continue

            valid_tiers.append(tier)

        # 5. Trier les tiers par min_capital croissant
        valid_tiers.sort(key=lambda x: x["min_capital"])

        # 6. Vérifier la continuité des paliers
        for i in range(1, len(valid_tiers)):
            if valid_tiers[i - 1]["min_capital"] >= valid_tiers[i]["min_capital"]:
                logger.error(
                    "Les paliers de capital doivent être en ordre croissant. "
                    "Palier %d (%s) a un min_capital >= au palier %d (%s)",
                    i - 1,
                    valid_tiers[i - 1]["name"],
                    i,
                    valid_tiers[i]["name"],
                )
                valid_tiers = []
                break

        # 7. Mettre à jour la liste des tiers valides
        self.capital_tiers = valid_tiers

        if not self.capital_tiers:
            logger.error(
                "Aucun palier de capital valide n'a été trouvé. "
                "Veuillez vérifier votre configuration."
            )
        else:
            logger.info(
                "%d paliers de capital chargés avec succès: %s",
                len(self.capital_tiers),
                ", ".join(
                    [f"{t['name']} ({t['min_capital']}+)" for t in self.capital_tiers]
                ),
            )

        # Position sizing configuration
        self.position_sizing_config = risk_management.get("position_sizing", {})
        self.concentration_limits = self.position_sizing_config.get(
            "concentration_limits", {}
        )

        # Trading protection flag (spot mode): when True, no new long trades are allowed
        self.trading_disabled: bool = False

        self.reset()

    def get_margin_level(self) -> float:
        """
        Returns the current margin level (margin used / available capital).

        Returns:
            float: The margin level as a ratio of used margin to initial capital.
        """
        if not self.futures_enabled:
            return 1.0  # Not applicable for spot trading

        # Calculate total margin used
        total_margin_used = 0.0
        for position in self.positions.values():
            if position.is_open:
                margin = (position.size * position.entry_price) / self.leverage
                total_margin_used += margin

        # Margin level is the ratio of margin used to initial capital
        if self.initial_capital > 0:
            return total_margin_used / self.initial_capital

        # Handle error state when initial capital is 0 or negative
        return 0.0

    def get_current_tier(self) -> Dict[str, Any]:
        """Détermine le tier de capital actuel en fonction de la valeur du portefeuille.

        Returns:
            dict: Configuration du tier actuel avec les clés :
                - name: Nom du palier
                - min_capital: Capital minimum du palier
                - max_capital: Capital max du palier (None si dernier palier)
                - max_position_size_pct: Taille max de position en %
                - leverage: Effet de levier autorisé
                - risk_per_trade_pct: Risque max par trade en %
                - max_drawdown_pct: Drawdown maximum autorisé en %

        Raises:
            RuntimeError: Si aucun tier de capital n'est défini ou si la
                configuration est invalide
        """
        if not self.capital_tiers:
            raise RuntimeError(
                "Configuration des capital_tiers invalide ou vide. "
                "Vérifiez la configuration."
            )

        # Trier les tiers par min_capital croissant (au cas où)
        sorted_tiers = sorted(self.capital_tiers, key=lambda x: x["min_capital"])

        # Trouver le premier tier où min_capital <= current_equity <
        # next_tier.min_capital
        current_equity = self.get_portfolio_value()

        for i, tier in enumerate(sorted_tiers):
            # Si c'est le dernier tier, on l'utilise
            if i == len(sorted_tiers) - 1:
                logger.debug(
                    "Palier actuel: %s (capital: %.2f >= %.2f)",
                    tier["name"],
                    current_equity,
                    tier["min_capital"],
                )
                return tier

            # Sinon, vérifier si on est dans l'intervalle [min_capital,
            # next_tier.min_capital)
            next_tier = sorted_tiers[i + 1]
            if tier["min_capital"] <= current_equity < next_tier["min_capital"]:
                logger.debug(
                    "Palier actuel: %s (%.2f <= capital: %.2f < %.2f)",
                    tier["name"],
                    tier["min_capital"],
                    current_equity,
                    next_tier["min_capital"],
                )
                return tier

        # Si on arrive ici, on utilise le dernier tier (ne devrait normalement
        # pas arriver)
        logger.warning(
            f"Aucun tier trouvé pour la valeur de portefeuille "
            f"{current_equity:.2f}. Utilisation du dernier tier disponible."
        )
        return sorted_tiers[-1]

    def calculate_position_size(
        self,
        price: float,
        stop_loss_pct: float = 0.02,
        risk_per_trade: float = 0.01,
        account_risk_multiplier: float = 1.0,
    ) -> float:
        """
        Calcule la taille de position en fonction du risque, du stop loss et des limites de position.

        La taille de la position est déterminée par la plus petite des deux valeurs suivantes :
        1. Taille basée sur le risque (risque_max / (prix * stop_loss_pct))
        2. Taille maximale autorisée par le palier (max_position_size_pct)

        Args:
            price: Prix actif
            stop_loss_pct: Pourcentage de stop loss (ex: 0.02 pour 2%)
            risk_per_trade: Fraction du capital à risquer (0.01 pour 1%)
            account_risk_multiplier: Multiplicateur de risque (défini par le DBE)

        Returns:
            float: Taille de position en unités de l'actif
        """
        if price <= 0 or stop_loss_pct <= 0:
            return 0.0

        tier = self.get_active_tier()

        # 1. Calcul du montant à risquer (% du capital défini dans le palier)
        risk_amount = self.portfolio_value * (tier["risk_per_trade_pct"] / 100.0)

        # 2. Application du multiplicateur de risque du DBE
        risk_amount *= account_risk_multiplier

        # 3. Calcul de la taille de position basée sur le risque
        risk_based_size = risk_amount / (price * stop_loss_pct)

        # 4. Calcul de la taille maximale autorisée par le palier
        max_position_value = self.portfolio_value * (
            tier["max_position_size_pct"] / 100.0
        )
        max_position_size = max_position_value / price

        # 5. Prendre la plus petite des deux tailles (risque ou taille max)
        position_size = min(risk_based_size, max_position_size)

        # 6. Vérifier la taille minimale de trade
        min_trade_size = self.min_trade_size
        if position_size > 0 and position_size < min_trade_size:
            position_size = min_trade_size
            logger.warning(
                f"Taille de position {position_size} inférieure au minimum autorisé "
                f"({min_trade_size}). Ajustement à {min_trade_size}"
            )

        # 7. Journalisation détaillée
        logger.info(
            f"[POSITION_SIZE] Risk: {tier['risk_per_trade_pct']}% of {self.portfolio_value:.2f} = {risk_amount:.4f} | "
            f"SL: {stop_loss_pct*100:.2f}% | RiskBasedSize: {risk_based_size:.8f} | "
            f"MaxSize: {max_position_size:.8f} | FinalSize: {position_size:.8f} ({position_size * price:.2f} USDT)"
        )

        return position_size

    def _calculate_volatility(self, window: int = 20) -> float:
        """Calcule la volatilité des rendements sur une fenêtre glissante.

        Args:
            window: Taille de la fenêtre de calcul (minimum 2, maximum 252)

        Returns:
            float: Volatilité annualisée des rendements sur la fenêtre, ou 0.0 si non calculable
        """
        # Validation des entrées
        window = max(2, min(window, 252))  # Borne la fenêtre entre 2 et 252

        # Vérification des données disponibles
        if not self.trade_history or len(self.trade_history) < 2:
            return 0.0

        try:
            # Conversion en array numpy et vérification des valeurs
            values = np.array(self.trade_history[-window:], dtype=np.float64)

            # Suppression des valeurs non finies (NaN, Inf)
            values = values[np.isfinite(values)]

            # Vérification après nettoyage
            if len(values) < 2 or np.any(values <= 0):
                return 0.0

            # Calcul des rendements logarithmiques avec protection contre les valeurs non positives
            returns = np.diff(np.log(values))

            # Vérification des rendements calculés
            if len(returns) < 1 or not np.all(np.isfinite(returns)):
                return 0.0

            # Calcul de la volatilité annualisée (252 jours de bourse par an)
            volatility = np.std(returns, ddof=1)  # ddof=1 pour l'estimation non biaisée

            # Protection contre les valeurs aberrantes
            if not np.isfinite(volatility) or volatility <= 0:
                return 0.0

            return volatility * np.sqrt(252)  # Annualisation

        except (ValueError, RuntimeWarning, ZeroDivisionError) as e:
            logger.warning(f"Erreur dans le calcul de la volatilité: {str(e)}")
            return 0.0

    def get_state(self) -> np.ndarray:
        """Return the current portfolio state as a numpy array with 17 dimensions.

        The state includes:
            0. Cash ratio (cash / portfolio value)
            1. Equity ratio (current equity / initial equity)
            2. Current margin level
            3. Ratio of open positions
            4. Realized PnL
            5. Unrealized PnL
            6. Max drawdown
            7. Sharpe ratio
            8. Sortino ratio (placeholder)
            9. Portfolio volatility
            10. Total fees paid
            11. Total commissions paid
            12. Number of trades
            13. Win rate
            14. Average gain per winning trade
            15. Average loss per losing trade
            16. Current drawdown

        Returns:
            np.ndarray: Array of shape (17,) containing portfolio state info
        """
        # Calculate basic metrics
        cash_ratio = 0.0
        if self.portfolio_value > 0:
            cash_ratio = self.cash / self.portfolio_value

        equity_ratio = 0.0
        if self.initial_equity > 0:
            equity_ratio = self.current_equity / self.initial_equity

        margin_level = self.get_margin_level()

        # Calculate position metrics
        open_positions = sum(1 for p in self.positions.values() if p.is_open)
        open_positions_ratio = open_positions / max(1, len(self.positions))

        # Calculate trade metrics
        def trade_filter(trade):
            return trade.get("type") == "close"

        def pnl_positive(trade):
            return trade.get("trade_pnl", 0) > 0

        def pnl_negative(trade):
            return trade.get("trade_pnl", 0) <= 0

        closed_trades = list(filter(trade_filter, self.trade_log))
        total_trades = len(closed_trades)
        winning_trades = list(filter(pnl_positive, closed_trades))
        losing_trades = list(filter(pnl_negative, closed_trades))

        win_rate = len(winning_trades) / max(1, total_trades)
        avg_win = (
            np.mean([t.get("trade_pnl", 0) for t in winning_trades])
            if winning_trades
            else 0.0
        )
        avg_loss = (
            np.mean([abs(t.get("trade_pnl", 0)) for t in losing_trades])
            if losing_trades
            else 0.0
        )

        # Calculate volatility
        volatility = self._calculate_volatility()

        # Create state vector with 17 dimensions
        state = np.array(
            [
                # 0-3: Basic metrics
                cash_ratio,  # 0: Cash ratio
                equity_ratio,  # 1: Equity ratio
                margin_level,  # 2: Margin level
                open_positions_ratio,  # 3: Open positions ratio
                # 4-7: PnL and risk metrics
                self.realized_pnl,  # 4: Realized PnL
                self.unrealized_pnl,  # 5: Unrealized PnL
                self.drawdown,  # 6: Max drawdown
                self.sharpe_ratio,  # 7: Sharpe ratio
                # 8-11: Advanced metrics
                0.0,  # 8: Sortino ratio (placeholder)
                volatility,  # 9: Portfolio volatility
                0.0,  # 10: Total fees paid (placeholder)
                0.0,  # 11: Commissions paid (placeholder)
                # 12-16: Trade statistics
                total_trades,  # 12: Number of trades
                win_rate,  # 13: Win rate
                avg_win,  # 14: Avg gain per winning trade
                avg_loss,  # 15: Avg loss per losing trade
                self.drawdown,  # 16: Current drawdown
            ],
            dtype=np.float32,
        )

        if len(state) != 17:
            err_msg = f"State vector must have 17 dimensions, got {len(state)}"
            raise ValueError(err_msg)

        return state

    def reset(self, new_epoch: bool = False) -> None:
        """
        Reset the portfolio to its initial state.

        Args:
            new_epoch: If True, resets to initial equity. If False, keeps current portfolio value.
        """
        # Get configuration sections
        portfolio_config = self.config.get("portfolio", {})
        environment_config = self.config.get("environment", {})

        # Close all open positions
        for asset in list(self.positions.keys()):
            if self.positions[asset].is_open:
                logger.debug("Closing position for %s during reset", asset)
                self.close_position(asset, self.positions[asset].entry_price)

        if new_epoch:
            # Full reset to initial equity - use the same logic as in __init__
            self.initial_equity = portfolio_config.get(
                "initial_balance", environment_config.get("initial_balance", 20.0)
            )
            self.initial_capital = self.initial_equity  # For backward compatibility
            self.cash = self.initial_equity
            self.portfolio_value = self.initial_equity
            self.current_equity = self.initial_equity
            self.peak_equity = self.initial_equity

            # Reset trade history and metrics
            self.trade_history = [self.initial_equity]
            self.trade_log = []

            logger.info(
                "New epoch - Portfolio reset to initial equity: %.2f",
                self.initial_equity,
            )
        else:
            # Keep current portfolio value but reset positions and metrics
            current_value = self.get_portfolio_value()
            self.cash = current_value
            self.portfolio_value = current_value
            self.current_equity = current_value
            self.peak_equity = current_value

            logger.debug(
                "Chunk reset - Portfolio value maintained at: %.2f", current_value
            )

        # Always reset these metrics
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.var = 0.0
        self.cvar = 0.0
        # Re-enable trading after a reset unless overridden by caller
        self.trading_disabled = False

    def get_current_tier_index(self) -> int:
        """Retourne l'index du palier actuel dans self.capital_tiers (0-based)."""
        equity = self.get_portfolio_value()
        sorted_tiers = sorted(self.capital_tiers, key=lambda x: x["min_capital"])

        for i, tier in enumerate(sorted_tiers):
            # Si c'est le dernier palier
            if i == len(sorted_tiers) - 1:
                if equity >= tier["min_capital"]:
                    return i
            else:
                next_min = sorted_tiers[i + 1]["min_capital"]
                if tier["min_capital"] <= equity < next_min:
                    return i
        return 0

    def get_current_tier_name(self) -> str:
        """Retourne le nom du palier actuel."""
        idx = self.get_current_tier_index()
        return sorted(self.capital_tiers, key=lambda x: x["min_capital"])[idx]["name"]

    def get_portfolio_value(self) -> float:
        """
        Returns the current total portfolio value (cash + open positions) with numerical safety.

        Returns:
            float: Total portfolio value in quote currency, never negative
        """
        try:
            # Calculate total value of open positions with protection
            positions_value = 0.0
            for pos in self.positions.values():
                if pos.is_open and hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                    positions_value = positions_value + float(pos.size) * float(pos.entry_price)

            # Update and return the total portfolio value with protection
            total_value = float(self.cash) + positions_value
            self.portfolio_value = max(0.0, total_value)  # Never go below zero
            return self.portfolio_value
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return max(0.0, float(getattr(self, 'cash', 0.0)))  # Fallback to cash if available

    def get_leverage(self) -> float:
        """
        Calculate the current leverage of the portfolio.

        Returns:
            float: Current leverage ratio (1.0 = no leverage, 2.0 = 2x, etc.)
        """
        if self.portfolio_value <= 0:
            return 1.0  # Avoid division by zero

        # Calculate total position value (sum of all open positions)
        total_position_value = 0.0
        for position in self.positions.values():
            if position.is_open:
                total_position_value += abs(position.size * position.entry_price)

        # Leverage = Total position value / Portfolio equity
        leverage = total_position_value / self.portfolio_value

        # Apply configured leverage cap if futures are enabled
        if self.futures_enabled:
            leverage = min(leverage, self.leverage)

        return max(1.0, leverage)  # Minimum leverage is 1.0

    def calculate_drawdown(self) -> float:
        """
        Calculate the current drawdown of the portfolio.

        Returns:
            float: Current drawdown as a percentage (0.0 to 1.0)
        """
        if not hasattr(self, "peak_equity") or self.peak_equity == 0:
            return 0.0

        current_equity = self.get_portfolio_value()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return 0.0

        return (self.peak_equity - current_equity) / self.peak_equity

    def get_available_capital(self) -> float:
        """
        Returns the available capital for new trades based on total portfolio value.

        For spot trading, available capital is calculated as the total portfolio value
        minus the value of all open positions. For margin trading, it considers the
        margin used and available leverage.

        Returns:
            float: The amount of capital available for new trades.
        """
        # Get current portfolio value (cash + open positions)
        portfolio_value = self.get_portfolio_value()

        # Calculate total value of open positions (absolute value to handle both long and short)
        open_positions_value = sum(
            abs(pos.size) * pos.entry_price
            for pos in self.positions.values()
            if pos.is_open
        )

        # Calculate available capital based on leverage
        if self.leverage > 1.0 and open_positions_value > 0:
            # For margin trading, available capital is reduced by margin used
            margin_used = open_positions_value / self.leverage
            available_capital = max(0, portfolio_value - margin_used)

            # Log detailed information for margin trading
            logger.debug(
                "[CAPITAL] Available: %.2f (Portfolio: %.2f, Margin Used: %.2f, "
                "Positions: %.2f, Leverage: %.1fx)",
                available_capital,
                portfolio_value,
                margin_used,
                open_positions_value,
                self.leverage,
            )
        else:
            # For spot trading, available capital is total value minus open positions
            available_capital = max(0, portfolio_value - open_positions_value)

            # Log detailed information for spot trading
            logger.debug(
                "[CAPITAL] Available: %.2f (Portfolio: %.2f, Positions: %.2f)",
                available_capital,
                portfolio_value,
                open_positions_value,
            )

        return available_capital

    def update_market_price(self, current_prices: Dict[str, float]) -> None:
        """
        Met à jour la valeur du portefeuille en fonction des prix actuels avec gestion robuste des erreurs.

        Args:
            current_prices: Dictionnaire associant les symboles d'actifs à leurs prix actuels.
        """
        if not isinstance(current_prices, dict):
            logger.warning("current_prices doit être un dictionnaire. Reçu: %s", type(current_prices))
            current_prices = {}

        try:
            previous_portfolio_value = float(self.portfolio_value) if hasattr(self, 'portfolio_value') else 0.0
            self.unrealized_pnl = 0.0
            total_positions_value = 0.0
            assets_with_missing_prices = []

            for asset, position in self.positions.items():
                if not position.is_open:
                    continue

                current_price = current_prices.get(asset)

                # Skip invalid prices
                if not isinstance(current_price, (int, float)) or current_price <= 0:
                    assets_with_missing_prices.append(asset)
                    continue

                try:
                    current_price = float(current_price)
                    entry_value = float(position.size) * float(position.entry_price)
                    current_value = float(position.size) * current_price
                    position_pnl = current_value - entry_value

                    # Update position metrics
                    self.unrealized_pnl += position_pnl
                    total_positions_value += current_value

                    # Update position attributes
                    position.current_price = current_price
                    position.current_value = current_value
                    position.unrealized_pnl = position_pnl

                    # Safe percentage calculation
                    if entry_value > 0:
                        position.pnl_pct = (position_pnl / entry_value) * 100.0
                    else:
                        position.pnl_pct = 0.0

                    # Log position details
                    logger.debug(
                        "[POSITION] %s: %.8f @ %.8f (Val: %.2f USDT, PnL: %.2f USDT, %.2f%%)",
                        asset,
                        position.size,
                        current_price,
                        current_value,
                        position_pnl,
                        position.pnl_pct,
                    )

                except (TypeError, ValueError) as e:
                    logger.error(f"Error updating position {asset}: {e}")
                    continue

            # Log assets with missing prices
            if assets_with_missing_prices:
                logger.warning(
                    "Prix manquants ou invalides pour %d actifs: %s. "
                    "Ces positions seront ignorées.",
                    len(assets_with_missing_prices),
                    ", ".join(assets_with_missing_prices),
                )

            # Update portfolio metrics
            self.portfolio_value = max(0.0, float(self.cash) + total_positions_value)
            self.current_equity = self.portfolio_value
            self.total_capital = self.portfolio_value

            # Calculate total PnL with protection
            try:
                total_pnl = self.portfolio_value - float(getattr(self, 'initial_equity', 0.0))
                pnl_pct = (total_pnl / float(self.initial_equity) * 100) if float(self.initial_equity) > 0 else 0.0
                logger.debug(
                    "[PORTFOLIO] Valeur totale: %.2f USDT (Cash: %.2f, Positions: %.2f, PnL: %.2f USDT, %.2f%%)",
                    self.portfolio_value,
                    float(self.cash),
                    total_positions_value,
                    total_pnl,
                    pnl_pct,
                )
            except (TypeError, ValueError) as e:
                logger.error(f"Error calculating portfolio PnL: {e}")

            # Update trade history (limit to 1000 entries)
            if not hasattr(self, "trade_history"):
                self.trade_history = [max(0.0, float(getattr(self, 'initial_equity', 0.0)))]

            self.trade_history.append(self.portfolio_value)
            if len(self.trade_history) > 1000:
                self.trade_history.pop(0)

            # Apply funding rates if futures are enabled
            if getattr(self, 'futures_enabled', False):
                # Funding rate logic would go here
                pass

            # Update metrics
            self.update_metrics()

            # Check protection limits with valid prices
            valid_prices = {
                k: float(v) for k, v in current_prices.items()
                if isinstance(v, (int, float)) and v > 0
            }

            if valid_prices:
                protection_triggered = self.check_protection_limits(valid_prices)
                if protection_triggered:
                    # Reset metrics after protection action
                    self.unrealized_pnl = 0.0
                    self.portfolio_value = max(0.0, float(self.cash))
                    self.current_equity = self.portfolio_value
                    self.total_capital = self.portfolio_value
                    logger.warning("Protection limits triggered - reset portfolio metrics")

        except Exception as e:
            logger.critical(
                "[CRITICAL] Erreur lors de la mise à jour des prix du marché: %s",
                str(e),
                exc_info=True
            )
            # Ensure portfolio value is always valid
            self.portfolio_value = max(0.0, float(getattr(self, 'cash', 0.0)))
            self.current_equity = self.portfolio_value
            self.total_capital = self.portfolio_value

            # Calculate drawdown for logging
            try:
                current_drawdown = float(getattr(self, 'initial_equity', 0.0)) - self.portfolio_value
                initial_equity = float(getattr(self, 'initial_equity', 1.0))
                drawdown_pct = (current_drawdown / initial_equity * 100) if initial_equity > 0 else 0.0
                logger.warning(
                    "Protection triggered - Portfolio: %.2f USDT, Drawdown: %.2f USDT (%.2f%%)",
                    self.portfolio_value,
                    current_drawdown,
                    drawdown_pct
                )
            except (TypeError, ValueError) as e:
                logger.error("Erreur lors du calcul du drawdown: %s", str(e))

        # Vérifier les ordres de protection si des positions sont ouvertes
        if any(pos.is_open for pos in self.positions.values()) and not valid_prices:
            logger.warning(
                "Impossible de vérifier les ordres de protection: "
                "aucun prix valide disponible"
            )

    def check_protection_limits(self, current_prices: Dict[str, float]) -> bool:
        """
        Vérifie si le portefeuille dépasse les limites de protection définies.

        Dans un contexte de trading spot, cette méthode vérifie les conditions de protection
        et ferme les positions si nécessaire pour protéger le capital.

        Args:
            current_prices: Dictionnaire des prix actuels par actif.

        Returns:
            bool: True si une action de protection a été déclenchée, False sinon.
        """
        # Récupérer le palier de capital actuel
        tier = self.get_active_tier()
        max_drawdown_pct = tier.get("max_drawdown_pct", 20.0) / 100.0

        # Vérifier le drawdown maximum autorisé
        max_drawdown_value = self.initial_equity * max_drawdown_pct
        current_drawdown = self.initial_equity - self.portfolio_value
        current_drawdown_pct = (
            (current_drawdown / self.initial_equity) * 100
            if self.initial_equity > 0
            else 0
        )

        # Vérifier le solde disponible pour éviter les positions trop importantes
        # Utilisation de la valeur totale du portefeuille moins la valeur des positions ouvertes
        # pour éviter de compter deux fois la même valeur
        open_positions_value = sum(
            pos.size * current_prices.get(asset, 0)
            for asset, pos in self.positions.items()
            if pos.is_open and asset in current_prices
        )
        available_balance = max(0, self.portfolio_value - open_positions_value)
        # Laisser 1% de marge pour les frais
        max_position_size = available_balance * 0.99
        # Journalisation des informations de risque
        logger.info(
            "[RISK] Drawdown actuel: %.2f/%.2f USDT (%.1f%%/%.1f%%), "
            "Solde dispo: %.2f USDT",
            current_drawdown,
            max_drawdown_value,
            current_drawdown_pct,
            max_drawdown_pct * 100,
            available_balance,
        )

        try:
            # Vérifier si le drawdown dépasse la limite du palier
            if current_drawdown > max_drawdown_value:
                tier = self.get_active_tier()
                logger.critical(
                    "[CRITICAL] Drawdown critique: %.2f/%.2f USDT (%.1f%%/%.1f%%), "
                    "Palier: %s (%.2f USDT), Solde: %.2f USDT",
                    current_drawdown,
                    max_drawdown_value,
                    current_drawdown_pct,
                    max_drawdown_pct * 100,
                    tier.get("name", "Inconnu"),
                    self.initial_equity,
                    self.portfolio_value,
                )

                if not self.futures_enabled:
                    # Spot mode: disable new buy trades, do NOT force-close positions
                    if not self.trading_disabled:
                        logger.warning(
                            "[PROTECTION] Spot mode: Disabling new BUY trades due to drawdown breach."
                        )
                    self.trading_disabled = True
                    # Keep positions open; environment should respect this via validation
                    return True

                # Futures/margin mode: proceed with liquidation as before
                positions_closed = False
                for asset in list(self.positions.keys()):
                    current_price = current_prices.get(asset)
                    if current_price is not None:
                        logger.info(
                            "[ACTION] Fermeture de la position %s à %.8f USDT",
                            asset,
                            current_price,
                        )
                        self.close_position(asset, current_price)
                        positions_closed = True
                    else:
                        logger.error(
                            "[ERROR] Impossible de fermer la position %s: prix manquant",
                            asset,
                        )

                # Mettre à jour les métriques (futures)
                if positions_closed:
                    self.unrealized_pnl = 0.0
                    self.total_capital = self.cash
                    self.portfolio_value = self.cash
                    self.current_equity = self.cash
                    logger.info(
                        "[STATUS] Portefeuille après fermeture: %.2f USDT (Cash: %.2f USDT)",
                        self.portfolio_value,
                        self.cash,
                    )
                return True

            # Si on arrive ici, c'est que le drawdown est dans les limites
            return False

        except Exception as e:
            # En cas d'erreur, on bloque par sécurité
            logger.critical(
                "[CRITICAL] Erreur lors de la vérification des limites de protection: %s",
                str(e),
                exc_info=True,
            )
            return True

            # Aucune position n'a pu être fermée
            return False

        # Vérifier également le niveau de marge pour les comptes sur marge
        if self.futures_enabled:
            liquidation_threshold = self.config["trading_rules"].get(
                "liquidation_threshold", 0.2
            )
            margin_level = self.get_margin_level()

            if margin_level < liquidation_threshold:
                logger.warning(
                    "Niveau de marge %.1f%% en dessous du seuil de liquidation de %.1f%%. "
                    "Liquidation des positions.",
                    margin_level * 100,
                    liquidation_threshold * 100,
                )

                # Fermer toutes les positions
                for asset in list(self.positions.keys()):
                    current_price = current_prices.get(asset)
                    if current_price is not None:
                        self.close_position(asset, current_price)
                    else:
                        logger.error(
                            "Impossible de fermer la position %s lors de la liquidation: "
                            "prix actuel manquant",
                            asset,
                        )

                # Mettre à jour les métriques
                self.unrealized_pnl = 0.0
                self.total_capital = self.cash
                self.portfolio_value = self.cash
                self.update_metrics()

                logger.critical(
                    "LIQUIDATION SUR MARGE EFFECTUÉE - Niveau de marge: %.1f%%",
                    margin_level * 100,
                )

                return True

        return False

    def check_protection_orders(self, current_prices: Dict[str, float]):
        """Checks if any open positions have hit their stop-loss or take-profit levels."""
        for asset, position in self.positions.items():
            if position.is_open:
                current_price = current_prices.get(asset)
                if current_price is None:
                    continue

                # Check stop-loss
                if (
                    position.stop_loss_pct > 0
                    and current_price
                    <= position.entry_price * (1 - position.stop_loss_pct)
                ):
                    logger.info("Stop-loss hit for %s. Closing position.", asset)
                    self.close_position(asset, current_price)
                # Check take-profit
                elif (
                    position.take_profit_pct > 0
                    and current_price
                    >= position.entry_price * (1 + position.take_profit_pct)
                ):
                    logger.info("Take-profit hit for %s. Closing position.", asset)
                    self.close_position(asset, current_price)

    def open_position(self, asset: str, price: float, size: float) -> bool:
        """
        Ouvre une nouvelle position longue pour un actif spécifique.

        Cette méthode gère l'ouverture d'une position en vérifiant les fonds disponibles
        et en mettant à jour la valeur du portefeuille. Elle prend en compte la commission
        et utilise la valeur totale du portefeuille pour les vérifications.

        Args:
            asset: L'actif pour lequel ouvrir une position.
            price: Le prix auquel ouvrir la position.
            size: La taille de la position à ouvrir.

        Returns:
            bool: True si la position a été ouverte avec succès, False sinon.
        """
        # Protection: in spot mode, block new BUY orders when trading is disabled
        if not self.futures_enabled and self.trading_disabled and size > 0:
            logger.warning(
                "[PROTECTION] open_position blocked: trading disabled for BUY orders (drawdown breach). %s size=%.8f @ %.8f",
                asset,
                size,
                price,
            )
            # Standardized guard log for easy grep during integration runs
            logger.warning(
                "[GUARD] Rejecting BUY due to trading_disabled (reason=spot_drawdown) asset=%s size=%.8f price=%.8f",
                asset,
                size,
                price,
            )
            return False
        # Vérifier si une position est déjà ouverte pour cet actif
        if self.positions[asset].is_open:
            logger.warning(
                "[ERREUR] Impossible d'ouvrir une position pour %s: position déjà ouverte",
                asset,
            )
            return False

        # Récupérer la configuration du worker (par défaut w1 si non spécifié)
        worker_config = self.config.get("workers", {}).get("w1", {})
        trading_config = worker_config.get("trading_config", {})

        # Récupérer les paramètres de gestion des risques avec valeurs par défaut
        stop_loss_pct = trading_config.get("stop_loss_pct", 0.05)  # 5% par défaut
        take_profit_pct = trading_config.get("take_profit_pct", 0.15)  # 15% par défaut

        # Journalisation des paramètres de trading
        logger.debug("[OUVERTURE] %s - Configuration: %s", asset, trading_config)
        logger.debug(
            "[OUVERTURE] %s - Stop-loss: %.2f%%, Take-profit: %.2f%%",
            asset,
            stop_loss_pct * 100,
            take_profit_pct * 100,
        )

        # Calculer la valeur notionnelle et la commission
        notional_value = size * price
        commission = notional_value * self.commission_pct
        total_cost = notional_value + commission

        # Vérifier les fonds disponibles en utilisant la valeur totale du portefeuille
        available_capital = self.get_available_capital()

        # Journalisation des détails financiers
        logger.debug(
            "[OUVERTURE] %s - Détails financiers - Valeur notionnelle: %.2f, Commission: %.2f, Coût total: %.2f, Capital disponible: %.2f",
            asset,
            notional_value,
            commission,
            total_cost,
            available_capital,
        )

        # Vérifier si les fonds sont suffisants
        if total_cost > available_capital:
            logger.warning(
                "[ERREUR] Fonds insuffisants pour %s - Coût total: %.2f > Disponible: %.2f",
                asset,
                total_cost,
                available_capital,
            )
            return False

        # Appliquer les coûts selon le type de trading
        if self.futures_enabled:
            # Pour les contrats à terme : réserver la marge (valeur notionnelle/levier) plus la commission
            margin_used = notional_value / self.leverage
            self.cash -= margin_used + commission
            logger.debug(
                "[OUVERTURE] %s - Marge utilisée: %.2f, Commission: %.2f, Cash restant: %.2f",
                asset,
                margin_used,
                commission,
                self.cash,
            )
        else:
            # Pour le spot : débiter le montant total (valeur notionnelle + commission)
            self.cash -= total_cost
            logger.debug(
                "[OUVERTURE] %s - Montant débité: %.2f, Cash restant: %.2f",
                asset,
                total_cost,
                self.cash,
            )

        try:
            # Ouvrir la position avec les paramètres de gestion des risques
            self.positions[asset].open(
                entry_price=price,
                size=size,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

            # Mettre à jour le compteur de trades
            self.trade_count += 1

            # Préparer les informations de suivi
            trade_info = {
                "type": "open",
                "asset": asset,
                "size": size,
                "price": price,
                "stop_loss": price * (1 - stop_loss_pct) if stop_loss_pct > 0 else None,
                "take_profit": price * (1 + take_profit_pct)
                if take_profit_pct > 0
                else None,
                "commission": commission,
                "timestamp": datetime.now().isoformat(),
                "current_cash": self.cash,
                "portfolio_value": self.portfolio_value,
                "available_capital": self.get_available_capital(),
                "leverage": self.leverage if self.futures_enabled else 1.0,
            }

            # Ajouter au journal des trades
            self.trade_log.append(trade_info)

            # Journalisation détaillée
            logger.info(
                "[POSITION OUVERTE] %s - Taille: %.8f @ %.8f | Valeur: %.2f | SL: %.8f | TP: %.8f | Commission: %.2f",
                asset.upper(),
                size,
                price,
                notional_value,
                trade_info["stop_loss"] if trade_info["stop_loss"] else 0.0,
                trade_info["take_profit"] if trade_info["take_profit"] else 0.0,
                commission,
            )

            # Journalisation des soldes
            logger.debug(
                "[SOLDE] Cash: %.2f | Capital disponible: %.2f | Valeur portefeuille: %.2f",
                self.cash,
                self.get_available_capital(),
                self.portfolio_value,
            )

            return True

        except Exception as e:
            logger.error(
                "[ERREUR] Échec de l'ouverture de la position pour %s: %s",
                asset,
                str(e),
                exc_info=True,
            )
            # Annuler les modifications en cas d'erreur
            if self.futures_enabled:
                self.cash += (notional_value / self.leverage) + commission
            else:
                self.cash += notional_value + commission

            return False

    def close_position(self, asset: str, price: float) -> float:
        """
        Ferme la position ouverte pour un actif spécifique.

        Cette méthode gère la fermeture d'une position, calcule le PnL réalisé,
        met à jour la trésorerie et journalise les détails de la transaction.

        Args:
            asset: L'actif pour lequel fermer la position.
            price: Le prix auquel fermer la position.

        Returns:
            float: Le PnL net réalisé (après commissions) ou 0 en cas d'erreur.
        """
        # Vérifier si une position est ouverte pour cet actif
        if asset not in self.positions or not self.positions[asset].is_open:
            logger.warning(
                "[FERMETURE] Impossible de fermer la position pour %s: aucune position ouverte",
                asset,
            )
            return 0.0

        position = self.positions[asset]
        position_size = position.size
        entry_price = position.entry_price

        # Calculer le PnL brut (sans commission)
        trade_pnl = (price - entry_price) * position_size

        # Calculer la valeur notionnelle et la commission
        notional_value = position_size * price
        commission = notional_value * self.commission_pct

        # Calculer le PnL net (après commission)
        net_pnl = trade_pnl - commission

        # Journalisation avant fermeture
        logger.debug(
            "[FERMETURE] Préparation de la fermeture pour %s - Taille: %.8f @ %.8f | Prix entrée: %.8f | Prix sortie: %.8f",
            asset.upper(),
            position_size,
            price,
            entry_price,
            price,
        )

        try:
            # Mettre à jour la trésorerie selon le type de trading
            if self.futures_enabled:
                # Pour les contrats à terme : libérer la marge + PnL - commission
                margin_released = (position_size * entry_price) / self.leverage
                self.cash += margin_released + trade_pnl - commission
                logger.debug(
                    "[FERMETURE] %s - Marge libérée: %.2f | PnL brut: %.2f | Commission: %.2f",
                    asset.upper(),
                    margin_released,
                    trade_pnl,
                    commission,
                )
            else:
                # Pour le spot : récupérer l'investissement initial + PnL - commission
                initial_investment = position_size * entry_price
                self.cash += initial_investment + trade_pnl - commission
                logger.debug(
                    "[FERMETURE] %s - Investissement initial: %.2f | PnL brut: %.2f | Commission: %.2f",
                    asset.upper(),
                    initial_investment,
                    trade_pnl,
                    commission,
                )

            # Mettre à jour le PnL réalisé (net des commissions)
            self.realized_pnl += net_pnl

            # Calculer le pourcentage de gain/perte
            pnl_pct = (
                ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            )

            # Préparer les informations de suivi
            trade_info = {
                "type": "close",
                "asset": asset,
                "size": position_size,
                "entry_price": entry_price,
                "exit_price": price,
                "pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "commission": commission,
                "timestamp": datetime.now().isoformat(),
                "trade_pnl": net_pnl,  # Pour rétrocompatibilité
                "leverage": self.leverage if self.futures_enabled else 1.0,
                "position_value": notional_value,
                "cash_after": self.cash,
                "portfolio_value_after": self.portfolio_value,
                "available_capital_after": self.get_available_capital(),
            }

            # Ajouter au journal des trades
            self.trade_log.append(trade_info)

            # Fermer la position
            position.close()

            # Journalisation de la fermeture
            logger.info(
                "[POSITION FERMÉE] %s - Taille: %.8f | Entrée: %.8f | Sortie: %.8f | PnL: %+.2f (%.2f%%)",
                asset.upper(),
                position_size,
                entry_price,
                price,
                net_pnl,
                pnl_pct,
            )

            # Journalisation des soldes après fermeture
            logger.debug(
                "[SOLDE APRÈS FERMETURE] Cash: %.2f | Capital disponible: %.2f | Valeur portefeuille: %.2f",
                self.cash,
                self.get_available_capital(),
                self.portfolio_value,
            )

            return net_pnl

        except Exception as e:
            logger.error(
                "[ERREUR] Échec de la fermeture de la position pour %s: %s",
                asset,
                str(e),
                exc_info=True,
            )
            return 0.0

        # Fermer la position
        position.close()

        # Return gross PnL (without commission) as expected by the tests
        return trade_pnl

    def update_metrics(self) -> None:
        """
        Met à jour les métriques du portefeuille de manière robuste.

        Cette méthode calcule et met à jour les métriques de performance clés
        comme le drawdown et le ratio de Sharpe, avec une gestion robuste des erreurs
        et une stabilité numérique améliorée.

        La méthode est conçue pour être tolérante aux erreurs et ne jamais lever d'exception.
        """
        # Valeurs par défaut sécurisées
        self.drawdown = 0.0
        self.sharpe_ratio = 0.0

        try:
            # Vérification de l'historique
            if not hasattr(self, 'trade_history') or not self.trade_history:
                logger.debug("Aucun historique de trades disponible pour le calcul des métriques")
                return

            # Conversion en tableau numpy avec gestion des erreurs
            try:
                history_array = np.asarray(self.trade_history, dtype=np.float64)
                if history_array.size == 0:
                    return
            except (TypeError, ValueError) as e:
                logger.error("Erreur lors de la conversion de l'historique: %s", str(e))
                return

            # 1. Calcul du drawdown
            self._update_drawdown_metrics(history_array)

            # 2. Calcul des métriques de rendement et risque
            self._update_return_metrics(history_array)

        except Exception as e:
            logger.error(
                "Erreur critique dans update_metrics: %s",
                str(e),
                exc_info=True
            )
            # En cas d'erreur, on conserve les valeurs par défaut
            self.drawdown = 0.0
            self.sharpe_ratio = 0.0

    def _update_drawdown_metrics(self, history_array: np.ndarray) -> None:
        """Met à jour les métriques de drawdown de manière robuste."""
        try:
            if len(history_array) < 2:
                return

            # Calcul des valeurs cumulées maximales avec protection numérique
            with np.errstate(divide='ignore', invalid='ignore'):
                # Remplacement des valeurs non finies par 0
                clean_history = np.nan_to_num(history_array, nan=0.0, posinf=0.0, neginf=0.0)
                cummax = np.maximum.accumulate(clean_history)

                # Calcul du drawdown avec protection contre division par zéro
                mask = cummax > 1e-10  # Évite les divisions par des valeurs très petites
                drawdowns = np.zeros_like(clean_history)
                drawdowns[mask] = (cummax[mask] - clean_history[mask]) / cummax[mask]

                # Calcul du drawdown maximum, limité à 100%
                self.drawdown = float(np.clip(np.max(drawdowns), 0.0, 1.0))

        except Exception as e:
            logger.error("Erreur dans _update_drawdown_metrics: %s", str(e))
            self.drawdown = 0.0

    def _update_return_metrics(self, history_array: np.ndarray) -> None:
        """Calcule les métriques de rendement (Sharpe ratio, etc.) de manière robuste."""
        try:
            if len(history_array) < 2:
                return

            # Nettoyage des données
            clean_history = np.nan_to_num(history_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Calcul des retours journaliers en pourcentage
            prev_values = clean_history[:-1]
            next_values = clean_history[1:]

            with np.errstate(divide='ignore', invalid='ignore'):
                # Calcul des rendements avec protection contre division par zéro
                valid_mask = prev_values > 1e-10  # Évite les divisions par des valeurs très petites
                returns = np.zeros_like(prev_values)
                returns[valid_mask] = (next_values[valid_mask] - prev_values[valid_mask]) / prev_values[valid_mask]

                # Suppression des valeurs aberrantes
                returns = returns[np.isfinite(returns)]

                # Calcul du ratio de Sharpe avec des conditions de protection
                if len(returns) >= 2:  # Au moins 2 points pour avoir une variance non nulle
                    returns_std = np.std(returns)
                    if returns_std > 1e-10:  # Évite la division par zéro
                        sharpe = np.mean(returns) / returns_std * np.sqrt(252)  # Annualisation
                        self.sharpe_ratio = float(np.clip(sharpe, -10.0, 10.0))  # Bornes raisonnables
                    else:
                        self.sharpe_ratio = 0.0
                else:
                    self.sharpe_ratio = 0.0

        except Exception as e:
            logger.error("Erreur dans _update_return_metrics: %s", str(e))
            self.sharpe_ratio = 0.0

        self.calculate_risk_metrics()

    def calculate_risk_metrics(self, confidence_level: float = 0.95) -> None:
        """
        Calcule la Value at Risk (VaR) et la Conditional Value at Risk (CVaR).

        Args:
            confidence_level: Niveau de confiance pour le calcul du VaR (par défaut: 0.95)
        """
        # Initialisation des valeurs par défaut
        self.var = 0.0
        self.cvar = 0.0

        # Vérification des conditions minimales
        if not hasattr(self, 'trade_history') or len(self.trade_history) < 2:
            return

        try:
            # Conversion sécurisée en tableau numpy
            history_array = np.asarray(self.trade_history, dtype=np.float64)

            # Vérification du tableau
            if history_array.size < 2:
                return

            # Calcul des retours avec protection
            prev_values = history_array[:-1]
            next_values = history_array[1:]

            # Calcul des rendements avec gestion des erreurs numériques
            with np.errstate(divide='ignore', invalid='ignore'):
                # Masque pour éviter les divisions par zéro
                valid_mask = (prev_values > 1e-10) & np.isfinite(prev_values)
                returns = np.zeros_like(prev_values)
                returns[valid_mask] = (next_values[valid_mask] - prev_values[valid_mask]) / prev_values[valid_mask]

            # Nettoyage des valeurs non finies
            returns = returns[np.isfinite(returns)]

            if len(returns) <= 1:  # Pas assez de données pour calculer le risque
                return

            # Tri des rendements
            sorted_returns = np.sort(returns)

            # Calcul de l'index pour le VaR avec protection des bornes
            var_index = int(np.floor(len(sorted_returns) * (1 - confidence_level)))
            var_index = max(0, min(var_index, len(sorted_returns) - 1))

            # Calcul du VaR (valeur absolue du quantile des pertes)
            if 0 <= var_index < len(sorted_returns):
                self.var = float(np.abs(sorted_returns[var_index]))

                # Calcul du CVaR (moyenne des pertes pires que le VaR)
                if var_index > 0:
                    cvar_returns = sorted_returns[:var_index]
                    if len(cvar_returns) > 0:
                        self.cvar = float(np.abs(np.mean(cvar_returns)))
                    else:
                        self.cvar = self.var  # Si pas de pertes pires, on prend le VaR
                else:
                    self.cvar = self.var  # Si pas de pertes pires, on prend le VaR

            # Protection contre les valeurs aberrantes
            self.var = min(self.var, 1.0)  # Ne peut pas dépasser 100%
            self.cvar = min(self.cvar, 1.0)  # Ne peut pas dépasser 100%

            # Log de débogage
            logger.debug(
                "Métriques de risque calculées - VaR: %.4f, CVaR: %.4f (n=%d)",
                self.var, self.cvar, len(returns)
            )

        except Exception as e:
            logger.error(
                "Erreur dans calculate_risk_metrics: %s",
                str(e),
                exc_info=True
            )
            # En cas d'erreur, on conserve les valeurs par défaut (0.0)
            self.cvar = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of current portfolio metrics.

        Returns:
            Dict containing comprehensive portfolio metrics
        """
        # Initialize default values for all metrics
        metrics = {
            "total_positions": 0,
            "positions": {},
            "trade_count": 0,
            "drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "var": 0.0,
            "cvar": 0.0,
            "total_pnl_pct": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "initial_capital": getattr(self, "initial_capital", 0.0),
            "current_equity": getattr(self, "current_equity", 0.0),
            "unrealized_pnl": getattr(self, "unrealized_pnl", 0.0),
            "realized_pnl": getattr(self, "realized_pnl", 0.0),
            "total_capital": getattr(self, "total_capital", 0.0),
            "cash": getattr(self, "cash", 0.0),
            "portfolio_value": getattr(self, "portfolio_value", 0.0),
            "leverage": getattr(self, "leverage", 1.0),
        }

        # Calculate basic metrics
        if hasattr(self, "initial_capital") and self.initial_capital > 0:
            metrics["total_pnl_pct"] = (
                (metrics["total_capital"] / self.initial_capital) - 1
            ) * 100

        # Initialize trade_log if it doesn't exist
        if not hasattr(self, "trade_log") or not isinstance(self.trade_log, list):
            self.trade_log = []
            logger.warning(
                "trade_log n'était pas initialisé, initialisation avec une liste vide"
            )
            return metrics

        # Calculate trade metrics with safe dictionary access
        try:
            # Get all closed trades safely
            closed_trades = [
                t
                for t in self.trade_log
                if isinstance(t, dict) and t.get("type") == "close"
            ]
            metrics["total_trades"] = len(closed_trades)

            # Calculate winning and losing trades with safe value access
            winning_trades = [
                t for t in closed_trades if float(t.get("trade_pnl", 0)) > 0
            ]
            losing_trades = [
                t for t in closed_trades if float(t.get("trade_pnl", 0)) <= 0
            ]

            metrics["winning_trades"] = len(winning_trades)
            metrics["losing_trades"] = len(losing_trades)

            # Calculate win rate
            if metrics["total_trades"] > 0:
                metrics["win_rate"] = (
                    metrics["winning_trades"] / metrics["total_trades"]
                ) * 100

            # Calculate average win/loss with safe value access
            if winning_trades:
                metrics["avg_win"] = float(
                    np.mean([float(t.get("trade_pnl", 0)) for t in winning_trades])
                )

            if losing_trades:
                metrics["avg_loss"] = abs(
                    float(
                        np.mean([float(t.get("trade_pnl", 0)) for t in losing_trades])
                    )
                )

            # Calculate profit factor safely
            if losing_trades and metrics["avg_loss"] > 0:
                total_win = metrics["avg_win"] * metrics["winning_trades"]
                total_loss = metrics["avg_loss"] * metrics["losing_trades"]
                if total_loss > 0:  # Avoid division by zero
                    metrics["profit_factor"] = total_win / total_loss

        except Exception as e:
            logger.error(
                "Erreur lors du calcul des métriques de trading: %s",
                str(e),
                exc_info=True,
            )
            # En cas d'erreur, on garde les valeurs par défaut déjà définies

        # Prepare position metrics with safe attribute access
        try:
            positions_metrics = {}
            for asset, position in getattr(self, "positions", {}).items():
                try:
                    position_data = {
                        "size": getattr(position, "size", 0.0),
                        "entry_price": getattr(position, "entry_price", 0.0),
                        "is_open": getattr(position, "is_open", False),
                        "unrealized_pnl": 0.0,
                        "leverage": getattr(position, "leverage", 1.0),
                    }

                    # Calculate unrealized PnL safely
                    if position_data["is_open"] and hasattr(position, "current_price"):
                        entry_price = position_data["entry_price"]
                        current_price = getattr(position, "current_price", entry_price)
                        position_data["unrealized_pnl"] = (
                            current_price - entry_price
                        ) * position_data["size"]

                    positions_metrics[asset] = position_data

                except Exception as pos_e:
                    logger.error(
                        "Erreur lors du calcul des métriques pour la position %s: %s",
                        asset,
                        str(pos_e),
                        exc_info=True,
                    )
                    continue

            # Update metrics with position data
            metrics.update(
                {
                    "total_positions": len(positions_metrics),
                    "positions": positions_metrics,
                }
            )

        except Exception as e:
            logger.error(
                "Erreur lors de la préparation des métriques de position: %s",
                str(e),
                exc_info=True,
            )
            # On continue avec les métriques déjà calculées

        # Update metrics with portfolio data safely
        try:
            metrics.update(
                {
                    "trade_count": getattr(self, "trade_count", 0),
                    "drawdown": getattr(self, "drawdown", 0.0),
                    "sharpe_ratio": getattr(self, "sharpe_ratio", 0.0),
                    "var": getattr(self, "var", 0.0),
                    "cvar": getattr(self, "cvar", 0.0),
                    "unrealized_pnl": getattr(self, "unrealized_pnl", 0.0),
                    "realized_pnl": getattr(self, "realized_pnl", 0.0),
                    "portfolio_value": getattr(self, "portfolio_value", 0.0),
                }
            )
        except Exception as e:
            logger.error(
                "Erreur lors de la mise à jour des métriques du portefeuille: %s",
                str(e),
                exc_info=True,
            )

        return metrics

    def get_state_features(self) -> np.ndarray:
        """
        Returns a numpy array of features representing the portfolio's state.
        """
        features = []
        for asset in self.config["assets"]:
            position = self.positions[asset]
            has_position = 1.0 if position.is_open else 0.0

            if position.is_open:
                entry_value = position.entry_price * position.size
                # This needs current price to be accurate, which is not available here.
                # Passing 0 for now, to be fixed in a later step.
                relative_pnl = 0.0
            else:
                relative_pnl = 0.0

            features.extend([has_position, relative_pnl])

        return np.array(features, dtype=np.float32)

    def get_feature_size(self) -> int:
        """
        Returns the number of features in the portfolio's state representation.
        """
        # For each asset, we have 'has_position' and 'relative_pnl'
        return len(self.config["assets"]) * 2

    def is_bankrupt(self) -> bool:
        """
        Checks if the portfolio value has fallen below a critical threshold.
        """
        # Consider bankrupt if capital is less than 1% of initial capital
        return self.total_capital < (self.initial_capital * 0.01)

    def start_new_chunk(self) -> None:
        """
        Call this method when starting to process a new chunk of data.
        This will finalize the previous chunk's PnL and start tracking a new chunk.
        """
        # Finalize the previous chunk's PnL if this isn't the first chunk
        if self.current_chunk_id > 0:
            self._finalize_chunk_pnl()

        # Start a new chunk
        self.current_chunk_id += 1
        self.chunk_start_equity = self.total_capital
        self.trade_count = 0  # Reset trade count for the new chunk
        logger.info(
            "Started tracking new chunk %d with starting equity: $%.2f",
            self.current_chunk_id,
            self.chunk_start_equity,
        )

    def _finalize_chunk_pnl(self) -> None:
        """Calculate and store the PnL for the current chunk."""
        if self.current_chunk_id == 0:
            return

        chunk_pnl_pct = (
            (self.total_capital - self.chunk_start_equity) / self.chunk_start_equity
        ) * 100

        self.chunk_pnl[self.current_chunk_id] = {
            "start_equity": self.chunk_start_equity,
            "end_equity": self.total_capital,
            "pnl_pct": chunk_pnl_pct,
            "n_trades": self.trade_count,  # Use self.trade_count directly
        }

        logger.info(
            "Chunk %d completed with PnL: %.2f%% (Equity: $%.2f -> $%.2f)",
            self.current_chunk_id,
            chunk_pnl_pct,
            self.chunk_start_equity,
            self.total_capital,
        )

    def get_chunk_performance_ratio(self, chunk_id: int, optimal_pnl: float) -> float:
        """
        Calculate the performance ratio for a specific chunk compared to the optimal PnL.

        Args:
            chunk_id: The ID of the chunk to calculate the ratio for.
            optimal_pnl: The optimal possible PnL for this chunk.

        Returns:
            float: The performance ratio (actual_pnl / optimal_pnl), clipped to [0, 1].
        """
        if chunk_id not in self.chunk_pnl:
            logger.warning("No PnL data found for chunk %d", chunk_id)
            return 0.0

        if optimal_pnl <= 0:
            return 0.0

        actual_pnl = self.chunk_pnl[chunk_id]["pnl_pct"]
        ratio = actual_pnl / optimal_pnl

        # Clip the ratio between 0 and 1 to prevent extreme values
        return max(0.0, min(1.0, ratio))

    def rebalance(self, current_prices: Dict[str, float]):
        """Rebalances the portfolio to match target allocations and concentration limits."""
        logger.info("Rebalancing portfolio...")

        if not current_prices:
            logger.warning("Cannot rebalance: no current prices provided.")
            return

        total_portfolio_value = self.get_portfolio_value()
        if total_portfolio_value <= 0:
            logger.warning(
                "Cannot rebalance: total portfolio value is zero or negative."
            )
            return

        max_single_asset_limit = self.concentration_limits.get(
            "max_single_asset", 1.0
        )  # Default to 100%

        for asset, position in self.positions.items():
            if position.is_open:
                current_price = current_prices.get(asset)
                if current_price is None:
                    logger.warning(
                        "Cannot rebalance %s: current price not available.", asset
                    )
                    continue

                position_value = position.size * current_price
                current_allocation = position_value / total_portfolio_value

                if current_allocation > max_single_asset_limit:
                    # Calculate the excess amount to sell
                    excess_value = position_value - (
                        max_single_asset_limit * total_portfolio_value
                    )
                    sell_size = excess_value / current_price

                    logger.info(
                        "Rebalancing %s: current allocation %.2f exceeds limit %.2f. "
                        "Selling %.4f units.",
                        asset,
                        current_allocation,
                        max_single_asset_limit,
                        sell_size,
                    )
                    # Simulate closing a portion of the position
                    # This is a simplified close; in a real scenario, you'd adjust the existing position object
                    # and potentially execute a partial sell order.
                    self.cash += excess_value * (
                        1 - self.commission_pct
                    )  # Deduct commission on sell
                    position.size -= sell_size
                    if position.size <= 0:
                        position.close()  # Close if size becomes zero or negative
                        logger.info(
                            "Position for %s fully closed during rebalancing.", asset
                        )

        self.update_metrics()
        logger.info("Portfolio rebalancing completed.")

    def validate_position(self, asset: str, size: float, price: float) -> bool:
        """
        Validates if a position can be opened with the given parameters.

        This method checks:
        1. If the price is valid
        2. If the position meets minimum/maximum size requirements
        3. If there's sufficient available capital including commissions
        4. If concentration limits are respected

        Args:
            asset: Asset symbol
            size: Size of the position (positive for long, negative for short)
            price: Entry price

        Returns:
            bool: True if the position is valid, False otherwise
        """
        # Check if price is valid
        if price <= 0:
            logger.warning("Invalid price: %.8f", price)
            return False

        # If protection is active in spot mode, block only new long entries (buys)
        if not self.futures_enabled and self.trading_disabled and size > 0:
            logger.warning(
                "[PROTECTION] Trading disabled for new BUY orders due to drawdown breach. Request blocked: %s size=%.8f @ %.8f",
                asset,
                size,
                price,
            )
            # Standardized guard log for easy grep during integration runs
            logger.warning(
                "[GUARD] Rejecting BUY due to trading_disabled (reason=spot_drawdown) asset=%s size=%.8f price=%.8f",
                asset,
                size,
                price,
            )
            return False

        # Check minimum trade size
        if abs(size) < self.min_trade_size:
            logger.warning(
                "Position size (%.8f) is less than minimum trade size (%.8f).",
                size,
                self.min_trade_size,
            )
            return False

        # Check against notional value limits
        notional_value = (
            abs(size) * price
        )  # Use absolute value to handle short positions

        if notional_value < self.min_notional_value:
            logger.warning(
                "[VALIDATION] Notional value %.2f < minimum %.2f for %s (size=%.8f @ %.8f)",
                notional_value,
                self.min_notional_value,
                asset,
                size,
                price,
            )
            return False

        if notional_value > self.max_notional_value:
            logger.warning(
                "[VALIDATION] Notional value %.2f > maximum %.2f for %s (size=%.8f @ %.8f)",
                notional_value,
                self.max_notional_value,
                asset,
                size,
                price,
            )
            return False

        # Calculate required capital including commission
        position_value = abs(size) * price
        commission = position_value * self.commission_pct

        # Calculate required margin based on trading mode
        if self.futures_enabled and self.leverage > 1.0:
            required_margin = (position_value / self.leverage) + commission
        else:
            required_margin = position_value + commission  # Spot trading or no leverage

        # Get available capital using the dedicated method
        available_capital = self.get_available_capital()

        # Check available capital
        if required_margin > available_capital:
            logger.warning(
                "[VALIDATION] Insufficient capital for %s: Required=%.2f, Available=%.2f, "
                "Portfolio=%.2f (size=%.8f @ %.8f, commission=%.4f)",
                asset,
                required_margin,
                available_capital,
                self.portfolio_value,
                size,
                price,
                commission,
            )
            return False

        # Check concentration limits if defined
        if self.concentration_limits:
            portfolio_value = self.get_portfolio_value()
            position_pct = position_value / portfolio_value

            # Check maximum position size
            max_position_pct = self.concentration_limits.get("max_position_pct", 1.0)
            if position_pct > max_position_pct:
                logger.warning(
                    "[VALIDATION] Position size %.2f%% > max %.2f%% for %s",
                    position_pct * 100,
                    max_position_pct * 100,
                    asset,
                )
                return False

            # Check per-asset concentration
            current_asset_value = 0.0
            if asset in self.positions and self.positions[asset].is_open:
                current_asset_value = abs(self.positions[asset].size * price)

            new_asset_value = current_asset_value + position_value
            max_asset_pct = self.concentration_limits.get("max_asset_pct", 0.5)

            if new_asset_value / portfolio_value > max_asset_pct:
                logger.warning(
                    "[VALIDATION] Asset concentration %.2f%% > max %.2f%% for %s",
                    (new_asset_value / portfolio_value) * 100,
                    max_asset_pct * 100,
                    asset,
                )
                return False

        return True

    def get_active_tier(self):
        """
        Détermine le palier de capital actif en fonction de la valeur du portefeuille.

        Returns:
            dict: Le palier de capital actif avec toutes ses propriétés.

        Raises:
            RuntimeError: Si aucun palier valide n'est trouvé.
        """
        if not self.capital_tiers:
            raise RuntimeError(
                "Aucun palier de capital n'est défini dans la configuration."
            )

        current_value = self.get_portfolio_value()

        # Parcourir les paliers du plus élevé au plus bas
        for tier in sorted(
            self.capital_tiers, key=lambda x: x["min_capital"], reverse=True
        ):
            if current_value >= tier["min_capital"]:
                logger.debug(
                    "Palier actif: %s (capital: %.2f >= %.2f)",
                    tier["name"],
                    current_value,
                    tier["min_capital"],
                )
                return tier

        # Si on arrive ici, utiliser le palier le plus bas
        min_tier = min(self.capital_tiers, key=lambda x: x["min_capital"])
        logger.warning(
            "La valeur du portefeuille (%.2f) est inférieure au palier minimum (%.2f). "
            "Utilisation du palier: %s",
            current_value,
            min_tier["min_capital"],
            min_tier["name"],
        )
        return min_tier
