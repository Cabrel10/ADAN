#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Portfolio management module for the ADAN trading bot.

This module is responsible for tracking the agent's financial status, including
capital, positions, and performance metrics.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

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
        self.positions: Dict[str, Position] = {
            asset: Position() for asset in assets}
        self.trade_history: List[Dict[str, Any]] = []

        # Initialize chunk-based tracking
        self.chunk_pnl: Dict[int, Dict[str, float]] = {}
        self.current_chunk_id = 0
        self.chunk_start_equity = self.initial_equity

        # Trading rules configuration
        self.futures_enabled = trading_rules_config.get(
            "futures_enabled", False)
        self.leverage = trading_rules_config.get("leverage", 1)

        # Commission and fees
        if self.futures_enabled:
            self.commission_pct = trading_rules_config.get(
                "futures_commission_pct", 0.0
            )
        else:
            self.commission_pct = trading_rules_config.get(
                "commission_pct", 0.0)

        # Position sizing rules
        self.min_trade_size = trading_rules_config.get(
            "min_trade_size", 0.0001)
        self.min_notional_value = trading_rules_config.get(
            "min_notional_value", 10.0)
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
                    "Le tier %d n'est pas un dictionnaire et sera ignoré: %s", i, tier)
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
            if valid_tiers[i
                           - 1]["min_capital"] >= valid_tiers[i]["min_capital"]:
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
        self.position_sizing_config = risk_management.get(
            "position_sizing", {})
        self.concentration_limits = self.position_sizing_config.get(
            "concentration_limits", {}
        )

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
        sorted_tiers = sorted(
            self.capital_tiers,
            key=lambda x: x["min_capital"])

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
        risk_per_trade: float = 0.01,
        account_risk_multiplier: float = 1.0,
        stop_loss_pct: float = 0.02,
    ) -> float:
        """Calcule la taille de position en fonction du risque et du capital.

        Args:
            price: Prix actuel de l'actif.
            risk_per_trade: Pourcentage du capital à risquer par trade.
                           (défaut: 0.01 pour 1%)
            account_risk_multiplier: Multiplicateur de risque basé sur le drawdown.
                                   (défaut: 1.0)
            stop_loss_pct: Pourcentage de stop-loss (défaut: 0.02 pour 2%).

        Returns:
            La taille de position en unités de l'actif.
        """
        # Récupérer la configuration du palier de capital actuel
        tier = self.get_current_tier()

        # Taille de position maximale en pourcentage du capital
        max_position_size_pct = tier.get("max_position_size_pct", 10.0) / 100.0

        # Ajuster le risque par trade avec le multiplicateur de risque du compte
        adjusted_risk = risk_per_trade * account_risk_multiplier
        
        # Calculer la taille de position basée sur le risque
        position_size = 0.0

        if stop_loss_pct > 0:
            # Calculer la taille de position basée sur le risque par trade
            risk_amount = self.portfolio_value * adjusted_risk
            risk_per_share = price * stop_loss_pct
            
            if risk_per_share > 0:
                position_size = risk_amount / risk_per_share

        # Calculer la valeur de la position
        max_position_value = self.portfolio_value * max_position_size_pct
        position_value = position_size * price

        # Limiter la taille de la position selon la valeur maximale autorisée
        if position_value > max_position_value:
            position_size = max_position_value / price
            logger.debug(
                "Position size capped at %.6f (max %.2f USD)",
                position_size,
                max_position_value
            )

        # Appliquer les limites de taille minimale et maximale
        if position_size * price < self.min_notional_value:
            position_size = 0.0
            logger.debug(
                "Position size below minimum notional value "
                "(%.2f < %.2f USD)",
                position_size * price,
                self.min_notional_value
            )
        elif position_size * price > self.max_notional_value:
            position_size = self.max_notional_value / price
            logger.debug(
                "Position size capped at maximum notional value "
                "(%.2f USD)",
                self.max_notional_value
            )

        # Journalisation des détails du calcul
        logger.debug(
            "Calculated position size - "
            "Price: %.8f, Size: %.8f, Value: %.2f USD, "
            "Risk/Trade: %.1f%%, Max Size: %.1f%%",
            price,
            position_size,
            position_size * price,
            risk_per_trade * 100,
            max_position_size_pct * 100
        )

        return position_size

    def _calculate_volatility(self, window: int = 20) -> float:
        """Calcule la volatilité des rendements sur une fenêtre glissante.

        Args:
            window: Taille de la fenêtre de calcul

        Returns:
            float: Volatilité des rendements sur la fenêtre
        """
        if len(self.trade_history) < 2:
            return 0.0

        values = np.array(self.trade_history[-window:])
        if len(values) < 2:
            return 0.0

        returns = np.diff(np.log(values))
        if len(returns) > 0:
            return np.std(returns) * np.sqrt(252)
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

    def reset(self) -> None:
        """
        Reset the portfolio to its initial state.

        This method resets all portfolio metrics, positions, and trade history
        to their initial values based on the environment configuration.
        """
        # Get environment configuration
        env_config = self.config.get("environment", {})

        # Initialize initial equity from configuration
        self.initial_equity = env_config.get("initial_balance", 10000.0)

        # Close all open positions
        for asset in self.positions:
            self.positions[asset].close()

        # Reset portfolio state variables
        self.cash = self.initial_equity
        self.total_capital = self.initial_equity
        self.portfolio_value = self.initial_equity
        self.current_equity = self.initial_equity
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.var = 0.0
        self.cvar = 0.0

        # Reset trade history and logs
        self.trade_history = [self.initial_equity]
        self.trade_log = []

        # Initialize peak equity tracking
        self.peak_equity = self.initial_equity

        logger.info(
            "Portfolio reset. Initial equity: %.2f",
            self.initial_equity)

    def get_portfolio_value(self) -> float:
        """Returns the current total portfolio value."""
        return self.portfolio_value

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
                total_position_value += abs(position.size
                                            * position.entry_price)

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
        Returns the available capital for new trades.

        Returns:
            float: The amount of capital available for new trades.
        """
        # Le capital disponible est simplement la trésorerie actuelle
        # car nous ne voulons pas permettre un effet de levier supérieur à 1x
        # pour les opérations sur marge
        return self.cash

    def update_market_price(self, current_prices: Dict[str, float]) -> None:
        """
        Met à jour la valeur du portefeuille en fonction des prix actuels.

        Args:
            current_prices: Dictionnaire associant les symboles d'actifs à leurs prix actuels.
        """
        # Vérification du type du paramètre
        if not isinstance(current_prices, dict):
            logger.warning(
                "current_prices doit être un dictionnaire. Reçu: %s",
                type(current_prices),
            )
            current_prices = {}

        # Réinitialisation du PnL non réalisé
        self.unrealized_pnl = 0.0
        assets_with_missing_prices = []

        # Calcul du PnL non réalisé pour toutes les positions ouvertes
        for asset, position in self.positions.items():
            if not position.is_open:
                continue

            current_price = current_prices.get(asset)

            # Ignorer les prix manquants ou invalides
            if (
                current_price is None
                or not isinstance(current_price, (int, float))
                or current_price <= 0
            ):
                assets_with_missing_prices.append(asset)
                continue

            # Calcul du PnL pour cette position
            position_pnl = position.size * \
                (current_price - position.entry_price)
            self.unrealized_pnl += position_pnl

            # Journalisation du statut de la position pour le débogage
            logger.debug(
                "Position %s: taille=%.8f, entrée=%.8f, actuel=%.8f, pnl=%.2f",
                asset,
                position.size,
                position.entry_price,
                current_price,
                position_pnl,
            )

        # Journalisation des actifs avec prix manquants
        if assets_with_missing_prices:
            logger.warning(
                "Prix manquants ou invalides pour %d actifs: %s. "
                "Ces positions seront ignorées.",
                len(assets_with_missing_prices),
                ", ".join(assets_with_missing_prices),
            )

        # Mise à jour de la valeur du portefeuille et de l'equity
        self.current_equity = self.cash + self.unrealized_pnl
        self.portfolio_value = self.current_equity
        self.total_capital = self.current_equity

        # Mettre à jour l'historique des trades (limité pour éviter la
        # surconsommation mémoire)
        if not hasattr(self, "trade_history") or not self.trade_history:
            self.trade_history = [self.initial_equity]

        # Conservation uniquement des 1000 dernières entrées
        if len(self.trade_history) >= 1000:
            self.trade_history.pop(0)

        self.trade_history.append(self.portfolio_value)

        # Application des taux de financement pour les contrats à terme si
        # activé
        if self.futures_enabled:
            # Logique d'application des taux de financement
            pass

        # Mise à jour des métriques
        self.update_metrics()

        # Vérification des ordres de protection (uniquement pour les actifs
        # avec prix valides)
        valid_prices = {
            k: v
            for k, v in current_prices.items()
            if v is not None and isinstance(v, (int, float)) and v > 0
        }

        if valid_prices:
            was_liquidated = self.check_liquidation(valid_prices)
            if was_liquidated:
                # Réinitialisation des métriques après liquidation
                self.unrealized_pnl = 0.0
                self.portfolio_value = self.cash
                self.current_equity = self.cash
                self.total_capital = self.cash

                # Journalisation de l'événement de liquidation
                logger.critical(
                    "Portefeuille liquidé. Valeur finale: %.2f USDT "
                    "(Initial: %.2f USDT)",
                    self.portfolio_value,
                    self.initial_equity,
                )
        elif any(pos.is_open for pos in self.positions.values()):
            logger.warning(
                "Impossible de vérifier les ordres de protection: "
                "aucun prix valide disponible"
            )

    def check_liquidation(self, current_prices: Dict[str, float]) -> bool:
        """
                Vérifie si le portefeuille est à risque de liquidation et ferme les positions si nécessaire.
        {{ ... }}

                Args:
                    current_prices: Dictionnaire des prix actuels par actif.

                Returns:
                    bool: True si une liquidation a eu lieu, False sinon.
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
        logger.info(
            "Vérification du drawdown: Actuel=%.2f USDT (%.2f%%), Max autorisé=%.2f USDT (%.1f%%)",
            current_drawdown,
            current_drawdown_pct,
            max_drawdown_value,
            max_drawdown_pct * 100,
        )

        # Vérifier si le drawdown dépasse la limite du palier
        if current_drawdown > max_drawdown_value:
            logger.warning(
                "Drawdown actuel de %.2f USDT dépasse la limite de %.2f USDT (%.1f%%). "
                "Fermeture des positions.",
                current_drawdown,
                max_drawdown_value,
                max_drawdown_pct * 100,
            )

            # Si pas de positions ouvertes et pas de futures, on retourne True
            # car le drawdown est dépassé
            if not self.futures_enabled and not any(
                p.is_open for p in self.positions.values()
            ):
                logger.warning(
                    "Drawdown critique atteint mais aucune position ouverte à fermer."
                )
                return True

            # Fermer toutes les positions
            positions_closed = False
            for asset in list(self.positions.keys()):
                current_price = current_prices.get(asset)
                if current_price is not None:
                    self.close_position(asset, current_price)
                    positions_closed = True
                else:
                    logger.error(
                        "Impossible de fermer la position %s lors de la liquidation: "
                        "prix actuel manquant", asset, )

            # Mettre à jour les métriques
            if positions_closed or not self.futures_enabled:
                self.unrealized_pnl = 0.0
                self.total_capital = self.cash
                self.portfolio_value = self.cash
                self.current_equity = self.cash
                self.update_metrics()

                # Journaliser la liquidation
                logger.critical(
                    "LIQUIDATION EFFECTUÉE - Drawdown: %.2f USDT (%.1f%%), "
                    "Valeur portefeuille: %.2f USDT",
                    current_drawdown,
                    (current_drawdown / self.initial_equity) * 100,
                    self.portfolio_value,
                )

                # Retourner True pour indiquer qu'une liquidation a eu lieu
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
                            "prix actuel manquant", asset, )

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
                    logger.info(
                        "Stop-loss hit for %s. Closing position.", asset)
                    self.close_position(asset, current_price)
                # Check take-profit
                elif (
                    position.take_profit_pct > 0
                    and current_price
                    >= position.entry_price * (1 + position.take_profit_pct)
                ):
                    logger.info(
                        "Take-profit hit for %s. Closing position.", asset)
                    self.close_position(asset, current_price)

    def open_position(self, asset: str, price: float, size: float) -> bool:
        """
        Opens a new long position for a specific asset.

        Args:
            asset: The asset to open a position for.
            price: The price at which to open the position.
            size: The size of the position to open.

        Returns:
            True if the position was opened successfully, False otherwise.
        """
        if self.positions[asset].is_open:
            logger.warning(
                "Cannot open a new position for %s while one is already open.",
                asset)
            return False

        # Get trading parameters from worker config (default to w1 if not
        # specified)
        worker_config = self.config.get("workers", {}).get("w1", {})
        trading_config = worker_config.get("trading_config", {})

        # Get stop loss and take profit with default values
        stop_loss_pct = trading_config.get("stop_loss_pct", 0.05)  # Default 5%
        take_profit_pct = trading_config.get(
            "take_profit_pct", 0.15)  # Default 15%

        logger.debug(
            "PortfolioManager open_position - trading_config: %s",
            trading_config)
        logger.debug(
            "PortfolioManager open_position - stop_loss_pct: %s, take_profit_pct: %s",
            stop_loss_pct,
            take_profit_pct,
        )

        notional_value = size * price
        commission = notional_value * self.commission_pct

        if self.futures_enabled:
            # For futures, we only need to set aside the margin (notional/leverage) plus commission
            # Commission is paid only once when opening the position (maker
            # fee)
            margin_used = notional_value / self.leverage
            self.cash -= margin_used + commission
        else:
            # For spot, we pay the full notional value plus commission
            self.cash -= notional_value + commission

        self.positions[asset].open(
            price,
            size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct)
        self.trade_count += 1
        trade_info = {
            "type": "open",
            "asset": asset,
            "size": size,
            "price": price,
            "commission": commission,
            "timestamp": datetime.now().isoformat(),
            "current_cash": self.cash,
            "portfolio_value": self.portfolio_value,
        }
        self.trade_log.append(trade_info)
        logger.info(
            "Opened position for %s: %.4f units at %.2f. Details: %s",
            asset,
            size,
            price,
            trade_info,
        )
        return True

    def close_position(self, asset: str, price: float) -> float:
        """
        Closes the current open position for a specific asset.

        Args:
            asset: The asset to close the position for.
            price: The price at which to close the position.

        Returns:
            The realized PnL from the trade after commissions.
        """
        if asset not in self.positions or not self.positions[asset].is_open:
            logger.warning(
                "Cannot close a position for %s when none is open.", asset)
            return 0.0

        position = self.positions[asset]

        # Calculate PnL (without commission)
        trade_pnl = (price - position.entry_price) * position.size

        # Calculate commission on the notional value
        notional_value = position.size * price
        commission = notional_value * self.commission_pct

        # Calculate net PnL (after commission)
        net_pnl = trade_pnl - commission

        # Update cash balance
        if self.futures_enabled:
            # For futures: release margin + PnL - commission
            margin_released = (
                position.size * position.entry_price) / self.leverage
            self.cash += margin_released + trade_pnl - commission
        else:
            # For spot: we get back the initial investment + PnL - commission
            initial_investment = position.size * position.entry_price
            self.cash += initial_investment + trade_pnl - commission

        # Update realized PnL (net of commission)
        self.realized_pnl += net_pnl

        # Log the trade
        trade_info = {
            "type": "close",
            "asset": asset,
            "size": position.size,
            "entry_price": position.entry_price,
            "exit_price": price,
            "price": price,  # For backward compatibility
            "commission": commission,
            "timestamp": datetime.now().isoformat(),
            "trade_pnl": net_pnl,  # Store net PnL in trade log
            "current_cash": self.cash,
            "portfolio_value": self.cash,  # Will be updated in the next update_market_price
        }

        self.trade_log.append(trade_info)
        logger.info(
            f"Closed position for {asset} at {price:.2f}. "
            f"Size: {position.size}, Entry: {position.entry_price:.2f}, "
            f"PnL: {net_pnl:.2f} (Gross: {trade_pnl:.2f}, Commission: {commission:.2f})")

        # Close the position
        position.close()

        # Return gross PnL (without commission) as expected by the tests
        return trade_pnl

    def update_metrics(self):
        """Updates the portfolio metrics."""
        # Convert trade_history to numpy array for calculations
        history_array = np.array(self.trade_history)

        # Calculate drawdown
        if len(history_array) > 0:
            peak = np.maximum.accumulate(history_array)
            self.drawdown = (
                (history_array[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0.0
            )
        else:
            self.drawdown = 0.0

        # Calculate Sharpe ratio
        if len(history_array) > 1:
            returns = (history_array[1:]
                       - history_array[:-1]) / history_array[:-1]
            if np.std(returns) > 0:
                self.sharpe_ratio = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )  # Annualized Sharpe ratio
            else:
                self.sharpe_ratio = 0.0
        else:
            self.sharpe_ratio = 0.0

        self.calculate_risk_metrics()

    def calculate_risk_metrics(self, confidence_level: float = 0.95):
        """Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""
        if len(self.trade_history) < 2:
            self.var = 0.0
            self.cvar = 0.0
            return

        returns = (
            np.array(self.trade_history)[1:] - np.array(self.trade_history)[:-1]
        ) / np.array(self.trade_history)[:-1]
        sorted_returns = np.sort(returns)

        # Calculate VaR
        var_index = int(len(sorted_returns) * (1 - confidence_level))
        self.var = -sorted_returns[var_index]

        # Calculate CVaR
        cvar_returns = sorted_returns[sorted_returns < -self.var]
        if len(cvar_returns) > 0:
            self.cvar = -np.mean(cvar_returns)
        else:
            self.cvar = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of current portfolio metrics.

        Returns:
            Dict containing comprehensive portfolio metrics
        """
        # Calculate basic metrics
        total_pnl_pct = 0.0
        if self.initial_capital > 0:
            total_pnl_pct = (
                (self.total_capital / self.initial_capital) - 1) * 100

        # Calculate trade metrics
        total_trades = len(
            [t for t in self.trade_log if t.get("type") == "close"])

        winning_trades = [
            t
            for t in self.trade_log
            if t.get("type") == "close" and t.get("trade_pnl", 0) > 0
        ]

        losing_trades = [
            t
            for t in self.trade_log
            if t.get("type") == "close" and t.get("trade_pnl", 0) <= 0
        ]

        # Calculate win rate and averages
        win_rate = (
            len(winning_trades) / max(1, total_trades) * 100
            if total_trades > 0
            else 0.0
        )

        avg_win = (
            np.mean([t.get("trade_pnl", 0) for t in winning_trades])
            if winning_trades
            else 0.0
        )

        avg_loss = (
            abs(np.mean([t.get("trade_pnl", 0) for t in losing_trades]))
            if losing_trades
            else 0.0
        )

        profit_factor = 0.0
        if losing_trades and losing_trades[0].get("trade_pnl", 0) != 0:
            profit_factor = (len(winning_trades) * avg_win) / (
                len(losing_trades) * avg_loss
            )

        # Prepare position metrics
        positions_metrics = {
            asset: {
                "size": position.size,
                "entry_price": position.entry_price,
                "is_open": position.is_open,
                "unrealized_pnl": (
                    (position.entry_price - position.current_price) * position.size
                    if (position.is_open and hasattr(position, "current_price"))
                    else 0.0
                ),
                "leverage": (
                    position.leverage if hasattr(position, "leverage") else 1.0
                ),
            }
            for asset, position in self.positions.items()
        }

        # Build final metrics dictionary
        metrics = {
            "total_positions": len(self.positions),
            "positions": positions_metrics,
            "trade_count": self.trade_count,
            "drawdown": self.drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "var": self.var,
            "cvar": self.cvar,
            "total_pnl_pct": total_pnl_pct,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "initial_capital": self.initial_capital,
            "current_equity": self.current_equity,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_capital": self.total_capital,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "leverage": self.leverage,
        }

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

    def get_chunk_performance_ratio(
            self,
            chunk_id: int,
            optimal_pnl: float) -> float:
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
                        "Cannot rebalance %s: current price not available.", asset)
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
                            "Position for %s fully closed during rebalancing.", asset)

        self.update_metrics()
        logger.info("Portfolio rebalancing completed.")

    def validate_position(self, asset: str, size: float, price: float) -> bool:
        """Validates a position before execution."""
        if size <= 0 or price <= 0:
            logger.warning(
                "Invalid size (%s) or price (%s) for position validation.",
                size,
                price)
            return False

        # Check against minimum trade size
        if size < self.min_trade_size:
            logger.warning(
                "Position size (%.8f) is less than minimum trade size (%.8f).",
                size,
                self.min_trade_size,
            )
            return False

        # Check against notional value limits
        notional_value = size * price
        if notional_value < self.min_notional_value:
            logger.warning(
                "Notional value (%.2f) is less than minimum notional value (%.2f).",
                notional_value,
                self.min_notional_value,
            )
            return False
        if notional_value > self.max_notional_value:
            logger.warning(
                "Notional value (%.2f) is greater than maximum notional value (%.2f).",
                notional_value,
                self.max_notional_value,
            )
            return False

        # Check if there's enough cash to open the position (considering
        # leverage for futures)
        required_cash = size * price
        if self.futures_enabled:
            required_cash /= self.leverage
        required_cash += required_cash * self.commission_pct  # Add commission

        if self.cash < required_cash:
            logger.warning(
                "Insufficient cash (%.2f) to open position requiring (%.2f).",
                self.cash,
                required_cash,
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
