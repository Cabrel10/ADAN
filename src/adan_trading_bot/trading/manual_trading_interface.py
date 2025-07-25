"""
Interface de trading manuel pour ADAN Trading Bot.
Implémente les tâches 10B.2.3, 10B.2.4.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import uuid
from decimal import Decimal, ROUND_DOWN

from .secure_api_manager import SecureAPIManager, ExchangeType, APICredentials

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types d'ordres"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Côtés d'ordre"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """États des ordres"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiskOverrideType(Enum):
    """Types d'override de risque"""
    FORCE_DEFENSIVE = "force_defensive"
    FORCE_AGGRESSIVE = "force_aggressive"
    DISABLE_DBE = "disable_dbe"
    CUSTOM_PARAMS = "custom_params"


@dataclass
class ManualOrder:
    """Ordre manuel"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    
    # Métadonnées
    exchange: Optional[ExchangeType] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Exécution
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    commission_asset: Optional[str] = None
    
    # Tracking
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.client_order_id is None:
            self.client_order_id = f"ADAN_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        if self.exchange:
            data['exchange'] = self.exchange.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.submitted_at:
            data['submitted_at'] = self.submitted_at.isoformat()
        if self.filled_at:
            data['filled_at'] = self.filled_at.isoformat()
        return data


@dataclass
class RiskOverride:
    """Override de risque"""
    override_id: str
    override_type: RiskOverrideType
    parameters: Dict[str, Any]
    reason: str
    created_by: str = "manual"
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None and self.override_type != RiskOverrideType.CUSTOM_PARAMS:
            # Override temporaire par défaut (1 heure)
            self.expires_at = datetime.now() + timedelta(hours=1)
    
    def is_expired(self) -> bool:
        """Vérifie si l'override a expiré"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['override_type'] = self.override_type.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data


class ManualTradingInterface:
    """Interface de trading manuel"""
    
    def __init__(self, api_manager: SecureAPIManager):
        self.api_manager = api_manager
        
        # Stockage des ordres
        self.orders: Dict[str, ManualOrder] = {}
        self.order_history: List[ManualOrder] = []
        
        # Gestion des overrides de risque
        self.risk_overrides: Dict[str, RiskOverride] = {}
        self.override_history: List[RiskOverride] = []
        
        # Callbacks
        self.order_callbacks: List[Callable] = []
        self.risk_override_callbacks: List[Callable] = []
        
        # Configuration
        self.default_exchange = ExchangeType.BINANCE
        self.confirmation_required = True
        self.max_order_value_usd = 10000  # Limite de sécurité
        
        # Threading
        self.order_monitor_thread = None
        self.stop_monitoring = False
        
        logger.info("ManualTradingInterface initialized")
    
    def set_default_exchange(self, exchange: ExchangeType) -> None:
        """Définit l'exchange par défaut"""
        self.default_exchange = exchange
        logger.info(f"Default exchange set to {exchange.value}")
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float,
                          exchange: Optional[ExchangeType] = None) -> str:
        """
        Crée un ordre au marché.
        
        Args:
            symbol: Symbole de trading (ex: BTCUSDT)
            side: Côté de l'ordre (BUY/SELL)
            quantity: Quantité
            exchange: Exchange à utiliser
            
        Returns:
            ID de l'ordre
        """
        order = ManualOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            exchange=exchange or self.default_exchange
        )
        
        return self._process_order(order)
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float,
                          exchange: Optional[ExchangeType] = None) -> str:
        """
        Crée un ordre limite.
        
        Args:
            symbol: Symbole de trading
            side: Côté de l'ordre
            quantity: Quantité
            price: Prix limite
            exchange: Exchange à utiliser
            
        Returns:
            ID de l'ordre
        """
        order = ManualOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            exchange=exchange or self.default_exchange
        )
        
        return self._process_order(order)
    
    def create_stop_loss_order(self, symbol: str, side: OrderSide, quantity: float, 
                              stop_price: float, limit_price: Optional[float] = None,
                              exchange: Optional[ExchangeType] = None) -> str:
        """
        Crée un ordre stop-loss.
        
        Args:
            symbol: Symbole de trading
            side: Côté de l'ordre
            quantity: Quantité
            stop_price: Prix de déclenchement
            limit_price: Prix limite (si None, ordre stop-market)
            exchange: Exchange à utiliser
            
        Returns:
            ID de l'ordre
        """
        order_type = OrderType.STOP_LIMIT if limit_price else OrderType.STOP_LOSS
        
        order = ManualOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            exchange=exchange or self.default_exchange
        )
        
        return self._process_order(order)
    
    def _process_order(self, order: ManualOrder) -> str:
        """Traite un ordre"""
        try:
            # Validation de base
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                order.error_message = "Order validation failed"
                self.orders[order.order_id] = order
                return order.order_id
            
            # Vérification des credentials
            credentials = self.api_manager.get_credentials(order.exchange)
            if not credentials:
                order.status = OrderStatus.REJECTED
                order.error_message = f"No credentials for {order.exchange.value}"
                self.orders[order.order_id] = order
                return order.order_id
            
            # Confirmation si requise
            if self.confirmation_required:
                order.status = OrderStatus.PENDING
                self.orders[order.order_id] = order
                self._notify_order_update(order)
                logger.info(f"Order {order.order_id} pending confirmation")
                return order.order_id
            
            # Soumettre directement
            return self._submit_order(order)
            
        except Exception as e:
            logger.error(f"Error processing order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.orders[order.order_id] = order
            return order.order_id
    
    def _validate_order(self, order: ManualOrder) -> bool:
        """Valide un ordre"""
        try:
            # Vérifications de base
            if order.quantity <= 0:
                logger.error("Invalid quantity")
                return False
            
            if order.price is not None and order.price <= 0:
                logger.error("Invalid price")
                return False
            
            if order.stop_price is not None and order.stop_price <= 0:
                logger.error("Invalid stop price")
                return False
            
            # Vérification de la valeur maximale (approximative)
            if order.price:
                estimated_value = order.quantity * order.price
                if estimated_value > self.max_order_value_usd:
                    logger.error(f"Order value too high: ${estimated_value}")
                    return False
            
            # Vérifications spécifiques au type d'ordre
            if order.order_type == OrderType.LIMIT and order.price is None:
                logger.error("Limit order requires price")
                return False
            
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and order.stop_price is None:
                logger.error("Stop order requires stop price")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False
    
    def confirm_order(self, order_id: str) -> bool:
        """Confirme un ordre en attente"""
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            logger.error(f"Order {order_id} not pending")
            return False
        
        return self._submit_order(order) == order_id
    
    def _submit_order(self, order: ManualOrder) -> str:
        """Soumet un ordre à l'exchange"""
        try:
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            self.orders[order.order_id] = order
            
            # Simuler la soumission (en production, utiliser l'API réelle)
            success = self._simulate_order_submission(order)
            
            if success:
                logger.info(f"Order {order.order_id} submitted successfully")
                self._notify_order_update(order)
                
                # Démarrer le monitoring si pas déjà actif
                if not self.order_monitor_thread or not self.order_monitor_thread.is_alive():
                    self._start_order_monitoring()
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = "Submission failed"
                logger.error(f"Order {order.order_id} submission failed")
            
            return order.order_id
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return order.order_id
    
    def _simulate_order_submission(self, order: ManualOrder) -> bool:
        """Simule la soumission d'ordre (pour tests)"""
        # En production, remplacer par l'appel API réel
        order.exchange_order_id = f"EX_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Simuler un délai
        time.sleep(0.1)
        
        # Simuler succès/échec (95% de succès)
        import random
        return random.random() > 0.05
    
    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre"""
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            logger.error(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False
        
        try:
            # Annuler sur l'exchange si soumis
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                success = self._cancel_order_on_exchange(order)
                if not success:
                    return False
            
            # Mettre à jour le statut
            order.status = OrderStatus.CANCELLED
            self._notify_order_update(order)
            
            logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def _cancel_order_on_exchange(self, order: ManualOrder) -> bool:
        """Annule un ordre sur l'exchange"""
        # En production, utiliser l'API réelle
        logger.info(f"Cancelling order {order.exchange_order_id} on {order.exchange.value}")
        time.sleep(0.1)  # Simuler délai
        return True
    
    def _start_order_monitoring(self) -> None:
        """Démarre le monitoring des ordres"""
        if self.order_monitor_thread and self.order_monitor_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.order_monitor_thread = threading.Thread(
            target=self._monitor_orders,
            daemon=True
        )
        self.order_monitor_thread.start()
        logger.info("Order monitoring started")
    
    def _monitor_orders(self) -> None:
        """Surveille les ordres actifs"""
        while not self.stop_monitoring:
            try:
                active_orders = [
                    order for order in self.orders.values()
                    if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
                ]
                
                for order in active_orders:
                    self._check_order_status(order)
                
                time.sleep(1)  # Vérifier toutes les secondes
                
            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                time.sleep(5)
    
    def _check_order_status(self, order: ManualOrder) -> None:
        """Vérifie le statut d'un ordre"""
        # En production, interroger l'API de l'exchange
        # Pour la simulation, on va simuler des remplissages aléatoires
        
        import random
        
        # Simuler remplissage progressif pour les ordres limite
        if order.order_type == OrderType.LIMIT and random.random() < 0.1:  # 10% de chance par seconde
            if order.filled_quantity < order.quantity:
                fill_amount = min(order.quantity - order.filled_quantity, order.quantity * 0.3)
                order.filled_quantity += fill_amount
                
                if order.filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now()
                    order.average_price = order.price
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                
                self._notify_order_update(order)
        
        # Simuler remplissage immédiat pour les ordres market
        elif order.order_type == OrderType.MARKET and order.filled_quantity == 0:
            order.filled_quantity = order.quantity
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.average_price = order.price or 50000  # Prix simulé
            self._notify_order_update(order)
    
    def get_order(self, order_id: str) -> Optional[ManualOrder]:
        """Récupère un ordre"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[ManualOrder]:
        """Récupère les ordres actifs"""
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
    
    def get_order_history(self, limit: int = 100) -> List[ManualOrder]:
        """Récupère l'historique des ordres"""
        all_orders = list(self.orders.values()) + self.order_history
        all_orders.sort(key=lambda x: x.created_at, reverse=True)
        return all_orders[:limit]
    
    # ==================== RISK OVERRIDE MANAGEMENT ====================
    
    def create_risk_override(self, override_type: RiskOverrideType, parameters: Dict[str, Any],
                           reason: str, duration_hours: Optional[int] = None) -> str:
        """
        Crée un override de risque.
        
        Args:
            override_type: Type d'override
            parameters: Paramètres spécifiques
            reason: Raison de l'override
            duration_hours: Durée en heures (None = permanent)
            
        Returns:
            ID de l'override
        """
        override_id = str(uuid.uuid4())
        
        expires_at = None
        if duration_hours:
            expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        override = RiskOverride(
            override_id=override_id,
            override_type=override_type,
            parameters=parameters,
            reason=reason,
            expires_at=expires_at
        )
        
        self.risk_overrides[override_id] = override
        self._notify_risk_override_update(override)
        
        logger.info(f"Risk override created: {override_type.value} - {reason}")
        return override_id
    
    def force_defensive_mode(self, reason: str, duration_hours: int = 1) -> str:
        """Force le mode défensif"""
        return self.create_risk_override(
            RiskOverrideType.FORCE_DEFENSIVE,
            {'mode': 'defensive', 'risk_multiplier': 0.5},
            reason,
            duration_hours
        )
    
    def force_aggressive_mode(self, reason: str, duration_hours: int = 1) -> str:
        """Force le mode agressif"""
        return self.create_risk_override(
            RiskOverrideType.FORCE_AGGRESSIVE,
            {'mode': 'aggressive', 'risk_multiplier': 1.5},
            reason,
            duration_hours
        )
    
    def disable_dbe(self, reason: str, duration_hours: int = 1) -> str:
        """Désactive temporairement le DBE"""
        return self.create_risk_override(
            RiskOverrideType.DISABLE_DBE,
            {'dbe_enabled': False},
            reason,
            duration_hours
        )
    
    def set_custom_risk_params(self, params: Dict[str, Any], reason: str) -> str:
        """Définit des paramètres de risque personnalisés"""
        return self.create_risk_override(
            RiskOverrideType.CUSTOM_PARAMS,
            params,
            reason,
            None  # Permanent jusqu'à révocation manuelle
        )
    
    def revoke_risk_override(self, override_id: str) -> bool:
        """Révoque un override de risque"""
        if override_id not in self.risk_overrides:
            logger.error(f"Risk override {override_id} not found")
            return False
        
        override = self.risk_overrides[override_id]
        override.active = False
        
        # Déplacer vers l'historique
        self.override_history.append(override)
        del self.risk_overrides[override_id]
        
        self._notify_risk_override_update(override)
        
        logger.info(f"Risk override {override_id} revoked")
        return True
    
    def get_active_risk_overrides(self) -> List[RiskOverride]:
        """Récupère les overrides actifs"""
        # Nettoyer les overrides expirés
        expired_ids = []
        for override_id, override in self.risk_overrides.items():
            if override.is_expired():
                expired_ids.append(override_id)
        
        for override_id in expired_ids:
            self.revoke_risk_override(override_id)
        
        return list(self.risk_overrides.values())
    
    def get_risk_override_history(self, limit: int = 50) -> List[RiskOverride]:
        """Récupère l'historique des overrides"""
        all_overrides = list(self.risk_overrides.values()) + self.override_history
        all_overrides.sort(key=lambda x: x.created_at, reverse=True)
        return all_overrides[:limit]
    
    # ==================== CALLBACKS ====================
    
    def add_order_callback(self, callback: Callable[[ManualOrder], None]) -> None:
        """Ajoute un callback pour les mises à jour d'ordres"""
        self.order_callbacks.append(callback)
    
    def add_risk_override_callback(self, callback: Callable[[RiskOverride], None]) -> None:
        """Ajoute un callback pour les overrides de risque"""
        self.risk_override_callbacks.append(callback)
    
    def _notify_order_update(self, order: ManualOrder) -> None:
        """Notifie les callbacks des mises à jour d'ordres"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    def _notify_risk_override_update(self, override: RiskOverride) -> None:
        """Notifie les callbacks des overrides de risque"""
        for callback in self.risk_override_callbacks:
            try:
                callback(override)
            except Exception as e:
                logger.error(f"Error in risk override callback: {e}")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Récupère un résumé de l'activité de trading"""
        active_orders = self.get_active_orders()
        recent_orders = [o for o in self.orders.values() if o.created_at > datetime.now() - timedelta(hours=24)]
        active_overrides = self.get_active_risk_overrides()
        
        return {
            'active_orders_count': len(active_orders),
            'recent_orders_count': len(recent_orders),
            'active_risk_overrides_count': len(active_overrides),
            'total_orders': len(self.orders),
            'order_success_rate': self._calculate_success_rate(),
            'active_overrides': [o.to_dict() for o in active_overrides]
        }
    
    def _calculate_success_rate(self) -> float:
        """Calcule le taux de succès des ordres"""
        if not self.orders:
            return 0.0
        
        completed_orders = [
            o for o in self.orders.values()
            if o.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
        ]
        
        if not completed_orders:
            return 0.0
        
        successful_orders = [o for o in completed_orders if o.status == OrderStatus.FILLED]
        return len(successful_orders) / len(completed_orders)
    
    def shutdown(self) -> None:
        """Arrêt propre de l'interface"""
        logger.info("Shutting down ManualTradingInterface...")
        
        # Arrêter le monitoring
        self.stop_monitoring = True
        if self.order_monitor_thread and self.order_monitor_thread.is_alive():
            self.order_monitor_thread.join(timeout=5.0)
        
        # Sauvegarder l'historique si nécessaire
        # (implémentation selon les besoins)
        
        logger.info("ManualTradingInterface shutdown completed")