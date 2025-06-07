"""
Order manager for the ADAN trading environment.
"""
import numpy as np
from ..common.utils import get_logger
from ..common.constants import (
    ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, ORDER_TYPE_STOP_LOSS, 
    ORDER_TYPE_TAKE_PROFIT, ORDER_TYPE_TRAILING_STOP, ORDER_TYPE_STOP_LIMIT,
    ORDER_TYPE_EXECUTED_LIMIT, ORDER_TYPE_EXECUTED_STOP_LOSS, 
    ORDER_TYPE_EXECUTED_TAKE_PROFIT, ORDER_TYPE_EXECUTED_TRAILING_STOP,
    ORDER_TYPE_EXECUTED_STOP_LIMIT,
    ORDER_STATUS_PENDING, ORDER_STATUS_EXECUTED, ORDER_STATUS_EXPIRED,
    INVALID_ORDER_TOO_SMALL, INVALID_ORDER_BELOW_TOLERABLE, 
    INVALID_NO_CAPITAL, INVALID_MAX_POSITIONS, INVALID_NO_POSITION,
    PENALTY_INVALID_ORDER, PENALTY_ORDER_TOO_SMALL, PENALTY_BELOW_TOLERABLE,
    PENALTY_NO_CAPITAL, PENALTY_MAX_POSITIONS, PENALTY_NO_POSITION,
    PENALTY_LIMIT_EXPIRY, PENALTY_STOP_LOSS_EXPIRY, PENALTY_TRAILING_STOP_EXPIRY,
    MIN_ORDER_VALUE_TOLERABLE, MIN_ORDER_VALUE_ABSOLUTE
)

logger = get_logger()

class OrderManager:
    """
    Manages order execution and pending orders in the trading environment.
    """
    
    def __init__(self, config, exchange_client=None):
        """
        Initialize the order manager.
        
        Args:
            config: Configuration dictionary.
            exchange_client: Optional CCXT exchange client for live/paper trading.
        """
        env_config = config.get('environment', {})
        
        # Store exchange client for live/paper trading
        self.exchange = exchange_client
        self.markets = None
        
        # Load market information if exchange is provided
        if self.exchange:
            try:
                self.markets = self.exchange.load_markets()
                logger.info(f"✅ OrderManager: Markets loaded for {self.exchange.id} ({len(self.markets)} pairs)")
            except Exception as e:
                logger.error(f"❌ OrderManager: Failed to load markets from {self.exchange.id}: {e}")
                self.exchange = None  # Fallback to simulation mode
        
        # Penalties configuration
        self.penalties_config = env_config.get('penalties', {})
        self.order_rules_config = env_config.get('order_rules', {})
        self.transaction_config = env_config.get('transaction', {})
        
        # Load penalties
        self.penalty_invalid_order_base = self.penalties_config.get('invalid_order_base', -0.5)
        self.penalty_order_below_tolerable = self.penalties_config.get('order_below_tolerable_if_not_adjusted', -0.2)
        self.penalty_order_expiry = self.penalties_config.get('order_expiry', -0.2)
        self.penalty_out_of_funds = self.penalties_config.get('out_of_funds', -1.0)
        
        # Load order rules
        self.min_order_value_tolerable = self.order_rules_config.get('min_value_tolerable', MIN_ORDER_VALUE_TOLERABLE)
        self.min_order_value_absolute = self.order_rules_config.get('min_value_absolute', MIN_ORDER_VALUE_ABSOLUTE)
        self.order_expiry_steps = self.order_rules_config.get('default_expiry_steps', 24)
        
        # Load transaction parameters
        self.fee_percent = self.transaction_config.get('fee_percent', 0.001)
        self.fixed_fee = self.transaction_config.get('fixed_fee', 0.0)
        
        # Initialize pending orders list
        self.pending_orders = []
        
        logger.info(f"OrderManager initialized with fee_percent={self.fee_percent}, "
                   f"fixed_fee={self.fixed_fee}, min_order_value_tolerable={self.min_order_value_tolerable}")
        
        if self.exchange:
            logger.info(f"🔗 Exchange integration enabled: {self.exchange.id}")
        else:
            logger.info("🔧 Simulation mode: No exchange client provided")
    
    def _convert_to_ccxt_symbol(self, asset_id):
        """
        Convert internal asset ID to CCXT symbol format.
        
        Args:
            asset_id: Internal asset ID (e.g., "ADAUSDT")
            
        Returns:
            str: CCXT symbol format (e.g., "ADA/USDT") or None if conversion fails
        """
        try:
            # Most crypto pairs end with USDT, BTC, ETH, etc.
            if asset_id.endswith('USDT'):
                base = asset_id[:-4]  # Remove 'USDT'
                return f"{base}/USDT"
            elif asset_id.endswith('BTC'):
                base = asset_id[:-3]  # Remove 'BTC'
                return f"{base}/BTC"
            elif asset_id.endswith('ETH'):
                base = asset_id[:-3]  # Remove 'ETH'
                return f"{base}/ETH"
            elif asset_id.endswith('USD'):
                base = asset_id[:-3]  # Remove 'USD'
                return f"{base}/USD"
            else:
                logger.warning(f"⚠️ Unknown quote currency for {asset_id}")
                return None
        except Exception as e:
            logger.error(f"❌ Error converting {asset_id} to CCXT symbol: {e}")
            return None
    
    def execute_order(self, asset_id, action_type, current_price, capital, positions, allocated_value_usdt=None, quantity=None, order_type=ORDER_TYPE_MARKET, current_step=0, **kwargs):
        """
        Execute an order.
        
        Args:
            asset_id: Asset ID.
            action_type: Order action type (BUY=1, SELL=2).
            current_price: Current price of the asset.
            capital: Available capital.
            positions: Current positions.
            allocated_value_usdt: Allocated value in USDT for BUY orders (replaces quantity).
            quantity: Quantity to trade for SELL orders. If None for SELL, will sell entire position.
            order_type: Order type (MARKET, LIMIT, etc.).
            current_step: Current step in the environment.
            
        Returns:
            tuple: (reward_modifier, status, info_dict)
        """
        action_name = "BUY" if action_type == 1 else "SELL"
        current_trade_penalty = 0.0
        
        # Exchange integration logic
        if self.exchange is not None:
            logger.info(f"🔗 Exchange mode: Validating {action_name} order for {asset_id}")
            
            # Convert asset_id to CCXT symbol (ex: "ADAUSDT" -> "ADA/USDT")
            symbol_ccxt = self._convert_to_ccxt_symbol(asset_id)
            if not symbol_ccxt:
                logger.error(f"❌ Cannot convert {asset_id} to CCXT symbol")
                return self.penalty_invalid_order_base, "INVALID_SYMBOL", {
                    "reason": f"Cannot convert {asset_id} to CCXT symbol",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
            
            # Get real market price from exchange
            try:
                ticker = self.exchange.fetch_ticker(symbol_ccxt)
                real_price = ticker['last']
                logger.info(f"📈 Real market price for {symbol_ccxt}: ${real_price:.6f}")
                
                # Use real price for calculations instead of normalized price
                price_for_calculations = real_price
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch real price for {symbol_ccxt}: {e}")
                # Fallback to provided price (could be normalized)
                price_for_calculations = abs(current_price)
                logger.warning(f"⚠️ Using fallback price: ${price_for_calculations:.6f}")
            
            # Get market limits and filters
            market_info = self.markets.get(symbol_ccxt)
            if not market_info:
                logger.error(f"❌ Market {symbol_ccxt} not found on {self.exchange.id}")
                return self.penalty_invalid_order_base, "MARKET_NOT_FOUND", {
                    "reason": f"Market {symbol_ccxt} not available",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
            
            limits = market_info.get('limits', {})
            amount_limits = limits.get('amount', {})
            cost_limits = limits.get('cost', {})
            precision = market_info.get('precision', {})
            
            # Calculate preliminary quantities for validation using REAL price
            if action_type == 1:  # BUY
                if allocated_value_usdt is None:
                    logger.error(f"❌ BUY {asset_id}: Missing allocated_value_usdt")
                    return self.penalty_invalid_order_base, "MISSING_ALLOCATION", {
                        "reason": "BUY order requires allocated_value_usdt",
                        "reward_mod": self.penalty_invalid_order_base,
                        "new_capital": capital
                    }
                
                quantity_approx = allocated_value_usdt / price_for_calculations
                value_of_trade = allocated_value_usdt
                
            else:  # SELL
                if asset_id not in positions or positions[asset_id]["qty"] <= 0:
                    logger.warning(f"❌ SELL {asset_id}: No position for exchange validation")
                    return self.penalty_invalid_order_base, INVALID_NO_POSITION, {
                        "reason": f"Cannot SELL {asset_id}: position does not exist",
                        "reward_mod": self.penalty_invalid_order_base,
                        "new_capital": capital
                    }
                
                available_qty = positions[asset_id]["qty"]
                quantity_to_trade = quantity if quantity is not None else available_qty
                quantity_approx = quantity_to_trade
                value_of_trade = quantity_to_trade * price_for_calculations
            
            # Validate against exchange filters
            min_amount = amount_limits.get('min', 0)
            max_amount = amount_limits.get('max', float('inf'))
            min_cost = cost_limits.get('min', 0)
            
            logger.debug(f"🔍 Exchange limits - Amount: [{min_amount}, {max_amount}], Cost: {min_cost}")
            
            # Check minimum amount
            if quantity_approx < min_amount:
                logger.warning(f"❌ {action_name} {asset_id}: Quantity {quantity_approx:.8f} < min {min_amount}")
                return self.penalty_invalid_order_base, "BELOW_MIN_AMOUNT", {
                    "reason": f"Quantity {quantity_approx:.8f} below exchange minimum {min_amount}",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
            
            # Check maximum amount
            if quantity_approx > max_amount:
                logger.warning(f"❌ {action_name} {asset_id}: Quantity {quantity_approx:.8f} > max {max_amount}")
                return self.penalty_invalid_order_base, "ABOVE_MAX_AMOUNT", {
                    "reason": f"Quantity {quantity_approx:.8f} above exchange maximum {max_amount}",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
            
            # Check minimum notional value
            if value_of_trade < min_cost:
                logger.warning(f"❌ {action_name} {asset_id}: Value ${value_of_trade:.2f} < min notional ${min_cost}")
                return self.penalty_invalid_order_base, "BELOW_MIN_NOTIONAL", {
                    "reason": f"Order value ${value_of_trade:.2f} below minimum notional ${min_cost}",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
            
            # Adjust quantity to exchange precision
            try:
                final_quantity = self.exchange.amount_to_precision(symbol_ccxt, quantity_approx)
                final_quantity = float(final_quantity)
                logger.debug(f"🔧 Precision adjusted: {quantity_approx:.8f} -> {final_quantity:.8f}")
            except Exception as e:
                logger.error(f"❌ Precision adjustment failed for {symbol_ccxt}: {e}")
                final_quantity = quantity_approx
            
            logger.info(f"✅ Exchange validation passed for {action_name} {asset_id}")
            
            # EXECUTE REAL ORDER ON EXCHANGE
            try:
                if action_type == 1:  # BUY
                    logger.info(f"📤 EXECUTING REAL BUY ORDER: {final_quantity:.6f} {symbol_ccxt} at ~${price_for_calculations:.6f}")
                    order_result = self.exchange.create_market_buy_order(symbol_ccxt, final_quantity)
                else:  # SELL
                    logger.info(f"📤 EXECUTING REAL SELL ORDER: {final_quantity:.6f} {symbol_ccxt} at ~${price_for_calculations:.6f}")
                    order_result = self.exchange.create_market_sell_order(symbol_ccxt, final_quantity)
                
                logger.info(f"✅ Order executed successfully: {order_result.get('id', 'N/A')}")
                logger.debug(f"📋 Order details: {order_result}")
                
                # Return special status for exchange orders
                return current_trade_penalty, "EXCHANGE_ORDER_EXECUTED", {
                    "asset_id": asset_id,
                    "symbol_ccxt": symbol_ccxt,
                    "quantity": final_quantity,
                    "price_real": price_for_calculations,
                    "order_id": order_result.get('id'),
                    "order_result": order_result,
                    "exchange_mode": True,
                    "reward_mod": current_trade_penalty,
                    "new_capital": capital  # Will be updated from exchange balance
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to execute order on exchange: {e}")
                # Fallback to simulation mode for this order
                logger.warning("⚠️ Falling back to simulation mode for this order")
                # Continue with normal simulation logic below
            
        else:
            logger.debug(f"🔧 Simulation mode: Processing {action_name} order for {asset_id}")
        
        # Check if we have the position for SELL
        if action_type == 2:  # SELL
            if asset_id not in positions or positions[asset_id]["qty"] <= 0:
                logger.warning(f"❌ SELL {asset_id}: No position")
                return self.penalty_invalid_order_base, INVALID_NO_POSITION, {
                    "reason": f"Cannot SELL {asset_id}: position does not exist",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
        
        # Determine quantity and order value
        if action_type == 2:  # SELL
            if quantity is None and asset_id in positions:
                quantity = positions[asset_id]["qty"]
            elif asset_id not in positions or positions[asset_id]["qty"] <= 0:
                return self.penalty_invalid_order_base, INVALID_NO_POSITION, {
                    "reason": f"Cannot SELL {asset_id}: position does not exist",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
            
            # Calculate order value for SELL
            quantity_to_sell = quantity
            # For threshold verification, use absolute price
            value_for_threshold = quantity_to_sell * abs(current_price)
            
        else:  # BUY
            if allocated_value_usdt is None:
                logger.warning("❌ BUY: Missing allocated_value_usdt")
                return self.penalty_invalid_order_base, INVALID_ORDER_TOO_SMALL, {
                    "reason": "BUY order requires allocated_value_usdt",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
        
            if allocated_value_usdt <= 0:
                logger.error(f"❌ BUY {asset_id}: Invalid allocation ${allocated_value_usdt:.2f}")
                return self.penalty_invalid_order_base, INVALID_ORDER_TOO_SMALL, {
                    "reason": f"Valeur allouée invalide: ${allocated_value_usdt:.2f}",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
        
            target_order_value_usdt = allocated_value_usdt
        
            if target_order_value_usdt < self.min_order_value_absolute:
                logger.warning(f"❌ BUY {asset_id}: ${target_order_value_usdt:.2f} < min ${self.min_order_value_absolute:.2f}")
                return self.penalty_invalid_order_base, INVALID_ORDER_TOO_SMALL, {
                    "reason": f"Order value {target_order_value_usdt:.2f} < absolute minimum {self.min_order_value_absolute}",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
        
            # Check minimum order value (tolerable threshold)
            elif target_order_value_usdt < self.min_order_value_tolerable:
                # Can we adjust the order value to min_order_value_tolerable?
                cost_for_min_tolerable = self.min_order_value_tolerable + self._calculate_fee(self.min_order_value_tolerable)
            
                if capital >= cost_for_min_tolerable:
                    logger.info(f"🔧 BUY {asset_id}: Auto-adjust ${target_order_value_usdt:.2f}→${self.min_order_value_tolerable:.2f}")
                    target_order_value_usdt = self.min_order_value_tolerable
                else:
                    logger.warning(f"⚠️ BUY {asset_id}: ${target_order_value_usdt:.2f} < tolerable, penalty applied")
                    current_trade_penalty = self.penalty_order_below_tolerable
                
                    # Calculate scaled penalty if configured
                    if 'order_below_tolerable_scaled' in self.penalties_config:
                        delta = self.min_order_value_tolerable - target_order_value_usdt
                        # Parse the formula from config, e.g., "delta * 0.1"
                        formula = self.penalties_config['order_below_tolerable_scaled']
                        if "delta" in formula:
                            try:
                                scaled_penalty = eval(formula.replace("delta", str(delta)))
                                current_trade_penalty = scaled_penalty
                            except Exception as e:
                                logger.error(f"❌ Penalty calc error: {e}")
        
            if abs(current_price) < 1e-9:
                logger.error(f"❌ {asset_id}: Price too small ${abs(current_price):.9f}")
                return self.penalty_invalid_order_base, "PRICE_NOT_AVAILABLE", {
                    "reason": f"Prix absolu trop faible (${abs(current_price):.9f}) pour {asset_id}",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
        
            quantity_to_buy = target_order_value_usdt / abs(current_price)
        
            if quantity_to_buy <= 1e-8:
                logger.warning(f"❌ BUY {asset_id}: Qty too small {quantity_to_buy:.8f}")
                return self.penalty_invalid_order_base, "INVALID_ORDER_TOO_SMALL", {
                    "reason": f"Quantité trop petite: {quantity_to_buy:.12f}",
                    "reward_mod": self.penalty_invalid_order_base,
                    "new_capital": capital
                }
        
            quantity = quantity_to_buy
            order_value_for_threshold = target_order_value_usdt
        
        # Calculate fee based on absolute order value (prix normalisés)
        if action_type == 1:  # BUY
            fee = self._calculate_fee(order_value_for_threshold)  # order_value_for_threshold est déjà positif
        else:  # SELL
            fee = self._calculate_fee(value_for_threshold)  # value_for_threshold utilise déjà abs(current_price)
        
        # For MARKET orders, execute immediately
        if order_type == ORDER_TYPE_MARKET:
            # Check if we have enough capital for BUY
            if action_type == 1:  # BUY
                total_cost = order_value_for_threshold + fee
                
                if total_cost > capital:
                    logger.warning(f"❌ BUY {asset_id}: Cost ${total_cost:.2f} > capital ${capital:.2f}")
                    return self.penalty_out_of_funds, INVALID_NO_CAPITAL, {
                        "reason": f"Coût total {total_cost:.2f} > capital disponible {capital:.2f}",
                        "reward_mod": self.penalty_out_of_funds,
                        "new_capital": capital
                    }
                
                # Execute BUY order
                new_capital = capital - total_cost
                
                if new_capital < 0:
                    logger.error(f"🚨 CRITICAL: Negative capital ${new_capital:.2f}")
                    return self.penalty_out_of_funds, INVALID_NO_CAPITAL, {
                        "reason": f"Capital insuffisant après calcul: ${new_capital:.2f}",
                        "reward_mod": self.penalty_out_of_funds,
                        "new_capital": capital
                    }
                
                # Update or create position
                if asset_id in positions:
                    # Update existing position (average down)
                    old_qty = positions[asset_id]["qty"]
                    old_price = positions[asset_id]["price"]
                    new_qty = old_qty + quantity
                    
                    if new_qty <= old_qty:
                        logger.error(f"🚨 BUY {asset_id}: Qty calc error")
                    
                    total_old_cost = old_qty * abs(old_price)
                    total_new_cost = quantity * abs(current_price)
                    weighted_avg_abs_price = (total_old_cost + total_new_cost) / new_qty
                    
                    positions[asset_id] = {"qty": new_qty, "price": current_price}
                    logger.info(f"📈 BUY {asset_id}: qty {old_qty:.3f}→{new_qty:.3f}, avg${weighted_avg_abs_price:.4f}")
                else:
                    if quantity <= 0:
                        logger.error(f"🚨 BUY {asset_id}: Invalid qty {quantity:.6f}")
                        return self.penalty_invalid_order_base, "INVALID_QUANTITY", {
                            "reason": f"Quantité invalide: {quantity:.6f}",
                            "reward_mod": self.penalty_invalid_order_base,
                            "new_capital": capital
                        }
                    positions[asset_id] = {"qty": quantity, "price": current_price}
                    logger.info(f"📈 NEW {asset_id}: qty={quantity:.6f}, price=${current_price:.6f}")
                
                if positions[asset_id]["qty"] <= 0:
                    logger.error(f"🚨 CRITICAL: Invalid position after BUY {asset_id}")
                    if asset_id in positions:
                        del positions[asset_id]
                    return self.penalty_invalid_order_base, "INVALID_POSITION_STATE", {
                        "reason": f"Position invalide après BUY: qty={positions[asset_id]['qty'] if asset_id in positions else 'N/A'}",
                        "reward_mod": self.penalty_invalid_order_base,
                        "new_capital": capital
                    }
                
                logger.info(f"✅ BUY {asset_id}: ${capital:.2f}→${new_capital:.2f}")
                return current_trade_penalty, "BUY_EXECUTED", {
                    "asset_id": asset_id,
                    "quantity": quantity,
                    "price": current_price,
                    "fee": fee,
                    "total_cost": total_cost,
                    "new_capital": new_capital,
                    "reward_mod": current_trade_penalty,
                    "status": "BUY_EXECUTED"
                }
            
            elif action_type == 2:  # SELL
                if asset_id not in positions or positions[asset_id]["qty"] <= 1e-8:
                    logger.warning(f"❌ SELL {asset_id}: No/insufficient position")
                    return self.penalty_invalid_order_base, INVALID_NO_POSITION, {
                        "reason": f"Cannot SELL {asset_id}: position does not exist or too small",
                        "reward_mod": self.penalty_invalid_order_base,
                        "new_capital": capital
                    }
                
                available_qty = positions[asset_id]["qty"]
                if quantity is None or quantity > available_qty:
                    quantity_to_sell = available_qty
                else:
                    quantity_to_sell = quantity
                
                if quantity_to_sell <= 1e-8:
                    logger.warning(f"❌ SELL {asset_id}: Qty too small {quantity_to_sell:.8f}")
                    return self.penalty_invalid_order_base, INVALID_NO_POSITION, {
                        "reason": f"Quantité à vendre trop petite: {quantity_to_sell:.8f}",
                        "reward_mod": self.penalty_invalid_order_base,
                        "new_capital": capital
                    }
                
                entry_price = positions[asset_id]["price"]
                
                gross_proceeds = quantity_to_sell * abs(current_price)
                net_proceeds = gross_proceeds - fee
                pnl = (current_price - entry_price) * quantity_to_sell - fee
                new_capital = capital + net_proceeds
                
                if new_capital < 0 and capital >= 0:
                    logger.error(f"🚨 CRITICAL: Negative capital after SELL ${capital:.2f}→${new_capital:.2f}")
                
                if quantity_to_sell >= available_qty - 1e-8:
                    logger.info(f"🔒 CLOSE {asset_id}: Complete position")
                    del positions[asset_id]
                else:
                    old_qty = positions[asset_id]["qty"]
                    positions[asset_id]["qty"] -= quantity_to_sell
                    
                    if positions[asset_id]["qty"] <= 1e-8:
                        logger.info(f"🧹 CLEAN {asset_id}: Residual position")
                        del positions[asset_id]
                
                logger.info(f"✅ SELL {asset_id}: qty={quantity_to_sell:.3f}, PnL=${pnl:.2f}, ${capital:.2f}→${new_capital:.2f}")
                
                return current_trade_penalty, "SELL_EXECUTED", {
                    "asset_id": asset_id,
                    "price": current_price,
                    "quantity": quantity_to_sell,
                    "value": gross_proceeds,
                    "fee": fee,
                    "pnl": pnl,
                    "proceeds": net_proceeds,
                    "order_type": order_type,
                    "reward_mod": current_trade_penalty,
                    "status": "SELL_EXECUTED",
                    "new_capital": new_capital
                }
        
        # For advanced orders, add to pending orders
        else:
            if action_type == 1:  # BUY
                total_cost = order_value_for_threshold + fee
                if total_cost > capital:
                    return self.penalty_out_of_funds, INVALID_NO_CAPITAL, {
                        "reason": f"Total cost {total_cost:.2f} > available capital {capital:.2f}",
                        "new_capital": capital
                    }
            # For SELL orders, check if we have the position
            elif action_type == 2:  # SELL
                if asset_id not in positions or positions[asset_id]["qty"] <= 0:
                    return self.penalty_invalid_order_base, INVALID_NO_POSITION, {
                        "reason": f"Cannot SELL {asset_id}: position does not exist",
                        "reward_mod": self.penalty_invalid_order_base,
                        "new_capital": capital
                    }
                
                if positions[asset_id]["qty"] < quantity:
                    quantity = positions[asset_id]["qty"]
                    value_for_threshold = quantity * abs(current_price)
            
            # Create pending order
            pending_order = {
                "asset_id": asset_id,
                "action_type": action_type,
                "quantity": quantity,
                "order_type": order_type,
                "created_step": current_step,
                "expiry_step": current_step + self.order_expiry_steps,
                "status": ORDER_STATUS_PENDING
            }
            
            # Add order-specific parameters
            if order_type == ORDER_TYPE_LIMIT:
                # For LIMIT orders, we need a limit price
                limit_price = kwargs.get("limit_price", current_price)
                pending_order["limit_price"] = limit_price
            
            elif order_type == ORDER_TYPE_STOP_LOSS:
                # For STOP_LOSS orders, we need a stop price
                stop_price = kwargs.get("stop_price", current_price * 0.95)
                pending_order["stop_price"] = stop_price
            
            elif order_type == ORDER_TYPE_TAKE_PROFIT:
                # For TAKE_PROFIT orders, we need a take profit price
                take_profit_price = kwargs.get("take_profit_price", current_price * 1.05)
                pending_order["take_profit_price"] = take_profit_price
            
            elif order_type == ORDER_TYPE_TRAILING_STOP:
                # For TRAILING_STOP orders, we need a trailing percentage
                trailing_pct = kwargs.get("trailing_pct", 0.05)
                pending_order["trailing_pct"] = trailing_pct
                pending_order["highest_price"] = current_price
                pending_order["stop_price"] = current_price * (1 - trailing_pct)
            
            elif order_type == ORDER_TYPE_STOP_LIMIT:
                # For STOP_LIMIT orders, we need both stop and limit prices
                stop_price = kwargs.get("stop_price", current_price * 0.95)
                limit_price = kwargs.get("limit_price", stop_price * 0.99)
                pending_order["stop_price"] = stop_price
                pending_order["limit_price"] = limit_price
            
            # Add the order to pending orders
            self.pending_orders.append(pending_order)
            
            return 0.0, f"{order_type}_CREATED", {
                "asset_id": asset_id,
                "quantity": quantity,
                "order_type": order_type,
                "expiry_step": pending_order["expiry_step"],
                **{k: v for k, v in pending_order.items() if k not in ["asset_id", "quantity", "order_type", "expiry_step"]}
            }
    
    def process_pending_orders(self, current_prices, capital, positions, current_step):
        """
        Process all pending orders.
        
        Args:
            current_prices: Dictionary of current prices for each asset.
            capital: Current available capital.
            positions: Dictionary of current positions.
            current_step: Current step in the environment.
            
        Returns:
            tuple: (total_reward_mod, executed_orders_info, new_capital, updated_positions)
        """
        total_reward_mod = 0.0
        executed_orders_info = []
        new_capital = capital
        updated_positions = positions.copy()
        
        # Process each pending order
        remaining_orders = []
        for order in self.pending_orders:
            asset_id = order["asset_id"]
            action_type = order["action_type"]
            quantity = order["quantity"]
            order_type = order["order_type"]
            
            # Skip if asset price is not available
            if asset_id not in current_prices:
                remaining_orders.append(order)
                continue
            
            current_price = current_prices[asset_id]
            
            # Check if order has expired
            if current_step >= order["expiry_step"]:
                # Apply penalty based on order type - utiliser la pénalité configurable pour l'expiration d'ordre
                # Toutes les expirations d'ordre utilisent la même pénalité configurable
                reward_mod = self.penalty_order_expiry
                
                total_reward_mod += reward_mod
                
                # Add to executed orders info
                executed_orders_info.append({
                    "asset_id": asset_id,
                    "order_type": order_type,
                    "status": ORDER_STATUS_EXPIRED,
                    "reward_mod": reward_mod,
                    "reason": "Order expired"
                })
                
                continue
            
            # Check if order conditions are met
            execute_order = False
            executed_order_type = order_type
            
            if order_type == ORDER_TYPE_LIMIT:
                # For BUY LIMIT, execute if price <= limit_price
                if action_type == 1 and current_price <= order["limit_price"]:
                    execute_order = True
                    executed_order_type = ORDER_TYPE_EXECUTED_LIMIT
                
                # For SELL LIMIT, execute if price >= limit_price
                elif action_type == 2 and current_price >= order["limit_price"]:
                    execute_order = True
                    executed_order_type = ORDER_TYPE_EXECUTED_LIMIT
            
            elif order_type == ORDER_TYPE_STOP_LOSS:
                # STOP_LOSS is for SELL only, execute if price <= stop_price
                if action_type == 2 and current_price <= order["stop_price"]:
                    execute_order = True
                    executed_order_type = ORDER_TYPE_EXECUTED_STOP_LOSS
            
            elif order_type == ORDER_TYPE_TAKE_PROFIT:
                # TAKE_PROFIT is for SELL only, execute if price >= take_profit_price
                if action_type == 2 and current_price >= order["take_profit_price"]:
                    execute_order = True
                    executed_order_type = ORDER_TYPE_EXECUTED_TAKE_PROFIT
            
            elif order_type == ORDER_TYPE_TRAILING_STOP:
                # Update highest price and stop price
                if current_price > order["highest_price"]:
                    order["highest_price"] = current_price
                    order["stop_price"] = current_price * (1 - order["trailing_pct"])
                
                # TRAILING_STOP is for SELL only, execute if price <= stop_price
                if action_type == 2 and current_price <= order["stop_price"]:
                    execute_order = True
                    executed_order_type = ORDER_TYPE_EXECUTED_TRAILING_STOP
            
            elif order_type == ORDER_TYPE_STOP_LIMIT:
                # STOP_LIMIT has two stages
                # 1. If price <= stop_price, convert to LIMIT order
                # 2. Then, if price >= limit_price, execute
                
                # For SELL STOP_LIMIT
                if action_type == 2:
                    if current_price <= order["stop_price"]:
                        # Convert to LIMIT
                        order["order_type"] = ORDER_TYPE_LIMIT
                        # Keep in pending orders
                        remaining_orders.append(order)
                        continue
            
            # If conditions are met, execute the order
            if execute_order:
                # Calculate order value
                order_value = quantity * current_price
                
                # Calculate fee
                fee = self._calculate_fee(order_value)
                
                # Execute BUY order
                if action_type == 1:
                    # Utiliser la valeur absolue pour le coût total
                    total_cost = abs(order_value) + fee
                    
                    # Check if we still have enough capital
                    if total_cost > new_capital:
                        # Not enough capital, keep order pending
                        remaining_orders.append(order)
                        continue
                    
                    # Update capital
                    new_capital = capital - total_cost
                    
                    # Add position - stocke le prix normalisé original comme prix d'entrée
                    if asset_id in updated_positions:
                        # Update existing position
                        old_qty = updated_positions[asset_id]["qty"]
                        old_price = updated_positions[asset_id]["price"]
                        new_qty = old_qty + quantity
                        
                        # Calculate new average price
                        new_price = (old_qty * old_price + quantity * current_price) / new_qty
                        updated_positions[asset_id] = {"qty": new_qty, "price": new_price, "value": abs(order_value)}
                    else:
                        # Create new position
                        updated_positions[asset_id] = {
                            "qty": quantity,
                            "price": current_price,  # Prix normalisé original (peut être négatif)
                            "value": abs(order_value)  # Valeur positive basée sur abs(price)
                        }
                    
                    # Add to executed orders info
                    executed_orders_info.append({
                        "asset_id": asset_id,
                        "price": current_price,
                        "quantity": quantity,
                        "value": order_value,
                        "fee": fee,
                        "total_cost": total_cost,
                        "order_type": executed_order_type,
                        "status": ORDER_STATUS_EXECUTED
                    })
                
                # Execute SELL order
                elif action_type == 2:
                    # Check if we still have the position
                    if asset_id not in updated_positions:
                        # No position, can't execute
                        continue
                    
                    # Check if we have enough quantity
                    if updated_positions[asset_id]["qty"] < quantity:
                        # Adjust quantity
                        quantity = updated_positions[asset_id]["qty"]
                        order_value = quantity * current_price
                        fee = self._calculate_fee(order_value)
                    
                    entry_price = positions[asset_id]["price"]
                    # Pour le calcul du PnL, utiliser les prix réels (normalisés)
                    # Le PnL est la différence entre le prix de vente et le prix d'achat, multiplié par la quantité, moins les frais
                    pnl = (current_price - entry_price) * quantity - fee
                    # Pour le calcul du capital, utiliser la valeur absolue du prix pour garantir un calcul cohérent
                    order_value_abs = quantity * abs(current_price)
                    new_capital = capital + order_value_abs - fee
                    
                    # Update position
                    if updated_positions[asset_id]["qty"] <= quantity:
                        # Close position completely
                        del updated_positions[asset_id]
                    else:
                        # Reduce position
                        updated_positions[asset_id]["qty"] -= quantity
                    
                    # Add to executed orders info
                    executed_orders_info.append({
                        "asset_id": asset_id,
                        "price": current_price,
                        "quantity": quantity,
                        "value": order_value,
                        "fee": fee,
                        "pnl": pnl,
                        "order_type": executed_order_type,
                        "status": ORDER_STATUS_EXECUTED
                    })
                    
                    # Add bonus for take profit orders with positive PnL
                    if executed_order_type == ORDER_TYPE_EXECUTED_TAKE_PROFIT and pnl > 0:
                        # Bonus of 1% of PnL, capped at 1.0
                        bonus = min(pnl * 0.01, 1.0)
                        total_reward_mod += bonus
                        executed_orders_info[-1]["bonus"] = bonus
            
            else:
                # Conditions not met, keep order pending
                remaining_orders.append(order)
        
        # Update pending orders
        self.pending_orders = remaining_orders
        
        return total_reward_mod, executed_orders_info, new_capital, updated_positions
    
    def _calculate_fee(self, amount):
        """
        Calculate the transaction fee.
        
        Args:
            amount: Transaction amount (should be absolute value).
            
        Returns:
            float: Fee amount.
        """
        return abs(amount) * self.fee_percent + self.fixed_fee
    
    def _get_position_size(self, asset_id, price, capital, tier=None):
        """
        Calculate the position size based on allocation.
        
        Args:
            asset_id: Asset ID.
            price: Current price.
            capital: Available capital.
            tier: Current tier (optional).
            
        Returns:
            float: Position size (quantity).
        """
        # This is a simplified version
        # In the real implementation, this would use tier allocation
        allocation_frac = 0.2  # Default allocation fraction
        
        if tier is not None:
            # Utiliser la clé correcte pour l'allocation
            allocation_frac = tier.get("allocation_frac_per_pos", 0.2)
            logger.info(f"[OrderManager._get_position_size] Using tier allocation: {allocation_frac:.4f}")
        else:
            logger.info(f"[OrderManager._get_position_size] Using default allocation: {allocation_frac:.4f}")
        
        # Calculate allocation amount
        allocation = capital * allocation_frac
        logger.info(f"[OrderManager._get_position_size] Capital: ${capital:.2f}, allocation: ${allocation:.2f}")
        logger.info(f"[OrderManager._get_position_size] Asset: {asset_id}, price: ${price:.4f}")
        
        # Calculate quantity based on allocation
        quantity = allocation / price
        logger.info(f"[OrderManager._get_position_size] Final quantity for {asset_id}: {quantity:.6f} (value: ${quantity * price:.2f})")
        
        return quantity
    
    def clear_pending_orders(self):
        """
        Clear all pending orders.
        """
        self.pending_orders = []
