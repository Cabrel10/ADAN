import os
import ccxt
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_open_positions():
    """
    Connects to Binance Testnet and checks for any open positions.
    """
    api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
    api_secret = os.environ.get('BINANCE_TESTNET_SECRET_KEY')

    if not api_key or not api_secret:
        logging.error("❌ Les variables d'environnement BINANCE_TESTNET_API_KEY et BINANCE_TESTNET_SECRET_KEY doivent être définies.")
        return

    logging.info("🔌 Connexion à Binance Testnet...")
    
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'options': {
            'defaultType': 'spot',
        },
    })
    exchange.set_sandbox_mode(True)

    try:
        # Test connection
        exchange.fetch_balance()
        logging.info("✅ Connexion réussie.")

        # Fetch open positions
        logging.info("🔍 Recherche de positions ouvertes...")
        positions = exchange.fetch_positions()
        
        # Filter for positions with a non-zero contract size
        open_positions = [p for p in positions if p.get('contracts') is not None and float(p['contracts']) != 0]

        if not open_positions:
            logging.info("✅ Aucune position ouverte détectée sur le compte.")
        else:
            logging.warning(f"⚠️ {len(open_positions)} position(s) ouverte(s) détectée(s) !")
            for pos in open_positions:
                # Use .get() for safe access
                symbol = pos.get('symbol', 'N/A')
                side = pos.get('side', 'N/A')
                contracts = pos.get('contracts', 0)
                entry_price = pos.get('entryPrice', 'N/A')
                unrealized_pnl = pos.get('unrealizedPnl', 'N/A')
                
                logging.info(f"  - Symbole: {symbol}, Côté: {side}, Taille: {contracts}, Prix d'entrée: {entry_price}, PnL non réalisé: {unrealized_pnl}")

    except ccxt.AuthenticationError as e:
        logging.error(f"❌ Erreur d'authentification: {e}. Vérifiez vos clés API.")
    except Exception as e:
        logging.error(f"❌ Une erreur est survenue: {e}")

if __name__ == "__main__":
    check_open_positions()
