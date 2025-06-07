import ccxt
import os
import json

# Lire les clés API depuis les variables d'environnement
api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
secret_key = os.environ.get("BINANCE_TESTNET_SECRET_KEY")

if not api_key or not secret_key:
    print("ERREUR: Veuillez définir les variables d'environnement BINANCE_TESTNET_API_KEY et BINANCE_TESTNET_SECRET_KEY")
    print("Exemple : export BINANCE_TESTNET_API_KEY='VotreCle'")
    exit()

print(f"Clé API Testnet chargée: {api_key[:5]}...") # Affiche seulement les 5 premiers caractères pour confirmation

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
    'options': {
        'defaultType': 'spot',
    },
    # CCXT gère les URLs du testnet avec set_sandbox_mode(True)
})
exchange.set_sandbox_mode(True) # Active le mode Testnet pour Binance

print(f"Connecté à {exchange.id} en mode Testnet.")

print("\n--- Tentative de chargement des marchés ---")
try:
    markets = exchange.load_markets()
    print(f"Nombre de marchés Testnet chargés: {len(markets)}")
    if 'BTC/USDT' in markets:
        print("Marché BTC/USDT trouvé sur Testnet.")
        # Afficher les limites pour BTC/USDT si elles existent
        if markets['BTC/USDT'].get('limits'):
            print(f"Limites pour BTC/USDT: {json.dumps(markets['BTC/USDT']['limits'], indent=2)}")
        else:
            print("Pas d'informations de limites détaillées pour BTC/USDT dans la réponse.")
    else:
        print("Marché BTC/USDT NON trouvé sur le Testnet.")

    print("\n--- Tentative de récupération du solde Testnet ---")
    balance = exchange.fetch_balance()
    print("Solde sur le Testnet (monnaies avec solde > 0):")
    testnet_funds_found = False
    for currency, amount_info in balance['total'].items():
        if amount_info > 0:
            print(f"{currency}: {amount_info}")
            testnet_funds_found = True
    if not testnet_funds_found:
        print("Aucun fonds Testnet trouvé ou solde nul pour toutes les monnaies affichables.")
        print("Note: Le Testnet Binance fournit automatiquement des fonds virtuels.")
    
    print("\n--- Tentative de récupération des dernières bougies pour BTC/USDT (1m) ---")
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=5)
    print("Dernières 5 bougies 1m pour BTC/USDT:")
    for candle in ohlcv:
        print(f"{exchange.iso8601(candle[0])} - O: {candle[1]}, H: {candle[2]}, L: {candle[3]}, C: {candle[4]}, V: {candle[5]}")

    # ----- Section pour tester un ordre (À ACTIVER AVEC PRUDENCE) -----
    # print("\n--- Tentative de placer un ordre MARKET BUY sur BTC/USDT (Testnet) ---")
    # symbol_to_trade = 'BTC/USDT'
    # order_type = 'market'
    # side = 'buy'
    
    # # Déterminer une quantité en fonction des filtres et du solde USDT (si disponible)
    # amount_to_buy = 0.0 # À initialiser
    # min_qty = markets.get(symbol_to_trade, {}).get('limits', {}).get('amount', {}).get('min')
    # min_notional = markets.get(symbol_to_trade, {}).get('limits', {}).get('cost', {}).get('min')
    # usdt_balance = balance.get('free', {}).get('USDT', 0)

    # if min_qty is not None and min_notional is not None and usdt_balance > 0:
    #     # Acheter pour un peu plus que le minimum notionnel
    #     # Obtenir le prix actuel pour estimer la quantité
    #     ticker = exchange.fetch_ticker(symbol_to_trade)
    #     current_price = ticker['last']
    #     if current_price > 0:
    #         qty_for_min_notional = (min_notional * 1.1) / current_price # 10% de plus que le minNotional
    #         amount_to_buy = max(min_qty, qty_for_min_notional)
            
    #         # S'assurer de ne pas dépasser une petite fraction du solde USDT
    #         amount_to_buy = min(amount_to_buy, usdt_balance * 0.01 / current_price, 0.001) # Limiter à 1% du solde ou 0.001 BTC
            
    #         print(f"Calcul de la quantité à acheter: {amount_to_buy} {symbol_to_trade.split('/')[0]}")
    #         print(f"  Basé sur: minQty={min_qty}, minNotional={min_notional}, solde USDT={usdt_balance}, prix actuel={current_price}")

    #         if amount_to_buy >= (min_qty or 0):
    #             try:
    #                 print(f"Passage de l'ordre: {side} {amount_to_buy} {symbol_to_trade}")
    #                 # order = exchange.create_order(symbol_to_trade, order_type, side, amount_to_buy)
    #                 # print("Réponse de l'ordre:")
    #                 # print(json.dumps(order, indent=2))
    #                 print("NOTE: La création d'ordre est commentée pour éviter des actions non désirées. Décommentez pour tester.")
    #             except Exception as e:
    #                 print(f"Erreur lors de la création de l'ordre: {e}")
    #         else:
    #             print(f"Quantité calculée {amount_to_buy} est inférieure au minQty {min_qty} ou les fonds sont insuffisants.")
    #     else:
    #         print("Impossible de récupérer le prix actuel pour calculer la quantité.")
    # else:
    #     print("Impossible de déterminer les quantités minimales ou solde USDT nul.")
        
except ccxt.NetworkError as e:
    print(f"Erreur réseau CCXT: {e}")
except ccxt.ExchangeError as e:
    print(f"Erreur d'exchange CCXT: {e}")
except Exception as e:
    print(f"Autre erreur: {e}")
    import traceback
    traceback.print_exc()