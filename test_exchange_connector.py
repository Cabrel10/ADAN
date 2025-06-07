"""
Script de test pour le connecteur d'exchange ADAN.
Teste la fonction get_exchange_client avec la configuration Binance Testnet.
"""

import os
import sys
import traceback
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH pour les imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.adan_trading_bot.common.utils import load_config, get_path
from src.adan_trading_bot.exchange_api.connector import (
    get_exchange_client, 
    test_exchange_connection, 
    validate_exchange_config,
    ExchangeConnectionError,
    ExchangeConfigurationError
)
import ccxt

def print_separator(title):
    """Affiche un séparateur avec titre."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_environment_variables():
    """Vérifie la présence des variables d'environnement nécessaires."""
    print_separator("VÉRIFICATION DES VARIABLES D'ENVIRONNEMENT")
    
    required_vars = ['BINANCE_TESTNET_API_KEY', 'BINANCE_TESTNET_SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {'*' * 10}{value[-5:]} (masqué)")
        else:
            print(f"❌ {var}: NON DÉFINIE")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️ Variables manquantes: {', '.join(missing_vars)}")
        print("\nPour les définir, utilisez:")
        for var in missing_vars:
            print(f'export {var}="VOTRE_VALEUR"')
        return False
    
    print("\n✅ Toutes les variables d'environnement sont définies")
    return True

def load_test_config():
    """Charge et prépare la configuration de test."""
    print_separator("CHARGEMENT DE LA CONFIGURATION")
    
    try:
        # Charger la configuration principale
        main_config_path = project_root / "config" / "main_config.yaml"
        print(f"Chargement de: {main_config_path}")
        
        main_config = load_config(str(main_config_path))
        print("✅ Configuration principale chargée")
        
        # Vérifier et afficher la section paper_trading
        paper_config = main_config.get('paper_trading', {})
        if paper_config:
            print(f"✅ Section paper_trading trouvée:")
            print(f"   - exchange_id: {paper_config.get('exchange_id')}")
            print(f"   - use_testnet: {paper_config.get('use_testnet')}")
        else:
            print("⚠️ Section paper_trading non trouvée, utilisation des valeurs par défaut")
            main_config['paper_trading'] = {
                'exchange_id': 'binance',
                'use_testnet': True
            }
        
        return main_config
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la configuration: {e}")
        print("\nUtilisation d'une configuration de test par défaut...")
        return {
            'paper_trading': {
                'exchange_id': 'binance',
                'use_testnet': True
            }
        }

def test_config_validation(config):
    """Teste la validation de la configuration."""
    print_separator("VALIDATION DE LA CONFIGURATION")
    
    try:
        is_valid = validate_exchange_config(config)
        if is_valid:
            print("✅ Configuration validée avec succès")
        else:
            print("❌ Configuration invalide")
        return is_valid
    except Exception as e:
        print(f"❌ Erreur lors de la validation: {e}")
        return False

def test_exchange_client_creation(config):
    """Teste la création du client d'exchange."""
    print_separator("CRÉATION DU CLIENT D'EXCHANGE")
    
    try:
        print("Initialisation du client CCXT...")
        exchange_client = get_exchange_client(config)
        
        print(f"✅ Client créé avec succès:")
        print(f"   - Exchange ID: {exchange_client.id}")
        print(f"   - Mode Sandbox: {getattr(exchange_client, 'sandbox', 'Non défini')}")
        print(f"   - URLs: {exchange_client.urls}")
        
        return exchange_client
        
    except ExchangeConfigurationError as e:
        print(f"❌ Erreur de configuration: {e}")
        return None
    except ExchangeConnectionError as e:
        print(f"❌ Erreur de connexion: {e}")
        return None
    except ValueError as e:
        print(f"❌ Erreur de valeur: {e}")
        return None
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        traceback.print_exc()
        return None

def test_exchange_connection_full(exchange_client):
    """Teste la connexion complète à l'exchange."""
    print_separator("TEST DE CONNEXION COMPLET")
    
    try:
        results = test_exchange_connection(exchange_client)
        
        print(f"\nRésultats des tests:")
        print(f"   - Exchange: {results['exchange_id']}")
        print(f"   - Mode Testnet: {results['testnet_mode']}")
        print(f"   - Heure serveur: {results['server_time']}")
        print(f"   - Marchés chargés: {results['markets_loaded']}")
        print(f"   - Nombre de marchés: {results['market_count']}")
        print(f"   - Solde accessible: {results['balance_accessible']}")
        print(f"   - Erreurs: {len(results['errors'])}")
        
        if results['errors']:
            print("\nErreurs détectées:")
            for error in results['errors']:
                print(f"   - {error}")
        
        return len(results['errors']) == 0
        
    except Exception as e:
        print(f"❌ Erreur lors du test de connexion: {e}")
        traceback.print_exc()
        return False

def test_market_data_retrieval(exchange_client):
    """Teste la récupération de données de marché."""
    print_separator("TEST DE RÉCUPÉRATION DES DONNÉES DE MARCHÉ")
    
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    
    for symbol in test_symbols:
        try:
            print(f"\nTest du symbole: {symbol}")
            
            # Test ticker
            ticker = exchange_client.fetch_ticker(symbol)
            print(f"   ✅ Ticker: Prix = {ticker['last']}, Volume = {ticker['baseVolume']}")
            
            # Test OHLCV (dernières 5 bougies)
            ohlcv = exchange_client.fetch_ohlcv(symbol, '1m', limit=5)
            print(f"   ✅ OHLCV: {len(ohlcv)} bougies récupérées")
            if ohlcv:
                last_candle = ohlcv[-1]
                print(f"      Dernière bougie: O={last_candle[1]}, H={last_candle[2]}, L={last_candle[3]}, C={last_candle[4]}")
            
        except Exception as e:
            print(f"   ❌ Erreur pour {symbol}: {e}")

def main():
    """Fonction principale du test."""
    print_separator("TEST DU CONNECTEUR D'EXCHANGE ADAN")
    print("Ce script teste la connexion au Binance Testnet via CCXT")
    
    success = True
    
    # Étape 1: Vérifier les variables d'environnement
    if not check_environment_variables():
        print("\n❌ Test arrêté: Variables d'environnement manquantes")
        return False
    
    # Étape 2: Charger la configuration
    config = load_test_config()
    
    # Étape 3: Valider la configuration
    if not test_config_validation(config):
        print("\n❌ Test arrêté: Configuration invalide")
        return False
    
    # Étape 4: Créer le client d'exchange
    exchange_client = test_exchange_client_creation(config)
    if not exchange_client:
        print("\n❌ Test arrêté: Impossible de créer le client d'exchange")
        return False
    
    # Étape 5: Tester la connexion complète
    if not test_exchange_connection_full(exchange_client):
        print("\n⚠️ Certains tests de connexion ont échoué")
        success = False
    
    # Étape 6: Tester la récupération de données de marché
    try:
        test_market_data_retrieval(exchange_client)
    except Exception as e:
        print(f"\n❌ Erreur lors du test des données de marché: {e}")
        success = False
    
    # Résultat final
    print_separator("RÉSULTAT FINAL")
    if success:
        print("🎉 TOUS LES TESTS ONT RÉUSSI !")
        print("✅ Le connecteur d'exchange est prêt pour l'intégration dans ADAN")
        print("\nProchaines étapes:")
        print("1. Intégrer le connecteur dans OrderManager")
        print("2. Créer les scripts de paper trading")
        print("3. Tester avec des ordres réels sur le testnet")
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez les erreurs ci-dessus avant de continuer")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
        traceback.print_exc()
        sys.exit(1)