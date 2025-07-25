"""
Gestionnaire sécurisé des API keys pour le trading live.
Implémente les tâches 10B.2.1, 10B.2.2.
"""

import os
import json
import logging
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import requests
import websocket
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Types d'exchanges supportés"""
    BINANCE = "binance"
    BINANCE_FUTURES = "binance_futures"
    BYBIT = "bybit"
    OKEX = "okex"
    KRAKEN = "kraken"


class ConnectionStatus(Enum):
    """États de connexion"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class APICredentials:
    """Informations d'identification API"""
    exchange: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # Pour OKEx
    sandbox: bool = True
    name: str = "Default"
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = {
            'exchange': self.exchange.value,
            'name': self.name,
            'sandbox': self.sandbox
        }
        
        if include_secrets:
            data.update({
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'passphrase': self.passphrase
            })
        else:
            # Masquer les secrets
            data.update({
                'api_key': self.api_key[:8] + "..." if self.api_key else None,
                'api_secret': "***" if self.api_secret else None,
                'passphrase': "***" if self.passphrase else None
            })
        
        return data


@dataclass
class ConnectionInfo:
    """Informations de connexion"""
    exchange: ExchangeType
    status: ConnectionStatus
    last_ping: Optional[datetime] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    reconnect_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['exchange'] = self.exchange.value
        data['status'] = self.status.value
        if self.last_ping:
            data['last_ping'] = self.last_ping.isoformat()
        return data


class SecureAPIManager:
    """Gestionnaire sécurisé des API keys"""
    
    def __init__(self, config_path: str = "config/api_keys.enc"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Chiffrement
        self._cipher_suite = None
        self._master_password = None
        
        # Stockage des credentials
        self.credentials: Dict[str, APICredentials] = {}
        
        # Monitoring des connexions
        self.connections: Dict[ExchangeType, ConnectionInfo] = {}
        self.connection_threads: Dict[ExchangeType, threading.Thread] = {}
        self.websockets: Dict[ExchangeType, websocket.WebSocketApp] = {}
        
        # Callbacks
        self.connection_callbacks: List[callable] = []
        
        # Configuration des endpoints
        self.endpoints = self._get_exchange_endpoints()
        
        logger.info("SecureAPIManager initialized")
    
    def _get_exchange_endpoints(self) -> Dict[ExchangeType, Dict[str, str]]:
        """Retourne les endpoints des exchanges"""
        return {
            ExchangeType.BINANCE: {
                'rest_url': 'https://api.binance.com',
                'rest_testnet': 'https://testnet.binance.vision',
                'ws_url': 'wss://stream.binance.com:9443/ws',
                'ws_testnet': 'wss://testnet.binance.vision/ws'
            },
            ExchangeType.BINANCE_FUTURES: {
                'rest_url': 'https://fapi.binance.com',
                'rest_testnet': 'https://testnet.binancefuture.com',
                'ws_url': 'wss://fstream.binance.com/ws',
                'ws_testnet': 'wss://stream.binancefuture.com/ws'
            },
            ExchangeType.BYBIT: {
                'rest_url': 'https://api.bybit.com',
                'rest_testnet': 'https://api-testnet.bybit.com',
                'ws_url': 'wss://stream.bybit.com/v5/public/spot',
                'ws_testnet': 'wss://stream-testnet.bybit.com/v5/public/spot'
            }
        }
    
    def set_master_password(self, password: str) -> bool:
        """Définit le mot de passe maître pour le chiffrement"""
        try:
            # Générer une clé de chiffrement à partir du mot de passe
            password_bytes = password.encode()
            salt = b'adan_trading_bot_salt'  # En production, utiliser un salt aléatoire
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            
            self._cipher_suite = Fernet(key)
            self._master_password = password
            
            # Essayer de charger les credentials existants
            if self.config_path.exists():
                self._load_credentials()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting master password: {e}")
            return False
    
    def add_credentials(self, credentials: APICredentials) -> bool:
        """Ajoute des credentials API"""
        if not self._cipher_suite:
            raise ValueError("Master password not set")
        
        # Tester la connexion avant de sauvegarder
        if not self._test_api_connection(credentials):
            logger.warning("API connection test failed, but saving credentials anyway")
        
        # Générer un ID unique
        cred_id = f"{credentials.exchange.value}_{credentials.name}"
        self.credentials[cred_id] = credentials
        
        # Sauvegarder
        self._save_credentials()
        
        logger.info(f"Added credentials for {credentials.exchange.value} ({credentials.name})")
        return True
    
    def get_credentials(self, exchange: ExchangeType, name: str = "Default") -> Optional[APICredentials]:
        """Récupère des credentials"""
        cred_id = f"{exchange.value}_{name}"
        return self.credentials.get(cred_id)
    
    def list_credentials(self) -> List[Dict[str, Any]]:
        """Liste tous les credentials (sans les secrets)"""
        return [cred.to_dict(include_secrets=False) for cred in self.credentials.values()]
    
    def remove_credentials(self, exchange: ExchangeType, name: str = "Default") -> bool:
        """Supprime des credentials"""
        cred_id = f"{exchange.value}_{name}"
        
        if cred_id in self.credentials:
            del self.credentials[cred_id]
            self._save_credentials()
            logger.info(f"Removed credentials for {exchange.value} ({name})")
            return True
        
        return False
    
    def _save_credentials(self) -> None:
        """Sauvegarde les credentials chiffrés"""
        if not self._cipher_suite:
            raise ValueError("Master password not set")
        
        try:
            # Préparer les données
            data = {
                'timestamp': datetime.now().isoformat(),
                'credentials': {
                    cred_id: cred.to_dict(include_secrets=True)
                    for cred_id, cred in self.credentials.items()
                }
            }
            
            # Chiffrer
            json_data = json.dumps(data)
            encrypted_data = self._cipher_suite.encrypt(json_data.encode())
            
            # Sauvegarder
            with open(self.config_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.debug("Credentials saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            raise
    
    def _load_credentials(self) -> None:
        """Charge les credentials chiffrés"""
        if not self._cipher_suite:
            raise ValueError("Master password not set")
        
        try:
            # Lire le fichier chiffré
            with open(self.config_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Déchiffrer
            decrypted_data = self._cipher_suite.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())
            
            # Reconstruire les credentials
            self.credentials = {}
            for cred_id, cred_data in data.get('credentials', {}).items():
                exchange = ExchangeType(cred_data['exchange'])
                credentials = APICredentials(
                    exchange=exchange,
                    api_key=cred_data['api_key'],
                    api_secret=cred_data['api_secret'],
                    passphrase=cred_data.get('passphrase'),
                    sandbox=cred_data.get('sandbox', True),
                    name=cred_data.get('name', 'Default')
                )
                self.credentials[cred_id] = credentials
            
            logger.info(f"Loaded {len(self.credentials)} credentials")
            
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            raise
    
    def _test_api_connection(self, credentials: APICredentials) -> bool:
        """Teste la connexion API"""
        try:
            exchange = credentials.exchange
            endpoints = self.endpoints.get(exchange)
            
            if not endpoints:
                logger.warning(f"No endpoints configured for {exchange.value}")
                return False
            
            # URL de base selon sandbox ou production
            base_url = endpoints['rest_testnet'] if credentials.sandbox else endpoints['rest_url']
            
            if exchange == ExchangeType.BINANCE:
                return self._test_binance_connection(base_url, credentials)
            elif exchange == ExchangeType.BINANCE_FUTURES:
                return self._test_binance_futures_connection(base_url, credentials)
            elif exchange == ExchangeType.BYBIT:
                return self._test_bybit_connection(base_url, credentials)
            else:
                logger.warning(f"Connection test not implemented for {exchange.value}")
                return True  # Assume OK for now
                
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def _test_binance_connection(self, base_url: str, credentials: APICredentials) -> bool:
        """Teste la connexion Binance"""
        try:
            import hmac
            import time
            
            # Endpoint de test
            endpoint = "/api/v3/account"
            timestamp = int(time.time() * 1000)
            
            # Paramètres
            params = f"timestamp={timestamp}"
            
            # Signature
            signature = hmac.new(
                credentials.api_secret.encode(),
                params.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Headers
            headers = {
                'X-MBX-APIKEY': credentials.api_key
            }
            
            # Requête
            url = f"{base_url}{endpoint}?{params}&signature={signature}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info("Binance API connection test successful")
                return True
            else:
                logger.warning(f"Binance API test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Binance connection test error: {e}")
            return False
    
    def _test_binance_futures_connection(self, base_url: str, credentials: APICredentials) -> bool:
        """Teste la connexion Binance Futures"""
        try:
            import hmac
            import time
            
            # Endpoint de test
            endpoint = "/fapi/v2/account"
            timestamp = int(time.time() * 1000)
            
            # Paramètres
            params = f"timestamp={timestamp}"
            
            # Signature
            signature = hmac.new(
                credentials.api_secret.encode(),
                params.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Headers
            headers = {
                'X-MBX-APIKEY': credentials.api_key
            }
            
            # Requête
            url = f"{base_url}{endpoint}?{params}&signature={signature}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info("Binance Futures API connection test successful")
                return True
            else:
                logger.warning(f"Binance Futures API test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Binance Futures connection test error: {e}")
            return False
    
    def _test_bybit_connection(self, base_url: str, credentials: APICredentials) -> bool:
        """Teste la connexion Bybit"""
        try:
            import hmac
            import time
            
            # Endpoint de test
            endpoint = "/v5/account/wallet-balance"
            timestamp = str(int(time.time() * 1000))
            
            # Paramètres
            params = f"accountType=UNIFIED&timestamp={timestamp}"
            
            # Signature
            signature = hmac.new(
                credentials.api_secret.encode(),
                timestamp.encode() + credentials.api_key.encode() + params.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Headers
            headers = {
                'X-BAPI-API-KEY': credentials.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': '5000'
            }
            
            # Requête
            url = f"{base_url}{endpoint}?{params}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info("Bybit API connection test successful")
                return True
            else:
                logger.warning(f"Bybit API test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Bybit connection test error: {e}")
            return False
    
    def start_connection_monitoring(self, exchange: ExchangeType) -> bool:
        """Démarre le monitoring de connexion pour un exchange"""
        if exchange in self.connection_threads:
            logger.warning(f"Connection monitoring already active for {exchange.value}")
            return False
        
        # Initialiser l'info de connexion
        self.connections[exchange] = ConnectionInfo(
            exchange=exchange,
            status=ConnectionStatus.CONNECTING
        )
        
        # Lancer le thread de monitoring
        thread = threading.Thread(
            target=self._monitor_connection,
            args=(exchange,),
            daemon=True
        )
        thread.start()
        self.connection_threads[exchange] = thread
        
        logger.info(f"Started connection monitoring for {exchange.value}")
        return True
    
    def _monitor_connection(self, exchange: ExchangeType) -> None:
        """Surveille la connexion WebSocket"""
        connection_info = self.connections[exchange]
        endpoints = self.endpoints.get(exchange)
        
        if not endpoints:
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = "No endpoints configured"
            return
        
        # Récupérer les credentials
        credentials = self.get_credentials(exchange)
        if not credentials:
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = "No credentials found"
            return
        
        # URL WebSocket
        ws_url = endpoints['ws_testnet'] if credentials.sandbox else endpoints['ws_url']
        
        def on_open(ws):
            connection_info.status = ConnectionStatus.CONNECTED
            connection_info.last_ping = datetime.now()
            connection_info.reconnect_count = 0
            logger.info(f"WebSocket connected for {exchange.value}")
            self._notify_connection_change(exchange, ConnectionStatus.CONNECTED)
        
        def on_message(ws, message):
            # Mettre à jour le ping
            connection_info.last_ping = datetime.now()
            
            # Calculer la latence si possible
            try:
                data = json.loads(message)
                if 'ping' in data:
                    # Répondre au ping
                    ws.send(json.dumps({'pong': data['ping']}))
            except:
                pass
        
        def on_error(ws, error):
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = str(error)
            logger.error(f"WebSocket error for {exchange.value}: {error}")
            self._notify_connection_change(exchange, ConnectionStatus.ERROR)
        
        def on_close(ws, close_status_code, close_msg):
            if connection_info.status != ConnectionStatus.ERROR:
                connection_info.status = ConnectionStatus.RECONNECTING
                connection_info.reconnect_count += 1
                logger.warning(f"WebSocket closed for {exchange.value}, reconnecting...")
                self._notify_connection_change(exchange, ConnectionStatus.RECONNECTING)
                
                # Reconnexion automatique après délai
                time.sleep(min(connection_info.reconnect_count * 2, 30))
                if exchange in self.connection_threads:  # Vérifier si pas arrêté
                    self._monitor_connection(exchange)
        
        # Créer et lancer la WebSocket
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.websockets[exchange] = ws
            ws.run_forever()
            
        except Exception as e:
            connection_info.status = ConnectionStatus.ERROR
            connection_info.error_message = str(e)
            logger.error(f"WebSocket connection failed for {exchange.value}: {e}")
    
    def stop_connection_monitoring(self, exchange: ExchangeType) -> bool:
        """Arrête le monitoring de connexion"""
        if exchange not in self.connection_threads:
            return False
        
        # Fermer la WebSocket
        if exchange in self.websockets:
            try:
                self.websockets[exchange].close()
                del self.websockets[exchange]
            except:
                pass
        
        # Arrêter le thread
        if exchange in self.connection_threads:
            del self.connection_threads[exchange]
        
        # Mettre à jour le statut
        if exchange in self.connections:
            self.connections[exchange].status = ConnectionStatus.DISCONNECTED
        
        logger.info(f"Stopped connection monitoring for {exchange.value}")
        return True
    
    def get_connection_status(self, exchange: ExchangeType) -> Optional[ConnectionInfo]:
        """Récupère le statut de connexion"""
        return self.connections.get(exchange)
    
    def get_all_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Récupère tous les statuts de connexion"""
        return {
            exchange.value: info.to_dict()
            for exchange, info in self.connections.items()
        }
    
    def add_connection_callback(self, callback: callable) -> None:
        """Ajoute un callback pour les changements de connexion"""
        self.connection_callbacks.append(callback)
    
    def _notify_connection_change(self, exchange: ExchangeType, status: ConnectionStatus) -> None:
        """Notifie les callbacks des changements de connexion"""
        for callback in self.connection_callbacks:
            try:
                callback(exchange, status)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")
    
    def shutdown(self) -> None:
        """Arrêt propre du gestionnaire"""
        logger.info("Shutting down SecureAPIManager...")
        
        # Arrêter tous les monitorings
        for exchange in list(self.connection_threads.keys()):
            self.stop_connection_monitoring(exchange)
        
        # Sauvegarder les credentials
        if self.credentials and self._cipher_suite:
            self._save_credentials()
        
        logger.info("SecureAPIManager shutdown completed")