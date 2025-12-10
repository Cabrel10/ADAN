#!/usr/bin/env python3
"""
Patch pour ajouter les appels au CentralLogger dans portfolio_manager.py
SANS arrêter l'entraînement
"""
import sys
from pathlib import Path

def patch_portfolio_manager():
    """Ajoute les imports et les appels au CentralLogger"""
    
    file_path = Path("src/adan_trading_bot/portfolio/portfolio_manager.py")
    
    print(f"🔧 Patching {file_path}...")
    
    # Lire le fichier
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Ajouter les imports du CentralLogger (après les autres imports)
    if "from ..common.central_logger import logger as central_logger" not in content:
        # Trouver la ligne après les imports locaux
        import_marker = "from ..performance.metrics import PerformanceMetrics"
        if import_marker in content:
            content = content.replace(
                import_marker,
                f"{import_marker}\n\ntry:\n    from ..common.central_logger import logger as central_logger\n    CENTRAL_LOGGER_AVAILABLE = True\nexcept ImportError:\n    CENTRAL_LOGGER_AVAILABLE = False\n    central_logger = None"
            )
            print("✅ Imports ajoutés")
    
    # 2. Ajouter l'appel au CentralLogger dans open_position
    if "central_logger.trade(" not in content:
        # Chercher la fin de la méthode open_position
        open_pos_marker = "return receipt"
        if open_pos_marker in content:
            # Ajouter l'appel avant le return
            patch_open = '''
        # Log le trade via CentralLogger
        if CENTRAL_LOGGER_AVAILABLE and central_logger:
            try:
                central_logger.trade(
                    action="BUY",
                    symbol=asset,
                    quantity=size,
                    price=entry_price,
                    pnl=None,
                    source="portfolio_manager"
                )
            except Exception as e:
                logger.debug(f"Erreur log trade: {e}")
        
        return receipt'''
            
            content = content.replace(
                "        return receipt",
                patch_open
            )
            print("✅ Appel CentralLogger ajouté à open_position")
    
    # 3. Ajouter l'appel au CentralLogger dans close_position
    if "# Log le trade de fermeture" not in content:
        close_pos_marker = "self.portfolio_value = new_portfolio_value"
        if close_pos_marker in content:
            patch_close = '''self.portfolio_value = new_portfolio_value
        
        # Log le trade de fermeture via CentralLogger
        if CENTRAL_LOGGER_AVAILABLE and central_logger:
            try:
                central_logger.trade(
                    action="SELL",
                    symbol=asset,
                    quantity=position.size,
                    price=current_price,
                    pnl=realized_pnl,
                    source="portfolio_manager"
                )
            except Exception as e:
                logger.debug(f"Erreur log trade fermeture: {e}")'''
            
            content = content.replace(
                close_pos_marker,
                patch_close
            )
            print("✅ Appel CentralLogger ajouté à close_position")
    
    # Écrire le fichier patché
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Patch appliqué avec succès!")
    print(f"📝 Fichier: {file_path}")

if __name__ == "__main__":
    try:
        patch_portfolio_manager()
        print("\n✅ PATCH COMPLET")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)
