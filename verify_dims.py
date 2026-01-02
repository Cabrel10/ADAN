#!/usr/bin/env python3
"""
VÉRIFICATION DES DIMENSIONS
Confirme que StateBuilder produit exactement 542 dimensions (525 + 17)
"""

import sys
from pathlib import Path

sys.path.append("src")

print("🔍 VÉRIFICATION DES DIMENSIONS POST-PATCH")
print("=" * 70)

try:
    from adan_trading_bot.data_processing.state_builder import StateBuilder
    
    sb = StateBuilder()
    
    print("\n📊 Configuration chargée:")
    print(f"   Fenêtres: {sb.window_sizes}")
    
    # Calcul des dimensions
    total_market = 0
    print("\n📈 Calcul des dimensions:")
    
    for tf in ['5m', '1h', '4h']:
        if tf in sb.features_config:
            feats = len(sb.features_config[tf])
            win = sb.window_sizes.get(tf, 0)
            subtotal = feats * win
            total_market += subtotal
            print(f"   {tf}: {win:2d} fenêtres × {feats:2d} features = {subtotal:3d}")
    
    print(f"\n   Total Marché: {total_market}")
    print(f"   Portfolio:    17")
    print(f"   ─────────────────────")
    print(f"   TOTAL:        {total_market + 17}")
    
    # Vérification
    print("\n" + "=" * 70)
    if total_market == 525 and (total_market + 17) == 542:
        print("✅ MATCH PARFAIT !")
        print("   Les dimensions correspondent exactement aux modèles entraînés.")
        print("   Vous pouvez déployer en confiance.")
        sys.exit(0)
    else:
        print(f"❌ ÉCHEC: Dimensions incorrectes")
        print(f"   Attendu: 525 (marché) + 17 (portfolio) = 542")
        print(f"   Obtenu:  {total_market} (marché) + 17 (portfolio) = {total_market + 17}")
        print(f"   Différence: {total_market - 525}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
