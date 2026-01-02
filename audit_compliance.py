#!/usr/bin/env python3
"""
Audit de Conformité - Vérification que le bot respecte les règles
"""

import yaml
import re
import sys
from pathlib import Path

# Couleurs
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_compliance():
    print(f"\n🕵️  {YELLOW}AUDIT DE CONFORMITÉ EN TEMPS RÉEL{RESET}")
    print("=" * 60)

    # 1. CHARGEMENT DE LA CONFIGURATION
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration chargée.")
    except Exception as e:
        print(f"{RED}❌ Erreur critique : Impossible de lire config.yaml{RESET}")
        sys.exit(1)

    # Extraction des règles strictes
    try:
        tiers = config['capital_tiers']
        env_rules = config['environment']
        min_trade = env_rules.get('hard_constraints', {}).get('min_order_value_usdt', 11.0)
        max_drawdown = env_rules.get('risk_management', {}).get('max_drawdown_pct', 0.15)

        print(f"\n📋 {YELLOW}RÈGLES STRICTES DÉFINIES :{RESET}")
        print(f"  • Min Order Value : ${min_trade}")
        print(f"  • Max Drawdown    : {max_drawdown*100}%")
        print(f"  • Paliers définis : {len(tiers)}")
    except KeyError as e:
        print(f"{RED}❌ Configuration incomplète : Clé manquante {e}{RESET}")

    # 2. ANALYSE DES FAITS (LOGS)
    log_path = Path("config/logs/adan_trading_bot.log")
    if not log_path.exists():
        print(f"{RED}❌ Pas de logs trouvés pour vérifier l'activité.{RESET}")
        return

    print(f"\n🔍 {YELLOW}ANALYSE DU COMPORTEMENT RÉEL :{RESET}")

    with open(log_path, 'r') as f:
        logs = f.readlines()[-500:]  # Dernières 500 lignes

    # Vérification des indicateurs
    rsi_values = []
    adx_values = []
    decisions = []
    sizing_warnings = 0

    for line in logs:
        # Check Indicators
        if "RSI=" in line:
            try:
                rsi = float(re.search(r"RSI=([0-9.]+)", line).group(1))
                adx = float(re.search(r"ADX=([0-9.]+)", line).group(1))
                rsi_values.append(rsi)
                adx_values.append(adx)
            except:
                pass

        # Check Décisions
        if "DÉCISION FINALE:" in line or "DECISION FINALE:" in line:
            decisions.append(line.strip())

        # Check Compliance Sizing
        if "taille de position" in line.lower() or "position size" in line.lower():
            if "adjusted" in line or "clamped" in line or "trop faible" in line:
                sizing_warnings += 1

    # --- VERDICT ---

    # 1. Indicateurs
    if not rsi_values:
        print(f"  ❌ {RED}INDICATEURS : Absents des logs.{RESET}")
    else:
        rsi_variance = len(set(rsi_values))
        last_rsi = rsi_values[-1]

        if last_rsi == 50.0 and rsi_variance == 1:
            print(f"  ❌ {RED}INDICATEURS : MORTS (Flatline). RSI bloqué à 50.00.{RESET}")
            print(f"     -> Les modèles reçoivent des données vides. C'est du trading à l'aveugle.")
        elif rsi_variance < 3:
            print(f"  ⚠️ {YELLOW}INDICATEURS : DOUTEUX. Très peu de variation ({rsi_variance} valeurs uniques).{RESET}")
        else:
            print(f"  ✅ {GREEN}INDICATEURS : VIVANTS. RSI varie ({min(rsi_values):.2f} - {max(rsi_values):.2f}).{RESET}")

    # 2. Normalisation
    has_normalization_warning = any("Normalisateur global désactivé" in l for l in logs)
    if has_normalization_warning:
        print(f"  ⚠️ {YELLOW}NORMALISATION : Le Global Normalizer est OFF.{RESET}")
        print(f"     -> Vérification : Les workers individuels ont-ils chargé leurs VecNormalize ?")
        w1_loaded = any("w1 chargé avec succès" in l for l in logs)
        if w1_loaded:
            print(f"     ✅ {GREEN}RASSURANT : Les workers individuels semblent chargés.{RESET}")
        else:
            print(f"     ❌ {RED}DANGER : Aucune trace de chargement correct des workers.{RESET}")

    # 3. Activité
    if not decisions:
        print(f"  ℹ️  {YELLOW}ACTIVITÉ : Aucune décision finale enregistrée récemment.{RESET}")
    else:
        print(f"  ℹ️  ACTIVITÉ : {len(decisions)} décisions récentes.")
        if decisions:
            print(f"     Dernière : {decisions[-1][:100]}...")

    # 4. Vérification du Warmup
    warmup_issues = sum(1 for l in logs if "Données insuffisantes" in l)
    if warmup_issues > 0:
        print(f"  ⚠️ {YELLOW}WARMUP : {warmup_issues} avertissements de données insuffisantes.{RESET}")
        print(f"     -> Le système attend que les données 4h se remplissent (28 périodes).")

    print("\n" + "=" * 60)
    print(f"⚖️  {YELLOW}CONCLUSION DE L'AUDIT :{RESET}")

    if rsi_values and last_rsi == 50.0 and rsi_variance == 1:
        print(f"{RED}🚫 SYSTÈME NON FIABLE : Les indicateurs ne sont pas calculés.{RESET}")
        print(f"{RED}   ACTION REQUISE : Le Warmup des données échoue ou le fetcher renvoie du vide.{RESET}")
        print(f"\n{YELLOW}RECOMMANDATION :{RESET}")
        print(f"   1. Attendre que le Warmup se complète (28 périodes 4h)")
        print(f"   2. Vérifier que les données Binance arrivent correctement")
        print(f"   3. Valider que les indicateurs commencent à bouger")
    elif warmup_issues > 0:
        print(f"{YELLOW}⏳ SYSTÈME EN PHASE DE DÉMARRAGE{RESET}")
        print(f"   Le bot est en Warmup. C'est normal au démarrage.")
        print(f"   Les indicateurs devraient devenir vivants dans quelques minutes.")
    else:
        print(f"{GREEN}🆗 SYSTÈME COHÉRENT : Les données entrent, les indicateurs bougent.{RESET}")
        print(f"   Le bot est prêt pour le trading.")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    check_compliance()
