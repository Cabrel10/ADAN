#!/usr/bin/python3
"""
Analyse de l'environnement de trading pour identifier les problèmes potentiels.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("env_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingEnvironmentAnalyzer:
    """Classe pour analyser l'environnement de trading"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise l'analyseur d'environnement.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()
        
        # Dossiers de sortie
        self.output_dir = Path("env_analysis")
        self.output_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Charge la configuration depuis le fichier"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration chargée depuis {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    def analyze_reward_structure(self):
        """Analyse la structure des récompenses"""
        logger.info("Analyse de la structure des récompenses...")
        
        if not self.config.get("reward_shaping"):
            logger.warning("Aucune configuration de récompense trouvée dans le fichier de configuration")
            return {}
        
        reward_config = self.config["reward_shaping"]
        
        # Extraire les paramètres de récompense
        reward_params = {
            "pnl_weight": reward_config.get("pnl_weight", 1.0),
            "inaction_penalty": reward_config.get("inaction_penalty", -0.01),
            "missed_opportunity_penalty": reward_config.get("missed_opportunity_penalty", -0.05),
            "take_profit_bonus": reward_config.get("take_profit_bonus", 1.0),
            "stop_loss_penalty": reward_config.get("stop_loss_penalty", -1.0),
            "overnight_penalty": reward_config.get("overnight_penalty", -0.1),
            "overnight_bonus": reward_config.get("overnight_bonus", 0.1),
            "scaling_factor": reward_config.get("scaling_factor", 1.0),
            "min_reward": reward_config.get("min_reward", -10.0),
            "max_reward": reward_config.get("max_reward", 10.0),
        }
        
        # Afficher les paramètres
        logger.info("Paramètres de récompense:")
        for param, value in reward_params.items():
            logger.info(f"  {param}: {value}")
        
        # Vérifier les déséquilibres potentiels
        issues = []
        
        # Vérifier si les pénalités sont trop élevées par rapport aux récompenses
        if abs(reward_params["inaction_penalty"]) > reward_params["take_profit_bonus"] * 0.5:
            issues.append("La pénalité d'inaction est élevée par rapport au bonus de prise de profit")
        
        if abs(reward_params["missed_opportunity_penalty"]) > reward_params["take_profit_bonus"] * 0.5:
            issues.append("La pénalité d'opportunité manquée est élevée par rapport au bonus de prise de profit")
        
        if abs(reward_params["stop_loss_penalty"]) > reward_params["take_profit_bonus"] * 1.5:
            issues.append("La pénalité de stop-loss est très élevée par rapport au bonus de prise de profit")
        
        # Afficher les problèmes détectés
        if issues:
            logger.warning("Problèmes potentiels détectés dans la structure des récompenses:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Aucun problème majeur détecté dans la structure des récompenses")
        
        # Générer un graphique de la structure des récompenses
        self._plot_reward_structure(reward_params, issues)
        
        return {"params": reward_params, "issues": issues}
    
    def _plot_reward_structure(self, reward_params: Dict, issues: List[str]):
        """Génère un graphique de la structure des récompenses"""
        # Préparer les données pour le graphique
        categories = [
            "P&L Weight", "Inaction Penalty", "Missed Opportunity",
            "Take Profit", "Stop Loss", "Overnight Penalty", "Overnight Bonus"
        ]
        
        values = [
            reward_params["pnl_weight"],
            reward_params["inaction_penalty"],
            reward_params["missed_opportunity_penalty"],
            reward_params["take_profit_bonus"],
            reward_params["stop_loss_penalty"],
            reward_params["overnight_penalty"],
            reward_params["overnight_bonus"]
        ]
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        
        # Utiliser des couleurs différentes pour les récompenses et les pénalités
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        bars = plt.bar(categories, values, color=colors)
        
        # Ajouter des étiquettes de valeur
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.01 if height >=0 else -0.05), 
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Structure des Récompenses et Pénalités')
        plt.ylabel('Valeur')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Ajouter une légende pour les couleurs
        import matplotlib.patches as mpatches
        green_patch = mpatches.Patch(color='green', label='Récompenses')
        red_patch = mpatches.Patch(color='red', label='Pénalités')
        plt.legend(handles=[green_patch, red_patch])
        
        # Ajouter des annotations pour les problèmes détectés
        if issues:
            plt.figtext(0.5, -0.3, 
                       "Problèmes détectés:\n- " + "\n- ".join(issues),
                       ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Sauvegarder le graphique
        plot_path = self.output_dir / 'reward_structure.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique de la structure des récompenses sauvegardé dans {plot_path}")
    
    def analyze_action_thresholds(self):
        """Analyse les seuils d'action et leur impact sur le trading"""
        logger.info("Analyse des seuils d'action...")
        
        if not self.config.get("trading_rules"):
            logger.warning("Aucune règle de trading trouvée dans la configuration")
            return {}
        
        trading_rules = self.config["trading_rults"]
        
        # Extraire les paramètres de seuil
        threshold_params = {
            "min_order_value_usdt": trading_rules.get("min_order_value_usdt", 10.0),
            "max_position_size_usdt": trading_rules.get("max_position_size_usdt", 1000.0),
            "max_open_trades": trading_rules.get("max_open_trades", 5),
            "max_risk_per_trade": trading_rules.get("max_risk_per_trade", 0.02),
            "take_profit_pct": trading_rules.get("take_profit_pct", 0.03),
            "stop_loss_pct": trading_rules.get("stop_loss_pct", 0.02),
        }
        
        # Afficher les paramètres
        logger.info("Paramètres de seuil de trading:")
        for param, value in threshold_params.items():
            logger.info(f"  {param}: {value}")
        
        # Vérifier les problèmes potentiels
        issues = []
        
        if threshold_params["min_order_value_usdt"] > 10:
            issues.append(f"La valeur minimale de commande est élevée ({threshold_params['min_order_value_usdt']} USDT)")
        
        if threshold_params["max_position_size_usdt"] < 100:
            issues.append(f"La taille de position maximale est faible ({threshold_params['max_position_size_usdt']} USDT)")
        
        if threshold_params["max_open_trades"] < 3:
            issues.append(f"Le nombre maximum de trades ouverts est faible ({threshold_params['max_open_trades']})")
        
        if threshold_params["take_profit_pct"] < 0.01 or threshold_params["take_profit_pct"] > 0.1:
            issues.append(f"Le take-profit semble extrême ({threshold_params['take_profit_pct']*100}%)")
        
        if threshold_params["stop_loss_pct"] < 0.005 or threshold_params["stop_loss_pct"] > 0.05:
            issues.append(f"Le stop-loss semble extrême ({threshold_params['stop_loss_pct']*100}%)")
        
        # Afficher les problèmes détectés
        if issues:
            logger.warning("Problèmes potentiels détectés dans les seuils de trading:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Aucun problème majeur détecté dans les seuils de trading")
        
        # Générer un graphique des seuils
        self._plot_trading_thresholds(threshold_params, issues)
        
        return {"params": threshold_params, "issues": issues}
    
    def _plot_trading_thresholds(self, threshold_params: Dict, issues: List[str]):
        """Génère un graphique des seuils de trading"""
        # Préparer les données pour le graphique
        categories = [
            "Ordre Min (USDT)", "Position Max (USDT)", "Trades Max",
            "Take Profit (%)", "Stop Loss (%)", "Risque/Trade"
        ]
        
        values = [
            threshold_params["min_order_value_usdt"],
            threshold_params["max_position_size_usdt"],
            threshold_params["max_open_trades"],
            threshold_params["take_profit_pct"] * 100,  # Convertir en pourcentage
            threshold_params["stop_loss_pct"] * 100,    # Convertir en pourcentage
            threshold_params["max_risk_per_trade"] * 100  # Convertir en pourcentage
        ]
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Créer un graphique à barres avec des couleurs différentes
        bars = ax.bar(categories, values, color=['blue', 'blue', 'blue', 'green', 'red', 'orange'])
        
        # Ajouter des étiquettes de valeur
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.01 * max(values)), 
                   f'{height:.2f}' if height < 10 else f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.title('Seuils de Trading')
        plt.ylabel('Valeur')
        plt.xticks(rotation=45, ha='right')
        
        # Ajouter une légende pour les couleurs
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label='Seuils de Trading')
        green_patch = mpatches.Patch(color='green', label='Take Profit')
        red_patch = mpatches.Patch(color='red', label='Stop Loss')
        orange_patch = mpatches.Patch(color='orange', label='Risque')
        plt.legend(handles=[blue_patch, green_patch, red_patch, orange_patch])
        
        # Ajouter des annotations pour les problèmes détectés
        if issues:
            plt.figtext(0.5, -0.3, 
                       "Problèmes détectés:\n- " + "\n- ".join(issues),
                       ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plot_path = self.output_dir / 'trading_thresholds.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique des seuils de trading sauvegardé dans {plot_path}")
    
    def analyze_environment(self):
        """Exécute toutes les analyses de l'environnement"""
        logger.info("Démarrage de l'analyse de l'environnement de trading...")
        
        results = {
            "reward_structure": self.analyze_reward_structure(),
            "trading_thresholds": self.analyze_action_thresholds(),
        }
        
        # Générer un rapport sommaire
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: Dict):
        """Génère un rapport sommaire des analyses"""
        report_path = self.output_dir / 'environment_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Rapport d'Analyse de l'Environnement de Trading\n\n")
            
            # Section Structure des Récompenses
            f.write("## 1. Structure des Récompenses\n\n")
            
            if "reward_structure" in results and results["reward_structure"]:
                reward_data = results["reward_structure"]
                
                f.write("### Paramètres de Récompense\n")
                for param, value in reward_data["params"].items():
                    f.write(f"- `{param}`: `{value}`\n")
                
                if reward_data["issues"]:
                    f.write("\n### Problèmes Détectés\n")
                    for issue in reward_data["issues"]:
                        f.write(f"- ⚠️ {issue}\n")
                else:
                    f.write("\n✅ Aucun problème majeur détecté dans la structure des récompenses.\n")
                
                f.write("\n![Structure des Récompenses](reward_structure.png)\n")
            else:
                f.write("Aucune donnée de structure de récompense disponible.\n")
            
            # Section Seuils de Trading
            f.write("\n## 2. Seuils de Trading\n\n")
            
            if "trading_thresholds" in results and results["trading_thresholds"]:
                threshold_data = results["trading_thresholds"]
                
                f.write("### Paramètres de Seuils\n")
                for param, value in threshold_data["params"].items():
                    f.write(f"- `{param}`: `{value}`\n")
                
                if threshold_data["issues"]:
                    f.write("\n### Problèmes Détectés\n")
                    for issue in threshold_data["issues"]:
                        f.write(f"- ⚠️ {issue}\n")
                else:
                    f.write("\n✅ Aucun problème majeur détecté dans les seuils de trading.\n")
                
                f.write("\n![Seuils de Trading](trading_thresholds.png)\n")
            else:
                f.write("Aucune donnée de seuil de trading disponible.\n")
            
            # Section Recommandations
            f.write("\n## 3. Recommandations\n\n")
            
            f.write("### Si le modèle est trop passif :\n")
            f.write("- Réduire les pénalités d'inaction et d'opportunités manquées\n")
            f.write("- Augmenter les récompenses pour les trades réussis\n")
            f.write("- Réduire la valeur minimale des ordres\n")
            f.write("- Augmenter le nombre maximum de trades ouverts\n\n")
            
            f.write("### Si le modèle prend trop de risques :\n")
            f.write("- Augmenter les pénalités de stop-loss\n")
            f.write("- Réduire la taille de position maximale\n")
            f.write("- Réduire le risque par trade\n")
            f.write("- Ajouter des pénalités pour le drawdown\n\n")
            
            f.write("### Pour améliorer l'apprentissage :\n")
            f.write("- Ajuster le facteur d'échelle des récompenses\n")
            f.write("- Normaliser les récompenses\n")
            f.write("- Implémenter un système de récompenses progressives\n")
        
        logger.info(f"Rapport d'analyse généré: {report_path}")

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse de l'environnement de trading")
    parser.add_argument('--config', type=str, default="config/config.yaml",
                      help="Chemin vers le fichier de configuration")
    
    args = parser.parse_args()
    
    try:
        # Initialiser l'analyseur
        analyzer = TradingEnvironmentAnalyzer(args.config)
        
        # Exécuter les analyses
        print("\n" + "="*50)
        print("ANALYSE DE L'ENVIRONNEMENT DE TRADING")
        print("="*50)
        
        results = analyzer.analyze_environment()
        
        print("\n" + "="*50)
        print("ANALYSE TERMINÉE")
        print("="*50)
        print("\n📊 Les résultats détaillés ont été sauvegardés dans le dossier 'env_analysis/'")
        print("   Consultez le rapport complet: env_analysis/environment_analysis_report.md")
        
        return 0
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de l'environnement: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
