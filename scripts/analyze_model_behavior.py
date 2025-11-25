#!/usr/bin/env python3
"""
ANALYSE APPROFONDIE DU COMPORTEMENT DU MODÈLE
Détecte les biais, mesure les poids, intentions et actions mal organisées
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import (
    MultiAssetChunkedEnv
)
from adan_trading_bot.common.config_loader import ConfigLoader


class ModelBehaviorAnalyzer:
    """Analyse le comportement du modèle PPO"""

    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.env = None
        
        # Métriques de comportement
        self.action_history = []
        self.action_distribution = defaultdict(int)
        self.action_confidence = []
        self.trade_sequence = []
        self.reward_history = []
        self.portfolio_history = []
        self.decision_patterns = defaultdict(list)
        
        # Analyse des biais
        self.bias_metrics = {
            'buy_bias': 0,
            'sell_bias': 0,
            'hold_bias': 0,
            'action_variance': 0,
            'action_entropy': 0,
            'decision_consistency': 0,
        }

    def setup_environment(self, asset: str, split: str = "train"):
        """Configure l'environnement"""
        logger.info(f"Configuration environnement: {asset} ({split})")
        
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)
        config['initial_capital'] = 20.5
        config['environment']['assets'] = [asset]

        self.env = MultiAssetChunkedEnv(
            config=config,
            worker_id=0,
            log_level="WARNING",
            data_split=split
        )
        
        logger.info("✅ Environnement configuré")

    def load_model(self):
        """Charge le modèle PPO"""
        logger.info(f"Chargement modèle: {self.model_path}")
        
        try:
            self.model = PPO.load(self.model_path, env=self.env)
            logger.info("✅ Modèle chargé")
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            return False
        
        return True

    def analyze_action_distribution(self, n_steps: int = 5000):
        """Analyse la distribution des actions"""
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSE 1: DISTRIBUTION DES ACTIONS")
        logger.info(f"{'='*80}")
        
        obs, _ = self.env.reset()
        
        for step in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            # Analyser l'action
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            self.action_distribution[self._categorize_action(action_value)] += 1
            
            if len(action) > 1:
                confidence = float(action[1])
                self.action_confidence.append(confidence)
            
            if (step + 1) % 1000 == 0:
                logger.info(f"  Step {step + 1}/{n_steps}")
            
            if terminated or truncated:
                break
        
        # Calculer les statistiques
        total_actions = len(self.action_history)
        
        logger.info(f"\n📊 DISTRIBUTION DES ACTIONS:")
        logger.info(f"  Total actions: {total_actions}")
        
        for action_type, count in sorted(self.action_distribution.items()):
            pct = (count / total_actions) * 100
            logger.info(f"  {action_type:15}: {count:5} ({pct:5.2f}%)")
        
        # Calculer les biais
        buy_count = self.action_distribution.get('BUY', 0)
        sell_count = self.action_distribution.get('SELL', 0)
        hold_count = self.action_distribution.get('HOLD', 0)
        
        self.bias_metrics['buy_bias'] = (buy_count / total_actions) * 100
        self.bias_metrics['sell_bias'] = (sell_count / total_actions) * 100
        self.bias_metrics['hold_bias'] = (hold_count / total_actions) * 100
        
        logger.info(f"\n🎯 BIAIS DÉTECTÉS:")
        logger.info(f"  Buy Bias:  {self.bias_metrics['buy_bias']:.2f}%")
        logger.info(f"  Sell Bias: {self.bias_metrics['sell_bias']:.2f}%")
        logger.info(f"  Hold Bias: {self.bias_metrics['hold_bias']:.2f}%")
        
        # Vérifier les biais extrêmes
        if self.bias_metrics['buy_bias'] > 60:
            logger.warning(f"⚠️ BIAIS EXTRÊME: Trop de BUY ({self.bias_metrics['buy_bias']:.2f}%)")
        if self.bias_metrics['sell_bias'] > 60:
            logger.warning(f"⚠️ BIAIS EXTRÊME: Trop de SELL ({self.bias_metrics['sell_bias']:.2f}%)")
        if self.bias_metrics['hold_bias'] > 80:
            logger.warning(f"⚠️ BIAIS EXTRÊME: Trop de HOLD ({self.bias_metrics['hold_bias']:.2f}%)")

    def analyze_action_variance(self):
        """Analyse la variance des actions"""
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSE 2: VARIANCE ET STABILITÉ DES ACTIONS")
        logger.info(f"{'='*80}")
        
        if not self.action_history:
            logger.warning("Pas d'historique d'actions")
            return
        
        actions_array = np.array([a[0] if isinstance(a, np.ndarray) else a 
                                 for a in self.action_history])
        
        variance = np.var(actions_array)
        std = np.std(actions_array)
        mean = np.mean(actions_array)
        
        self.bias_metrics['action_variance'] = variance
        
        logger.info(f"\n📊 STATISTIQUES DES ACTIONS:")
        logger.info(f"  Mean:     {mean:.4f}")
        logger.info(f"  Std Dev:  {std:.4f}")
        logger.info(f"  Variance: {variance:.4f}")
        logger.info(f"  Min:      {np.min(actions_array):.4f}")
        logger.info(f"  Max:      {np.max(actions_array):.4f}")
        
        # Vérifier la stabilité
        if variance < 0.01:
            logger.warning(f"⚠️ ACTIONS TROP STABLES: Variance très faible ({variance:.6f})")
            logger.warning("   Le modèle pourrait être figé ou mal entraîné")
        elif variance > 0.5:
            logger.warning(f"⚠️ ACTIONS TROP INSTABLES: Variance très élevée ({variance:.6f})")
            logger.warning("   Le modèle pourrait être chaotique")

    def analyze_action_entropy(self):
        """Calcule l'entropie des actions"""
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSE 3: ENTROPIE ET DIVERSITÉ DES DÉCISIONS")
        logger.info(f"{'='*80}")
        
        if not self.action_distribution:
            logger.warning("Pas de distribution d'actions")
            return
        
        total = sum(self.action_distribution.values())
        probabilities = [count / total for count in self.action_distribution.values()]
        
        # Entropie Shannon
        entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities)
        max_entropy = np.log2(len(self.action_distribution))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        self.bias_metrics['action_entropy'] = normalized_entropy
        
        logger.info(f"\n📊 ENTROPIE DES DÉCISIONS:")
        logger.info(f"  Entropy:           {entropy:.4f}")
        logger.info(f"  Max Entropy:       {max_entropy:.4f}")
        logger.info(f"  Normalized:        {normalized_entropy:.4f}")
        
        if normalized_entropy < 0.3:
            logger.warning(f"⚠️ DÉCISIONS PEU DIVERSIFIÉES: Entropie faible ({normalized_entropy:.4f})")
            logger.warning("   Le modèle utilise peu de types d'actions différentes")
        elif normalized_entropy > 0.9:
            logger.info(f"✅ DÉCISIONS BIEN DIVERSIFIÉES: Entropie élevée ({normalized_entropy:.4f})")

    def analyze_action_sequences(self):
        """Analyse les séquences d'actions"""
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSE 4: SÉQUENCES D'ACTIONS ET PATTERNS")
        logger.info(f"{'='*80}")
        
        if len(self.action_history) < 2:
            logger.warning("Pas assez d'actions")
            return
        
        # Analyser les transitions
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.action_history) - 1):
            current = self._categorize_action(self.action_history[i][0] 
                                             if isinstance(self.action_history[i], np.ndarray) 
                                             else self.action_history[i])
            next_action = self._categorize_action(self.action_history[i + 1][0] 
                                                 if isinstance(self.action_history[i + 1], np.ndarray) 
                                                 else self.action_history[i + 1])
            
            transitions[current][next_action] += 1
        
        logger.info(f"\n📊 MATRICE DE TRANSITION:")
        
        for current_action in sorted(transitions.keys()):
            logger.info(f"\n  De {current_action}:")
            total = sum(transitions[current_action].values())
            
            for next_action in sorted(transitions[current_action].keys()):
                count = transitions[current_action][next_action]
                pct = (count / total) * 100
                logger.info(f"    → {next_action:10}: {count:4} ({pct:5.2f}%)")
        
        # Vérifier les patterns anormaux
        for current_action in transitions:
            next_actions = transitions[current_action]
            if len(next_actions) == 1:
                logger.warning(f"⚠️ PATTERN FIGÉ: Après {current_action}, toujours {list(next_actions.keys())[0]}")

    def analyze_decision_consistency(self):
        """Analyse la cohérence des décisions"""
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSE 5: COHÉRENCE DES DÉCISIONS")
        logger.info(f"{'='*80}")
        
        if len(self.action_history) < 100:
            logger.warning("Pas assez d'actions pour analyse")
            return
        
        # Diviser en chunks
        chunk_size = len(self.action_history) // 5
        chunks = [self.action_history[i:i+chunk_size] 
                 for i in range(0, len(self.action_history), chunk_size)]
        
        chunk_distributions = []
        
        for i, chunk in enumerate(chunks):
            distribution = defaultdict(int)
            for action in chunk:
                action_val = action[0] if isinstance(action, np.ndarray) else action
                distribution[self._categorize_action(action_val)] += 1
            
            chunk_distributions.append(distribution)
            
            logger.info(f"\n  Chunk {i+1}:")
            for action_type in sorted(distribution.keys()):
                pct = (distribution[action_type] / len(chunk)) * 100
                logger.info(f"    {action_type:10}: {pct:5.2f}%")
        
        # Calculer la cohérence (variance entre chunks)
        buy_pcts = [(d.get('BUY', 0) / sum(d.values())) * 100 for d in chunk_distributions]
        consistency = 100 - np.std(buy_pcts)
        
        self.bias_metrics['decision_consistency'] = consistency
        
        logger.info(f"\n📊 COHÉRENCE:")
        logger.info(f"  Std Dev BUY %: {np.std(buy_pcts):.2f}%")
        logger.info(f"  Cohérence:     {consistency:.2f}%")
        
        if consistency < 30:
            logger.warning(f"⚠️ DÉCISIONS INCOHÉRENTES: Comportement très variable")
        elif consistency > 90:
            logger.info(f"✅ DÉCISIONS COHÉRENTES: Comportement stable")

    def analyze_reward_correlation(self):
        """Analyse la corrélation entre actions et récompenses"""
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSE 6: CORRÉLATION ACTIONS-RÉCOMPENSES")
        logger.info(f"{'='*80}")
        
        if not self.action_history or not self.reward_history:
            logger.warning("Pas d'historique")
            return
        
        actions_array = np.array([a[0] if isinstance(a, np.ndarray) else a 
                                 for a in self.action_history[:len(self.reward_history)]])
        rewards_array = np.array(self.reward_history)
        
        correlation = np.corrcoef(actions_array, rewards_array)[0, 1]
        
        logger.info(f"\n📊 CORRÉLATION:")
        logger.info(f"  Pearson Correlation: {correlation:.4f}")
        logger.info(f"  Reward Mean:         {np.mean(rewards_array):.4f}")
        logger.info(f"  Reward Std:          {np.std(rewards_array):.4f}")
        
        if np.isnan(correlation):
            logger.warning("⚠️ Corrélation indéfinie (variance zéro?)")
        elif abs(correlation) < 0.1:
            logger.warning(f"⚠️ FAIBLE CORRÉLATION: Actions et récompenses peu liées")
        elif correlation > 0.5:
            logger.info(f"✅ BONNE CORRÉLATION: Actions bien récompensées")

    def analyze_model_weights(self):
        """Analyse les poids du modèle"""
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSE 7: POIDS DU MODÈLE")
        logger.info(f"{'='*80}")
        
        try:
            policy = self.model.policy
            
            logger.info(f"\n📊 ARCHITECTURE DU MODÈLE:")
            logger.info(f"  Policy Type: {type(policy).__name__}")
            
            # Analyser les couches
            if hasattr(policy, 'mlp_extractor'):
                logger.info(f"\n  Feature Extractor:")
                for name, param in policy.mlp_extractor.named_parameters():
                    if 'weight' in name:
                        mean = param.data.mean().item()
                        std = param.data.std().item()
                        logger.info(f"    {name}: mean={mean:.4f}, std={std:.4f}")
            
            # Analyser la couche de sortie
            if hasattr(policy, 'action_net'):
                logger.info(f"\n  Action Network:")
                for name, param in policy.action_net.named_parameters():
                    if 'weight' in name:
                        mean = param.data.mean().item()
                        std = param.data.std().item()
                        logger.info(f"    {name}: mean={mean:.4f}, std={std:.4f}")
            
            if hasattr(policy, 'value_net'):
                logger.info(f"\n  Value Network:")
                for name, param in policy.value_net.named_parameters():
                    if 'weight' in name:
                        mean = param.data.mean().item()
                        std = param.data.std().item()
                        logger.info(f"    {name}: mean={mean:.4f}, std={std:.4f}")
        
        except Exception as e:
            logger.warning(f"⚠️ Impossible d'analyser les poids: {e}")

    def generate_report(self, output_file: str = "model_behavior_analysis.json"):
        """Génère un rapport d'analyse"""
        logger.info(f"\n{'='*80}")
        logger.info("RAPPORT FINAL")
        logger.info(f"{'='*80}")
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_path": self.model_path,
            "total_actions_analyzed": len(self.action_history),
            "bias_metrics": self.bias_metrics,
            "action_distribution": dict(self.action_distribution),
        }
        
        # Verdict
        verdict = self._generate_verdict()
        report["verdict"] = verdict
        
        logger.info(f"\n🎯 VERDICT FINAL:")
        logger.info(f"\n{verdict['summary']}")
        
        logger.info(f"\n⚠️ PROBLÈMES DÉTECTÉS:")
        for issue in verdict['issues']:
            logger.info(f"  - {issue}")
        
        logger.info(f"\n✅ POINTS POSITIFS:")
        for positive in verdict['positives']:
            logger.info(f"  - {positive}")
        
        # Sauvegarder le rapport
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📁 Rapport sauvegardé: {output_path}")
        
        return report

    def _categorize_action(self, action_value: float) -> str:
        """Catégorise une action"""
        if action_value < -0.5:
            return "SELL"
        elif action_value > 0.5:
            return "BUY"
        else:
            return "HOLD"

    def _generate_verdict(self) -> dict:
        """Génère un verdict sur le comportement du modèle"""
        issues = []
        positives = []
        
        # Vérifier les biais
        if self.bias_metrics['buy_bias'] > 60:
            issues.append(f"Biais BUY extrême: {self.bias_metrics['buy_bias']:.2f}%")
        if self.bias_metrics['sell_bias'] > 60:
            issues.append(f"Biais SELL extrême: {self.bias_metrics['sell_bias']:.2f}%")
        if self.bias_metrics['hold_bias'] > 80:
            issues.append(f"Biais HOLD extrême: {self.bias_metrics['hold_bias']:.2f}%")
        
        # Vérifier la variance
        if self.bias_metrics['action_variance'] < 0.01:
            issues.append(f"Actions trop stables (variance: {self.bias_metrics['action_variance']:.6f})")
        elif self.bias_metrics['action_variance'] > 0.5:
            issues.append(f"Actions trop instables (variance: {self.bias_metrics['action_variance']:.6f})")
        else:
            positives.append(f"Variance d'actions acceptable: {self.bias_metrics['action_variance']:.4f}")
        
        # Vérifier l'entropie
        if self.bias_metrics['action_entropy'] < 0.3:
            issues.append(f"Décisions peu diversifiées (entropie: {self.bias_metrics['action_entropy']:.4f})")
        elif self.bias_metrics['action_entropy'] > 0.8:
            positives.append(f"Décisions bien diversifiées (entropie: {self.bias_metrics['action_entropy']:.4f})")
        
        # Vérifier la cohérence
        if self.bias_metrics['decision_consistency'] < 30:
            issues.append(f"Décisions incohérentes (cohérence: {self.bias_metrics['decision_consistency']:.2f}%)")
        elif self.bias_metrics['decision_consistency'] > 80:
            positives.append(f"Décisions cohérentes (cohérence: {self.bias_metrics['decision_consistency']:.2f}%)")
        
        # Résumé
        if len(issues) > 3:
            summary = "❌ MODÈLE BIAISÉ ET MAL ORGANISÉ - Nombreux problèmes détectés"
        elif len(issues) > 0:
            summary = "⚠️ MODÈLE AVEC PROBLÈMES - Certains biais détectés"
        else:
            summary = "✅ MODÈLE SAIN - Pas de biais majeur détecté"
        
        return {
            "summary": summary,
            "issues": issues,
            "positives": positives,
        }

    def run_full_analysis(self, asset: str, split: str = "train"):
        """Lance l'analyse complète"""
        logger.info("=" * 80)
        logger.info("ANALYSE COMPLÈTE DU COMPORTEMENT DU MODÈLE")
        logger.info("=" * 80)
        
        # Setup
        self.setup_environment(asset, split)
        if not self.load_model():
            return None
        
        # Analyses
        self.analyze_action_distribution(n_steps=5000)
        self.analyze_action_variance()
        self.analyze_action_entropy()
        self.analyze_action_sequences()
        self.analyze_decision_consistency()
        self.analyze_reward_correlation()
        self.analyze_model_weights()
        
        # Rapport
        report = self.generate_report(f"model_behavior_analysis_{asset}_{split}.json")
        
        return report


def main():
    analyzer = ModelBehaviorAnalyzer(
        model_path="checkpoints_final/adan_model_checkpoint_640000_steps.zip",
        config_path="config/config.yaml"
    )
    
    # Analyser sur différents splits
    for split in ["train", "test"]:
        for asset in ["BTCUSDT", "XRPUSDT"]:
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"# ANALYSE: {asset} ({split})")
            logger.info(f"{'#'*80}\n")
            
            analyzer.run_full_analysis(asset, split)


if __name__ == "__main__":
    main()
