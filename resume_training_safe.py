#!/usr/bin/env python3
"""
Safe Resume Training Script - Garantit un VRAI resume, pas un relancement
Vérifie à chaque étape que c'est bien un resume
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_checkpoint(checkpoint_path):
    """Vérifier que le checkpoint est valide"""
    logger.info(f"🔍 Vérification du checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint NOT FOUND: {checkpoint_path}")
        return False
    
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    logger.info(f"   Taille: {size_mb:.1f} MB")
    
    if size_mb < 2.5:
        logger.error(f"❌ Checkpoint trop petit (< 2.5 MB): {size_mb:.1f} MB")
        return False
    
    logger.info("✅ Checkpoint valide")
    return True

def load_and_verify_model(checkpoint_path, env):
    """Charger le modèle et vérifier que c'est un resume"""
    from stable_baselines3 import PPO
    
    logger.info("📦 Chargement du modèle...")
    
    try:
        model = PPO.load(checkpoint_path, env=env)
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        return None
    
    # VÉRIFICATION CRITIQUE: num_timesteps
    num_steps = model.num_timesteps
    logger.info(f"📊 num_timesteps après chargement: {num_steps}")
    
    if num_steps < 160000:
        logger.error(f"❌ ERREUR: num_timesteps trop bas ({num_steps})")
        logger.error("   Cela signifie que le checkpoint n'a pas été chargé correctement!")
        return None
    
    if num_steps > 180000:
        logger.warning(f"⚠️  num_timesteps plus haut que prévu ({num_steps})")
        logger.warning("   Mais c'est OK, on continue")
    
    logger.info(f"✅ Resume confirmé: {num_steps} steps déjà effectués")
    return model

def test_resume_with_small_batch(model, steps=1000):
    """Tester le resume avec un petit batch (1000 steps)"""
    logger.info(f"\n🧪 TEST: Entraînement sur {steps} steps...")
    
    initial_steps = model.num_timesteps
    logger.info(f"   Steps avant: {initial_steps}")
    
    try:
        model.learn(total_timesteps=steps, log_interval=100)
        logger.info("✅ Entraînement test complété")
    except Exception as e:
        logger.error(f"❌ Erreur pendant l'entraînement: {e}")
        return False
    
    final_steps = model.num_timesteps
    logger.info(f"   Steps après: {final_steps}")
    
    expected_steps = initial_steps + steps
    if abs(final_steps - expected_steps) > 10:
        logger.error(f"❌ ERREUR: num_timesteps n'a pas augmenté correctement")
        logger.error(f"   Attendu: ~{expected_steps}, Obtenu: {final_steps}")
        return False
    
    logger.info(f"✅ Resume confirmé: +{final_steps - initial_steps} steps")
    return True

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("🚀 RESUME TRAINING - SAFE MODE")
    logger.info("=" * 80)
    
    # Configuration
    checkpoint_path = "/mnt/new_data/t10_training/checkpoints/w1/w1_model_170000_steps.zip"
    config_path = "/home/morningstar/Documents/trading/bot/config/config.yaml"
    
    # Step 1: Vérifier le checkpoint
    logger.info("\n[STEP 1] Vérification du checkpoint")
    if not verify_checkpoint(checkpoint_path):
        logger.error("❌ Checkpoint invalide, abandon")
        return False
    
    # Step 2: Charger la config
    logger.info("\n[STEP 2] Chargement de la configuration")
    try:
        from adan_trading_bot.common.config_loader import ConfigLoader
        config = ConfigLoader.load_config(config_path)
        logger.info("✅ Configuration chargée")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement de la config: {e}")
        return False
    
    # Step 3: Créer l'environnement
    logger.info("\n[STEP 3] Création de l'environnement")
    try:
        from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
        env = MultiAssetChunkedEnv(config=config)
        logger.info("✅ Environnement créé")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création de l'environnement: {e}")
        return False
    
    # Step 4: Charger le modèle
    logger.info("\n[STEP 4] Chargement du modèle")
    model = load_and_verify_model(checkpoint_path, env)
    if model is None:
        logger.error("❌ Impossible de charger le modèle, abandon")
        return False
    
    # Step 5: Test avec petit batch
    logger.info("\n[STEP 5] Test du resume avec 1000 steps")
    if not test_resume_with_small_batch(model, steps=1000):
        logger.error("❌ Test échoué, abandon")
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TOUS LES TESTS PASSÉS - RESUME CONFIRMÉ")
    logger.info("=" * 80)
    logger.info("\n🎯 Prochaine étape: Lancer le resume complet (80k steps)")
    logger.info("   Commande: python resume_training_full.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
