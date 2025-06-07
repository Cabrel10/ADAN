#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'entraînement ADAN amélioré avec barre de progression et métriques claires.
"""
import os
import sys
import argparse
import time
import signal
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import get_path, load_config
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.data_processing.data_loader import load_training_data
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.agent.ppo_agent import create_ppo_agent
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import logging

# Désactiver les logs verbeux
logging.getLogger().setLevel(logging.WARNING)
for handler in logging.getLogger().handlers:
    handler.setLevel(logging.WARNING)

class ProgressCallback(BaseCallback):
    """Callback pour afficher la
