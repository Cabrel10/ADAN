"""
ADAN - Agent de Décision Algorithmique Neuronal
==============================================

Un agent de trading algorithmique basé sur l'apprentissage par renforcement.
"""

__version__ = "0.1.0"
__author__ = "ADAN Team"

# Expose les modules principaux
from .data_processing import state_builder
from .common import config_validator
from .common import utils
from .data_processing import feature_engineer
from .trading import action_translator
from .training import trainer
