#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chargeur de données pour le projet ADAN.
Charge les données de trading à partir de fichiers parquet organisés par actif et timeframe.
"""
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..common.config_loader import ConfigLoader

# Configuration du logger
logger = logging.getLogger(__name__)


class ChunkedDataLoader:
    """
    Chargeur de données pour le projet ADAN.
    Charge les données de trading à partir de fichiers parquet organisés par actif et timeframe.
    """

    def __init__(self, config, worker_config):
        """
        Initialise le chargeur de données.

        Args:
            config: Configuration principale de l'application
            worker_config: Configuration spécifique au worker contenant:
                - timeframes: liste des timeframes à charger (ex: ["5m", "1h"])
                - data_split: jeu de données à charger (ex: 'train', 'validation', 'test')
                - assets: liste des actifs à charger (optionnel, utilise la config par défaut si non spécifié)
        """
        self.config = ConfigLoader.resolve_env_vars(config)
        self.worker_config = ConfigLoader.resolve_env_vars(
            worker_config, root_config=self.config
        )

        # 1) Timeframes à charger
        self.timeframes = self.worker_config.get(
            'timeframes',
            self.config['data'].get('timeframes', [])
        )

        # 2) Split (train / test / train_stress_test…)
        self.data_split = self.worker_config.get(
            'data_split_override',
            self.config['data'].get('data_split', 'train')
        )

        # 3) Actifs (permet override si besoin)
        self.assets_list = self.worker_config.get(
            'assets',  # Utilise directement 'assets' du worker_config
            self.config.get('environment', {}).get('assets', [])  # Fallback sur la config environment
        )

        # Initialise le dictionnaire des features par timeframe
        self.features_by_timeframe = self._init_features_by_timeframe()
        
        # Vérifie que tous les timeframes demandés sont pris en charge
        self._validate_timeframes()
        
        # Initialise le nombre total de chunks (1 car toutes les données sont chargées en une fois)
        self.total_chunks = 1

        logger.info(f"Initialisation du ChunkedDataLoader avec {len(self.assets_list)} actifs et {len(self.timeframes)} timeframes")
        logger.debug(f"Actifs: {self.assets_list}")
        logger.debug(f"Timeframes: {self.timeframes}")
        logger.debug(f"Jeu de données: {self.data_split}")

        # Vérifie que nous avons des actifs et des timeframes
        if not self.assets_list:
            raise ValueError(
                "Aucun actif défini dans la configuration du worker ou la configuration principale."
            )
        if not self.timeframes:
            raise ValueError("Aucun timeframe défini dans la configuration du worker.")

        logger.info(
            f"Chargement des données pour les actifs: {self.assets_list} "
            f"et timeframes: {self.timeframes}"
        )

    def _validate_timeframes(self):
        """
        Vérifie que tous les timeframes demandés sont pris en charge par la configuration.
        
        Raises:
            ValueError: Si un timeframe demandé n'est pas pris en charge
        """
        supported_timeframes = self.config.get("data", {}).get("features_per_timeframe", {}).keys()
        
        for tf in self.timeframes:
            if tf not in supported_timeframes:
                raise ValueError(
                    f"Le timeframe '{tf}' n'est pas pris en charge. "
                    f"Timeframes disponibles: {list(supported_timeframes)}"
                )
    
    def _init_features_by_timeframe(self) -> Dict[str, List[str]]:
        """
        Initialise le dictionnaire des features par timeframe à partir de la configuration.

        Returns:
            Dictionnaire des features par timeframe
        """
        features_by_timeframe = {}
        timeframe_features = self.config.get("data", {}).get("features_per_timeframe", {})

        for timeframe in self.timeframes:
            if timeframe in timeframe_features:
                features = timeframe_features[timeframe]
                if not isinstance(features, list):
                    features = [features]
                features_by_timeframe[timeframe] = features

        if not features_by_timeframe:
            raise ValueError(
                "Aucune configuration de features trouvée pour les timeframes spécifiés."
            )

        return features_by_timeframe

    def _get_data_path(self, asset: str, timeframe: str) -> Path:
        """
        Construit le chemin vers le fichier de données pour un actif et un timeframe donnés.
        
        Cherche dans plusieurs emplacements possibles :
        1. {base_dir}/data/processed/indicators/{asset}/{timeframe}.parquet
        2. {base_dir}/{asset}/{timeframe}.parquet
        3. {base_dir}/indicators/{asset}/{timeframe}.parquet

        Args:
            asset: Symbole de l'actif (ex: 'BTC')
            timeframe: Période de temps (ex: '5m', '1h')

        Returns:
            Chemin vers le fichier de données

        Raises:
            FileNotFoundError: Si le fichier n'est trouvé dans aucun des emplacements
        """
        try:
            # Liste des emplacements à essayer
            possible_paths = []
            
            # 1. Chemin de base depuis la configuration
            if "paths" in self.config and "base_dir" in self.config["paths"]:
                base_dir = Path(self.config["paths"]["base_dir"]).expanduser().resolve()
                possible_paths.append(base_dir / "data" / "processed" / "indicators" / asset / f"{timeframe}.parquet")
                
            # 2. Chemin depuis indicators_data_dir
            if "paths" in self.config and "indicators_data_dir" in self.config["paths"]:
                indicators_dir = Path(self.config["paths"]["indicators_data_dir"]).expanduser().resolve()
                possible_paths.append(indicators_dir / asset / f"{timeframe}.parquet")
            
            # 3. Chemin relatif au répertoire courant
            possible_paths.append(Path.cwd() / "data" / "processed" / "indicators" / asset / f"{timeframe}.parquet")
            
            # 4. Chemin relatif pour compatibilité ascendante
            possible_paths.append(Path.cwd() / "indicators" / asset / f"{timeframe}.parquet")
            
            # Essayer chaque chemin jusqu'à en trouver un qui existe
            for file_path in possible_paths:
                file_path = file_path.resolve()
                if file_path.exists():
                    logger.info(f"Fichier trouvé à l'emplacement: {file_path}")
                    return file_path
                else:
                    logger.debug(f"Fichier non trouvé à l'emplacement: {file_path}")
                    logger.debug(f"Attempting to load file: {file_path}")
                    logger.debug(f"Attempting to load file: {file_path}")
            
            # Si aucun chemin n'a fonctionné, essayer de créer le répertoire
            last_path = possible_paths[0]
            try:
                last_path.parent.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Aucun fichier trouvé, création du répertoire: {last_path.parent}")
                return last_path
            except Exception as e:
                logger.error(f"Impossible de créer le répertoire {last_path.parent}: {e}")
                raise FileNotFoundError(f"Impossible de trouver ou créer le fichier pour {asset} {timeframe} dans aucun des emplacements: {[str(p) for p in possible_paths]}")

        except Exception as e:
            error_msg = f"Erreur lors de la construction du chemin des données pour {asset} {timeframe}: {str(e)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e

    def load_chunk(self, chunk_index: int = 0) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Charge un morceau de données pour les actifs et timeframes configurés.

        Args:
            chunk_index: Index du morceau à charger (non utilisé pour l'instant)

        Returns:
            Dictionnaire imbriqué de DataFrames: {asset: {timeframe: df}}
        """
        data = {}

        for asset in self.assets_list:
            data[asset] = {}

            for timeframe in self.timeframes:
                file_path = self._get_data_path(asset, timeframe)

                try:
                    # Charge les données depuis le fichier parquet
                    df = pd.read_parquet(file_path)

                    # Vérifie que le DataFrame n'est pas vide
                    if df.empty:
                        logger.warning(f"Le fichier {file_path} est vide.")
                        continue

                    # Vérifie que les colonnes requises sont présentes
                    required_columns = {'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'}
                    missing_columns = required_columns - set(df.columns)
                    if missing_columns:
                        raise ValueError(
                            f"Colonnes manquantes dans {file_path}: {missing_columns}"
                        )

                    # Stocke le DataFrame dans la structure de données de sortie
                    data[asset][timeframe] = df

                    logger.info(
                        f"Données chargées pour {asset} {timeframe}: {len(df)} lignes"
                    )

                except FileNotFoundError:
                    logger.error(f"Fichier introuvable: {file_path}")
                    raise
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de {file_path}: {str(e)}")
                    raise

        return data
