#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chargeur de données pour le projet ADAN.
Charge les données de trading à partir de fichiers parquet organisés par actif et timeframe.
"""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

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

        # Configuration de la taille des chunks par timeframe
        # Priorité: worker_config.chunk_sizes > config.data.chunk_sizes > défauts optimisés
        default_chunk_sizes = {"5m": 5328, "1h": 242, "4h": 111}
        cfg_chunk_sizes = (
            (self.worker_config.get("chunk_sizes") or {})
            or self.config.get("data", {}).get("chunk_sizes", {})
        )
        # Conserver uniquement les timeframes demandés, sinon fallback sur défaut
        self.chunk_sizes = {}
        for tf in self.timeframes:
            if isinstance(cfg_chunk_sizes, dict) and tf in cfg_chunk_sizes:
                self.chunk_sizes[tf] = int(cfg_chunk_sizes[tf])
            else:
                # défaut si dispo, sinon utiliser longueur complète (sera bornée plus tard)
                self.chunk_sizes[tf] = int(default_chunk_sizes.get(tf, 10_000_000))

        # Initialise le nombre total de chunks en fonction des données disponibles
        self.total_chunks = self._calculate_total_chunks()
        logger.info(f"Total chunks disponibles: {self.total_chunks}")

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

        # Configuration du parallélisme
        self.max_workers = min(8, (os.cpu_count() or 4) * 2)
        logger.info(
            f"Chargement des données pour {len(self.assets_list)} actifs et "
            f"{len(self.timeframes)} timeframes en parallèle (max {self.max_workers} workers)"
        )
        logger.debug(f"Actifs: {self.assets_list}")
        logger.debug(f"Timeframes: {self.timeframes}")
        logger.debug(f"Jeu de données: {self.data_split}")

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

        Cherche dans l'emplacement :
        {base_dir}/{data_split}/{asset}/{timeframe}.parquet

        Args:
            asset: Symbole de l'actif (ex: 'BTC')
            timeframe: Période de temps (ex: '5m', '1h' ou '5m_1h' pour une combinaison)

        Returns:
            Chemin vers le fichier de données

        Raises:
            FileNotFoundError: Si le fichier n'est pas trouvé
        """
        # Mapping for asset names to file system names (e.g., BTC -> BTCUSDT)
        asset_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
            # Ajout des timeframes comme actifs si nécessaire
            "5m": "BTCUSDT",
            "1h": "BTCUSDT",
            "4h": "BTCUSDT"
        }
        
        # Utiliser le nom d'actif mappé pour la construction du chemin
        file_system_asset = asset_mapping.get(asset, asset)
        
        # Obtenir le répertoire de base des indicateurs
        indicators_dir = Path(self.config["paths"]["indicators_data_dir"]).expanduser().resolve()
        
        # Si le timeframe contient un underscore, c'est une combinaison
        # On prend le premier timeframe de la combinaison
        base_timeframe = timeframe.split('_')[0] if '_' in timeframe else timeframe
        
        # Construire le chemin complet du fichier
        file_path = indicators_dir / self.data_split / file_system_asset / f"{base_timeframe}.parquet"
        
        # Vérifier si le fichier existe
        if not file_path.exists():
            # Essayer un autre format de chemin si le premier échoue
            file_path = indicators_dir / self.data_split / f"{file_system_asset}_{base_timeframe}.parquet"
            
            if not file_path.exists():
                # Vérifier si le répertoire parent existe
                parent_dir = file_path.parent
                if not parent_dir.exists():
                    raise FileNotFoundError(
                        f"Le répertoire n'existe pas: {parent_dir}\n"
                        f"Vérifiez que le chemin de base est correct: {indicators_dir}"
                    )
                raise FileNotFoundError(
                    f"Le fichier n'existe pas: {file_path}\n"
                    f"Vérifiez que le fichier existe et que les permissions sont correctes."
                )
                    
            # Vérifier s'il y a des fichiers dans le répertoire
            if parent_dir.exists() and any(parent_dir.iterdir()):
                available_files = [f.name for f in parent_dir.glob('*') if f.is_file()]
                raise FileNotFoundError(
                    f"Fichier {base_timeframe}.parquet introuvable dans {parent_dir}\n"
                    f"Fichiers disponibles: {', '.join(available_files) if available_files else 'Aucun fichier trouvé'}"
                )
            else:
                raise FileNotFoundError(
                    f"Aucun fichier trouvé dans le répertoire: {parent_dir}\n"
                    f"Vérifiez que les données ont été correctement générées."
                )
        
        logger.debug(f"Fichier trouvé: {file_path}")
        return file_path

    def _load_asset_timeframe(self, asset: str, timeframe: str) -> pd.DataFrame:
        """
        Charge les données d'un actif et d'un timeframe spécifique.

        Args:
            asset: Symbole de l'actif (ex: 'BTC')
            timeframe: Période de temps (ex: '5m', '1h')

        Returns:
            DataFrame contenant les données demandées

        Raises:
            FileNotFoundError: Si le fichier n'est pas trouvé
            ValueError: Si les données sont corrompues ou incomplètes
        """
        file_path = self._get_data_path(asset, timeframe)

        try:
            # Charge les données depuis le fichier parquet
            df = pd.read_parquet(file_path)

            # Vérifie que le DataFrame n'est pas vide
            if df.empty:
                raise ValueError(f"Le fichier {file_path} est vide.")

            # Vérifie les colonnes requises
            required_columns = {'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'}
            missing_columns = required_columns - set(df.columns)

            if missing_columns:
                raise ValueError(
                    f"Colonnes manquantes dans {file_path}: {missing_columns}"
                )

            logger.debug(f"Données chargées pour {asset} {timeframe}: {len(df)} lignes")
            return df

        except Exception as e:
            logger.error(f"Erreur lors du chargement de {file_path}: {str(e)}")
            raise

    def _load_asset_timeframe_parallel(self, asset: str, tf: str, max_retries: int = 3) -> Tuple[str, str, pd.DataFrame]:
        """
        Charge les données d'un actif et d'un timeframe spécifique.
        Méthode utilisée pour le chargement parallèle.

        Args:
            asset: Symbole de l'actif
            tf: Timeframe à charger
            max_retries: Nombre maximum de tentatives en cas d'échec

        Returns:
            Tuple (asset, timeframe, DataFrame)

        Raises:
            Exception: Si le chargement échoue après plusieurs tentatives
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Tentative {attempt + 1}/{max_retries} pour {asset} {tf}")
                df = self._load_asset_timeframe(asset, tf)
                logger.debug(f"Chargement réussi de {asset} {tf}: {len(df)} lignes")
                return asset, tf, df

            except Exception as e:
                last_error = e
                wait_time = (attempt + 1) * 2  # Attente exponentielle
                logger.warning(
                    f"Échec du chargement de {asset} {tf} (tentative {attempt + 1}/{max_retries}): {str(e)}. "
                    f"Nouvelle tentative dans {wait_time}s..."
                )
                time.sleep(wait_time)
        
        # Si on arrive ici, toutes les tentatives ont échoué
        error_msg = f"Impossible de charger {asset} {tf} après {max_retries} tentatives: {str(last_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def load_chunk(self, chunk_idx: int = 0) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Charge un chunk de données pour tous les actifs et timeframes configurés en parallèle.

        Args:
            chunk_idx: Index du chunk à charger (non utilisé, conservé pour compatibilité)

        Returns:
            Dictionnaire imbriqué {actif: {timeframe: DataFrame}}

        Raises:
            RuntimeError: Si le chargement échoue pour un ou plusieurs actifs/timeframes
        """
        start_time = time.time()
        data = {asset: {} for asset in self.assets_list}
        total_tasks = len(self.assets_list) * len(self.timeframes)
        failed_loads = []

        logger.info(f"Début du chargement du chunk {chunk_idx} pour {len(self.assets_list)} actifs et {len(self.timeframes)} timeframes")

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Création des tâches de chargement
                futures = {
                    executor.submit(self._load_asset_timeframe_parallel, asset, tf): (asset, tf)
                    for asset in self.assets_list
                    for tf in self.timeframes
                }

                # Traitement des résultats au fur et à mesure
                with tqdm(total=total_tasks, desc="Chargement des données") as pbar:
                    for future in as_completed(futures):
                        asset, tf = futures[future]
                        try:
                            asset, tf, df = future.result()
                            # Appliquer la taille de chunk cible par timeframe (prend les dernières lignes)
                            target_len = int(self.chunk_sizes.get(tf, len(df)))
                            if target_len <= 0:
                                target_len = len(df)
                            if len(df) > target_len:
                                df = df.iloc[-target_len:].reset_index(drop=True)
                            else:
                                # Si les données sont plus courtes que la cible, on log un avertissement
                                logger.warning(
                                    f"{asset} {tf}: données ({len(df)}) plus courtes que la taille de chunk cible ({target_len})."
                                )
                            data[asset][tf] = df
                            pbar.update(1)
                            pbar.set_postfix_str(f"{asset} {tf} - {len(df)} lignes")
                        except Exception as e:
                            logger.error(f"Échec du chargement de {asset} {tf}: {str(e)}")
                            failed_loads.append((asset, tf, str(e)))
                            # Continue processing other assets/timeframes even if one fails

            # Vérifier si tous les chargements ont réussi
            if failed_loads:
                failed_str = ", ".join(f"{asset}/{tf}" for asset, tf, _ in failed_loads)
                error_msg = (
                    f"Échec du chargement pour {len(failed_loads)}/{total_tasks} combinaisons actif/timeframe: {failed_str}"
                    f"\nDétails des erreurs:\n"
                )
                for asset, tf, err in failed_loads:
                    error_msg += f"- {asset} {tf}: {err}\n"
                
                # Si tous les chargements ont échoué, on lève une exception
                if len(failed_loads) == total_tasks:
                    raise RuntimeError(f"Tous les chargements ont échoué:\n{error_msg}")
                
                # Sinon, on log l'erreur mais on continue
                logger.error(error_msg)

            # Vérifier que toutes les données attendues ont été chargées
            for asset in self.assets_list:
                if asset not in data or not data[asset]:
                    logger.error(f"Aucune donnée chargée pour l'actif {asset}")
                    continue
                    
                for tf in self.timeframes:
                    if tf not in data[asset]:
                        logger.warning(f"Données manquantes pour {asset} {tf}")

            duration = time.time() - start_time
            logger.info(f"Chargement du chunk {chunk_idx} terminé en {duration:.2f} secondes")
            
            return data

        except Exception as e:
            logger.error(f"Erreur critique lors du chargement du chunk {chunk_idx}: {str(e)}")
            logger.exception("Détails de l'erreur:")
            raise

    def _calculate_total_chunks(self) -> int:
        """
        Calcule le nombre total de chunks disponibles pour les données chargées.

        Le calcul est basé sur la taille des données disponibles et la taille des chunks
        configurée pour chaque timeframe. Le nombre de chunks est déterminé par le timeframe
        ayant le moins de données par rapport à la taille de ses chunks.

        Returns:
            int: Nombre total de chunks disponibles
        """
        # Si max_chunks_per_episode est défini dans la configuration, on l'utilise
        max_chunks = self.config.get('environment', {}).get('max_chunks_per_episode')
        if max_chunks is not None:
            return int(max_chunks)

        # Sinon, on calcule le nombre de chunks en fonction des données disponibles
        min_chunks = float('inf')

        # Parcourir tous les actifs et timeframes pour trouver le plus petit nombre de chunks
        for asset in self.assets_list:
            for tf in self.timeframes:
                try:
                    # Charger les données pour cet actif et ce timeframe
                    df = self._load_asset_timeframe(asset, tf)
                    chunk_size = self.chunk_sizes.get(tf, len(df))

                    # Calculer le nombre de chunks pour ce timeframe
                    num_chunks = max(1, len(df) // chunk_size)

                    # Prendre le plus petit nombre de chunks parmi tous les timeframes
                    if num_chunks < min_chunks:
                        min_chunks = num_chunks

                except Exception as e:
                    logger.warning(
                        f"Erreur lors du calcul des chunks pour {asset} {tf}: {str(e)}"
                    )
                    continue

        # Si on n'a pas pu déterminer le nombre de chunks, on retourne 1 par défaut
        if min_chunks == float('inf'):
            logger.warning(
                "Impossible de déterminer le nombre de chunks, utilisation de la valeur par défaut (1)"
            )
            return 1

        logger.info(f"Nombre total de chunks calculé : {min_chunks}")
        return min_chunks
