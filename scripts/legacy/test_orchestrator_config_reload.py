#!/usr/bin/env python3
"""
Test d'intégration du ConfigWatcher avec le TrainingOrchestrator.

Ce script permet de tester les mises à jour
dynamiques des paramètres pendant l'exécution
avec un fichier de configuration externe.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.common.utils import get_logger  # noqa: E402
from adan_trading_bot.training.training_orchestrator import (  # noqa: E402
    TrainingOrchestrator
)

# Configuration du logger
logger = get_logger(__name__)


class LoggingCallback(BaseCallback):
    """
    Callback personnalisé pour surveiller et mettre à jour
    dynamiquement les paramètres d'entraînement.

    Ce callback permet de:
    - Surveiller les paramètres (learning rate, coefficient d'entropie)
    - Détecter les changements de configuration en temps réel
    - Appliquer dynamiquement les mises à jour
    - Journaliser les changements pour analyse
    """

    def __init__(
        self,
        config_watcher: Any,
        verbose: int = 0,
    ) -> None:
        """
        Initialise le callback avec le config_watcher.
        
        Args:
            config_watcher: Instance du ConfigWatcher pour surveiller les
                changements de configuration pendant l'entraînement
            verbose: Niveau de verbosité (0=aucun, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.config_watcher = config_watcher
        self.last_lr: Optional[float] = None
        self.last_ent_coef: Optional[float] = None
        self.updated_lr: Optional[float] = None
        self.updated_ent_coef: Optional[float] = None
        self.updated_clip_range: Optional[float] = None
        self.last_clip_range: Optional[float] = None

    def _on_rollout_start(self) -> None:
        """
        Appelé par Stable-Baselines3 avant chaque nouveau rollout.
        
        Cette méthode est le point d'entrée principal pour appliquer les mises à jour
        des paramètres d'entraînement. Elle est appelée avant le début de chaque
        nouvel épisode d'entraînement.
        """
        try:
            self._apply_parameter_updates()
        except Exception as e:
            logger.error(
                "[Callback] Erreur dans _on_rollout_start: %s",
                str(e),
                exc_info=True
            )

    def _on_step(self) -> bool:
        """
        Callback appelé à chaque étape d'entraînement.

        Cette méthode est appelée par Stable-Baselines3 à chaque étape
        d'entraînement. Elle effectue les actions suivantes :
        1. Vérifie les mises à jour de configuration (toutes les 10 étapes)
        2. Journalise les paramètres actuels (toutes les 10 étapes)
        3. Met à jour les dernières valeurs connues des paramètres
        
        La journalisation n'est effectuée que toutes les 10 étapes pour éviter
        une surcharge des logs et maintenir de bonnes performances.

        Returns:
            bool: Toujours retourne True pour indiquer que l'entraînement doit
                  continuer. Retourner False arrêterait l'entraînement.
                  
        Note:
            Les paramètres suivis sont enregistrés pour TensorBoard :
            - train/learning_rate: Le taux d'apprentissage actuel
            - train/ent_coef: Le coefficient d'entropie actuel
        """
        # Vérifier les mises à jour et journaliser toutes les 10 étapes
        if self.n_calls % 10 == 0:
            try:
                # Vérifier les mises à jour de configuration
                self._check_for_config_updates()
                
                # Récupérer les valeurs actuelles
                current_lr = self._get_current_learning_rate()
                current_ent_coef = self._get_current_entropy_coef()
                current_clip_range = self._get_current_clip_range()
                
                if current_lr is not None and current_ent_coef is not None and current_clip_range is not None:
                    # Mettre à jour les dernières valeurs connues
                    self._update_last_parameters(current_lr, current_ent_coef, current_clip_range)
                    
                    # Enregistrer les valeurs pour TensorBoard
                    self.logger.record("train/learning_rate", current_lr)
                    self.logger.record("train/ent_coef", current_ent_coef)
                    self.logger.record("train/clip_range", current_clip_range)
                    
                    # Journaliser en mode détaillé si activé
                    if self.verbose >= 2:
                        self._log_current_parameters(current_lr, current_ent_coef, current_clip_range)
                        
            except Exception as e:
                logger.error(
                    "Erreur lors de la journalisation (étape %d): %s",
                    self.num_timesteps, str(e),
                    exc_info=self.verbose >= 2
                )
                
        return True

    def _get_current_learning_rate(self) -> float:
        """
        Récupère le taux d'apprentissage actuel du modèle.
        
        Cette méthode accède directement à l'optimiseur du modèle pour obtenir
        le taux d'apprentissage actuel. Si l'optimiseur n'est pas disponible ou
        en cas d'erreur, retourne 0.0.
        
        La méthode est conçue pour être robuste et ne jamais lever d'exception,
        même si la structure du modèle change ou si l'optimiseur n'est pas
        correctement initialisé.
        
        Returns:
            float: Le taux d'apprentissage actuel ou 0.0 en cas d'erreur
            
        Note:
            - Vérifie d'abord si le modèle et la politique existent
            - Vérifie si l'optimiseur est disponible
            - Accède au premier groupe de paramètres de l'optimiseur
            - Retourne la valeur du learning rate ou 0.0 en cas d'échec
            
        Exemple:
            >>> lr = callback._get_current_learning_rate()
            >>> print(f"Taux d'apprentissage actuel: {lr}")
        """
        try:
            # Vérifications de sécurité
            if not hasattr(self, 'model') or not hasattr(self.model, 'policy'):
                logger.debug("Modèle ou politique non disponible")
                return 0.0
                
            if not hasattr(self.model.policy, 'optimizer'):
                logger.debug("Aucun optimiseur trouvé dans la politique du modèle")
                return 0.0
                
            # Accès sécurisé au learning rate
            optimizer = self.model.policy.optimizer
            if not hasattr(optimizer, 'param_groups') or not optimizer.param_groups:
                logger.debug("Aucun groupe de paramètres trouvé dans l'optimiseur")
                return 0.0
                
            # Récupération du learning rate du premier groupe de paramètres
            lr = float(optimizer.param_groups[0]['lr'])
            
            # Validation de la valeur récupérée
            if not isinstance(lr, (int, float)) or lr <= 0:
                logger.warning("Valeur de learning rate invalide: %s", lr)
                return 0.0
                
            return lr
            
        except Exception as e:
            logger.debug(
                "Erreur lors de la récupération du taux d'apprentissage: %s",
                str(e),
                exc_info=self.verbose >= 2
            )
            return 0.0

    def _get_current_entropy_coef(self) -> float:
        """
        Récupère le coefficient d'entropie actuel du modèle.
        
        Cette méthode tente de récupérer le coefficient d'entropie actuel
        utilisé par l'algorithme PPO. Le coefficient d'entropie est un
        hyperparamètre important qui contrôle l'exploration en pénalisant
        les politiques trop certaines.
        
        La méthode est conçue pour être robuste et ne jamais lever d'exception,
        même si la structure du modèle change ou si le coefficient n'est pas
        directement accessible.
        
        Returns:
            float: Le coefficient d'entropie actuel ou 0.0 en cas d'erreur
            
        Note:
            - Vérifie d'abord si le modèle existe
            - Vérifie si le coefficient d'entropie est disponible
            - Convertit la valeur en float pour assurer la cohérence
            - Retourne 0.0 si le coefficient n'est pas trouvé ou en cas d'erreur
            
        Exemple:
            >>> ent_coef = callback._get_current_entropy_coef()
            >>> print(f"Coefficient d'entropie actuel: {ent_coef}")
        """
        try:
            # Vérifications de sécurité
            if not hasattr(self, 'model'):
                logger.debug("Modèle non disponible pour la récupération du coefficient d'entropie")
                return 0.0
                
            # Vérifier si le coefficient d'entropie est disponible
            if not hasattr(self.model, 'ent_coef'):
                logger.debug("Aucun coefficient d'entropie trouvé dans le modèle")
                return 0.0
                
            # Récupération et conversion du coefficient
            ent_coef = self.model.ent_coef
            
            # Si c'est un tenseur (cas de SB3), on extrait la valeur scalaire
            if hasattr(ent_coef, 'item'):
                ent_coef = ent_coef.item()
                
            # Conversion en float et validation
            ent_coef = float(ent_coef)
            
            # Validation de la valeur récupérée
            if not isinstance(ent_coef, (int, float)) or ent_coef < 0:
                logger.warning("Valeur de coefficient d'entropie invalide: %s", ent_coef)
                return 0.0
                
            return ent_coef
            
        except Exception as e:
            logger.debug(
                "Erreur lors de la récupération du coefficient d'entropie: %s",
                str(e),
                exc_info=self.verbose >= 2
            )
            return 0.0
    
    def _get_current_clip_range(self) -> float:
        """
        Récupère le clip range actuel du modèle.
        
        Cette méthode tente de récupérer le clip range actuel
        utilisé par l'algorithme PPO. Le clip range est un hyperparamètre
        important qui contrôle la plage de valeurs pour le clipping de l'objectif PPO.
        
        La méthode est conçue pour être robuste et ne jamais lever d'exception,
        même si la structure du modèle change ou si le clip range n'est pas
        directement accessible.
        
        Returns:
            float: Le clip range actuel ou 0.0 en cas d'erreur
            
        Note:
            - Vérifie d'abord si le modèle existe
            - Vérifie si le clip range est disponible
            - Convertit la valeur en float pour assurer la cohérence
            - Retourne 0.0 si le clip range n'est pas trouvé ou en cas d'erreur
            
        Exemple:
            >>> clip_range = callback._get_current_clip_range()
            >>> print(f"Clip range actuel: {clip_range}")
        """
        try:
            # Vérifications de sécurité
            if not hasattr(self, 'model'):
                logger.debug("Modèle non disponible pour la récupération du clip range")
                return 0.0
                
            if not hasattr(self.model, 'clip_range'):
                logger.debug("Aucun clip range trouvé dans le modèle")
                return 0.0
                
            # Récupération et conversion du clip range
            clip_range = self.model.clip_range
            
            # Si c'est un callable (cas de SB3), on l'appelle avec une valeur par défaut
            if callable(clip_range):
                clip_range = clip_range(1.0) # Passer une valeur arbitraire, car elle est souvent ignorée pour les constantes
                
            # Conversion en float et validation
            clip_range = float(clip_range)
            
            # Validation de la valeur récupérée
            if not isinstance(clip_range, (int, float)) or clip_range < 0:
                logger.warning("Valeur de clip range invalide: %s", clip_range)
                return 0.0
                
            return clip_range
            
        except Exception as e:
            logger.debug(
                "Erreur lors de la récupération du clip range: %s",
                str(e),
                exc_info=self.verbose >= 2
            )
            return 0.0
    
    def _log_current_parameters(self, lr: float, ent_coef: float, clip_range: float) -> None:
        """
        Affiche les paramètres actuels du modèle dans les logs.
        
        Cette méthode formate et enregistre les paramètres d'entraînement actuels
        (taux d'apprentissage, coefficient d'entropie et clip range) dans les logs avec un format
        clair et cohérent.
        
        Args:
            lr: Taux d'apprentissage actuel (doit être un nombre positif)
            ent_coef: Coefficient d'entropie actuel (doit être un nombre positif ou nul)
            clip_range: Clip range actuel (doit être un nombre positif ou nul)
            
        Note:
            - Le taux d'apprentissage est affiché en notation scientifique avec 2 décimales
            - Le coefficient d'entropie est affiché avec 6 décimales pour une bonne précision
            - Le clip range est affiché avec 6 décimales pour une bonne précision
            - Le numéro d'étape actuel est inclus pour le suivi temporel
            
        Exemple de sortie:
            [Callback] Étape 100: Taux d'apprentissage: 3.00e-04, Coef. entropie: 0.010000, Clip range: 0.200000
            
        Raises:
            TypeError: Si les paramètres ne sont pas numériques
            ValueError: Si les paramètres sont négatifs
        """
        try:
            # Validation des entrées
            if not isinstance(lr, (int, float)) or not isinstance(ent_coef, (int, float)) or not isinstance(clip_range, (int, float)):
                raise TypeError("Les paramètres doivent être des nombres")
                
            if lr < 0 or ent_coef < 0 or clip_range < 0:
                raise ValueError("Les paramètres ne peuvent pas être négatifs")
                
            # Journalisation avec formatage approprié
            logger.info(
                "[Callback] Étape %d: Taux d'apprentissage: %.2e, Coef. entropie: %.6f, Clip range: %.6f",
                self.n_calls, lr, ent_coef, clip_range
            )
            
        except Exception as e:
            logger.error(
                "Erreur lors de la journalisation des paramètres: %s",
                str(e),
                exc_info=self.verbose >= 1
            )

    def _update_last_parameters(self, lr: float, ent_coef: float, clip_range: float) -> None:
        """
        Affiche les paramètres actuels du modèle dans les logs.
        
        Cette méthode formate et enregistre les paramètres d'entraînement actuels
        (taux d'apprentissage et coefficient d'entropie) dans les logs avec un format
        clair et cohérent.
        
        Args:
            lr: Taux d'apprentissage actuel (doit être un nombre positif)
            ent_coef: Coefficient d'entropie actuel (doit être un nombre positif ou nul)
            
        Note:
            - Le taux d'apprentissage est affiché en notation scientifique avec 2 décimales
            - Le coefficient d'entropie est affiché avec 6 décimales pour une bonne précision
            - Le numéro d'étape actuel est inclus pour le suivi temporel
            
        Exemple de sortie:
            [Callback] Étape 100: Taux d'apprentissage: 3.00e-04, Coef. entropie: 0.010000
            
        Raises:
            TypeError: Si les paramètres ne sont pas numériques
            ValueError: Si les paramètres sont négatifs
        """
        try:
            # Validation des entrées
            if not isinstance(lr, (int, float)) or not isinstance(ent_coef, (int, float)):
                raise TypeError("Les paramètres doivent être des nombres")
                
            if lr < 0 or ent_coef < 0:
                raise ValueError("Les paramètres ne peuvent pas être négatifs")
                
            # Journalisation avec formatage approprié
            logger.info(
                "[Callback] Étape %d: Taux d'apprentissage: %.2e, Coef. entropie: %.6f",
                self.n_calls, lr, ent_coef
            )
            
        except Exception as e:
            logger.error(
                "Erreur lors de la journalisation des paramètres: %s",
                str(e),
                exc_info=self.verbose >= 1
            )

    def _update_last_parameters(self, lr: float, ent_coef: float, clip_range: float) -> None:
        """
        Met à jour la référence des derniers paramètres connus du modèle.
        
        Cette méthode stocke en mémoire les dernières valeurs connues des paramètres
        d'entraînement pour permettre une détection efficace des changements lors
        des prochaines itérations.
        
        Args:
            lr: Dernier taux d'apprentissage connu (doit être un nombre positif)
            ent_coef: Dernier coefficient d'entropie connu (doit être un nombre positif ou nul)
            
        Note:
            - Les valeurs sont stockées telles quelles, sans validation supplémentaire
            - Cette méthode est typiquement appelée après avoir validé les nouvelles valeurs
            - Les valeurs sont utilisées pour détecter les changements de configuration
            
        Raises:
            TypeError: Si les paramètres ne sont pas numériques
            
        Exemple:
            >>> callback._update_last_parameters(0.001, 0.01)
        """
        try:
            # Validation des types
            if not isinstance(lr, (int, float)) or not isinstance(ent_coef, (int, float)):
                raise TypeError("Les paramètres doivent être des nombres")
                
            # Mise à jour des attributs
            self.last_lr = float(lr)
            self.last_ent_coef = float(ent_coef)
            self.last_clip_range = float(clip_range)
            
            if self.verbose >= 2:
                logger.debug(
                    "Mise à jour des références - LR: %.2e, Entropy: %.6f, ClipRange: %.6f",
                    self.last_lr, self.last_ent_coef, self.last_clip_range
                )
                
        except Exception as e:
            logger.error(
                "Erreur lors de la mise à jour des références des paramètres: %s",
                str(e),
                exc_info=self.verbose >= 1
            )

    def _check_for_config_updates(self) -> None:
        """
        Vérifie les mises à jour de configuration et les traite si nécessaire.
        
        Cette méthode interroge le config_watcher pour détecter les changements
        dans la configuration d'entraînement et déclenche le traitement des mises à jour.
        
        Le processus de vérification inclut :
        1. Récupération de la configuration mise à jour via le config_watcher
        2. Vérification de la validité de la configuration
        3. Traitement des mises à jour pour chaque paramètre (learning rate, coefficient d'entropie)
        
        Note:
            - La méthode est conçue pour être tolérante aux erreurs
            - Les échecs de traitement d'un paramètre n'empêchent pas le traitement des autres
            - Les erreurs sont enregistrées dans les logs avec un niveau de détail dépendant de self.verbose
            
        Raises:
            AttributeError: Si config_watcher n'est pas initialisé
            
        Exemple:
            >>> callback._check_for_config_updates()
        """
        try:
            # Vérification de l'initialisation du config_watcher
            if not hasattr(self, 'config_watcher') or self.config_watcher is None:
                logger.error("ConfigWatcher non initialisé")
                return
                
            # Récupération de la configuration mise à jour
            updated_config = self.config_watcher.get_config('training')
            
            # Vérification de la configuration
            if not isinstance(updated_config, dict):
                if self.verbose >= 2:
                    logger.debug(
                        "[Callback] Aucune configuration d'entraînement valide disponible"
                    )
                return
                
            # Créer une copie du dictionnaire pour éviter de modifier l'original
            updated_config = updated_config.copy()
            
            # Convertir explicitement les valeurs numériques
            if 'learning_rate' in updated_config:
                try:
                    updated_config['learning_rate'] = float(
                        updated_config['learning_rate']
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Impossible de convertir le learning_rate en nombre: %s",
                        updated_config['learning_rate']
                    )
                    
            if 'ent_coef' in updated_config:
                try:
                    updated_config['ent_coef'] = float(
                        updated_config['ent_coef']
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Impossible de convertir ent_coef en nombre: %s",
                        updated_config['ent_coef']
                    )
                    
            if 'clip_range' in updated_config:
                try:
                    updated_config['clip_range'] = float(
                        updated_config['clip_range']
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Impossible de convertir clip_range en nombre: %s",
                        updated_config['clip_range']
                    )
                
            if self.verbose >= 2:
                logger.debug(
                    "[Callback] Configuration reçue: %s", 
                    {k: v for k, v in updated_config.items() if k in ['learning_rate', 'ent_coef']}
                )
            
            # Traitement des mises à jour pour chaque paramètre
            try:
                self._process_learning_rate_update(updated_config)
            except Exception as e:
                logger.error(
                    "Erreur lors du traitement de la mise à jour du learning rate: %s",
                    str(e),
                    exc_info=self.verbose >= 1
                )
                
            try:
                self._process_entropy_coef_update(updated_config)
            except Exception as e:
                logger.error(
                    "Erreur lors du traitement de la mise à jour du coefficient d'entropie: %s",
                    str(e),
                    exc_info=self.verbose >= 1
                )
            try:
                self._process_clip_range_update(updated_config)
            except Exception as e:
                logger.error(
                    "Erreur lors du traitement de la mise à jour du clip range: %s",
                    str(e),
                    exc_info=self.verbose >= 1
                )
                
        except Exception as e:
            logger.error(
                "Erreur inattendue lors de la vérification des mises à jour de configuration: %s",
                str(e),
                exc_info=self.verbose >= 1
            )

    def _process_learning_rate_update(self, config: Dict[str, Any]) -> None:
        """
        Traite la mise à jour du taux d'apprentissage à partir de la configuration.
        
        Cette méthode vérifie si une nouvelle valeur de taux d'apprentissage est présente
        dans la configuration et si elle diffère significativement de la dernière valeur connue.
        En cas de changement détecté, la nouvelle valeur est stockée pour application ultérieure.
        
        Args:
            config: Dictionnaire de configuration contenant potentiellement la clé 'learning_rate'
            
        Note:
            - La méthode utilise une tolérance de 1e-10 pour détecter les changements significatifs
            - Seules les valeurs numériques positives sont acceptées
            - Les changements sont enregistrés dans les logs avec l'ancienne et la nouvelle valeur
            
        Raises:
            ValueError: Si la valeur du learning rate est invalide ou négative
            TypeError: Si le type de la configuration ou de la valeur est incorrect
            
        Exemple:
            >>> config = {'learning_rate': 0.001}
            >>> callback._process_learning_rate_update(config)
        """
        try:
            # Vérification de la présence de la clé
            if 'learning_rate' not in config:
                if self.verbose >= 3:
                    logger.debug("Aucun taux d'apprentissage dans la configuration")
                return
                
            # Vérification du type de configuration
            if not isinstance(config, dict):
                raise TypeError("La configuration doit être un dictionnaire")
                
            # Extraction et validation de la valeur
            try:
                new_lr = float(config['learning_rate'])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Le taux d'apprentissage doit être un nombre, reçu: {config['learning_rate']}"
                ) from e
                
            if new_lr <= 0:
                raise ValueError(
                    f"Le taux d'apprentissage doit être positif, reçu: {new_lr}"
                )
                
            # Vérification du changement significatif
            if self.last_lr is None or abs(new_lr - self.last_lr) > 1e-10:
                self.updated_lr = new_lr
                logger.info(
                    "[Callback] Nouveau taux d'apprentissage détecté: %.2e -> %.2e",
                    self.last_lr or 0, new_lr
                )
                
                if self.verbose >= 1:
                    logger.debug(
                        "Taux d'apprentissage mis à jour de %.2e à %.2e",
                        self.last_lr or 0, new_lr
                    )
                    
        except Exception as e:
            logger.error(
                "Erreur lors du traitement de la mise à jour du taux d'apprentissage: %s",
                str(e),
                exc_info=self.verbose >= 1
            )
            raise  # Propager l'exception pour une gestion plus poussée

    def _process_entropy_coef_update(self, config: Dict[str, Any]) -> None:
        """
        Traite la mise à jour du coefficient d'entropie à partir de la configuration.
        
        Cette méthode vérifie si une nouvelle valeur de coefficient d'entropie est présente
        dans la configuration et si elle diffère significativement de la dernière valeur connue.
        En cas de changement détecté, la nouvelle valeur est stockée pour application ultérieure.
        
        Args:
            config: Dictionnaire de configuration contenant potentiellement la clé 'ent_coef'
            
        Note:
            - La méthode utilise une tolérance de 1e-10 pour détecter les changements significatifs
            - Seules les valeurs numériques positives ou nulles sont acceptées
            - Les changements sont enregistrés dans les logs avec l'ancienne et la nouvelle valeur
            
        Raises:
            ValueError: Si la valeur du coefficient d'entropie est invalide ou négative
            TypeError: Si le type de la configuration ou de la valeur est incorrect
            
        Exemple:
            >>> config = {'ent_coef': 0.01}
            >>> callback._process_entropy_coef_update(config)
        """
        try:
            # Vérification de la présence de la clé
            if 'ent_coef' not in config:
                if self.verbose >= 3:
                    logger.debug("Aucun coefficient d'entropie dans la configuration")
                return
                
            # Vérification du type de configuration
            if not isinstance(config, dict):
                raise TypeError("La configuration doit être un dictionnaire")
                
            # Extraction et validation de la valeur
            try:
                new_ent_coef = float(config['ent_coef'])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Le coefficient d'entropie doit être un nombre, reçu: {config['ent_coef']}"
                ) from e
                
            if new_ent_coef < 0:
                raise ValueError(
                    f"Le coefficient d'entropie ne peut pas être négatif, reçu: {new_ent_coef}"
                )
                
            # Vérification du changement significatif
            if (self.last_ent_coef is None or 
                    abs(new_ent_coef - self.last_ent_coef) > 1e-10):
                self.updated_ent_coef = new_ent_coef
                
                # Journalisation du changement
                logger.info(
                    "[Callback] Nouveau coefficient d'entropie détecté: %.6f -> %.6f",
                    self.last_ent_coef or 0, new_ent_coef
                )
                
                if self.verbose >= 1:
                    logger.debug(
                        "Coefficient d'entropie mis à jour de %.6f à %.6f",
                        self.last_ent_coef or 0, new_ent_coef
                    )
                    
        except Exception as e:
            logger.error(
                "Erreur lors du traitement de la mise à jour du coefficient d'entropie: %s",
                str(e),
                exc_info=self.verbose >= 1
            )
            raise  # Propager l'exception pour une gestion plus poussée

    def _process_clip_range_update(self, config: Dict[str, Any]) -> None:
        """
        Traite la mise à jour du clip range à partir de la configuration.
        
        Cette méthode vérifie si une nouvelle valeur de clip range est présente
        dans la configuration et si elle diffère significativement de la dernière valeur connue.
        En cas de changement détecté, la nouvelle valeur est stockée pour application ultérieure.
        
        Args:
            config: Dictionnaire de configuration contenant potentiellement la clé 'clip_range'
            
        Note:
            - La méthode utilise une tolérance de 1e-10 pour détecter les changements significatifs
            - Seules les valeurs numériques positives ou nulles sont acceptées
            - Les changements sont enregistrés dans les logs avec l'ancienne et la nouvelle valeur
            
        Raises:
            ValueError: Si la valeur du clip range est invalide ou négative
            TypeError: Si le type de la configuration ou de la valeur est incorrect
            
        Exemple:
            >>> config = {'clip_range': 0.01}
            >>> callback._process_clip_range_update(config)
        """
        try:
            # Vérification de la présence de la clé
            if 'clip_range' not in config:
                if self.verbose >= 3:
                    logger.debug("Aucun clip range dans la configuration")
                return
                
            # Vérification du type de configuration
            if not isinstance(config, dict):
                raise TypeError("La configuration doit être un dictionnaire")
                
            # Extraction et validation de la valeur
            try:
                new_clip_range = float(config['clip_range'])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Le clip range doit être un nombre, reçu: {config['clip_range']}"
                ) from e
                
            if new_clip_range < 0:
                raise ValueError(
                    f"Le clip range ne peut pas être négatif, reçu: {new_clip_range}"
                )
                
            # Vérification du changement significatif
            if (self.updated_clip_range is None or 
                    abs(new_clip_range - self.updated_clip_range) > 1e-10):
                self.updated_clip_range = new_clip_range
                
                # Journalisation du changement
                logger.info(
                    "[Callback] Nouveau clip range détecté: %.6f -> %.6f",
                    self.last_clip_range or 0, new_clip_range
                )
                
                if self.verbose >= 1:
                    logger.debug(
                        "Clip range mis à jour de %.6f à %.6f",
                        self.last_clip_range or 0, new_clip_range
                    )
                    
        except Exception as e:
            logger.error(
                "Erreur lors du traitement de la mise à jour du clip range: %s",
                str(e),
                exc_info=self.verbose >= 1
            )
            raise  # Propager l'exception pour une gestion plus poussée

    def _apply_parameter_updates(self) -> None:
        """
        Applique les mises à jour de paramètres en attente de manière sécurisée.
        
        Cette méthode orchestre l'application des mises à jour des paramètres d'entraînement
        en appelant séquentiellement les méthodes spécifiques à chaque paramètre.
        
        Le processus d'application inclut :
        1. Vérification de la disponibilité des mises à jour
        2. Application des mises à jour dans un ordre logique (d'abord le learning rate, puis l'entropie)
        3. Gestion robuste des erreurs pour chaque type de paramètre
        
        Note:
            - Les échecs d'application d'un paramètre n'empêchent pas l'application des autres
            - Toutes les exceptions sont capturées et journalisées avec un niveau de détail approprié
            - La méthode est conçue pour être tolérante aux erreurs et ne jamais interrompre l'entraînement
            
        Raises:
            Aucune exception n'est propagée pour éviter de perturber la boucle d'entraînement
            
        Exemple:
            >>> callback._apply_parameter_updates()
            # Applique toutes les mises à jour en attente de manière sécurisée
        """
        try:
            if self.verbose >= 2:
                logger.debug("[Callback] Vérification des mises à jour de paramètres à appliquer")
                
            # Appliquer d'abord le learning rate puis le coefficient d'entropie
            self._apply_learning_rate_update()
            self._apply_entropy_coef_update()
            self._apply_clip_range_update()
            
            if self.verbose >= 3:
                logger.debug("[Callback] Toutes les mises à jour de paramètres ont été traitées")
                
        except Exception as e:
            # Journalisation de l'erreur avec le niveau de détail approprié
            log_level = logger.error if self.verbose >= 1 else logger.debug
            log_level(
                "[Callback] Erreur lors de l'application des mises à jour de paramètres: %s",
                str(e),
                exc_info=self.verbose >= 1
            )
            
            # Réinitialisation des indicateurs de mise à jour en cas d'échec
            if self.verbose >= 2:
                logger.debug("[Callback] Réinitialisation des indicateurs de mise à jour après erreur")
            self.updated_lr = None
            self.updated_ent_coef = None

    def _apply_clip_range_update(self) -> None:
        """
        Applique la mise à jour du clip range si nécessaire.
        
        Cette méthode effectue les opérations suivantes :
        1. Vérifie si une mise à jour du clip range est en attente
        2. Vérifie que le modèle est correctement initialisé
        3. Applique la nouvelle valeur du clip range
        4. Met à jour les références internes et journalise le changement
        
        Note:
            - La méthode est tolérante aux erreurs et ne lève pas d'exception
            - Les erreurs sont journalisées avec un niveau de détail approprié
            - Les références sont mises à jour uniquement si la modification réussit
            
        Raises:
            Aucune exception n'est propagée pour éviter de perturber l'entraînement
            
        Exemple:
            >>> callback._apply_clip_range_update()
            # Met à jour le clip range si une mise à jour est en attente
        """
        try:
            # Vérification des prérequis
            if self.updated_clip_range is None:
                if self.verbose >= 3:
                    logger.debug("Aucune mise à jour de clip range en attente")
                return
                
            if not hasattr(self, 'model'):
                logger.warning("Modèle non initialisé pour la mise à jour du clip range")
                return
            
            # Sauvegarde de l'ancienne valeur pour le log
            old_clip_range = self.last_clip_range or 0.0
            new_clip_range = self.updated_clip_range
            
            # Validation de la nouvelle valeur
            if not isinstance(new_clip_range, (int, float)) or new_clip_range < 0:
                raise ValueError(
                    f"Valeur de clip range invalide: {new_clip_range}. "
                    f"Doit être un nombre positif ou nul."
                )
            
            # Application de la mise à jour
            self.model.clip_range = lambda _: new_clip_range
            
            # Journalisation du changement
            logger.info(
                "[Callback] Clip range mis à jour: %.6f -> %.6f",
                old_clip_range, new_clip_range
            )
            
            if self.verbose >= 1:
                logger.debug(
                    "Clip range appliqué avec succès à %.6f (précédent: %.6f)",
                    new_clip_range, old_clip_range
                )
            
            # Mise à jour des références
            self.last_clip_range = new_clip_range
            self.updated_clip_range = None
            
        except Exception as e:
            logger.error(
                "Erreur lors de la mise à jour du clip range: %s",
                str(e),
                exc_info=self.verbose >= 1
            )
            # Réinitialisation de la mise à jour en cas d'échec
            self.updated_clip_range = None

    def _apply_learning_rate_update(self) -> None:
        """
        Applique la mise à jour du taux d'apprentissage si nécessaire.
        
        Cette méthode effectue les opérations suivantes :
        1. Vérifie si une mise à jour du taux d'apprentissage est en attente
        2. Vérifie que le modèle et son optimiseur sont correctement initialisés
        3. Applique la nouvelle valeur de taux d'apprentissage à tous les groupes de paramètres
        4. Met à jour les références internes et journalise le changement
        
        Note:
            - La méthode est tolérante aux erreurs et ne lève pas d'exception
            - Les erreurs sont journalisées avec un niveau de détail approprié
            - Les références sont mises à jour uniquement si la modification réussit
            
        Raises:
            Aucune exception n'est propagée pour éviter de perturber l'entraînement
            
        Exemple:
            >>> callback._apply_learning_rate_update()
            # Met à jour le taux d'apprentissage si une mise à jour est en attente
        """
        try:
            # Vérification des prérequis
            if self.updated_lr is None:
                if self.verbose >= 3:
                    logger.debug("Aucune mise à jour de taux d'apprentissage en attente")
                return
                
            if not hasattr(self, 'model') or not hasattr(self.model, 'policy'):
                logger.warning("Modèle ou politique non initialisé pour la mise à jour du taux d'apprentissage")
                return
                
            if not hasattr(self.model.policy, 'optimizer'):
                logger.warning("Aucun optimiseur trouvé pour la mise à jour du taux d'apprentissage")
                return
            
            # Sauvegarde de l'ancienne valeur pour le log
            old_lr = self.last_lr or 0.0
            new_lr = self.updated_lr
            
            # Validation de la nouvelle valeur
            if not isinstance(new_lr, (int, float)) or new_lr <= 0:
                raise ValueError(
                    f"Valeur de taux d'apprentissage invalide: {new_lr}. "
                    f"Doit être un nombre positif."
                )
            
            # Application de la mise à jour à tous les groupes de paramètres
            optimizer = self.model.policy.optimizer
            if not hasattr(optimizer, 'param_groups') or not optimizer.param_groups:
                logger.warning("Aucun groupe de paramètres trouvé dans l'optimiseur")
                return
                
            for param_group in optimizer.param_groups:
                if 'lr' in param_group:
                    param_group['lr'] = new_lr
                else:
                    logger.warning("Clé 'lr' non trouvée dans un groupe de paramètres")
            
            # Journalisation du changement
            logger.info(
                "[Callback] Taux d'apprentissage mis à jour: %.2e -> %.2e",
                old_lr, new_lr
            )
            
            if self.verbose >= 1:
                logger.debug(
                    "Taux d'apprentissage appliqué avec succès à %.2e (précédent: %.2e)",
                    new_lr, old_lr
                )
            
            # Mise à jour des références
            self.last_lr = new_lr
            self.updated_lr = None
            
        except Exception as e:
            logger.error(
                "Erreur lors de la mise à jour du taux d'apprentissage: %s",
                str(e),
                exc_info=self.verbose >= 1
            )
            # Réinitialisation de la mise à jour en cas d'échec
            self.updated_lr = None

    def _apply_entropy_coef_update(self) -> None:
        """
        Applique la mise à jour du coefficient d'entropie si nécessaire.
        
        Cette méthode effectue les opérations suivantes :
        1. Vérifie si une mise à jour du coefficient d'entropie est en attente
        2. Vérifie que le modèle et son coefficient d'entropie sont accessibles
        3. Applique la nouvelle valeur du coefficient d'entropie
        4. Met à jour les références internes et journalise le changement
        
        Note:
            - La méthode est tolérante aux erreurs et ne lève pas d'exception
            - Les erreurs sont journalisées avec un niveau de détail approprié
            - Les références sont mises à jour uniquement si la modification réussit
            
        Raises:
            Aucune exception n'est propagée pour éviter de perturber l'entraînement
            
        Exemple:
            >>> callback._apply_entropy_coef_update()
            # Met à jour le coefficient d'entropie si une mise à jour est en attente
        """
        try:
            # Vérification des prérequis
            if self.updated_ent_coef is None:
                if self.verbose >= 3:
                    logger.debug("Aucune mise à jour du coefficient d'entropie en attente")
                return
                
            if not hasattr(self, 'model'):
                logger.warning("Modèle non initialisé pour la mise à jour du coefficient d'entropie")
                return
                
            if not hasattr(self.model, 'ent_coef'):
                logger.warning("Le modèle ne prend pas en charge la modification du coefficient d'entropie")
                return
            
            # Sauvegarde de l'ancienne valeur pour le log
            old_ent_coef = self.last_ent_coef or 0.0
            new_ent_coef = self.updated_ent_coef
            
            # Validation de la nouvelle valeur
            if not isinstance(new_ent_coef, (int, float)) or new_ent_coef < 0:
                raise ValueError(
                    f"Valeur de coefficient d'entropie invalide: {new_ent_coef}. "
                    f"Doit être un nombre positif ou nul."
                )
            
            # Application de la mise à jour
            self.model.ent_coef = new_ent_coef
            
            # Journalisation du changement
            logger.info(
                "[Callback] Coefficient d'entropie mis à jour: %.6f -> %.6f",
                old_ent_coef, new_ent_coef
            )
            
            if self.verbose >= 1:
                logger.debug(
                    "Coefficient d'entropie appliqué avec succès à %.6f (précédent: %.6f)",
                    new_ent_coef, old_ent_coef
                )
            
            # Mise à jour des références
            self.last_ent_coef = new_ent_coef
            self.updated_ent_coef = None
            
        except Exception as e:
            logger.error(
                "Erreur lors de la mise à jour du coefficient d'entropie: %s",
                str(e),
                exc_info=self.verbose >= 1
            )
            # Réinitialisation de la mise à jour en cas d'échec
            self.updated_ent_coef = None

def create_minimal_test_config() -> Dict[str, Any]:
    """
    Crée une configuration minimale pour les tests.
    
    Returns:
        Dict[str, Any]: Dictionnaire contenant la configuration de test minimale
    """
    # Chemin vers le dossier de données de test
    test_data_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "data", 
        "test_assets"
    )
    
    return {
        "num_environments": 1,
        "curriculum_learning": False,
        "shared_experience_buffer": False,
        "dynamic_adaptation": {"enabled": True},
        "config_dir": "config",
        "environment_config": {
            "data": {
                "data_dir": test_data_dir,
                "chunk_size": 100,  # Petit pour accélérer les tests
                "assets": ["BTC"],
                "timeframes": ["5m"],
                "file_format": "parquet"
            },
            "environment": {
                "initial_capital": 100.0
            },
            "portfolio": {},
            "trading": {
                "stop_loss": 0.01,  # 1% de stop-loss
                "take_profit": 0.02  # 2% de take-profit
            },
            "state": {
                "window_size": 5,  # Petit pour accélérer les tests
                "timeframes": ["5m"],
                "features_per_timeframe": {"5m": ["open", "high", "low", "close", "volume"]}
            }
        }
    }

def create_minimal_agent_config() -> Dict[str, Any]:
    """
    Crée une configuration minimale pour l'agent de test.
    
    Returns:
        Dict[str, Any]: Configuration minimale pour un agent PPO
    """
    return {
        # Configuration de la politique
        "policy": "MlpPolicy",
        
        # Hyperparamètres d'apprentissage
        "learning_rate": 3e-4,  # Taux d'apprentissage initial
        "n_steps": 10,  # Nombre d'étapes par mise à jour
        "batch_size": 16,  # Taille du batch pour les mises à jour
        "n_epochs": 1,  # Nombre d'époques par mise à jour
        "gamma": 0.99,  # Facteur de remise
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "verbose": 1,
        "device": "cpu"
    }

def modify_training_config_after_delay(
    config_path: Union[str, Path],
    initial_delay: float = 2.0,
    update_interval: float = 10.0,
    num_updates: int = 3
) -> None:
    """
    Modifie périodiquement la configuration d'entraînement pendant l'exécution.
    
    Cette fonction est conçue pour être exécutée dans un thread séparé et modifie
    dynamiquement les paramètres d'entraînement à intervalles réguliers.
    
    Args:
        config_path: Chemin vers le fichier de configuration à modifier
        initial_delay: Délai initial avant la première modification (en secondes)
        update_interval: Intervalle entre les modifications (en secondes)
        num_updates: Nombre total de modifications à effectuer
        
    Note:
        Les modifications incluent l'alternance du taux d'apprentissage entre
        deux valeurs et l'ajout progressif d'un coefficient d'entropie.
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error("❌ Fichier de configuration non trouvé: %s", config_path)
            return
            
        # Attente initiale pour laisser l'entraînement démarrer
        logger.info("⏳ Attente de %.1f secondes avant la première mise à jour...", initial_delay)
        time.sleep(initial_delay)
        
        for update_num in range(1, num_updates + 1):
            try:
                # 1. Lire la configuration actuelle
                with open(config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file) or {}
                
                # 2. Préparer les nouvelles valeurs
                old_lr = float(config.get('learning_rate', 3e-4))
                # Alterne entre deux valeurs de learning rate
                new_lr = 1e-5 if old_lr > 1e-4 else 1e-4
                
                old_ent_coef = float(config.get('ent_coef', 0.0))
                # Augmente progressivement le coefficient d'entropie
                new_ent_coef = round(0.02 * update_num, 2)
                
                # 3. Appliquer les modifications
                updated_config = {
                    **config,
                    'learning_rate': str(new_lr),  # Conversion en chaîne
                    'ent_coef': new_ent_coef,
                    'last_modified': time.time()  # Timestamp pour détection
                }
                
                # 4. Écrire les modifications
                with open(config_path, 'w', encoding='utf-8') as file:
                    yaml.dump(updated_config, file, default_flow_style=False)
                
                logger.info(
                    "🔄 Mise à jour %d/%d - LR: %.2e → %.2e, Entropy: %.2f → %.2f",
                    update_num, num_updates, old_lr, new_lr, old_ent_coef, new_ent_coef
                )
                
                # Attente avant la prochaine mise à jour (sauf après la dernière)
                if update_num < num_updates:
                    time.sleep(update_interval)
                    
            except Exception as e:
                logger.error(
                    "❌ Erreur lors de la mise à jour %d: %s",
                    update_num, str(e),
                    exc_info=True
                )
                # Pause plus longue en cas d'erreur
                time.sleep(5)
                
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(restored_config, file, default_flow_style=False)
        
        logger.info("🔄 Configuration restaurée aux valeurs d'origine")
        
    except Exception as e:
        logger.error("Erreur lors de la modification de la configuration: %s", e, exc_info=True)

def setup_logging() -> None:
    """Configure la journalisation pour l'application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_handlers = [
        logging.StreamHandler(),
        logging.FileHandler('test_orchestrator.log', encoding='utf-8')
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=log_handlers
    )
    
    # Activer les logs de débogage pour le module adan_trading_bot
    logging.getLogger('adan_trading_bot').setLevel(logging.DEBUG)


def log_test_overview() -> None:
    """Affiche un aperçu du test dans les logs."""
    logger.info("🚀 Démarrage du test de rechargement dynamique de configuration")
    logger.info("=" * 80)
    logger.info("Ce test va :")
    logger.info("1. Démarrer un entraînement avec des paramètres initiaux")
    logger.info("2. Modifier dynamiquement les paramètres pendant l'exécution")
    logger.info("3. Vérifier que les changements sont bien appliqués")
    logger.info("=" * 80)


def initialize_training_environment() -> Tuple[Dict, Dict, Path]:
    """Initialise l'environnement d'entraînement et retourne les configurations."""
    # Créer les répertoires nécessaires
    os.makedirs("config", exist_ok=True)
    
    # Créer les configurations
    test_config = create_minimal_test_config()
    agent_config = create_minimal_agent_config()
    
    # Sauvegarder la configuration d'entraînement
    train_config_path = Path("config/train_config.yaml")
    with open(train_config_path, 'w', encoding='utf-8') as file:
        yaml.dump(agent_config, file, default_flow_style=False)
    
    logger.info("✅ Configuration initiale sauvegardée dans %s", train_config_path)
    logger.info(
        "🔧 Paramètres initiaux : LR=%.1e, EntCoef=%.2f",
        float(agent_config.get('learning_rate', 0)),
        float(agent_config.get('ent_coef', 0))
    )
    
    return test_config, agent_config, train_config_path


def cleanup_resources(
    train_config_path: Optional[Path] = None,
    orchestrator: Optional[Any] = None,
    config_thread: Optional[threading.Thread] = None
) -> None:
    """Nettoie les ressources utilisées pendant le test."""
    try:
        # Fermer l'orchestrateur s'il existe
        if orchestrator is not None:
            logger.info("Nettoyage de l'orchestrateur...")
            orchestrator.close()
        
        # Attendre la fin du thread de modification
        if config_thread is not None and config_thread.is_alive():
            logger.info("Attente de la fin du thread de modification...")
            config_thread.join(timeout=5.0)
        
        # Supprimer le fichier de configuration temporaire
        if train_config_path is not None and train_config_path.exists():
            train_config_path.unlink()
            
    except Exception as e:
        logger.error("Erreur lors du nettoyage des ressources: %s", e, exc_info=True)


def main() -> int:
    """
    Point d'entrée principal pour le test de rechargement dynamique.
    
    Cette fonction orchestre l'ensemble du processus de test en:
    1. Configurant la journalisation
    2. Initialisant l'environnement de test
    3. Lançant l'orchestrateur d'entraînement
    4. Démarrant un thread pour modifier dynamiquement la configuration
    5. Exécutant l'entraînement avec suivi des paramètres
    6. Nettoyant les ressources utilisées
    
    Returns:
        int: Code de sortie (0 pour succès, 1 pour échec)
    """
    # Configuration initiale
    logger.info("🚀 Démarrage du test de rechargement dynamique")
    setup_logging()
    log_test_overview()
    
    # Variables pour le nettoyage
    train_config_path = None
    orchestrator = None
    config_thread = None
    
    try:
        # Initialisation de l'environnement
        logger.info("🔄 Initialisation de l'environnement de test...")
        test_config, agent_config, train_config_path = initialize_training_environment()
        
        # Initialiser l'orchestrateur
        logger.info("🤖 Initialisation du TrainingOrchestrator...")
        orchestrator = TrainingOrchestrator(
            config=test_config,
            agent_class=PPO,
            agent_config=agent_config
        )
        
        # Créer le callback de journalisation
        logging_callback = LoggingCallback(
            config_watcher=orchestrator.config_watcher,
            verbose=1
        )
        
        # Démarrer le thread de modification de configuration
        logger.info("⚙️  Lancement du thread de modification de configuration...")
        config_thread = threading.Thread(
            target=modify_training_config_after_delay,
            kwargs={
                'config_path': train_config_path,
                'initial_delay': 5,
                'update_interval': 10,
                'num_updates': 3
            },
            daemon=True,
            name="ConfigModifierThread"
        )
        config_thread.start()
        
        # Démarrer l'entraînement
        logger.info("🏃 Démarrage de l'entraînement (1000 pas)...")
        orchestrator.train_agent(
            total_timesteps=1000,
            callback=logging_callback
        )
        
        logger.info("✅ Entraînement terminé avec succès !")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n🛑 Arrêt manuel de l'utilisateur détecté")
        return 1
        
    except Exception as e:
        logger.error("❌ Échec critique du test: %s", str(e), exc_info=True)
        return 1
        
    finally:
        logger.info("🧹 Nettoyage des ressources...")
        try:
            cleanup_resources(
                train_config_path=train_config_path,
                orchestrator=orchestrator,
                config_thread=config_thread
            )
            logger.info("✅ Nettoyage terminé avec succès")
        except Exception as e:
            logger.error("❌ Erreur lors du nettoyage: %s", str(e), exc_info=True)

if __name__ == "__main__":
    main()