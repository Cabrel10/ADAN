# Module `common`

Ce module contient les utilitaires partagés, les constantes et la configuration du logger utilisés dans tout le projet ADAN.

## Contenu

* `constants.py`: Définition des constantes globales utilisées dans le projet, comme les codes d'action, les noms de colonnes standards, les valeurs par défaut, etc.
* `utils.py`: Fonctions utilitaires génériques pour la manipulation de dates, la sauvegarde/chargement de fichiers, la gestion des chemins, etc.
* `custom_logger.py`: Configuration du système de logging personnalisé pour ADAN, permettant un suivi cohérent des événements à travers les différents modules.

## Utilisation

Ce module est importé par presque tous les autres modules du projet. Il fournit une base commune pour assurer la cohérence et éviter la duplication de code. Par exemple:

```python
from adan_trading_bot.common.constants import ACTION_BUY, ACTION_SELL, ACTION_HOLD
from adan_trading_bot.common.utils import save_to_json, load_from_json
from adan_trading_bot.common.custom_logger import setup_logger

# Utilisation des constantes
if action == ACTION_BUY:
    # Logique d'achat...

# Utilisation des utilitaires
save_to_json(results, "path/to/results.json")

# Utilisation du logger
logger = setup_logger("environment", log_level="INFO")
logger.info("Environnement initialisé avec succès")
```

## Bonnes Pratiques

1. Toutes les constantes globales doivent être définies dans `constants.py` et non dispersées dans le code
2. Les fonctions utilitaires doivent être génériques et bien testées
3. Le logger doit être configuré de manière cohérente dans tous les modules
