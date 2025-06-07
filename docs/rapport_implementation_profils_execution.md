# Rapport d'implémentation des profils d'exécution

## Résumé

Ce rapport détaille l'implémentation des profils d'exécution (CPU et GPU) dans le projet ADAN. Cette fonctionnalité permet de configurer facilement le projet pour s'exécuter sur différentes architectures matérielles sans modifier le code source.

## Contexte et objectifs

L'objectif principal était d'implémenter un mécanisme permettant de charger différentes configurations en fonction du matériel disponible (CPU ou GPU). Cela permet :

1. D'optimiser les performances en adaptant les paramètres d'entraînement au matériel
2. De faciliter le passage d'une configuration à l'autre sans modifier le code
3. D'assurer la portabilité du projet entre différents environnements d'exécution

## Modifications apportées

### 1. Scripts modifiés

Les scripts suivants ont été modifiés pour prendre en charge les profils d'exécution :

| Script | Modifications |
|--------|---------------|
| `train_rl_agent.py` | Ajout des arguments `--exec_profile` et `--device`, modification du chargement des configurations |
| `test_environment_with_merged_data.py` | Ajout de l'argument `--exec_profile`, modification de la fonction `load_configurations` |
| `fetch_data_ccxt.py` | Ajout de l'argument `--exec_profile`, modification du chargement des configurations |
| `process_data.py` | Ajout de l'argument `--exec_profile`, modification du chargement des configurations |
| `merge_processed_data.py` | Ajout de l'argument `--exec_profile`, modification de la fonction `load_configs` |

### 2. Nouveaux scripts

Un nouveau script de test a été créé pour vérifier la configuration des profils d'exécution :

- `test_exec_profiles.py` : Vérifie l'existence et le chargement des fichiers de configuration pour un profil donné

### 3. Structure des fichiers de configuration

La structure des fichiers de configuration a été modifiée pour prendre en charge les profils d'exécution :

```
config/
├── main_config.yaml            # Configuration générale (commune)
├── data_config_cpu.yaml        # Configuration des données pour CPU
├── data_config_gpu.yaml        # Configuration des données pour GPU
├── environment_config.yaml     # Configuration de l'environnement (commune)
├── agent_config_cpu.yaml       # Configuration de l'agent pour CPU
├── agent_config_gpu.yaml       # Configuration de l'agent pour GPU
└── logging_config.yaml         # Configuration du logging (commune)
```

### 4. Documentation

La documentation a été mise à jour pour refléter ces changements :

- Mise à jour du `README.md` pour inclure les informations sur les profils d'exécution
- Création d'un guide des commandes détaillé (`guide_commandes.md`)
- Création de ce rapport d'implémentation

## Implémentation technique

### Argument de ligne de commande

Tous les scripts principaux ont été modifiés pour accepter l'argument `--exec_profile` avec les valeurs possibles `cpu` (par défaut) ou `gpu` :

```python
parser.add_argument(
    '--exec_profile', 
    type=str, 
    default='cpu',
    choices=['cpu', 'gpu'],
    help="Profil d'exécution ('cpu' ou 'gpu') pour charger les configurations appropriées."
)
```

### Chargement des configurations

Les fonctions de chargement des configurations ont été modifiées pour utiliser le profil spécifié :

```python
# Exemple de modification
data_config_path = f'config/data_config_{profile}.yaml'
agent_config_path = f'config/agent_config_{profile}.yaml'
```

### Paramètre device

Pour le script d'entraînement, un argument supplémentaire `--device` a été ajouté pour spécifier l'appareil à utiliser (auto, cpu, cuda) :

```python
parser.add_argument(
    '--device',
    type=str,
    default='auto',
    choices=['auto', 'cpu', 'cuda'],
    help="Appareil à utiliser pour l'entraînement ('auto', 'cpu', 'cuda')."
)
```

## Tests et validation

Les tests suivants ont été effectués pour valider l'implémentation :

1. **Test des configurations** : Vérification que les fichiers de configuration sont correctement chargés pour chaque profil
2. **Test des scripts** : Vérification que tous les scripts acceptent l'argument `--exec_profile` et l'utilisent correctement

Les résultats des tests montrent que l'implémentation fonctionne comme prévu. Le script `test_exec_profiles.py` peut être utilisé pour vérifier la configuration à tout moment.

## Recommandations pour l'utilisation

1. **Profil CPU** : Utiliser ce profil pour le développement, les tests et l'exécution sur des machines sans GPU
   ```bash
   python scripts/train_rl_agent.py --exec_profile cpu --device cpu
   ```

2. **Profil GPU** : Utiliser ce profil pour l'entraînement sur des machines équipées de GPU compatibles CUDA
   ```bash
   python scripts/train_rl_agent.py --exec_profile gpu --device cuda
   ```

3. **Personnalisation** : Pour personnaliser davantage les configurations, modifier les fichiers correspondants dans le répertoire `config/`

## Conclusion

L'implémentation des profils d'exécution dans le projet ADAN offre une flexibilité accrue pour l'exécution sur différentes architectures matérielles. Cette fonctionnalité facilite le passage entre les environnements de développement et de production, tout en permettant d'optimiser les performances en fonction du matériel disponible.

Les modifications apportées sont non-intrusives et maintiennent la compatibilité avec le code existant. La documentation a été mise à jour pour refléter ces changements et faciliter l'utilisation des profils d'exécution.

---

*Rapport généré le 27 mai 2025*
