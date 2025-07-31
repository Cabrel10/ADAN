# 🚀 Guide de démarrage rapide ADAN Trading Bot

Ce guide vous aidera à configurer et exécuter ADAN Trading Bot en quelques étapes simples.

## 📋 Prérequis

- **Python 3.8+** - [Télécharger Python](https://www.python.org/downloads/)
- **pip** - Gestionnaire de paquets Python (inclus avec Python 3.4+)
- **Git** - [Télécharger Git](https://git-scm.com/downloads)
- **Bibliothèques scientifiques** (recommandé) :
  - NumPy
  - pandas
  - scikit-learn

## 🛠 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/adan-trading-bot.git
cd adan-trading-bot
```

### 2. Configuration de l'environnement

#### Créer un environnement virtuel (recommandé)

```bash
# Linux/MacOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

#### Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configuration initiale

1. Copiez le fichier de configuration exemple :
   ```bash
   cp config/config.example.yaml config/config.yaml
   ```

2. Modifiez `config/config.yaml` selon vos besoins :
   - Configurer les paramètres du réseau de neurones
   - Définir les paires de trading
   - Configurer les paramètres de risque

## 🚦 Exécution

### Mode développement

Pour exécuter en mode développement avec des données de test :

```bash
python -m src.adan_trading_bot.main --mode=dev
```

### Mode backtest

Pour exécuter un backtest sur des données historiques :

```bash
python -m src.adan_trading_bot.main --mode=backtest --start-date=20230101 --end-date=20231231
```

### Mode live (attention !)

⚠️ **Utiliser avec précaution en environnement réel**

```bash
python -m src.adan_trading_bot.main --mode=live --paper-trading
```

## 🧪 Tests

### Exécuter tous les tests

```bash
pytest
```

### Exécuter une catégorie spécifique de tests

```bash
# Tests unitaires
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Un test spécifique
pytest tests/unit/test_shared_buffer.py -v
```

### Couverture de code

```bash
pytest --cov=src tests/
```

## 🏗 Structure du projet

```
.
├── config/                 # Fichiers de configuration
│   ├── config.yaml         # Configuration principale
│   └── config.example.yaml # Exemple de configuration
├── data/                   # Données brutes et traitées
│   ├── raw/               # Données brutes (non versionnées)
│   └── processed/         # Données prétraitées (non versionnées)
├── docs/                  # Documentation technique
├── notebooks/             # Notebooks d'analyse
├── scripts/               # Scripts utilitaires
│   ├── prepare_adan_data.py
│   └── test_orchestrator_integration.py
├── src/                   # Code source
│   └── adan_trading_bot/
│       ├── environment/   # Environnements de trading
│       ├── models/        # Modèles d'IA
│       ├── training/      # Logique d'entraînement
│       ├── utils/         # Utilitaires communs
│       ├── __init__.py
│       └── main.py        # Point d'entrée principal
└── tests/                 # Tests automatisés
    ├── unit/             # Tests unitaires
    └── integration/      # Tests d'intégration
```

## 🛠 Développement

### Standards de code

Le projet utilise :
- **Black** pour le formatage du code
- **isort** pour le tri des imports
- **Flake8** pour le linting

Pour formater le code avant de commiter :

```bash
black .
isort .
flake8
```

### Workflow Git

1. Créez une branche pour votre fonctionnalité :
   ```bash
   git checkout -b feature/nom-de-la-fonctionnalite
   ```

2. Committez vos changements :
   ```bash
   git add .
   git commit -m "Description claire des modifications"
   ```

3. Poussez vos changements :
   ```bash
   git push origin feature/nom-de-la-fonctionnalite
   ```

4. Créez une Pull Request

## 🚨 Dépannage

### Problèmes courants

#### Erreurs d'importation
- Vérifiez que votre environnement virtuel est activé
- Vérifiez que vous êtes dans le bon répertoire
- Exécutez `pip install -e .` pour une installation en mode développement

#### Problèmes de configuration
- Vérifiez l'indentation YAML
- Vérifiez les chemins des fichiers
- Consultez les logs dans `logs/` pour plus de détails

#### Problèmes de performance
- Vérifiez l'utilisation de la mémoire
- Réduisez la taille des lots d'entraînement si nécessaire
- Vérifiez les paramètres de parallélisme

## 📚 Documentation complète

Pour une documentation plus détaillée, consultez :
- [Guide de configuration avancée](docs/configuration_guide.md)
- [Guide de développement](docs/development_guide.md)
- [Architecture du système](docs/architecture.md)

## ⚠️ Avertissement important

**Ce logiciel est fourni à des fins éducatives et de recherche uniquement.**

Le trading de cryptomonnaies comporte des risques importants de perte en capital. Ne tradez pas avec de l'argent que vous ne pouvez pas vous permettre de perdre. Les performances passées ne sont pas indicatives des résultats futurs. Les développeurs ne peuvent être tenus responsables des pertes éventuelles encourues lors de l'utilisation de ce logiciel.

Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt du projet.
