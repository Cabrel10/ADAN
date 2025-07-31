# ADAN Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📝 Aperçu

ADAN (Adaptive Deep Algorithmic Network) est un système de trading algorithmique avancé conçu pour les marchés de cryptomonnaies. Basé sur des techniques d'apprentissage par renforcement profond, ADAN permet de développer et déployer des stratégies de trading adaptatives et évolutives.

## 🚀 Fonctionnalités

### 🎯 Principales
- **Apprentissage par renforcement** : Implémentation d'algorithmes de DRL (Deep Reinforcement Learning)
- **Traitement parallèle** : Exécution multi-processus pour une meilleure performance
- **Gestion du risque** : Mécanismes intégrés de gestion des risques
- **Support multi-actifs** : Trading sur plusieurs paires de cryptomonnaies
- **Multi-timeframes** : Analyse sur différentes échelles de temps

### ⚙️ Composants clés
- **SharedExperienceBuffer** : Mémoire de rejeu d'expériences priorisées
- **TrainingOrchestrator** : Orchestrateur de l'entraînement distribué
- **Environnements de trading** : Simulation de marché pour le backtesting
- **API d'échange** : Connecteurs pour différentes plateformes de trading

## 🛠 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Git

### Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-utilisateur/adan-trading-bot.git
cd adan-trading-bot
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# OU
.\venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

La configuration s'effectue via le fichier `config/config.yaml`. Consultez le [Guide de configuration](docs/configuration_guide.md) pour les options détaillées.

## 🏗 Structure du projet

```
.
├── config/               # Fichiers de configuration
├── data/                 # Données brutes et traitées
│   ├── raw/             # Données brutes
│   └── processed/       # Données prétraitées
├── docs/                # Documentation technique
├── notebooks/           # Notebooks d'analyse et d'expérimentation
├── scripts/             # Scripts utilitaires
├── src/                 # Code source principal
│   └── adan_trading_bot/
│       ├── environment/  # Environnements de trading
│       ├── models/      # Modèles d'IA
│       ├── training/    # Logique d'entraînement
│       ├── utils/       # Utilitaires communs
│       └── __init__.py
└── tests/               # Tests automatisés
    ├── unit/           # Tests unitaires
    └── integration/    # Tests d'intégration
```

## 🧪 Exécution des tests

Pour exécuter tous les tests :
```bash
pytest
```

Pour exécuter une catégorie spécifique de tests :
```bash
pytest tests/unit/       # Tests unitaires
pytest tests/integration # Tests d'intégration
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📜 Licence

Distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## ⚠️ Avertissement

**Ce logiciel est fourni à des fins éducatives et de recherche uniquement.**

Le trading de cryptomonnaies comporte des risques importants de perte en capital. Ne tradez pas avec de l'argent que vous ne pouvez pas vous permettre de perdre. Les performances passées ne sont pas indicatives des résultats futurs. Les développeurs ne peuvent être tenus responsables des pertes éventuelles encourues lors de l'utilisation de ce logiciel.
