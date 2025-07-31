#!/bin/bash
# Script pour démarrer Jupyter Notebook avec la configuration appropriée

# Se placer dans le répertoire du projet
cd "$(dirname "$0")"

# Vérifier que Jupytext est installé
pip show jupytext >/dev/null 2>&1 || {
    echo "Installation de Jupytext..."
    pip install jupytext
}

# Créer le répertoire des notebooks s'il n'existe pas
mkdir -p notebooks

# Démarrer Jupyter Notebook avec la configuration
jupyter notebook \
    --config=notebooks/jupyter_notebook_config.py \
    --no-browser \
    --port=8888 \
    --notebook-dir=notebooks/

# Pour ouvrir automatiquement dans le navigateur, décommentez la ligne suivante :
# xdg-open http://localhost:8888
