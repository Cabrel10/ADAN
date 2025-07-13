# 04_DBE_Analysis.ipynb

## Analyse du Moteur Comportemental Dynamique (DBE)

Ce notebook Jupyter est dédié à l'analyse approfondie du comportement et de l'impact du Dynamic Behavior Engine (DBE) sur l'agent de trading ADAN. Il permet de visualiser et de comprendre comment le DBE module les paramètres de trading et d'apprentissage en fonction des conditions de marché et des performances de l'agent.

### Objectifs

*   **Visualiser l'évolution des paramètres du DBE** : Comprendre comment le `sl_pct` (stop-loss percentage) et le `tp_pct` (take-profit percentage) sont ajustés dynamiquement au fil du temps.
*   **Corréler le drawdown avec le mode de risque** : Analyser la relation entre le drawdown du portefeuille et l'activation du mode de risque `DEFENSIVE` par le DBE.
*   **Évaluer l'efficacité de l'agent** : Examiner la distribution des `reward_boost` pour déterminer la fréquence à laquelle l'agent est jugé efficace par le DBE.
*   **Suivre les pénalités d'inaction** : Visualiser l'application des pénalités d'inaction pour comprendre quand et pourquoi l'agent est pénalisé pour son manque d'activité.

### Utilisation

1.  **Exécuter un test d'endurance** : Avant d'utiliser ce notebook, assurez-vous d'avoir exécuté le script `scripts/endurance_test.py` sur une longue période. Ce script générera un fichier de log `dbe_replay.jsonl` (ou un nom similaire configuré dans le DBE) qui contient les données nécessaires à l'analyse.

    ```bash
    python scripts/endurance_test.py --duration_hours 24
    ```

2.  **Ouvrir le notebook** : Lancez Jupyter Lab ou Jupyter Notebook depuis le répertoire racine du projet et ouvrez `notebooks/04_DBE_Analysis.ipynb`.

3.  **Charger les données** : Le notebook commencera par charger le fichier `dbe_replay.jsonl` généré par le test d'endurance. Assurez-vous que le chemin d'accès au fichier est correct dans le notebook.

4.  **Exécuter les cellules** : Exécutez les cellules du notebook séquentiellement pour générer les visualisations et les analyses.

### Visualisations Clés

Le notebook inclura les graphiques suivants :

*   **Évolution de SL/TP** : Un graphique linéaire montrant `sl_pct` et `tp_pct` sur l'axe Y et le temps (ou le numéro de pas) sur l'axe X.
*   **Drawdown vs. Mode Défensif** : Un graphique combinant l'évolution du drawdown du portefeuille et des marqueurs visuels indiquant les périodes où le `risk_mode` du DBE est passé en `DEFENSIVE`.
*   **Distribution des Reward Boosts** : Un histogramme ou un graphique de densité des valeurs de `reward_boost` appliquées par le DBE.
*   **Pénalités d'Inaction** : Un graphique montrant les points dans le temps où des `penalty_inaction` ont été appliquées, avec leur magnitude.

### Dépendances

Ce notebook nécessite les bibliothèques Python suivantes :

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `jsonlines` (ou `json` pour lire ligne par ligne si le format est `jsonl`)

Assurez-vous qu'elles sont installées dans votre environnement Python.
