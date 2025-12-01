# Plan d'Action (Task Breakdown) : Correction et Clarification des Rapports de Backtesting

Ce document détaille les tâches nécessaires pour implémenter les exigences et la conception définies précédemment.

## 1. Amélioration du `PortfolioManager`

**Fichier concerné :** `src/adan_trading_bot/portfolio/portfolio_manager.py`

*   **Tâche 1.1 : Ajout des Attributs de Suivi des Trades et PnL**
    *   Ajouter les attributs suivants à la classe `PortfolioManager` :
        *   `self.total_opened_positions: int = 0`
        *   `self.total_closed_positions: int = 0`
        *   `self.realized_pnl_total: float = 0.0`
        *   `self.unrealized_pnl_total: float = 0.0`
    *   Initialiser ces attributs dans `__init__` et les réinitialiser dans `reset()`.

*   **Tâche 1.2 : Incrémentation des Trades Ouverts**
    *   Modifier la méthode `open_position(...)` pour incrémenter `self.total_opened_positions` après une ouverture réussie.

*   **Tâche 1.3 : Incrémentation des Trades Clôturés et Accumulation du PnL Réalisé**
    *   Modifier la méthode `close_position(...)` pour :
        *   Incrémenter `self.total_closed_positions` après une clôture réussie.
        *   Ajouter le PnL du trade clôturé (`receipt.get("pnl")`) à `self.realized_pnl_total`.

*   **Tâche 1.4 : Calcul et Mise à Jour du PnL Latent**
    *   Modifier la méthode `update_market_price(...)` pour :
        *   Calculer le PnL latent de chaque position ouverte.
        *   Sommer ces PnL latents pour mettre à jour `self.unrealized_pnl_total`.

*   **Tâche 1.5 : Exposition des Nouvelles Métriques**
    *   Modifier la méthode `get_metrics()` pour inclure les nouvelles métriques (`total_opened_positions`, `total_closed_positions`, `realized_pnl_total`, `unrealized_pnl_total`) dans le dictionnaire retourné.

*   **Tâche 1.6 : Implémentation de `close_all_positions`**
    *   Créer la nouvelle méthode `close_all_positions(reason: str, current_prices: Dict[str, float], timestamp: Any) -> float` :
        *   Itérer sur toutes les positions ouvertes (`self.positions`).
        *   Pour chaque position ouverte, appeler `self.close_position(asset, current_price, timestamp, reason=reason)`.
        *   Retourner la somme du PnL réalisé par ces clôtures.

## 2. Amélioration de l'Environnement

**Fichier concerné :** `src/adan_trading_bot/environment/multi_asset_chunked_env.py`

*   **Tâche 2.1 : Ajout du Paramètre `close_positions_on_episode_end`**
    *   Ajouter le paramètre `close_positions_on_episode_end: bool = False` à la méthode `__init__()`.
    *   Initialiser `self.steps_since_last_trade_with_open_pos: int = 0` dans `__init__()`.

*   **Tâche 2.2 : Logique de Pénalité de Passivité dans `step()`**
    *   Dans la méthode `step()`, après l'appel à `_execute_trades()` :
        *   Vérifier si `trade_executed_this_step` est `False` ET si `len(self.portfolio_manager._get_open_positions()) > 0`.
        *   Si vrai, incrémenter `self.steps_since_last_trade_with_open_pos`.
        *   Si `trade_executed_this_step` est `True`, réinitialiser `self.steps_since_last_trade_with_open_pos` à 0.

*   **Tâche 2.3 : Logique de Clôture en Fin d'Épisode dans `step()`**
    *   Dans la méthode `step()`, juste avant de retourner le tuple `(observation, reward, terminated, truncated, info)` :
        *   Si `terminated` ou `truncated` est `True` ET `self.close_positions_on_episode_end` est `True` :
            *   Appeler `self.portfolio_manager.close_all_positions("EPISODE_END", current_prices, current_timestamp)`.
            *   S'assurer que le PnL résultant est correctement pris en compte dans les métriques finales de l'épisode.

*   **Tâche 2.4 : Intégration de la Pénalité de Passivité dans la Récompense**
    *   Modifier la méthode `_calculate_reward()` pour :
        *   Ajouter une nouvelle composante de récompense `passivity_penalty`.
        *   Calculer `passivity_penalty = -self.config.get("reward_shaping", {}).get("passivity_penalty_weight", 0.01) * self.steps_since_last_trade_with_open_pos`.
        *   Ajouter `passivity_penalty` à la `total_reward`.
        *   Mettre à jour le dictionnaire `reward_components` pour inclure `passivity_penalty`.

*   **Tâche 2.5 : Configuration du Poids de la Pénalité de Passivité**
    *   Ajouter un paramètre `passivity_penalty_weight` (ex: `0.01`) dans le fichier `config.yaml` sous la section `reward_shaping`.

## 3. Amélioration du Script de Rapport de Backtesting

**Fichier concerné :** À identifier (ex: `scripts/backtest_final_rigorous.py` ou `scripts/evaluate_model_comprehensive.py`)

*   **Tâche 3.1 : Identification du Script de Rapport**
    *   Confirmer quel script est utilisé pour générer le rapport de backtesting final affiché à l'utilisateur.

*   **Tâche 3.2 : Récupération des Nouvelles Métriques**
    *   Modifier le script identifié pour récupérer les nouvelles métriques détaillées (`total_opened_positions`, `total_closed_positions`, `realized_pnl_total`, `unrealized_pnl_total`) via l'environnement et le `PortfolioManager`.

*   **Tâche 3.3 : Mise à Jour du Format du Rapport**
    *   Modifier le format d'affichage du rapport pour inclure clairement :
        *   "Trades : X ouverts, Y complétés"
        *   "PnL Réalisé : $Z"
        *   "PnL Latent : $W"
        *   "PnL Total : $(Z + W)"

## 4. Tests et Validation

*   **Tâche 4.1 : Tests Unitaires du `PortfolioManager`**
    *   Écrire des tests unitaires pour les nouveaux attributs et la méthode `close_all_positions` du `PortfolioManager`.
*   **Tâche 4.2 : Tests Unitaires de l'Environnement**
    *   Écrire des tests unitaires pour la logique de pénalité de passivité et la clôture en fin d'épisode dans `MultiAssetChunkedEnv`.
*   **Tâche 4.3 : Exécution du Backtest**
    *   Exécuter un backtest avec l'environnement et le script de rapport modifiés sur le jeu de données problématique.
*   **Tâche 4.4 : Vérification du Rapport**
    *   Vérifier que le rapport affiche correctement les trades ouverts/complétés et la ventilation du PnL.
*   **Tâche 4.5 : (Optionnel) Validation du Comportement de l'Agent**
    *   Lancer une courte session d'entraînement avec la pénalité de passivité activée et observer si l'agent modifie son comportement (ex: clôture les positions plus activement).
