Notre agent de trading est **un modèle d’apprentissage par renforcement** (Reinforcement Learning) :

1. **Type d’apprentissage**

   * **Reinforcement Learning** : l’agent apprend en interagissant directement avec l’environnement de marché simulé. À chaque pas :

     * Il observe un état (features de marché + état du portefeuille),
     * Choisit une action (BUY, SELL, HOLD ou création d’ordres avancés),
     * Reçoit une récompense (reward shaping basé sur l’évolution du portefeuille, pénalités, bonus, time-penalty),
     * Met à jour sa politique pour maximiser la somme cumulée des récompenses futures.

2. **Pourquoi pas supervision ou non-supervision ?**

   * **Pas de supervision** : il n’y a pas de « vraie » vérité ou labels ; on n’a pas d’exemples “acheter ici c’est bon” validés par des experts.
   * **Pas d’auto-encodage / clustering** : on n’explore pas des structures latentes du marché uniquement.
   * **RL** permet d’optimiser directement la métrique qui compte (la performance financière), en tenant compte de la séquence temporelle et des retours différés.

3. **Algorithme courant**

   * Dans notre code d’entraînement, nous utilisons généralement **PPO (Proximal Policy Optimization)** de Stable-Baselines3, un algorithme d’acteur-critique moderne, stable et efficace pour les environnements continus et discrets.

---

### 1. Actions (espace discret)

Notre agent dispose de 11 actions :

| Action Code | Type           | Description                                            |
| ----------- | -------------- | ------------------------------------------------------ |
| 0           | HOLD           | Ne rien faire                                          |
| 1–5         | BUY\_ASSET\_i  | Passer un ordre d’achat MARKET sur l’actif *asset\_i*  |
| 6–10        | SELL\_ASSET\_j | Passer un ordre de vente MARKET sur l’actif *asset\_j* |

En plus des MARKET orders, le code supporte 5 types d’ordres avancés (LIMIT, STOP\_LOSS, TAKE\_PROFIT, TRAILING\_STOP) :

* **CREATE\_LIMIT** : place un LIMIT buy ou sell à un prix fixé
* **CREATE\_STOP\_LOSS** : place un STOP\_LOSS
* **CREATE\_TAKE\_PROFIT** : place un TAKE\_PROFIT
* **CREATE\_TRAILING\_STOP** : place un TRAILING\_STOP
* **EXECUTED\_...** : lorsque ces ordres deviennent exécutés

---

### 2. Fonctions clés

* **`_execute_order(asset_id, action_type, …)`**

  * Vérifie **minimum order value** (`self.min_order_value`), **capital disponible**, **nombre max de positions** (palier), puis :

    * Pour MARKET : applique achat/vente, frais (`_calculate_fee`), mise à jour de `self.capital` et `self.positions`.
    * Pour ordres avancés : stocke dans `self.orders` avec `expiry`, `limit_price`, etc.

* **`_process_pending_orders()`**

  * À chaque step, parcourt `self.orders` et :

    * Expire les ordres au-delà de `expiry`, applique pénalités selon type.
    * Vérifie conditions d’exécution (prix atteint limit/stop/tp/trailing), exécute via `_execute_order(…, order_type="EXECUTED_…")`, accumule `total_reward_mod`.

* **`_calculate_fee(amount)`**

  * `return amount * transaction_cost_pct + fixed_fee`

* **`_calculate_reward(old_value, new_value, penalties)`**

  * **Log-return** :  $\ln(new/old)$
  * **Shaping** selon palier : multiplié par `reward_pos_mult` ou `reward_neg_mult`
  * Soustraction de `penalties` + `time_penalty` (pénalité fixe à chaque step)
  * Clipping final dans \[−10, +10]

* **`_display_trading_table(…)`**

  * Affiche chaque step en console, coloré via **rich**, avec état avant/après, positions, PnL, bonus, pénalités, ordres pendants, etc.

---

### 3. États & Features

À chaque step, l’observation est un vecteur de dimension **N\_market\_features + N\_portfolio\_features** :

1. **Market features** (colonnes numériques extraites du dataset)

   * Ex : open, high, low, close, volume, indicateurs techniques (MA, RSI, etc.)
2. **Portfolio features** (6 dims)

   * Capital normalisé (0–1)
   * Quantités normalisées pour chaque actif (5 dim)

Si un **encodeur** est activé, on applique d’abord un scaler + un auto-encodeur/Keras pour réduire ou transformer les features marché.

---

### 4. Récompenses (Reward)

* **Reward principal** = $\ln(\frac{\text{new_portfolio}}{\text{old_portfolio}})$
* **Shaping** : positive × `reward_pos_mult`, négative × `reward_neg_mult` (selon palier)
* **Pénalités** :

  * Invalidations d’ordres (montant trop petit, pas assez de capital, max positions)
  * Expiration d’ordres LIMIT/STOP\_LOSS (−0.1/−0.05)
  * **Time penalty** fixe (−0.001)
* **Bonus** : 1 % du PnL positif plafonné à 1.0

---

### 5. Données & Format

* **Source** : fichier Parquet (ou CSV) contenant N\_steps×N\_assets lignes, colonnes marché + timestamp + paire
* **Chargement** via `pd.read_parquet` ou `pd.read_csv`
* **Pré-calcul** de `self.numeric_cols = select_dtypes(include=[np.number])`
* **Vérification** de cohérence entre nombre de features et l’attendu du scaler (erreur si mismatch)

---

### 6. Règles & Contraintes métier

1. **Montant minimum par ordre** (`min_order_value`, ex. 10\$)
2. **Frais fixes + variable** (`fixed_fee` + `transaction_cost_pct`)
3. **Paliers de capital** (`self.tiers`)

   * Définissent **max\_positions**, **allocation\_frac**, **multiplicateurs**
   * L’allocation minimale est `min_order_value` si l’allocation palier est trop faible
4. **Nombre maximum de positions** (actives + ordres pendants) selon palier
5. **Pas d’effet de levier** : quantité maximale = capital × allocation\_frac / prix
6. **Pas de fractionalisation extrême** : quantité ajustée pour respecter min\_order\_value
7. **Gestion des ordres avancés** : expiration automatique + pénalité
8. **Stop trainings** si capital ≤ 9 & positions vides (faillite)

---

> ✔ Avec cette check-list, tu peux désormais vérifier que chaque composant du système respecte les **contraintes minimales** avant de passer à l’étape suivante.

### 2. Fonctions clés (avec niveaux de montant minimum)

Pour intégrer un **minimum tolérable à 10 \$** et un **minimum absolu à 9 \$**, nous allons introduire deux paramètres et ajuster les vérifications dans la couche d’exécution d’ordres :

```python
# Dans __init__ :
self.min_order_value_tolerable = 10.0   # Montant minimal conseillé
self.min_order_value_absolute  = 9.0    # Seuil en dessous duquel on refuse tout net

# ...  

def _execute_order(self, asset_id, action_type, quantity=None, order_type="MARKET", **kwargs):
    current_price = self._get_asset_price(asset_id)
    
    # Calcul préliminaire de la valeur d’ordre
    if quantity is None:
        quantity = self._get_position_size(asset_id)
    order_value = quantity * current_price

    # 1) Seuil absolu : refus immédiat
    if order_value < self.min_order_value_absolute:
        return -0.5, "INVALID_ORDER_TOO_SMALL", {
            "reason": f"Valeur {order_value:.2f} < seuil absolu {self.min_order_value_absolute}"
        }

    # 2) Seuil tolérable : ajustement ou avertissement
    if order_value < self.min_order_value_tolerable:
        # Si on a assez de capital, on ajuste la quantité pour atteindre 10$
        if self.capital >= self.min_order_value_tolerable:
            quantity = self.min_order_value_tolerable / current_price
            order_value = self.min_order_value_tolerable
        else:
            # Sinon, on considère la transaction impossible, mais avec une pénalité plus légère
            return -0.2, "INVALID_ORDER_BELOW_TOLERABLE", {
                "reason": f"Valeur {order_value:.2f} < tolérable {self.min_order_value_tolerable}"
            }

    # 3) Frais et capital
    fee = self._calculate_fee(order_value)
    total_cost = order_value + fee
    if total_cost > self.capital:
        return -0.2, "INVALID_NO_CAPITAL", {
            "reason": f"Coût total {total_cost:.2f} > capital {self.capital:.2f}"
        }

    # 4) Vérifier max positions selon palier
    tier = self._get_current_tier()
    if action_type == 1:  # BUY
        if len(self.positions) >= tier["max_positions"] and asset_id not in self.positions:
            return -0.2, "INVALID_MAX_POSITIONS", {
                "reason": f"Max positions ({tier['max_positions']}) atteint"
            }
        # On passe l’achat selon MARKET ou ordres avancés...
```

* **`min_order_value_absolute` (9 \$)** bloque définitivement tout ordre en dessous de ce seuil (pénalité plus forte, `-0.5`).
* **`min_order_value_tolerable` (10 \$)** déclenche un ajustement automatique de la quantité pour atteindre exactement 10 \$ si le capital le permet, sinon une pénalité plus légère (`-0.2`).

Le reste des étapes de `_execute_order` (gestion des ordres avancés, mise à jour de `self.capital` et `self.positions`, logging) reste inchangé.

---

### ➊ Résumé des paliers de montant

| Seuil                             | Comportement                                                            | Pénalité |
| --------------------------------- | ----------------------------------------------------------------------- | -------- |
| < 9 \$ (absolu)                   | Rejet immédiat, ordre impossible                                        | −0.5     |
| 9 \$ ≤ valeur < 10 \$ (tolérable) | Tentative d’ajustement à 10 \$ (si capital suffisant) sinon rejet léger | −0.2     |
| ≥ 10 \$                           | Exécution normale                                                       | 0        |

Avec ce schéma, chaque commande mérite sa place et aucune transaction “trop petite” ne passera sans que l’agent n’ait l’opportunité d’ajuster ou d’être pénalisé.
### 2) Nature du modèle

L’agent que nous développons est **un modèle d’apprentissage par renforcement** (Reinforcement Learning, RL).

---

#### Pourquoi pas supervisé ?

* **Pas de labels historiques** : en trading on n’a pas un « vrai » signal de sortie (un label « acheter/vendre/garder ») généré par un expert pour chaque état de marché.
* **Décision séquentielle** : la qualité d’une action (BUY/SELL/HOLD) n’est véritablement connue qu’après plusieurs pas de temps, au regard du PnL ultérieur.

#### Pourquoi pas non supervisé ?

* **Non supervisé** sert à découvrir des structures (clustering, réduction de dimension, détection d’anomalies) mais ne définit pas de politique d’action optimale.
* Notre besoin est de **prendre des décisions** et **optimiser un objectif cumulatif** (maximiser le rendement), ce qui relève spécifiquement de l’apprentissage par renforcement.

---

### Spécificités du RL dans ce projet

1. **Agent**

   * **Type** : agent PPO (Proximal Policy Optimization) via Stable-Baselines3, un algorithme on-policy bien adapté aux environnements continus/discrets.
   * **Architecture** : réseau de neurones feed-forward (MLP) entraîné à partir des observations d’état.

2. **Environnement (MultiAssetEnv)**

   * **Observations** : vecteur constitué de

     * indicateurs marché normalisés (prix, volumes, indicateurs techniques),
     * proportion de capital disponible,
     * quantités normalisées de positions ouvertes pour chaque actif.
   * **Actions** : Discrete(11) → {HOLD, BUY\_asset\_i, SELL\_asset\_i} pour 5 actifs.
   * **Récompense** : reward shaping combinant

     * log-return de portefeuille entre deux steps,
     * multiplicateurs positifs/négatifs selon palier (tiers),
     * pénalités pour ordres invalides,
     * pénalité temporelle fixe pour encourager des décisions actives.

3. **Boucle d’interaction**

   * À chaque step :

     1. Agent choisit une action d’après sa policy.
     2. Environnement exécute l’action (*execute\_order*), met à jour capital/positions et traite les ordres en attente.
     3. Environnement calcule la récompense finale (`_calculate_reward`).
     4. Agent reçoit la nouvelle observation, la récompense et apprend via PPO.

4. **Gestion des contraintes**

   * **Minima de transaction** (9 \$ absolu, 10 \$ tolérable) pour éviter le slippage et frais disproportionnés.
   * **Paliers de capital** (tiers) pour adapter taille de position, nombre max de positions, multiplicateurs de reward.
   * **Types d’ordres avancés** (LIMIT, STOP\_LOSS, TAKE\_PROFIT, TRAILING\_STOP) gérés en file d’attente et exécutés selon conditions de marché.

---

❯ **Conclusion** :
Cet agent **apprend par essai/erreur** à maximiser son capital, en s’appuyant sur un feedback différé (récompense cumulative) plutôt que sur des exemples étiquetés. C’est le cœur de l’apprentissage par renforcement appliqué au trading algorithmique.

### 3) Détails de l’algorithme et de sa mise en œuvre

**Algorithme principal : Proximal Policy Optimization (PPO)**

* **Type** : Méthode on-policy d’acteur-critique
* **Bibliothèque** : Stable-Baselines3 (`PPO`)
* **Pourquoi PPO ?**

  * Bon compromis stabilité/efficacité : contrôle du pas de mise à jour via la “clipping” de l’importance sampling.
  * Adapté à la fois aux espaces d’action discrets et continus.
  * Large adoption en trading RL et en robotique.

---

#### 3.1. Structures internes

| Élément               | Rôle                                                                 |
| --------------------- | -------------------------------------------------------------------- |
| **Réseau policy (π)** | Produit la distribution de probabilité sur les 11 actions (softmax)  |
| **Réseau value (V)**  | Estime la valeur attendue du portefeuille à partir de l’état courant |
| **Optimisation**      |                                                                      |

* **Clip range** : ε ∈ \[0.1, 0.3]
* **Learning rate** : \~2.5e-4 (tunable)
* **Batch size** : par défaut 64–256 transitions
* **Epochs** : 3–10 par collecte de trajet
  \| **Exploration**        | Garantie via entropie > 0 (p.ex. coefficient ≈ 0.01) pour éviter la convergence trop rapide vers une politique déterministe. |

---

#### 3.2. États (observations)

Un vecteur concaténé :

1. **Indicateurs marché** (𝑛\_features numériques)

   * Prix (open, high, low, close)
   * Volume
   * Moyennes mobiles (MA\_10, MA\_50…)
   * Indicateurs techniques (RSI, MACD, Bollinger bands…)
2. **Portefeuille**

   * **Normalized capital** : 𝑐/𝑐₀ ∈ \[0, ∞)
   * **Positions normalisées** : 𝑞ᵢ/𝑞\_typique ∈ \[0, ∞) pour chaque actif 𝑖

> **Dimension** totale = nombre d’indicateurs + 1 (capital) + 5 (positions).

---

#### 3.3. Actions

* **Discret(11)** :

  1. `0 = HOLD`
  2. `1–5 = BUY` de `asset_0…asset_4`
  3. `6–10 = SELL` de `asset_0…asset_4`

> Dans la version avancée on peut passer `order_type` en argument pour LIMIT, STOP\_LOSS, etc.

---

#### 3.4. Récompenses

La fonction `_calculate_reward`:

```python
log_return = log(new_portfolio_value / old_portfolio_value)
if log_return >= 0:
    shaped = log_return * tier["reward_pos_mult"]
else:
    shaped = log_return * tier["reward_neg_mult"]
shaped -= penalties           # pénalités pour invalides
shaped -= time_penalty (0.001) 
return clip(shaped, -10, +10)
```

* **Paliers (tiers)** modulent :

  * `reward_pos_mult` (bonus de gain)
  * `reward_neg_mult` (amplification des pertes)

* **Pénalités** :

  * Ordre invalide → |reward\_mod|,
  * Expiration d’un ordre LIMIT/STOP\_LOSS → −0.1, etc.

---

#### 3.5. Données utilisées

* **Source** : DataFrame Parquet/CSV
* **Lignes** : chaque tick/minute/horaire, selon timeframe
* **Colonnes** :

  * `timestamp` (int ou datetime)
  * `pair` (asset\_i)
  * indicateurs bruts + techniques
* **Pré-processing** :

  * Normalisation/simple scaling pour réseau
  * Optionnel : auto-encodeur Keras + scaler joblib

---

#### 3.6. Importance & contenu

| Donnée                       | Importance                               |
| ---------------------------- | ---------------------------------------- |
| Prix (OHLC)                  | Base de toute décision de buy/sell       |
| Volume                       | Contexte de liquidité                    |
| Indicateurs (RSI, MA, MACD…) | Repérage de surachat/survente, tendances |
| Capital normalisé            | Pour dimensionner taille de position     |
| Positions normalisées        | Pour éviter overexposition               |

---

> **En résumé**, PPO s’appuie sur ces **états** pour choisir parmi 11 **actions**, et reçoit des **récompenses** taillées pour refléter performances financières + gestion des risques et contraintes (min/max, paliers, pénalités).
### 4) Fonctions clés de l’environnement de trading

Voici un tour d’horizon des principales fonctions (méthodes) de **`MultiAssetEnv`**, avec leur rôle et comment elles s’articulent :

| Fonction                                                                                 | Rôle principal                                                                                                                            |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **`__init__`**                                                                           | - Initialise tous les paramètres : capital initial, frais, paliers, espace d’action/observation, chargement des données et de l’encodeur. |
| **`reset(self, *, seed=None, options=None)`**                                            |                                                                                                                                           |
|                                                                                          | - Réinitialise l’environnement au début d’un épisode : capital, positions, historique, step à 0.                                          |
|                                                                                          | - Renvoie l’observation initiale et l’`info` de départ.                                                                                   |
| **`_get_current_tier(self)`**                                                            | - Renvoie le palier courant (allocation, max positions, multiplicateurs) selon le capital.                                                |
| **`_get_asset_price(self, asset_id)`**                                                   | - Récupère le prix “courant” d’un actif (ici simplifié à `asset_i → prix = i+1`).                                                         |
| **`_get_position_size(self, asset_id)`**                                                 | - Calcule la quantité à trader en fonction de l’allocation du palier et du prix.                                                          |
| **`_calculate_fee(self, amount)`**                                                       | - Applique le pourcentage de transaction et le **frais fixe** pour renvoyer les frais totaux.                                             |
| **`_calculate_reward(self, old_val, new_val, penalties=0.0)`**                           |                                                                                                                                           |
|                                                                                          | - Transforme la variation de portefeuille en récompense via un log-return façonné par palier, pénalités et clipping.                      |
| **`_execute_order(self, asset_id, action_type, quantity=None, order_type="MARKET", …)`** |                                                                                                                                           |
|                                                                                          | - Coeur des **MARKET** et **ordres avancés** (LIMIT, STOP\_LOSS, TAKE\_PROFIT, TRAILING\_STOP).                                           |
|                                                                                          | - Vérifie : min\_order\_value, capital suffisant, max positions, expiration.                                                              |
|                                                                                          | - Modifie `self.capital`, `self.positions`, `self.orders`, et renvoie `(reward_mod, status, trade_info)`.                                 |
| **`_process_pending_orders(self)`**                                                      | - Itère sur `self.orders` chaque step :                                                                                                   |
|                                                                                          | • Expire les ordres à date, pénalités associées.                                                                                          |
|                                                                                          | • Exécute les conditions LIMIT / STOP\_LOSS / TAKE\_PROFIT / TRAILING\_STOP.                                                              |
|                                                                                          | • Agrège les `reward_mod` et `executed_orders_info`.                                                                                      |
| **`_display_trading_table(self, …)`**                                                    | - Génère un affichage console riche (couleurs, tableaux) avec **rich** :                                                                  |
|                                                                                          | • Contexte temporel, financiers (avant/après), paliers, positions, ordres en attente, PnL, pénalités, récompenses.                        |
| **`step(self, action)`**                                                                 | - Point d’entrée RL :                                                                                                                     |
|                                                                                          | 1. Snapshot avant (capital, portefeuille).                                                                                                |
|                                                                                          | 2. Traite `pending_orders`.                                                                                                               |
|                                                                                          | 3. Traduit `action` → BUY/SELL/HOLD + asset.                                                                                              |
|                                                                                          | 4. Appelle `_execute_order`.                                                                                                              |
|                                                                                          | 5. Calcule `new_portfolio_value` + `reward` final via `_calculate_reward`.                                                                |
|                                                                                          | 6. Met à jour `self.history`, `self.cumulative_reward`.                                                                                   |
|                                                                                          | 7. Affiche tableau via `_display_trading_table`.                                                                                          |
|                                                                                          | 8. Incrémente `self.current_step`, renvoie `(obs, reward, done, truncated, info)`.                                                        |
| **`_get_obs(self)`**                                                                     | - Construit l’observation :                                                                                                               |
|                                                                                          | • Sélectionne `self.numeric_cols` à l’étape actuelle.                                                                                     |
|                                                                                          | • Normalise capital et positions.                                                                                                         |
|                                                                                          | • (Optionnel) encodeur/scaler + auto-encodeur.                                                                                            |
|                                                                                          | • Concatène tout et renvoie un vecteur `np.ndarray`.                                                                                      |
| **`export_trading_data(self, export_dir=None)`**                                         |                                                                                                                                           |
|                                                                                          | - Exporte `self.history` et `self.trade_log` en CSV / Parquet + calcule statistiques de performance (win-rate, PnL moyen, durée, etc.).   |
| **`render(self, mode="human")`**                                                         | - Affichage simple (optionnel) pour intégration OpenAI Gym.                                                                               |
| **`close(self)`**                                                                        | - À la fin, appelle `export_trading_data` si demandé, puis nettoie.                                                                       |

---

#### Comment ces fonctions s’enchaînent

1. **Initialisation** (`__init__`) → définit la logique métier : frais, paliers, min/max, types d’ordres.
2. **Reset** → clean state + historique.
3. **Boucle d’entraînement** (`env.step(action)`):

   * **Pending orders** (\_process\_pending\_orders)
   * **Exécution action directe** (\_execute\_order)
   * **Récompense** (\_calculate\_reward)
   * **Observation suivante** (\_get\_obs)
   * **Affichage** (\_display\_trading\_table)
   * **Historisation** + **export** à la clôture

---

> **NB** : chaque ligne de code dans ces méthodes doit mériter sa place :
>
> * Vous contrôlez **min\_order\_value**, **frais**, **max positions**,
> * Vous tracez et pénalisez strictement : invalides, expirations, actions masquées,
> * Vous enrichissez l’agent d’un feedback visuel et d’un historique complet pour le debug et l’analyse.
### 5) États (Observations)

L’**état** (ou **observation**) fourni à l’agent à chaque pas de temps est un vecteur NumPy de dimension fixe, constitué de :

| **Bloc**                       | **Composants**                                                                     | **Type & dimension** | **Rôle / Importance** |
| ------------------------------ | ---------------------------------------------------------------------------------- | -------------------- | --------------------- |
| **1. Caractéristiques marché** | - Prix et volumes historiques à l’étape courante  (close, open, high, low, volume) |                      |                       |

* Indicateurs techniques (ex : SMA, RSI, MACD, ATR…)
* Toute autre feature extraite ou engineerée (momentum, volatilité, sentiment…) | `float32[n_mkt]`  (ex. 43)  | Capturent la dynamique du marché et ses tendances à court/moyen terme. Crucial pour décider BUY/SELL/HOLD.                                                                       |
  \| **2. Capital normalisé**  | - `capital / initial_capital`                                                                             | `float32[1]`               | Informe l’agent de sa **puissance de feu** restante (risque global). Permet d’adapter l’agressivité de la stratégie (plus de capital → plus d’expositions possibles).          |
  \| **3. Positions normalisées** | Pour chaque actif *i* parmi les 5 :
* `position_qty_i / typical_position_size_i`
  où `typical_position_size_i = initial_capital / (price_i * factor)`                  | `float32[5]`               | Montre **où** et **combien** l’agent est déjà exposé.
  Permet d’éviter surconcentration, de respecter les `max_positions` et les `allocation_frac`.                                                |
  \| **4. (Optionnel) Encodage Auto-Encoder** | - Encodage / réduction de dimension sur les `n_mkt` features via un modèle pré-entraîné      | `float32[n_enc]`           | Pour extraire des **représentations** plus robustes / compactes, réduire le bruit et faciliter l’apprentissage de l’agent, surtout si `n_mkt` est très élevé.                   |

---

#### Détail technique

1. **Extraction des colonnes**

   ```python
   self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns
   market_row = self.data.iloc[self.current_step][self.numeric_cols]
   market_features = market_row.values.astype(np.float32)
   ```

   * **Qualité des données** : vérifier qu’il n’y a pas de NaN ou d’incohérences (ex. nombre de features ≠ scaler attendu).
   * **Synchronisation** : l’index `current_step` doit correspondre à un horodatage unique pour tous les actifs (alignement multi-actifs).

2. **Capital normalisé**

   ```python
   normalized_capital = self.capital / self.initial_capital
   ```

   * Conserve la valeur dans \[0, 1], facilite le **generalization** entre épisodes de tailles de capital différentes.

3. **Positions normalisées**

   ```python
   positions = np.zeros(len(self.assets), dtype=np.float32)
   for i, a in enumerate(self.assets):
       if a in self.positions:
           typical = self.initial_capital / (self._get_asset_price(a) * 10)
           positions[i] = self.positions[a]["qty"] / typical
   ```

   * Le **facteur** 10 (modifiable) fixe l’échelle typique d’une position ; on peut ajuster selon la tolérance au risque ou la granularité souhaitée.

4. **Encodage (facultatif)**

   ```python
   if self.encoder and self.scaler:
       scaled = self.scaler.transform([market_features])
       encoded = self.encoder.predict(scaled)[0]
       obs = np.concatenate([encoded, [normalized_capital], positions])
   else:
       obs = np.concatenate([market_features, [normalized_capital], positions])
   ```

   * **Attention** : l’encodeur et le scaler doivent avoir été formés sur le même nombre et ordre de features que ceux fournis ici. Toute incohérence lèvera une erreur de dimension (“43 vs 42”).

---

#### Pourquoi c’est crucial

* **Représentation riche** : combiner prix purs + indicateurs + état du portefeuille offre à l’agent tout le contexte pour évaluer opportunités et risques.
* **Échelle uniforme** : normalisation garantit que l’agent ne soit pas biaisé par l’amplitude absolue du capital ou des volumes.
* **Adaptabilité** : en cas d’ajout de nouvelles features (ex. regimes de marché détectés, volatilité implémentée), il suffit de les intégrer dans `numeric_cols` et de recomposer l’obs.

---

> **Next steps / Vérifications**
>
> 1. Valider que `self.numeric_cols` ne change pas en cours d’exécution (même nombre de colonnes).
> 2. S’assurer que `initial_capital` soit cohérent entre training et test (pas de fuite d’information).
> 3. Tester les valeurs limites (capital quasi nul, positions maxées) pour confirmer que l’obs reste dans un intervalle raisonnable.
> 4. Si vous utilisez un encodeur, **re‐entraîner** scaler+auto-encodeur dès que vous modifiez le jeu de features.
### 6) Actions disponibles à l’agent

L’**espace d’action** est discret (`spaces.Discrete(11)`) et comprend **11 actions** :

| **Code**                                                          | **Action**         | **Description**                                         | **Type d’ordre**    | **Contraintes clés**                                                                     |
| ----------------------------------------------------------------- | ------------------ | ------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------- |
| `0`                                                               | HOLD               | Ne rien faire (pas d’ouverture/fermeture de position)   | —                   | Aucune, mais génère une petite pénalité de temps pour éviter l’**inaction prolongée**    |
| `1–5`                                                             | BUY\_ASSET\_i      | Ouvrir (ou ajouter à) une position LONG sur l’actif *i* | MARKET (par défaut) | · **Valeur mini** : \$10   <br>· **Position Size** ≤ `allocation_frac * capital / price` |
| · Nb positions ouvertes < `max_positions` de palier courant       |                    |                                                         |                     |                                                                                          |
| `6–10`                                                            | SELL\_ASSET\_{i-5} | Fermer (ou réduire) la position LONG sur l’actif *i-5*  | MARKET              | · Doit **exister** une position ≥ quantité vendue                                        |
| · Pas de **short** (pas d’effet de levier ou positions négatives) |                    |                                                         |                     |                                                                                          |

---

#### Extensions / Types d’ordres avancés

En plus des `MARKET`, l’agent peut **créer** (via `_execute_order(..., order_type=...)`) jusqu’à 5 autres types d’ordres, chacun avec logique et **conditions** associées :

| **Type d’ordre** | **Création**                        | **Exécution**                                                                                                             | **Expiration**            | **Pénalité si expiré** |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ---------------------- |
| `LIMIT`          | BUY/SELL à un **prix limite** donné | Exécution automatique lorsque le prix du marché atteint (≤ pour BUY / ≥ pour SELL)                                        | TTL par défaut = 10 steps | –0.1                   |
| `STOP_LOSS`      | SELL à un **prix stop** spécifié    | Active une vente si le prix retombe ≤ `stop_price`                                                                        | TTL = 10 steps            | –0.1                   |
| `TAKE_PROFIT`    | SELL à un **prix cible**            | Active une vente si le prix atteint ≥ `take_profit_price`                                                                 | TTL = 10 steps            | –0.0 (aucune pénalité) |
| `TRAILING_STOP`  | SELL avec **suivi dynamique**       | Met à jour `stop_price` au fil de la hausse du cours (`highest_price − trailing_pct%`) et vend si le prix chute ≤ ce stop | TTL = 10 steps            | –0.05                  |
| `STOP_LIMIT`     | STOP\_LOSS + LIMIT                  | Après déclenchement du stop, passe un ordre LIMIT au prix spécifié (évite slippage)                                       | TTL = 10 steps            | –0.1                   |

> **NB** : Tous les ordres avancés respectent aussi :
>
> * **Montant minimal** de transaction (`min_order_value` = \$10)
> * **Capital disponible** après frais (`transaction_cost_pct` + `fixed_fee`)
> * **Max positions** du palier courant (en cumulé : positions ouvertes + ordres en attente)

---

#### Sélection de la paire et multi-actifs

* **Actifs** : 5 paires (nommées `asset_0` à `asset_4`)
* L’agent choisit une seule paire par BUY/SELL, ce qui permet de diversifier les expositions **par palier** :

  * Palier 0 (capital < \$30) → ≤ 1 position
  * Palier 1 (30–75) → ≤ 2 positions
  * etc.

---

#### Pourquoi ces actions ?

1. **Couverture fonctionnelle** : couvre tous les scénarios de trading spot (achat, vente, limit, stop, profit, trailing).
2. **Gestion du risque** : respect des minimas, nombre de positions limité, frais anticipés.
3. **Modularité** : on peut ajouter facilement de nouveaux types d’ordres (ex. `OCO`, `ICEBERG`) en reprenant la même structure `_execute_order` / `_process_pending_orders`.

---

> **Checks à implémenter** avant la phase finale de training RL
>
> * **Validation** que chaque action codée entre 0–10 passe bien toutes les vérifications (`min_order_value`, `max_positions`, `capital_available`)
> * **Simulation** d’exemples d’ordres avancés pour confirmer l’exécution/expiration/annulation
> * **Logging**: conserver en `info["trades"]` le détail complet (`order_type`, `original_order_type`, `reason`, `pnl`, etc.)
> * **Test unitaires** couvrant toutes les combinaisons action–état–palier pour éviter des **invalid actions loops**

Avec cette définition précise des actions et de leurs contraintes, votre agent pourra trader de façon robuste et conforme à vos règles de gestion. Lorsque vous êtes prêt, dites **7** pour aborder la partie *récompenses* (reward shaping) !


### 7) Récompenses (Reward Shaping)

Pour guider l’agent vers un trading rentable et discipliné, nous définissons une **fonction de récompense** combinant plusieurs composantes :

| Composante                                                             | Formule / Mécanisme                                                                      | Objectif                                                                                  |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Rendement logarithmique**                                            | $r_{\text{log}} = \ln\!\bigl(\tfrac{\text{valeur\_après}}{\text{valeur\_avant}}\bigr)$   | Capturer la performance relative (gain ou perte en %), symétrique pour hausses et baisses |
| **Multiplicateurs de palier**                                          | Si $r_{\text{log}}\ge0$,                                                                 |                                                                                           |
|   $r_{\text{shaped}} = r_{\text{log}} \times \text{reward\_pos\_mult}$ |                                                                                          |                                                                                           |
| Sinon                                                                  |                                                                                          |                                                                                           |
|   $r_{\text{shaped}} = r_{\text{log}} \times \text{reward\_neg\_mult}$ | Renforcer les gains et/ou pénaliser plus fortement les pertes selon la taille du capital |                                                                                           |
| **Pénalités d’action invalides**                                       |                                                                                          |                                                                                           |
| & **Hold forcé**                                                       | –0.2 par ordre rejeté (min order, pas de position, max positions…)                       |                                                                                           |
| –0.05 par step de HOLD invalidé                                        |                                                                                          |                                                                                           |
| –0.001 par step de délai (time penalty)                                | Éviter que l’agent reste inactif ou enchaîne des actions impossibles                     |                                                                                           |
| **Pénalités d’expiration**                                             | –0.1 pour un LIMIT/STOP\_LOSS expiré, –0.05 pour un TRAILING\_STOP expiré                | Encourager l’agent à placer des ordres réalistes et à suivre leurs exécutions rapidement  |
| **Bonus take-profit**                                                  | +1% du PnL lorsque `pnl > 0`, plafonné à \$1                                             | Récompenser la prise de profits et la bonne détection des points hauts                    |
| **Clipping final**                                                     | $\text{reward} = \text{clip}(r_{\text{shaped}} - \text{pénalités},\; -10,\; +10)$        | Éviter les valeurs extrêmes qui destabilisent l’apprentissage                             |

---

#### Processus de calcul dans `step()`

1. **Snapshot “avant”**

   * $V_{\text{avant}} = \text{capital}_{t} + \sum \text{positions}_t$.
2. **Traitement des ordres pendants**

   * Appliquer pénalités d’expiration et gains de take-profit automatiques, accumuler `pending_reward`.
3. **Exécution de l’action choisie**

   * Acheter/vendre ou rejeter (+ pénalité immédiate si invalid).
4. **Snapshot “après”**

   * $V_{\text{après}} = \text{capital}_{t+1} + \sum \text{positions}_{t+1}$.
5. **Calcul du reward brut**

   $$
     r_{\text{log}} = \ln\!\bigl(\tfrac{V_{\text{après}}}{V_{\text{avant}}}\bigr), \quad
     r_{\text{shaped}} = 
       \begin{cases}
         r_{\text{log}} \times \text{pos\_mult}, & r_{\text{log}}\ge0,\\
         r_{\text{log}} \times \text{neg\_mult}, & r_{\text{log}}<0.
       \end{cases}
   $$
6. **Déduction des pénalités**

   * `r = r_shaped – penalties – time_penalty`.
7. **Bonus**

   * Ajouter 1% du PnL pour les trades gagnants.
8. **Clipping**

   * `r = clip(r, –10, +10)`.
9. **Mise à jour cumulative**

   * `cumulative_reward += r`.

---

#### Pourquoi ce shaping ?

* **Log-return** : bien adapté aux finances (invariance multiplicative).
* **Paliers** : adaptent l’aversion au risque selon la taille du capital.
* **Pénalités d’invalides** : empêchent l’agent d’exploiter des boucles de HOLD ou d’ordres impossibles pour maximiser la récompense…
* **Bonus take-profit** : favorise la réalisation concrète des gains plutôt que la simple valorisation théorique.
* **Clipping** : stabilise l’entraînement RL en évitant des extrêmes.

Lorsque vous êtes prêt, dites **8** pour discuter des **états et features** utilisés par l’agent !


### 8) États & Features (Représentation du State)

L’**état** observé par l’agent à chaque pas de temps combine :

1. **Features de marché** (historiques/prédictives)
2. **Features de portefeuille** (capital & positions)

---

#### 1. Market Features (𝐌 ∈ ℝᴺ)

Ces colonnes sont extraites directement du jeu de données (CSV / Parquet) et typiquement calculées en amont :

* Cours : `open`, `high`, `low`, `close`
* Volume
* Indicateurs techniques classiques : `ma_10` (moyenne mobile 10 périodes), `rsi` (Relative Strength Index), etc.
* (Éventuellement) autres momentum, volatilité, bandes de Bollinger…

> **Dimension** : N = nombre de colonnes numériques dans vos données.
> **Qualité & importance** :
>
> * Choisir des indicateurs fiables pour votre horizon (1 min, 5 min, 1 h…).
> * Plus vous donnez de features pertinentes, mieux l’agent peut capturer la structure du marché.
> * Évitez la redondance et le bruit excessif (feature selection).

---

#### 2. Portfolio Features (𝑷 ∈ ℝᵏ)

Nous ajoutons **k = 1 + (# actifs)** dimensions pour informer l’agent sur son portefeuille :

| Feature                   | Description                                  |
| ------------------------- | -------------------------------------------- |
| **Normalized Capital**    | capital / initial\_capital ∈ \[0,∞)          |
| **Positions Normalisées** | pour chaque actif *i* : qtyᵢ / typical\_qtyᵢ |

* **typical\_qtyᵢ** = (initial\_capital) ÷ (prixᵢ × facteur), où `facteur` est une constante (ex : 10)
  → met toutes les quantités sur une même échelle relative

> **Dimension** : k = 1 + 5 (pour 5 paires/actifs).
> **Pourquoi ?**
>
> * L’agent doit savoir **combien** il détient de chaque actif pour décider de SELL vs HOLD vs BUY.
> * Le capital normalisé l’aide à comprendre sa taille relative (paliers).

---

#### 3. Encodage / Scaling (optionnel)

Si vous avez un auto‐encodeur ou un `scaler` entraîné (PCA, Keras-AE…) :

1. **Scaler** standardise les features de marché (moyenne 0, écart-type 1)
2. **Encodeur** (auto‐encodeur) réduit la dimensionnalité et extrait des représentations non-linéaires

> **Avantage** : réduire le bruit, accélérer l’entraînement, extraire des facteurs latents.
> **Attention** : veillez à garder cohérents le nombre de features et l’ordre lors du `fit` / `transform` pour éviter les incohérences (erreur “43 vs 42”).

---

### ➤ **Résumé du vecteur d’observation**

$$
\text{state} =  
\bigl[\,\underbrace{f_1, f_2, \dots, f_N}_{\text{market features}},\;\;
\underbrace{\tfrac{\text{capital}}{\text{initial\_capital}}}_{\text{normalized cap}},\;
\underbrace{\tfrac{\text{qty}_0}{\text{typical}_0},\dots,\tfrac{\text{qty}_4}{\text{typical}_4}}_{\text{positions norm.}}\bigr]
\in \mathbb{R}^{N + 6}.
$$

* **N** = # de colonnes numériques
* **+6** = 1 (capital) + 5 (actifs)

---

> **Prochaine étape** : dites **9** pour passer aux **actions & espace d’action** (ordre, dimension, mapping).

### 9) Actions & Espace d’Action

L’agent dispose de **11 actions discrètes** (espace `Discrete(11)`) réparties ainsi :

| Code | Action             | Description                                                           |
| ---- | ------------------ | --------------------------------------------------------------------- |
| 0    | **HOLD**           | Ne rien faire ce pas de temps                                         |
| 1    | **BUY\_ASSET\_0**  | Acheter l’actif `asset_0` au prix du marché (order\_type=MARKET)      |
| 2    | **BUY\_ASSET\_1**  | Acheter l’actif `asset_1`                                             |
| 3    | **BUY\_ASSET\_2**  | Acheter l’actif `asset_2`                                             |
| 4    | **BUY\_ASSET\_3**  | Acheter l’actif `asset_3`                                             |
| 5    | **BUY\_ASSET\_4**  | Acheter l’actif `asset_4`                                             |
| 6    | **SELL\_ASSET\_0** | Vendre la position sur `asset_0` (order\_type=MARKET), si elle existe |
| 7    | **SELL\_ASSET\_1** | Vendre la position sur `asset_1`                                      |
| 8    | **SELL\_ASSET\_2** | Vendre la position sur `asset_2`                                      |
| 9    | **SELL\_ASSET\_3** | Vendre la position sur `asset_3`                                      |
| 10   | **SELL\_ASSET\_4** | Vendre la position sur `asset_4`                                      |

---

#### Mapping & Logique interne

1. **Interpréter l’action**

   ```python
   if action == 0:          action_type=0, asset_idx=None
   elif 1 ≤ action ≤ 5:     action_type=1 (BUY),  asset_idx=action-1
   elif 6 ≤ action ≤ 10:    action_type=2 (SELL), asset_idx=action-6
   ```

2. **Vérifications avant exécution**

   * **Min Order Value** : chaque ordre (`quantity × price`) ≥ `min_order_value` (ex. 10 \$).
   * **Capital disponible** : coût total (valeur + frais) ≤ `capital`.
   * **Max Positions** : nombre de positions ouvertes ≤ `tier["max_positions"]`.

3. **Types d’ordres**

   * **MARKET** (exécution immédiate)
   * **LIMIT**, **STOP\_LOSS**, **TAKE\_PROFIT**, **TRAILING\_STOP** (ordres avancés via `_process_pending_orders`).

4. **Frais & pénalités**

   * `fee = transaction_cost_pct * value + fixed_fee`
   * Invalidations donnent pénalité fixe (ex. −0.2) et masque l’action en `HOLD` avec petite pénalité de temps (−0.001).

---

#### Pourquoi ce choix ?

* **Discrete** est simple à entraîner (PPO, DQN, etc.).
* **Mapping clair** : chaque index correspond à (type, actif).
* **Extensible** : on peut ajouter ordres avancés si on enrichit l’espace (mais complexité RL augmente).

---

> **Prochaine étape** : tapez **10** pour passer au **Reward Design** (fonction de récompense).

### 10) Reward Design (Fonction de Récompense)

La fonction de récompense est au cœur de l’apprentissage par renforcement : elle guide l’agent vers le comportement souhaité. Pour notre environnement de trading, voici les composantes et justifications :

| Composante                     | Formule / Valeur                                                       | But                                                             |
| ------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Rendement Logarithmique**    | $r_t = \ln\!\Big(\frac{V_{t+1} + \varepsilon}{V_t + \varepsilon}\Big)$ | Capture les gains proportionnels, symétrique pour gains/pertes. |
| **Multiplicateurs par Palier** |                                                                        |                                                                 |
| – Gains                        | $r_t^+ = r_t \times \text{reward\_pos\_mult}$                          | Renforce plus fortement les gains aux paliers supérieurs.       |
| – Pertes                       | $r_t^- = r_t \times \text{reward\_neg\_mult}$                          | Dissuade les pertes en les pénalisant plus aux paliers élevés.  |
| **Pénalités**                  |                                                                        |                                                                 |
| – Invalidation d’ordre         | $-0.2$                                                                 | Décourage l’agent de placer des ordres non exécutables.         |
| – Expiration d’ordre           | $-0.1$ (Limit/SL), $-0.05$ (Trailing Stop)                             | Évite les ordres dormants trop longs sans exécution.            |
| – Temps (pas de trade)         | $-0.001$                                                               | Encourage à agir plutôt qu’à rester passif indéfiniment.        |
| **Clipping**                   | $\text{clip}(r_t, -10, +10)$                                           | Stabilise l’entraînement en bornant l’amplitude extrême.        |

#### Calcul complet par pas de temps :

1. **Avant** :

   * $V_t =$ valeur du portefeuille au début du step.
2. **Exécution** :

   * Appliquer ordres, frais, PnL, bonus sur trades ponctuels.
3. **Après** :

   * $V_{t+1} =$ valeur du portefeuille après exécution.
4. **Log-return** :

   * $r = \ln\frac{V_{t+1}}{V_t}$.
5. **Shaping** :

   * Si $r\ge0$ → $r \times \text{reward\_pos\_mult}$
   * Si $r<0$ → $r \times \text{reward\_neg\_mult}$
6. **Soustraction des pénalités** (invalidations, expiration, inactivité).
7. **Clipping** et sortie comme `reward`.

---

#### Pourquoi ?

* **Log-return** normalise les gains/pertes quel que soit le capital (scale invariant).
* **Multiplicateurs** adaptent l’aversion/accélération du risk-taking selon le capital.
* **Pénalités** disciplinent le comportement (éviter spam d’ordres invalides ou passivité).
* **Clipping** assure une stabilité numérique et évite les pics extrêmes qui déstabilisent l’agent.

---

🎯 **Objectif** : maximiser la croissance nette du portefeuille tout en se conformant aux contraintes de risque et de palier.

> Tapez **11** pour passer aux **Observations & Features** (structure des états).

### 11) Observations & Features (États)

L’agent perçoit à chaque pas un vecteur d’**état** qui combine :

---

#### A. **Données de marché**

Pour chaque actif considéré (ici 5 paires), on extrait un ensemble de **features techniques** issues du dataset :

* **Cours OHLCV** : open, high, low, close, volume
* **Indicateurs dérivés** (exemple) :

  * Moyennes mobiles (MA – ex. MA\_10, MA\_50)
  * Indicateur de force relative (RSI)
  * Bandes de Bollinger, MACD, etc.
* **Dimension** : si vous gardez 7 indicateurs par actif → 5 × 7 = 35 features

> **Pourquoi ?**
> Ces indicateurs capturent tendances, momentum et volatilité, permettant à l’agent de détecter des signaux d’achat/vente.

---

#### B. **Features de portefeuille**

1. **Capital Normalisé**

   $$
     c_t^\text{norm} = \frac{\text{capital}_t}{\text{capital}_\text{initial}}
   $$
2. **Positions** (pour chacun des 5 actifs) :

   $$
     p_{i,t} = \frac{\text{qty}_{i,t}}{\text{position\_typique}_i}
     \quad\text{où}\quad
     \text{position\_typique}_i = \frac{\text{capital}_\text{initial}}{\text{prix}_i \times 10}
   $$

   → donne une valeur relative entre 0 et 1

> **Dimension** : 1 (capital) + 5 (positions) = **6 features**

---

#### C. **Etat de l’ordre & du palier** *(optionnel mais utile)*

* **Tier actuel** (one-hot ou index numérique)
* **Nombre de positions ouvertes** vs **max\_positions**
* **Exposition relative** (allocation\_frac)
* **Nombre d’ordres en attente**

> **Pourquoi ?**
> Permet à l’agent de savoir s’il a atteint ses limites de risque/exposition et d’adapter son comportement.

---

#### D. **Encodage & Normalisation**

* **Scaler** (StandardScaler ou MinMax) sur les features de marché
* **Encodage** (auto-encodeur ou PCA) pour réduire la dimension si nécessaire

> **Attention** :
>
> * **Incohérence** “43 vs 42 features” → vérifier que le scaler et le vecteur d’état utilisent **exactement** la même dimension et les mêmes colonnes, dans le même ordre.
> * Toujours recalibrer le **scaler** lorsque vous modifiez le jeu de données ou le nombre d’indicateurs.

---

### 👉 **Résumé**

| Catégorie              | Features                    | Dimension  |
| ---------------------- | --------------------------- | ---------- |
| Marché                 | OHLCV + indicateurs (\~7×5) | \~35       |
| Portefeuille           | Capital norm. + Positions   | 6          |
| Palier & Ordres (opt.) | Tier, expos., n\_orders     | 3–5        |
| **Total**              |                             | **≈44–46** |

Ces **états** représentent le **contexte complet** (marché + portefeuille + contraintes) nécessaire à l’agent pour décider et apprendre efficacement.
Voici une proposition de **standardisation** pour vos features, un jeu de **7 indicateurs clés** par timeframe, un **workflow Colab** pour récupérer vos données (en tenant compte des restrictions régionales) et quelques suggestions « bonus » pour rendre ADAN encore plus robuste.

---

## 1. Convention de nommage

| Catégorie           | Exemple de nom                             | Description                                     |
| ------------------- | ------------------------------------------ | ----------------------------------------------- |
| Prix bruts          | `open`, `high`, `low`, `close`, `volume`   | Valeurs OHLCV classiques                        |
| Moyennes            | `ema_10`, `ema_20`, `sma_50`               | Moyennes exponentielles (10, 20) et simple (50) |
| Momentum            | `rsi_14`, `macd_12_26_9`                   | RSI sur 14 périodes, MACD (12,26, signal 9)     |
| Volatilité          | `bb_upper_20_2`, `bb_lower_20_2`, `atr_14` | Bandes de Bollinger (20, écart 2), ATR 14       |
| Tendance            | `adx_14`, `ma_200`                         | ADX 14 pour la force de tendance, MA 200 long   |
| Volume              | `obv`, `fvwap_1h`                          | On-Balance Volume, VWAP intra-horaire           |
| Cycle / Oscillateur | `stoch_k_14_3`, `stoch_d_14_3`             | Stochastique %K et %D                           |

> **Exemple de colonnes** pour 1 timeframe (e.g. 1 h) :
>
> ```
> ['open','high','low','close','volume',
>  'ema_10','ema_20','sma_50',
>  'rsi_14','macd_12_26_9',
>  'bb_upper_20_2','bb_lower_20_2','atr_14',
>  'adx_14','ma_200',
>  'obv','fvwap_1h',
>  'stoch_k_14_3','stoch_d_14_3']
> ```
>
> Vous pouvez bien sûr ajuster les périodes.

---

## 2. 7 indicateurs prioritaires par timeframe

1. **Tendance long terme** : `ma_200`
2. **Tendance court terme** : `ema_10`
3. **Volatilité** : `atr_14`
4. **Momentum** : `rsi_14`
5. **Cross-momentum** : `macd_12_26_9`
6. **Bandes de Bollinger** : `bb_upper_20_2` / `bb_lower_20_2`
7. **Volume** : `obv`

> Pour chaque timeframe (1 min, 5 min, 1 h, 24 h), dupliquez ces colonnes en suffixant le timeframe (e.g. `rsi_14_5m`, `atr_14_1h`).

---

## 3. Workflow Colab pour assembler votre dataset

1. **Installer les librairies**

   ```python
   !pip install python-binance pandas ta
   ```

2. **Configurer vos clés API**

   ```python
   from binance.client import Client
   import os
   API_KEY = os.environ.get("BINANCE_API_KEY")
   API_SECRET = os.environ.get("BINANCE_API_SECRET")
   client = Client(API_KEY, API_SECRET)
   ```

3. **Gérer les restrictions géographiques**

   * Certains pays (États-Unis, Canada, UK…) exigent une API dédiée (Binance .US, etc.).
   * **Astuce** : dans Colab, utilisez un VPN ou un proxy autorisé, ou basculez vers un exchange local (Coinbase Pro, Kraken) si Binance est bloqué.

4. **Télécharger les données OHLCV**

   ```python
   import pandas as pd
   def fetch_ohlcv(symbol, interval, start_str, end_str=None):
       klines = client.get_historical_klines(symbol, interval, start_str, end_str)
       df = pd.DataFrame(klines, columns=[
           'open_time','open','high','low','close','volume', *_ 
       ])  # adapter
       df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
       return df.set_index('open_time').astype(float)

   df_1h = fetch_ohlcv("BTCUSDT", "1h", "1 year ago UTC")
   ```

5. **Calculer les indicateurs**

   ```python
   import ta
   df = df_1h.copy()
   # Exemple : RSI
   df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
   # ATR
   df['atr_14'] = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close'],window=14).average_true_range()
   # ... et ainsi de suite pour vos 7 indicateurs
   ```

6. **Assembler plusieurs timeframes**

   * Téléchargez 1 m, 5 m, 24 h de la même façon.
   * **Merge** sur l’`open_time` (par exemple en upsampling / forward-fill).
   * Conservez uniquement les pas 1 m pour l’entraînement, en intégrant les colonnes des autres timeframes.

7. **Sauvegarde**

   ```python
   df.to_parquet("dataset_adan.parquet")
   ```

---

## 4. Bonus & recommandations « ADAN »

* **Backtesting & validation** :

  * Séparez votre jeu en **train/val/test** (80 %/10 %/10 %).
  * Vérifiez le **surapprentissage** : comparez performances en backtest & paper-trading.

* **Feature Drift detection** :

  * Intégrez un module pour détecter les dérives de vos indicateurs (p. ex. un changement soudain de volatilité).

* **Monitoring & Alerting** :

  * Loggez les décisions d’ADAN et mettez en place des alertes (Slack, email) si un comportement anormal se déclenche.

* **Robustesse géo-politique** :

  * Préparez des **fallbacks** d’exchanges (p. ex. Kraken, FTX US) si l’accès à Binance devient impossible.

* **Documentez vos versions** :

  * Versionnez vos colonnes (ex. `v1.0_indicators.yaml`) pour garder la traçabilité.

* **Esthétique & UX** :

  * Donnez à ADAN un **dashboard minimaliste** (Plotly, Streamlit) montrant : capital, PnL cumulatif, heatmap de positions, signaux d’entrée/sortie.

---

🎯 Avec cette structure, ADAN aura :

1. **Un état cohérent** et complet (44–46 dims),
2. **7+ indicateurs** par timeframe, bien nommés,
3. **Un pipeline Colab** robuste face aux restrictions régionales,
4. **Des modules bonus** pour garantir traçabilité, robustesse et présentation claire.

Bonne mise en place !

