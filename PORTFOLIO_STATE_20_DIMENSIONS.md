# 📊 PORTFOLIO STATE - 20 DIMENSIONS EXACTES

**Source**: `src/adan_trading_bot/portfolio/portfolio_manager.py` - Méthode `get_state_vector()` (ligne 1228)

---

## 🎯 STRUCTURE COMPLÈTE

Le vecteur `portfolio_state` contient **exactement 20 dimensions** :

### **Bloc 1: 10 Features de Base**

```python
[0]  cash                          # Solde en espèces
[1]  total_value                   # Valeur totale du portefeuille
[2]  trading_pnl_pct               # PnL de trading pur (%)
[3]  external_flow_pct             # Impact des flux externes (%)
[4]  total_deposits_pct            # Total des dépôts (% du capital initial)
[5]  total_withdrawals_pct         # Total des retraits (% du capital initial)
[6]  sharpe_ratio                  # Ratio de Sharpe
[7]  drawdown_ratio                # Drawdown (converti de % à ratio)
[8]  open_positions_count          # Nombre de positions ouvertes
[9]  allocation_ratio              # Allocation = (total_value - cash) / total_value
```

### **Bloc 2: 10 Features pour les Positions (5 positions × 2 features)**

```python
[10] position_1_size               # Taille de la position 1
[11] position_1_asset_encoded      # Asset 1 encodé (hash % 1000 / 1000)
[12] position_2_size               # Taille de la position 2
[13] position_2_asset_encoded      # Asset 2 encodé
[14] position_3_size               # Taille de la position 3
[15] position_3_asset_encoded      # Asset 3 encodé
[16] position_4_size               # Taille de la position 4
[17] position_4_asset_encoded      # Asset 4 encodé
[18] position_5_size               # Taille de la position 5
[19] position_5_asset_encoded      # Asset 5 encodé
```

---

## 📝 CODE SOURCE EXACT

```python
def get_state_vector(self) -> np.ndarray:
    """Construit et retourne l'état du portefeuille sous forme de vecteur numpy."""
    try:
        metrics = self.get_metrics()
        total_value = metrics.get("total_value", 0.0)
        cash = metrics.get("cash", 0.0)

        # Obtenir les informations sur les flux de fonds
        fund_analysis = self.get_trading_pnl_vs_external_flows()
        trading_pnl_pct = (
            fund_analysis["trading_pnl"] / fund_analysis["adjusted_initial_capital"]
            if fund_analysis["adjusted_initial_capital"] > 0
            else 0.0
        )
        external_flow_pct = (
            fund_analysis["net_external_flow"] / self.initial_capital
            if self.initial_capital > 0
            else 0.0
        )

        # 10 features de base (incluant les flux de fonds)
        state = [
            cash,                                    # [0]
            total_value,                             # [1]
            trading_pnl_pct,                         # [2]
            external_flow_pct,                       # [3]
            fund_analysis["total_deposits"] / self.initial_capital if self.initial_capital > 0 else 0.0,  # [4]
            fund_analysis["total_withdrawals"] / self.initial_capital if self.initial_capital > 0 else 0.0,  # [5]
            metrics.get("sharpe_ratio", 0.0),       # [6]
            metrics.get("drawdown", 0.0) / 100.0,   # [7]
            metrics.get("open_positions_count", 0), # [8]
            (total_value - cash) / total_value if total_value > 0 else 0.0,  # [9]
        ]

        # 10 features pour les positions (5 positions * 2 features)
        sorted_positions = sorted(
            metrics.get("positions", {}).items(),
            key=lambda item: abs(
                item[1].get("size", 0.0) * item[1].get("current_price", 0.0)
            ),
            reverse=True,
        )[:5]

        for asset, pos_obj in sorted_positions:
            state.append(pos_obj.get("size", 0.0))                    # Position size
            state.append(hash(asset) % 1000 / 1000.0)                 # Asset encoded

        # Remplir les slots de positions restants avec des zéros
        num_pos_features = len(sorted_positions) * 2
        padding_needed = 10 - num_pos_features
        state.extend([0.0] * padding_needed)

        return np.array(state, dtype=np.float32)  # ← RETOURNE 20 DIMENSIONS

    except Exception as e:
        logger.error(f"Erreur lors de la construction du vecteur d'état: {e}")
        return np.zeros(20, dtype=np.float32)  # ← FALLBACK: 20 ZÉROS
```

---

## ✅ VALIDATION

**Dimensions totales**:
- Bloc 1 (features de base): 10
- Bloc 2 (positions): 10
- **Total: 20** ✅

**Type**: `np.float32`

**Fallback**: Si erreur, retourne `np.zeros(20, dtype=np.float32)`

---

## 🔧 CORRECTION REQUISE

### StateBuilder doit retourner 20 dimensions

Modifier `src/adan_trading_bot/data_processing/state_builder.py`:

```python
def get_portfolio_state_dim(self) -> int:
    """Retourne la dimension de l'état du portefeuille."""
    return 20  # ← DOIT ÊTRE 20, PAS 17

def build_portfolio_state(self, portfolio_manager: Any) -> np.ndarray:
    """Build portfolio state information."""
    if not self.include_portfolio_state or portfolio_manager is None:
        return np.zeros(20, dtype=np.float32)  # ← DOIT ÊTRE 20
    
    try:
        return portfolio_manager.get_state_vector()  # ← Retourne 20 dimensions
    except Exception as e:
        return np.zeros(20, dtype=np.float32)  # ← DOIT ÊTRE 20
```

---

## 📊 NOUVELLES DIMENSIONS TOTALES

Après correction:

```
5m:  19 fenêtres × 15 features = 285
1h:  10 fenêtres × 16 features = 160
4h:   5 fenêtres × 16 features =  80
Portfolio:                        = 20
─────────────────────────────────────
TOTAL:                          = 545
```

⚠️ **ATTENTION**: Le total passe de 542 à 545 !

Les modèles attendent peut-être 545, pas 542. À vérifier.

---

## 🎯 PROCHAINES ÉTAPES

1. ✅ Identifier les 20 dimensions (FAIT)
2. ⏭️ Modifier StateBuilder pour retourner 20
3. ⏭️ Vérifier que les modèles acceptent 545 dimensions
4. ⏭️ Relancer l'audit
5. ⏭️ Déployer

---

**Status**: 🟡 **CORRECTION EN COURS**
