# Documentation Technique: Intégration du CNN et Stabilisation du Système

Ce document technique détaille les modifications apportées au projet ADAN pour intégrer le CNN Feature Extractor et stabiliser le système de récompenses.

## 1. Architecture du CNN Feature Extractor

### 1.1 Structure du modèle

Le CNN Feature Extractor implémenté dans `feature_extractors.py` est conçu pour traiter les données de marché sous forme d'images 2D, où:
- Les lignes représentent différents indicateurs techniques
- Les colonnes représentent différents pas de temps

L'architecture comprend:
- Des couches convolutionnelles pour extraire des motifs spatiaux
- Des couches de pooling pour réduire la dimensionnalité
- Des couches fully-connected pour combiner les features extraites avec les features vectorielles

### 1.2 Améliorations de stabilité

Les modifications suivantes ont été apportées pour améliorer la stabilité numérique:

```python
# Vérification et correction des valeurs NaN/Inf
if not th.all(th.isfinite(image_features)):
    logger.warning(f"Valeurs non finies détectées dans image_features")
    image_features = th.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)

# Normalisation des features pour éviter les valeurs extrêmes
image_features = th.clamp(image_features, min=-10.0, max=10.0)
vector_features = th.clamp(vector_features, min=-10.0, max=10.0)
```

Des vérifications similaires ont été ajoutées à chaque étape du traitement pour garantir que les valeurs restent finies tout au long du forward pass.

### 1.3 Gestion des erreurs

Un mécanisme de secours a été implémenté pour gérer les erreurs potentielles:

```python
try:
    # Traitement normal
    # ...
except Exception as e:
    logger.error(f"Erreur dans le forward pass du CNN: {e}")
    # En cas d'erreur, retourner un tenseur de zéros de la bonne dimension
    batch_size = image_features.shape[0]
    return th.zeros((batch_size, self.features_dim), device=image_features.device)
```

## 2. Stabilisation du Système de Récompenses

### 2.1 Modifications du calcul des récompenses

Le calcul des récompenses dans `reward_calculator.py` a été modifié pour améliorer la stabilité:

```python
# Plafonner les valeurs de portefeuille
MAX_PORTFOLIO_VALUE = 1e6  # 1 million USD maximum
capped_old_value = min(old_portfolio_value, MAX_PORTFOLIO_VALUE)
capped_new_value = min(new_portfolio_value, MAX_PORTFOLIO_VALUE)

# Vérifier que les valeurs sont positives
capped_old_value = max(capped_old_value, 1e-6)  # Éviter la division par zéro
capped_new_value = max(capped_new_value, 1e-6)

# Réduire le multiplicateur de log-return
adjusted_multiplier = 10.0  # Réduit de 100.0 à 10.0
log_return = log_return * adjusted_multiplier
```

### 2.2 Limitation des multiplicateurs

Les multiplicateurs de récompense ont été limités à des valeurs raisonnables:

```python
# Limiter le multiplicateur à une valeur raisonnable (entre 0.5 et 2.0)
reward_pos_mult = max(0.5, min(reward_pos_mult, 2.0))
reward_neg_mult = max(0.5, min(reward_neg_mult, 2.0))
```

### 2.3 Clipping des récompenses

Les limites de clipping des récompenses ont été réduites pour éviter les valeurs extrêmes:

```python
# Utiliser des limites plus strictes (-2.0, 2.0) au lieu de (-5.0, 5.0)
reward = np.clip(reward, -2.0, 2.0)
```

## 3. Gestion des Paliers (Tiers)

### 3.1 Plafonnement du capital

Le capital est désormais plafonné pour la sélection du palier:

```python
# Plafonner le capital à une valeur raisonnable pour éviter les overflows
MAX_CAPITAL = 1e6  # 1 million USD maximum pour la sélection du palier
capped_capital = min(capital, MAX_CAPITAL)
```

### 3.2 Vérification des clés nécessaires

Des vérifications ont été ajoutées pour s'assurer que les paliers contiennent toutes les clés requises:

```python
# Vérifier que le palier contient toutes les clés nécessaires
if 'allocation_frac_per_pos' not in current_tier:
    logger.warning(f"Palier sans allocation_frac_per_pos: {current_tier}")
    current_tier['allocation_frac_per_pos'] = 0.95  # Valeur par défaut
```

### 3.3 Limitation de l'allocation

L'allocation par position est limitée à des valeurs raisonnables:

```python
# Limiter l'allocation à une valeur raisonnable (entre 0.05 et 0.95)
current_tier['allocation_frac_per_pos'] = max(0.05, min(current_tier['allocation_frac_per_pos'], 0.95))
```

## 4. Configuration des Pénalités

Les pénalités ont été ajustées dans `environment_config.yaml` pour améliorer la stabilité:

```yaml
penalties:
  invalid_order_base: -0.3         # Réduit de -0.5 à -0.3
  time_step: -0.0005               # Réduit de -0.001 à -0.0005
  order_expiry: -0.1               # Réduit de -0.2 à -0.1
  out_of_funds: -0.5               # Réduit de -1.0 à -0.5
  # ...

reward_shaping:
  log_return_multiplier: 10.0      # Réduit de 100.0 à 10.0
  clip_min: -2.0                   # Réduit de -5.0 à -2.0
  clip_max: 2.0                    # Réduit de 5.0 à 2.0
```

## 5. Gestion des Indicateurs par Timeframe

### 5.1 Dynamique d'ajout des indicateurs

Dans `multi_asset_env.py`, la logique d'ajout des indicateurs a été améliorée pour gérer les indicateurs spécifiques à chaque timeframe:

```python
# Ajouter les indicateurs spécifiques au timeframe actuel
timeframe_indicators = [col for col in market_data.columns if f'_{timeframe}' in col]
for indicator in timeframe_indicators:
    if indicator not in base_features:
        base_features.append(indicator)
```

### 5.2 Transformation des features pour le CNN

Dans `state_builder.py`, la méthode `_get_market_features_as_image` a été modifiée pour traiter correctement les indicateurs spécifiques à chaque timeframe:

```python
# Vérifier si des indicateurs spécifiques au timeframe sont disponibles
timeframe_indicators = [col for col in market_data.columns if f'_{timeframe}' in col]
if timeframe_indicators:
    logger.info(f"Indicateurs spécifiques au timeframe {timeframe} trouvés: {timeframe_indicators}")
    # Ajouter ces indicateurs à la liste des features
    for indicator in timeframe_indicators:
        if indicator not in feature_list:
            feature_list.append(indicator)
```

## 6. Tests et Validation

Pour valider les modifications apportées, il est recommandé de:

1. Exécuter des tests unitaires pour vérifier le comportement du CNN Feature Extractor
2. Effectuer des tests d'intégration pour valider le flux complet de données
3. Surveiller les logs pendant l'entraînement pour détecter d'éventuelles valeurs non finies
4. Comparer les performances avant et après les modifications

## 7. Considérations pour le Futur

Pour continuer à améliorer la stabilité et les performances du système:

1. Envisager l'utilisation de techniques de normalisation plus avancées (BatchNorm, LayerNorm)
2. Implémenter des mécanismes d'early stopping basés sur la stabilité des récompenses
3. Explorer des architectures CNN alternatives (ResNet, EfficientNet) pour le feature extractor
4. Mettre en place un monitoring plus détaillé des gradients pendant l'entraînement
