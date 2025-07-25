# Documentation des Scripts ADAN

## Scripts Principaux

### Entraînement et Évaluation
- `train_rl_agent.py`: Entraînement de l'agent RL
- `train_rl_agent_no_rich.py`: Version simplifiée de l'entraînement
- `evaluate_final.py`: Évaluation finale des modèles
- `evaluate_models.py`: Évaluation des différents modèles
- `evaluate_performance.py`: Évaluation des performances
- `endurance_test.py`: Tests d'endurance
- `generate_endurance_metrics.py`: Génération des métriques d'endurance
- `validate_configs.py`: Validation des configurations
- `validate_orchestration_complete.py`: Validation de l'orchestration

### Trading
- `online_learning_agent.py`: Agent d'apprentissage en ligne
- `paper_trade_agent.py`: Agent pour le trading en mode paper
- `paper_trading_agent.py`: Agent de trading en mode paper
- `run_online_learning.py`: Exécution de l'apprentissage en ligne

### Données
- `generate_sample_data.py`: Génération de données de test
- `profile_performance_bottlenecks.py`: Profiling des performances
- `train_parallel_agents.py`: Entraînement parallèle des agents
- `vectorize_critical_calculations.py`: Vectorisation des calculs critiques
- `verify_indicators.py`: Vérification des indicateurs

## Scripts Dépréciés (dans scripts/legacy)

### Scripts de Préparation des Données
- `prepare_data.py`
- `process_data.py`
- `preprocess_indicators.py`
- `merge_processed_data.py`
- `simple_merge.py`
- `convert_real_data.py`
- `fetch_data_ccxt.py`
- `download_historical_data.py`
- `run_test_training.py`
- `simulate_capital_tiers.py`

### Scripts d'Analyse
- `analyze_dbe_logs.py`
- `analyze_dbe_metrics.py`
- `analyze_dynamic_behavior.py`
- `analyze_tensorboard_logs.py`

### Scripts de Migration
- `convert_real_data.py`
- `fetch_data_ccxt.py`
- `download_historical_data.py`
