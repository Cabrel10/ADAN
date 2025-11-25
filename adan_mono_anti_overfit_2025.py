# adan_mono_anti_overfit_2025.py
import pandas as pd
import numpy as np
import optuna
from mlfinlab.cross_validation import CombinatorialPurgedKFold
from mlfinlab.backtest_statistics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting
)
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PLACEHOLDERS: Remplacez ces fonctions par votre logique ADAN v1 ---

def train_adan_mono(train_df: pd.DataFrame, params: dict):
    """
    Fonction de remplacement pour entraîner le modèle ADAN v1 mono-timeframe.
    
    Args:
        train_df (pd.DataFrame): Le DataFrame d'entraînement.
        params (dict): Les hyperparamètres suggérés par Optuna.
        
    Returns:
        Un objet modèle factice.
    """
    logging.info(f"Entraînement du modèle avec les paramètres : {params} sur {len(train_df)} lignes.")
    # Simule un modèle entraîné
    model = {'params': params, 'trained_on': len(train_df)}
    return model

def backtest_adan_mono(model: dict, test_df: pd.DataFrame) -> pd.Series:
    """
    Fonction de remplacement pour backtester le modèle ADAN v1.
    
    Args:
        model (dict): Le modèle "entraîné".
        test_df (pd.DataFrame): Le DataFrame de test.
        
    Returns:
        pd.Series: Une série représentant la courbe de l'equity.
    """
    logging.info(f"Backtest du modèle sur {len(test_df)} lignes.")
    # Simule une courbe de performance en générant des rendements aléatoires
    # avec une légère tendance positive pour simuler une stratégie.
    random_returns = np.random.randn(len(test_df)) * 0.001 + 5e-5
    equity_curve = pd.Series(100 * (1 + random_returns).cumprod(), index=test_df.index)
    return equity_curve

# --- Fin des Placeholders ---


def objective(trial: optuna.Trial, df: pd.DataFrame, cpkf: CombinatorialPurgedKFold) -> float:
    """
    Fonction objectif pour l'optimisation Optuna.
    """
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 64, 512),
        'n_layers': trial.suggest_int('n_layers', 2, 8),
        'dropout': trial.suggest_float('dropout', 0.1, 0.7),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        # Ajoutez ici tous vos autres hyperparamètres ADAN v1
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }

    equity_curves = []
    logging.info(f"Début du Trial #{trial.number} avec les paramètres : {trial.params}")

    for i, (train_idx, test_idx) in enumerate(cpkf.split(df)):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]
        
        logging.info(f"  Fold {i+1}/{cpkf.n_test_splits}: Train size={len(train_df)}, Test size={len(test_df)}")

        # Entraîne ton ADAN v1 mono ici
        model = train_adan_mono(train_df, params)
        
        # Backtest et retourne une pd.Series de l'equity
        equity = backtest_adan_mono(model, test_df)
        equity_curves.append(equity.pct_change().fillna(0))

    # Calcul PBO et Deflated SR sur les N folds de test
    returns_matrix = np.column_stack([c.values for c in equity_curves])
    
    # Calcul du Sharpe Ratio annualisé
    # (288 périodes de 5m par jour, 252 jours de trading par an)
    sr_annualized = np.mean(returns_matrix.mean(axis=0) / returns_matrix.std(axis=0) * np.sqrt(252 * 288))
    
    if np.isnan(sr_annualized) or np.isinf(sr_annualized):
        logging.warning("Sharpe Ratio invalide (NaN ou Inf). Pruning du trial.")
        raise optuna.exceptions.TrialPruned()

    # Calcul de la probabilité d'overfitting
    pbo = probability_of_backtest_overfitting(returns_matrix, n_trials=len(equity_curves))
    
    # Calcul du Deflated Sharpe Ratio
    # n_trials = nombre de stratégies testées (ici, le nombre de folds)
    # years = durée totale des données en années
    total_years = (df.index.max() - df.index.min()).days / 365.25
    dsr = deflated_sharpe_ratio(sr_annualized, n_trials=len(equity_curves), years=total_years)

    logging.info(f"Trial #{trial.number} terminé. SR: {sr_annualized:.2f}, PBO: {pbo:.3f}, DSR: {dsr:.2f}")

    # Condition de pruning pour tuer les trials non performants
    if pbo > 0.25 or dsr < 1.5:
        logging.info(f"Pruning du trial #{trial.number} (PBO > 0.25 ou DSR < 1.5)")
        raise optuna.exceptions.TrialPruned()

    return dsr  # On maximise le Deflated Sharpe Ratio

def main():
    """
    Fonction principale pour lancer l'optimisation.
    """
    logging.info("1. Chargement des données 5m (doit commencer en 2018)")
    try:
        # Assurez-vous que ce fichier existe et est au bon format
        df = pd.read_parquet("data/btcusdt_5m_2018_2025.parquet")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        logging.info(f"Données chargées : {len(df)} lignes, de {df.index.min()} à {df.index.max()}")
    except FileNotFoundError:
        logging.error("ERREUR: Le fichier 'data/btcusdt_5m_2018_2025.parquet' est introuvable.")
        logging.error("Veuillez vous assurer que les données sont présentes avant de lancer le script.")
        return

    logging.info("2. Initialisation de Combinatorial Purged K-Fold (CPCV)")
    cpkf = CombinatorialPurgedKFold(
        n_train_windows=12,
        n_test_windows=6,
        purge=48,      # 4h de purge (48 * 5 minutes)
        embargo=24     # 2h d'embargo (24 * 5 minutes)
    )

    logging.info("3. Lancement de l'étude Optuna (maximisation du Deflated Sharpe Ratio)")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # Enveloppez l'objectif dans un lambda pour passer les arguments supplémentaires
    objective_with_args = lambda trial: objective(trial, df, cpkf)
    
    # Le nombre de trials est élevé, assurez-vous d'avoir la puissance de calcul nécessaire
    # 5000 trials peuvent prendre plusieurs jours.
    n_trials = 5000
    logging.info(f"Optimisation sur {n_trials} trials. Ceci peut prendre beaucoup de temps.")
    
    study.optimize(objective_with_args, n_trials=n_trials, timeout=None)

    logging.info("Optimisation terminée.")
    logging.info(f"Meilleurs hyperparamètres trouvés : {study.best_params}")
    logging.info(f"Meilleur Deflated Sharpe Ratio : {study.best_value}")

if __name__ == "__main__":
    main()
