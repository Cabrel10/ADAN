"""
Training module for the ADAN trading bot.
"""
import os
import time
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

from ..common.utils import get_logger, get_path, ensure_dir_exists, load_config
from ..data_processing.feature_engineer import prepare_data_pipeline
from ..environment.multi_asset_env import MultiAssetEnv
from ..agent.ppo_agent import create_ppo_agent, save_agent, TradingCallback
from .callbacks import CustomTrainingInfoCallback, EvaluationCallback

logger = get_logger()
console = Console()

def adapt_config_for_training_timeframe(config):
    """
    Adapte automatiquement la configuration selon le training_timeframe.
    Cette fonction est maintenant simplifiée car la logique d'initialisation des features
    est centralisée dans MultiAssetEnv.
    
    Args:
        config: Configuration complète
        
    Returns:
        config: Configuration adaptée (sans modification)
    """
    training_timeframe = config.get('data', {}).get('training_timeframe', '1h')
    data_source_type = config.get('data', {}).get('data_source_type', 'legacy_pipeline')
    
    logger.info(f"⏰ Training timeframe: {training_timeframe}")
    logger.info(f"🔧 Data source type: {data_source_type}")
    
    # La logique d'initialisation des base_feature_names est maintenant dans MultiAssetEnv
    # Plus besoin de warning car la sélection des features est automatique selon le timeframe
    
    return config

def make_env(env_id, rank, seed=0, log_dir_base=None, df_env=None, config_env=None, scaler_env=None, encoder_env=None, max_episode_steps_override=None):
    """
    Utility function to create a single environment for parallel training.
    
    Args:
        env_id: Environment identifier
        rank: Process rank
        seed: Random seed
        log_dir_base: Base directory for logging
        df_env: DataFrame for this environment
        config_env: Configuration
        scaler_env: Scaler
        encoder_env: Encoder
        max_episode_steps_override: Max steps override
        
    Returns:
        Function that creates the environment
    """
    def _init():
        env_log_dir = os.path.join(log_dir_base, f"worker_{rank}") if log_dir_base else None
        env = MultiAssetEnv(df_env, config_env, scaler_env, encoder_env, max_episode_steps_override)
        
        if env_log_dir:
            ensure_dir_exists(env_log_dir)
            env = Monitor(env, env_log_dir)
        return env
    return _init

def setup_training_environment(config, train_df, val_df=None, scaler=None, encoder=None, max_episode_steps_override=None):
    """
    Setup training and validation environments with optional parallel processing.
    
    Args:
        config: Configuration dictionary.
        train_df: Training data.
        val_df: Validation data (optional).
        scaler: Pre-fitted scaler (optional).
        encoder: Pre-fitted encoder (optional).
        max_episode_steps_override: Optional override for maximum steps per episode.
        
    Returns:
        tuple: (train_env, val_env)
    """
    # Get n_envs from agent config
    n_envs = config.get('agent', {}).get('n_envs', 1)
    seed = config.get('agent', {}).get('seed', 42)
    
    # Base log directory for monitor logs
    log_dir_base_monitor = os.path.join(get_path('reports'), 'monitor_logs_vec')
    ensure_dir_exists(log_dir_base_monitor)
    
    # Setup training environment(s)
    if n_envs > 1:
        logger.info(f"Setting up parallel training with {n_envs} environments using SubprocVecEnv")
        train_env = SubprocVecEnv([
            make_env(0, i, seed + i, log_dir_base_monitor, train_df, config, scaler, encoder, max_episode_steps_override) 
            for i in range(n_envs)
        ])
    else:
        logger.info("Setting up single training environment with DummyVecEnv")
        train_env = DummyVecEnv([
            make_env(0, 0, seed, log_dir_base_monitor, train_df, config, scaler, encoder, max_episode_steps_override)
        ])
    
    # Setup validation environment (always single environment)
    val_env = None
    if val_df is not None and not val_df.empty:
        logger.info("Setting up validation environment")
        val_env = DummyVecEnv([
            make_env(1, 0, seed, None, val_df, config, scaler, encoder, max_episode_steps_override)
        ])
    
    return train_env, val_env

def train_agent(config_paths=None, config=None, override_params=None):
    """
    Train the trading agent.
    
    Args:
        config_paths: Dictionary of paths to configuration files.
        config: Pre-loaded configuration dictionary.
        override_params: Dictionary of parameters to override in the configuration.
        
    Returns:
        tuple: (trained_agent, train_env)
    """
    # Load configurations if not provided
    if config is None:
        if config_paths is None:
            config_paths = {
                'main': os.path.join(get_path('config'), 'main_config.yaml'),
                'data': os.path.join(get_path('config'), 'data_config.yaml'),
                'environment': os.path.join(get_path('config'), 'environment_config.yaml'),
                'agent': os.path.join(get_path('config'), 'agent_config.yaml')
            }
        
        config = {}
        for key, path in config_paths.items():
            config[key] = load_config(path)
        
        # Combine configurations
        config = {
            'general': config['main'].get('general', {}),
            'paths': config['main'].get('paths', {}),
            'data': config['data'],
            'environment': config['environment'],
            'agent': config['agent']
        }
        
        # Adapter la configuration selon le training_timeframe
        config = adapt_config_for_training_timeframe(config)
        # Apply override parameters if provided
        if override_params:
            logger.info("Applying override parameters to configuration")
            
            # Override initial_capital
            if 'initial_capital' in override_params:
                config['environment']['initial_capital'] = override_params['initial_capital']
                logger.info(f"Overriding initial_capital: {override_params['initial_capital']}")
            
            # Override training_data_file
            if 'training_data_file' in override_params:
                config['data']['training_data_file'] = override_params['training_data_file']
                logger.info(f"Overriding training_data_file: {override_params['training_data_file']}")
            
            # Override validation_data_file
            if 'validation_data_file' in override_params:
                config['data']['validation_data_file'] = override_params['validation_data_file']
                logger.info(f"Overriding validation_data_file: {override_params['validation_data_file']}")

            # Override training_timeframe in data config
            if 'training_timeframe' in override_params and override_params['training_timeframe'] is not None:
                if 'data' not in config:
                    config['data'] = {} # Ensure 'data' key exists
                config['data']['training_timeframe'] = override_params['training_timeframe']
                logger.info(f"Overriding training_timeframe in config['data'] to: {override_params['training_timeframe']}")
            
            # Override total_timesteps
            if 'total_timesteps' in override_params:
                config['agent']['total_timesteps'] = override_params['total_timesteps']
                logger.info(f"Overriding total_timesteps: {override_params['total_timesteps']}")
            
            # Override learning_rate
            if 'learning_rate' in override_params:
                config['agent']['policy']['learning_rate'] = override_params['learning_rate']
                logger.info(f"Overriding learning_rate: {override_params['learning_rate']}")
            
            # Override batch_size
            if 'batch_size' in override_params:
                config['agent']['ppo']['batch_size'] = override_params['batch_size']
                logger.info(f"Overriding batch_size: {override_params['batch_size']}")

            # Override PPO n_steps
            if 'n_steps' in override_params and override_params['n_steps'] is not None:
                if 'ppo' not in config['agent']: # Ensure 'ppo' sub-dictionary exists
                    config['agent']['ppo'] = {}
                config['agent']['ppo']['n_steps'] = override_params['n_steps']
                logger.info(f"Overriding PPO n_steps in config['agent']['ppo'] to: {override_params['n_steps']}")
        
    # Set random seed for reproducibility
    random_seed = config.get('general', {}).get('random_seed', 42)
    np.random.seed(random_seed)
    
    # Prepare data
    logger.info("Preparing data for training...")
    train_df, val_df, test_df = prepare_data_pipeline(config, is_training=True)
    
    # Logs détaillés pour vérifier le format du DataFrame fusionné
    logger.info(f"TRAINER - train_df chargé par prepare_data_pipeline. Shape: {train_df.shape if train_df is not None else 'None'}")
    if train_df is not None and not train_df.empty:
        logger.info(f"TRAINER - Colonnes de train_df APRÈS prepare_data_pipeline (premières 30): {train_df.columns.tolist()[:30] if train_df is not None and not train_df.empty else 'DataFrame vide/None'}")
        if train_df is not None and not train_df.empty and 'open_ADAUSDT' not in train_df.columns:  # Vérifier une colonne typique
            logger.error("TRAINER - ERREUR : train_df ne contient PAS les colonnes fusionnées attendues (ex: open_ADAUSDT) !")
        elif train_df is not None and not train_df.empty:
            logger.info("TRAINER - SUCCÈS : train_df semble contenir les colonnes fusionnées.")
        
        # Vérification cruciale de la présence des colonnes fusionnées
        assets = config.get('data', {}).get('assets', [])
        if assets:
            for asset in assets[:2]:  # Juste vérifier les 2 premiers actifs pour éviter trop de logs
                logger.info(f"TRAINER - train_df contient-il 'open_{asset}' ? {'open_' + asset in train_df.columns}")
                logger.info(f"TRAINER - train_df contient-il 'close_{asset}' ? {'close_' + asset in train_df.columns}")
        else:
            logger.warning("TRAINER - Aucun actif défini dans config['data']['assets']")
    else:
        logger.critical("TRAINER - ERREUR CRITIQUE - train_df est vide ou None après prepare_data_pipeline!")
        return None, None  # Arrêter si pas de données
    
    if train_df is None or train_df.empty:
        logger.critical("ERREUR CRITIQUE: Données d'entraînement non disponibles ou vides")
        return None, None
    
    # Setup training environment
    logger.info("Setting up training environment...")
    max_episode_steps_override = override_params.get('max_episode_steps') if override_params else None
    train_env, val_env = setup_training_environment(config, train_df, val_df, max_episode_steps_override=max_episode_steps_override)
    
    # Access training_timeframe for naming
    training_timeframe = config.get('data', {}).get('training_timeframe', 'default_tf')
    model_name_suffix = override_params.get('model_name_suffix', None) if override_params else None
    logger.info(f"Using training_timeframe='{training_timeframe}' and suffix='{model_name_suffix}' for model save paths.")

    # Create agent
    logger.info("Creating PPO agent...")
    # Tensorboard log directory could also be timeframe specific if desired, but not requested here.
    tensorboard_log_dir = os.path.join(get_path('reports'), 'tensorboard_logs')
    ensure_dir_exists(tensorboard_log_dir)
    agent = create_ppo_agent(train_env, config, tensorboard_log_dir)
    
    # Setup callbacks
    callbacks = []
    
    models_base_dir = get_path('models') # e.g., 'output/models'

    # Evaluation callback if validation environment is available
    if val_env is not None:
        eval_suffix = f"_{model_name_suffix}" if model_name_suffix else ""
        best_model_dir = os.path.join(models_base_dir, f'best_model_{training_timeframe}{eval_suffix}')
        # EvalCallback saves the model as 'best_model.zip' inside this path
        ensure_dir_exists(best_model_dir)
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=best_model_dir,
            log_path=os.path.join(get_path('reports'), 'eval_logs'), # Eval logs can stay common or be specific
            eval_freq=config.get('agent', {}).get('eval_freq', 10000),
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_freq = config.get('agent', {}).get('checkpoint_freq', 50000)
    checkpoint_suffix = f"_{model_name_suffix}" if model_name_suffix else ""
    checkpoints_dir = os.path.join(models_base_dir, f'checkpoints_{training_timeframe}{checkpoint_suffix}')
    # CheckpointCallback creates this directory if it doesn't exist.
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix=f"ppo_trading_{training_timeframe}{checkpoint_suffix}"
    )
    callbacks.append(checkpoint_callback)
    
    # Custom trading callback
    trading_cb_suffix = f"_{model_name_suffix}" if model_name_suffix else ""
    best_trading_model_path = os.path.join(models_base_dir, f'best_trading_model_{training_timeframe}{trading_cb_suffix}.zip')
    ensure_dir_exists(os.path.dirname(best_trading_model_path))
    trading_callback = TradingCallback(
        check_freq=config.get('agent', {}).get('check_freq', 10000),
        save_path=best_trading_model_path,
        verbose=1
    )
    callbacks.append(trading_callback)
    
    # Custom training info callback pour un affichage riche pendant l'entraînement
    custom_info_freq = config.get('agent', {}).get('custom_log_freq_rollouts', 1)  # Log tous les X rollouts
    custom_info_callback = CustomTrainingInfoCallback(check_freq=custom_info_freq, verbose=1)
    callbacks.append(custom_info_callback)
    
    # Note: Un callback EarlyStopping avancé pourra être ajouté dans une version future
    # pour arrêter l'entraînement selon des critères de performance spécifiques
    
    # Train the agent
    logger.info("Starting training...")
    total_timesteps = config.get('agent', {}).get('total_timesteps', 1000000)
    
    start_time = time.time()
    # Use models_base_dir defined earlier for consistency
    interrupted_model_save_dir = models_base_dir
    ensure_dir_exists(interrupted_model_save_dir)
    interrupted_suffix = f"_{model_name_suffix}" if model_name_suffix else ""
    interrupted_model_path = os.path.join(interrupted_model_save_dir, f'interrupted_model_{training_timeframe}{interrupted_suffix}.zip')
    
    try:
        logger.info("Starting agent training...")
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100 # Default SB3 log interval for console output
        )
        
        # Save the final model
        final_model_save_dir = models_base_dir
        ensure_dir_exists(final_model_save_dir)
        final_suffix = f"_{model_name_suffix}" if model_name_suffix else ""
        final_model_path = os.path.join(final_model_save_dir, f'final_model_{training_timeframe}{final_suffix}.zip')
        save_agent(agent, final_model_path)
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final model saved to {final_model_path}")
        
        # Afficher un tableau récapitulatif de la session d'entraînement
        summary_table = Table(title="[bold magenta]Résumé de la Session d'Entraînement ADAN[/bold magenta]")
        summary_table.add_column("Métrique", style="dim cyan")
        summary_table.add_column("Valeur")
        
        summary_table.add_row("Durée Totale Entraînement", f"{training_time:.2f}s")
        summary_table.add_row("Nombre Total de Timesteps", f"{agent.num_timesteps} / {total_timesteps}")
        summary_table.add_row("Modèle Final Sauvegardé", final_model_path)
        
        # Récupérer les dernières valeurs de TensorBoard (si possible)
        try:
            latest_values = agent.logger.name_to_value
            
            if 'rollout/ep_rew_mean' in latest_values:
                final_ep_rew = latest_values['rollout/ep_rew_mean']
                summary_table.add_row("Dernière Récompense Moyenne Épisode", f"{final_ep_rew:.4f}")
            
            if 'train/policy_loss' in latest_values:
                final_policy_loss = latest_values['train/policy_loss']
                summary_table.add_row("Dernière Perte Politique", f"{final_policy_loss:.6f}")
            
            if 'train/value_loss' in latest_values:
                final_value_loss = latest_values['train/value_loss']
                summary_table.add_row("Dernière Perte Valeur", f"{final_value_loss:.6f}")
        except Exception as e:
            logger.warning(f"Erreur lors de la récupération des métriques finales: {e}")
        
        console.print(summary_table)
        
        # Evaluate the final model
        if val_env is not None:
            mean_reward, std_reward = evaluate_policy(agent, val_env, n_eval_episodes=10)
            logger.info(f"Final model evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Ajouter les résultats d'évaluation au tableau récapitulatif
            eval_table = Table(title="[bold cyan]Évaluation Finale du Modèle[/bold cyan]")
            eval_table.add_column("Métrique", style="dim cyan")
            eval_table.add_column("Valeur")
            
            eval_table.add_row("Récompense Moyenne", f"{mean_reward:.4f}")
            eval_table.add_row("Écart-Type Récompense", f"{std_reward:.4f}")
            
            console.print(eval_table)
            
    except KeyboardInterrupt:
        # Save the model if training is interrupted
        logger.info("Training interrupted. Saving model...")
        save_agent(agent, interrupted_model_path)
        logger.info(f"Interrupted model saved to {interrupted_model_path}")
        console.print(Panel(f"[bold yellow]Entraînement interrompu. Modèle sauvegardé à:[/bold yellow]\n{interrupted_model_path}", title="Interruption"))
    except Exception as e:
        logger.error(f"Error during training: {e}")
        # Try to save the model if possible
        try:
            save_agent(agent, interrupted_model_path)
            logger.info(f"Model saved to {interrupted_model_path} despite error")
            console.print(Panel(f"[bold red]Erreur pendant l'entraînement:[/bold red]\n{str(e)}\n\n[bold yellow]Modèle sauvegardé à:[/bold yellow]\n{interrupted_model_path}", title="Erreur"))
        except Exception as save_error:
            logger.error(f"Could not save model: {save_error}")
            console.print(Panel(f"[bold red]Erreur pendant l'entraînement:[/bold red]\n{str(e)}\n\n[bold red]Impossible de sauvegarder le modèle:[/bold red]\n{str(save_error)}", title="Erreur Critique"))
    
    return agent, train_env

def resume_training(model_path, config_paths=None, config=None, train_df=None, override_params=None):
    """
    Resume training from a saved model.
    
    Args:
        model_path: Path to the saved model.
        config_paths: Dictionary of paths to configuration files.
        config: Pre-loaded configuration dictionary.
        train_df: Training data (if None, will be loaded from config).
        override_params: Dictionary of parameters to override in the configuration.
        
    Returns:
        tuple: (trained_agent, train_env)
    """
    from ..agent.ppo_agent import load_agent
    
    # Load configurations if not provided
    if config is None:
        if config_paths is None:
            config_paths = {
                'main': os.path.join(get_path('config'), 'main_config.yaml'),
                'data': os.path.join(get_path('config'), 'data_config.yaml'),
                'environment': os.path.join(get_path('config'), 'environment_config.yaml'),
                'agent': os.path.join(get_path('config'), 'agent_config.yaml')
            }
        
        config = {}
        for key, path in config_paths.items():
            config[key] = load_config(path)
        
        # Combine configurations
        combined_config = {
            'general': config['main'].get('general', {}),
            'data': config['data'],
            'environment': config['environment'],
            'agent': config['agent']
        }
        # Apply override parameters if provided
        if override_params:
            logger.info("Applying override parameters to configuration")
            
            # Override initial_capital
            if 'initial_capital' in override_params:
                config['environment']['initial_capital'] = override_params['initial_capital']
                logger.info(f"Overriding initial_capital: {override_params['initial_capital']}")
            
            # Override training_data_file
            if 'training_data_file' in override_params:
                config['data']['training_data_file'] = override_params['training_data_file']
                logger.info(f"Overriding training_data_file: {override_params['training_data_file']}")
            
            # Override validation_data_file
            if 'validation_data_file' in override_params:
                config['data']['validation_data_file'] = override_params['validation_data_file']
                logger.info(f"Overriding validation_data_file: {override_params['validation_data_file']}")
            
            # Override total_timesteps
            if 'total_timesteps' in override_params:
                # Vérifier si total_timesteps est dans agent ou dans ppo
                if 'total_timesteps' in config['agent']:
                    config['agent']['total_timesteps'] = override_params['total_timesteps']
                elif 'ppo' in config:
                    config['ppo']['total_timesteps'] = override_params['total_timesteps']
                else:
                    # Créer la structure si elle n'existe pas
                    config.setdefault('ppo', {})['total_timesteps'] = override_params['total_timesteps']
                logger.info(f"Overriding total_timesteps: {override_params['total_timesteps']}")
            
            # Override learning_rate
            if 'learning_rate' in override_params:
                # Vérifier si learning_rate est dans policy ou dans ppo
                if 'policy' in config and 'learning_rate' in config['policy']:
                    config['policy']['learning_rate'] = override_params['learning_rate']
                elif 'ppo' in config:
                    config['ppo']['learning_rate'] = override_params['learning_rate']
                else:
                    # Créer la structure si elle n'existe pas
                    config.setdefault('policy', {})['learning_rate'] = override_params['learning_rate']
                logger.info(f"Overriding learning_rate: {override_params['learning_rate']}")
            
            # Override batch_size
            if 'batch_size' in override_params:
                # Vérifier si batch_size est dans ppo
                if 'ppo' in config:
                    config['ppo']['batch_size'] = override_params['batch_size']
                else:
                    # Créer la structure si elle n'existe pas
                    config.setdefault('ppo', {})['batch_size'] = override_params['batch_size']
                logger.info(f"Overriding batch_size: {override_params['batch_size']}")
    
    # Prepare data if not provided
    if train_df is None:
        logger.info("Preparing data for training...")
        train_df, val_df, _ = prepare_data_pipeline(config, is_training=True)
        
        if train_df.empty:
            logger.error("No training data available")
            return None, None
    else:
        # Split the provided data for validation
        data_config = config.get('data', {})
        train_ratio = data_config.get('train_ratio', 0.7)
        val_ratio = data_config.get('val_ratio', 0.15)
        
        from ..data_processing.feature_engineer import split_data
        train_df, val_df, _ = split_data(
            train_df,
            train_ratio=train_ratio / (train_ratio + val_ratio),
            val_ratio=val_ratio / (train_ratio + val_ratio),
            test_ratio=0.0
        )
    
    # Setup training environment
    logger.info("Setting up training environment...")
    max_episode_steps_override = override_params.get('max_episode_steps') if override_params else None
    train_env, val_env = setup_training_environment(config, train_df, val_df, max_episode_steps_override=max_episode_steps_override)
    
    # Load the agent
    logger.info(f"Loading agent from {model_path}...")
    agent = load_agent(model_path, train_env)
    
    # Setup callbacks (same as in train_agent)
    callbacks = []
    
    # Access training_timeframe for naming in resume_training as well
    training_timeframe = config.get('data', {}).get('training_timeframe', 'default_tf') # Already available in config
    model_name_suffix = override_params.get('model_name_suffix', None) if override_params else None # Get suffix for resume
    logger.info(f"Using training_timeframe='{training_timeframe}' and suffix='{model_name_suffix}' for resumed model save paths.")

    models_base_dir = get_path('models') # e.g., 'output/models'

    # Evaluation callback if validation environment is available
    if val_env is not None:
        eval_suffix_resumed = f"_{model_name_suffix}" if model_name_suffix else ""
        best_model_dir_resumed = os.path.join(models_base_dir, f'best_model_{training_timeframe}{eval_suffix_resumed}_resumed')
        ensure_dir_exists(best_model_dir_resumed)
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=best_model_dir_resumed, # Dir for best_model.zip
            log_path=os.path.join(get_path('reports'), 'eval_logs'),
            eval_freq=config.get('agent', {}).get('eval_freq', 10000),
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_freq = config.get('agent', {}).get('checkpoint_freq', 50000)
    checkpoint_suffix_resumed = f"_{model_name_suffix}" if model_name_suffix else ""
    checkpoints_dir_resumed = os.path.join(models_base_dir, f'checkpoints_{training_timeframe}{checkpoint_suffix_resumed}_resumed')
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoints_dir_resumed,
        name_prefix=f"ppo_trading_{training_timeframe}{checkpoint_suffix_resumed}_resumed"
    )
    callbacks.append(checkpoint_callback)
    
    # Custom trading callback
    trading_cb_suffix_resumed = f"_{model_name_suffix}" if model_name_suffix else ""
    best_trading_model_resumed_path = os.path.join(models_base_dir, f'best_trading_model_{training_timeframe}{trading_cb_suffix_resumed}_resumed.zip')
    ensure_dir_exists(os.path.dirname(best_trading_model_resumed_path))
    trading_callback = TradingCallback(
        check_freq=config.get('agent', {}).get('check_freq', 10000),
        save_path=best_trading_model_resumed_path,
        verbose=1
    )
    callbacks.append(trading_callback)
    
    # Continue training
    logger.info("Resuming training...")
    additional_timesteps = config.get('agent', {}).get('additional_timesteps', 500000)
    
    start_time = time.time()
    agent.learn(
        total_timesteps=additional_timesteps,
        callback=callbacks,
        log_interval=100,
        reset_num_timesteps=False  # Continue from previous timesteps
    )
    training_time = time.time() - start_time
    
    logger.info(f"Training resumed and completed in {training_time:.2f} seconds")
    
    # Save the final model
    final_model_resumed_save_dir = models_base_dir
    ensure_dir_exists(final_model_resumed_save_dir)
    final_suffix_resumed = f"_{model_name_suffix}" if model_name_suffix else ""
    final_model_resumed_path = os.path.join(final_model_resumed_save_dir, f'final_model_{training_timeframe}{final_suffix_resumed}_resumed.zip')
    save_agent(agent, final_model_resumed_path)
    logger.info(f"Resumed final model saved to {final_model_resumed_path}")
    
    return agent, train_env
