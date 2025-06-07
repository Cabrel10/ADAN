"""
PPO agent implementation for the ADAN trading bot.
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ..common.utils import get_logger, get_path, ensure_dir_exists
from .feature_extractors import CustomCNNFeatureExtractor

logger = get_logger()

def create_ppo_agent(env, config, tensorboard_log=None):
    """
    Create a PPO agent for the trading environment.
    
    Args:
        env: Trading environment.
        config: Configuration dictionary.
        tensorboard_log: Path for tensorboard logs.
        
    Returns:
        PPO: Configured PPO agent.
    """
    agent_config = config.get('agent', {})
    
    # Obtenir les configurations spécifiques à PPO et à la politique
    ppo_config = config.get('ppo', {})
    policy_config = config.get('policy', {})
    
    # Get PPO hyperparameters avec priorité : policy > ppo > agent
    # Paramètres liés à l'apprentissage
    learning_rate = policy_config.get('learning_rate', 
                   ppo_config.get('learning_rate', 
                   agent_config.get('learning_rate', 2.5e-4)))
    
    # Paramètres spécifiques à PPO
    n_steps = ppo_config.get('n_steps', 
              agent_config.get('n_steps', 2048))
    batch_size = ppo_config.get('batch_size', 
                agent_config.get('batch_size', 64))
    n_epochs = ppo_config.get('n_epochs', 
              agent_config.get('n_epochs', 10))
    
    # Paramètres de récompense et d'avantage
    gamma = policy_config.get('gamma', 
           ppo_config.get('gamma', 
           agent_config.get('gamma', 0.99)))
    gae_lambda = ppo_config.get('gae_lambda', 
               agent_config.get('gae_lambda', 0.95))
    
    # Paramètres de clipping et de coefficients
    clip_range = ppo_config.get('clip_range', 
               agent_config.get('clip_range', 0.2))
    clip_range_vf = ppo_config.get('clip_range_vf', 
                   agent_config.get('clip_range_vf', None))
    ent_coef = ppo_config.get('ent_coef', 
             agent_config.get('ent_coef', 0.01))
    vf_coef = ppo_config.get('vf_coef', 
            agent_config.get('vf_coef', 0.5))
    
    # Paramètres de gradient
    max_grad_norm = policy_config.get('max_grad_norm', 
                   ppo_config.get('max_grad_norm', 
                   agent_config.get('max_grad_norm', 0.5)))
    
    # Préparer les policy_kwargs pour SB3
    policy_kwargs = {}
    
    # 1. Architecture du réseau (net_arch)
    # Priorité : policy_kwargs > policy > network
    if 'policy_kwargs' in agent_config and 'net_arch' in agent_config['policy_kwargs']:
        policy_kwargs['net_arch'] = agent_config['policy_kwargs']['net_arch']
    elif 'net_arch' in policy_config:
        policy_kwargs['net_arch'] = policy_config['net_arch']
    elif 'network' in config:
        network_config = config['network']
        # Vérifier si nous avons des architectures séparées pour policy et value
        if 'policy_net_arch' in network_config and 'value_net_arch' in network_config:
            policy_kwargs['net_arch'] = [
                dict(
                    pi=network_config['policy_net_arch'],
                    vf=network_config['value_net_arch']
                )
            ]
        elif 'policy_net_arch' in network_config:
            policy_kwargs['net_arch'] = network_config['policy_net_arch']
        else:
            # Architecture par défaut si rien n'est spécifié
            policy_kwargs['net_arch'] = [dict(pi=[128, 64], vf=[128, 64])]
    else:
        # Architecture par défaut si rien n'est spécifié
        policy_kwargs['net_arch'] = [dict(pi=[128, 64], vf=[128, 64])]
    
    # 2. Fonction d'activation
    # Priorité : policy_kwargs > policy > network
    activation_fn = None
    if 'policy_kwargs' in agent_config and 'activation_fn' in agent_config['policy_kwargs']:
        activation_fn = agent_config['policy_kwargs']['activation_fn']
    elif 'activation_fn' in policy_config:
        activation_fn = policy_config['activation_fn']
    elif 'network' in config and 'activation_fn' in config['network']:
        activation_fn = config['network']['activation_fn']
    
    # Convertir la chaîne d'activation en fonction PyTorch si nécessaire
    if activation_fn is not None:
        if isinstance(activation_fn, str):
            import torch as th
            if activation_fn.lower() == 'relu':
                policy_kwargs['activation_fn'] = th.nn.ReLU
            elif activation_fn.lower() == 'tanh':
                policy_kwargs['activation_fn'] = th.nn.Tanh
            elif activation_fn.lower() == 'elu':
                policy_kwargs['activation_fn'] = th.nn.ELU
            elif activation_fn.lower() == 'leaky_relu':
                policy_kwargs['activation_fn'] = th.nn.LeakyReLU
        else:
            policy_kwargs['activation_fn'] = activation_fn
    
    # 3. Extracteur de caractéristiques (feature extractor)
    # Configurer l'extracteur de caractéristiques CNN
    policy_kwargs['features_extractor_class'] = CustomCNNFeatureExtractor
    
    # Récupérer les kwargs de l'extracteur de caractéristiques
    features_extractor_kwargs = {}
    
    # Récupérer la configuration CNN depuis data_config
    data_config = config.get('data', {})
    cnn_config = data_config.get('cnn_config', {})
    
    # Configurer les dimensions de l'extracteur de caractéristiques
    features_dim = cnn_config.get('features_dim', 64)
    features_extractor_kwargs['features_dim'] = features_dim
    
    # Récupérer le nombre de canaux d'entrée
    num_input_channels = cnn_config.get('num_input_channels', 1)
    features_extractor_kwargs['num_input_channels'] = num_input_channels
    
    # Récupérer la configuration CNN complète
    if cnn_config:
        features_extractor_kwargs['cnn_config'] = cnn_config
    
    # Fusionner avec les kwargs existants si présents
    if 'features_extractor_kwargs' in agent_config:
        # Priorité aux configurations explicites
        agent_features_kwargs = agent_config['features_extractor_kwargs']
        features_extractor_kwargs.update(agent_features_kwargs)
    
    # Définir les kwargs finaux
    policy_kwargs['features_extractor_kwargs'] = features_extractor_kwargs
    
    # Journaliser les paramètres de l'extracteur pour débogage
    logger.info(f"CNN features extractor configuration:")
    logger.info(f"  - features_dim: {features_extractor_kwargs.get('features_dim')}")
    logger.info(f"  - num_input_channels: {features_extractor_kwargs.get('num_input_channels')}")
    
    # Journaliser les paramètres de policy_kwargs pour débogage
    logger.info(f"Using policy_kwargs: {policy_kwargs}")
    
    # Set tensorboard log path if not provided
    if tensorboard_log is None:
        tensorboard_log = os.path.join(get_path('reports'), 'tensorboard_logs')
        ensure_dir_exists(tensorboard_log)
    
    # Get policy type from config
    policy_type = agent_config.get('policy_type', 'MultiInputPolicy')
    
    # Create the agent
    agent = PPO(
        policy=policy_type,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    logger.info(f"Created PPO agent with learning_rate={learning_rate}, batch_size={batch_size}, n_epochs={n_epochs}")
    
    return agent

def save_agent(agent, save_path):
    """
    Save a trained agent.
    
    Args:
        agent: Trained agent.
        save_path: Path to save the agent.
        
    Returns:
        str: Path where the agent was saved.
    """
    # Ensure directory exists
    ensure_dir_exists(os.path.dirname(save_path))
    
    # Save the agent (PPO n'accepte pas l'argument save_replay_buffer)
    agent.save(save_path)
    logger.info(f"Agent saved to {save_path}")
    
    return save_path

def load_agent(load_path, env=None):
    """
    Load a trained agent.
    
    Args:
        load_path: Path to load the agent from.
        env: Environment to use with the loaded agent.
        
    Returns:
        PPO: Loaded agent.
    """
    try:
        agent = PPO.load(load_path, env=env)
        logger.info(f"Agent loaded from {load_path}")
        return agent
    except Exception as e:
        logger.error(f"Error loading agent from {load_path}: {e}")
        raise

class TradingCallback(BaseCallback):
    """
    Callback for saving the agent during training.
    """
    
    def __init__(self, check_freq, save_path, verbose=1):
        """
        Initialize the callback.
        
        Args:
            check_freq: Frequency to check for saving.
            save_path: Path to save the agent.
            verbose: Verbosity level.
        """
        super(TradingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
    
    def _init_callback(self):
        """
        Initialize the callback.
        """
        # Create folder if needed
        if self.save_path is not None:
            ensure_dir_exists(os.path.dirname(self.save_path))
    
    def _on_step(self):
        """
        Called at each step of training.
        
        Returns:
            bool: Whether to continue training.
        """
        if self.n_calls % self.check_freq == 0:
            # Get current reward
            try:
                # Méthode 1: Utiliser les valeurs du logger SB3
                mean_reward = -np.inf
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                    latest_values = self.model.logger.name_to_value
                    if 'rollout/ep_rew_mean' in latest_values:
                        mean_reward = latest_values['rollout/ep_rew_mean']
                
                # Méthode 2: Fallback - calculer manuellement si ep_info_buffer existe
                if mean_reward == -np.inf and hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    if rewards:
                        mean_reward = np.mean(rewards)
                
                if mean_reward != -np.inf:
                    if self.verbose > 0:
                        logger.info(f"Num timesteps: {self.num_timesteps}")
                        logger.info(f"Mean reward: {mean_reward:.2f}")
                    
                    # Save if better than previous best
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            logger.info(f"Saving new best model to {self.save_path}")
                        # Nous supprimons explicitement l'argument save_replay_buffer
                        self.model.save(self.save_path)
            except Exception as e:
                import traceback
                logger.warning(f"Erreur lors de la récupération des métriques SB3: {e}")
                logger.warning(f"Trace: {traceback.format_exc()}")
        
        return True
