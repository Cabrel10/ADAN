"""Modèles d'attention améliorés pour l'optimisation ADAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class EnhancedTemporalAttention(nn.Module):
    """Module d'attention temporelle amélioré avec suivi des poids"""

    def __init__(self, num_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = num_features // num_heads

        if num_features % num_heads != 0:
            raise ValueError(f"num_features ({num_features}) must be divisible by num_heads ({num_heads})")

        self.query = nn.Linear(num_features, num_features)
        self.key = nn.Linear(num_features, num_features)
        self.value = nn.Linear(num_features, num_features)
        self.out_proj = nn.Linear(num_features, num_features)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features)

        # Pour le suivi des poids d'attention
        self.attention_weights = []
        self.register_buffer('scale', torch.tensor(1.0 / (self.head_dim ** 0.5)))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Projections Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calcul des scores d'attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Stocker les poids pour analyse (seulement en mode entraînement)
        if self.training:
            self.attention_weights.append(attn_weights.detach().cpu().numpy())

        # Application des poids aux valeurs
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        # Connexion résiduelle et normalisation
        return self.layer_norm(x + output)

    def clear_attention_weights(self):
        """Réinitialise le suivi des poids d'attention"""
        self.attention_weights = []

    def get_attention_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des poids d'attention"""
        if not self.attention_weights:
            return {}

        weights = np.array(self.attention_weights)
        return {
            'mean_attention': np.mean(weights, axis=(0, 1, 2)),
            'std_attention': np.std(weights, axis=(0, 1, 2)),
            'max_attention': np.max(weights, axis=(0, 1, 2)),
            'min_attention': np.min(weights, axis=(0, 1, 2))
        }


class ChannelAttention(nn.Module):
    """Module d'attention sur les canaux amélioré"""

    def __init__(self, num_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_channels = max(num_channels // reduction_ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec attention sur les canaux"""
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))

        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights.view(x.size(0), x.size(1), 1, 1)


class MultiScaleFeatureExtractor(nn.Module):
    """Extracteur de caractéristiques multi-échelle"""

    def __init__(self, in_channels: int, out_channels_per_branch: int = 32,
                 kernel_sizes: Optional[List[int]] = None):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9]

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=k,
                         padding=k//2, bias=False),
                nn.BatchNorm1d(out_channels_per_branch),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ) for k in kernel_sizes
        ])

        self.fusion = nn.Sequential(
            nn.Linear(len(kernel_sizes) * out_channels_per_branch, out_channels_per_branch * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extraction multi-échelle"""
        branch_outputs = [branch(x) for branch in self.branches]
        combined = torch.cat(branch_outputs, dim=1)
        return self.fusion(combined)


class OptimizedMultiTimeframeCNN(nn.Module):
    """CNN multi-timeframe optimisé"""

    def __init__(self, input_shape: Tuple[int, int, int], n_features_per_tf: int = 10,
                 use_attention: bool = True, use_multiscale: bool = True):
        super().__init__()

        self.n_timeframes = input_shape[0]
        self.window_size = input_shape[1]
        self.n_features = input_shape[2]

        # Couches CNN par timeframe
        self.tf_convs = nn.ModuleList([
            self._build_tf_cnn(n_features_per_tf, use_attention, use_multiscale)
            for _ in range(self.n_timeframes)
        ])

        # Couche de fusion
        tf_output_size = 128  # Sortie après CNN
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.n_timeframes * tf_output_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self._initialize_weights()

    def _build_tf_cnn(self, n_features: int, use_attention: bool, use_multiscale: bool) -> nn.Module:
        """Construit le réseau CNN pour un timeframe"""
        layers = []

        if use_multiscale:
            layers.append(MultiScaleFeatureExtractor(n_features, 32))
        else:
            layers.append(nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ))

        if use_attention:
            layers.append(ChannelAttention(64))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Traitement avant: [batch, timeframes, window, features]"""
        batch_size = x.shape[0]

        # Traiter chaque timeframe
        tf_features = []
        for i in range(self.n_timeframes):
            tf_data = x[:, i, :, :].transpose(1, 2)  # [batch, features, window]
            features = self.tf_convs[i](tf_data)
            tf_features.append(features)

        # Fusion des features
        combined_features = torch.cat(tf_features, dim=1)
        fused = self.fusion_layer(combined_features)

        return fused

    def _initialize_weights(self):
        """Initialisation optimisée des poids"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)


class OptimizedPPOMemoryNetwork(nn.Module):
    """Réseau PPO optimisé avec mémoire"""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim

        # Réseau politique
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim * 2)
        )

        # Réseau de valeur
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Mémoire LSTM
        self.memory_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1 if hidden_dim > 64 else 0.0
        )

        # Attention temporelle
        self.temporal_attention = EnhancedTemporalAttention(hidden_dim // 2, num_heads=4)

    def forward(self, state: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Dict:
        """Traitement avec mémoire et attention"""

        # LSTM pour mémoire
        if hidden_state is not None:
            lstm_out, new_hidden = self.memory_lstm(state.unsqueeze(1), hidden_state)
        else:
            lstm_out, new_hidden = self.memory_lstm(state.unsqueeze(1))

        lstm_features = lstm_out.squeeze(1)

        # Attention temporelle
        attn_features = self.temporal_attention(lstm_features.unsqueeze(1)).squeeze(1)

        # Fusion des features
        combined_input = lstm_features + attn_features

        # Politique et valeur
        policy_logits = self.policy_net(combined_input)
        action_mean, action_logstd = policy_logits.chunk(2, dim=1)
        action_std = torch.exp(torch.clamp(action_logstd, -20, 2))

        value = self.value_net(combined_input)

        return {
            'action_mean': action_mean,
            'action_std': action_std,
            'value': value,
            'hidden_state': new_hidden,
            'lstm_features': lstm_features,
            'attention_features': attn_features
        }


class OptimizedIntegratedCNNPPOModel(nn.Module):
    """Modèle intégré CNN+PPO optimisé"""

    def __init__(self, input_shape: Tuple, action_dim: int = 2,
                 use_attention: bool = True, use_multiscale: bool = True):
        super().__init__()

        # CNN multi-timeframe
        self.cnn = OptimizedMultiTimeframeCNN(
            input_shape,
            use_attention=use_attention,
            use_multiscale=use_multiscale
        )

        cnn_output_dim = 128

        # PPO avec mémoire
        self.ppo = OptimizedPPOMemoryNetwork(cnn_output_dim, action_dim)

        # Métriques pour analyse
        self.feature_importance = {}

    def forward(self, observation: Dict[str, torch.Tensor],
                hidden_state: Optional[Tuple] = None) -> Dict:
        """Flux complet optimisé"""

        # Concaténer les observations des timeframes
        timeframe_tensors = []
        for tf_key in ['5m', '1h', '4h']:
            if tf_key in observation:
                timeframe_tensors.append(observation[tf_key])

        if not timeframe_tensors:
            raise ValueError("Aucune donnée de timeframe trouvée dans l'observation")

        # Stack les timeframes: [batch, timeframes, window, features]
        x = torch.stack(timeframe_tensors, dim=1)

        # CNN processing
        cnn_features = self.cnn(x)

        # PPO processing
        ppo_output = self.ppo(cnn_features, hidden_state)

        return ppo_output

    def update_attention_heads(self, new_num_heads: int):
        """Met à jour le nombre de têtes d'attention"""
        self.ppo.temporal_attention = EnhancedTemporalAttention(
            self.ppo.temporal_attention.layer_norm.normalized_shape[0],
            num_heads=new_num_heads
        )

    def get_attention_weights(self) -> List[np.ndarray]:
        """Récupère les poids d'attention"""
        return self.ppo.temporal_attention.attention_weights

    def clear_attention_weights(self):
        """Efface les poids d'attention"""
        self.ppo.temporal_attention.clear_attention_weights()


class OptimizedCNNPPOAgent:
    """Agent CNN-PPO optimisé"""

    def __init__(self, observation_shape: Tuple, action_dim: int = 2,
                 use_attention: bool = True, use_multiscale: bool = True):
        self.observation_shape = observation_shape
        self.action_dim = action_dim

        # Modèle intégré
        self.model = OptimizedIntegratedCNNPPOModel(
            observation_shape, action_dim, use_attention, use_multiscale
        )

        # État caché
        self.hidden_state = None

        # Métriques
        self.step_count = 0
        self.action_history = []
        self.reward_history = []

    def act(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Sélection d'action optimisée"""
        self.model.eval()

        with torch.no_grad():
            # Conversion en tenseur
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0)
                         for k, v in observation.items()}

            # Forward pass
            output = self.model(obs_tensor, self.hidden_state)

            # Sélection d'action
            action_mean = output['action_mean'].squeeze().numpy()
            action_std = output['action_std'].squeeze().numpy()

            # Échantillonnage gaussien
            action = np.random.normal(action_mean, action_std)

            # Clipping des actions
            action = np.clip(action, -1.0, 1.0)

            # Mise à jour de l'état caché
            self.hidden_state = output['hidden_state']

            # Stockage des métriques
            self.step_count += 1
            self.action_history.append(action)

        return action, {
            'action_mean': action_mean,
            'action_std': action_std,
            'value': output['value'].item(),
            'hidden_state': self.hidden_state
        }

    def remember(self, observation: np.ndarray, action: np.ndarray,
                 reward: float, next_observation: np.ndarray, done: bool):
        """Mémorisation optimisée"""
        self.reward_history.append(reward)

    def learn(self):
        """Apprentissage (à implémenter selon les besoins)"""
        pass

    def get_attention_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques d'attention"""
        return self.model.ppo.temporal_attention.get_attention_stats()

    def clear_attention_weights(self):
        """Efface les poids d'attention"""
        self.model.clear_attention_weights()


# Configuration pour Optuna
def create_model_for_optuna(trial, input_shape: Tuple, action_dim: int = 2) -> OptimizedIntegratedCNNPPOModel:
    """Crée un modèle pour l'optimisation avec Optuna"""

    # Hyperparamètres à optimiser
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    use_multiscale = trial.suggest_categorical('use_multiscale', [True, False])
    num_heads = trial.suggest_categorical('num_attention_heads', [4, 8, 16])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.1, 0.3)

    # Créer le modèle avec les hyperparamètres
    model = OptimizedIntegratedCNNPPOModel(
        input_shape=input_shape,
        action_dim=action_dim,
        use_attention=use_attention,
        use_multiscale=use_multiscale
    )

    # Mettre à jour les paramètres d'attention
    model.ppo.temporal_attention = EnhancedTemporalAttention(
        model.ppo.input_dim, num_heads=num_heads, dropout=dropout
    )

    return model
