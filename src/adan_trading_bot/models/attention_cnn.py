#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture CNN avancée avec mécanismes d'attention inter-canaux pour ADAN Trading Bot.
Implémente la tâche 7.2.1 - Ajouter mécanismes d'attention inter-canaux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ChannelAttention(nn.Module):
    """
    Mécanisme d'attention entre canaux (timeframes) pour pondérer dynamiquement
    l'importance de chaque timeframe selon le contexte du marché.
    """
    
    def __init__(self, n_channels: int = 3, reduction_ratio: int = 4):
        """
        Initialize channel attention module.
        
        Args:
            n_channels: Number of input channels (timeframes)
            reduction_ratio: Reduction ratio for the attention bottleneck
        """
        super(ChannelAttention, self).__init__()
        
        self.n_channels = n_channels
        self.reduction_ratio = reduction_ratio
        
        # Global average pooling and max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for attention computation
        hidden_dim = max(1, n_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_channels)
        )
        
        # Sigmoid activation for attention weights
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of channel attention.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, seq_len, n_features)
            
        Returns:
            Attention-weighted tensor of same shape as input
        """
        batch_size, n_channels, seq_len, n_features = x.shape
        
        # Global pooling to get channel-wise statistics
        avg_out = self.avg_pool(x).view(batch_size, n_channels)  # (B, C)
        max_out = self.max_pool(x).view(batch_size, n_channels)  # (B, C)
        
        # Compute attention weights
        avg_attention = self.shared_mlp(avg_out)  # (B, C)
        max_attention = self.shared_mlp(max_out)  # (B, C)
        
        # Combine and normalize attention weights
        attention_weights = self.sigmoid(avg_attention + max_attention)  # (B, C)
        
        # Apply attention weights
        attention_weights = attention_weights.view(batch_size, n_channels, 1, 1)
        attended_x = x * attention_weights
        
        return attended_x

class SpatialAttention(nn.Module):
    """
    Mécanisme d'attention spatiale pour identifier les features et timesteps
    les plus importants dans chaque timeframe.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention module.
        
        Args:
            kernel_size: Kernel size for the convolutional layer
        """
        super(SpatialAttention, self).__init__()
        
        self.kernel_size = kernel_size
        
        # Convolutional layer for spatial attention
        self.conv = nn.Conv2d(
            in_channels=2,  # avg and max pooled features
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, seq_len, n_features)
            
        Returns:
            Attention-weighted tensor of same shape as input
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, S, F)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, S, F)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_out, max_out], dim=1)  # (B, 2, S, F)
        
        # Compute spatial attention weights
        attention_weights = self.sigmoid(self.conv(pooled))  # (B, 1, S, F)
        
        # Apply attention weights
        attended_x = x * attention_weights
        
        return attended_x

class TimeframeInteractionModule(nn.Module):
    """
    Module pour modéliser les interactions entre timeframes avec des connexions
    croisées et de la fusion d'informations multi-échelles.
    """
    
    def __init__(self, n_channels: int = 3, hidden_dim: int = 64):
        """
        Initialize timeframe interaction module.
        
        Args:
            n_channels: Number of timeframes
            hidden_dim: Hidden dimension for interaction modeling
        """
        super(TimeframeInteractionModule, self).__init__()
        
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        
        # Cross-timeframe interaction layers
        self.interaction_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_channels)
        ])
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * n_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, timeframe_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of timeframe interaction.
        
        Args:
            timeframe_features: List of feature tensors for each timeframe
                                Each tensor has shape (batch_size, hidden_dim)
            
        Returns:
            Fused features tensor of shape (batch_size, hidden_dim)
        """
        batch_size = timeframe_features[0].shape[0]
        
        # Apply interaction transformations
        interacted_features = []
        for i, (features, interaction_layer) in enumerate(zip(timeframe_features, self.interaction_layers)):
            # Cross-timeframe attention
            attention_weights = []
            for j, other_features in enumerate(timeframe_features):
                if i != j:
                    # Compute attention between timeframes
                    attention = torch.softmax(
                        torch.sum(features * other_features, dim=-1, keepdim=True), 
                        dim=0
                    )
                    attention_weights.append(attention * other_features)
            
            # Combine with self-features
            if attention_weights:
                cross_attention = torch.stack(attention_weights, dim=0).mean(dim=0)
                combined_features = features + cross_attention
            else:
                combined_features = features
            
            # Apply interaction transformation
            interacted = interaction_layer(combined_features)
            interacted_features.append(interacted)
        
        # Fuse all timeframe features
        concatenated = torch.cat(interacted_features, dim=-1)  # (B, hidden_dim * n_channels)
        fused = self.fusion_layer(concatenated)  # (B, hidden_dim)
        
        # Output projection
        output = self.output_proj(fused)  # (B, hidden_dim)
        
        return output

class AttentionCNN(nn.Module):
    """
    Architecture CNN avancée avec mécanismes d'attention inter-canaux
    optimisée pour les données de trading multi-timeframes 3D.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (3, 100, 28),
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 dropout_rate: float = 0.1):
        """
        Initialize attention-based CNN.
        
        Args:
            input_shape: Input shape (n_channels, seq_len, n_features)
            hidden_dim: Hidden dimension for feature extraction
            n_layers: Number of CNN layers
            dropout_rate: Dropout rate for regularization
        """
        super(AttentionCNN, self).__init__()
        
        self.input_shape = input_shape
        self.n_channels, self.seq_len, self.n_features = input_shape
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Channel attention for timeframe weighting
        self.channel_attention = ChannelAttention(n_channels=self.n_channels)
        
        # Spatial attention for feature/time importance
        self.spatial_attention = SpatialAttention()
        
        # CNN layers for feature extraction
        self.cnn_layers = nn.ModuleList()
        
        # First layer
        self.cnn_layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, hidden_dim // 2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            )
        )
        
        # Hidden layers
        for i in range(n_layers - 2):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_rate)
                )
            )
        
        # Final layer
        self.cnn_layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            )
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Timeframe interaction module
        self.timeframe_interaction = TimeframeInteractionModule(
            n_channels=self.n_channels,
            hidden_dim=hidden_dim
        )
        
        # Final feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention CNN.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, seq_len, n_features)
            
        Returns:
            Feature tensor of shape (batch_size, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # Apply channel attention (timeframe weighting)
        x = self.channel_attention(x)
        
        # Apply spatial attention (feature/time importance)
        x = self.spatial_attention(x)
        
        # CNN feature extraction
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Global pooling to get per-channel features
        pooled = self.global_pool(x)  # (B, hidden_dim, 1, 1)
        pooled = pooled.view(batch_size, self.hidden_dim)  # (B, hidden_dim)
        
        # Extract features for each timeframe (simulate separate processing)
        timeframe_features = []
        for i in range(self.n_channels):
            # In practice, this would be more sophisticated
            # For now, we use the same pooled features for all timeframes
            timeframe_features.append(pooled)
        
        # Apply timeframe interaction
        interacted_features = self.timeframe_interaction(timeframe_features)
        
        # Final feature extraction
        output_features = self.feature_extractor(interacted_features)
        
        return output_features
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for visualization and analysis.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, seq_len, n_features)
            
        Returns:
            Dictionary containing attention weights
        """
        with torch.no_grad():
            # Channel attention weights
            batch_size, n_channels, seq_len, n_features = x.shape
            
            # Get channel attention weights
            avg_out = self.channel_attention.avg_pool(x).view(batch_size, n_channels)
            max_out = self.channel_attention.max_pool(x).view(batch_size, n_channels)
            
            avg_attention = self.channel_attention.shared_mlp(avg_out)
            max_attention = self.channel_attention.shared_mlp(max_out)
            
            channel_weights = torch.sigmoid(avg_attention + max_attention)
            
            # Spatial attention weights
            avg_spatial = torch.mean(x, dim=1, keepdim=True)
            max_spatial, _ = torch.max(x, dim=1, keepdim=True)
            pooled_spatial = torch.cat([avg_spatial, max_spatial], dim=1)
            spatial_weights = torch.sigmoid(self.spatial_attention.conv(pooled_spatial))
            
            return {
                'channel_attention': channel_weights,  # (B, C)
                'spatial_attention': spatial_weights,  # (B, 1, S, F)
                'timeframe_names': ['5m', '1h', '4h']
            }

class AttentionCNNPolicy(nn.Module):
    """
    Policy network utilisant l'architecture CNN avec attention pour
    l'apprentissage par renforcement dans le trading.
    """
    
    def __init__(self,
                 observation_space_shape: Tuple[int, int, int],
                 action_space_size: int,
                 hidden_dim: int = 128,
                 n_layers: int = 3):
        """
        Initialize attention CNN policy.
        
        Args:
            observation_space_shape: Shape of observation space
            action_space_size: Size of action space
            hidden_dim: Hidden dimension
            n_layers: Number of CNN layers
        """
        super(AttentionCNNPolicy, self).__init__()
        
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size
        
        # Feature extractor with attention
        self.feature_extractor = AttentionCNN(
            input_shape=observation_space_shape,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_space_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy network.
        
        Args:
            observations: Batch of observations
            
        Returns:
            Tuple of (action_logits, values)
        """
        # Extract features with attention
        features = self.feature_extractor(observations)
        
        # Compute policy and value
        action_logits = self.policy_head(features)
        values = self.value_head(features)
        
        return action_logits, values.squeeze(-1)
    
    def get_attention_analysis(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention analysis for the given observations.
        
        Args:
            observations: Batch of observations
            
        Returns:
            Dictionary containing attention analysis
        """
        return self.feature_extractor.get_attention_weights(observations)

def create_attention_cnn_model(observation_space_shape: Tuple[int, int, int],
                              action_space_size: int,
                              **kwargs) -> AttentionCNNPolicy:
    """
    Factory function to create an attention CNN model.
    
    Args:
        observation_space_shape: Shape of observation space
        action_space_size: Size of action space
        **kwargs: Additional arguments for model configuration
        
    Returns:
        Configured AttentionCNNPolicy model
    """
    return AttentionCNNPolicy(
        observation_space_shape=observation_space_shape,
        action_space_size=action_space_size,
        **kwargs
    )