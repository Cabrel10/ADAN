import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, List, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for 3D observations (channels, height, width).
    Designed to process multi-timeframe market data.
    """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        channels: int = 3,
        kernel_sizes: List[int] = [3, 3, 3],
        strides: List[int] = [1, 1, 1],
        paddings: List[int] = [1, 1, 1],
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_batch_norm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)
        
        # Input shape: (channels, height, width)
        self.channels = channels
        self.height = observation_space.shape[1]
        self.width = observation_space.shape[2]
        
        # Store architecture parameters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.use_batch_norm = use_batch_norm
        
        # Calculate output dimensions after each conv layer
        def conv2d_size_out(size, kernel_size, stride, padding):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        # Build the CNN architecture
        layers = []
        in_channels = self.channels
        
        for i, (kernel_size, stride, padding) in enumerate(zip(kernel_sizes, strides, paddings)):
            # Add conv layer
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=64 * (2 ** i),  # Double channels each layer
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            
            # Add batch norm if enabled
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(64 * (2 ** i)))
            
            # Add activation
            layers.append(activation_fn())
            
            # Add dropout if enabled
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            # Update height and width after conv
            self.height = conv2d_size_out(self.height, kernel_size, stride, padding)
            self.width = conv2d_size_out(self.width, kernel_size, stride, padding)
            
            # Update input channels for next layer
            in_channels = 64 * (2 ** i)
        
        # Store the CNN layers
        self.cnn = nn.Sequential(*layers)
        
        # Calculate the size of the flattened features
        self.flattened_size = in_channels * self.height * self.width
        
        # Final linear layer to get to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, features_dim),
            nn.LayerNorm(features_dim),
            activation_fn(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights using He initialization.
        
        Args:
            module: The module to initialize weights for.
        """
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            observations: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature vector of shape (batch_size, features_dim)
        """
        # Ensure input is float32
        x = observations.float()
        
        # Pass through CNN
        x = self.cnn(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Pass through final linear layer
        features = self.fc(x)
        
        return features

# Dictionary to map string names to feature extractor classes
FEATURE_EXTRACTORS = {
    'cnn': CustomCNNFeatureExtractor,
    # Add other feature extractors here if needed
}

def get_feature_extractor(
    name: str,
    observation_space: gym.spaces.Space,
    **kwargs
) -> BaseFeaturesExtractor:
    """
    Get a feature extractor by name.
    
    Args:
        name: Name of the feature extractor ('cnn', etc.)
        observation_space: The observation space
        **kwargs: Additional arguments to pass to the feature extractor
        
    Returns:
        An instance of the specified feature extractor
        
    Raises:
        ValueError: If the feature extractor name is unknown
    """
    if name not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature extractor: {name}")
    
    return FEATURE_EXTRACTORS[name](observation_space, **kwargs)
