"""
Feature extractors for the ADAN trading bot.

This module contains custom feature extractors for the RL agent,
including a CNN-based feature extractor for processing market data as images.
"""
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..common.utils import get_logger

logger = get_logger()


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for the ADAN trading bot.
    
    This extractor processes market data as image-like tensors and extracts
    features using convolutional layers. It's designed to work with
    observation spaces that are dictionaries containing 'image_features'
    and 'vector_features'.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, 
                 num_input_channels: int = 3, cnn_config: dict = None):
        """
        Initialize the CNN feature extractor for multi-timeframe market data.
        
        Args:
            observation_space: Observation space (must be a Dict space)
            features_dim: Dimension of the feature vector output (default: 256)
            num_input_channels: Number of input channels for the CNN (default: 3 for 1m, 1h, 3h timeframes)
            cnn_config: Configuration dictionary for the CNN architecture
        """
        # Initialize the parent class
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Ensure we have a Dict observation space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(observation_space)}")
        
        # Extract image shape from observation space
        if "image_features" not in observation_space.spaces:
            raise ValueError("Observation space must contain 'image_features'")
        
        # Get the shape of the image features
        image_space = observation_space.spaces["image_features"]
        if len(image_space.shape) == 3:
            # Shape is (channels, height, width)
            self.input_shape = image_space.shape
            # Override num_input_channels if shape already includes channels
            num_input_channels = self.input_shape[0]
        else:
            # Assume shape is (height, width) and use specified number of channels
            self.input_shape = (num_input_channels, *image_space.shape)
        
        logger.info(f"CNN input shape: {self.input_shape}")
        
        # Get vector features dimension
        if "vector_features" not in observation_space.spaces:
            raise ValueError("Observation space must contain 'vector_features'")
        self.vector_dim = observation_space.spaces["vector_features"].shape[0]
        
        # Default CNN configuration for market data
        default_cnn_config = {
            # First conv block: capture short-term patterns
            "conv_layers": [
                {"out_channels": 32, "kernel_size": (3, 3), "stride": 1, "padding": 1},
                {"out_channels": 32, "kernel_size": (3, 3), "stride": 1, "padding": 1},
                # Second conv block: capture medium-term patterns
                {"out_channels": 64, "kernel_size": (3, 3), "stride": 1, "padding": 1},
                {"out_channels": 64, "kernel_size": (3, 3), "stride": 1, "padding": 1},
                # Third conv block: capture long-term patterns
                {"out_channels": 128, "kernel_size": (3, 3), "stride": 1, "padding": 1}
            ],
            # Pooling layers for downsampling
            "pool_layers": [
                {"kernel_size": 2, "stride": 2},  # 1/2
                {"kernel_size": 2, "stride": 2},  # 1/4
                {"kernel_size": 2, "stride": 2},  # 1/8
                {"kernel_size": 2, "stride": 2}   # 1/16
            ],
            "activation": "leaky_relu",  # Better for financial data with negative returns
            "batch_norm": True,  # Add batch normalization
            "dropout": 0.3,  # Slightly higher dropout for regularization
            "fc_layers": [512, 256]  # Larger fully connected layers
        }
        
        # Use provided config or default
        self.cnn_config = cnn_config or default_cnn_config
        
        # Build CNN layers
        cnn_layers = []
        in_channels = self.input_shape[0]  # Number of input channels
        height, width = self.input_shape[1], self.input_shape[2]
        
        # Add convolutional layers
        for i, (conv_config, pool_config) in enumerate(
            zip(self.cnn_config["conv_layers"], self.cnn_config["pool_layers"])
        ):
            # Add convolutional layer
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_config["out_channels"],
                    kernel_size=conv_config["kernel_size"],
                    stride=conv_config["stride"],
                    padding=conv_config["padding"]
                )
            )
            
            # Add activation
            if self.cnn_config["activation"].lower() == "relu":
                cnn_layers.append(nn.ReLU())
            elif self.cnn_config["activation"].lower() == "leaky_relu":
                cnn_layers.append(nn.LeakyReLU(0.1))
            
            # Add pooling layer
            cnn_layers.append(
                nn.MaxPool2d(
                    kernel_size=pool_config["kernel_size"],
                    stride=pool_config["stride"]
                )
            )
            
            # Update dimensions after convolution and pooling
            # After convolution: (H + 2*padding - kernel_size) / stride + 1
            height = int((height + 2*conv_config["padding"] - conv_config["kernel_size"]) / conv_config["stride"] + 1)
            width = int((width + 2*conv_config["padding"] - conv_config["kernel_size"]) / conv_config["stride"] + 1)
            
            # After pooling: (H - kernel_size) / stride + 1
            height = int((height - pool_config["kernel_size"]) / pool_config["stride"] + 1)
            width = int((width - pool_config["kernel_size"]) / pool_config["stride"] + 1)
            
            # Update channels for next layer
            in_channels = conv_config["out_channels"]
        
        # Add flatten layer
        cnn_layers.append(nn.Flatten())
        
        # Calculate flattened size
        self.flattened_size = in_channels * height * width
        logger.info(f"CNN flattened size: {self.flattened_size} (channels: {in_channels}, height: {height}, width: {width})")
        
        # Create CNN module
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Add dropout
        self.dropout = nn.Dropout(self.cnn_config["dropout"])
        
        # Create fully connected layers for CNN output
        fc_layers = []
        fc_input_dim = self.flattened_size
        
        for fc_dim in self.cnn_config["fc_layers"]:
            fc_layers.append(nn.Linear(fc_input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_input_dim = fc_dim
        
        # Final layer to match features_dim
        fc_layers.append(nn.Linear(fc_input_dim, features_dim))
        
        # Create fully connected module
        self.fc = nn.Sequential(*fc_layers)
        
        # Linear layer for vector features
        self.vector_fc = nn.Sequential(
            nn.Linear(self.vector_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Combine CNN and vector features
        self.combined_fc = nn.Sequential(
            nn.Linear(features_dim + 32, features_dim),
            nn.ReLU()
        )
        
        logger.info(f"Initialized CustomCNNFeatureExtractor with features_dim={features_dim}")
    
    def forward(self, observations):
        """
        Process the observations through the CNN feature extractor.
        
        Args:
            observations: Dictionary containing 'image_features' and 'vector_features'
            
        Returns:
            th.Tensor: Extracted features
        """
        # Extract image and vector features
        image_features = observations["image_features"]
        vector_features = observations["vector_features"]
        
        # Debug: Log input shapes and value ranges
        if self.training:  # Only log during training to avoid cluttering logs
            logger.debug(f"Input image features shape: {image_features.shape}, dtype: {image_features.dtype}")
            logger.debug(f"Image features range: [{image_features.min():.4f}, {image_features.max():.4f}]")
            logger.debug(f"Input vector features shape: {vector_features.shape}")
        
        # Vérifier et corriger les valeurs NaN/Inf dans les features d'entrée
        if not th.all(th.isfinite(image_features)):
            logger.warning(f"Valeurs non finies détectées dans image_features")
            image_features = th.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if not th.all(th.isfinite(vector_features)):
            logger.warning(f"Valeurs non finies détectées dans vector_features")
            vector_features = th.nan_to_num(vector_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure we have a batch dimension
        if len(image_features.shape) == 3:
            # If batch dimension is missing, add it
            image_features = image_features.unsqueeze(0)
        
        # Normaliser les features pour éviter les valeurs extrêmes
        image_features = th.clamp(image_features, min=-10.0, max=10.0)
        vector_features = th.clamp(vector_features, min=-10.0, max=10.0)
        
        
        # Process image features through CNN
        batch_size = image_features.size(0)
        
        # Ensure the input has the correct shape: [batch, channels, height, width]
        if len(image_features.shape) == 3:
            # Add batch dimension if missing
            image_features = image_features.unsqueeze(0)
        
        # Process through CNN
        cnn_features = self.cnn(image_features)
        
        # Apply dropout if in training mode
        if self.training:
            cnn_features = self.dropout(cnn_features)
        
        # Process through fully connected layers
        cnn_features = self.fc(cnn_features)
        
        # Process vector features
        if len(vector_features.shape) == 1:
            # Add batch dimension if missing
            vector_features = vector_features.unsqueeze(0)
        
        vector_features = self.vector_fc(vector_features)
        
        # Combine features
        combined = th.cat([cnn_features, vector_features], dim=1)
        features = self.combined_fc(combined)
        
        # Debug: Log output shapes and value ranges
        if self.training:
            logger.debug(f"CNN features shape: {cnn_features.shape}")
            logger.debug(f"Vector features shape: {vector_features.shape}")
            logger.debug(f"Combined features shape: {features.shape}")
            logger.debug(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
        
        return features
