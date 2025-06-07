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
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64, 
                 num_input_channels: int = 1, cnn_config: dict = None):
        """
        Initialize the CNN feature extractor.
        
        Args:
            observation_space: Observation space (must be a Dict space)
            features_dim: Dimension of the feature vector output
            num_input_channels: Number of input channels for the CNN
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
        else:
            # Assume shape is (height, width) and add channel dimension
            self.input_shape = (num_input_channels, *image_space.shape)
        
        # Get vector features dimension
        if "vector_features" not in observation_space.spaces:
            raise ValueError("Observation space must contain 'vector_features'")
        self.vector_dim = observation_space.spaces["vector_features"].shape[0]
        
        # Default CNN configuration
        default_cnn_config = {
            "conv_layers": [
                {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
            ],
            "pool_layers": [
                {"kernel_size": 2, "stride": 2},
                {"kernel_size": 2, "stride": 2}
            ],
            "activation": "relu",
            "dropout": 0.2,
            "fc_layers": [128]
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
        
        # Vérifier et corriger les valeurs NaN/Inf dans les features d'entrée
        if not th.all(th.isfinite(image_features)):
            logger.warning(f"Valeurs non finies détectées dans image_features")
            image_features = th.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if not th.all(th.isfinite(vector_features)):
            logger.warning(f"Valeurs non finies détectées dans vector_features")
            vector_features = th.nan_to_num(vector_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normaliser les features pour éviter les valeurs extrêmes
        image_features = th.clamp(image_features, min=-10.0, max=10.0)
        vector_features = th.clamp(vector_features, min=-10.0, max=10.0)
        
        # Ensure image features have the right shape
        if len(image_features.shape) == 3:
            # If batch dimension is missing, add it
            image_features = image_features.unsqueeze(0)
        
        # If channels dimension is missing (e.g., shape is [batch, height, width]), add it
        if len(image_features.shape) == 3 and self.input_shape[0] == 1:
            image_features = image_features.unsqueeze(1)
        
        # Process image features through CNN
        try:
            cnn_output = self.cnn(image_features)
            cnn_output = self.dropout(cnn_output)
            cnn_features = self.fc(cnn_output)
            
            # Vérifier les valeurs NaN/Inf après le CNN
            if not th.all(th.isfinite(cnn_features)):
                logger.warning(f"Valeurs non finies détectées dans cnn_features")
                cnn_features = th.nan_to_num(cnn_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Process vector features
            vector_output = self.vector_fc(vector_features)
            
            # Vérifier les valeurs NaN/Inf après le traitement des features vectorielles
            if not th.all(th.isfinite(vector_output)):
                logger.warning(f"Valeurs non finies détectées dans vector_output")
                vector_output = th.nan_to_num(vector_output, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Combine features
            combined_features = th.cat([cnn_features, vector_output], dim=1)
            output_features = self.combined_fc(combined_features)
            
            # Vérification finale des valeurs NaN/Inf
            if not th.all(th.isfinite(output_features)):
                logger.warning(f"Valeurs non finies détectées dans output_features")
                output_features = th.nan_to_num(output_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Assert pour vérifier que les logits sont valides
            assert th.all(th.isfinite(output_features)), "Logits non finis détectés!"
            
            return output_features
        except Exception as e:
            logger.error(f"Erreur dans le forward pass du CNN: {e}")
            # En cas d'erreur, retourner un tenseur de zéros de la bonne dimension
            batch_size = image_features.shape[0]
            return th.zeros((batch_size, self.features_dim), device=image_features.device)
