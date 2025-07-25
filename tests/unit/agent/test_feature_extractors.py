import pytest
import torch as th
import numpy as np
import gymnasium as gym
from adan_trading_bot.agent.feature_extractors import CustomCNNFeatureExtractor, ChannelAttention

class TestFeatureExtractors:

    @pytest.fixture
    def observation_space(self):
        # Define a dummy observation space for testing
        return gym.spaces.Dict({
            "image_features": gym.spaces.Box(low=-1, high=1, shape=(3, 64, 10), dtype=np.float32),
            "vector_features": gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        })

    def test_cnn_feature_extractor_init(self, observation_space):
        # Test initialization without custom config
        extractor = CustomCNNFeatureExtractor(observation_space)
        assert isinstance(extractor, CustomCNNFeatureExtractor)
        assert extractor.features_dim == 256

        # Test initialization with custom config
        custom_cnn_config = {
            "conv_layers": [
                {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}
            ],
            "pool_layers": [
                {"kernel_size": 2, "stride": 2}
            ],
            "activation": "relu",
            "batch_norm": False,
            "dropout": 0.0,
            "fc_layers": [64],
            "use_channel_attention": True
        }
        extractor_custom = CustomCNNFeatureExtractor(observation_space, cnn_config=custom_cnn_config)
        assert isinstance(extractor_custom, CustomCNNFeatureExtractor)
        assert extractor_custom.cnn_config["use_channel_attention"] is True

    def test_cnn_feature_extractor_forward(self, observation_space):
        extractor = CustomCNNFeatureExtractor(observation_space)
        
        # Create dummy observations
        dummy_image_features = th.randn(1, 3, 64, 10)
        dummy_vector_features = th.randn(1, 5)
        observations = {
            "image_features": dummy_image_features,
            "vector_features": dummy_vector_features
        }

        # Test forward pass
        output = extractor(observations)
        assert output.shape == (1, extractor.features_dim)

        # Test with batch size > 1
        dummy_image_features_batch = th.randn(4, 3, 64, 10)
        dummy_vector_features_batch = th.randn(4, 5)
        observations_batch = {
            "image_features": dummy_image_features_batch,
            "vector_features": dummy_vector_features_batch
        }
        output_batch = extractor(observations_batch)
        assert output_batch.shape == (4, extractor.features_dim)

    def test_cnn_feature_extractor_nan_inf_handling(self, observation_space):
        extractor = CustomCNNFeatureExtractor(observation_space)

        # Create dummy observations with NaNs and Infs
        dummy_image_features_nan = th.randn(1, 3, 64, 10)
        dummy_image_features_nan[0, 0, 0, 0] = float('nan')
        dummy_image_features_nan[0, 1, 0, 0] = float('inf')

        dummy_vector_features_nan = th.randn(1, 5)
        dummy_vector_features_nan[0, 0] = float('nan')
        dummy_vector_features_nan[0, 1] = float('-inf')

        observations_nan = {
            "image_features": dummy_image_features_nan,
            "vector_features": dummy_vector_features_nan
        }

        # Test forward pass with NaNs/Infs (should handle them gracefully)
        output_nan = extractor(observations_nan)
        assert not th.isnan(output_nan).any()
        assert not th.isinf(output_nan).any()
        assert output_nan.shape == (1, extractor.features_dim)

    def test_channel_attention(self):
        # Test ChannelAttention module
        num_channels = 32
        x = th.randn(1, num_channels, 10, 10) # Batch, Channels, Height, Width
        attention_module = ChannelAttention(num_channels)
        output = attention_module(x)
        assert output.shape == x.shape
        # Check if attention weights are applied (output should not be identical to input unless weights are all 1)
        assert not th.allclose(output, x)

    def test_cnn_feature_extractor_with_channel_attention(self, observation_space):
        custom_cnn_config = {
            "conv_layers": [
                {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}
            ],
            "pool_layers": [
                {"kernel_size": 2, "stride": 2}
            ],
            "activation": "relu",
            "batch_norm": False,
            "dropout": 0.0,
            "fc_layers": [64],
            "use_channel_attention": True
        }
        extractor = CustomCNNFeatureExtractor(observation_space, cnn_config=custom_cnn_config)

        dummy_image_features = th.randn(1, 3, 64, 10)
        dummy_vector_features = th.randn(1, 5)
        observations = {
            "image_features": dummy_image_features,
            "vector_features": dummy_vector_features
        }

        output = extractor(observations)
        assert output.shape == (1, extractor.features_dim)
        # Further checks could involve inspecting the model's layers to ensure ChannelAttention is present
        # This is implicitly tested by the successful forward pass and the config setting.
