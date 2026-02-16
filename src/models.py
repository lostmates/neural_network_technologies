"""
Model Architectures Module
==========================

Defines LSTM, Transformer, and CNN-LSTM hybrid models for
crop yield prediction from time series data.
"""

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from typing import Optional


class LSTMModel(keras.Model):
    """
    LSTM-based model for crop yield prediction.
    
    Architecture:
    - Multiple LSTM layers with dropout
    - Dense layers for regression output
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        lstm_units: list = [128, 64],
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        self.lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            self.lstm_layers.append(
                keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    input_shape=(sequence_length, n_features) if i == 0 else None
                )
            )
            self.lstm_layers.append(keras.layers.Dropout(dropout_rate))
        
        self.dense_layers = [
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(1, activation='linear')
        ]
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x, training=training)
        for layer in self.dense_layers:
            x = layer(x, training=training)
        return x


class CNNLSTMModel(keras.Model):
    """
    CNN-LSTM hybrid model for crop yield prediction.
    
    Architecture:
    - 1D CNN for feature extraction
    - LSTM for temporal dependencies
    - Dense layers for regression
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        cnn_filters: list = [64, 32],
        kernel_size: int = 3,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.conv_layers = []
        for filters in cnn_filters:
            self.conv_layers.extend([
                keras.layers.Conv1D(filters, kernel_size, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling1D(pool_size=2)
            ])
        
        self.lstm = keras.layers.LSTM(lstm_units, return_sequences=False)
        self.dropout = keras.layers.Dropout(dropout_rate)
        
        self.dense_layers = [
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(1, activation='linear')
        ]
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x, training=training)
        x = self.lstm(x, training=training)
        x = self.dropout(x, training=training)
        for layer in self.dense_layers:
            x = layer(x, training=training)
        return x


class TransformerBlock(keras.layers.Layer):
    """
    Transformer encoder block with multi-head attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(keras.Model):
    """
    Transformer-based model for crop yield prediction.
    
    Architecture:
    - Positional encoding
    - Multiple transformer blocks
    - Global average pooling
    - Dense layers for regression
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_projection = keras.layers.Dense(embed_dim)
        self.pos_embedding = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_blocks)
        ]
        
        self.global_pool = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(dropout_rate)
        
        self.dense_layers = [
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(1, activation='linear')
        ]
    
    def call(self, inputs, training=False):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        x = self.input_projection(inputs)
        x = x + self.pos_embedding(positions)
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        return x


def build_model(
    model_type: str,
    sequence_length: int,
    n_features: int,
    **kwargs
) -> keras.Model:
    """
    Factory function to build models.
    
    Args:
        model_type: One of 'lstm', 'cnn_lstm', 'transformer'
        sequence_length: Length of input sequences
        n_features: Number of input features
        **kwargs: Additional model-specific parameters
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'lstm':
        model = LSTMModel(sequence_length, n_features, **kwargs)
    elif model_type == 'cnn_lstm':
        model = CNNLSTMModel(sequence_length, n_features, **kwargs)
    elif model_type == 'transformer':
        model = TransformerModel(sequence_length, n_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=kwargs.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )
    
    return model
