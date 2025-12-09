"""
Utility functions for deep learning models in financial time series prediction.
Contains model builders and training utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, GRU, 
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    BatchNormalization, LayerNormalization, MultiHeadAttention
)

# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_lstm_model(
    sequence_length,
    n_features,
    lstm_units=128,
    lstm_layers=2,
    dropout_rate=0.3,
    dense_units=64,
    learning_rate=0.001
):
    """
    Build LSTM model for binary classification.
    
    Args:
        sequence_length: Number of time steps
        n_features: Number of features per time step
        lstm_units: Number of units in each LSTM layer
        lstm_layers: Number of LSTM layers (1, 2, or 3)
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in dense layer
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='LSTM_Model')
    
    # Input layer
    model.add(Input(shape=(sequence_length, n_features)))
    
    # LSTM layers
    for i in range(lstm_layers):
        return_sequences = (i < lstm_layers - 1)  # True for all but last layer
        model.add(LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            name=f'lstm_{i+1}'
        ))
        model.add(Dropout(dropout_rate, name=f'dropout_lstm_{i+1}'))
    
    # Dense layers
    model.add(Dense(dense_units, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout_dense'))
    
    # Output layer (binary classification)
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def build_gru_model(
    sequence_length,
    n_features,
    gru_units=128,
    gru_layers=2,
    dropout_rate=0.3,
    dense_units=64,
    learning_rate=0.001
):
    """
    Build GRU model for binary classification.
    
    Args:
        sequence_length: Number of time steps
        n_features: Number of features per time step
        gru_units: Number of units in each GRU layer
        gru_layers: Number of GRU layers
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in dense layer
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='GRU_Model')
    
    # Input layer
    model.add(Input(shape=(sequence_length, n_features)))
    
    # GRU layers
    for i in range(gru_layers):
        return_sequences = (i < gru_layers - 1)
        model.add(GRU(
            units=gru_units,
            return_sequences=return_sequences,
            name=f'gru_{i+1}'
        ))
        model.add(Dropout(dropout_rate, name=f'dropout_gru_{i+1}'))
    
    # Dense layers
    model.add(Dense(dense_units, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout_dense'))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def build_cnn_model(
    sequence_length,
    n_features,
    filters=[64, 128, 256],
    kernel_size=3,
    pool_size=2,
    dropout_rate=0.3,
    dense_units=128,
    learning_rate=0.001
):
    """
    Build 1D CNN model for binary classification.
    
    Args:
        sequence_length: Number of time steps
        n_features: Number of features per time step
        filters: List of filter counts for each Conv layer
        kernel_size: Size of convolutional kernel
        pool_size: Size of max pooling window
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in dense layer
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='CNN_Model')
    
    # Input layer
    model.add(Input(shape=(sequence_length, n_features)))
    
    # Convolutional blocks
    for i, n_filters in enumerate(filters):
        model.add(Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
        model.add(MaxPooling1D(pool_size=pool_size, name=f'max_pool_{i+1}'))
        model.add(Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))
    
    # Global average pooling (alternative to flatten)
    model.add(GlobalAveragePooling1D(name='global_avg_pool'))
    
    # Dense layers
    model.add(Dense(dense_units, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout_dense'))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout_rate=0.1):
    """
    Single Transformer encoder block.
    
    Args:
        inputs: Input tensor
        head_size: Size of each attention head
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed-forward network
        dropout_rate: Dropout rate
    
    Returns:
        Output tensor
    """
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout_rate
    )(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    
    # Add & Norm
    x1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(x1)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    
    # Add & Norm
    x2 = LayerNormalization(epsilon=1e-6)(x1 + ff_output)
    
    return x2

def build_transformer_model(
    sequence_length,
    n_features,
    num_transformer_blocks=2,
    head_size=256,
    num_heads=4,
    ff_dim=256,
    dropout_rate=0.2,
    dense_units=128,
    learning_rate=0.001
):
    """
    Build Transformer model for binary classification.
    
    Args:
        sequence_length: Number of time steps
        n_features: Number of features per time step
        num_transformer_blocks: Number of transformer encoder blocks
        head_size: Size of each attention head
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in feed-forward network
        dropout_rate: Dropout rate
        dense_units: Number of units in final dense layer
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(sequence_length, n_features))
    x = inputs
    
    # Stack transformer encoder blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout_rate)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Transformer_Model')
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def build_hybrid_cnn_lstm_model(
    sequence_length,
    n_features,
    conv_filters=[64, 128],
    kernel_size=3,
    lstm_units=128,
    lstm_layers=2,
    dropout_rate=0.3,
    dense_units=64,
    learning_rate=0.001
):
    """
    Build Hybrid CNN-LSTM model for binary classification.
    
    Args:
        sequence_length: Number of time steps
        n_features: Number of features per time step
        conv_filters: List of filter counts for Conv layers
        kernel_size: Size of convolutional kernel
        lstm_units: Number of units in LSTM layers
        lstm_layers: Number of LSTM layers
        dropout_rate: Dropout rate
        dense_units: Number of units in dense layer
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='Hybrid_CNN_LSTM_Model')
    
    # Input layer
    model.add(Input(shape=(sequence_length, n_features)))
    
    # CNN layers for feature extraction
    for i, n_filters in enumerate(conv_filters):
        model.add(Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
        model.add(Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))
    
    # LSTM layers for temporal modeling
    for i in range(lstm_layers):
        return_sequences = (i < lstm_layers - 1)
        model.add(LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            name=f'lstm_{i+1}'
        ))
        model.add(Dropout(dropout_rate, name=f'dropout_lstm_{i+1}'))
    
    # Dense layers
    model.add(Dense(dense_units, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate, name='dropout_dense'))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def plot_training_history(history, model_name, save_path=None):
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Keras History object
        model_name: Name of model for title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[1, 0].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    axes[1, 0].set_title('Model AUC', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision', linewidth=2, linestyle='--')
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Precision & Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def get_callbacks(model_name, patience=10, min_delta=0.001, MODELS_DIR='../models/'):
    """
    Get standard callbacks for model training.
    
    Args:
        model_name: Name for checkpoint file
        patience: Patience for early stopping
        min_delta: Minimum change to qualify as improvement
    
    Returns:
        List of callbacks
    """
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=f'{MODELS_DIR}{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    return callbacks_list

def load_sequences(asset, horizon, sequences_dir='../data_new/sequences/'):
    """
    Load preprocessed sequences for a given asset and horizon.
    """
    filepath = f'{sequences_dir}{asset}_{horizon}_sequences.npz'
    data = np.load(filepath)
    
    return (
        data['X_train'], data['X_val'], data['X_test'],
        data['y_train'], data['y_val'], data['y_test'],
        int(data['sequence_length']), int(data['n_features'])
    )

def load_class_weights(sequences_dir='../data_new/sequences/'):
    """Load pre-computed class weights."""
    with open(f'{sequences_dir}class_weights.pkl', 'rb') as f:
        return pickle.load(f)