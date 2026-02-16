"""
Training Module
===============

Script for training crop yield prediction models.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Optional

from data_preprocessing import DataPreprocessor
from models import build_model
from evaluation import ModelEvaluator


def load_data(data_dir: str) -> Dict[str, np.ndarray]:
    """
    Load preprocessed data from data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
    """
    data = {}
    for split in ['train', 'val', 'test']:
        X = np.load(os.path.join(data_dir, f'X_{split}.npy'))
        y = np.load(os.path.join(data_dir, f'y_{split}.npy'))
        data[f'X_{split}'] = X
        data[f'y_{split}'] = y
    return data


def train_model(
    model_type: str,
    data_dir: str,
    output_dir: str,
    sequence_length: int = 12,
    n_features: int = 10,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    **model_kwargs
) -> None:
    """
    Train a crop yield prediction model.
    
    Args:
        model_type: Type of model (lstm, cnn_lstm, transformer)
        data_dir: Directory containing preprocessed data
        output_dir: Directory to save model and results
        sequence_length: Length of input sequences
        n_features: Number of input features
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping_patience: Patience for early stopping
        **model_kwargs: Additional model parameters
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data(data_dir)
    
    # Build model
    print(f"Building {model_type} model...")
    model = build_model(
        model_type=model_type,
        sequence_length=sequence_length,
        n_features=n_features,
        learning_rate=learning_rate,
        **model_kwargs
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("Training model...")
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']]
    }
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Evaluate
    print("Evaluating model...")
    evaluator = ModelEvaluator()
    
    # Plot training history
    evaluator.plot_training_history(
        history_dict,
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # Predictions
    y_pred = model.predict(data['X_test']).flatten()
    y_true = data['y_test']
    
    # Load preprocessor for inverse transform
    preprocessor = DataPreprocessor()
    # Note: In practice, you would load the fitted scaler here
    # preprocessor.scaler_target = load_scaler(...)
    
    # Generate report
    report = evaluator.generate_report(y_true, y_pred, model_type)
    report.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Plot results
    evaluator.plot_predictions(
        y_true, y_pred,
        save_path=os.path.join(output_dir, 'predictions.png')
    )
    evaluator.plot_residuals(
        y_true, y_pred,
        save_path=os.path.join(output_dir, 'residuals.png')
    )
    
    print(f"Training complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train crop yield prediction model')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'cnn_lstm', 'transformer'],
                        help='Model type')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        data_dir=args.data_dir,
        output_dir=os.path.join(args.output_dir, args.model),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()
