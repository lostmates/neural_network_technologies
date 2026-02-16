"""
Evaluation Module
=================

Provides metrics calculation and visualization tools for
crop yield prediction model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """
    Class for evaluating crop yield prediction models.
    """
    
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary with MAE, RMSE, R2 metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Additional agricultural-specific metric: percentage error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regions: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            regions: List of region names (optional)
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Actual Yield (c/ha)', fontsize=12)
        ax1.set_ylabel('Predicted Yield (c/ha)', fontsize=12)
        ax1.set_title('Predicted vs Actual Yield', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time series plot
        ax2 = axes[1]
        x = np.arange(len(y_true))
        ax2.plot(x, y_true, 'b-', label='Actual', linewidth=2)
        ax2.plot(x, y_pred, 'r--', label='Predicted', linewidth=2)
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Yield (c/ha)', fontsize=12)
        ax2.set_title('Yield Prediction Over Time', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot residual analysis.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals distribution
        ax1 = axes[0]
        ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Residuals (c/ha)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Residuals', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Residuals vs predicted
        ax2 = axes[1]
        ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Yield (c/ha)', fontsize=12)
        ax2.set_ylabel('Residuals (c/ha)', fontsize=12)
        ax2.set_title('Residuals vs Predicted', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regional_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regions: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create regional comparison visualization.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            regions: List of region names
            save_path: Path to save the plot
        """
        df = pd.DataFrame({
            'Region': regions,
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        # Calculate regional metrics
        regional_metrics = df.groupby('Region').apply(
            lambda x: pd.Series({
                'MAE': mean_absolute_error(x['Actual'], x['Predicted']),
                'Mean_Actual': x['Actual'].mean(),
                'Mean_Predicted': x['Predicted'].mean()
            })
        ).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(regional_metrics))
        width = 0.35
        
        ax.bar(x - width/2, regional_metrics['Mean_Actual'], width, 
               label='Actual', color='steelblue', edgecolor='black')
        ax.bar(x + width/2, regional_metrics['Mean_Predicted'], width,
               label='Predicted', color='coral', edgecolor='black')
        
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Average Yield (c/ha)', fontsize=12)
        ax.set_title('Regional Yield Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(regional_metrics['Region'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return regional_metrics
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot model training history.
        
        Args:
            history: Training history dictionary from Keras
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1 = axes[0]
        ax1.plot(history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2 = axes[1]
        ax2.plot(history['mae'], label='Train MAE', linewidth=2)
        ax2.plot(history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE (c/ha)', fontsize=12)
        ax2.set_title('Training and Validation MAE', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        regions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Name of the model
            regions: Optional list of regions
            
        Returns:
            DataFrame with metrics
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        report = pd.DataFrame({
            'Model': [model_name],
            'MAE': [metrics['MAE']],
            'RMSE': [metrics['RMSE']],
            'R2': [metrics['R2']],
            'MAPE': [metrics['MAPE']]
        })
        
        print(f"\n{'='*50}")
        print(f"Evaluation Report: {model_name}")
        print(f"{'='*50}")
        print(f"MAE:  {metrics['MAE']:.4f} c/ha")
        print(f"RMSE: {metrics['RMSE']:.4f} c/ha")
        print(f"R²:   {metrics['R2']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"{'='*50}\n")
        
        return report
