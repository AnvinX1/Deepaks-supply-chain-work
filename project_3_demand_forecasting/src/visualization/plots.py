"""
Visualization Plots

Plotting utilities for time series forecasts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_forecast(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Forecast",
    save_path: Optional[str] = None
) -> None:
    """Plot time series forecast."""
    plt.figure(figsize=(14, 5))
    
    plt.plot(dates, actual, label='Actual', linewidth=1.5)
    plt.plot(dates, predicted, label='Forecast', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_components(
    df: pd.DataFrame,
    components: List[str],
    title: str = "Time Series Components",
    save_path: Optional[str] = None
) -> None:
    """Plot time series components."""
    n_components = len(components)
    fig, axes = plt.subplots(n_components, 1, figsize=(14, 3 * n_components))
    
    if n_components == 1:
        axes = [axes]
    
    for ax, comp in zip(axes, components):
        if comp in df.columns:
            ax.plot(df.index, df[comp])
            ax.set_ylabel(comp)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plot residual analysis."""
    residuals = actual - predicted
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residual distribution
    axes[0].hist(residuals, bins=50, edgecolor='white')
    axes[0].set_title('Residual Distribution')
    axes[0].set_xlabel('Residual')
    
    # Residuals over time
    axes[1].plot(residuals)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_title('Residuals Over Time')
    axes[1].set_xlabel('Time Step')
    
    # Actual vs Predicted
    axes[2].scatter(actual, predicted, alpha=0.5)
    min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[2].set_title('Actual vs Predicted')
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Predicted')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
