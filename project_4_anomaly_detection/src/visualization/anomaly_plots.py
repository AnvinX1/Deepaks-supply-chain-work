"""
Anomaly Visualization

Plotting utilities for anomaly detection results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


def plot_anomalies(
    data: pd.DataFrame,
    anomaly_labels: np.ndarray,
    feature_x: str,
    feature_y: str,
    title: str = "Anomaly Detection",
    save_path: Optional[str] = None
) -> None:
    """Plot anomalies on 2D feature space."""
    plt.figure(figsize=(10, 6))
    
    normal_mask = anomaly_labels == 0
    anomaly_mask = anomaly_labels == 1
    
    plt.scatter(
        data.loc[normal_mask, feature_x],
        data.loc[normal_mask, feature_y],
        c='blue', alpha=0.5, label='Normal', s=20
    )
    
    plt.scatter(
        data.loc[anomaly_mask, feature_x],
        data.loc[anomaly_mask, feature_y],
        c='red', alpha=0.8, label='Anomaly', s=50, marker='x'
    )
    
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_feature_distributions(
    data: pd.DataFrame,
    features: List[str],
    anomaly_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot feature distributions with anomalies highlighted."""
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, feature in enumerate(features):
        if feature not in data.columns:
            continue
        
        ax = axes[i]
        
        if anomaly_labels is not None:
            normal_data = data.loc[anomaly_labels == 0, feature]
            anomaly_data = data.loc[anomaly_labels == 1, feature]
            
            ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', color='blue')
            ax.hist(anomaly_data, bins=50, alpha=0.8, label='Anomaly', color='red')
            ax.legend()
        else:
            ax.hist(data[feature], bins=50, alpha=0.7, color='blue')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution: {feature}')
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_anomaly_scores(
    scores: np.ndarray,
    threshold: float,
    title: str = "Anomaly Score Distribution",
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of anomaly scores with threshold."""
    plt.figure(figsize=(10, 5))
    
    plt.hist(scores, bins=50, alpha=0.7, edgecolor='white')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_time_series_anomalies(
    timestamps: pd.Series,
    values: np.ndarray,
    anomaly_labels: np.ndarray,
    title: str = "Time Series with Anomalies",
    save_path: Optional[str] = None
) -> None:
    """Plot time series with anomalies highlighted."""
    plt.figure(figsize=(14, 5))
    
    plt.plot(timestamps, values, 'b-', alpha=0.7, label='Values')
    
    anomaly_mask = anomaly_labels == 1
    plt.scatter(
        timestamps[anomaly_mask],
        values[anomaly_mask],
        c='red', s=50, zorder=5, label='Anomalies'
    )
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
