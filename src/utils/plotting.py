import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_style():
    """Set professional plotting style for book figures."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def plot_forecast(y_true, y_pred, title="Forecast vs Actual", xlabel="Time", ylabel="Value", save_path=None):
    """Plot actual vs predicted values."""
    set_style()
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='#1f77b4', linewidth=1.5)
    plt.plot(y_pred, label='Forecast', color='#ff7f0e', linestyle='--', linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt.gca()

def plot_probabilistic_forecast(y_true, quantiles, quantile_levels, title="Probabilistic Forecast", save_path=None):
    """Plot median forecast with uncertainty bands."""
    set_style()
    plt.figure(figsize=(10, 6))
    
    # Assuming quantiles are ordered and levels are symmetric (e.g., [0.1, 0.5, 0.9])
    median_idx = len(quantile_levels) // 2
    median = quantiles[:, median_idx]
    
    plt.plot(y_true, label='Actual', color='black', linewidth=1, alpha=0.7)
    plt.plot(median, label='Median Forecast', color='#1f77b4', linewidth=2)
    
    # Plot uncertainty bands
    for i in range(median_idx):
        lower = quantiles[:, i]
        upper = quantiles[:, -(i+1)]
        alpha_val = (i + 1) / (median_idx + 1) * 0.5
        plt.fill_between(range(len(y_true)), lower, upper, color='#1f77b4', alpha=alpha_val, 
                         label=f'{int((quantile_levels[-(i+1)] - quantile_levels[i])*100)}% Confidence Interval')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt.gca()
