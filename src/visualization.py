"""
Visualization utilities for building damage classification.

This module provides functions for visualizing model performance,
training history, and comparing different models.
"""

import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, FIGURES_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_history(history_path):
    """
    Load training history from a pickle file.

    Parameters
    ----------
    history_path : str
        Path to the history pickle file

    Returns
    -------
    dict
        Dictionary containing training history
    """
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        return history
    except Exception as e:
        logger.error(f"Error loading history from {history_path}: {e}")
        return None


def plot_training_history(history_path, save_dir=None):
    """
    Plot training history from a pickle file.

    Parameters
    ----------
    history_path : str
        Path to the history pickle file
    save_dir : str, optional
        Directory to save the plots, by default None

    Returns
    -------
    tuple
        (accuracy_fig, loss_fig) - Figure objects for accuracy and loss plots
    """
    # Load history
    history = load_history(history_path)
    if not history:
        return None, None

    # Create directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Get model name from file path
    model_name = os.path.basename(history_path).split('_')[1].split('.')[0]

    # Plot accuracy
    acc_fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['accuracy'], 'b-', label='Training', linewidth=2)
    ax.plot(history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    ax.set_title(f'Model Accuracy ({model_name})', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11)

    # Set y-axis to start from 0.5 or min value, whichever is lower
    min_acc = min(min(history['accuracy']), min(history['val_accuracy']))
    ax.set_ylim([min(0.5, min_acc * 0.9), 1.0])

    # Save accuracy figure
    if save_dir:
        acc_path = os.path.join(save_dir, f'accuracy_{model_name}.png')
        acc_fig.savefig(acc_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved accuracy plot to {acc_path}")

    # Plot loss
    loss_fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['loss'], 'b-', label='Training', linewidth=2)
    ax.plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_title(f'Model Loss ({model_name})', fontsize=14)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=11)

    # Save loss figure
    if save_dir:
        loss_path = os.path.join(save_dir, f'loss_{model_name}.png')
        loss_fig.savefig(loss_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss plot to {loss_path}")

    return acc_fig, loss_fig


def compare_models(history_paths, save_path=None):
    """
    Compare training histories from multiple models.

    Parameters
    ----------
    history_paths : list
        List of paths to history pickle files
    save_path : str, optional
        Path to save the comparison figure, by default None

    Returns
    -------
    tuple
        (accuracy_fig, loss_fig) - Figure objects for accuracy and loss comparisons
    """
    # Set up the style
    sns.set_style("whitegrid")

    # Create figures
    acc_fig, acc_ax = plt.subplots(figsize=(12, 7))
    loss_fig, loss_ax = plt.subplots(figsize=(12, 7))

    # Color cycle for different models
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    for i, history_path in enumerate(history_paths):
        # Load history
        history = load_history(history_path)
        if not history:
            continue

        # Get model name from file path
        model_name = os.path.basename(history_path).split('_')[1].split('.')[0]
        color = colors[i % len(colors)]

        # Plot accuracy
        acc_ax.plot(history['accuracy'],
                    linestyle='-',
                    color=color,
                    alpha=0.7,
                    linewidth=2,
                    label=f'{model_name} (Train)')

        acc_ax.plot(history['val_accuracy'],
                    linestyle='--',
                    color=color,
                    alpha=1.0,
                    linewidth=2,
                    label=f'{model_name} (Val)')

        # Plot loss
        loss_ax.plot(history['loss'],
                     linestyle='-',
                     color=color,
                     alpha=0.7,
                     linewidth=2,
                     label=f'{model_name} (Train)')

        loss_ax.plot(history['val_loss'],
                     linestyle='--',
                     color=color,
                     alpha=1.0,
                     linewidth=2,
                     label=f'{model_name} (Val)')

    # Configure accuracy plot
    acc_ax.set_title('Model Accuracy Comparison', fontsize=16)
    acc_ax.set_ylabel('Accuracy', fontsize=14)
    acc_ax.set_xlabel('Epoch', fontsize=14)
    acc_ax.legend(loc='lower right', fontsize=12)
    acc_ax.grid(True, linestyle='--', alpha=0.7)

    # Configure loss plot
    loss_ax.set_title('Model Loss Comparison', fontsize=16)
    loss_ax.set_ylabel('Loss', fontsize=14)
    loss_ax.set_xlabel('Epoch', fontsize=14)
    loss_ax.legend(loc='upper right', fontsize=12)
    loss_ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    acc_fig.tight_layout()
    loss_fig.tight_layout()

    # Save figures if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        acc_fig.savefig(f"{save_path}_accuracy.png", dpi=300, bbox_inches='tight')
        loss_fig.savefig(f"{save_path}_loss.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plots to {save_path}_accuracy.png and {save_path}_loss.png")

    return acc_fig, loss_fig


def plot_model_metrics_comparison(report_paths, save_path=None):
    """
    Compare performance metrics across multiple models using bar charts.

    Parameters
    ----------
    report_paths : list
        List of paths to classification report CSV files
    save_path : str, optional
        Path to save the comparison figure, by default None

    Returns
    -------
    matplotlib.figure.Figure
        Figure object for the metrics comparison
    """
    # Set up the style
    sns.set_style("whitegrid")

    # Lists to store data
    model_names = []
    precision = []
    recall = []
    f1_score = []
    accuracy = []

    # Load metrics from each report
    for report_path in report_paths:
        try:
            # Get model name from file path
            model_name = os.path.basename(report_path).split('_')[0]
            model_names.append(model_name)

            # Load report
            report_df = pd.read_csv(report_path, index_col=0)

            # Extract macro-average metrics
            if 'macro avg' in report_df.index:
                precision.append(report_df.loc['macro avg', 'precision'])
                recall.append(report_df.loc['macro avg', 'recall'])
                f1_score.append(report_df.loc['macro avg', 'f1-score'])

                # Extract accuracy from the last row (assuming it's the accuracy row)
                if 'accuracy' in report_df.index:
                    accuracy.append(report_df.loc['accuracy', 'f1-score'])  # accuracy is stored in f1-score column
                else:
                    accuracy.append(None)
            else:
                # If no macro avg, calculate mean of class metrics
                precision.append(report_df.iloc[:-1]['precision'].mean())
                recall.append(report_df.iloc[:-1]['recall'].mean())
                f1_score.append(report_df.iloc[:-1]['f1-score'].mean())

                # Extract accuracy from the last row
                accuracy.append(report_df.iloc[-1]['precision'])

        except Exception as e:
            logger.error(f"Error loading report from {report_path}: {e}")

    # Create a dataframe for plotting
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Accuracy': accuracy
    })

    # Melt the dataframe for easier plotting
    melted_df = pd.melt(metrics_df, id_vars=['Model'],
                        value_vars=['Precision', 'Recall', 'F1 Score', 'Accuracy'],
                        var_name='Metric', value_name='Value')

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create grouped bar chart
    sns.barplot(x='Model', y='Value', hue='Metric', data=melted_df, ax=ax)

    # Configure plot
    ax.set_title('Model Performance Metrics Comparison', fontsize=16)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.legend(title='Metric', fontsize=12)
    ax.set_ylim([0, 1])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout
    fig.tight_layout()

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics comparison plot to {save_path}")

    return fig


def main():
    """
    Main function to parse arguments and run visualization.
    """
    parser = argparse.ArgumentParser(description='Visualize model performance')

    subparsers = parser.add_subparsers(dest='command', help='Visualization command')

    # Single model history visualization
    history_parser = subparsers.add_parser('history', help='Plot training history for a single model')
    history_parser.add_argument('--history-path', type=str, required=True,
                                help='Path to history pickle file')
    history_parser.add_argument('--save-dir', type=str, default=FIGURES_DIR,
                                help='Directory to save plots')

    # Model comparison visualization
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--history-paths', type=str, nargs='+', required=True,
                                help='Paths to history pickle files')
    compare_parser.add_argument('--save-path', type=str,
                                default=os.path.join(FIGURES_DIR, 'model_comparison'),
                                help='Base path for saving comparison plots')

    # Metrics comparison visualization
    metrics_parser = subparsers.add_parser('metrics', help='Compare model metrics')
    metrics_parser.add_argument('--report-paths', type=str, nargs='+', required=True,
                                help='Paths to classification report CSV files')
    metrics_parser.add_argument('--save-path', type=str,
                                default=os.path.join(FIGURES_DIR, 'metrics_comparison.png'),
                                help='Path for saving metrics comparison plot')

    args = parser.parse_args()

    # Process the command
    if args.command == 'history':
        plot_training_history(args.history_path, args.save_dir)

    elif args.command == 'compare':
        compare_models(args.history_paths, args.save_path)

    elif args.command == 'metrics':
        plot_model_metrics_comparison(args.report_paths, args.save_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()