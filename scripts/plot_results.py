#!/usr/bin/env python3
"""
Generate publication-quality plots for GRN evaluation results
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_results():
    """Load evaluation results"""
    with open('results/predictions.json', 'r') as f:
        return json.load(f)

def plot_roc_curve(predictions, labels, metrics):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/roc_curve.png")
    plt.close()

def plot_pr_curve(predictions, labels, metrics):
    """Plot Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, auc
    
    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline (proportion of positives)
    baseline = np.mean(labels)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
             label=f'Random (AUC = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/pr_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/pr_curve.png")
    plt.close()

def plot_confusion_matrix(metrics):
    """Plot confusion matrix"""
    cm = np.array([
        [metrics['true_positives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_negatives']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'],
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 16})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/confusion_matrix.png")
    plt.close()

def plot_metrics_comparison():
    """Plot comparison of different metrics"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'AUPRC']
    
    # Load metrics
    with open('results/evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['auroc'],
        metrics['auprc']
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.axhline(y=0.70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='70% Target')
    plt.axhline(y=0.80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='80% Excellent')
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/metrics_comparison.png")
    plt.close()

def plot_score_distribution(predictions, labels):
    """Plot distribution of prediction scores"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    pos_scores = predictions[labels == 1]
    neg_scores = predictions[labels == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Negative (No Edge)', color='red', density=True)
    plt.hist(pos_scores, bins=50, alpha=0.5, label='Positive (True Edge)', color='green', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    plt.xlabel('Prediction Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Prediction Scores', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/score_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/score_distribution.png")
    plt.close()

def main():
    print("=== Generating Evaluation Plots ===\n")
    
    # Check if sklearn is available
    try:
        import sklearn
    except ImportError:
        print("⚠️  Warning: scikit-learn not installed. Installing...")
        import subprocess
        subprocess.run(['pip3', 'install', 'scikit-learn'], check=True)
        import sklearn
    
    # Load results
    results = load_results()
    predictions = results['predictions']
    labels = results['labels']
    metrics = results['metrics']
    
    print(f"Loaded {len(predictions)} predictions\n")
    
    # Generate plots
    print("Generating plots...")
    plot_roc_curve(predictions, labels, metrics)
    plot_pr_curve(predictions, labels, metrics)
    plot_confusion_matrix(metrics)
    plot_metrics_comparison()
    plot_score_distribution(predictions, labels)
    
    print("\n✅ All plots generated successfully!")
    print("📊 Results saved to results/ directory")
    print("\nFiles created:")
    print("  - roc_curve.png")
    print("  - pr_curve.png")
    print("  - confusion_matrix.png")
    print("  - metrics_comparison.png")
    print("  - score_distribution.png")

if __name__ == '__main__':
    main()
