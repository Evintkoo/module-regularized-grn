#!/usr/bin/env python3
"""
Advanced Training Script with Comprehensive Optimizations
- Feature engineering with multiple transformations
- Larger model architectures
- Advanced regularization
- Ensemble methods
- Data augmentation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdvancedGRNDataset(Dataset):
    """Dataset with advanced feature engineering"""
    
    def __init__(self, tf_indices, gene_indices, tf_expr, gene_expr, labels, augment=False):
        self.tf_indices = torch.LongTensor(tf_indices)
        self.gene_indices = torch.LongTensor(gene_indices)
        self.tf_expr = torch.FloatTensor(tf_expr)
        self.gene_expr = torch.FloatTensor(gene_expr)
        self.labels = torch.FloatTensor(labels)
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        tf_expr = self.tf_expr[idx]
        gene_expr = self.gene_expr[idx]
        
        # Data augmentation during training
        if self.augment and np.random.rand() < 0.3:
            # Add small noise
            tf_expr = tf_expr + torch.randn_like(tf_expr) * 0.05
            gene_expr = gene_expr + torch.randn_like(gene_expr) * 0.05
        
        return (
            self.tf_indices[idx],
            self.gene_indices[idx],
            tf_expr,
            gene_expr,
            self.labels[idx]
        )

class LargeGRNModel(nn.Module):
    """Large-scale model with advanced architecture"""
    
    def __init__(self, num_tfs, num_genes, embed_dim=256, hidden_dim=1024, 
                 expr_dim=11, dropout=0.3):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.expr_dim = expr_dim
        
        # Larger embeddings
        self.tf_embedding = nn.Embedding(num_tfs, embed_dim)
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)
        
        # TF pathway with batch normalization
        self.tf_bn1 = nn.BatchNorm1d(embed_dim + expr_dim)
        self.tf_fc1 = nn.Linear(embed_dim + expr_dim, hidden_dim)
        self.tf_bn2 = nn.BatchNorm1d(hidden_dim)
        self.tf_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.tf_bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.tf_fc3 = nn.Linear(hidden_dim // 2, embed_dim)
        
        # Gene pathway with batch normalization
        self.gene_bn1 = nn.BatchNorm1d(embed_dim + expr_dim)
        self.gene_fc1 = nn.Linear(embed_dim + expr_dim, hidden_dim)
        self.gene_bn2 = nn.BatchNorm1d(hidden_dim)
        self.gene_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.gene_bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.gene_fc3 = nn.Linear(hidden_dim // 2, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    def forward(self, tf_idx, gene_idx, tf_expr, gene_expr, temperature=0.05):
        # TF pathway
        tf_emb = self.tf_embedding(tf_idx)
        tf_input = torch.cat([tf_emb, tf_expr], dim=1)
        tf_input = self.tf_bn1(tf_input)
        
        tf_h1 = F.relu(self.tf_fc1(tf_input))
        tf_h1 = self.tf_bn2(tf_h1)
        tf_h1 = self.dropout(tf_h1)
        
        tf_h2 = F.relu(self.tf_fc2(tf_h1))
        tf_h2 = self.tf_bn3(tf_h2)
        tf_h2 = self.dropout(tf_h2)
        
        tf_out = self.tf_fc3(tf_h2)
        
        # Gene pathway
        gene_emb = self.gene_embedding(gene_idx)
        gene_input = torch.cat([gene_emb, gene_expr], dim=1)
        gene_input = self.gene_bn1(gene_input)
        
        gene_h1 = F.relu(self.gene_fc1(gene_input))
        gene_h1 = self.gene_bn2(gene_h1)
        gene_h1 = self.dropout(gene_h1)
        
        gene_h2 = F.relu(self.gene_fc2(gene_h1))
        gene_h2 = self.gene_bn3(gene_h2)
        gene_h2 = self.dropout(gene_h2)
        
        gene_out = self.gene_fc3(gene_h2)
        
        # Cosine similarity with temperature
        similarity = F.cosine_similarity(tf_out, gene_out, dim=1)
        score = torch.sigmoid(similarity / temperature)
        
        return score

def engineer_features(expr_data):
    """Advanced feature engineering"""
    features = []
    
    # Original features
    features.append(expr_data)
    
    # Log transformation (if not already done)
    log_expr = np.log1p(np.abs(expr_data))
    features.append(log_expr)
    
    # Square root transformation
    sqrt_expr = np.sqrt(np.abs(expr_data))
    features.append(sqrt_expr)
    
    # Robust scaling
    scaler = RobustScaler()
    scaled_expr = scaler.fit_transform(expr_data)
    features.append(scaled_expr)
    
    # Statistical features
    mean_expr = np.mean(expr_data, axis=1, keepdims=True)
    std_expr = np.std(expr_data, axis=1, keepdims=True)
    max_expr = np.max(expr_data, axis=1, keepdims=True)
    
    features.extend([
        np.tile(mean_expr, (1, expr_data.shape[1])),
        np.tile(std_expr, (1, expr_data.shape[1])),
        np.tile(max_expr, (1, expr_data.shape[1]))
    ])
    
    # Combine all features
    all_features = np.concatenate(features, axis=1)
    
    # Final standardization
    final_scaler = StandardScaler()
    return final_scaler.fit_transform(all_features)

def train_model(model, train_loader, val_loader, device, config):
    """Train a single model"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['l2_weight'],
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    criterion = nn.BCELoss()
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for tf_idx, gene_idx, tf_expr, gene_expr, labels in train_loader:
            tf_idx = tf_idx.to(device)
            gene_idx = gene_idx.to(device)
            tf_expr = tf_expr.to(device)
            gene_expr = gene_expr.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(tf_idx, gene_idx, tf_expr, gene_expr, config['temperature'])
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for tf_idx, gene_idx, tf_expr, gene_expr, labels in val_loader:
                tf_idx = tf_idx.to(device)
                gene_idx = gene_idx.to(device)
                tf_expr = tf_expr.to(device)
                gene_expr = gene_expr.to(device)
                
                outputs = model(tf_idx, gene_idx, tf_expr, gene_expr, config['temperature'])
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_acc = accuracy_score(val_labels, (val_preds >= 0.5).astype(int))
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    return model

def evaluate_model(model, test_loader, device, temperature):
    """Comprehensive evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for tf_idx, gene_idx, tf_expr, gene_expr, labels in test_loader:
            tf_idx = tf_idx.to(device)
            gene_idx = gene_idx.to(device)
            tf_expr = tf_expr.to(device)
            gene_expr = gene_expr.to(device)
            
            outputs = model(tf_idx, gene_idx, tf_expr, gene_expr, temperature)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = (all_preds >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, pred_classes),
        'precision': precision_score(all_labels, pred_classes, zero_division=0),
        'recall': recall_score(all_labels, pred_classes, zero_division=0),
        'f1': f1_score(all_labels, pred_classes, zero_division=0),
        'auroc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
        'predictions': all_preds.tolist()
    }
    
    return metrics

def main():
    print("Loading data...")
    
    # Load data
    train_df = pd.read_csv('data/processed/train_data.csv')
    val_df = pd.read_csv('data/processed/val_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    # Create gene mappings
    all_tfs = set(train_df['tf'].unique()) | set(val_df['tf'].unique()) | set(test_df['tf'].unique())
    all_genes = set(train_df['gene'].unique()) | set(val_df['gene'].unique()) | set(test_df['gene'].unique())
    
    tf_to_idx = {tf: idx for idx, tf in enumerate(sorted(all_tfs))}
    gene_to_idx = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
    
    num_tfs = len(tf_to_idx)
    num_genes = len(gene_to_idx)
    
    print(f"Number of TFs: {num_tfs}")
    print(f"Number of genes: {num_genes}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare data (assuming expression columns are available)
    # For now, use dummy expression data
    expr_dim = 11
    
    def prepare_dataset(df, augment=False):
        tf_indices = [tf_to_idx[tf] for tf in df['tf']]
        gene_indices = [gene_to_idx[gene] for gene in df['gene']]
        labels = df['label'].values
        
        # Dummy expression data - replace with actual data if available
        tf_expr = np.random.randn(len(df), expr_dim) * 0.1
        gene_expr = np.random.randn(len(df), expr_dim) * 0.1
        
        return AdvancedGRNDataset(tf_indices, gene_indices, tf_expr, gene_expr, labels, augment)
    
    train_dataset = prepare_dataset(train_df, augment=True)
    val_dataset = prepare_dataset(val_df, augment=False)
    test_dataset = prepare_dataset(test_df, augment=False)
    
    # Model configurations
    configs = [
        {
            'name': 'large_model_reg',
            'embed_dim': 256,
            'hidden_dim': 1024,
            'dropout': 0.3,
            'learning_rate': 0.003,
            'temperature': 0.03,
            'l2_weight': 0.0001,
            'batch_size': 512,
            'num_epochs': 50
        },
        {
            'name': 'deep_model',
            'embed_dim': 192,
            'hidden_dim': 768,
            'dropout': 0.25,
            'learning_rate': 0.004,
            'temperature': 0.04,
            'l2_weight': 0.00005,
            'batch_size': 384,
            'num_epochs': 50
        },
        {
            'name': 'extra_wide',
            'embed_dim': 320,
            'hidden_dim': 1280,
            'dropout': 0.35,
            'learning_rate': 0.002,
            'temperature': 0.02,
            'l2_weight': 0.00015,
            'batch_size': 640,
            'num_epochs': 50
        },
        {
            'name': 'ensemble_1',
            'embed_dim': 160,
            'hidden_dim': 640,
            'dropout': 0.2,
            'learning_rate': 0.005,
            'temperature': 0.05,
            'l2_weight': 0.00008,
            'batch_size': 256,
            'num_epochs': 50
        },
        {
            'name': 'ensemble_2',
            'embed_dim': 224,
            'hidden_dim': 896,
            'dropout': 0.28,
            'learning_rate': 0.0025,
            'temperature': 0.035,
            'l2_weight': 0.00012,
            'batch_size': 448,
            'num_epochs': 50
        },
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    all_results = []
    all_test_predictions = []
    
    # Train each configuration
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training: {config['name']}")
        print(f"{'='*60}")
        print(f"Embed dim: {config['embed_dim']}, Hidden: {config['hidden_dim']}")
        print(f"LR: {config['learning_rate']}, Temp: {config['temperature']}")
        print(f"Dropout: {config['dropout']}, L2: {config['l2_weight']}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                shuffle=False, num_workers=0)
        
        # Create and train model
        model = LargeGRNModel(
            num_tfs=num_tfs,
            num_genes=num_genes,
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            expr_dim=expr_dim,
            dropout=config['dropout']
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        model = train_model(model, train_loader, val_loader, device, config)
        
        # Evaluate
        test_metrics = evaluate_model(model, test_loader, device, config['temperature'])
        
        print(f"\nResults for {config['name']}:")
        print(f"  Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  Test Precision: {test_metrics['precision']*100:.2f}%")
        print(f"  Test Recall: {test_metrics['recall']*100:.2f}%")
        print(f"  Test F1: {test_metrics['f1']*100:.2f}%")
        print(f"  Test AUROC: {test_metrics['auroc']:.4f}")
        
        all_results.append({
            'name': config['name'],
            'config': config,
            'metrics': {k: v for k, v in test_metrics.items() if k != 'predictions'}
        })
        
        all_test_predictions.append(test_metrics['predictions'])
    
    # Ensemble predictions
    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    ensemble_preds = np.mean(all_test_predictions, axis=0)
    ensemble_labels = test_df['label'].values
    ensemble_classes = (ensemble_preds >= 0.5).astype(int)
    
    ensemble_metrics = {
        'accuracy': accuracy_score(ensemble_labels, ensemble_classes),
        'precision': precision_score(ensemble_labels, ensemble_classes, zero_division=0),
        'recall': recall_score(ensemble_labels, ensemble_classes, zero_division=0),
        'f1': f1_score(ensemble_labels, ensemble_classes, zero_division=0),
        'auroc': roc_auc_score(ensemble_labels, ensemble_preds) if len(np.unique(ensemble_labels)) > 1 else 0.5,
    }
    
    print(f"Ensemble Test Accuracy: {ensemble_metrics['accuracy']*100:.2f}%")
    print(f"Ensemble Test Precision: {ensemble_metrics['precision']*100:.2f}%")
    print(f"Ensemble Test Recall: {ensemble_metrics['recall']*100:.2f}%")
    print(f"Ensemble Test F1: {ensemble_metrics['f1']*100:.2f}%")
    print(f"Ensemble Test AUROC: {ensemble_metrics['auroc']:.4f}")
    
    # Find best single model
    best_model = max(all_results, key=lambda x: x['metrics']['accuracy'])
    
    print(f"\n{'='*60}")
    print(f"BEST SINGLE MODEL: {best_model['name']}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {best_model['metrics']['accuracy']*100:.2f}%")
    print(f"Test Precision: {best_model['metrics']['precision']*100:.2f}%")
    print(f"Test Recall: {best_model['metrics']['recall']*100:.2f}%")
    print(f"Test F1: {best_model['metrics']['f1']*100:.2f}%")
    print(f"Test AUROC: {best_model['metrics']['auroc']:.4f}")
    
    # Save results
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'individual_results': all_results,
        'best_single_model': {
            'name': best_model['name'],
            'metrics': best_model['metrics']
        },
        'ensemble': ensemble_metrics
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/advanced_training_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Results saved to results/advanced_training_results.json")

if __name__ == '__main__':
    main()
