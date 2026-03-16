#!/usr/bin/env python3
"""
Train the attention-enhanced expression model
"""
import subprocess
import json
import time

def run_rust_training():
    """Run the Rust training with attention model"""
    
    configs = [
        {
            "name": "attention_base",
            "hidden_dim1": 512,
            "hidden_dim2": 256,
            "output_dim": 128,
            "num_heads": 4,
            "learning_rate": 0.001,
            "epochs": 50,
        },
        {
            "name": "attention_large",
            "hidden_dim1": 768,
            "hidden_dim2": 384,
            "output_dim": 192,
            "num_heads": 8,
            "learning_rate": 0.0005,
            "epochs": 50,
        },
        {
            "name": "attention_deep",
            "hidden_dim1": 1024,
            "hidden_dim2": 512,
            "output_dim": 256,
            "num_heads": 8,
            "learning_rate": 0.0003,
            "epochs": 50,
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Training: {config['name']}")
        print(f"{'='*80}")
        print(f"Architecture: {config['hidden_dim1']} -> {config['hidden_dim2']} -> {config['output_dim']}")
        print(f"Attention heads: {config['num_heads']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Epochs: {config['epochs']}")
        
        start_time = time.time()
        
        # Build command
        cmd = [
            "cargo", "run", "--release", "--",
            "train-attention",
            "--hidden1", str(config['hidden_dim1']),
            "--hidden2", str(config['hidden_dim2']),
            "--output-dim", str(config['output_dim']),
            "--num-heads", str(config['num_heads']),
            "--learning-rate", str(config['learning_rate']),
            "--epochs", str(config['epochs']),
            "--batch-size", "256",
        ]
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        elapsed = time.time() - start_time
        
        print(f"\n{result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        print(f"\nTime elapsed: {elapsed:.2f}s")
        
        # Parse results from output
        output_lines = result.stdout.split('\n')
        metrics = {}
        for line in output_lines:
            if 'Accuracy:' in line:
                metrics['accuracy'] = float(line.split(':')[1].strip().rstrip('%'))
            elif 'AUROC:' in line:
                metrics['auroc'] = float(line.split(':')[1].strip())
            elif 'AUPRC:' in line:
                metrics['auprc'] = float(line.split(':')[1].strip())
        
        results.append({
            'config': config,
            'metrics': metrics,
            'time': elapsed
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    for r in results:
        print(f"{r['config']['name']}:")
        print(f"  Accuracy: {r['metrics'].get('accuracy', 'N/A')}%")
        print(f"  AUROC: {r['metrics'].get('auroc', 'N/A')}")
        print(f"  AUPRC: {r['metrics'].get('auprc', 'N/A')}")
        print(f"  Time: {r['time']:.2f}s")
        print()
    
    # Find best model
    best = max(results, key=lambda x: x['metrics'].get('accuracy', 0))
    print(f"Best model: {best['config']['name']}")
    print(f"  Accuracy: {best['metrics'].get('accuracy')}%")
    print(f"  AUROC: {best['metrics'].get('auroc')}")
    print(f"  AUPRC: {best['metrics'].get('auprc')}")
    
    # Save results
    with open('results/attention_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/attention_model_results.json")

if __name__ == "__main__":
    run_rust_training()
