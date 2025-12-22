use module_regularized_grn::{Config, Trainer};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== GRN Training Pipeline ===\n");

    // Load configuration from config.toml
    let config = Config::load_default()?;
    
    println!("✓ Loaded configuration from config.toml\n");
    
    println!("Configuration:");
    println!("  Epochs: {}", config.training.num_epochs);
    println!("  Batch size: {}", config.training.batch_size);
    println!("  Learning rate: {}", config.training.learning_rate);
    println!("  Early stopping patience: {}", config.training.early_stopping_patience);
    println!();

    // Create trainer
    let mut trainer = Trainer::new(config.clone());

    println!("Starting training...\n");

    // Training loop (simplified - no actual model yet)
    for epoch in 0..config.training.num_epochs {
        // Placeholder training
        let train_loss = trainer.train_epoch_placeholder();
        let (val_loss, metrics) = trainer.validate_epoch_placeholder();

        // Update history
        trainer.history.add_epoch(train_loss, val_loss, metrics.clone());

        // Print progress
        if epoch % config.training.save_every_n_epochs == 0 {
            trainer.print_epoch(epoch, train_loss, val_loss, &metrics);
        }

        // Save checkpoint
        if epoch % config.training.save_every_n_epochs == 0 {
            trainer.save_checkpoint(epoch, train_loss, val_loss, "two_tower")?;
        }

        // Check early stopping
        if trainer.should_stop(epoch) {
            println!("\nEarly stopping triggered at epoch {}", epoch);
            break;
        }
    }

    println!("\n=== Training Complete ===");
    println!("Best validation loss: {:.4}", trainer.history.best_val_loss());
    println!("Best epoch: {}", trainer.history.best_epoch());

    Ok(())
}
