use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use anyhow::Result;

/// Model checkpoint for saving/loading
#[derive(Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub best_val_loss: f32,
    pub metadata: CheckpointMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub model_type: String,
    pub timestamp: String,
    pub hyperparameters: std::collections::HashMap<String, String>,
}

impl Checkpoint {
    pub fn new(epoch: usize, train_loss: f32, val_loss: f32, best_val_loss: f32, model_type: &str) -> Self {
        Self {
            epoch,
            train_loss,
            val_loss,
            best_val_loss,
            metadata: CheckpointMetadata {
                model_type: model_type.to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                hyperparameters: std::collections::HashMap::new(),
            },
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }
}

/// Save model weights to binary format
pub fn save_weights<P: AsRef<Path>>(path: P, weights: &[(String, Array2<f32>)]) -> Result<()> {
    let mut file = File::create(path)?;
    
    // Write number of weight matrices
    let n_weights = weights.len() as u32;
    file.write_all(&n_weights.to_le_bytes())?;
    
    for (name, weight) in weights {
        // Write name length and name
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        file.write_all(&name_len.to_le_bytes())?;
        file.write_all(name_bytes)?;
        
        // Write shape
        let shape = weight.shape();
        file.write_all(&(shape[0] as u32).to_le_bytes())?;
        file.write_all(&(shape[1] as u32).to_le_bytes())?;
        
        // Write data
        let data = weight.as_slice().unwrap();
        for &val in data {
            file.write_all(&val.to_le_bytes())?;
        }
    }
    
    Ok(())
}

/// Load model weights from binary format
pub fn load_weights<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Array2<f32>)>> {
    use std::io::Read;
    
    let mut file = File::open(path)?;
    let mut weights = Vec::new();
    
    // Read number of weight matrices
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let n_weights = u32::from_le_bytes(buf);
    
    for _ in 0..n_weights {
        // Read name
        file.read_exact(&mut buf)?;
        let name_len = u32::from_le_bytes(buf) as usize;
        let mut name_bytes = vec![0u8; name_len];
        file.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)?;
        
        // Read shape
        file.read_exact(&mut buf)?;
        let rows = u32::from_le_bytes(buf) as usize;
        file.read_exact(&mut buf)?;
        let cols = u32::from_le_bytes(buf) as usize;
        
        // Read data
        let mut data = vec![0.0f32; rows * cols];
        for val in data.iter_mut() {
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        
        let weight = Array2::from_shape_vec((rows, cols), data)?;
        weights.push((name, weight));
    }
    
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");
        
        let checkpoint = Checkpoint::new(10, 0.5, 0.6, 0.55, "two_tower");
        checkpoint.save(&path).unwrap();
        
        let loaded = Checkpoint::load(&path).unwrap();
        assert_eq!(loaded.epoch, 10);
        assert!((loaded.train_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_weights_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        
        let weights = vec![
            ("fc1".to_string(), arr2(&[[1.0, 2.0], [3.0, 4.0]])),
            ("fc2".to_string(), arr2(&[[5.0, 6.0], [7.0, 8.0]])),
        ];
        
        save_weights(&path, &weights).unwrap();
        let loaded = load_weights(&path).unwrap();
        
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "fc1");
        assert_eq!(loaded[0].1, arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    }
}
