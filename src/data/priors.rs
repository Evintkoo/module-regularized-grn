/// Prior knowledge loader for GRN training
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use anyhow::Result;

/// Prior knowledge network: TF -> [target genes]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorKnowledge {
    #[serde(flatten)]
    pub tf_to_genes: HashMap<String, Vec<String>>,
}

impl PriorKnowledge {
    /// Load from JSON file
    pub fn from_file(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let tf_to_genes: HashMap<String, Vec<String>> = serde_json::from_reader(reader)?;
        Ok(Self { tf_to_genes })
    }
    
    /// Get all unique TFs
    pub fn get_tfs(&self) -> Vec<String> {
        self.tf_to_genes.keys().cloned().collect()
    }
    
    /// Get all unique target genes
    pub fn get_genes(&self) -> Vec<String> {
        let mut genes = std::collections::HashSet::new();
        for targets in self.tf_to_genes.values() {
            for gene in targets {
                genes.insert(gene.clone());
            }
        }
        genes.into_iter().collect()
    }
    
    /// Get all TF-Gene edges
    pub fn get_edges(&self) -> Vec<(String, String)> {
        let mut edges = Vec::new();
        for (tf, targets) in &self.tf_to_genes {
            for gene in targets {
                edges.push((tf.clone(), gene.clone()));
            }
        }
        edges
    }
    
    /// Count statistics
    pub fn stats(&self) -> PriorStats {
        let num_tfs = self.tf_to_genes.len();
        let edges = self.get_edges();
        let num_edges = edges.len();
        let num_genes = self.get_genes().len();
        let avg_targets = num_edges as f32 / num_tfs as f32;
        
        PriorStats {
            num_tfs,
            num_genes,
            num_edges,
            avg_targets_per_tf: avg_targets,
        }
    }
    
    /// Check if TF->Gene edge exists
    pub fn has_edge(&self, tf: &str, gene: &str) -> bool {
        if let Some(targets) = self.tf_to_genes.get(tf) {
            targets.contains(&gene.to_string())
        } else {
            false
        }
    }
    
    /// Create TF and Gene vocabularies with indices
    pub fn create_vocabularies(&self) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let tfs = self.get_tfs();
        let genes = self.get_genes();
        
        let tf_vocab: HashMap<String, usize> = tfs.into_iter()
            .enumerate()
            .map(|(i, tf)| (tf, i))
            .collect();
        
        let gene_vocab: HashMap<String, usize> = genes.into_iter()
            .enumerate()
            .map(|(i, gene)| (gene, i))
            .collect();
        
        (tf_vocab, gene_vocab)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorStats {
    pub num_tfs: usize,
    pub num_genes: usize,
    pub num_edges: usize,
    pub avg_targets_per_tf: f32,
}

/// Dataset builder from prior knowledge
pub struct PriorDatasetBuilder {
    priors: PriorKnowledge,
    tf_vocab: HashMap<String, usize>,
    gene_vocab: HashMap<String, usize>,
}

impl PriorDatasetBuilder {
    pub fn new(priors: PriorKnowledge) -> Self {
        let (tf_vocab, gene_vocab) = priors.create_vocabularies();
        Self {
            priors,
            tf_vocab,
            gene_vocab,
        }
    }
    
    /// Get positive examples (edges that exist in priors)
    pub fn get_positive_examples(&self) -> Vec<(usize, usize)> {
        let edges = self.priors.get_edges();
        edges.into_iter()
            .filter_map(|(tf, gene)| {
                let tf_idx = self.tf_vocab.get(&tf)?;
                let gene_idx = self.gene_vocab.get(&gene)?;
                Some((*tf_idx, *gene_idx))
            })
            .collect()
    }
    
    /// Sample negative examples (edges that don't exist in priors)
    pub fn sample_negative_examples(&self, n_samples: usize, seed: u64) -> Vec<(usize, usize)> {
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(seed);
        let num_tfs = self.tf_vocab.len();
        let num_genes = self.gene_vocab.len();
        
        let mut negatives = Vec::new();
        let mut attempts = 0;
        let max_attempts = n_samples * 100; // Prevent infinite loop
        
        while negatives.len() < n_samples && attempts < max_attempts {
            let tf_idx = rng.gen_range(0..num_tfs);
            let gene_idx = rng.gen_range(0..num_genes);
            
            // Check if this edge exists in priors
            let tf_name = self.get_tf_name(tf_idx);
            let gene_name = self.get_gene_name(gene_idx);
            
            if let (Some(tf), Some(gene)) = (tf_name, gene_name) {
                if !self.priors.has_edge(&tf, &gene) {
                    negatives.push((tf_idx, gene_idx));
                }
            }
            
            attempts += 1;
        }
        
        negatives
    }
    
    /// Get TF name from index
    fn get_tf_name(&self, idx: usize) -> Option<String> {
        self.tf_vocab.iter()
            .find(|(_, &i)| i == idx)
            .map(|(name, _)| name.clone())
    }
    
    /// Get gene name from index
    fn get_gene_name(&self, idx: usize) -> Option<String> {
        self.gene_vocab.iter()
            .find(|(_, &i)| i == idx)
            .map(|(name, _)| name.clone())
    }
    
    pub fn num_tfs(&self) -> usize {
        self.tf_vocab.len()
    }
    
    pub fn num_genes(&self) -> usize {
        self.gene_vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_priors() {
        let priors = PriorKnowledge::from_file("data/priors/merged_priors.json");
        assert!(priors.is_ok());
        
        let priors = priors.unwrap();
        let stats = priors.stats();
        
        println!("Stats: {:?}", stats);
        assert!(stats.num_tfs > 0);
        assert!(stats.num_edges > 0);
    }
    
    #[test]
    fn test_vocabularies() {
        let priors = PriorKnowledge::from_file("data/priors/merged_priors.json").unwrap();
        let (tf_vocab, gene_vocab) = priors.create_vocabularies();
        
        assert!(tf_vocab.len() > 0);
        assert!(gene_vocab.len() > 0);
    }
    
    #[test]
    fn test_dataset_builder() {
        let priors = PriorKnowledge::from_file("data/priors/merged_priors.json").unwrap();
        let builder = PriorDatasetBuilder::new(priors);
        
        let positives = builder.get_positive_examples();
        assert!(positives.len() > 0);
        
        let negatives = builder.sample_negative_examples(1000, 42);
        assert_eq!(negatives.len(), 1000);
    }
}
