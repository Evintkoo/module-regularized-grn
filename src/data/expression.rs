/// Expression data loader for H5AD-processed data
use ndarray::{Array2};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

/// Expression data for a dataset
#[derive(Debug, Clone)]
pub struct ExpressionData {
    /// Expression matrix: [n_cell_types, n_genes]
    pub expression: Array2<f32>,
    /// Cell type labels
    pub cell_type_labels: Vec<String>,
    /// Gene names (ENSEMBL IDs)
    pub gene_names: Vec<String>,
    /// Number of cell types
    pub n_cell_types: usize,
    /// Number of genes
    pub n_genes: usize,
}

impl ExpressionData {
    /// Load from processed H5AD directory
    pub fn from_processed_dir(dir_path: &str) -> Result<Self> {
        let path = Path::new(dir_path);
        
        // Load expression matrix (.npy file)
        let expr_path = path.join("pseudobulk_expression.npy");
        let expr_file = File::open(&expr_path)
            .context(format!("Failed to open expression file: {:?}", expr_path))?;
        let reader = BufReader::new(expr_file);
        let expression: Array2<f32> = Array2::read_npy(reader)
            .context("Failed to read NPY file")?;
        
        // Load cell type labels (.json file)
        let labels_path = path.join("pseudobulk_labels.json");
        let labels_file = File::open(&labels_path)
            .context(format!("Failed to open labels file: {:?}", labels_path))?;
        let cell_type_labels: Vec<String> = serde_json::from_reader(labels_file)
            .context("Failed to parse labels JSON")?;
        
        // Load gene names (.txt file)
        let genes_path = path.join("genes.txt");
        let gene_names = std::fs::read_to_string(&genes_path)
            .context(format!("Failed to read genes file: {:?}", genes_path))?
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        
        let n_cell_types = expression.nrows();
        let n_genes = expression.ncols();
        
        // Validate dimensions
        if cell_type_labels.len() != n_cell_types {
            anyhow::bail!(
                "Mismatch: {} cell type labels but {} rows in expression matrix",
                cell_type_labels.len(),
                n_cell_types
            );
        }
        
        if gene_names.len() != n_genes {
            anyhow::bail!(
                "Mismatch: {} gene names but {} columns in expression matrix",
                gene_names.len(),
                n_genes
            );
        }
        
        Ok(Self {
            expression,
            cell_type_labels,
            gene_names,
            n_cell_types,
            n_genes,
        })
    }
    
    /// Get expression for a specific cell type
    pub fn get_cell_type_expression(&self, cell_type_idx: usize) -> Option<ndarray::ArrayView1<f32>> {
        if cell_type_idx < self.n_cell_types {
            Some(self.expression.row(cell_type_idx))
        } else {
            None
        }
    }
    
    /// Get expression for a specific gene across all cell types
    pub fn get_gene_expression(&self, gene_idx: usize) -> Option<ndarray::ArrayView1<f32>> {
        if gene_idx < self.n_genes {
            Some(self.expression.column(gene_idx))
        } else {
            None
        }
    }
    
    /// Get cell type index by name
    pub fn get_cell_type_index(&self, name: &str) -> Option<usize> {
        self.cell_type_labels.iter().position(|ct| ct == name)
    }
    
    /// Get gene index by name
    pub fn get_gene_index(&self, name: &str) -> Option<usize> {
        self.gene_names.iter().position(|g| g == name)
    }
    
    /// Get statistics
    pub fn stats(&self) -> ExpressionStats {
        let data = self.expression.as_slice().unwrap();
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        
        ExpressionStats {
            n_cell_types: self.n_cell_types,
            n_genes: self.n_genes,
            min,
            max,
            mean,
            median,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionStats {
    pub n_cell_types: usize,
    pub n_genes: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_expression_data() {
        // This test requires processed H5AD data
        let result = ExpressionData::from_processed_dir(
            "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
        );
        
        if result.is_ok() {
            let expr_data = result.unwrap();
            println!("Expression data loaded:");
            println!("  Cell types: {}", expr_data.n_cell_types);
            println!("  Genes: {}", expr_data.n_genes);
            
            let stats = expr_data.stats();
            println!("Stats: {:?}", stats);
            
            assert!(expr_data.n_cell_types > 0);
            assert!(expr_data.n_genes > 0);
            assert_eq!(expr_data.cell_type_labels.len(), expr_data.n_cell_types);
            assert_eq!(expr_data.gene_names.len(), expr_data.n_genes);
        } else {
            println!("Skipping test - processed data not available");
        }
    }
}
