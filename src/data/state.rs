use super::types::{State, StateId, StateManifest};
use anyhow::{Result, Context};
use std::collections::HashMap;

pub struct StatePartitioner {
    min_cell_count: usize,
}

impl StatePartitioner {
    pub fn new(min_cell_count: usize) -> Self {
        Self { min_cell_count }
    }
    
    pub fn create_states(
        &self,
        dissections: &[String],
        superclusters: &[String],
        cell_counts: &HashMap<(String, String), usize>,
        pseudobulk_data: &HashMap<(String, String), Vec<f32>>,
        gene_ids: &[String],
    ) -> Result<StateManifest> {
        let mut states = Vec::new();
        
        for dissection in dissections {
            for supercluster in superclusters {
                let key = (dissection.clone(), supercluster.clone());
                
                if let Some(&count) = cell_counts.get(&key) {
                    if count >= self.min_cell_count {
                        let expression = pseudobulk_data
                            .get(&key)
                            .context("Missing pseudobulk data")?
                            .clone();
                        
                        if !expression.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                            states.push(State {
                                id: StateId::new(dissection.clone(), supercluster.clone()),
                                cell_count: count,
                                pseudobulk_expression: expression,
                                gene_ids: gene_ids.to_vec(),
                            });
                        }
                    }
                }
            }
        }
        
        let min = states.iter().map(|s| s.cell_count).min().unwrap_or(0);
        let max = states.iter().map(|s| s.cell_count).max().unwrap_or(0);
        
        Ok(StateManifest {
            total_states: states.len(),
            min_cell_count: min,
            max_cell_count: max,
            states,
        })
    }
    
    pub fn validate_manifest(&self, manifest: &StateManifest) -> Result<()> {
        anyhow::ensure!(
            manifest.total_states >= 50,
            "Insufficient states: {} < 50",
            manifest.total_states
        );
        
        for state in &manifest.states {
            anyhow::ensure!(
                state.cell_count >= self.min_cell_count,
                "State {} has too few cells: {}",
                state.id.to_string(),
                state.cell_count
            );
            
            anyhow::ensure!(
                !state.pseudobulk_expression.iter().any(|&x| x.is_nan() || x.is_infinite()),
                "State {} contains invalid values",
                state.id.to_string()
            );
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_creation() {
        let partitioner = StatePartitioner::new(50);
        let dissections = vec!["brain1".to_string(), "brain2".to_string()];
        let superclusters = vec!["cluster1".to_string(), "cluster2".to_string()];
        let mut cell_counts = HashMap::new();
        let mut pseudobulk_data = HashMap::new();
        let gene_ids = vec!["gene1".to_string(), "gene2".to_string()];
        
        for d in &dissections {
            for s in &superclusters {
                cell_counts.insert((d.clone(), s.clone()), 100);
                pseudobulk_data.insert((d.clone(), s.clone()), vec![1.0, 2.0]);
            }
        }
        
        let manifest = partitioner
            .create_states(&dissections, &superclusters, &cell_counts, &pseudobulk_data, &gene_ids)
            .unwrap();
        
        assert_eq!(manifest.total_states, 4);
    }
}
