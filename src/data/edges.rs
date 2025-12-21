use super::types::{CandidateEdgeSet, Edge, PriorKnowledge, StateId};
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};

pub struct CandidateEdgeBuilder {
    correlation_threshold: f32,
    negatives_per_positive: usize,
}

impl CandidateEdgeBuilder {
    pub fn new(correlation_threshold: f32, negatives_per_positive: usize) -> Self {
        Self {
            correlation_threshold,
            negatives_per_positive,
        }
    }
    
    pub fn build_candidates(
        &self,
        state: &StateId,
        priors: &PriorKnowledge,
        expression: &[f32],
        tf_indices: &HashMap<String, usize>,
        gene_indices: &HashMap<String, usize>,
        rng: &mut impl Rng,
    ) -> Result<CandidateEdgeSet> {
        let mut positive_edges = Vec::new();
        
        for (tf, targets) in &priors.tf_target_pairs {
            for target in targets {
                if tf_indices.contains_key(tf) && gene_indices.contains_key(target) {
                    positive_edges.push((tf.clone(), target.clone()));
                }
            }
        }
        
        let all_tfs: Vec<_> = tf_indices.keys().cloned().collect();
        let all_genes: Vec<_> = gene_indices.keys().cloned().collect();
        let positive_set: HashSet<_> = positive_edges.iter().cloned().collect();
        
        let mut negative_edges = Vec::new();
        let target_negatives = positive_edges.len() * self.negatives_per_positive;
        
        while negative_edges.len() < target_negatives {
            let tf = all_tfs.choose(rng).unwrap().clone();
            let gene = all_genes.choose(rng).unwrap().clone();
            let pair = (tf.clone(), gene.clone());
            
            if !positive_set.contains(&pair) && !negative_edges.contains(&pair) {
                negative_edges.push(pair);
            }
        }
        
        Ok(CandidateEdgeSet {
            state: state.clone(),
            positive_edges,
            negative_edges,
        })
    }
    
    pub fn expand_with_correlation(
        &self,
        candidates: &mut CandidateEdgeSet,
        expression: &[f32],
        tf_indices: &HashMap<String, usize>,
        gene_indices: &HashMap<String, usize>,
    ) -> Result<()> {
        let mut new_positives = Vec::new();
        
        for (tf, _) in &candidates.positive_edges {
            if let Some(&tf_idx) = tf_indices.get(tf) {
                let tf_expr = expression[tf_idx];
                
                for (gene, &gene_idx) in gene_indices {
                    let gene_expr = expression[gene_idx];
                    let corr = (tf_expr * gene_expr).abs();
                    
                    if corr > self.correlation_threshold {
                        let pair = (tf.clone(), gene.clone());
                        if !candidates.positive_edges.contains(&pair) {
                            new_positives.push(pair);
                        }
                    }
                }
            }
        }
        
        candidates.positive_edges.extend(new_positives);
        Ok(())
    }
}

pub struct EdgeSampler {
    pub k_negatives: usize,
}

impl EdgeSampler {
    pub fn new(k_negatives: usize) -> Self {
        Self { k_negatives }
    }
    
    pub fn sample_batch(
        &self,
        candidates: &CandidateEdgeSet,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> Vec<Edge> {
        let mut edges = Vec::new();
        
        let positive_sample = candidates
            .positive_edges
            .choose_multiple(rng, batch_size.min(candidates.positive_edges.len()))
            .cloned()
            .collect::<Vec<_>>();
        
        for (tf, target) in positive_sample {
            edges.push(Edge {
                tf: tf.clone(),
                target: target.clone(),
                state: candidates.state.clone(),
                is_positive: true,
            });
            
            let negatives = candidates
                .negative_edges
                .choose_multiple(rng, self.k_negatives)
                .cloned();
            
            for (neg_tf, neg_target) in negatives {
                edges.push(Edge {
                    tf: neg_tf,
                    target: neg_target,
                    state: candidates.state.clone(),
                    is_positive: false,
                });
            }
        }
        
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    
    #[test]
    fn test_candidate_edge_building() {
        let mut rng = StdRng::seed_from_u64(42);
        let builder = CandidateEdgeBuilder::new(0.5, 5);
        
        let mut priors = PriorKnowledge {
            tf_target_pairs: HashMap::new(),
            source: "test".to_string(),
        };
        priors.tf_target_pairs.insert("TF1".to_string(), vec!["Gene1".to_string()]);
        
        let state = StateId::new("diss1".to_string(), "cluster1".to_string());
        let expression = vec![1.0, 2.0];
        
        let mut tf_indices = HashMap::new();
        tf_indices.insert("TF1".to_string(), 0);
        
        let mut gene_indices = HashMap::new();
        gene_indices.insert("Gene1".to_string(), 1);
        
        let candidates = builder
            .build_candidates(&state, &priors, &expression, &tf_indices, &gene_indices, &mut rng)
            .unwrap();
        
        assert_eq!(candidates.positive_edges.len(), 1);
        assert_eq!(candidates.negative_edges.len(), 5);
    }
}
