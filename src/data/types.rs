use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StateId {
    pub dissection: String,
    pub supercluster: String,
}

impl StateId {
    pub fn new(dissection: String, supercluster: String) -> Self {
        Self { dissection, supercluster }
    }
    
    pub fn to_string(&self) -> String {
        format!("{}_{}", self.dissection, self.supercluster)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub id: StateId,
    pub cell_count: usize,
    pub pseudobulk_expression: Vec<f32>,
    pub gene_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub tf: String,
    pub target: String,
    pub state: StateId,
    pub is_positive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEdgeSet {
    pub state: StateId,
    pub positive_edges: Vec<(String, String)>,
    pub negative_edges: Vec<(String, String)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StateManifest {
    pub states: Vec<State>,
    pub total_states: usize,
    pub min_cell_count: usize,
    pub max_cell_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorKnowledge {
    pub tf_target_pairs: HashMap<String, Vec<String>>,
    pub source: String,
}
