use super::types::{CandidateEdgeSet, Edge};
use super::edges::EdgeSampler;
use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;

pub struct DataSplit {
    pub train_states: Vec<String>,
    pub val_states: Vec<String>,
    pub test_states: Vec<String>,
}

impl DataSplit {
    pub fn new(
        all_states: &[String],
        train_ratio: f64,
        val_ratio: f64,
        rng: &mut impl Rng,
    ) -> Self {
        use rand::seq::SliceRandom;
        
        let mut shuffled = all_states.to_vec();
        shuffled.shuffle(rng);
        
        let train_end = (all_states.len() as f64 * train_ratio) as usize;
        let val_end = train_end + (all_states.len() as f64 * val_ratio) as usize;
        
        Self {
            train_states: shuffled[..train_end].to_vec(),
            val_states: shuffled[train_end..val_end].to_vec(),
            test_states: shuffled[val_end..].to_vec(),
        }
    }
}

pub struct EdgeDataset {
    candidates: HashMap<String, CandidateEdgeSet>,
    state_ids: Vec<String>,
    sampler: EdgeSampler,
}

impl EdgeDataset {
    pub fn new(candidates: HashMap<String, CandidateEdgeSet>, sampler: EdgeSampler) -> Self {
        let state_ids = candidates.keys().cloned().collect();
        Self {
            candidates,
            state_ids,
            sampler,
        }
    }
    
    pub fn len(&self) -> usize {
        self.state_ids.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.state_ids.is_empty()
    }
    
    pub fn get_batch(
        &self,
        state_id: &str,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<Edge>> {
        let candidates = self.candidates.get(state_id)
            .ok_or_else(|| anyhow::anyhow!("State not found: {}", state_id))?;
        Ok(self.sampler.sample_batch(candidates, batch_size, rng))
    }
}

pub struct DataLoader {
    dataset: EdgeDataset,
    batch_size: usize,
    shuffle: bool,
}

impl DataLoader {
    pub fn new(dataset: EdgeDataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
        }
    }
    
    pub fn iter<'a, R: Rng>(
        &'a self,
        rng: &'a mut R,
    ) -> DataLoaderIter<'a, R> {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        
        if self.shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(rng);
        }
        
        DataLoaderIter {
            loader: self,
            indices,
            current: 0,
            rng,
        }
    }
}

pub struct DataLoaderIter<'a, R: Rng> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    current: usize,
    rng: &'a mut R,
}

impl<'a, R: Rng> Iterator for DataLoaderIter<'a, R> {
    type Item = Result<Vec<Edge>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }
        
        let idx = self.indices[self.current];
        let state_id = &self.loader.dataset.state_ids[idx];
        self.current += 1;
        
        Some(self.loader.dataset.get_batch(state_id, self.loader.batch_size, self.rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::types::StateId;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    
    #[test]
    fn test_data_split() {
        let mut rng = StdRng::seed_from_u64(42);
        let states = vec!["s1".to_string(), "s2".to_string(), "s3".to_string(), "s4".to_string()];
        
        let split = DataSplit::new(&states, 0.7, 0.15, &mut rng);
        
        assert_eq!(split.train_states.len(), 2);
        assert_eq!(split.val_states.len(), 0);
        assert_eq!(split.test_states.len(), 2);
    }
}
