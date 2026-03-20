/// Bilinear Two-Tower model: same architecture as HybridEmbeddingModel
/// but replaces cosine similarity scoring with a learnable bilinear form:
///   score = sigmoid(e_TF^T W e_gene)
/// where W is a (output_dim x output_dim) matrix.
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

#[derive(Clone)]
pub struct BilinearTwoTowerModel {
    // TF encoder
    pub tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    pub tf_fc1: LinearLayer,
    pub tf_fc2: LinearLayer,

    // Gene encoder
    pub gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,
    pub gene_fc1: LinearLayer,
    pub gene_fc2: LinearLayer,

    // Bilinear scoring matrix W: (output_dim x output_dim)
    pub bilinear_w: Array2<f32>,
    pub bilinear_w_grad: Array2<f32>,

    embed_dim: usize,
    #[allow(dead_code)]
    expr_dim: usize,

    // Caches
    tf_indices_cache: Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_embed_out: Option<Array2<f32>>,
    gene_embed_out: Option<Array2<f32>>,
    tf_expr_cache: Option<Array2<f32>>,
    gene_expr_cache: Option<Array2<f32>>,
    tf_concat: Option<Array2<f32>>,
    gene_concat: Option<Array2<f32>>,
    tf_h1: Option<Array2<f32>>,
    gene_h1: Option<Array2<f32>>,
    tf_final: Option<Array2<f32>>,
    gene_final: Option<Array2<f32>>,
}

impl BilinearTwoTowerModel {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,
        expr_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        std_dev: f32,
        seed: u64,
    ) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, std_dev).unwrap();

        let tf_embed = Array2::random_using((num_tfs, embed_dim), dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, embed_dim), dist, &mut rng);

        let input_dim = embed_dim + expr_dim;
        let tf_fc1 = LinearLayer::new(input_dim, hidden_dim, seed);
        let tf_fc2 = LinearLayer::new(hidden_dim, output_dim, seed + 1);
        let gene_fc1 = LinearLayer::new(input_dim, hidden_dim, seed + 2);
        let gene_fc2 = LinearLayer::new(hidden_dim, output_dim, seed + 3);

        // Initialize bilinear W as identity + small noise for stable start
        let w_noise = Array2::random_using((output_dim, output_dim), Normal::new(0.0, 0.01).unwrap(), &mut rng);
        let bilinear_w = Array2::eye(output_dim) + &w_noise;

        Self {
            tf_embed,
            tf_embed_grad: Array2::zeros((num_tfs, embed_dim)),
            tf_fc1,
            tf_fc2,
            gene_embed,
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            gene_fc1,
            gene_fc2,
            bilinear_w,
            bilinear_w_grad: Array2::zeros((output_dim, output_dim)),
            embed_dim,
            expr_dim,
            tf_indices_cache: None,
            gene_indices_cache: None,
            tf_embed_out: None,
            gene_embed_out: None,
            tf_expr_cache: None,
            gene_expr_cache: None,
            tf_concat: None,
            gene_concat: None,
            tf_h1: None,
            gene_h1: None,
            tf_final: None,
            gene_final: None,
        }
    }

    /// Forward pass: returns sigmoid(e_TF^T W e_gene) per sample
    pub fn forward(
        &mut self,
        tf_indices: &[usize],
        gene_indices: &[usize],
        tf_expr: &Array2<f32>,
        gene_expr: &Array2<f32>,
    ) -> Array1<f32> {
        let batch_size = tf_indices.len();

        self.tf_indices_cache = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());
        self.tf_expr_cache = Some(tf_expr.clone());
        self.gene_expr_cache = Some(gene_expr.clone());

        // Embedding lookup
        let mut tf_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        let mut gene_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        for (i, &idx) in tf_indices.iter().enumerate() {
            tf_embed_batch.row_mut(i).assign(&self.tf_embed.row(idx));
        }
        for (i, &idx) in gene_indices.iter().enumerate() {
            gene_embed_batch.row_mut(i).assign(&self.gene_embed.row(idx));
        }
        self.tf_embed_out = Some(tf_embed_batch.clone());
        self.gene_embed_out = Some(gene_embed_batch.clone());

        // Concat + encode
        let tf_concat = ndarray::concatenate![Axis(1), tf_embed_batch, tf_expr.clone()];
        let gene_concat = ndarray::concatenate![Axis(1), gene_embed_batch, gene_expr.clone()];
        self.tf_concat = Some(tf_concat.clone());
        self.gene_concat = Some(gene_concat.clone());

        let tf_h1_pre = self.tf_fc1.forward(&tf_concat);
        let tf_h1 = relu(&tf_h1_pre);
        let tf_out = self.tf_fc2.forward(&tf_h1);
        self.tf_h1 = Some(tf_h1);
        self.tf_final = Some(tf_out.clone());

        let gene_h1_pre = self.gene_fc1.forward(&gene_concat);
        let gene_h1 = relu(&gene_h1_pre);
        let gene_out = self.gene_fc2.forward(&gene_h1);
        self.gene_h1 = Some(gene_h1);
        self.gene_final = Some(gene_out.clone());

        // Bilinear scoring: score_i = sigmoid(tf_out_i^T W gene_out_i)
        // Compute tf_out @ W  → [batch, output_dim], then element-wise dot with gene_out
        let tf_w = tf_out.dot(&self.bilinear_w); // [batch, output_dim]
        let mut scores = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let logit = tf_w.row(i).dot(&gene_out.row(i));
            scores[i] = 1.0 / (1.0 + (-logit).exp());
        }

        scores
    }

    /// Backward pass
    pub fn backward(&mut self, grad_output: &Array1<f32>) {
        let batch_size = grad_output.len();
        let tf_out = self.tf_final.as_ref().unwrap();
        let gene_out = self.gene_final.as_ref().unwrap();

        let tf_w = tf_out.dot(&self.bilinear_w); // [batch, output_dim]

        let mut grad_tf_out = Array2::zeros(tf_out.raw_dim());
        let mut grad_gene_out = Array2::zeros(gene_out.raw_dim());

        // Zero the bilinear grad accumulator
        self.bilinear_w_grad.fill(0.0);

        for i in 0..batch_size {
            let logit = tf_w.row(i).dot(&gene_out.row(i));
            let score = 1.0 / (1.0 + (-logit).exp());
            let grad_sigmoid = score * (1.0 - score) * grad_output[i];

            // d(logit)/d(gene_out) = W^T tf_out
            // d(logit)/d(tf_out)   = W gene_out
            // d(logit)/d(W)        = tf_out^T gene_out (outer product)
            let w_gene = self.bilinear_w.dot(&gene_out.row(i).to_owned());
            let wt_tf = self.bilinear_w.t().dot(&tf_out.row(i).to_owned());

            grad_tf_out.row_mut(i).assign(&(&w_gene * grad_sigmoid));
            grad_gene_out.row_mut(i).assign(&(&wt_tf * grad_sigmoid));

            // Accumulate W gradient: outer product of tf_out[i] and gene_out[i]
            for r in 0..self.bilinear_w.nrows() {
                for c in 0..self.bilinear_w.ncols() {
                    self.bilinear_w_grad[[r, c]] += grad_sigmoid * tf_out[[i, r]] * gene_out[[i, c]];
                }
            }
        }

        // Backward through gene encoder
        let grad_gene_h1 = self.gene_fc2.backward(&grad_gene_out);
        let gene_h1 = self.gene_h1.as_ref().unwrap();
        let grad_gene_h1_pre = relu_backward(gene_h1, &grad_gene_h1);
        let grad_gene_concat = self.gene_fc1.backward(&grad_gene_h1_pre);

        // Backward through TF encoder
        let grad_tf_h1 = self.tf_fc2.backward(&grad_tf_out);
        let tf_h1 = self.tf_h1.as_ref().unwrap();
        let grad_tf_h1_pre = relu_backward(tf_h1, &grad_tf_h1);
        let grad_tf_concat = self.tf_fc1.backward(&grad_tf_h1_pre);

        // Split embedding gradients
        let grad_tf_embed = grad_tf_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();
        let grad_gene_embed = grad_gene_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();

        let tf_indices = self.tf_indices_cache.as_ref().unwrap();
        let gene_indices = self.gene_indices_cache.as_ref().unwrap();

        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            let mut row = self.tf_embed_grad.row_mut(tf_idx);
            row += &grad_tf_embed.row(i);
        }
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            let mut row = self.gene_embed_grad.row_mut(gene_idx);
            row += &grad_gene_embed.row(i);
        }
    }

    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.bilinear_w_grad.fill(0.0);
        self.tf_fc1.zero_grad();
        self.tf_fc2.zero_grad();
        self.gene_fc1.zero_grad();
        self.gene_fc2.zero_grad();
    }

    pub fn num_parameters(&self) -> usize {
        let embed_params = self.tf_embed.len() + self.gene_embed.len();
        let tf_params = self.tf_fc1.weights.len() + self.tf_fc1.bias.len()
                      + self.tf_fc2.weights.len() + self.tf_fc2.bias.len();
        let gene_params = self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
                        + self.gene_fc2.weights.len() + self.gene_fc2.bias.len();
        let bilinear_params = self.bilinear_w.len();
        embed_params + tf_params + gene_params + bilinear_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_forward() {
        let mut model = BilinearTwoTowerModel::new(100, 500, 64, 11, 128, 64, 0.01, 42);
        let tf_indices = vec![0, 1, 2];
        let gene_indices = vec![10, 20, 30];
        let tf_expr = Array2::zeros((3, 11));
        let gene_expr = Array2::zeros((3, 11));
        let scores = model.forward(&tf_indices, &gene_indices, &tf_expr, &gene_expr);
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_bilinear_backward_runs() {
        let mut model = BilinearTwoTowerModel::new(10, 20, 8, 4, 16, 8, 0.01, 42);
        let tf_indices = vec![0, 1];
        let gene_indices = vec![5, 6];
        let tf_expr = Array2::zeros((2, 4));
        let gene_expr = Array2::zeros((2, 4));
        let scores = model.forward(&tf_indices, &gene_indices, &tf_expr, &gene_expr);
        let grad = Array1::from(vec![scores[0] - 1.0, scores[1] - 0.0]);
        model.backward(&grad);
        // Check gradients are non-zero
        assert!(model.bilinear_w_grad.iter().any(|&x| x != 0.0));
    }
}
