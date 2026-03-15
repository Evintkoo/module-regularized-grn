/// Cross-encoder model for GRN edge prediction.
/// Input: [TF_emb | Gene_emb | TF_expr | Gene_expr | TF_emb⊙Gene_emb]
/// Output: raw logit [batch, 1]. Use bce_loss/bce_loss_backward from nn.rs.
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array2, Axis};

pub struct CrossEncoderModel {
    // Embedding tables
    pub tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    pub gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,

    embed_dim: usize,
    expr_dim: usize,

    // FC layers (pub so Adam in training script can access grad_weights/grad_bias)
    pub fc1: LinearLayer,  // (3*embed+2*expr) → hidden, ReLU
    pub fc2: LinearLayer,  // hidden → hidden, ReLU
    pub fc3: LinearLayer,  // hidden → 1, raw logit

    // Caches for backward pass
    tf_indices_cache:   Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_embed_cache:     Option<Array2<f32>>,
    gene_embed_cache:   Option<Array2<f32>>,
    h1_pre_cache:       Option<Array2<f32>>,  // pre-ReLU input to fc1 (for relu_backward)
    h2_pre_cache:       Option<Array2<f32>>,  // pre-ReLU input to fc2 (for relu_backward)
}

impl CrossEncoderModel {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,
        expr_dim: usize,
        hidden_dim: usize,
        seed: u64,
    ) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0f32, 0.1).unwrap();
        let tf_embed   = Array2::random_using((num_tfs,   embed_dim), dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, embed_dim), dist, &mut rng);
        let input_dim  = 3 * embed_dim + 2 * expr_dim;

        Self {
            tf_embed_grad:   Array2::zeros((num_tfs,   embed_dim)),
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            fc1: LinearLayer::new(input_dim,  hidden_dim, seed),
            fc2: LinearLayer::new(hidden_dim, hidden_dim, seed + 1),
            fc3: LinearLayer::new(hidden_dim, 1,          seed + 2),
            embed_dim,
            expr_dim,
            tf_embed,
            gene_embed,
            tf_indices_cache:   None,
            gene_indices_cache: None,
            tf_embed_cache:     None,
            gene_embed_cache:   None,
            h1_pre_cache:       None,
            h2_pre_cache:       None,
        }
    }

    /// Forward pass. Returns raw logits [batch, 1]. No sigmoid applied.
    pub fn forward(
        &mut self,
        tf_indices:   &[usize],
        gene_indices: &[usize],
        tf_expr:      &Array2<f32>,
        gene_expr:    &Array2<f32>,
    ) -> Array2<f32> {
        let batch = tf_indices.len();

        // Embedding lookup
        let mut tf_emb   = Array2::zeros((batch, self.embed_dim));
        let mut gene_emb = Array2::zeros((batch, self.embed_dim));
        for (i, &idx) in tf_indices.iter().enumerate() {
            tf_emb.row_mut(i).assign(&self.tf_embed.row(idx));
        }
        for (i, &idx) in gene_indices.iter().enumerate() {
            gene_emb.row_mut(i).assign(&self.gene_embed.row(idx));
        }

        self.tf_indices_cache   = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());
        self.tf_embed_cache     = Some(tf_emb.clone());
        self.gene_embed_cache   = Some(gene_emb.clone());

        // Interaction: element-wise product [batch, embed_dim]
        let interaction = &tf_emb * &gene_emb;

        // Build input [batch, 3*embed+2*expr]
        let input = ndarray::concatenate![
            Axis(1),
            tf_emb.view(),
            gene_emb.view(),
            tf_expr.view(),
            gene_expr.view(),
            interaction.view()
        ];

        // FC1 → ReLU (cache pre-activation for relu_backward)
        let h1_pre = self.fc1.forward(&input);
        self.h1_pre_cache = Some(h1_pre.clone());
        let h1 = relu(&h1_pre);

        // FC2 → ReLU (cache pre-activation for relu_backward)
        let h2_pre = self.fc2.forward(&h1);
        self.h2_pre_cache = Some(h2_pre.clone());
        let h2 = relu(&h2_pre);

        // FC3 → raw logit (no activation)
        self.fc3.forward(&h2)
    }

    /// Backward pass. `grad_output` is [batch, 1] from bce_loss_backward.
    pub fn backward(&mut self, grad_output: &Array2<f32>) {
        // Backward through fc3
        let grad_h2 = self.fc3.backward(grad_output);

        // Backward through ReLU + fc2 (relu_backward takes pre-activation as first arg)
        let h2_pre = self.h2_pre_cache.as_ref().unwrap();
        let grad_h2_pre = relu_backward(h2_pre, &grad_h2);
        let grad_h1 = self.fc2.backward(&grad_h2_pre);

        // Backward through ReLU + fc1 (relu_backward takes pre-activation as first arg)
        let h1_pre = self.h1_pre_cache.as_ref().unwrap();
        let grad_h1_pre = relu_backward(h1_pre, &grad_h1);
        let grad_input = self.fc1.backward(&grad_h1_pre);
        // grad_input: [batch, 3*embed_dim + 2*expr_dim]

        let e = self.embed_dim;
        let x = self.expr_dim;

        // Split gradients from concatenated input
        let g_tf_emb      = grad_input.slice(ndarray::s![.., 0..e]).to_owned();
        let g_gene_emb    = grad_input.slice(ndarray::s![.., e..2*e]).to_owned();
        // Indices 2e..(2e+2x) are expression grads — no embedding params there, discard
        let g_interaction = grad_input.slice(ndarray::s![.., (2*e+2*x)..]).to_owned();

        // Interaction term backprop: interaction = tf_emb * gene_emb
        //   d(interaction)/d(tf_emb)   = gene_emb * g_interaction
        //   d(interaction)/d(gene_emb) = tf_emb   * g_interaction
        let tf_emb   = self.tf_embed_cache.as_ref().unwrap();
        let gene_emb = self.gene_embed_cache.as_ref().unwrap();

        let total_g_tf   = g_tf_emb   + &(gene_emb * &g_interaction);
        let total_g_gene = g_gene_emb + &(tf_emb   * &g_interaction);

        // Scatter into embedding gradient tables
        let tf_indices   = self.tf_indices_cache.as_ref().unwrap();
        let gene_indices = self.gene_indices_cache.as_ref().unwrap();
        for (i, &idx) in tf_indices.iter().enumerate() {
            let mut row = self.tf_embed_grad.row_mut(idx);
            row += &total_g_tf.row(i);
        }
        for (i, &idx) in gene_indices.iter().enumerate() {
            let mut row = self.gene_embed_grad.row_mut(idx);
            row += &total_g_gene.row(i);
        }
    }

    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.fc3.zero_grad();
    }

    pub fn num_parameters(&self) -> usize {
        self.tf_embed.len() + self.gene_embed.len()
            + self.fc1.weights.len() + self.fc1.bias.len()
            + self.fc2.weights.len() + self.fc2.bias.len()
            + self.fc3.weights.len() + self.fc3.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::nn::{bce_loss, bce_loss_backward};
    use ndarray::Array1;

    #[test]
    fn test_forward_output_shape() {
        // 10 TFs, 20 genes, embed=4, expr=3, hidden=8
        let mut model = CrossEncoderModel::new(10, 20, 4, 3, 8, 42);
        let tf_idx    = vec![0usize, 1, 2, 3];
        let gene_idx  = vec![5usize, 6, 7, 8];
        let tf_expr   = Array2::<f32>::zeros((4, 3));
        let gene_expr = Array2::<f32>::zeros((4, 3));
        let out = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        assert_eq!(out.dim(), (4, 1), "forward must return [batch, 1] raw logits");
    }

    #[test]
    fn test_loss_decreases_after_sgd_step() {
        let mut model = CrossEncoderModel::new(10, 20, 4, 3, 8, 99);
        let tf_idx    = vec![0usize, 1, 2, 3];
        let gene_idx  = vec![5usize, 6, 7, 8];
        let tf_expr   = Array2::<f32>::zeros((4, 3));
        let gene_expr = Array2::<f32>::zeros((4, 3));
        let labels    = Array1::<f32>::from(vec![1.0, 0.0, 1.0, 0.0]);

        let logits1 = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        let loss1   = bce_loss(&logits1, &labels);
        let grad    = bce_loss_backward(&logits1, &labels);
        model.backward(&grad);

        let lr = 0.5f32;
        model.tf_embed.scaled_add(-lr, &model.tf_embed_grad);
        model.gene_embed.scaled_add(-lr, &model.gene_embed_grad);
        model.fc1.update(lr);
        model.fc2.update(lr);
        model.fc3.update(lr);
        model.zero_grad();

        let logits2 = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        let loss2   = bce_loss(&logits2, &labels);
        assert!(loss2 < loss1, "loss must decrease: {:.4} → {:.4}", loss1, loss2);
    }
}
