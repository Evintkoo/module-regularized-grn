/// Hybrid model with Classification Head (Top Tower)
/// Instead of dot product, uses an MLP classifier on concatenated embeddings
use crate::models::nn::{LinearLayer, relu, relu_backward, sigmoid};
use ndarray::{Array1, Array2, Axis};

/// Classification Head - MLP that takes concatenated embeddings and predicts interaction
pub struct ClassificationHead {
    fc1: LinearLayer,  // (2 * output_dim) → head_hidden
    fc2: LinearLayer,  // head_hidden → head_hidden
    fc3: LinearLayer,  // head_hidden → 1

    // Caches for backward pass
    input_cache: Option<Array2<f32>>,
    h1_cache: Option<Array2<f32>>,
    h2_cache: Option<Array2<f32>>,
}

impl ClassificationHead {
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        Self {
            fc1: LinearLayer::new(input_dim, hidden_dim, seed),
            fc2: LinearLayer::new(hidden_dim, hidden_dim / 2, seed + 1),
            fc3: LinearLayer::new(hidden_dim / 2, 1, seed + 2),
            input_cache: None,
            h1_cache: None,
            h2_cache: None,
        }
    }

    pub fn forward(&mut self, combined: &Array2<f32>) -> Array1<f32> {
        self.input_cache = Some(combined.clone());

        // Layer 1
        let h1_pre = self.fc1.forward(combined);
        let h1 = relu(&h1_pre);
        self.h1_cache = Some(h1.clone());

        // Layer 2
        let h2_pre = self.fc2.forward(&h1);
        let h2 = relu(&h2_pre);
        self.h2_cache = Some(h2.clone());

        // Output layer
        let out = self.fc3.forward(&h2);

        // Sigmoid to get probabilities
        let probs = sigmoid(&out);

        // Return as 1D array
        probs.column(0).to_owned()
    }

    pub fn backward(&mut self, grad_output: &Array1<f32>) -> Array2<f32> {
        let h2 = self.h2_cache.as_ref().unwrap();

        // Reshape grad to 2D
        let grad_2d = grad_output.clone().insert_axis(Axis(1));

        // Get output for sigmoid gradient
        let out = self.fc3.forward(h2);
        let probs = sigmoid(&out);
        let grad_sigmoid = &probs * &(1.0 - &probs) * &grad_2d;

        // Backward through fc3
        let grad_h2 = self.fc3.backward(&grad_sigmoid);

        // Backward through ReLU and fc2
        let grad_h2_pre = relu_backward(&grad_h2, h2);
        let grad_h1 = self.fc2.backward(&grad_h2_pre);

        // Backward through ReLU and fc1
        let h1 = self.h1_cache.as_ref().unwrap();
        let grad_h1_pre = relu_backward(&grad_h1, h1);
        let grad_input = self.fc1.backward(&grad_h1_pre);

        grad_input
    }

    pub fn update(&mut self, learning_rate: f32) {
        self.fc1.update(learning_rate);
        self.fc2.update(learning_rate);
        self.fc3.update(learning_rate);
    }

    pub fn update_with_weight_decay(&mut self, learning_rate: f32, weight_decay: f32) {
        self.fc1.update_with_weight_decay(learning_rate, weight_decay);
        self.fc2.update_with_weight_decay(learning_rate, weight_decay);
        self.fc3.update_with_weight_decay(learning_rate, weight_decay);
    }

    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.fc3.zero_grad();
    }

    pub fn num_parameters(&self) -> usize {
        self.fc1.weights.len() + self.fc1.bias.len()
            + self.fc2.weights.len() + self.fc2.bias.len()
            + self.fc3.weights.len() + self.fc3.bias.len()
    }
}

/// Hybrid model with Classification Head
/// Uses MLP classifier instead of dot product for final prediction
pub struct HybridWithClassifier {
    // TF encoder
    tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    tf_fc1: LinearLayer,
    tf_fc2: LinearLayer,

    // Gene encoder
    gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,
    gene_fc1: LinearLayer,
    gene_fc2: LinearLayer,

    // Classification head (top tower)
    classifier: ClassificationHead,

    // Dimensions
    embed_dim: usize,
    output_dim: usize,

    // Caches
    tf_indices_cache: Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_embed_out: Option<Array2<f32>>,
    gene_embed_out: Option<Array2<f32>>,
    tf_concat: Option<Array2<f32>>,
    gene_concat: Option<Array2<f32>>,
    tf_h1: Option<Array2<f32>>,
    gene_h1: Option<Array2<f32>>,
    tf_final: Option<Array2<f32>>,
    gene_final: Option<Array2<f32>>,
    combined_cache: Option<Array2<f32>>,
}

impl HybridWithClassifier {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,
        expr_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        classifier_hidden: usize,
        std_dev: f32,
        seed: u64,
    ) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, std_dev).unwrap();

        // Initialize embeddings
        let tf_embed = Array2::random_using((num_tfs, embed_dim), dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, embed_dim), dist, &mut rng);

        // Input size: embed + expression
        let input_dim = embed_dim + expr_dim;

        // Initialize encoder layers
        let tf_fc1 = LinearLayer::new(input_dim, hidden_dim, seed);
        let tf_fc2 = LinearLayer::new(hidden_dim, output_dim, seed + 1);
        let gene_fc1 = LinearLayer::new(input_dim, hidden_dim, seed + 2);
        let gene_fc2 = LinearLayer::new(hidden_dim, output_dim, seed + 3);

        // Classification head: takes concatenated TF and Gene embeddings
        // Input: 2 * output_dim (TF output + Gene output)
        // Also include element-wise product and difference for richer features
        let classifier_input = output_dim * 4; // concat + product + diff + abs_diff
        let classifier = ClassificationHead::new(classifier_input, classifier_hidden, seed + 100);

        Self {
            tf_embed,
            tf_embed_grad: Array2::zeros((num_tfs, embed_dim)),
            tf_fc1,
            tf_fc2,
            gene_embed,
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            gene_fc1,
            gene_fc2,
            classifier,
            embed_dim,
            output_dim,
            tf_indices_cache: None,
            gene_indices_cache: None,
            tf_embed_out: None,
            gene_embed_out: None,
            tf_concat: None,
            gene_concat: None,
            tf_h1: None,
            gene_h1: None,
            tf_final: None,
            gene_final: None,
            combined_cache: None,
        }
    }

    /// Forward pass with classification head
    pub fn forward(
        &mut self,
        tf_indices: &[usize],
        gene_indices: &[usize],
        tf_expr: &Array2<f32>,
        gene_expr: &Array2<f32>,
    ) -> Array1<f32> {
        let batch_size = tf_indices.len();

        // Cache inputs
        self.tf_indices_cache = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());

        // Lookup embeddings
        let mut tf_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        let mut gene_embed_batch = Array2::zeros((batch_size, self.embed_dim));

        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            tf_embed_batch.row_mut(i).assign(&self.tf_embed.row(tf_idx));
        }
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            gene_embed_batch.row_mut(i).assign(&self.gene_embed.row(gene_idx));
        }

        self.tf_embed_out = Some(tf_embed_batch.clone());
        self.gene_embed_out = Some(gene_embed_batch.clone());

        // Concatenate embeddings + expression
        let tf_concat = ndarray::concatenate![Axis(1), tf_embed_batch, tf_expr.clone()];
        let gene_concat = ndarray::concatenate![Axis(1), gene_embed_batch, gene_expr.clone()];

        self.tf_concat = Some(tf_concat.clone());
        self.gene_concat = Some(gene_concat.clone());

        // TF encoding
        let tf_h1_pre = self.tf_fc1.forward(&tf_concat);
        let tf_h1 = relu(&tf_h1_pre);
        let tf_out = self.tf_fc2.forward(&tf_h1);
        self.tf_h1 = Some(tf_h1);
        self.tf_final = Some(tf_out.clone());

        // Gene encoding
        let gene_h1_pre = self.gene_fc1.forward(&gene_concat);
        let gene_h1 = relu(&gene_h1_pre);
        let gene_out = self.gene_fc2.forward(&gene_h1);
        self.gene_h1 = Some(gene_h1);
        self.gene_final = Some(gene_out.clone());

        // Create rich combined features:
        // 1. Concatenation: [tf_out, gene_out]
        // 2. Element-wise product: tf_out * gene_out
        // 3. Difference: tf_out - gene_out
        // 4. Absolute difference: |tf_out - gene_out|
        let product = &tf_out * &gene_out;
        let diff = &tf_out - &gene_out;
        let abs_diff = diff.mapv(|x| x.abs());

        let combined = ndarray::concatenate![Axis(1), tf_out, gene_out, product, abs_diff];
        self.combined_cache = Some(combined.clone());

        // Pass through classification head
        self.classifier.forward(&combined)
    }

    /// Backward pass
    pub fn backward(&mut self, grad_output: &Array1<f32>) {
        // Backward through classifier
        let grad_combined = self.classifier.backward(grad_output);

        let tf_out = self.tf_final.as_ref().unwrap();
        let gene_out = self.gene_final.as_ref().unwrap();

        // Split gradients for combined features
        // combined = [tf_out, gene_out, product, abs_diff]
        let out_dim = self.output_dim;

        let grad_tf_concat_part = grad_combined.slice(ndarray::s![.., 0..out_dim]).to_owned();
        let grad_gene_concat_part = grad_combined.slice(ndarray::s![.., out_dim..2*out_dim]).to_owned();
        let grad_product = grad_combined.slice(ndarray::s![.., 2*out_dim..3*out_dim]).to_owned();
        let grad_abs_diff = grad_combined.slice(ndarray::s![.., 3*out_dim..]).to_owned();

        // Gradient through element-wise product
        let grad_tf_product = &grad_product * gene_out;
        let grad_gene_product = &grad_product * tf_out;

        // Gradient through absolute difference
        let diff = tf_out - gene_out;
        let sign = diff.mapv(|x| if x >= 0.0 { 1.0 } else { -1.0 });
        let grad_tf_abs = &grad_abs_diff * &sign;
        let grad_gene_abs = &grad_abs_diff * &(-&sign);

        // Total gradients for TF and Gene outputs
        let grad_tf_out = &grad_tf_concat_part + &grad_tf_product + &grad_tf_abs;
        let grad_gene_out = &grad_gene_concat_part + &grad_gene_product + &grad_gene_abs;

        // Backward through gene encoder
        let grad_gene_h1 = self.gene_fc2.backward(&grad_gene_out);
        let gene_h1 = self.gene_h1.as_ref().unwrap();
        let grad_gene_h1_pre = relu_backward(&grad_gene_h1, gene_h1);
        let grad_gene_concat = self.gene_fc1.backward(&grad_gene_h1_pre);

        // Backward through TF encoder
        let grad_tf_h1 = self.tf_fc2.backward(&grad_tf_out);
        let tf_h1 = self.tf_h1.as_ref().unwrap();
        let grad_tf_h1_pre = relu_backward(&grad_tf_h1, tf_h1);
        let grad_tf_concat = self.tf_fc1.backward(&grad_tf_h1_pre);

        // Extract embedding gradients
        let grad_tf_embed = grad_tf_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();
        let grad_gene_embed = grad_gene_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();

        // Accumulate embedding gradients
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

    /// Update parameters with weight decay
    pub fn update_with_weight_decay(&mut self, learning_rate: f32, weight_decay: f32) {
        let decay_factor = 1.0 - learning_rate * weight_decay;

        // Update embeddings
        self.tf_embed.scaled_add(-learning_rate, &self.tf_embed_grad);
        self.tf_embed *= decay_factor;

        self.gene_embed.scaled_add(-learning_rate, &self.gene_embed_grad);
        self.gene_embed *= decay_factor;

        // Update encoder layers
        self.tf_fc1.update_with_weight_decay(learning_rate, weight_decay);
        self.tf_fc2.update_with_weight_decay(learning_rate, weight_decay);
        self.gene_fc1.update_with_weight_decay(learning_rate, weight_decay);
        self.gene_fc2.update_with_weight_decay(learning_rate, weight_decay);

        // Update classifier
        self.classifier.update_with_weight_decay(learning_rate, weight_decay);
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.tf_fc1.zero_grad();
        self.tf_fc2.zero_grad();
        self.gene_fc1.zero_grad();
        self.gene_fc2.zero_grad();
        self.classifier.zero_grad();
    }

    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.tf_embed.len() + self.gene_embed.len();
        let tf_params = self.tf_fc1.weights.len() + self.tf_fc1.bias.len()
                      + self.tf_fc2.weights.len() + self.tf_fc2.bias.len();
        let gene_params = self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
                        + self.gene_fc2.weights.len() + self.gene_fc2.bias.len();
        let classifier_params = self.classifier.num_parameters();
        embed_params + tf_params + gene_params + classifier_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_head() {
        let mut head = ClassificationHead::new(256, 128, 42);
        let input = Array2::ones((4, 256));
        let output = head.forward(&input);
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_hybrid_with_classifier() {
        let mut model = HybridWithClassifier::new(
            100, 500, 64, 11, 128, 64, 128, 0.01, 42
        );

        let tf_indices = vec![0, 1, 2];
        let gene_indices = vec![10, 20, 30];
        let tf_expr = Array2::zeros((3, 11));
        let gene_expr = Array2::zeros((3, 11));

        let scores = model.forward(&tf_indices, &gene_indices, &tf_expr, &gene_expr);

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}
