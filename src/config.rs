use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub project: ProjectConfig,
    pub data: DataConfig,
    pub priors: PriorsConfig,
    pub preprocessing: PreprocessingConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub loss: LossConfig,
    pub evaluation: EvaluationConfig,
    pub experiments: ExperimentsConfig,
    pub output: OutputConfig,
    pub compute: ComputeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub name: String,
    pub version: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub root: String,
    pub brain: BrainDataConfig,
    pub processed: ProcessedDataConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainDataConfig {
    pub collection_id: String,
    pub collection_name: String,
    pub base_dir: String,
    pub metadata_file: String,
    pub h5ad_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDataConfig {
    pub output_dir: String,
    pub manifest_file: String,
    pub pseudobulk_dir: String,
    pub states_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorsConfig {
    pub base_dir: String,
    pub dorothea: DatabaseConfig,
    pub trrust: DatabaseConfig,
    pub merged: MergedPriorsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub raw_file: String,
    pub processed_file: String,
    pub source_url: String,
    pub tf_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedPriorsConfig {
    pub output_file: String,
    pub stats_file: String,
    pub tf_count: usize,
    pub edge_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub min_cells_per_state: usize,
    pub filter_nan: bool,
    pub filter_inf: bool,
    pub normalize_expression: bool,
    pub log_transform: bool,
    pub scale_features: bool,
    pub min_gene_expression: f64,
    pub min_cells_expressing: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub r#type: String,
    pub embeddings: EmbeddingConfig,
    pub two_tower: TwoTowerConfig,
    pub monolithic: MonolithicConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub gene_embedding_dim: usize,
    pub tf_embedding_dim: usize,
    pub state_embedding_dim: usize,
    pub combined_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoTowerConfig {
    pub tf_encoder_hidden: Vec<usize>,
    pub tf_encoder_output: usize,
    pub tf_encoder_dropout: f64,
    pub gene_encoder_hidden: Vec<usize>,
    pub gene_encoder_output: usize,
    pub gene_encoder_dropout: f64,
    pub scoring: String,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonolithicConfig {
    pub hidden_layers: Vec<usize>,
    pub output_dim: usize,
    pub dropout: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub train_ratio: f64,
    pub val_ratio: f64,
    pub test_ratio: f64,
    pub stratify_by: String,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub early_stopping_patience: usize,
    pub optimizer: String,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub betas: Vec<f64>,
    pub use_scheduler: bool,
    pub scheduler_type: String,
    pub warmup_epochs: usize,
    pub gradient_clip_norm: f64,
    pub checkpoint_dir: String,
    pub save_every_n_epochs: usize,
    pub save_best_only: bool,
    pub monitor_metric: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    pub use_contrastive: bool,
    pub contrastive_weight: f64,
    pub num_negative_samples: usize,
    pub use_reconstruction: bool,
    pub reconstruction_weight: f64,
    pub reconstruction_type: String,
    pub use_prior_regularization: bool,
    pub prior_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub enrichment: EnrichmentConfig,
    pub reproducibility: ReproducibilityConfig,
    pub prediction: PredictionConfig,
    pub calibration: CalibrationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    pub use_gsea: bool,
    pub databases: Vec<String>,
    pub fdr_threshold: f64,
    pub target_enrichment_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityConfig {
    pub top_k: Vec<usize>,
    pub min_jaccard: f64,
    pub min_correlation: f64,
    pub bootstrap_iterations: usize,
    pub stability_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    pub metrics: Vec<String>,
    pub min_r2: f64,
    pub min_correlation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub num_bins: usize,
    pub max_ece: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentsConfig {
    pub output_dir: String,
    pub log_format: String,
    pub seeds: Vec<u64>,
    pub configs: Vec<ExperimentConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub model_type: String,
    pub description: String,
    #[serde(default)]
    pub use_reconstruction: Option<bool>,
    #[serde(default)]
    pub use_contrastive: Option<bool>,
    #[serde(default)]
    pub use_prior_regularization: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub results_dir: String,
    pub figures_dir: String,
    pub models_dir: String,
    pub results: ResultFilesConfig,
    pub figures: FigureFilesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFilesConfig {
    pub enrichment_scores: String,
    pub reproducibility_metrics: String,
    pub prediction_metrics: String,
    pub top_predictions: String,
    pub novel_edges: String,
    pub validation_hits: String,
    pub statistical_tests: String,
    pub summary_table: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigureFilesConfig {
    pub main_results: String,
    pub ablation_study: String,
    pub calibration_plot: String,
    pub training_curves: String,
    pub embeddings_tsne: String,
    pub performance_vs_params: String,
    pub stability_vs_sparsity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConfig {
    pub device: String,
    pub num_threads: usize,
    pub use_mixed_precision: bool,
    pub batch_accumulation_steps: usize,
    pub max_memory_gb: usize,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref())
            .context("Failed to read config file")?;
        let config: Config = toml::from_str(&contents)
            .context("Failed to parse config TOML")?;
        Ok(config)
    }

    pub fn load_default() -> Result<Self> {
        Self::from_file("config.toml")
    }
}
