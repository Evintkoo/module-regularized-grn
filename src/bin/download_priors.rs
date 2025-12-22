use anyhow::{Context, Result};
use module_regularized_grn::Config;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize)]
struct DorotheaInteraction {
    #[serde(alias = "source_genesymbol", alias = "source")]
    tf: String,
    #[serde(alias = "target_genesymbol", alias = "target")]
    target: String,
    #[serde(default)]
    dorothea_level: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PriorDatabase {
    pub tf_targets: HashMap<String, Vec<String>>,
    pub tf_count: usize,
    pub total_edges: usize,
    pub avg_targets_per_tf: f32,
}

impl PriorDatabase {
    fn new(tf_targets: HashMap<String, Vec<String>>) -> Self {
        let tf_count = tf_targets.len();
        let total_edges: usize = tf_targets.values().map(|v| v.len()).sum();
        let avg_targets_per_tf = if tf_count > 0 {
            total_edges as f32 / tf_count as f32
        } else {
            0.0
        };

        Self {
            tf_targets,
            tf_count,
            total_edges,
            avg_targets_per_tf,
        }
    }

    fn merge(databases: Vec<&PriorDatabase>) -> Self {
        let mut merged_map: HashMap<String, Vec<String>> = HashMap::new();
        
        for db in databases {
            for (tf, targets) in &db.tf_targets {
                merged_map
                    .entry(tf.clone())
                    .or_insert_with(Vec::new)
                    .extend(targets.clone());
            }
        }

        // Deduplicate targets
        for targets in merged_map.values_mut() {
            targets.sort();
            targets.dedup();
        }

        PriorDatabase::new(merged_map)
    }
}

fn download_dorothea(client: &Client, config: &Config) -> Result<PriorDatabase> {
    println!("======================================================");
    println!("Downloading DoRothEA database from OmniPath...");
    println!("======================================================");

    let response = client
        .get(&config.priors.dorothea.source_url)
        .timeout(Duration::from_secs(120))
        .send()
        .context("Failed to download DoRothEA")?;

    let text = response.text()?;
    
    // Save raw data
    let priors_dir = Path::new(&config.priors.base_dir);
    fs::create_dir_all(priors_dir)?;
    fs::write(&config.priors.dorothea.raw_file, &text)?;
    println!("✓ Saved raw data to {}", config.priors.dorothea.raw_file);

    // Parse TSV
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(text.as_bytes());

    let mut tf_targets: HashMap<String, Vec<String>> = HashMap::new();
    let mut interaction_count = 0;

    for result in rdr.deserialize() {
        let record: HashMap<String, String> = result?;
        
        // Extract TF and target (handle different column names)
        let tf = record.get("source_genesymbol")
            .or_else(|| record.get("source"))
            .context("Missing TF column")?
            .clone();
        
        let target = record.get("target_genesymbol")
            .or_else(|| record.get("target"))
            .context("Missing target column")?
            .clone();

        tf_targets.entry(tf).or_insert_with(Vec::new).push(target);
        interaction_count += 1;
    }

    // Deduplicate targets
    for targets in tf_targets.values_mut() {
        targets.sort();
        targets.dedup();
    }

    let db = PriorDatabase::new(tf_targets);
    
    println!("✓ Downloaded {} interactions", interaction_count);
    println!("  Unique TFs: {}", db.tf_count);
    println!("  Total edges: {}", db.total_edges);
    println!("  Avg targets per TF: {:.1}", db.avg_targets_per_tf);

    Ok(db)
}

fn download_trrust(client: &Client, config: &Config) -> Result<PriorDatabase> {
    println!("\n======================================================");
    println!("Downloading TRRUST v2 database...");
    println!("======================================================");

    let response = client
        .get(&config.priors.trrust.source_url)
        .timeout(Duration::from_secs(60))
        .send()
        .context("Failed to download TRRUST")?;

    let text = response.text()?;
    
    // Save raw data
    fs::write(&config.priors.trrust.raw_file, &text)?;
    println!("✓ Saved raw data to {}", config.priors.trrust.raw_file);

    // Parse TSV (no header: TF, Target, Regulation, PMID)
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(text.as_bytes());

    let mut tf_targets: HashMap<String, Vec<String>> = HashMap::new();
    let mut interaction_count = 0;
    let mut activation_count = 0;
    let mut repression_count = 0;

    for result in rdr.records() {
        let record = result?;
        if record.len() >= 3 {
            let tf = record[0].to_string();
            let target = record[1].to_string();
            let regulation = &record[2];

            tf_targets.entry(tf).or_insert_with(Vec::new).push(target);
            interaction_count += 1;

            if regulation == "Activation" {
                activation_count += 1;
            } else if regulation == "Repression" {
                repression_count += 1;
            }
        }
    }

    // Deduplicate targets
    for targets in tf_targets.values_mut() {
        targets.sort();
        targets.dedup();
    }

    let db = PriorDatabase::new(tf_targets);
    
    println!("✓ Downloaded {} interactions", interaction_count);
    println!("  Unique TFs: {}", db.tf_count);
    println!("  Total edges: {}", db.total_edges);
    println!("  Activation: {}", activation_count);
    println!("  Repression: {}", repression_count);
    println!("  Avg targets per TF: {:.1}", db.avg_targets_per_tf);

    Ok(db)
}

fn save_priors(dorothea: &PriorDatabase, trrust: &PriorDatabase, merged: &PriorDatabase, config: &Config) -> Result<()> {
    println!("\n======================================================");
    println!("Saving processed priors...");
    println!("======================================================");

    // Save individual databases
    let dorothea_json = serde_json::to_string_pretty(&dorothea.tf_targets)?;
    fs::write(&config.priors.dorothea.processed_file, dorothea_json)?;
    println!("✓ Saved {}", config.priors.dorothea.processed_file);

    let trrust_json = serde_json::to_string_pretty(&trrust.tf_targets)?;
    fs::write(&config.priors.trrust.processed_file, trrust_json)?;
    println!("✓ Saved {}", config.priors.trrust.processed_file);

    let merged_json = serde_json::to_string_pretty(&merged.tf_targets)?;
    fs::write(&config.priors.merged.output_file, merged_json)?;
    println!("✓ Saved {}", config.priors.merged.output_file);

    // Save statistics
    let stats = serde_json::json!({
        "dorothea": {
            "tf_count": dorothea.tf_count,
            "total_edges": dorothea.total_edges,
            "avg_targets_per_tf": dorothea.avg_targets_per_tf
        },
        "trrust": {
            "tf_count": trrust.tf_count,
            "total_edges": trrust.total_edges,
            "avg_targets_per_tf": trrust.avg_targets_per_tf
        },
        "merged": {
            "tf_count": merged.tf_count,
            "total_edges": merged.total_edges,
            "avg_targets_per_tf": merged.avg_targets_per_tf
        }
    });

    let stats_json = serde_json::to_string_pretty(&stats)?;
    fs::write(&config.priors.merged.stats_file, stats_json)?;
    println!("✓ Saved {}", config.priors.merged.stats_file);

    Ok(())
}

fn main() -> Result<()> {
    println!("======================================================");
    println!("TF-Target Prior Knowledge Download");
    println!("======================================================\n");

    // Load configuration
    let config = Config::load_default()
        .context("Failed to load config.toml")?;
    
    println!("✓ Loaded configuration from config.toml\n");

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    // Download DoRothEA
    let dorothea = download_dorothea(&client, &config)?;

    std::thread::sleep(Duration::from_secs(2)); // Rate limiting

    // Download TRRUST
    let trrust = download_trrust(&client, &config)?;

    // Merge databases
    println!("\n======================================================");
    println!("Merging databases...");
    println!("======================================================");
    
    let merged = PriorDatabase::merge(vec![&dorothea, &trrust]);
    
    println!("✓ Merged database:");
    println!("  Total TFs: {}", merged.tf_count);
    println!("  Total edges: {}", merged.total_edges);
    println!("  Avg targets per TF: {:.1}", merged.avg_targets_per_tf);

    // Save all
    save_priors(&dorothea, &trrust, &merged, &config)?;

    println!("\n======================================================");
    println!("✅ Prior knowledge download complete!");
    println!("======================================================");

    Ok(())
}
