use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

const DOROTHEA_URL: &str = "https://omnipathdb.org/interactions?datasets=dorothea&fields=sources,references&license=academic";
const TRRUST_URL: &str = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv";

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

fn download_dorothea(client: &Client) -> Result<PriorDatabase> {
    println!("======================================================");
    println!("Downloading DoRothEA database from OmniPath...");
    println!("======================================================");

    let response = client
        .get(DOROTHEA_URL)
        .timeout(Duration::from_secs(120))
        .send()
        .context("Failed to download DoRothEA")?;

    let text = response.text()?;
    
    // Save raw data
    let priors_dir = Path::new("data/priors");
    fs::create_dir_all(priors_dir)?;
    fs::write(priors_dir.join("dorothea_raw.tsv"), &text)?;
    println!("✓ Saved raw data to data/priors/dorothea_raw.tsv");

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

fn download_trrust(client: &Client) -> Result<PriorDatabase> {
    println!("\n======================================================");
    println!("Downloading TRRUST v2 database...");
    println!("======================================================");

    let response = client
        .get(TRRUST_URL)
        .timeout(Duration::from_secs(60))
        .send()
        .context("Failed to download TRRUST")?;

    let text = response.text()?;
    
    // Save raw data
    let priors_dir = Path::new("data/priors");
    fs::write(priors_dir.join("trrust_raw.tsv"), &text)?;
    println!("✓ Saved raw data to data/priors/trrust_raw.tsv");

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

fn save_priors(dorothea: &PriorDatabase, trrust: &PriorDatabase, merged: &PriorDatabase) -> Result<()> {
    println!("\n======================================================");
    println!("Saving processed priors...");
    println!("======================================================");

    let priors_dir = Path::new("data/priors");

    // Save individual databases
    let dorothea_json = serde_json::to_string_pretty(&dorothea.tf_targets)?;
    fs::write(priors_dir.join("dorothea_priors.json"), dorothea_json)?;
    println!("✓ Saved data/priors/dorothea_priors.json");

    let trrust_json = serde_json::to_string_pretty(&trrust.tf_targets)?;
    fs::write(priors_dir.join("trrust_priors.json"), trrust_json)?;
    println!("✓ Saved data/priors/trrust_priors.json");

    let merged_json = serde_json::to_string_pretty(&merged.tf_targets)?;
    fs::write(priors_dir.join("merged_priors.json"), merged_json)?;
    println!("✓ Saved data/priors/merged_priors.json");

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
    fs::write(priors_dir.join("priors_stats.json"), stats_json)?;
    println!("✓ Saved data/priors/priors_stats.json");

    Ok(())
}

fn main() -> Result<()> {
    println!("======================================================");
    println!("TF-Target Prior Knowledge Download");
    println!("======================================================\n");

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    // Download DoRothEA
    let dorothea = download_dorothea(&client)?;

    std::thread::sleep(Duration::from_secs(2)); // Rate limiting

    // Download TRRUST
    let trrust = download_trrust(&client)?;

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
    save_priors(&dorothea, &trrust, &merged)?;

    println!("\n======================================================");
    println!("✅ Prior knowledge download complete!");
    println!("======================================================");

    Ok(())
}
