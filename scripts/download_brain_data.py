#!/usr/bin/env python3
"""
Download Human Brain Cell Atlas v1.0 data from CELLxGENE
Based on collection: 283d65eb-dd53-496d-adb7-7570c7caa443
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List
import time

# Collection ID for Human Brain Cell Atlas v1.0
COLLECTION_ID = "283d65eb-dd53-496d-adb7-7570c7caa443"
BASE_URL = "https://api.cellxgene.cziscience.com"
DATA_DIR = Path("data/brain_v1_0")

def fetch_collection_metadata() -> Dict:
    """Fetch metadata for the Brain v1.0 collection"""
    url = f"{BASE_URL}/curation/v1/collections/{COLLECTION_ID}"
    print(f"Fetching collection metadata from {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def list_datasets(collection_metadata: Dict) -> List[Dict]:
    """Extract dataset information from collection metadata"""
    datasets = []
    for dataset in collection_metadata.get("datasets", []):
        dataset_info = {
            "id": dataset["dataset_id"],
            "name": dataset.get("title", dataset["dataset_id"]),
            "cell_count": dataset.get("cell_count", 0),
            "assets": dataset.get("assets", [])
        }
        datasets.append(dataset_info)
    return datasets

def download_dataset(dataset: Dict, download_dir: Path):
    """Download a specific dataset"""
    dataset_dir = download_dir / dataset["id"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading: {dataset['name']}")
    print(f"  Cell count: {dataset['cell_count']:,}")
    
    for asset in dataset["assets"]:
        if asset["filetype"] == "H5AD":
            file_url = asset["url"]
            file_name = f"{dataset['id']}.h5ad"
            file_path = dataset_dir / file_name
            
            if file_path.exists():
                print(f"  ✓ Already downloaded: {file_name}")
                continue
            
            print(f"  Downloading: {file_name}")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if file_size > 0:
                            pct = (downloaded / file_size) * 100
                            print(f"    Progress: {pct:.1f}%", end='\r')
            
            print(f"    ✓ Downloaded: {file_size / (1024**2):.1f} MB")
            time.sleep(1)  # Rate limiting

def main():
    """Main download function"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Human Brain Cell Atlas v1.0 Data Download")
    print("=" * 60)
    
    try:
        # Fetch collection metadata
        metadata = fetch_collection_metadata()
        
        # Save metadata
        metadata_file = DATA_DIR / "collection_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_file}")
        
        # Get datasets
        datasets = list_datasets(metadata)
        print(f"\nFound {len(datasets)} datasets")
        print(f"Total cells: {sum(d['cell_count'] for d in datasets):,}")
        
        # Download datasets (sample first few for testing)
        print("\nDownloading sample datasets (first 5)...")
        for i, dataset in enumerate(datasets[:5]):
            download_dataset(dataset, DATA_DIR)
            if i < len(datasets) - 1:
                time.sleep(2)  # Rate limiting
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
