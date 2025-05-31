#!/usr/bin/env python3
"""
Debug script to check metadata paths.
"""
import json
from pathlib import Path
from ursakit.client import UrsaClient
from sklearn.ensemble import RandomForestClassifier
from app.config import settings, REPO_ROOT

def check_metadata_paths():
    print("=== Checking Metadata Paths ===")
    
    sdk_dir = REPO_ROOT / "storage" / "models"
    sdk_dir.mkdir(parents=True, exist_ok=True)
    client = UrsaClient(dir=sdk_dir, use_server=False)
    
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    model.fit([[1, 2], [3, 4]], [0, 1])
    
    model_id = client.save(model, name='test')
    print(f"Model saved with ID: {model_id}")
    
    # Read the metadata
    metadata_file = sdk_dir / 'models' / model_id / 'metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Artifact path: {metadata['artifacts']['model']['path']}")
    print(f"Is absolute: {Path(metadata['artifacts']['model']['path']).is_absolute()}")
    print(f"SDK dir: {sdk_dir}")
    print(f"Model dir: {sdk_dir / 'models' / model_id}")

if __name__ == "__main__":
    check_metadata_paths() 