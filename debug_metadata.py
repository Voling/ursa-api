#!/usr/bin/env python3
"""
Debug script to check metadata paths.
"""
import tempfile
import json
from pathlib import Path
from ursakit.client import UrsaClient
from sklearn.ensemble import RandomForestClassifier

def check_metadata_paths():
    print("=== Checking Metadata Paths ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        client = UrsaClient(dir=temp_path, use_server=False)
        
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit([[1, 2], [3, 4]], [0, 1])
        
        model_id = client.save(model, name='test')
        print(f"Model saved with ID: {model_id}")
        
        # Read the metadata
        metadata_file = temp_path / 'models' / model_id / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Artifact path: {metadata['artifacts']['model']['path']}")
        print(f"Is absolute: {Path(metadata['artifacts']['model']['path']).is_absolute()}")
        print(f"Temp dir: {temp_path}")
        print(f"Model dir: {temp_path / 'models' / model_id}")

if __name__ == "__main__":
    check_metadata_paths() 