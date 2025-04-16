#!/usr/bin/env python3
"""
Script to download and store all scikit-learn datasets and generate examples 
from generator functions. Data is saved in appropriate formats for integrity.
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn import datasets
import warnings
import time

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# full_path = os.path.dirname(os.path.abspath(__file__))
full_path = ''

# Create main directory for storage
BASE_DIR = os.path.join(full_path, "sklearn_datasets")
os.makedirs(BASE_DIR, exist_ok=True)

# Create subdirectories
BUILTIN_DIR = os.path.join(BASE_DIR, "builtin")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")
FETCHED_DIR = os.path.join(BASE_DIR, "fetched")

for directory in [BUILTIN_DIR, GENERATED_DIR, FETCHED_DIR]:
    os.makedirs(directory, exist_ok=True)

def save_dataset(data, name, directory):
    """Save a dataset in multiple formats for integrity"""
    path_base = os.path.join(directory, name)
    
    # Save as compressed joblib (efficient for numpy arrays)
    joblib.dump(data, f"{path_base}.joblib", compress=3)
    
    # Also save as CSV for human readability if possible
    try:
        if hasattr(data, 'data') and hasattr(data, 'target'):
            # Try to create DataFrame with feature names if available
            if hasattr(data, 'feature_names'):
                df = pd.DataFrame(data.data, columns=data.feature_names)
            else:
                df = pd.DataFrame(data.data)
            
            # Add target column
            if hasattr(data, 'target_names') and len(data.target_names) == len(np.unique(data.target)):
                # Map numeric targets to their names
                target_map = {i: name for i, name in enumerate(data.target_names)}
                vectorized_map = np.vectorize(target_map.get)
                df['target'] = vectorized_map(data.target)
            else:
                df['target'] = data.target
                
            df.to_csv(f"{path_base}.csv", index=False)
            print(f"  - Saved as CSV: {name}.csv")
    except Exception as e:
        print(f"  - Could not save as CSV: {e}")
    
    print(f"  - Saved as joblib: {name}.joblib")
    return path_base

def process_builtin_datasets():
    """Download and save all built-in datasets"""
    print("\n=== Processing Built-in Datasets ===")
    
    # List of all built-in datasets to download
    builtin_loaders = {
        'iris': datasets.load_iris,
        'digits': datasets.load_digits,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'diabetes': datasets.load_diabetes,
    }
    
    # Try to load Boston housing if available (deprecated in newer versions)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin_loaders['boston'] = datasets.load_boston
    except:
        print("Boston housing dataset not available in this sklearn version")
    
    # Load and save each dataset
    for name, loader in builtin_loaders.items():
        print(f"Processing {name}...")
        try:
            data = loader()
            save_dataset(data, name, BUILTIN_DIR)
        except Exception as e:
            print(f"  - Error loading {name}: {e}")

class Dataset:
    """A simple container for dataset data and target attributes."""
    pass

def process_generated_datasets():
    """Generate and save examples from generator functions"""
    print("\n=== Processing Generated Datasets ===")
    
    # Define generator functions with parameters
    generators = {
        'classification': (datasets.make_classification, 
                          {'n_samples': 10000, 'n_features': 20, 'n_informative': 10, 
                           'n_redundant': 5, 'random_state': 42}),
        'regression': (datasets.make_regression, 
                      {'n_samples': 10000, 'n_features': 20, 'n_informative': 10, 
                       'noise': 0.5, 'random_state': 42}),
        'blobs': (datasets.make_blobs, 
                 {'n_samples': 10000, 'centers': 5, 'cluster_std': 1.0, 
                  'random_state': 42}),
        'circles': (datasets.make_circles, 
                   {'n_samples': 10000, 'noise': 0.05, 'factor': 0.5, 
                    'random_state': 42}),
        'moons': (datasets.make_moons, 
                 {'n_samples': 10000, 'noise': 0.1, 'random_state': 42}),
        'swiss_roll': (datasets.make_swiss_roll, 
                      {'n_samples': 10000, 'noise': 0.1, 'random_state': 42})
    }
    
    # Generate and save each dataset
    for name, (generator, params) in generators.items():
        print(f"Generating {name}...")
        try:
            if name == 'swiss_roll':
                X, target = generator(**params)
                data = {'data': X, 'target': target}
                data = {'data': X, 'target': target}
            else:
                X, y = generator(**params)
                data = {'data': X, 'target': y}
            
            # Store as an object with data and target attributes for consistency
            dataset = Dataset()
            dataset.data = data['data']
            dataset.target = data['target']
            
            save_dataset(dataset, name, GENERATED_DIR)
            
            # Also save as NumPy arrays for raw access
            np.savez_compressed(
                os.path.join(GENERATED_DIR, f"{name}_raw.npz"),
                X=data['data'], 
                y=data['target']
            )
            print(f"  - Saved as NumPy: {name}_raw.npz")
            
        except Exception as e:
            print(f"  - Error generating {name}: {e}")

def process_fetched_datasets():
    """Download and save larger datasets from remote sources"""
    print("\n=== Processing Fetched Datasets ===")
    
    # Define fetchers with any needed parameters
    fetchers = {
        'california_housing': (datasets.fetch_california_housing, {}),
        'olivetti_faces': (datasets.fetch_olivetti_faces, {}),
        '20newsgroups': (datasets.fetch_20newsgroups, 
                          {'subset': 'train', 'remove': ('headers', 'footers', 'quotes')}),
    }
    
    # Try to add fetch_lfw_people if it doesn't take too long
    try:
        fetchers['lfw_people'] = (datasets.fetch_lfw_people, 
                                  {'min_faces_per_person': 20, 'resize': 0.4})
    except:
        print("LFW People dataset might take too long - skipped")
    
    # Fetch and save each dataset
    for name, (fetcher, params) in fetchers.items():
        print(f"Fetching {name}...")
        try:
            start_time = time.time()
            data = fetcher(**params)
            fetch_time = time.time() - start_time
            print(f"  - Downloaded in {fetch_time:.2f} seconds")
            
            save_dataset(data, name, FETCHED_DIR)
            
            # Special handling for text datasets
            if name == '20newsgroups':
                # Save raw text data separately
                texts_file = os.path.join(FETCHED_DIR, f"{name}_texts.txt")
                with open(texts_file, 'w', encoding='utf-8') as f:
                    limit = 1000  # Configurable limit for the number of samples
                    for i, (text, target) in enumerate(zip(data.data[:limit], data.target[:limit])):  # Save only up to the limit
                        f.write(f"--- Document {i} (Category {target}: {data.target_names[target]}) ---\n")
                        f.write(text[:10000] + "...\n\n" if len(text) > 10000 else text + "\n\n")
                print(f"  - Saved text samples: {name}_texts.txt")
                
        except Exception as e:
            print(f"  - Error fetching {name}: {e}")

def create_dataset_info():
    """Create a summary info file of all datasets"""
    print("\n=== Creating Dataset Summary ===")
    
    info_path = os.path.join(BASE_DIR, "dataset_info.txt")
    
    with open(info_path, 'w') as f:
        f.write("SCIKIT-LEARN DATASET COLLECTION\n")
        f.write("===============================\n\n")
        f.write(f"Created on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Get total size of dataset directory
        total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                         for dirpath, _, filenames in os.walk(BASE_DIR) 
                         for filename in filenames)
        
        f.write(f"Total storage used: {total_size / (1024*1024):.2f} MB\n\n")
        
        # List all files by category
        for directory, title in [(BUILTIN_DIR, "Built-in Datasets"), 
                                (GENERATED_DIR, "Generated Datasets"),
                                (FETCHED_DIR, "Fetched Datasets")]:
            
            f.write(f"\n{title}\n")
            f.write("-" * len(title) + "\n")
            
            files = os.listdir(directory)
            if not files:
                f.write("  No datasets found\n")
                continue
                
            for filename in sorted(files):
                file_path = os.path.join(directory, filename)
                try:
                    size_mb = os.path.getsize(file_path) / (1024*1024)
                except OSError as e:
                    print(f"  - Could not access file size for {filename}: {e}")
                    size_mb = 0
                f.write(f"  - {filename} ({size_mb:.2f} MB)\n")
    
    print(f"Dataset summary created at: {info_path}")

def main():
    """Main function to execute all dataset processing"""
    print("Starting scikit-learn dataset downloader and generator...")
    print(f"Data will be stored in: {os.path.abspath(BASE_DIR)}")
    
    # Step 1: Process and save all built-in datasets provided by scikit-learn
    process_builtin_datasets()
    
    # Step 2: Generate synthetic datasets using scikit-learn's generator functions
    process_generated_datasets()
    
    # Step 3: Fetch larger datasets from remote sources and save them locally
    process_fetched_datasets()
    
    # Step 4: Create a summary file containing information about all saved datasets
    create_dataset_info()
    
    print("\nAll datasets processed and saved successfully!")
    print(f"Data location: {os.path.abspath(BASE_DIR)}")

if __name__ == "__main__":
    main()