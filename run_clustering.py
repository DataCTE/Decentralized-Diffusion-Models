#!/usr/bin/env python3
"""Generate cluster assignments for Decentralized Diffusion Models"""

import os
import torch
from tqdm import tqdm
import logging

from data.clustering import DDMClustering
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = get_config("config.py")
    
    # Define paths
    feature_path = os.path.join(config.feature_cache_path, "features")
    cluster_path = os.path.join(config.feature_cache_path, "clusters")
    
    # Ensure directories exist
    os.makedirs(cluster_path, exist_ok=True)
    
    # Verify features exist
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Features directory not found at {feature_path}. Please run feature extraction first.")
    
    feature_files = [f for f in os.listdir(feature_path) if f.endswith(".pt")]
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {feature_path}. Please run feature extraction first.")
    
    logger.info(f"Found {len(feature_files)} feature files in {feature_path}")
    
    # Load features
    logger.info("Loading features...")
    all_features = []
    for feature_file in tqdm(feature_files, desc="Loading features"):
        feature_tensor = torch.load(os.path.join(feature_path, feature_file), map_location='cpu')
        all_features.append(feature_tensor)
    
    features = torch.cat(all_features, dim=0)
    logger.info(f"Loaded features with shape: {features.shape}")
    
    # Initialize clusterer with configuration parameters
    logger.info("Initializing clusterer...")
    clusterer = DDMClustering(
        num_coarse_clusters=config.num_experts,
        num_fine_clusters=config.num_fine_clusters,
        default_feature_path=config.feature_cache_path
    )
    
    # Run clustering
    logger.info("Starting clustering process...")
    cluster_assignments = clusterer.cluster(features_list=[features])
    
    logger.info(f"Clustering complete! Generated assignments for {len(cluster_assignments)} samples")
    logger.info(f"Cluster distribution: {torch.bincount(cluster_assignments)}")
    
    # Verify cluster files were created
    cluster_files = [f for f in os.listdir(cluster_path) if f.endswith(".cluster.pt")]
    logger.info(f"Created {len(cluster_files)} cluster assignment files in {cluster_path}")
    
    return 0

if __name__ == "__main__":
    main() 