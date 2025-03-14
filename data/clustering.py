"""Clustering functionality for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
import os
import pickle
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from data.feature_extractor import DINOv2FeatureExtractor

# Import centralized utilities
from utils.distributed import (
    broadcast_numpy_array, 
    broadcast_object, 
    is_main_process, 
    get_rank, 
    get_world_size, 
    synchronize
)
from utils.logging import setup_logger
from utils.visualization import visualize_embeddings, create_image_grid, tensor_to_pil

logger = logging.getLogger(__name__)

class ClusterManager:
    """Manages clustering operations for DDM following paper's two-stage approach"""
    
    def __init__(self, local_rank=0, dataset_path=None, config=None):
        """
        Initialize the cluster manager
        
        Args:
            local_rank: Local process rank for distributed training
            dataset_path: Path to the dataset
            config: Configuration object
        """
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.dataset_path = dataset_path
        self.config = config
        
        # Initialize logger
        self.logger = setup_logger(name="ClusterManager", rank=local_rank)
        
        # Initialize vision encoder
        self._init_vision_encoder()
        
        # Initialize other attributes
        self.features = None
        self.cluster_labels = None
        self.fine_kmeans = None
        self.coarse_kmeans = None
        self.old_cluster_labels = None
        self.fine_cluster_labels = None
        
        # Setup cache directory for features and clusters
        self.cache_dir = os.path.join(
            os.path.dirname(dataset_path), 
            "ddm_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Feature cache file
        self.feature_cache_file = os.path.join(
            self.cache_dir, 
            f"features_dino_{config.dino_size}.pkl"
        )
        
        # Cluster cache files
        self.fine_cluster_cache_file = os.path.join(
            self.cache_dir, 
            f"fine_clusters_{config.num_experts * 8}.pkl"
        )
        
        self.coarse_cluster_cache_file = os.path.join(
            self.cache_dir, 
            f"coarse_clusters_{config.num_experts}.pkl"
        )
        
    def _init_vision_encoder(self):
        """Initialize vision encoder for feature extraction"""
        try:
            self.logger.info("Initializing vision encoder for feature extraction")
            self.vision_encoder = DINOv2FeatureExtractor(self.device)
            
            # Freeze encoder parameters
            for param in self.vision_encoder.parameters():
                param.requires_grad_(False)
                
            self.logger.info("Vision encoder initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize vision encoder: {str(e)}")
            raise

    def extract_features(self, dataloader):
        """
        Extract features from dataset images with caching
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            Tensor of extracted features
        """
        # Try to load features from cache
        if is_main_process() and os.path.exists(self.feature_cache_file):
            self.logger.info(f"Loading features from cache: {self.feature_cache_file}")
            try:
                with open(self.feature_cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    features = cached_data["features"]
                    image_paths = cached_data["paths"]
                    
                    # Verify cached features shape
                    if len(features) > 0:
                        self.logger.info(f"Loaded {features.shape[0]} cached features")
                        return torch.from_numpy(features), image_paths
                    else:
                        self.logger.warning("Cached features are empty, re-extracting")
            except Exception as e:
                self.logger.warning(f"Failed to load cached features: {e}. Extracting new features.")
        
        # Extract features if not loaded from cache
        self.logger.info("Extracting features from dataset")
        features = []
        image_paths = []
        
        # Only extract on main process in distributed setting
        if get_rank() == 0:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Extracting features"):
                    # Get images and paths from batch
                    if isinstance(batch, dict):
                        images = batch["image"]
                        paths = batch.get("path", [None] * images.size(0))
                    else:
                        images = batch
                        paths = [None] * images.size(0)
                        
                    # Move images to device
                    images = images.to(self.device)
                    
                    # Extract features in smaller batches to avoid OOM
                    sub_batch_size = getattr(self.config, "feature_sub_batch_size", 32)
                    
                    if images.size(0) > sub_batch_size:
                        # Process in sub-batches
                        batch_features = []
                        for i in range(0, images.size(0), sub_batch_size):
                            end_idx = min(i + sub_batch_size, images.size(0))
                            sub_features = self.vision_encoder(images[i:end_idx])
                            batch_features.append(sub_features.cpu())
                        batch_features = torch.cat(batch_features, dim=0)
                    else:
                        # Process whole batch
                        batch_features = self.vision_encoder(images).cpu()
                    
                    # Store features and paths
                    features.append(batch_features)
                    image_paths.extend(paths)
                
                # Concatenate features
                if features:
                    features = torch.cat(features, dim=0)
                else:
                    features = torch.empty((0, self.vision_encoder.feature_dim))
                
                self.logger.info(f"Extracted features for {features.size(0)} images")
                
                # Save to cache
                try:
                    with open(self.feature_cache_file, "wb") as f:
                        pickle.dump({
                            "features": features.numpy(),
                            "paths": image_paths
                        }, f)
                    self.logger.info(f"Saved features to cache: {self.feature_cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save features to cache: {e}")
        
        # Broadcast features in distributed setting
        if get_world_size() > 1:
            # Convert to numpy for broadcasting
            if get_rank() == 0 and features is not None:
                features_np = features.numpy()
            else:
                features_np = None
                
            # Broadcast features
            features_np = broadcast_numpy_array(features_np)
            
            # Convert back to tensor
            features = torch.from_numpy(features_np)
            
            # Broadcast paths
            image_paths = broadcast_object(image_paths)
            
        self.features = features
        return features, image_paths

    def cluster_dataset(self, features):
        """
        Cluster features using two-stage K-means as in Paper Section 4.1
        
        Args:
            features: Tensor of image features
            
        Returns:
            Array of coarse cluster assignments
        """
        # Save old labels if available
        if self.cluster_labels is not None:
            self.old_cluster_labels = self.cluster_labels.copy()
        
        # Step 1: First stage clustering into fine-grained clusters (1024 clusters as in paper)
        fine_clusters = self._create_fine_clusters(features)
        
        # Step 2: Second stage clustering to consolidate into coarse clusters
        coarse_clusters = self._consolidate_clusters(features, fine_clusters)
        
        # Store final cluster labels
        self.cluster_labels = coarse_clusters
        
        # Log cluster distribution
        if get_rank() == 0:
            unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                self.logger.info(f"Cluster {label}: {count} images ({count / len(self.cluster_labels) * 100:.2f}%)")
            
        return self.cluster_labels
    
    def _create_fine_clusters(self, features):
        """Create fine-grained clusters (1024 as recommended in paper)"""
        # Check cache first
        if is_main_process() and os.path.exists(self.fine_cluster_cache_file):
            self.logger.info(f"Loading fine clusters from cache: {self.fine_cluster_cache_file}")
            try:
                with open(self.fine_cluster_cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    fine_cluster_labels = cache_data["labels"]
                    self.fine_kmeans = cache_data["model"]
                    self.logger.info(f"Loaded {len(np.unique(fine_cluster_labels))} fine clusters from cache")
                    
                    # Broadcast to all processes
                    if get_world_size() > 1:
                        fine_cluster_labels = broadcast_numpy_array(fine_cluster_labels)
                    
                    self.fine_cluster_labels = fine_cluster_labels
                    return fine_cluster_labels
            except Exception as e:
                self.logger.warning(f"Failed to load fine clusters from cache: {e}")
        
        # Compute fine clusters
        num_fine_clusters = min(self.config.num_experts * 8, features.shape[0] // 100)
        self.logger.info(f"Creating {num_fine_clusters} fine-grained clusters")
        
        # Only perform clustering on main process
        if get_rank() == 0:
            # Convert features to numpy for sklearn
            features_np = features.numpy()
            
            # Initialize MiniBatchKMeans for memory efficiency
            fine_kmeans = MiniBatchKMeans(
                n_clusters=num_fine_clusters,
                batch_size=min(4096, features_np.shape[0]),
                init="k-means++",
                max_iter=300,
                random_state=42
            )
            
            # Fit K-means
            self.logger.info("Fitting fine-grained K-means")
            fine_kmeans.fit(features_np)
            
            # Get cluster assignments
            fine_cluster_labels = fine_kmeans.labels_
            
            # Store model
            self.fine_kmeans = fine_kmeans
            
            # Save to cache
            try:
                with open(self.fine_cluster_cache_file, "wb") as f:
                    pickle.dump({
                        "labels": fine_cluster_labels,
                        "model": fine_kmeans
                    }, f)
                self.logger.info(f"Saved fine clusters to cache: {self.fine_cluster_cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save fine clusters to cache: {e}")
        else:
            fine_cluster_labels = None
            
        # Broadcast to all processes
        if get_world_size() > 1:
            fine_cluster_labels = broadcast_numpy_array(fine_cluster_labels)
            
        self.fine_cluster_labels = fine_cluster_labels
        return fine_cluster_labels
    
    def _consolidate_clusters(self, features, fine_cluster_labels):
        """Consolidate fine-grained clusters into coarse clusters"""
        # Check cache first
        if is_main_process() and os.path.exists(self.coarse_cluster_cache_file):
            self.logger.info(f"Loading coarse clusters from cache: {self.coarse_cluster_cache_file}")
            try:
                with open(self.coarse_cluster_cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    coarse_cluster_labels = cache_data["labels"]
                    cluster_mapping = cache_data.get("mapping", {})
                    self.coarse_kmeans = cache_data["model"]
                    self.logger.info(f"Loaded {len(np.unique(coarse_cluster_labels))} coarse clusters from cache")
                    
                    # Broadcast to all processes
                    if get_world_size() > 1:
                        coarse_cluster_labels = broadcast_numpy_array(coarse_cluster_labels)
                    
                    return coarse_cluster_labels
            except Exception as e:
                self.logger.warning(f"Failed to load coarse clusters from cache: {e}")
        
        # Compute coarse clusters
        num_coarse_clusters = self.config.num_experts
        self.logger.info(f"Consolidating fine clusters into {num_coarse_clusters} coarse clusters")
        
        # Only perform clustering on main process
        if get_rank() == 0:
            # Method 1: Cluster the centroids of fine clusters (More memory efficient)
            if hasattr(self.fine_kmeans, "cluster_centers_"):
                # Use fine cluster centroids as input for coarse clustering
                centroids = self.fine_kmeans.cluster_centers_
                
                # Initialize KMeans for coarse clustering
                coarse_kmeans = KMeans(
                    n_clusters=num_coarse_clusters,
                    init="k-means++",
                    max_iter=500,
                    random_state=42
                )
                
                # Fit KMeans on centroids
                self.logger.info("Fitting coarse K-means on fine cluster centroids")
                coarse_kmeans.fit(centroids)
                
                # Map fine clusters to coarse clusters
                fine_to_coarse = coarse_kmeans.labels_
                
                # Map each sample to its coarse cluster
                coarse_cluster_labels = fine_to_coarse[fine_cluster_labels]
                
                # Store model
                self.coarse_kmeans = coarse_kmeans
            else:
                # Method 2: Direct clustering on image features
                # Convert features to numpy for sklearn
                features_np = features.numpy()
                
                # Initialize KMeans for coarse clustering
                coarse_kmeans = KMeans(
                    n_clusters=num_coarse_clusters,
                    init="k-means++",
                    max_iter=500,
                    random_state=42
                )
                
                # Fit KMeans
                self.logger.info("Fitting coarse K-means directly on features")
                coarse_kmeans.fit(features_np)
                
                # Get cluster assignments
                coarse_cluster_labels = coarse_kmeans.labels_
                
                # Store model
                self.coarse_kmeans = coarse_kmeans
            
            # Create mapping between old and new clusters if available
            cluster_mapping = {}
            if self.old_cluster_labels is not None:
                # For each old cluster, find which new cluster contains the most samples
                for old_idx in range(self.config.num_experts):
                    old_mask = (self.old_cluster_labels == old_idx)
                    if np.sum(old_mask) > 0:
                        new_clusters, counts = np.unique(
                            coarse_cluster_labels[old_mask], 
                            return_counts=True
                        )
                        # Map to new cluster with most overlap
                        cluster_mapping[old_idx] = new_clusters[np.argmax(counts)]
            
            # Save to cache
            try:
                with open(self.coarse_cluster_cache_file, "wb") as f:
                    pickle.dump({
                        "labels": coarse_cluster_labels,
                        "model": coarse_kmeans,
                        "mapping": cluster_mapping
                    }, f)
                self.logger.info(f"Saved coarse clusters to cache: {self.coarse_cluster_cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save coarse clusters to cache: {e}")
        else:
            coarse_cluster_labels = None
            
        # Broadcast to all processes
        if get_world_size() > 1:
            coarse_cluster_labels = broadcast_numpy_array(coarse_cluster_labels)
            
        return coarse_cluster_labels

    def get_clusters(self):
        """Get current cluster assignments"""
        return self.cluster_labels

    def perform_clustering(self, dataloader=None):
        """
        Extract features and perform clustering
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            Array of cluster assignments
        """
        if dataloader is None:
            self.logger.error("No dataloader provided for clustering")
            return None
            
        # Extract features
        features, _ = self.extract_features(dataloader)
        
        # Perform clustering
        cluster_labels = self.cluster_dataset(features)
        
        # Visualize clusters if main process and logging enabled
        if is_main_process() and hasattr(self.config, 'visualize_clusters') and self.config.visualize_clusters:
            try:
                # Plot cluster visualization
                fig, _ = visualize_embeddings(
                    features.numpy(), 
                    cluster_labels,
                    method='tsne' if features.size(0) < 10000 else 'pca'
                )
                
                # Save visualization
                import matplotlib.pyplot as plt
                os.makedirs("visualizations", exist_ok=True)
                fig.savefig(f"visualizations/clusters_{time.strftime('%Y%m%d-%H%M%S')}.png")
                plt.close(fig)
                
                self.logger.info("Saved cluster visualization to visualizations/")
            except Exception as e:
                self.logger.warning(f"Failed to visualize clusters: {str(e)}")
        
        return cluster_labels 