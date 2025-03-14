"""Clustering functionality for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from data.feature_extractor import DINOv2FeatureExtractor
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from torch.utils.data import DataLoader

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
    """Manages clustering operations for DDM with ownership of all clustering logic"""
    
    def __init__(self, local_rank=0, dataset_path=None, config=None):
        """
        Initialize the cluster manager
        
        Args:
            local_rank: Local process rank for distributed training
            dataset_path: Path to the dataset
            config: Configuration object
        """
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.dataset_path = dataset_path
        self.config = config
        
        # Initialize logger
        self.logger = setup_logger(name="ClusterManager", rank=local_rank)
        
        # Initialize vision encoder
        self._init_vision_encoder()
        
        # Initialize other attributes
        self.features = None
        self.cluster_labels = None
        self.kmeans = None
        self.old_cluster_labels = None
        
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
        Extract features from dataset images
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            Tensor of extracted features
        """
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
                    
                    # Extract features
                    batch_features = self.vision_encoder(images)
                    
                    # Store features and paths
                    features.append(batch_features.cpu())
                    image_paths.extend(paths)
                
                # Concatenate features
                if features:
                    features = torch.cat(features, dim=0)
                else:
                    features = torch.empty((0, self.vision_encoder.feature_dim))
                
                self.logger.info(f"Extracted features for {features.size(0)} images")
        
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
        Cluster features using K-means
        
        Args:
            features: Tensor of image features
            
        Returns:
            Array of cluster assignments
        """
        self.logger.info(f"Clustering {features.size(0)} images into {self.config.num_experts} clusters")
        
        # Save old labels if available
        if self.cluster_labels is not None:
            self.old_cluster_labels = self.cluster_labels.copy()
        
        # Only perform clustering on main process in distributed setting
        if get_rank() == 0:
            # Initialize K-means
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.config.num_experts,
                batch_size=min(1024, features.size(0)),
                init="k-means++",
                max_iter=300,
                random_state=42
            )
            
            # Convert features to numpy
            features_np = features.numpy()
            
            # Fit K-means
            self.kmeans.fit(features_np)
            
            # Get cluster assignments
            self.cluster_labels = self.kmeans.labels_
            
            # Log cluster distribution
            unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                self.logger.info(f"Cluster {label}: {count} images ({count / len(self.cluster_labels) * 100:.2f}%)")
        
        # Broadcast cluster labels in distributed setting
        if get_world_size() > 1:
            self.cluster_labels = self._broadcast_labels(self.cluster_labels)
            
        return self.cluster_labels

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
        
    def _broadcast_labels(self, labels):
        """
        Broadcast cluster labels from rank 0 to all processes
        
        Args:
            labels: Cluster labels array
            
        Returns:
            Broadcasted labels
        """
        # Broadcast labels using centralized distributed utility
        return broadcast_numpy_array(labels)

    def perform_reclustering(self, dataloader=None):
        """
        Re-cluster dataset and map old clusters to new ones
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            Tuple of (new_labels, cluster_mapping)
        """
        self.logger.info("Performing reclustering")
        
        # Store old labels
        old_labels = self.cluster_labels
        
        # Perform clustering
        new_labels = self.perform_clustering(dataloader)
        
        # Get mapping from old to new clusters
        cluster_mapping = {}
        
        # Only calculate mapping on main process
        if get_rank() == 0 and old_labels is not None:
            cluster_mapping = self.get_cluster_mapping(old_labels, new_labels)
            
            # Log cluster mapping
            self.logger.info("Cluster mapping:")
            for old_cluster, new_cluster in cluster_mapping.items():
                self.logger.info(f"Old cluster {old_cluster} -> New cluster {new_cluster}")
                
        # Broadcast mapping in distributed setting
        if get_world_size() > 1:
            cluster_mapping = broadcast_object(cluster_mapping)
            
        return new_labels, cluster_mapping

    def get_cluster_mapping(self, old_clusters, new_clusters):
        """
        Calculate mapping from old clusters to new clusters
        
        Args:
            old_clusters: Old cluster assignments
            new_clusters: New cluster assignments
            
        Returns:
            Dictionary mapping old cluster indices to new ones
        """
        # Create overlap matrix
        overlap = np.zeros((self.config.num_experts, self.config.num_experts), dtype=np.int32)
        
        # Calculate overlap between old and new clusters
        for i in range(len(old_clusters)):
            overlap[old_clusters[i], new_clusters[i]] += 1
            
        # For each old cluster, find the new cluster with maximum overlap
        mapping = {}
        for old_idx in range(self.config.num_experts):
            mapping[old_idx] = np.argmax(overlap[old_idx])
            
        # Handle collisions - some old clusters might map to the same new cluster
        used_new_clusters = set(mapping.values())
        unused_new_clusters = set(range(self.config.num_experts)) - used_new_clusters
        
        # If we have collisions, find unused new clusters and assign them to collided old clusters
        if len(used_new_clusters) < len(mapping):
            # Find old clusters that map to the same new cluster
            collision_groups = {}
            for old, new in mapping.items():
                if new not in collision_groups:
                    collision_groups[new] = []
                collision_groups[new].append(old)
                
            # Handle collisions
            for new, old_groups in collision_groups.items():
                if len(old_groups) > 1:
                    # Keep the mapping for the old cluster with highest overlap
                    overlaps = [overlap[old, new] for old in old_groups]
                    max_idx = np.argmax(overlaps)
                    keep_old = old_groups[max_idx]
                    
                    # Reassign other old clusters to unused new clusters
                    for i, old in enumerate(old_groups):
                        if old != keep_old and unused_new_clusters:
                            new_target = unused_new_clusters.pop()
                            mapping[old] = new_target
                            
        return mapping 