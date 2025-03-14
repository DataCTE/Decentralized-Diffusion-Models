"""Clustering functionality for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import time
import logging
from utils.feature_extractor import DINOv2FeatureExtractor
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.distributed import broadcast_numpy_array, broadcast_object

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
        
        # Initialize vision encoder
        self.feature_extractor = self._init_vision_encoder()
        
        # Storage for cluster labels
        self.cluster_labels = None
        self.n_clusters = config.num_experts if config else 8
        
        # Additional state for reclustering
        self.features = None
        self.prev_cluster_labels = None
        
        if self.local_rank == 0:
            logger.info(f"ClusterManager initialized on device cuda:{self.local_rank}")
        
    def _init_vision_encoder(self):
        """Initialize DINOv2 for feature extraction"""
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
        
        # Set to evaluation mode and freeze parameters
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
            
        model = model.to(self.device)
        if self.local_rank == 0:
            logger.info(f"DINOv2 encoder initialized on device cuda:{self.local_rank}")
        return model
    
    def extract_features(self, dataloader):
        """Extract DINO features from dataset"""
        all_features = []
        total_batches = len(dataloader)
        start_time = time.time()
        processed_images = 0
        total_images = len(dataloader.dataset)
        
        if self.local_rank == 0:
            logger.info(f"Starting feature extraction on {total_images:,} images across {dist.get_world_size()} GPUs")
            logger.info(f"Process will take approximately {total_images * 0.01 / dist.get_world_size():.1f} minutes (estimate)")
        
        log_interval = max(1, total_batches // 20)  # Log 20 times during extraction
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Log progress periodically from main process
                if self.local_rank == 0 and (batch_idx % log_interval == 0 or batch_idx == total_batches - 1):
                    elapsed = time.time() - start_time
                    images_per_sec = processed_images / max(1, elapsed)
                    progress = batch_idx / max(1, total_batches) * 100
                    eta = (total_batches - batch_idx) / max(1, batch_idx) * elapsed if batch_idx > 0 else 0
                    logger.info(f"Feature extraction: {progress:.1f}% complete | "
                                f"Batch {batch_idx+1}/{total_batches} | "
                                f"Images: {processed_images:,}/{total_images:,} | "
                                f"Speed: {images_per_sec:.1f} img/s | "
                                f"ETA: {eta/60:.1f} min")
                
                # Process batch
                try:
                    images = batch.to(self.device)  # Direct tensor input
                    features = self.feature_extractor(images).cpu().numpy()
                    all_features.append(features)
                    processed_images += len(images)
                except Exception as e:
                    logger.error(f"Error in feature extraction at batch {batch_idx}: {str(e)}")
                    continue
        
        # Final stats
        if self.local_rank == 0:
            total_time = time.time() - start_time
            logger.info(f"Feature extraction complete: {processed_images:,} images processed in {total_time/60:.1f} minutes "
                       f"({processed_images/total_time:.1f} images/sec)")
                
        # Synchronize before continuing to ensure all processes are complete
        dist.barrier()
        
        return np.concatenate(all_features)
    
    def cluster_dataset(self, features):
        """Two-stage clustering as described in the paper"""
        # Dynamically adjust clusters based on sample size
        n_samples = features.shape[0]
        n_clusters_fine = min(n_samples, 1024)  # Don't exceed available samples
        n_clusters_coarse = min(n_clusters_fine, 8)  # Ensure coarse <= fine
        
        # Reinitialize clusterers with adjusted counts
        self.fine_clusterer = MiniBatchKMeans(n_clusters=n_clusters_fine)
        self.coarse_clusterer = MiniBatchKMeans(n_clusters=n_clusters_coarse)

        if self.local_rank == 0:
            logger.info(f"Running fine-grained clustering ({n_clusters_fine} clusters)...")
            start_time = time.time()
            
        fine_labels = self.fine_clusterer.fit_predict(features)
        
        if self.local_rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Fine clustering completed in {elapsed:.1f} seconds")
        
        # Create centroids for each fine-grained cluster
        fine_centroids = np.zeros((n_clusters_fine, features.shape[1]))
        for i in range(n_clusters_fine):
            mask = fine_labels == i
            if mask.sum() > 0:  # Handle empty clusters
                fine_centroids[i] = features[mask].mean(axis=0)
        
        if self.local_rank == 0:
            logger.info(f"Running coarse clustering ({n_clusters_coarse} clusters)...")
            start_time = time.time()
            
        coarse_labels = self.coarse_clusterer.fit_predict(fine_centroids)
        
        if self.local_rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Coarse clustering completed in {elapsed:.1f} seconds")
        
        # Map fine clusters to coarse clusters
        sample_to_coarse = coarse_labels[fine_labels]
        
        if self.local_rank == 0:
            cluster_distribution = np.bincount(sample_to_coarse, minlength=n_clusters_coarse)
            logger.info(f"Final cluster distribution: {cluster_distribution}")
            logger.info(f"Clusters assigned to {len(sample_to_coarse):,} images")
        
        return sample_to_coarse 

    def get_clusters(self):
        """Return the current cluster labels"""
        return self.cluster_labels

    def perform_clustering(self, dataloader=None):
        """
        Perform initial clustering of the dataset
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            Cluster labels for the dataset
        """
        # Extract features if needed
        if dataloader is not None:
            self.features = self.extract_features(dataloader)
        
        # Perform clustering
        if self.local_rank == 0 and self.features is not None:
            logger.info(f"Clustering {self.features.shape[0]} samples into {self.n_clusters} clusters")
            self.cluster_labels = self.cluster_dataset(self.features)
            logger.info(f"Clustering complete - {len(self.cluster_labels):,} samples assigned to clusters")
        
        # Broadcast cluster labels to all processes
        self.cluster_labels = self._broadcast_labels(self.cluster_labels)
        
        return self.cluster_labels
    
    def _broadcast_labels(self, labels):
        """
        Broadcasts clustering labels from rank 0 to all processes
        
        Args:
            labels: Cluster labels (numpy array) on rank 0
            
        Returns:
            Broadcasted labels on all ranks
        """
        # Use the centralized broadcast utility
        return broadcast_numpy_array(labels, src_rank=0)
            
    def perform_reclustering(self, dataloader=None):
        """Perform reclustering during training to adapt to distribution shifts"""
        if self.local_rank == 0:
            logger.info("Starting dynamic reclustering process")
            
        if dataloader is None:
            if self.dataset_path is None or self.config is None:
                raise ValueError("dataset_path and config must be provided if dataloader is None")
                
            # Import here to avoid circular imports
            from data.dataset import FeatureDataset
            
            feature_dataset = FeatureDataset(self.dataset_path, self.config)
            dataloader = DataLoader(
                feature_dataset,
                batch_size=self.config.feature_batch_size,
                num_workers=self.config.feature_workers,
                pin_memory=True
            )
            
        # Extract new features
        if self.local_rank == 0:
            logger.info("Extracting updated features for reclustering")
            
        features = self.extract_features(dataloader)
        
        # Save old clusters for reference
        old_clusters = self.cluster_labels.copy() if self.cluster_labels is not None else None
        
        # Run clustering with extracted features instead of re-extracting them
        if self.local_rank == 0:
            logger.info("Clustering extracted features")
        
        # Instead of calling perform_clustering (which would extract features again),
        # we directly use the clustering logic with our already extracted features
        n_samples = features.shape[0]
        n_fine_clusters = min(1024, n_samples // 100)
        n_coarse_clusters = min(8, n_fine_clusters)
        
        if self.local_rank == 0:
            logger.info(f"Reclustering {n_samples:,} samples into {n_fine_clusters} fine clusters")
        
        # Initialize fresh clusterers
        self.fine_clusterer = MiniBatchKMeans(n_clusters=n_fine_clusters)
        self.coarse_clusterer = MiniBatchKMeans(n_clusters=n_coarse_clusters)
        
        # Fit fine clusters
        start_time = time.time()
        if self.local_rank == 0:
            logger.info("Performing fine-grained clustering for reclustering")
            with tqdm(total=100, desc="Fine clustering", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                self.fine_clusterer.fit(features)
                pbar.update(100)
        else:
            self.fine_clusterer.fit(features)
        
        if self.local_rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Fine clustering completed in {elapsed:.1f} seconds")
        
        # Get fine centroids
        fine_centroids = self.fine_clusterer.cluster_centers_
        
        # Perform coarse clustering
        start_time = time.time()
        if self.local_rank == 0:
            logger.info(f"Grouping {len(fine_centroids)} centroids into {n_coarse_clusters} coarse clusters")
            with tqdm(total=100, desc="Coarse clustering", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                self.coarse_clusterer.fit(fine_centroids)
                pbar.update(100)
        else:
            self.coarse_clusterer.fit(fine_centroids)
        
        if self.local_rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Coarse clustering completed in {elapsed:.1f} seconds")
        
        # Map samples to coarse clusters
        if self.local_rank == 0:
            logger.info("Assigning new cluster labels")
            fine_labels = []
            with tqdm(total=len(features), desc="Cluster assignment", unit="img") as pbar:
                for i in range(0, len(features), 10000):
                    batch = features[i:i+10000]
                    fine_labels.extend(self.fine_clusterer.predict(batch))
                    pbar.update(len(batch))
        
            self.cluster_labels = self.coarse_clusterer.labels_[fine_labels]
        else:
            # Just predict all at once for non-main processes
            fine_labels = self.fine_clusterer.predict(features)
            self.cluster_labels = self.coarse_clusterer.labels_[fine_labels]
        
        # Broadcast labels to all processes
        self.cluster_labels = self._broadcast_labels(self.cluster_labels)
        
        if self.local_rank == 0:
            cluster_distribution = np.bincount(self.cluster_labels, minlength=n_coarse_clusters)
            logger.info(f"Reclustering result - cluster distribution: {cluster_distribution}")
            logger.info(f"Reclustering complete - {len(self.cluster_labels):,} images assigned to clusters")
        
        # Return both old and new cluster assignments
        return old_clusters, self.cluster_labels
        
    def get_cluster_mapping(self, old_clusters, new_clusters):
        """Map old cluster IDs to new cluster IDs based on maximum overlap"""
        if old_clusters is None or new_clusters is None:
            return {}
            
        # Create overlap matrix
        n_old = max(old_clusters) + 1
        n_new = max(new_clusters) + 1
        overlap = np.zeros((n_old, n_new), dtype=int)
        
        # Count overlaps
        for old, new in zip(old_clusters, new_clusters):
            overlap[old, new] += 1
            
        # Create mapping from old to new
        mapping = {}
        for old_idx in range(n_old):
            if old_idx in np.unique(old_clusters):
                # Get the new cluster with maximum overlap
                new_idx = np.argmax(overlap[old_idx])
                mapping[old_idx] = new_idx
                
        if self.local_rank == 0:
            logger.info(f"Created cluster mapping from {n_old} old clusters to {n_new} new clusters")
            logger.info(f"Mapping: {mapping}")
            
        return mapping 