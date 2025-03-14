"""Clustering functionality for Decentralized Diffusion Models."""

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import logging
import time
import torch.distributed as dist

logger = logging.getLogger(__name__)

class ClusterManager:
    """Handles dataset clustering following the paper's approach"""
    def __init__(self, local_rank=0):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Initialize DINO for feature extraction
        self.dino = self._init_vision_encoder()
        
        # Clustering components
        self.fine_clusterer = MiniBatchKMeans(n_clusters=1024)
        self.coarse_clusterer = MiniBatchKMeans(n_clusters=8)
        self.cluster_labels = None  # Add cluster storage
        
    def _init_vision_encoder(self):
        """Initialize DINOv2 for feature extraction"""
        if self.local_rank == 0:
            logger.info("Initializing DINOv2 vision encoder")
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
                    features = self.dino(images).cpu().numpy()
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
        """Retrieve cluster assignments"""
        if self.cluster_labels is None:
            raise RuntimeError("Clusters not initialized. Call perform_clustering() first")
        return self.cluster_labels

    def perform_clustering(self, dataloader):
        """Paper's two-stage clustering procedure from Section 3.2"""
        # Stage 1: Extract features and create fine clusters
        if self.local_rank == 0:
            logger.info("Starting DINOv2 feature extraction")
        features = self.extract_features(dataloader)
        
        # Paper's dynamic cluster count based on dataset size
        n_samples = features.shape[0]
        n_fine_clusters = min(1024, n_samples // 100)
        n_coarse_clusters = 8

        # Initialize fresh clusterers
        self.fine_clusterer = MiniBatchKMeans(n_clusters=n_fine_clusters)
        self.coarse_clusterer = MiniBatchKMeans(n_clusters=n_coarse_clusters)

        # Fit fine clusters with progress
        if self.local_rank == 0:
            logger.info(f"Clustering {n_samples:,} samples into {n_fine_clusters} fine clusters")
            with tqdm(total=100, desc="Fine clustering", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                self.fine_clusterer.fit(features)
                pbar.update(100)
        
        # Stage 2: Cluster fine centroids into coarse groups
        fine_centroids = self.fine_clusterer.cluster_centers_
        if self.local_rank == 0:
            logger.info(f"Grouping {len(fine_centroids)} centroids into {n_coarse_clusters} coarse clusters")
            with tqdm(total=100, desc="Coarse clustering", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                self.coarse_clusterer.fit(fine_centroids)
                pbar.update(100)

        # Map samples to coarse clusters with progress
        if self.local_rank == 0:
            logger.info("Assigning final cluster labels")
            fine_labels = []
            with tqdm(total=len(features), desc="Cluster assignment", unit="img") as pbar:
                for i in range(0, len(features), 10000):
                    batch = features[i:i+10000]
                    fine_labels.extend(self.fine_clusterer.predict(batch))
                    pbar.update(len(batch))
            
            self.cluster_labels = self.coarse_clusterer.labels_[fine_labels]
        else:
            self.cluster_labels = None
            
        # Broadcast labels to all processes
        self.cluster_labels = self._broadcast_labels(self.cluster_labels)
        
        return self.cluster_labels 
        
    def _broadcast_labels(self, labels):
        """Broadcast cluster labels from rank 0 to all processes"""
        if self.local_rank == 0:
            logger.info(f"Broadcasting cluster labels to all processes")
        
        # Synchronize before broadcasting
        dist.barrier()
        
        # Handle the case where labels is None
        if dist.get_world_size() > 1:
            # First broadcast the size of the tensor
            if self.local_rank == 0:
                if labels is None:
                    size = torch.tensor([0], dtype=torch.long, device=self.device)
                else:
                    size = torch.tensor([len(labels)], dtype=torch.long, device=self.device)
                    logger.info(f"Broadcasting {size.item():,} cluster labels to all processes")
            else:
                size = torch.tensor([0], dtype=torch.long, device=self.device)
            
            # Broadcast size first
            dist.broadcast(size, 0)
            tensor_size = size.item()
            
            # If size is 0, return empty array
            if tensor_size == 0:
                if self.local_rank == 0:
                    logger.warning("No cluster labels to broadcast")
                return np.array([], dtype=np.int64)
            
            # Create tensor of the right size
            if self.local_rank == 0:
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
                logger.info(f"Prepared labels tensor with shape {labels_tensor.shape}")
            else:
                labels_tensor = torch.empty(tensor_size, dtype=torch.long, device=self.device)
            
            # Broadcast the actual tensor
            dist.broadcast(labels_tensor, 0)
            
            if self.local_rank == 0:
                logger.info(f"Cluster labels broadcast complete")
            
            return labels_tensor.cpu().numpy()
        else:
            # Single process case
            if self.local_rank == 0:
                logger.info(f"Single process mode - no broadcasting needed")
            return labels if labels is not None else np.array([], dtype=np.int64) 