"""Clustering functionality for Decentralized Diffusion Models."""

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import logging

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
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
        
        # Set to evaluation mode and freeze parameters
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
            
        model = model.to(self.device)
        return model
    
    def extract_features(self, dataloader):
        """Extract DINO features from dataset"""
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch.to(self.device)  # Direct tensor input
                features = self.dino(images).cpu().numpy()
                all_features.append(features)
                
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

        print(f"Running fine-grained clustering ({n_clusters_fine} clusters)...")
        fine_labels = self.fine_clusterer.fit_predict(features)
        
        # Create centroids for each fine-grained cluster
        fine_centroids = np.zeros((n_clusters_fine, features.shape[1]))
        for i in range(n_clusters_fine):
            mask = fine_labels == i
            if mask.sum() > 0:  # Handle empty clusters
                fine_centroids[i] = features[mask].mean(axis=0)
        
        print(f"Running coarse clustering ({n_clusters_coarse} clusters)...")
        coarse_labels = self.coarse_clusterer.fit_predict(fine_centroids)
        
        # Map fine clusters to coarse clusters
        sample_to_coarse = coarse_labels[fine_labels]
        
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