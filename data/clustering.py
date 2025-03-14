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
        
        # Feature cache file - use chunked format for large datasets
        self.feature_chunks_dir = os.path.join(self.cache_dir, f"feature_chunks_dino_{config.dino_size}")
        os.makedirs(self.feature_chunks_dir, exist_ok=True)
        
        self.feature_index_file = os.path.join(
            self.cache_dir, 
            f"features_index_dino_{config.dino_size}.pkl"
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
        
        # New: Track cluster statistics
        self.cluster_stats = {}
        
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
        Extract features from dataset images with efficient chunked caching
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            Tensor of extracted features
        """
        # Check if chunked features exist
        if is_main_process() and os.path.exists(self.feature_index_file):
            self.logger.info(f"Loading feature index from: {self.feature_index_file}")
            try:
                with open(self.feature_index_file, "rb") as f:
                    index_data = pickle.load(f)
                    num_chunks = index_data["num_chunks"]
                    total_features = index_data["total_features"]
                    feature_dim = index_data["feature_dim"]
                    image_paths = index_data["paths"]
                    
                    # Verify chunks exist
                    chunks_exist = True
                    for i in range(num_chunks):
                        chunk_file = os.path.join(self.feature_chunks_dir, f"chunk_{i}.npy")
                        if not os.path.exists(chunk_file):
                            chunks_exist = False
                            break
                    
                    if chunks_exist:
                        self.logger.info(f"Found {num_chunks} feature chunks with {total_features} total features")
                        
                        # Load and concatenate chunks efficiently
                        features_list = []
                        for i in range(num_chunks):
                            chunk_file = os.path.join(self.feature_chunks_dir, f"chunk_{i}.npy")
                            chunk = np.load(chunk_file, mmap_mode='r')  # Memory-mapped loading
                            features_list.append(chunk)
                            
                        # Concatenate features
                        features = np.concatenate(features_list, axis=0)
                        
                        # Verify loaded features shape
                        if features.shape[0] > 0:
                            self.logger.info(f"Loaded {features.shape[0]} cached features")
                            return torch.from_numpy(features), image_paths
                        else:
                            self.logger.warning("Cached features are empty, re-extracting")
                    else:
                        self.logger.warning("Some feature chunks are missing, re-extracting")
            except Exception as e:
                self.logger.warning(f"Failed to load cached features: {e}. Extracting new features.")
        
        # Extract features in chunks if not loaded from cache
        self.logger.info("Extracting features from dataset")
        features_list = []
        image_paths = []
        
        # Only extract on main process in distributed setting
        if get_rank() == 0:
            # Determine chunk size and maximum memory usage
            max_chunk_size = getattr(self.config, "feature_chunk_size", 10000)
            chunk_idx = 0
            current_chunk = []
            current_paths = []
            
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
                    current_chunk.append(batch_features)
                    current_paths.extend(paths)
                    
                    # Check if we should save this chunk
                    if len(current_chunk) * sub_batch_size >= max_chunk_size:
                        # Concatenate features in current chunk
                        chunk_tensor = torch.cat(current_chunk, dim=0)
                        features_list.append(chunk_tensor)
                        
                        # Save chunk to disk
                        chunk_file = os.path.join(self.feature_chunks_dir, f"chunk_{chunk_idx}.npy")
                        np.save(chunk_file, chunk_tensor.numpy())
                        
                        self.logger.info(f"Saved feature chunk {chunk_idx} with {chunk_tensor.size(0)} features")
                        
                        # Update chunk index and reset current chunk
                        chunk_idx += 1
                        current_chunk = []
                
                # Save final chunk if not empty
                if current_chunk:
                    # Concatenate features in current chunk
                    chunk_tensor = torch.cat(current_chunk, dim=0)
                    features_list.append(chunk_tensor)
                    
                    # Save chunk to disk
                    chunk_file = os.path.join(self.feature_chunks_dir, f"chunk_{chunk_idx}.npy")
                    np.save(chunk_file, chunk_tensor.numpy())
                    
                    self.logger.info(f"Saved final feature chunk {chunk_idx} with {chunk_tensor.size(0)} features")
                
                # Concatenate all features
                if features_list:
                    features = torch.cat(features_list, dim=0)
                    
                    # Save feature index
                    try:
                        with open(self.feature_index_file, "wb") as f:
                            pickle.dump({
                                "num_chunks": chunk_idx + 1,
                                "total_features": features.size(0),
                                "feature_dim": features.size(1),
                                "paths": current_paths
                            }, f)
                        self.logger.info(f"Saved feature index to: {self.feature_index_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save feature index: {e}")
                else:
                    features = torch.empty((0, self.vision_encoder.feature_dim))
                
                self.logger.info(f"Extracted features for {features.size(0)} images")
                image_paths = current_paths
        
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
        """Create fine-grained clusters with efficient MiniBatch K-means"""
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
        
        # Run clustering on main process only
        if get_rank() == 0:
            self.logger.info(f"Creating {self.config.num_experts * 8} fine-grained clusters")
            try:
                # Use MiniBatchKMeans for large datasets for efficiency (paper recommends this)
                # Very large datasets benefit from this approach
                if len(features) > 100000:
                    n_clusters = self.config.num_experts * 8
                    
                    # Initialize with scalable MiniBatchKMeans
                    self.fine_kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        batch_size=4096,  # Large batch for faster convergence
                        init='k-means++',
                        max_iter=100,
                        random_state=42
                    )
                    
                    # Fit with progress bar
                    batch_size = 4096
                    n_batches = int(np.ceil(len(features) / batch_size))
                    
                    with tqdm(total=n_batches, desc="MiniBatchKMeans") as pbar:
                        for i in range(n_batches):
                            start = i * batch_size
                            end = min((i + 1) * batch_size, len(features))
                            self.fine_kmeans.partial_fit(features[start:end])
                            pbar.update(1)
                else:
                    # For smaller datasets, use standard K-means
                    self.fine_kmeans = KMeans(
                        n_clusters=self.config.num_experts * 8,
                        init='k-means++',
                        n_init=10,
                        random_state=42
                    )
                    self.fine_kmeans.fit(features)
                
                # Get cluster assignments
                fine_cluster_labels = self.fine_kmeans.predict(features)
                
                # Save to cache for future use
                try:
                    with open(self.fine_cluster_cache_file, "wb") as f:
                        pickle.dump({
                            "labels": fine_cluster_labels,
                            "model": self.fine_kmeans,
                            "num_clusters": self.config.num_experts * 8
                        }, f)
                    self.logger.info(f"Saved fine clusters to cache: {self.fine_cluster_cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save fine clusters to cache: {e}")
            except Exception as e:
                self.logger.error(f"Error in fine clustering: {e}")
                # Fallback to random assignments
                fine_cluster_labels = np.random.randint(0, self.config.num_experts * 8, size=len(features))
                self.logger.warning("Using random fine cluster assignments due to error")
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

    def update_clusters_incremental(self, new_features, old_features_ratio=0.9):
        """
        Perform incremental clustering update with new features
        
        Args:
            new_features: New features to incorporate [N, D]
            old_features_ratio: Ratio of old features to keep (0.9 means 90% old, 10% new)
            
        Returns:
            Updated cluster assignments
        """
        if self.features is None or len(self.features) == 0:
            self.logger.warning("No existing features, cannot update incrementally")
            return None
            
        if get_rank() == 0:
            self.logger.info(f"Updating clusters with {len(new_features)} new features")
            
            # Combine old and new features with proper ratio
            old_keep_count = int(len(self.features) * old_features_ratio)
            
            # Randomly select features to keep
            keep_indices = np.random.choice(
                len(self.features), 
                size=old_keep_count, 
                replace=False
            )
            
            # Combine old (selected) and new features
            combined_features = torch.cat([
                self.features[keep_indices],
                new_features
            ], dim=0)
            
            self.logger.info(f"Combined {old_keep_count} old features with {len(new_features)} new features")
            
            # Update internal feature store
            self.features = combined_features
            
            # Perform clustering on updated features
            cluster_labels = self.cluster_dataset(combined_features)
            
            return cluster_labels
        else:
            return None

    def get_clusters(self):
        """Get current cluster assignments"""
        return self.cluster_labels

    def visualize_cluster_samples(self, dataloader, output_dir="cluster_samples", samples_per_cluster=10):
        """
        Visualize sample images from each cluster
        
        Args:
            dataloader: DataLoader for dataset
            output_dir: Directory to save visualizations
            samples_per_cluster: Number of sample images per cluster
        """
        if not is_main_process() or self.cluster_labels is None:
            return
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Visualizing cluster samples to {output_dir}")
            
            # Collect sample indices for each cluster
            cluster_samples = {}
            for i, cluster in enumerate(self.cluster_labels):
                if cluster not in cluster_samples:
                    cluster_samples[cluster] = []
                if len(cluster_samples[cluster]) < samples_per_cluster:
                    cluster_samples[cluster].append(i)
            
            # Create visualization for each cluster
            for cluster, indices in cluster_samples.items():
                sample_images = []
                sample_count = 0
                
                # Collect actual images
                for batch in dataloader:
                    if isinstance(batch, dict):
                        images = batch["image"]
                        idx_offset = batch.get("index", list(range(len(images))))
                    else:
                        images = batch
                        idx_offset = list(range(len(images)))
                    
                    for i, idx in enumerate(idx_offset):
                        if idx in indices and sample_count < samples_per_cluster:
                            sample_images.append(images[i])
                            sample_count += 1
                    
                    if sample_count >= samples_per_cluster:
                        break
                
                # Create and save grid
                if sample_images:
                    grid = create_image_grid(sample_images, nrow=min(5, len(sample_images)))
                    grid_pil = tensor_to_pil(grid)
                    grid_pil.save(os.path.join(output_dir, f"cluster_{cluster}.png"))
            
            self.logger.info(f"Saved cluster visualizations to {output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to visualize cluster samples: {e}")

    def perform_clustering(self, dataloader=None):
        """
        Extract features and perform clustering with improved efficiency
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            Array of cluster assignments
        """
        if dataloader is None:
            self.logger.error("No dataloader provided for clustering")
            return None
            
        # Extract features with chunking for memory efficiency
        features, _ = self.extract_features(dataloader)
        
        # Perform clustering
        cluster_labels = self.cluster_dataset(features)
        
        # Analyze and log cluster statistics
        if is_main_process():
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            self.cluster_stats["distribution"] = {c: int(count) for c, count in zip(unique_clusters, counts)}
            self.cluster_stats["num_clusters"] = len(unique_clusters)
            self.cluster_stats["largest_cluster"] = int(max(counts))
            self.cluster_stats["smallest_cluster"] = int(min(counts))
            
            # Log detailed statistics
            self.logger.info(f"Clustering Statistics:")
            self.logger.info(f"  Number of clusters: {self.cluster_stats['num_clusters']}")
            self.logger.info(f"  Largest cluster: {self.cluster_stats['largest_cluster']} samples")
            self.logger.info(f"  Smallest cluster: {self.cluster_stats['smallest_cluster']} samples")
            
            # Compute cluster balance metric (coefficient of variation)
            cv = np.std(counts) / np.mean(counts)
            self.cluster_stats["balance_cv"] = float(cv)
            self.logger.info(f"  Cluster balance (CV): {cv:.4f} (lower is better)")
            
            # Save statistics
            try:
                stats_file = os.path.join(self.cache_dir, "cluster_stats.pkl")
                with open(stats_file, "wb") as f:
                    pickle.dump(self.cluster_stats, f)
                self.logger.info(f"Saved cluster statistics to {stats_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save cluster statistics: {e}")
        
        # Visualize clusters if main process and logging enabled
        if is_main_process() and hasattr(self.config, 'visualize_clusters') and self.config.visualize_clusters:
            try:
                # Memory-efficient visualization using PCA
                from sklearn.decomposition import PCA
                
                # Always use PCA first to reduce dimensionality
                pca = PCA(n_components=min(50, features.size(1)))
                reduced_features = pca.fit_transform(features.numpy())
                
                if len(reduced_features) < 10000 and hasattr(self.config, 'use_tsne') and self.config.use_tsne:
                    # Further reduce with t-SNE for smaller datasets
                    from sklearn.manifold import TSNE
                    vis_features = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(reduced_features)
                    method = 'tsne'
                else:
                    # Just use first 2 PCA components for large datasets
                    vis_features = reduced_features[:, :2]
                    method = 'pca'
                
                # Plot visualization
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 10))
                scatter = ax.scatter(vis_features[:, 0], vis_features[:, 1], 
                                    c=cluster_labels, cmap='tab20', s=2, alpha=0.5)
                ax.set_title(f"Cluster Visualization ({method.upper()})")
                fig.colorbar(scatter, ax=ax, label="Cluster ID")
                
                # Save visualization
                os.makedirs("visualizations", exist_ok=True)
                viz_path = f"visualizations/clusters_{time.strftime('%Y%m%d-%H%M%S')}.png"
                fig.savefig(viz_path)
                plt.close(fig)
                
                self.logger.info(f"Saved cluster visualization to {viz_path}")
                
                # Also save sample images from each cluster for better interpretability
                self.visualize_cluster_samples(dataloader)
            except Exception as e:
                self.logger.warning(f"Failed to visualize clusters: {str(e)}")
        
        return cluster_labels 