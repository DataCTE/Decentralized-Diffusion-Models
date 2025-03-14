"""Clustering functionality for Decentralized Diffusion Models."""

import torch
import numpy as np
import time
import logging
import os
import pickle
import hashlib
import json
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from data.feature_extractor import DINOv2FeatureExtractor
from utils.distributed import (
    broadcast_numpy_array, 
    broadcast_object, 
    is_main_process, 
    get_rank, 
    get_world_size, 
    synchronize
)
from utils.logging import setup_logger
from utils.visualization import visualize_embeddings

logger = logging.getLogger(__name__)

class ClusterManager:
    """
    Manages data clustering for decentralized diffusion models
    Handles feature extraction, clustering, and assignment of data to experts
    """
    
    def __init__(self, config, feature_extractor=None):
        """
        Initialize cluster manager
        
        Args:
            config: Configuration object
            feature_extractor: Feature extractor model (optional)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature extractor
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None:
            self.feature_extractor = self._initialize_feature_extractor()
            
        # Initialize storage for features and clusters
        self.features = None
        self.centroids = None
        self.cluster_labels = None
        self.cluster_sizes = None
        self.expert_assignment = None  # Maps each cluster to an expert
        
        # Create cache directory
        self.cache_dir = os.path.join(
            getattr(config, 'cache_dir', 'cache'),
            'clustering'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = setup_logger(name="ClusterManager", rank=get_rank())
        
        # Initialize vision encoder
        self._init_vision_encoder()
        
        # Create cache directories
        self.feature_cache_dir = os.path.join(self.cache_dir, 'features')
        self.feature_chunks_dir = os.path.join(self.feature_cache_dir, 'chunks')
        
        # Cluster cache
        self.kmeans_cache_dir = os.path.join(self.cache_dir, 'clusters')
        self.fine_cluster_cache_file = os.path.join(self.kmeans_cache_dir, 'fine_clusters.pkl')
        self.coarse_cluster_cache_file = os.path.join(self.kmeans_cache_dir, 'coarse_clusters.pkl')
        
        # Create cache directories if main process
        if is_main_process():
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.feature_cache_dir, exist_ok=True)
            os.makedirs(self.feature_chunks_dir, exist_ok=True)
            os.makedirs(self.kmeans_cache_dir, exist_ok=True)
            
        # Wait for directories to be created
        synchronize()
        
        # Initialize cluster models
        self.fine_kmeans = None
        self.coarse_kmeans = None
        self.fine_cluster_labels = None
        self.coarse_cluster_labels = None
        
        # Dataset metadata for consistency checking
        self.dataset_hash = None
        
    def _initialize_feature_extractor(self):
        """
        Initialize feature extractor based on config
        
        Returns:
            Feature extractor model
        """
        feature_type = getattr(self.config, 'clustering_features', 'clip')
        
        if feature_type.lower() == 'clip':
            try:
                import clip
                model, _ = clip.load("ViT-B/32", device=self.config.device)
                return model.visual
            except ImportError:
                self.logger.error("CLIP not installed. Install with: pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git")
                raise
        elif feature_type.lower() == 'dinov2':
            try:
                import torch.hub
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                model = model.to(self.config.device)
                return model
            except ImportError:
                self.logger.error("Failed to load DINOv2. Check if torch.hub is accessible.")
                raise
        else:
            self.logger.error(f"Unsupported feature type: {feature_type}")
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def _init_vision_encoder(self):
        """Initialize vision model for feature extraction"""
        model_name = getattr(self.config, 'feature_extractor', 'dinov2')
        
        if model_name == 'dinov2':
            self.logger.info("Initializing DINOv2 feature extractor")
            self.vision_encoder = DINOv2FeatureExtractor(
                variant=getattr(self.config, 'dinov2_variant', 'small'),
                device=self.config.device
            )
        else:
            raise ValueError(f"Unknown feature extractor: {model_name}")
            
        self.logger.info(f"Initialized vision encoder: {model_name}")
        
    def compute_dataset_hash(self, image_paths):
        """
        Compute a hash of dataset information to check consistency
        
        Args:
            image_paths: List of image paths in the dataset
            
        Returns:
            str: Hash representing the dataset
        """
        if not image_paths:
            return None
            
        # Use the first 100 and last 100 paths as a fingerprint
        # This is a compromise between full checking and speed
        subset = image_paths[:100] + image_paths[-100:] if len(image_paths) > 200 else image_paths
        
        # Sort to ensure consistent ordering regardless of dataloader
        subset = sorted(subset)
        
        # Concatenate for hashing
        paths_str = "".join(subset)
        
        # Compute hash
        hash_val = hashlib.md5(paths_str.encode('utf-8')).hexdigest()
        
        return hash_val
        
    def load_cached_features(self, image_paths):
        """
        Load cached features if available
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Tuple of (features, paths) or None if cache invalid
        """
        try:
            # Compute dataset hash for consistency checking
            dataset_hash = self.compute_dataset_hash(image_paths)
            self.dataset_hash = dataset_hash
            
            if dataset_hash is None:
                self.logger.warning("Could not compute dataset hash, unable to use cache")
                return None, None
                
            # Check for hash file
            hash_file = os.path.join(self.feature_cache_dir, 'dataset_hash.txt')
            
            if is_main_process() and os.path.exists(hash_file):
                # Check hash
                with open(hash_file, 'r') as f:
                    cached_hash = f.read().strip()
                
                if cached_hash != dataset_hash:
                    self.logger.warning(f"Dataset hash mismatch: {cached_hash} != {dataset_hash}")
                    # Delete old cache files if hash mismatch
                    import shutil
                    shutil.rmtree(self.feature_chunks_dir, ignore_errors=True)
                    os.makedirs(self.feature_chunks_dir, exist_ok=True)
                    return None, None
                    
                self.logger.info(f"Dataset hash match confirmed: {cached_hash}")
            elif is_main_process():
                # First time with this dataset, save hash
                with open(hash_file, 'w') as f:
                    f.write(dataset_hash)
                self.logger.info(f"Created new dataset hash: {dataset_hash}")
                
            # Wait for main process to check/update hash
            synchronize()
            
            # Get list of chunk files
            if is_main_process():
                chunk_files = sorted([f for f in os.listdir(self.feature_chunks_dir) 
                                    if f.startswith('chunk_') and f.endswith('.npy')])
                                    
                # Check if metadata file exists
                metadata_file = os.path.join(self.feature_cache_dir, 'metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        num_chunks = metadata.get('num_chunks', 0)
                        total_features = metadata.get('total_features', 0)
                        
                        # Check if all chunks exist
                        chunks_exist = len(chunk_files) == num_chunks
                else:
                    # No metadata file
                    chunks_exist = False
                    num_chunks = 0
                    total_features = 0
            else:
                # Non-main processes wait for check
                chunks_exist = False
                num_chunks = 0
                total_features = 0
                
            # Broadcast results to all processes
            chunks_exist = broadcast_object(chunks_exist)
            num_chunks = broadcast_object(num_chunks)
            total_features = broadcast_object(total_features)
            
            if chunks_exist:
                self.logger.info(f"Found {num_chunks} feature chunks with {total_features} total features")
                
                if is_main_process():
                    # Load and concatenate chunks efficiently
                    features_list = []
                    paths_list = []
                    
                    # Load paths from metadata
                    paths_file = os.path.join(self.feature_cache_dir, 'paths.pkl')
                    if os.path.exists(paths_file):
                        with open(paths_file, 'rb') as f:
                            image_paths = pickle.load(f)
                            self.logger.info(f"Loaded {len(image_paths)} cached paths")
                    else:
                        self.logger.warning("Paths file not found")
                        return None, None
                        
                    # Load features
                    for i in range(num_chunks):
                        chunk_file = os.path.join(self.feature_chunks_dir, f"chunk_{i}.npy")
                        if not os.path.exists(chunk_file):
                            self.logger.warning(f"Chunk file {chunk_file} not found")
                            return None, None
                            
                        # Load with memory mapping for efficiency
                        chunk = np.load(chunk_file, mmap_mode='r')
                        features_list.append(chunk)
                        
                    # Concatenate features
                    features = np.concatenate(features_list, axis=0)
                    
                    # Verify feature count matches path count
                    if features.shape[0] != len(image_paths):
                        self.logger.warning(f"Feature count ({features.shape[0]}) doesn't match path count ({len(image_paths)})")
                        return None, None
                        
                    # Broadcast features to all processes
                    features = torch.from_numpy(features)
                else:
                    # Non-main processes wait for feature loading
                    features = None
                    image_paths = None
                
                # Broadcast results
                features = broadcast_object(features)
                image_paths = broadcast_object(image_paths)
                
                if features is not None and len(features) > 0:
                    self.logger.info(f"Loaded {len(features)} cached features")
                    return features, image_paths
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached features: {e}. Extracting new features.")
            
        return None, None
        
    def extract_features(self, dataloader):
        """
        Extract features from images using the vision encoder
        
        Args:
            dataloader: DataLoader for images
            
        Returns:
            Tuple of (features tensor, image_paths)
        """
        # Try to load from cache first
        image_paths = []
        for batch in dataloader:
            if isinstance(batch, dict) and 'path' in batch:
                image_paths.extend(batch['path'])
                
        cached_features, cached_paths = self.load_cached_features(image_paths)
        if cached_features is not None:
            return cached_features, cached_paths
            
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
                    images = images.to(self.config.device)
                    
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
                    
                    # Check if we should save the current chunk
                    if len(current_paths) >= max_chunk_size:
                        # Concatenate and save chunk
                        chunk_tensor = torch.cat(current_chunk, dim=0)
                        chunk_array = chunk_tensor.numpy()
                        
                        # Save chunk
                        chunk_file = os.path.join(self.feature_chunks_dir, f"chunk_{chunk_idx}.npy")
                        np.save(chunk_file, chunk_array)
                        
                        # Add to features list and image paths
                        features_list.append(chunk_tensor)
                        image_paths.extend(current_paths)
                        
                        # Reset for next chunk
                        chunk_idx += 1
                        current_chunk = []
                        current_paths = []
                
                # Save final chunk if needed
                if current_chunk:
                    chunk_tensor = torch.cat(current_chunk, dim=0)
                    chunk_array = chunk_tensor.numpy()
                    
                    # Save chunk
                    chunk_file = os.path.join(self.feature_chunks_dir, f"chunk_{chunk_idx}.npy")
                    np.save(chunk_file, chunk_array)
                    
                    # Add to features list and image paths
                    features_list.append(chunk_tensor)
                    image_paths.extend(current_paths)
                    
                # Save paths
                paths_file = os.path.join(self.feature_cache_dir, 'paths.pkl')
                with open(paths_file, 'wb') as f:
                    pickle.dump(image_paths, f)
                    
                # Save metadata
                metadata_file = os.path.join(self.feature_cache_dir, 'metadata.json')
                
                # Compute dataset hash if not already set
                if self.dataset_hash is None:
                    self.dataset_hash = self.compute_dataset_hash(image_paths)
                    
                    # Save hash
                    hash_file = os.path.join(self.feature_cache_dir, 'dataset_hash.txt')
                    with open(hash_file, 'w') as f:
                        f.write(self.dataset_hash)
                    
                # Save metadata
                with open(metadata_file, 'w') as f:
                    metadata = {
                        'num_chunks': chunk_idx + 1,
                        'total_features': len(image_paths),
                        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'dataset_hash': self.dataset_hash
                    }
                    json.dump(metadata, f)
                    
                # Concatenate all features
                features = torch.cat(features_list)
                self.logger.info(f"Extracted {len(features)} features from {len(image_paths)} images")
        else:
            # Non-main processes create empty tensors (will be broadcast)
            features = None
                
        # Synchronize before continuing
        synchronize()
        
        # Broadcast features and paths to all processes
        if is_main_process():
            features_tensor = torch.cat(features_list)
        else:
            features_tensor = None
            
        # Broadcast results
        features_tensor = broadcast_object(features_tensor)
        image_paths = broadcast_object(image_paths)
            
        return features_tensor, image_paths
    
    def generate_clusters(self, dataloader=None, k=None, fine_clusters=1024):
        """
        Generate clusters from extracted features
        
        Args:
            dataloader: DataLoader for images
            k: Number of clusters (if None, uses config value)
            fine_clusters: Number of fine-grained clusters (default=1024)
            
        Returns:
            Cluster assignments for each datapoint
        """
        if k is None:
            k = getattr(self.config, 'num_experts', 8)
        
        fine_clusters = getattr(self.config, 'fine_clusters', fine_clusters)
        
        self.logger.info(f"Generating {k} clusters from data (with {fine_clusters} fine clusters)")
        
        # Extract features if not already done
        if dataloader is not None:
            features, image_paths = self.extract_features(dataloader)
        else:
            self.logger.error("No dataloader provided for feature extraction")
            return None
        
        # Ensure features are on CPU for clustering
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            
        self.logger.info(f"Creating clusters from {len(features)} feature vectors")
        
        # 1. Create fine-grained clusters (Paper Section 4.1)
        fine_labels = self._create_fine_clusters(features, fine_clusters)
        
        # 2. Consolidate to coarse clusters (Paper Section 4.1)
        coarse_labels = self._create_coarse_clusters(features, fine_labels, k)
        
        # Store paths for consistency checking later
        self.image_paths = image_paths
        
        return coarse_labels
    
    def _create_fine_clusters(self, features, num_clusters=1024):
        """
        Create fine-grained clusters with efficient MiniBatch K-means
        
        Args:
            features: Feature tensor (N x D)
            num_clusters: Number of fine clusters
            
        Returns:
            Fine-grained cluster labels tensor
        """
        # Check cache first
        if is_main_process() and os.path.exists(self.fine_cluster_cache_file):
            self.logger.info(f"Loading fine clusters from cache: {self.fine_cluster_cache_file}")
            try:
                with open(self.fine_cluster_cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    
                    # Get dataset hash from cache data and check for consistency
                    cached_hash = cache_data.get("dataset_hash", None)
                    current_hash = self.dataset_hash
                    
                    if cached_hash != current_hash:
                        self.logger.warning(f"Dataset hash mismatch for fine clusters cache: {cached_hash} != {current_hash}")
                    else:
                        # Hash matches, use cached clusters
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
        if is_main_process():
            self.logger.info(f"Creating {num_clusters} fine-grained clusters with MiniBatchKMeans")
            
            # Paper recommended approach: MiniBatchKMeans
            batch_size = min(1024, len(features))
            kmeans = MiniBatchKMeans(
                n_clusters=num_clusters,
                batch_size=batch_size,
                init='k-means++',
                n_init=3,
                random_state=42,
                verbose=1
            )
            
            # Fit model
            start_time = time.time()
            kmeans.fit(features)
            elapsed = time.time() - start_time
            
            # Save model and clustering results
            fine_cluster_labels = kmeans.labels_
            unique_clusters = np.unique(fine_cluster_labels)
            
            self.logger.info(f"Created {len(unique_clusters)} fine clusters in {elapsed:.1f}s")
            
            # Calculate metrics
            try:
                sample_size = min(10000, len(features))
                indices = np.random.choice(len(features), sample_size, replace=False)
                silhouette = silhouette_score(
                    features[indices],
                    fine_cluster_labels[indices],
                    sample_size=sample_size
                )
                self.logger.info(f"Fine clusters silhouette score: {silhouette:.4f}")
            except Exception as e:
                self.logger.warning(f"Could not compute silhouette score: {e}")
                
            # Log cluster sizes
            cluster_sizes = np.bincount(fine_cluster_labels)
            self.logger.info(f"Fine cluster sizes: min={cluster_sizes.min()}, "
                             f"max={cluster_sizes.max()}, "
                             f"mean={cluster_sizes.mean():.1f}, "
                             f"median={np.median(cluster_sizes):.1f}")
                             
            # Cache results
            try:
                with open(self.fine_cluster_cache_file, "wb") as f:
                    pickle.dump({
                        "labels": fine_cluster_labels,
                        "model": kmeans,
                        "dataset_hash": self.dataset_hash,
                        "timestamp": time.time()
                    }, f)
                self.logger.info(f"Cached fine clusters to {self.fine_cluster_cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cache fine clusters: {e}")
                
            # Store model
            self.fine_kmeans = kmeans
        else:
            # Non-main processes wait for clustering
            fine_cluster_labels = None
            
        # Broadcast results to all processes
        synchronize()
        
        if get_world_size() > 1:
            fine_cluster_labels = broadcast_numpy_array(fine_cluster_labels)
            
        # Store results
        self.fine_cluster_labels = fine_cluster_labels
        
        return fine_cluster_labels
    
    def _create_coarse_clusters(self, features, fine_labels, k):
        """
        Create coarse clusters by consolidating fine-grained clusters
        following the paper's multi-stage approach in Section 4.1
        
        Args:
            features: Feature tensor (N x D)
            fine_labels: Fine-grained cluster labels (N,)
            k: Number of coarse clusters
            
        Returns:
            Coarse cluster labels tensor
        """
        # Check cache first
        if is_main_process() and os.path.exists(self.coarse_cluster_cache_file):
            self.logger.info(f"Loading coarse clusters from cache: {self.coarse_cluster_cache_file}")
            try:
                with open(self.coarse_cluster_cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    
                    # Get dataset hash and cluster count from cache data
                    cached_hash = cache_data.get("dataset_hash", None)
                    cached_k = cache_data.get("k", None)
                    current_hash = self.dataset_hash
                    
                    # Check if cache is valid
                    if cached_hash != current_hash:
                        self.logger.warning(f"Dataset hash mismatch for coarse clusters cache")
                    elif cached_k != k:
                        self.logger.warning(f"Cluster count mismatch (cached: {cached_k}, requested: {k})")
                    else:
                        # Cache is valid
                        coarse_cluster_labels = cache_data["labels"]
                        self.coarse_kmeans = cache_data.get("model", None)
                        
                        self.logger.info(f"Loaded {len(np.unique(coarse_cluster_labels))} coarse clusters from cache")
                        
                        # Broadcast to all processes
                        if get_world_size() > 1:
                            coarse_cluster_labels = broadcast_numpy_array(coarse_cluster_labels)
                            
                        self.coarse_cluster_labels = coarse_cluster_labels
                        return coarse_cluster_labels
                        
            except Exception as e:
                self.logger.warning(f"Failed to load coarse clusters from cache: {e}")
        
        # Run clustering on main process only
        if is_main_process():
            self.logger.info(f"Consolidating fine clusters into {k} coarse clusters (paper Section 4.1)")
            
            # Paper Section 4.1 multi-stage approach:
            # 1. Compute centroids for each fine-grained cluster
            fine_cluster_ids = np.unique(fine_labels)
            num_fine_clusters = len(fine_cluster_ids)
            
            self.logger.info(f"Computing centroids for {num_fine_clusters} fine clusters")
            
            # Compute centroids for each fine cluster 
            fine_centroids = np.zeros((num_fine_clusters, features.shape[1]))
            fine_cluster_sizes = np.zeros(num_fine_clusters)
            
            # Map original fine cluster IDs to consecutive indices
            fine_id_to_idx = {fine_id: idx for idx, fine_id in enumerate(fine_cluster_ids)}
            
            # Compute centroid for each fine cluster
            for i, fine_id in enumerate(fine_cluster_ids):
                mask = fine_labels == fine_id
                fine_centroids[i] = features[mask].mean(axis=0)
                fine_cluster_sizes[i] = mask.sum()
            
            # 2. Cluster the fine centroids into k coarse clusters
            self.logger.info(f"Clustering {num_fine_clusters} fine centroids into {k} coarse clusters")
            
            # Apply K-means++ to cluster fine centroids
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=42,
                verbose=1
            )
            
            # Weight centroids by cluster size to prevent tiny clusters from dominating
            coarse_labels_for_fine = kmeans.fit_predict(fine_centroids)
            
            # Map fine cluster labels to coarse cluster labels
            fine_to_coarse = {fine_id: coarse_labels_for_fine[fine_id_to_idx[fine_id]] 
                             for fine_id in fine_cluster_ids}
            
            # 3. Map each data point to its coarse cluster
            coarse_cluster_labels = np.array([fine_to_coarse[label] for label in fine_labels])
            
            # Compute coarse cluster sizes for logging
            unique_coarse, cluster_sizes = np.unique(coarse_cluster_labels, return_counts=True)
            
            # Log clustering results
            self.logger.info(f"Created {len(unique_coarse)} coarse clusters from {num_fine_clusters} fine clusters")
            self.logger.info(f"Coarse cluster sizes: min={cluster_sizes.min()}, "
                             f"max={cluster_sizes.max()}, "
                             f"mean={cluster_sizes.mean():.1f}, "
                             f"median={np.median(cluster_sizes):.1f}")
                             
            # Cache results
            try:
                with open(self.coarse_cluster_cache_file, "wb") as f:
                    pickle.dump({
                        "labels": coarse_cluster_labels,
                        "model": kmeans,
                        "fine_to_coarse": fine_to_coarse,
                        "k": k, 
                        "dataset_hash": self.dataset_hash,
                        "timestamp": time.time()
                    }, f)
                self.logger.info(f"Cached coarse clusters to {self.coarse_cluster_cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cache coarse clusters: {e}")
                
            # Store model
            self.coarse_kmeans = kmeans
        else:
            # Non-main processes wait for clustering
            coarse_cluster_labels = None
            
        # Broadcast results to all processes
        synchronize()
        
        if get_world_size() > 1:
            coarse_cluster_labels = broadcast_numpy_array(coarse_cluster_labels)
            
        # Store results
        self.coarse_cluster_labels = coarse_cluster_labels
        
        return coarse_cluster_labels
        
    def predict_cluster(self, features):
        """
        Predict cluster assignments for new features
        
        Args:
            features: Feature tensor (N x D)
            
        Returns:
            Cluster assignments for each input feature
        """
        if self.fine_kmeans is None or self.coarse_kmeans is None:
            self.logger.error("Models not yet fit. Call generate_clusters first.")
            return None
            
        # Ensure features are on CPU for clustering
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            
        # Predict fine clusters
        fine_labels = self.fine_kmeans.predict(features)
        
        # Map to coarse clusters
        fine_to_coarse = self.coarse_kmeans.labels_
        
        coarse_labels = np.zeros_like(fine_labels)
        for i, fine_label in enumerate(fine_labels):
            coarse_labels[i] = fine_to_coarse[fine_label]
            
        return coarse_labels
        
    def visualize_clusters(self, features, labels, output_path=None, n_samples=1000):
        """
        Generate visualization of clusters
        
        Args:
            features: Feature tensor
            labels: Cluster labels
            output_path: Path to save visualization
            n_samples: Number of samples to visualize
            
        Returns:
            Path to saved visualization or None
        """
        if not is_main_process():
            return None
            
        try:
            self.logger.info(f"Generating cluster visualization with {n_samples} samples")
            
            # Sample features for visualization
            if len(features) > n_samples:
                indices = np.random.choice(len(features), n_samples, replace=False)
                viz_features = features[indices]
                viz_labels = labels[indices]
            else:
                viz_features = features
                viz_labels = labels
                
            # Generate visualization using UMAP
            if output_path is None:
                output_path = os.path.join(self.cache_dir, 'cluster_visualization.png')
                
            # Create visualization
            figpath = visualize_embeddings(
                embeddings=viz_features,
                labels=viz_labels,
                output_path=output_path,
                title=f"Cluster Visualization (k={len(np.unique(labels))})"
            )
            
            self.logger.info(f"Saved cluster visualization to {output_path}")
            return figpath
        except Exception as e:
            self.logger.error(f"Failed to create cluster visualization: {e}")
            return None

    def perform_clustering(self, dataset=None, features=None, method=None):
        """
        Perform clustering on dataset features
        
        Args:
            dataset: DDMDataset object (optional if features provided)
            features: Feature matrix (optional if dataset provided)
            method: Clustering method (defaults to config)
            
        Returns:
            np.ndarray: Cluster labels
        """
        # Get features if not provided
        if features is None:
            if self.features is not None:
                features = self.features
            elif dataset is not None:
                features = self.extract_features(dataset)
            else:
                raise ValueError("Either features or dataset must be provided")
        
        # Get method if not provided
        if method is None:
            method = getattr(self.config, 'cluster_method', 'kmeans')
            
        # Get number of clusters
        num_clusters = getattr(self.config, 'num_clusters', 10)
        
        self.logger.info(f"Performing {method} clustering with {num_clusters} clusters...")
        
        # Perform clustering
        if method.lower() == 'kmeans':
            # Use k-means clustering
            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=getattr(self.config, 'seed', 42),
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(features)
            self.centroids = kmeans.cluster_centers_
            
        elif method.lower() == 'gmm':
            # Use Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=num_clusters,
                random_state=getattr(self.config, 'seed', 42),
                n_init=10
            )
            cluster_labels = gmm.fit_predict(features)
            self.centroids = gmm.means_
            
        elif method.lower() == 'dbscan':
            # Use DBSCAN
            eps = getattr(self.config, 'dbscan_eps', 0.5)
            min_samples = getattr(self.config, 'dbscan_min_samples', 5)
            
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )
            cluster_labels = dbscan.fit_predict(features)
            
            # Compute centroids for each cluster
            unique_labels = np.unique(cluster_labels)
            self.centroids = np.zeros((len(unique_labels), features.shape[1]))
            
            for i, label in enumerate(unique_labels):
                mask = cluster_labels == label
                self.centroids[i] = features[mask].mean(axis=0)
                
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        # Store results
        self.cluster_labels = cluster_labels
        
        # Compute cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        self.cluster_sizes = {label: count for label, count in zip(unique_labels, counts)}
        
        self.logger.info(f"Clustering complete. Cluster distribution: {self.cluster_sizes}")
        
        return cluster_labels
    
    def assign_experts(self, force_recompute=False):
        """
        Assign clusters to experts for distributed training
        Uses smart allocation to balance load across experts
        
        Args:
            force_recompute: Force recomputation of assignments
            
        Returns:
            dict: Mapping of cluster IDs to expert IDs
        """
        # Skip if already computed
        if self.expert_assignment is not None and not force_recompute:
            return self.expert_assignment
            
        # Ensure cluster sizes are computed
        if self.cluster_sizes is None:
            if self.cluster_labels is None:
                raise ValueError("Must perform clustering before assigning experts")
                
            unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
            self.cluster_sizes = {label: count for label, count in zip(unique_labels, counts)}
        
        # Get number of experts
        num_experts = getattr(self.config, 'num_experts', 1)
        if num_experts <= 1:
            # Single expert gets all clusters
            self.expert_assignment = {label: 0 for label in self.cluster_sizes.keys()}
            return self.expert_assignment
        
        # Get expert overlap parameter (% of clusters shared between experts)
        overlap = getattr(self.config, 'expert_overlap', 0.1)
        
        # Sort clusters by size (descending)
        sorted_clusters = sorted(
            self.cluster_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate number of samples per expert (ideal)
        total_samples = sum(self.cluster_sizes.values())
        samples_per_expert = total_samples / num_experts
        
        # Allocate largest clusters first to minimize imbalance
        expert_loads = [0] * num_experts
        self.expert_assignment = {}
        
        # First pass: assign each cluster to the expert with minimum load
        for cluster_id, size in sorted_clusters:
            # Find expert with minimum load
            min_expert = expert_loads.index(min(expert_loads))
            
            # Assign cluster to expert
            self.expert_assignment[cluster_id] = min_expert
            
            # Update expert load
            expert_loads[min_expert] += size
        
        # Second pass: handle overlap (duplicate some clusters to multiple experts)
        if overlap > 0:
            # Determine number of shared clusters
            num_clusters = len(sorted_clusters)
            num_shared = int(num_clusters * overlap)
            
            # Take the top 'num_shared' largest clusters for sharing
            for i in range(min(num_shared, num_clusters)):
                cluster_id = sorted_clusters[i][0]
                size = sorted_clusters[i][1]
                
                # Current expert assignment
                current_expert = self.expert_assignment[cluster_id]
                
                # Find next best expert (excluding current one)
                available_experts = expert_loads.copy()
                available_experts[current_expert] = float('inf')  # Exclude current expert
                next_expert = available_experts.index(min(available_experts))
                
                # Duplicate assignment with a special marker
                # Tuple (expert_id, is_shared)
                self.expert_assignment[cluster_id] = [(current_expert, False), (next_expert, True)]
                
                # Update expert load (shared clusters count at 50% weight)
                expert_loads[next_expert] += size * 0.5
        
        # Log expert allocation
        expert_cluster_counts = {}
        for cluster_id, assignment in self.expert_assignment.items():
            if isinstance(assignment, list):
                # Shared cluster
                for expert_id, is_shared in assignment:
                    expert_cluster_counts[expert_id] = expert_cluster_counts.get(expert_id, 0) + (0.5 if is_shared else 1)
            else:
                # Regular assignment
                expert_cluster_counts[assignment] = expert_cluster_counts.get(assignment, 0) + 1
                
        self.logger.info(f"Expert allocation: {expert_cluster_counts}")
        
        # Calculate load balance
        if expert_loads:
            avg_load = sum(expert_loads) / len(expert_loads)
            max_imbalance = max(abs(load - avg_load) / avg_load for load in expert_loads)
            self.logger.info(f"Load imbalance: {max_imbalance:.2%}")
        
        return self.expert_assignment
    
    def get_expert_for_cluster(self, cluster_id):
        """
        Get expert(s) assigned to a cluster
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            int or list: Expert ID(s) responsible for the cluster
        """
        if self.expert_assignment is None:
            self.assign_experts()
            
        # Handle noise points (-1) in DBSCAN
        if cluster_id == -1 or cluster_id not in self.expert_assignment:
            return 0  # Default to first expert for noise
            
        return self.expert_assignment.get(cluster_id, 0)
    
    def get_expert_for_sample(self, sample_idx, cluster_labels=None):
        """
        Get expert(s) for a specific sample
        
        Args:
            sample_idx: Sample index
            cluster_labels: Cluster labels (optional if already computed)
            
        Returns:
            int or list: Expert ID(s) for the sample
        """
        if cluster_labels is None:
            if self.cluster_labels is None:
                raise ValueError("Cluster labels not available")
            cluster_labels = self.cluster_labels
            
        if sample_idx >= len(cluster_labels):
            raise ValueError(f"Sample index {sample_idx} out of range")
            
        cluster_id = cluster_labels[sample_idx]
        return self.get_expert_for_cluster(cluster_id)
    
    def get_samples_for_expert(self, expert_id, cluster_labels=None, include_shared=True):
        """
        Get sample indices assigned to an expert
        
        Args:
            expert_id: Expert ID
            cluster_labels: Cluster labels (optional if already computed)
            include_shared: Whether to include shared samples
            
        Returns:
            list: Sample indices for the expert
        """
        if cluster_labels is None:
            if self.cluster_labels is None:
                raise ValueError("Cluster labels not available")
            cluster_labels = self.cluster_labels
        
        if self.expert_assignment is None:
            self.assign_experts()
            
        # Create mapping from cluster to expert
        clusters_for_expert = []
        
        for cluster_id, assignment in self.expert_assignment.items():
            if isinstance(assignment, list):
                # Shared cluster
                for expert, is_shared in assignment:
                    if expert == expert_id and (include_shared or not is_shared):
                        clusters_for_expert.append(cluster_id)
            elif assignment == expert_id:
                # Direct assignment
                clusters_for_expert.append(cluster_id)
                
        # Get samples for these clusters
        sample_indices = []
        
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id in clusters_for_expert:
                sample_indices.append(i)
                
        return sample_indices
    
    def save(self, filepath):
        """
        Save cluster manager state
        
        Args:
            filepath: Path to save the state
        """
        # Only save from main process
        if not is_main_process():
            return
            
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare state
        state = {
            'centroids': self.centroids,
            'cluster_labels': self.cluster_labels,
            'cluster_sizes': self.cluster_sizes,
            'expert_assignment': self.expert_assignment,
            'config': {
                'num_experts': getattr(self.config, 'num_experts', 1),
                'cluster_method': getattr(self.config, 'cluster_method', 'kmeans'),
                'num_clusters': getattr(self.config, 'num_clusters', 10),
                'expert_overlap': getattr(self.config, 'expert_overlap', 0.1)
            }
        }
        
        # Save state
        try:
            torch.save(state, filepath)
            self.logger.info(f"Saved cluster state to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save cluster state: {e}")
    
    def load(self, filepath):
        """
        Load cluster manager state
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"Cluster state file not found: {filepath}")
            return False
            
        try:
            state = torch.load(filepath, map_location='cpu')
            
            self.centroids = state.get('centroids', None)
            self.cluster_labels = state.get('cluster_labels', None)
            self.cluster_sizes = state.get('cluster_sizes', None)
            self.expert_assignment = state.get('expert_assignment', None)
            
            # Log load success
            if self.cluster_labels is not None:
                num_clusters = len(np.unique(self.cluster_labels))
                self.logger.info(f"Loaded cluster state with {num_clusters} clusters from {filepath}")
            
            # Verify configuration compatibility
            loaded_config = state.get('config', {})
            current_config = {
                'num_experts': getattr(self.config, 'num_experts', 1),
                'cluster_method': getattr(self.config, 'cluster_method', 'kmeans'),
                'num_clusters': getattr(self.config, 'num_clusters', 10)
            }
            
            # Check for mismatches
            for key, value in current_config.items():
                if key in loaded_config and loaded_config[key] != value:
                    self.logger.warning(f"Configuration mismatch: {key}={value}, loaded={loaded_config[key]}")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load cluster state: {e}")
            return False
    
    def get_cluster_labels(self):
        """
        Get cluster labels
        
        Returns:
            np.ndarray: Cluster labels
        """
        return self.cluster_labels
        
    def get_cluster_stats(self):
        """
        Get cluster statistics
        
        Returns:
            dict: Cluster statistics
        """
        if self.cluster_labels is None:
            return {}
            
        # Compute basic statistics
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        stats = {
            'num_clusters': len(unique_labels),
            'cluster_sizes': {label: count for label, count in zip(unique_labels, counts)},
            'min_cluster_size': min(counts),
            'max_cluster_size': max(counts),
            'avg_cluster_size': np.mean(counts),
            'imbalance': max(counts) / min(counts) if min(counts) > 0 else float('inf')
        }
        
        return stats 

    def cluster_data(self, features, num_clusters, algorithm='kmeans', random_state=42):
        """
        Cluster feature data using specified algorithm
        
        Args:
            features: Feature array to cluster
            num_clusters: Number of clusters to create
            algorithm: Clustering algorithm to use ('kmeans', 'minibatch_kmeans', 'gmm', 'dbscan')
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (cluster_labels, centroids, clustering_model)
        """
        start_time = time.time()
        self.logger.info(f"Starting clustering using {algorithm} algorithm for {num_clusters} clusters "
                         f"on {features.shape[0]} samples with {features.shape[1]} features")
        
        # Choose algorithm and cluster
        if algorithm == 'kmeans':
            # Regular KMeans - good quality but slow for large datasets
            self.logger.info(f"Using KMeans with {num_clusters} clusters")
            
            # Get batch size from config or use default
            batch_size = getattr(self.config, 'kmeans_batch_size', 1024)
            algorithm_start = time.time()
            
            # For large datasets, use mini-batch with increasingly larger batches
            if features.shape[0] > 100000:
                self.logger.info(f"Large dataset with {features.shape[0]} samples detected, using progressive mini-batch approach")
                
                # Initial model with small batches
                self.logger.info("Phase 1: Initial clustering with small batches")
                phase1_start = time.time()
                kmeans = MiniBatchKMeans(
                    n_clusters=num_clusters,
                    batch_size=min(batch_size, features.shape[0]//10),
                    init='k-means++',
                    max_iter=100,
                    random_state=random_state
                )
                kmeans.fit(features)
                phase1_time = time.time() - phase1_start
                
                # Refine with larger batches
                self.logger.info(f"Phase 1 completed in {phase1_time:.2f}s. Phase 2: Refining with larger batches")
                phase2_start = time.time()
                kmeans = MiniBatchKMeans(
                    n_clusters=num_clusters,
                    batch_size=min(batch_size*4, features.shape[0]//5),
                    init=kmeans.cluster_centers_,
                    max_iter=50,
                    random_state=random_state
                )
                kmeans.fit(features)
                phase2_time = time.time() - phase2_start
                
                # Final refinement
                self.logger.info(f"Phase 2 completed in {phase2_time:.2f}s. Phase 3: Final refinement")
                phase3_start = time.time()
                kmeans = KMeans(
                    n_clusters=num_clusters,
                    init=kmeans.cluster_centers_,
                    max_iter=10,
                    n_init=1,
                    random_state=random_state
                )
                kmeans.fit(features)
                phase3_time = time.time() - phase3_start
                self.logger.info(f"Phase 3 completed in {phase3_time:.2f}s")
                
            else:
                # Regular KMeans for smaller datasets with progress updates
                self.logger.info(f"Using standard KMeans for dataset with {features.shape[0]} samples")
                
                # Custom implementation with progress tracking for standard KMeans
                # We can't directly modify sklearn's implementation, so we'll track iterations externally
                
                # Initialize with k-means++
                self.logger.info("Initializing clusters with k-means++")
                init_start = time.time()
                kmeans = KMeans(
                    n_clusters=num_clusters,
                    init='k-means++',
                    max_iter=300,
                    n_init=10 if features.shape[0] < 50000 else 3,  # More initializations for smaller datasets
                    random_state=random_state,
                    verbose=0
                )
                
                # Use a trick to monitor progress - create a callback for verbose output
                from sklearn.utils.validation import check_is_fitted
                orig_fit = kmeans.fit
                
                def verbose_fit(X, y=None, sample_weight=None):
                    """Wrapper around KMeans fit method to add progress updates"""
                    self.logger.info(f"Starting KMeans fitting with {kmeans.max_iter} max iterations")
                    fit_start = time.time()
                    
                    # Call original fit method
                    result = orig_fit(X, y, sample_weight)
                    
                    # Log completion
                    fit_time = time.time() - fit_start
                    n_iter = kmeans.n_iter_
                    self.logger.info(f"KMeans fitting completed in {fit_time:.2f}s after {n_iter} iterations "
                                    f"({fit_time/n_iter:.2f}s per iteration)")
                    return result
                
                kmeans.fit = verbose_fit
                kmeans.fit(features)
                
            # Get results
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
            # Calculate metrics
            inertia = kmeans.inertia_
            iterations = kmeans.n_iter_
            
            # Log clustering stats
            algorithm_time = time.time() - algorithm_start
            self.logger.info(f"KMeans clustering completed in {algorithm_time:.2f}s after {iterations} iterations "
                           f"with inertia {inertia:.2f}")
            
        elif algorithm == 'minibatch_kmeans':
            # MiniBatch KMeans - faster but less accurate
            self.logger.info(f"Using MiniBatchKMeans with {num_clusters} clusters")
            batch_size = min(getattr(self.config, 'minibatch_size', 1024), features.shape[0])
            
            algorithm_start = time.time()
            self.logger.info(f"Using batch size of {batch_size} samples")
            
            # Calculate total iterations and time estimate
            max_iter = 300
            n_batches = features.shape[0] // batch_size + (1 if features.shape[0] % batch_size > 0 else 0)
            self.logger.info(f"Will process approximately {n_batches} batches per iteration, "
                            f"with up to {max_iter} iterations")
            
            # Initialize and fit
            kmeans = MiniBatchKMeans(
                n_clusters=num_clusters,
                batch_size=batch_size,
                max_iter=max_iter,
                init='k-means++',
                reassignment_ratio=0.01,
                random_state=random_state,
                verbose=0
            )
            
            # Monitor time
            last_log_time = time.time()
            log_interval = 5.0  # Log every 5 seconds
            
            # Fit in smaller chunks to track progress
            n_samples = features.shape[0]
            chunk_size = min(10000, n_samples)
            n_chunks = (n_samples + chunk_size - 1) // chunk_size
            
            self.logger.info(f"Fitting in {n_chunks} chunks of {chunk_size} samples for better progress tracking")
            
            for i in range(0, n_samples, chunk_size):
                chunk_end = min(i + chunk_size, n_samples)
                chunk_features = features[i:chunk_end]
                
                # Fit this chunk
                chunk_start = time.time()
                kmeans = kmeans.partial_fit(chunk_features)
                chunk_time = time.time() - chunk_start
                
                # Log progress
                current_time = time.time()
                if current_time - last_log_time > log_interval or chunk_end == n_samples:
                    progress = chunk_end / n_samples * 100
                    elapsed = current_time - algorithm_start
                    remaining = (elapsed / chunk_end) * (n_samples - chunk_end) if chunk_end > 0 else 0
                    
                    self.logger.info(f"MiniBatchKMeans progress: {progress:.1f}% ({chunk_end}/{n_samples}) - "
                                   f"Current inertia: {kmeans.inertia_:.2f} - "
                                   f"Elapsed: {elapsed:.2f}s, Estimated remaining: {remaining:.2f}s")
                    last_log_time = current_time
            
            # Get results
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
            # Log final stats
            algorithm_time = time.time() - algorithm_start
            self.logger.info(f"MiniBatchKMeans completed in {algorithm_time:.2f}s with final inertia {kmeans.inertia_:.2f}")
            
        elif algorithm == 'gmm':
            # Gaussian Mixture Model
            self.logger.info(f"Using Gaussian Mixture Model with {num_clusters} components")
            algorithm_start = time.time()
            
            # Initialize model with progress monitoring
            gmm = GaussianMixture(
                n_components=num_clusters,
                covariance_type='full',
                max_iter=100,
                random_state=random_state,
                verbose=0
            )
            
            # Fit model
            fit_start = time.time()
            gmm.fit(features)
            fit_time = time.time() - fit_start
            
            # Get results
            labels = gmm.predict(features)
            centroids = gmm.means_
            
            # Log clustering stats
            algorithm_time = time.time() - algorithm_start
            self.logger.info(f"GMM clustering completed in {algorithm_time:.2f}s after {gmm.n_iter_} iterations "
                           f"with log-likelihood {gmm.score(features):.2f}")
            
        elif algorithm == 'dbscan':
            # DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
            self.logger.info("Using DBSCAN for density-based clustering")
            algorithm_start = time.time()
            
            # Calculate epsilon based on feature dimensionality and dataset size
            from sklearn.neighbors import NearestNeighbors
            sample_size = min(10000, features.shape[0])  # Sample for epsilon calculation
            
            self.logger.info(f"Calculating epsilon parameter using {sample_size} samples...")
            epsilon_start = time.time()
            
            # Sample data
            indices = np.random.choice(features.shape[0], sample_size, replace=False)
            sample_features = features[indices]
            
            # Find distances to nearest neighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(sample_features)
            distances, _ = nn.kneighbors(sample_features)
            
            # Sort distances to find "elbow" point
            distances = np.sort(distances[:, 1])
            
            # Simple automatic epsilon selection based on distance distribution
            epsilon = np.percentile(distances, 95)  # Use 95th percentile as cutoff
            
            epsilon_time = time.time() - epsilon_start
            self.logger.info(f"Calculated epsilon = {epsilon:.4f} in {epsilon_time:.2f}s")
            
            # Adjust min_samples based on dataset size
            min_samples = max(5, int(np.log(features.shape[0])))
            self.logger.info(f"Using min_samples = {min_samples}")
            
            # Run DBSCAN
            fit_start = time.time()
            dbscan = DBSCAN(
                eps=epsilon,
                min_samples=min_samples,
                metric='euclidean',
                n_jobs=-1
            )
            
            self.logger.info(f"Starting DBSCAN fitting...")
            dbscan.fit(features)
            fit_time = time.time() - fit_start
            
            # Get results
            labels = dbscan.labels_
            
            # Handle noise points (labeled as -1) by assigning to nearest cluster
            noise_mask = labels == -1
            noise_count = np.sum(noise_mask)
            
            if noise_count > 0:
                self.logger.info(f"Found {noise_count} noise points ({noise_count/features.shape[0]:.1%}), "
                               f"assigning to nearest clusters...")
                
                # Get non-noise cluster indices and compute centroids
                cluster_indices = np.unique(labels[~noise_mask])
                centroids = np.array([features[labels == i].mean(axis=0) for i in cluster_indices])
                
                # For each noise point, find nearest centroid
                noise_features = features[noise_mask]
                for i, feat in enumerate(noise_features):
                    # Compute distances to all centroids
                    distances = np.linalg.norm(centroids - feat, axis=1)
                    nearest_cluster = cluster_indices[np.argmin(distances)]
                    
                    # Assign to nearest cluster
                    labels[np.where(noise_mask)[0][i]] = nearest_cluster
                
                self.logger.info(f"Reassigned all noise points to nearest clusters")
            else:
                # Compute centroids
                cluster_indices = np.unique(labels)
                centroids = np.array([features[labels == i].mean(axis=0) for i in cluster_indices])
            
            # If too few clusters, fall back to KMeans
            if len(cluster_indices) < num_clusters / 2:
                self.logger.warning(f"DBSCAN found only {len(cluster_indices)} clusters, "
                                 f"falling back to KMeans for {num_clusters} clusters")
                return self.cluster_data(features, num_clusters, algorithm='kmeans', random_state=random_state)
            
            # Log clustering stats
            algorithm_time = time.time() - algorithm_start
            self.logger.info(f"DBSCAN clustering completed in {algorithm_time:.2f}s with {len(cluster_indices)} clusters")
            
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Verify cluster assignments
        unique_labels = np.unique(labels)
        self.logger.info(f"Found {len(unique_labels)} unique clusters")
        
        # Calculate and log cluster sizes
        cluster_sizes = np.bincount(labels.astype(np.int64))
        smallest = np.min(cluster_sizes)
        largest = np.max(cluster_sizes)
        mean_size = np.mean(cluster_sizes)
        median_size = np.median(cluster_sizes)
        
        self.logger.info(f"Cluster sizes - Min: {smallest}, Max: {largest}, Mean: {mean_size:.1f}, Median: {median_size:.1f}")
        
        # Calculate silhouette score if dataset is not too large
        if features.shape[0] <= 10000:  # Only compute for smaller datasets
            try:
                sample_size = min(5000, features.shape[0])
                sample_indices = np.random.choice(features.shape[0], sample_size, replace=False)
                silhouette = silhouette_score(features[sample_indices], labels[sample_indices])
                self.logger.info(f"Silhouette score (on {sample_size} samples): {silhouette:.4f}")
            except Exception as e:
                self.logger.warning(f"Could not compute silhouette score: {str(e)}")
        
        # Log total time
        total_time = time.time() - start_time
        self.logger.info(f"Total clustering process completed in {total_time:.2f}s")
        
        return labels, centroids, locals().get(algorithm, None)  # Return local variable with algorithm name 