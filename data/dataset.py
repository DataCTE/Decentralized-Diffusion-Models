"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from PIL import Image
import random
from collections import defaultdict, OrderedDict
import logging
import time  
import glob
import bisect

import io
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized utilities
from utils.distributed import is_main_process, broadcast_object, get_rank, get_local_rank, get_world_size
from utils.logging import setup_distributed_logger
from data.transforms import resize_image, normalize
import threading


# Setup logging
logger = logging.getLogger(__name__)

import math  # For BucketBatchSampler

# Add global cache to avoid duplicate validation
_GLOBAL_DATASET_CACHE = {
    "initialized": False,
    "image_files": [],
    "caption_files": [],
    "dim_cache": None,
}

def chunks(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_tensor_length(file_path):
    """Helper function for parallel metadata loading"""
    with open(file_path, 'rb') as f:
        return torch.load(f, map_location='cpu').shape[0]

class DDMDataset(Dataset):
    """GPU-optimized dataset pipeline for decentralized diffusion models with precomputed latents"""
    
    def __init__(self, config, split='train', transforms=None, hf_split=None):
        self.config = config
        self.split = split
        self.device = torch.device('cpu')
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Load precomputed paths
        self.feature_cache_path = config.feature_cache_path
        self.feature_path = os.path.join(self.feature_cache_path, "features")
        self.cluster_path = os.path.join(self.feature_cache_path, "clusters")
        self.latent_path = os.path.join(self.feature_cache_path, "latents")
        self.clip_embedding_path = os.path.join(self.feature_cache_path, "clip_embeddings")

        # Load precomputed file lists
        self.image_files = sorted([f.replace(".latent.pt", "") for f in os.listdir(self.latent_path)])
        self.caption_files = [os.path.join(self.config.dataset_path, f+".txt") for f in self.image_files]
        
        if is_main_process(): # Only load metadata on rank 0
            # Cluster file metadata with parallel loading
            self.cluster_files = sorted(glob.glob(os.path.join(self.cluster_path, "*.cluster.pt")))
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(get_tensor_length, cf) for cf in self.cluster_files]
                with tqdm(total=len(futures), desc="Loading cluster metadata") as pbar:
                    cluster_lengths = [] # Local variable for rank 0
                    for future in as_completed(futures):
                        cluster_lengths.append(future.result())
                        pbar.update(1)
            self.cluster_lengths = cluster_lengths # Assign to self for rank 0
            self.cumulative_clusters = np.cumsum([0] + self.cluster_lengths).tolist()

            # Feature file metadata with parallel loading
            self.feature_files = sorted(glob.glob(os.path.join(self.feature_path, "*.pt")))
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(get_tensor_length, ff) for ff in self.feature_files]
                with tqdm(total=len(futures), desc="Loading feature metadata") as pbar:
                    feature_lengths = [] # Local variable for rank 0
                for future in as_completed(futures):
                        feature_lengths.append(future.result())
                        pbar.update(1)
            self.feature_lengths = feature_lengths # Assign to self for rank 0
            self.cumulative_features = np.cumsum([0] + self.feature_lengths).tolist()

            # Broadcast metadata to other ranks
            broadcast_object((self.cluster_lengths, self.cumulative_clusters, self.feature_lengths, self.cumulative_features))
        else: # Receive broadcasted metadata on other ranks
            (self.cluster_lengths, self.cumulative_clusters, self.feature_lengths, self.cumulative_features) = broadcast_object(None)

        # Limited caches
        self.cluster_cache = OrderedDict()
        self.feature_cache = OrderedDict()
        self.cache_size = 5  # Keep 5 files in memory at once

        # Initialize buckets
        self._init_buckets()

    def __getitem__(self, idx):
        return {
            'latent': self._load_latent(idx),
            'expert': self._load_cluster(idx),
            'features': self._load_feature(idx),
            'clip_embedding': self._load_clip_embedding(idx),
            'bucket': self.bucket_assignments[idx]
        }

    def _load_cluster(self, idx):
        # Find which cluster file contains the index
        file_idx = bisect.bisect_right(self.cumulative_clusters, idx) - 1
        file_path = self.cluster_files[file_idx]
        
        # Load with caching
        if file_path not in self.cluster_cache:
            if len(self.cluster_cache) >= self.cache_size:
                self.cluster_cache.popitem(last=False)
            self.cluster_cache[file_path] = torch.load(file_path)
            
        # Get position within file
        pos = idx - self.cumulative_clusters[file_idx]
        return self.cluster_cache[file_path][pos]

    def _load_feature(self, idx):
        # Find which feature file contains the index
        file_idx = bisect.bisect_right(self.cumulative_features, idx) - 1
        file_path = self.feature_files[file_idx]
        
        # Load with caching
        if file_path not in self.feature_cache:
            if len(self.feature_cache) >= self.cache_size:
                self.feature_cache.popitem(last=False)
            self.feature_cache[file_path] = torch.load(file_path)
            
        # Get position within file
        pos = idx - self.cumulative_features[file_idx]
        return self.feature_cache[file_path][pos]

    def _init_buckets(self):
        """CPU-based bucket initialization"""
        # Before bucket assignment, handle train/val split if using cache
        if self.split == 'val' and _GLOBAL_DATASET_CACHE["initialized"]:
            # If this is the validation dataset and we're using cached data, 
            # select only a subset for validation
            val_size = getattr(self.config, 'val_size', 1000)
            if val_size < len(self.image_files):
                # Use deterministic selection to ensure consistency
                all_indices = np.arange(len(self.image_files))
                np.random.seed(42)  # Fixed seed for reproducibility
                val_indices = np.random.choice(all_indices, size=val_size, replace=False)
                
                # Filter files for validation
                self.image_files = [self.image_files[i] for i in val_indices]
                self.caption_files = [self.caption_files[i] for i in val_indices]
                self.dim_cache = self.dim_cache[val_indices]
                self.logger.info(f"Rank {get_rank()}: Selected {len(self.image_files)} files for validation split")
        elif self.split == 'train' and _GLOBAL_DATASET_CACHE["initialized"]:
            # For training, exclude validation samples if specified
            val_size = getattr(self.config, 'val_size', 1000)
            if val_size > 0 and val_size < len(self.image_files):
                # Use same deterministic selection as above
                all_indices = np.arange(len(self.image_files))
                np.random.seed(42)  # Fixed seed for reproducibility
                val_indices = np.random.choice(all_indices, size=val_size, replace=False)
                train_indices = np.setdiff1d(all_indices, val_indices)
                
                # Filter files for training
                self.image_files = [self.image_files[i] for i in train_indices]
                self.caption_files = [self.caption_files[i] for i in train_indices]
                self.dim_cache = self.dim_cache[train_indices]
                self.logger.info(f"Rank {get_rank()}: Selected {len(self.image_files)} files for training split (excluded {val_size} validation files)")
        
        self.logger.info(f"Rank {get_rank()}: Starting bucket assignment for {len(self.image_files)} images...")
        bucket_start = time.time()
        
        # Calculate aspect ratios
        bucket_aspects = self.bucket_dims[:,0] / self.bucket_dims[:,1]
        print(f"Shape of self.dim_cache: {self.dim_cache.shape}")
        image_aspects = self.dim_cache[:,0] / self.dim_cache[:,1]

        # Add progress bar for bucket assignment
        pbar_bucket_assign = tqdm(
            range(len(image_aspects)),
            desc=f"Rank {get_rank()}: Assigning buckets",
            unit="image",
            dynamic_ncols=True
        )

        # Find closest bucket using matrix ops
        diffs = torch.abs(image_aspects.unsqueeze(1) - bucket_aspects)
        self.bucket_assignments = torch.argmin(diffs, dim=1)

        pbar_bucket_assign.update(len(image_aspects)) # Complete progress bar
        pbar_bucket_assign.close()
        
        # Count images per bucket for logging
        bucket_counts = {}
        for i in range(self.bucket_dims.shape[0]):
            count = torch.sum(self.bucket_assignments == i).item()
            if count > 0:
                bucket_counts[i] = count
        
        bucket_time = time.time() - bucket_start
        self.logger.info(f"Rank {get_rank()}: Bucket assignment completed in {bucket_time:.2f}s - {len(bucket_counts)} buckets used")
        
        # Log distribution stats
        top_buckets = sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for bucket_idx, count in top_buckets:
            bucket_size = tuple(self.bucket_dims[bucket_idx].tolist())
            self.logger.info(f"Rank {get_rank()}: Bucket {bucket_idx} ({bucket_size}): {count} images")

    def _load_latent(self, idx):
        """Load precomputed latent tensor from disk, with caching and thread safety"""
        latent_file = self.image_files[idx] + ".latent.pt"
        latent_path = os.path.join(self.latent_path, latent_file)

        with self._latent_loading_lock[idx]: # Thread lock for loading
            if latent_path in self.latent_cache: # Check cache first
                latent = self.latent_cache[latent_path] # Load from cache
            else:
                latent = torch.load(latent_path, map_location=self.device) # Load from disk
                self._update_latent_cache(latent_path, latent) # Update cache
            logger.debug(f"Loaded latent tensor shape: {latent.shape} from {latent_path}") # Log shape
        return latent

    def _load_clip_embedding(self, idx):
        """Load precomputed CLIP embedding tensor from file, using cache"""
        clip_embedding_file_index = torch.searchsorted(self.cumulative_feature_counts, idx, right=True).item()
        if clip_embedding_file_index > 0:
            clip_embedding_index_in_file = idx - self.cumulative_feature_counts[clip_embedding_file_index - 1].item()
        else:
            clip_embedding_index_in_file = idx

        clip_embedding_file_path = os.path.join(self.clip_embedding_path, self.clip_embedding_files[clip_embedding_file_index])

        try:
            # Check cache first
            if self.clip_embedding_cache is not None and clip_embedding_file_path in self.clip_embedding_cache:
                clip_embeddings = self.clip_embedding_cache[clip_embedding_file_path]
                clip_embedding = clip_embeddings[clip_embedding_index_in_file]
                return clip_embedding
            else:
                # Load clip embeddings from disk
                if clip_embedding_file_path not in self._clip_embedding_loading_lock:
                    self._clip_embedding_loading_lock[clip_embedding_file_path] = threading.Lock()

                with self._clip_embedding_loading_lock[clip_embedding_file_path]:
                    if self.clip_embedding_cache is not None and clip_embedding_file_path in self.clip_embedding_cache:
                        # Re-check cache in case it was loaded while waiting for lock
                        clip_embeddings = self.clip_embedding_cache[clip_embedding_file_path]
                        clip_embedding = clip_embeddings[clip_embedding_index_in_file]
                        return clip_embedding
                    else:
                        clip_embeddings = torch.load(clip_embedding_file_path, map_location='cpu')
                        if self.clip_embedding_cache is not None:
                            self.clip_embedding_cache[clip_embedding_file_path] = clip_embeddings
                            # Manage cache size - LRU eviction
                            if len(self.clip_embedding_cache) > self.clip_embedding_cache_max_size:
                                self.clip_embedding_cache.popitem(last=False) # Remove LRU item

                        clip_embedding = clip_embeddings[clip_embedding_index_in_file]
                        return clip_embedding
        except Exception as e:
            self.logger.error(f"Error loading clip embedding from {clip_embedding_file_path} at index {idx}: {e}")
            return None

    def __len__(self):
        """Get dataset length (number of latents, same as images)"""
        return len(self.image_files)
        
    def get_status_summary(self):
        """Generate a user-friendly status summary of dataset processing"""
        if not hasattr(self, 'image_files') or len(self.image_files) == 0:
            return {
                "status": "incomplete",
                "message": "Dataset processing has not completed or failed",
                "images_found": 0
            }
            
        # Count buckets actually used
        bucket_counts = {}
        if hasattr(self, 'bucket_assignments'):
            for i in range(self.bucket_dims.shape[0]):
                count = torch.sum(self.bucket_assignments == i).item()
                if count > 0:
                    bucket_counts[i] = count
        
        # Count images per expert
        expert_counts = {}
        if hasattr(self, 'expert_assignments'):
            for i in range(self.num_experts.item()):
                count = torch.sum(self.expert_assignments == i).item()
                if count > 0:
                    expert_counts[i] = count
        
        # Images for this rank
        this_rank_count = 0
        if hasattr(self, 'expert_assignments'):
            this_rank_count = torch.sum(self.expert_assignments == get_rank()).item()
        
        return {
            "status": "complete",
            "rank": get_rank(),
            "total_images": len(self.image_files),
            "total_buckets": len(bucket_counts),
            "total_experts": len(expert_counts),
            "images_for_this_rank": this_rank_count,
            "percent_for_this_rank": f"{this_rank_count/len(self.image_files)*100:.1f}%",
            "top_buckets": sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "expert_distribution": expert_counts
        }

    def _default_transform(self, img, bucket_idx=None):
        """Apply default transformations based on bucket dimensions"""
        # Get target dimensions from bucket if provided
        if bucket_idx is not None and 0 <= bucket_idx < len(self.buckets):
            width, height = self.buckets[bucket_idx]
        else:
            # Fallback to default image size
            _, height, width = self.config.image_size
        
        # Resize image to target dimensions
        if isinstance(img, Image.Image):
            # PIL Image
            img = resize_image(img, (width, height))
            img = transforms.ToTensor()(img)
        elif isinstance(img, torch.Tensor):
            # Already a tensor, resize with torch functions
            if img.shape[-2] != height or img.shape[-1] != width:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), 
                    size=(height, width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
        
        # Normalize
        img = normalize(img)
        return img
        
    def _create_bucket_samplers(self):
        """Create bucket-specific samplers to ensure consistent shapes in each batch"""
        self.bucket_samplers = {}
        for bucket_idx, _ in enumerate(self.bucket_dims):
            # Get indices of samples in this bucket
            bucket_indices = [i for i, sample in enumerate(self.samples) 
                             if sample.get('bucket_idx', 0) == bucket_idx]
            
            if bucket_indices:
                self.bucket_samplers[bucket_idx] = SubsetRandomSampler(bucket_indices)
        
        logger.info(f"Created {len(self.bucket_samplers)} bucket-specific samplers")

    def _get_feature_count(self, file_path):
        """Helper function to load a feature file and get the count of features"""
        sample_features = torch.load(file_path, map_location='cpu')
        return sample_features.shape[0]

    def clear_cache(self):
        """Explicitly clear all caches to free RAM."""
        if self.feature_cache:
            self.logger.info("Clearing feature cache")
            self.feature_cache.clear()
        if self.cluster_assignments_cache:
            self.logger.info("Clearing cluster assignments cache")
            self.cluster_assignments_cache.clear()
        if self.dataset_pair_cache:
            self.logger.info("Clearing dataset pair cache")
            self.dataset_pair_cache.clear()
        if self.latent_cache: # New: clear latent cache
            self.logger.info("Clearing latent cache")
            self.latent_cache.clear()
        if self.clip_embedding_cache: # New: clear clip embedding cache
            self.logger.info("Clearing clip embedding cache")
            self.clip_embedding_cache.clear()
        torch.cuda.empty_cache() # Also clear CUDA cache just in case
        self.logger.info("Caches cleared.")

class CombinedBatchSampler(Sampler):
    """Combines multiple BatchSamplers to ensure each batch has consistent dimensions"""
    def __init__(self, batch_samplers):
        self.batch_samplers = batch_samplers
        self.batch_indices = []
        
        # Generate all batch indices upfront
        for sampler in self.batch_samplers:
            self.batch_indices.extend(list(sampler))
        
        # Shuffle batches
        random.shuffle(self.batch_indices)
    
    def __iter__(self):
        for batch in self.batch_indices:
            yield batch
    
    def __len__(self):
        return len(self.batch_indices) 

class BucketBatchSampler(torch.utils.data.Sampler):
    """GPU-optimized bucket batch sampler with tensor-based operations"""
    
    def __init__(self, bucket_indices, batch_size, device, shuffle=True, drop_last=True):
        """
        Args:
            bucket_indices: Dictionary of {bucket_idx: list of indices}
            batch_size: Target batch size
            device: Target device for tensor operations
            shuffle: Whether to shuffle batches
            drop_last: Whether to drop last incomplete batch
        """
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Convert indices to GPU tensors
        self.bucket_tensors = {
            bucket: torch.tensor(indices, device=device, dtype=torch.long)
            for bucket, indices in bucket_indices.items()
        }
        
        # Precompute batch counts using GPU ops
        self.batch_counts = torch.zeros(len(bucket_indices), device=device, dtype=torch.long)
        for i, (bucket, indices) in enumerate(bucket_indices.items()):
            count = len(indices) // batch_size if drop_last else math.ceil(len(indices) / batch_size)
            self.batch_counts[i] = count
            
        self.total_batches = torch.sum(self.batch_counts).item()
        
    def __iter__(self):
        # Generate batches using GPU-accelerated operations
        all_batches = []
        
        for bucket_idx, indices in self.bucket_tensors.items():
            # Shuffle on GPU if needed
            if self.shuffle:
                indices = indices[torch.randperm(len(indices), device=self.device)]
            
            # Split into batches using tensor operations
            batches = torch.split(indices, self.batch_size)
            
            if self.drop_last and len(indices) % self.batch_size != 0:
                batches = batches[:-1]
                
            all_batches.extend(batches)
        
        # Shuffle across buckets if needed
        if self.shuffle:
            perm = torch.randperm(len(all_batches), device=self.device)
            all_batches = [all_batches[i] for i in perm.cpu().numpy()]
            
        return iter(all_batches)
            
    def __len__(self):
        return self.total_batches

def create_expert_bucket_loaders(dataset, config, world_size=1, rank=0):
    """
    Create GPU-optimized loaders for each expert's data with:
    - Pinned memory buffers
    - Async GPU transfers
    - Optimized worker processes
    """
    # Use the correct local device for this process

    # Use CPU for bucket sampler to avoid NCCL conflicts
    device = torch.device('cpu')
    logger = setup_distributed_logger(name="ExpertLoaders", rank=rank)
    logger.info(f"Rank {rank}: Using CPU for bucket sampler to avoid NCCL conflicts")
    
    loader_start = time.time()
    logger.info(f"Rank {rank}: Starting DataLoader creation for {dataset.num_experts.item()} experts")
    
    # Get expert assignments directly from GPU tensor
    expert_assignments = dataset.expert_assignments.cpu().numpy()
    expert_indices = defaultdict(list)
    
    # Use vectorized operations for expert index collection
    for idx in np.nditer(np.where(expert_assignments >= 0)):
        expert_idx = expert_assignments[idx]
        expert_indices[expert_idx].append(idx.item())
    
    # Log expert distribution stats
    total_indices = sum(len(indices) for indices in expert_indices.values())
    logger.info(f"Rank {rank}: Collected {total_indices} indices across {len(expert_indices)} active experts")

    expert_loaders = {}
    expert_pbar = tqdm(
        total=len(expert_indices),
        desc="Creating Expert Loaders",
        unit="expert", 
        dynamic_ncols=True
    )
    
    for expert_idx, indices in expert_indices.items():
        # Create GPU-optimized bucket indices
        bucket_indices = defaultdict(list)
        for idx in indices:
            bucket_idx = dataset.bucket_assignments[idx].item()
            bucket_indices[bucket_idx].append(idx)
        
        logger.info(f"Rank {rank}: Expert {expert_idx} uses {len(bucket_indices)} different buckets with {len(indices)} total images")
        
        # Create GPU-accelerated sampler
        sampler_start = time.time()
        sampler = BucketBatchSampler(
            bucket_indices=bucket_indices,
            batch_size=config.expert_batch_size,
            device=device,
            shuffle=True,
            drop_last=True
        )
        sampler_time = time.time() - sampler_start
        logger.info(f"Rank {rank}: Created sampler for expert {expert_idx} in {sampler_time:.2f}s")
        
        # Configure loader with GPU optimizations
        loader_config_start = time.time()
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,  # No worker processes = no pickling needed
            pin_memory=True
        )
        loader_config_time = time.time() - loader_config_start
        
        # Warmup pipeline (no need to load directly to GPU here)
        warmup_start = time.time()
        try:
            for _ in range(1):
                next(iter(loader), None)
                break
            warmup_time = time.time() - warmup_start
            logger.info(f"Rank {rank}: Warmup for expert {expert_idx} completed in {warmup_time:.2f}s")
        except Exception as e:
            logger.warning(f"Rank {rank}: Warmup failed for expert {expert_idx}: {str(e)}. This is non-critical and training will continue.")
        
        expert_loaders[expert_idx] = loader
        logger.info(f"Rank {rank}: Created optimized loader for expert {expert_idx} "
                   f"with {len(sampler)} batches (config: {loader_config_time:.2f}s)")
        
        expert_pbar.update(1)
    
    expert_pbar.close()
    total_loader_time = time.time() - loader_start
    logger.info(f"Rank {rank}: DataLoader creation complete in {total_loader_time:.2f}s - {len(expert_loaders)} expert loaders created")
        
    return expert_loaders 
