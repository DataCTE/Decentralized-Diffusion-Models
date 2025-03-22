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
import signal


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

class DDMDataset(Dataset):
    """GPU-optimized dataset pipeline for decentralized diffusion models with precomputed latents"""
    
    def __init__(self, config, split='train', transforms=None, hf_split=None):
        self.config = config
        self.split = split
        self.device = torch.device('cpu')
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Initialize paths
        self.feature_cache_path = config.feature_cache_path
        self.feature_path = os.path.join(self.feature_cache_path, "features")
        self.cluster_path = os.path.join(self.feature_cache_path, "clusters")
        self.latent_path = os.path.join(self.feature_cache_path, "latents")
        self.clip_embedding_path = os.path.join(self.feature_cache_path, "clip_embeddings")
        self.dim_cache_path = os.path.join(self.feature_cache_path, "dimensions")

        # 1. Latent-first initialization -------------------------------------------------
        latent_files = sorted(os.listdir(self.latent_path))
        self.image_files = [f.replace(".latent.pt", "") for f in latent_files]
        latent_basenames = set(self.image_files)
        
        # 2. Parallel filtering with progress tracking ----------------------------------------
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Create futures with task descriptions
            futures = {
                executor.submit(
                    lambda: [
                        f for f in glob.glob(os.path.join(self.cluster_path, "*.cluster.pt"))
                        if os.path.basename(f).replace(".cluster.pt", "") in latent_basenames
                    ]
                ): "Clusters",
                executor.submit(
                    lambda: [
                        f for f in glob.glob(os.path.join(self.dim_cache_path, "*.pt"))
                        if os.path.basename(f).replace(".pt", "") in latent_basenames
                    ]
                ): "Dimensions",
                executor.submit(
                    lambda: sorted([
                        f for f in glob.glob(os.path.join(self.clip_embedding_path, "*.clip_emb.pt"))
                        if os.path.basename(f).replace(".clip_emb.pt", "") in latent_basenames
                    ])
                ): "CLIP Embeddings"
            }

            # Progress bar setup
            with tqdm(total=len(futures), desc="Filtering dependencies") as pbar:
                results = {}
                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        results[task_name] = future.result()
                        pbar.set_postfix_str(f"Completed {task_name}")
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing {task_name}: {str(e)}")
                        raise

            # Assign results with type checking
            self.cluster_files = results.get("Clusters", [])
            self.dimension_files = results.get("Dimensions", [])
            self.clip_embedding_files = results.get("CLIP Embeddings", [])

        # 3. Validation gate -------------------------------------------------------------
        assert len(self.image_files) == len(self.dimension_files), \
            f"Latent/dimension mismatch: {len(self.image_files)} vs {len(self.dimension_files)}"
        
        # 4. Optimized dimension loading -------------------------------------------------
        def load_dims_batch(file_batch):
            return [torch.load(f, map_location='cpu') for f in file_batch]

        # Preserve file order using latent file ordering
        dim_file_map = {os.path.basename(f).replace(".pt", ""): f for f in self.dimension_files}
        ordered_dim_files = [dim_file_map[base] for base in self.image_files if base in dim_file_map]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            file_batches = [ordered_dim_files[i:i+512] for i in range(0, len(ordered_dim_files), 512)]
            
            with tqdm(total=len(ordered_dim_files), 
                     desc="Loading dimensions",
                     unit="file",
                     dynamic_ncols=True,
                     bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
                futures = [executor.submit(load_dims_batch, batch) for batch in file_batches]
                dim_cache = []
                for future in as_completed(futures):
                    dim_cache.extend(future.result())
                    pbar.update(len(future.result()))
                
        self.dim_cache = torch.stack(dim_cache)

        # 5. Final validation ------------------------------------------------------------
        assert len(self.image_files) == len(self.dim_cache), \
            f"Final mismatch: {len(self.image_files)} latents vs {len(self.dim_cache)} dimensions"

        # 6. Initialize core dataset properties before any broadcasting
        self.caption_files = [os.path.join(config.dataset_path, f+".txt") for f in self.image_files]
        self.bucket_dims = torch.tensor(config.buckets, dtype=torch.float32)
        self._latent_loading_lock = defaultdict(threading.Lock)
        
        # Initialize caches
        self.cluster_cache = OrderedDict()
        self.feature_cache = OrderedDict()
        self.latent_cache = OrderedDict()
        self.clip_embedding_cache = OrderedDict()
        self.cache_size = 5
        self.clip_embedding_cache_max_size = 5

        # 7. Calculate bucket assignments
        self._init_buckets()  # Now creates self.bucket_assignments

        # 8. Broadcast all necessary data together
        if is_main_process():
            # Calculate and store cumulative feature counts before broadcasting
            self.cumulative_feature_counts = self._calculate_cumulative_counts(self.clip_embedding_files, self.clip_embedding_path)
            
            # Add bucket_assignments to broadcast data
            broadcast_data = (
                self.image_files,
                self.caption_files,
                self.dim_cache,
                self.clip_embedding_files,
                self.cumulative_feature_counts,
                self.bucket_assignments,
                self.bucket_dims
            )
            broadcast_object(broadcast_data)
        else:
            (self.image_files,
             self.caption_files,
             self.dim_cache,
             self.clip_embedding_files,
             self.cumulative_feature_counts,
             self.bucket_assignments,
             self.bucket_dims) = broadcast_object(None)

        # 9. Final validation
        self._validate_distributed_state()

        # Load dimension cache
        self.dim_cache_path = os.path.join(self.feature_cache_path, "dimensions") # Path to dimensions directory
        logger.info(f"Dimensions path: {self.dim_cache_path}") # Log dimensions path - removed rank info

        # After line 194
        assert len(self.image_files) == len(self.bucket_assignments), \
            f"Broadcast mismatch: {len(self.image_files)} vs {len(self.bucket_assignments)}"

        # 10. Add final synchronization check (NEW)
        self._verify_global_consistency()

    def __getitem__(self, idx):
        return {
            'latent': self._load_latent(idx),
            'expert': self._load_cluster(idx),
            'features': self._load_feature(idx),
            'clip_embedding': self._load_clip_embedding(idx),
            'bucket': self.bucket_assignments[idx]
        }

    def _load_cluster(self, idx):
        # Get image filename and construct cluster path
        image_file = self.image_files[idx]
        cluster_file = os.path.join(self.cluster_path, f"{os.path.basename(image_file)}.cluster.pt")
        
        # LRU Caching Mechanism
        if cluster_file not in self.cluster_cache:
            # Remove oldest entry if cache full
            if len(self.cluster_cache) >= self.cache_size:
                self.cluster_cache.popitem(last=False) 
            
            # Load cluster data and add to cache
            self.cluster_cache[cluster_file] = torch.load(cluster_file)
        
        return self.cluster_cache[cluster_file]

    def _load_feature(self, idx):
        # Similar structure but for features
        image_file = self.image_files[idx]
        feature_file = os.path.join(self.feature_path, f"{os.path.basename(image_file)}.pt")
        
        if feature_file not in self.feature_cache:
            if len(self.feature_cache) >= self.cache_size:
                self.feature_cache.popitem(last=False)
            
            self.feature_cache[feature_file] = torch.load(feature_file)
        
        return self.feature_cache[feature_file]

    def _init_buckets(self):
        """Distributed-safe bucket initialization with full data sync"""
        # Calculate bucket assignments first
        self.bucket_assignments = self._calculate_bucket_assignments()
        
        if is_main_process():
            # Add progress bar for validation/train filtering
            if self.split == 'val' and _GLOBAL_DATASET_CACHE["initialized"]:
                val_size = getattr(self.config, 'val_size', 1000)
                if val_size < len(self.image_files):
                    all_indices = np.arange(len(self.image_files))
                    np.random.seed(42)
                    
                    # Add progress bar for validation sample selection
                    with tqdm(total=val_size, desc="Selecting validation samples") as pbar:
                        val_indices = []
                        while len(val_indices) < val_size:
                            batch = np.random.choice(all_indices, size=min(1000, val_size-len(val_indices)), replace=False)
                            val_indices.extend(batch.tolist())
                            pbar.update(len(batch))
                        val_indices = np.array(val_indices[:val_size])
                    
                    # Add filtering progress bar
                    with tqdm(total=len(val_indices), desc="Filtering validation files") as pbar:
                        self.image_files = [self.image_files[i] for i in val_indices]
                        self.caption_files = [self.caption_files[i] for i in val_indices]
                        self.dim_cache = self.dim_cache[val_indices]
                        pbar.update(len(val_indices))
                        
                # After filtering image_files, re-filter clip_embedding_files with proper extension handling
                current_image_basenames = {os.path.basename(f) for f in self.image_files}
                self.clip_embedding_files = [
                    f for f in glob.glob(os.path.join(self.clip_embedding_path, "*.clip_emb.pt"))
                    if os.path.basename(f).replace(".clip_emb.pt", "") in current_image_basenames
                ]
                
                # Add validation before recalculating counts
                assert len(self.clip_embedding_files) > 0, "No CLIP embedding files match filtered dataset"
                
                # Recalculate cumulative counts AFTER filtering with proper path handling
                self.cumulative_feature_counts = self._calculate_cumulative_counts(
                    self.clip_embedding_files,
                    self.clip_embedding_path
                )
                
                # Verify counts match before broadcasting
                total_features = self.cumulative_feature_counts[-1].item()
                assert total_features == len(self.image_files), \
                    f"CLIP features ({total_features}) don't match images ({len(self.image_files)})"
                
            elif self.split == 'train' and _GLOBAL_DATASET_CACHE["initialized"]:
                val_size = getattr(self.config, 'val_size', 1000)
                if val_size > 0 and val_size < len(self.image_files):
                    all_indices = np.arange(len(self.image_files))
                    np.random.seed(42)
                    
                    # Add progress bar for train sample selection
                    with tqdm(total=len(all_indices)-val_size, desc="Selecting training samples") as pbar:
                        train_indices = []
                        for i in all_indices:
                            if i not in val_indices:
                                train_indices.append(i)
                                pbar.update(1)
                        train_indices = np.array(train_indices)
                    
                    # Add filtering progress bar
                    with tqdm(total=len(train_indices), desc="Filtering training files") as pbar:
                        self.image_files = [self.image_files[i] for i in train_indices]
                        self.caption_files = [self.caption_files[i] for i in train_indices]
                        self.dim_cache = self.dim_cache[train_indices]
                        pbar.update(len(train_indices))

                # After filtering image_files, re-filter clip_embedding_files with proper extension handling
                current_image_basenames = {os.path.basename(f) for f in self.image_files}
                self.clip_embedding_files = [
                    f for f in glob.glob(os.path.join(self.clip_embedding_path, "*.clip_emb.pt"))
                    if os.path.basename(f).replace(".clip_emb.pt", "") in current_image_basenames
                ]
                
                # Add validation before recalculating counts
                assert len(self.clip_embedding_files) > 0, "No CLIP embedding files match filtered dataset"
                
                # Recalculate cumulative counts AFTER filtering with proper path handling
                self.cumulative_feature_counts = self._calculate_cumulative_counts(
                    self.clip_embedding_files,
                    self.clip_embedding_path
                )
                
                # Verify counts match before broadcasting
                total_features = self.cumulative_feature_counts[-1].item()
                assert total_features == len(self.image_files), \
                    f"CLIP features ({total_features}) don't match images ({len(self.image_files)})"

            # Recalculate cumulative counts AFTER filtering
            self.cumulative_feature_counts = self._calculate_cumulative_counts(
                self.clip_embedding_files, 
                self.clip_embedding_path
            )
            
            # Update broadcast data with filtered clip_embedding_files
            broadcast_data = (
                self.image_files,
                self.caption_files,
                self.dim_cache,
                self.clip_embedding_files,  # Now contains filtered files
                self.cumulative_feature_counts,
                self.bucket_assignments
            )
            broadcast_object(broadcast_data)
        else:
            # Receive tensor directly
            (self.image_files, 
             self.caption_files,
             self.dim_cache,  # Receive as tensor
             self.clip_embedding_files,
             self.cumulative_feature_counts,
             self.bucket_assignments) = broadcast_object(None)  # Add bucket_assignments here

        # Final validation (critical for distributed sync)
        assert len(self.image_files) == len(self.bucket_assignments), \
            f"Dataset/Bucket mismatch: {len(self.image_files)} vs {len(self.bucket_assignments)}"
        assert self.dim_cache.shape[0] == len(self.image_files), \
            f"Dimension cache mismatch: {self.dim_cache.shape[0]} vs {len(self.image_files)}"

    def _calculate_bucket_assignments(self):
        """Calculate bucket assignments based on image dimensions"""
        # Convert bucket dims to tensor for vectorized operations
        bucket_dims_tensor = self.bucket_dims
        
        # Get image dimensions from cache [width, height]
        image_dims = self.dim_cache[:, [1, 0]].float()  # Swap W/H if needed
        
        # Calculate distances to all buckets
        diffs = bucket_dims_tensor.unsqueeze(0) - image_dims.unsqueeze(1)
        distances = torch.norm(diffs, dim=2)
        
        # Find closest bucket for each image
        _, assignments = torch.min(distances, dim=1)
        return assignments

    def _load_latent(self, idx):
        """Load precomputed latent tensor from disk, with caching and thread safety"""
        latent_file = self.image_files[idx] + ".latent.pt"
        latent_path = os.path.join(self.latent_path, latent_file)

        with self._latent_loading_lock[idx]:  # Thread lock for loading
            if latent_path in self.latent_cache:  # Check cache first
                latent = self.latent_cache[latent_path]  # Load from cache
            else:
                latent = torch.load(latent_path, map_location=self.device)  # Load from disk
                self.latent_cache[latent_path] = latent  # Update cache
                
                # Update progress bar on main process
                if is_main_process():
                    # Initialize progress bar if it doesn't exist
                    if not hasattr(self, '_latent_progress'):
                        self._latent_progress = tqdm(
                            total=len(self.image_files),
                            desc="Loading latent tensors",
                            unit="file",
                            position=3,  # Position below other progress bars
                            leave=False  # Don't persist after completion
                        )
                    
                    # Update with number of cached items (more accurate than +=1)
                    cached = len(self.latent_cache)
                    self._latent_progress.n = cached
                    self._latent_progress.refresh()
                
                if len(self.latent_cache) > self.cache_size:  # LRU eviction
                    self.latent_cache.popitem(last=False)  # Remove LRU item

            logger.debug(f"Loaded latent tensor shape: {latent.shape} from {latent_path}")
        return latent

    def _calculate_cumulative_counts(self, files, path):
        """Optimized cumulative counts calculation with parallel processing"""
        # Pre-allocate tensor for counts
        counts = torch.zeros(len(files), dtype=torch.long)
        
        # Use maximum available workers
        num_workers = min(32, os.cpu_count())
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create future map with proper filename construction
            future_to_idx = {
                executor.submit(
                    self._get_feature_count, 
                    os.path.join(path, f"{f}.clip_emb.pt")  # Add extension here
                ): i 
                for i, f in enumerate(files)
            }
            
            # Progress bar with manual updates
            with tqdm(total=len(files), desc="Calculating feature counts") as pbar:
                # Process completed futures as they come in
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        counts[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing file {files[idx]}: {e}")
                        counts[idx] = 0
                    pbar.update(1)

        # Calculate cumulative sum with initial zero
        cumulative_counts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, dim=0)])
        
        return cumulative_counts

    def _get_feature_count(self, file_path):
        """Helper function to load a feature file and get the count of features"""
        sample_features = torch.load(file_path, map_location='cpu')
        return sample_features.shape[0]

    def _load_clip_embedding(self, idx):
        """Load precomputed CLIP embedding tensor from file, using cache"""
        # Add bounds checking
        if idx >= self.cumulative_feature_counts[-1]:
            self.logger.error(f"Requested index {idx} exceeds total features {self.cumulative_feature_counts[-1]}")
            return torch.zeros_like(next(iter(self.clip_embedding_cache.values()))[0])

        # Find the first index where cumulative_counts > idx
        file_idx = torch.searchsorted(self.cumulative_feature_counts, idx, right=True).item() - 1
        file_idx = max(0, min(file_idx, len(self.clip_embedding_files) - 1))

        # Calculate index within file
        idx_in_file = idx - self.cumulative_feature_counts[file_idx].item()
        if idx_in_file < 0:
            self.logger.error(f"Negative index calculation: {idx} - {self.cumulative_feature_counts[file_idx]} = {idx_in_file}")
            return torch.zeros_like(next(iter(self.clip_embedding_cache.values()))[0])

        file_path = os.path.join(self.clip_embedding_path, self.clip_embedding_files[file_idx])

        # Locking and caching mechanism
        if file_path not in self._clip_embedding_loading_lock:
            self._clip_embedding_loading_lock[file_path] = threading.Lock()
            
        with self._clip_embedding_loading_lock[file_path]:
            if file_path in self.clip_embedding_cache:
                embeddings = self.clip_embedding_cache[file_path]
            else:
                try:
                    embeddings = torch.load(file_path, map_location='cpu')
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
                    return torch.zeros(self.config.clip_embedding_dim)
                
                self.clip_embedding_cache[file_path] = embeddings
                
                # Update progress
                if is_main_process():
                    if not hasattr(self, '_clip_progress'):
                        self._clip_progress = tqdm(
                            total=len(self.clip_embedding_files),
                            desc="Loading CLIP embeddings",
                            unit="file",
                            position=2
                        )
                    self._clip_progress.update(1)
                
                # LRU eviction
                if len(self.clip_embedding_cache) > self.clip_embedding_cache_max_size:
                    self.clip_embedding_cache.popitem(last=False)

            # Final bounds check
            if idx_in_file >= len(embeddings):
                self.logger.error(f"Index {idx_in_file} out of range for {file_path} (size {len(embeddings)})")
                return torch.zeros_like(embeddings[0])

            return embeddings[idx_in_file]

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

    def _load_dimension_cache(self):
        """Load dimension cache with progress tracking"""
        self.dimension_files = sorted(glob.glob(os.path.join(
            self.dim_cache_path, "*.pt"
        )))
        
        logger.info(f"Loading {len(self.dimension_files)} dimension files...")
        
        dim_cache_list = []
        with tqdm(total=len(self.dimension_files), 
                 desc="Loading dimension files") as pbar:
            # Process in batches
            file_batches = [self.dimension_files[i:i+512] 
                           for i in range(0, len(self.dimension_files), 512)]
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for batch in file_batches:
                    futures.append(executor.submit(
                        lambda x: [torch.load(f) for f in x],
                        batch
                    ))
                
                for future in as_completed(futures):
                    dim_cache_list.extend(future.result())
                    pbar.update(len(batch))
        
        return torch.stack(dim_cache_list)

    def _validate_distributed_state(self):
        """Validate synchronized state across all processes"""
        assert len(self.image_files) == len(self.bucket_assignments), \
            f"Image/bucket mismatch: {len(self.image_files)} vs {len(self.bucket_assignments)}"
        assert self.dim_cache.shape[0] == len(self.image_files), \
            f"Dimension cache mismatch: {self.dim_cache.shape[0]} vs {len(self.image_files)}"
        assert torch.allclose(self.bucket_dims, torch.tensor(self.config.buckets, dtype=torch.float32)), \
            "Bucket dimensions mismatch between config and loaded data"

    def _verify_global_consistency(self):
        """Final distributed consistency verification"""
        if is_main_process():
            # Calculate total expected samples
            total_samples = len(self.image_files)
            
            # Verify CLIP embeddings coverage with explicit file checking
            valid_clip_files = [
                f for f in self.clip_embedding_files
                if os.path.exists(os.path.join(self.clip_embedding_path, f))
            ]
            self.clip_embedding_files = valid_clip_files
            
            # Recalculate with verified files
            self.cumulative_feature_counts = self._calculate_cumulative_counts(
                self.clip_embedding_files,
                self.clip_embedding_path
            )
            
            # Final alignment check
            total_clip_features = self.cumulative_feature_counts[-1].item()
            if total_samples != total_clip_features:
                self.logger.warning(f"Adjusting dataset size to match CLIP features ({total_clip_features})")
                self.image_files = self.image_files[:total_clip_features]
                self.dim_cache = self.dim_cache[:total_clip_features]

            # Create verification payload with adjusted counts
            verification_data = (len(self.image_files), self.cumulative_feature_counts[-1].item())
            broadcast_object(verification_data)
        else:
            main_total, main_clip_total = broadcast_object(None)
            
            # Trim datasets on other ranks to match main
            self.image_files = self.image_files[:main_total]
            self.dim_cache = self.dim_cache[:main_total]
            self.cumulative_feature_counts = torch.tensor([main_clip_total])

        # Final assertion with synchronized values
        assert len(self.image_files) == self.cumulative_feature_counts[-1].item(), \
            f"Final mismatch: {len(self.image_files)} vs {self.cumulative_feature_counts[-1].item()}"

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
    """Groups samples by bucket dimensions for efficient batching"""
    
    def __init__(self, bucket_indices, batch_size, device, shuffle=True, drop_last=True):
        # Add missing attribute assignments
        self.shuffle = shuffle
        self.device = device
        self.drop_last = drop_last
        self.batch_size = batch_size
        
        # Existing initialization
        self.bucket_tensors = {
            bucket: torch.tensor(indices, device=device, dtype=torch.long)
            for bucket, indices in bucket_indices.items()
        }
        
        # Precomputes number of batches per bucket
        self.batch_counts = torch.tensor([
            len(indices) // batch_size if drop_last 
            else math.ceil(len(indices) / batch_size)
            for indices in bucket_indices.values()
        ], device=device)

    def __iter__(self):
        # GPU-accelerated shuffling and batching
        all_batches = []
        for bucket_idx, indices in self.bucket_tensors.items():
            # Shuffle on GPU using tensor operations
            if self.shuffle:  # Now using properly initialized attribute
                indices = indices[torch.randperm(len(indices), device=self.device)]
            
            # Split into batches using tensor slicing
            batches = torch.split(indices, self.batch_size)
            
            # Handle partial batch using initialized attribute
            if self.drop_last and (len(indices) % self.batch_size != 0):
                batches = batches[:-1]
            
            all_batches.extend(batches)
        
        # Final shuffle across buckets using initialized attribute
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
