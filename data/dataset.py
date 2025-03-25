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
import struct

import io
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import centralized utilities
from utils.distributed import is_main_process, broadcast_object, get_rank, get_local_rank, get_world_size
from utils.logging import setup_distributed_logger
from data.transforms import resize_image, normalize
import threading
import signal

# Setup logging
logger = logging.getLogger(__name__)

import math  # For BucketBatchSampler
import torch.distributed as dist

from queue import Queue, Empty
from threading import Thread, Event

def chunks(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class DDMDataset(Dataset):
    """Distributed-optimized dataset pipeline with lazy loading and prefetching"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.feature_dir = config.feature_cache_path
        self.device = torch.device('cuda')  # Use GPU by default
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # Cache directories
        self.latent_dir = os.path.join(config.feature_cache_path, "latents")
        self.clip_dir = os.path.join(config.feature_cache_path, "clip")
        self.cluster_dir = os.path.join(config.feature_cache_path, "clusters")
        self.dim_dir = os.path.join(config.feature_cache_path, "dims")
        self.bucket_dir = os.path.join(config.feature_cache_path, "buckets")
        
        # Verify directories exist
        self._verify_cache_dirs()
        
        # Load all preprocessed files for this rank
        self.latent_files = sorted(glob.glob(os.path.join(self.latent_dir, f"*_rank{self.rank}.pt")))
        self.num_samples = len(self.latent_files)
        
        # Extract base names without rank suffix for loading other features
        self.base_names = [Path(f).stem.rsplit('_rank', 1)[0] for f in self.latent_files]
        
        # Initialize memory-mapped cache
        self.cache = {}
        self.cache_size = min(100, self.num_samples)  # Larger cache size
        
        # Pre-load frequently accessed data
        logger.info(f"Pre-loading data for rank {self.rank}")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for idx in range(min(self.cache_size, self.num_samples)):
                futures.append(executor.submit(self._load_sample, idx))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Rank {self.rank} preloading"):
                idx, data = future.result()
                if data is not None:
                    self.cache[idx] = data
        
        # Load global cluster assignments to GPU
        self.expert_assignments = self._lazy_load_clusters()
        
        # Load bucket assignments to GPU
        self.bucket_assignments = self._load_bucket_assignments()
        
        logger.info(f"[Rank {self.rank}] Initialized dataset with {self.num_samples} samples")
        logger.info(f"[Rank {self.rank}] Preloaded {len(self.cache)} samples")

    def _verify_cache_dirs(self):
        """Validate cache directory structure"""
        required_dirs = {
            'latents': self.latent_dir,
            'clip': self.clip_dir,
            'clusters': self.cluster_dir,
            'dims': self.dim_dir
        }
        
        for name, path in required_dirs.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required cache directory: {name} ({path})")

    def _lazy_load_clusters(self):
        """Load precomputed cluster assignments"""
        cluster_path = os.path.join(self.cluster_dir, "final_clusters.pt")
        try:
            # Load the precomputed cluster assignments with weights_only=False
            # since we're loading numpy arrays
            cluster_assignments = torch.load(cluster_path, weights_only=False)
            
            # Convert numpy array to tensor if needed
            if isinstance(cluster_assignments, np.ndarray):
                cluster_assignments = torch.from_numpy(cluster_assignments)
            
            # Validate cluster assignments
            num_clusters = cluster_assignments.max().item() + 1
            if num_clusters != self.config.num_experts:
                logger.warning(
                    f"Found {num_clusters} clusters but config specifies {self.config.num_experts} experts"
                )
            
            # Log cluster distribution
            unique_clusters, counts = torch.unique(cluster_assignments, return_counts=True)
            for cluster, count in zip(unique_clusters.tolist(), counts.tolist()):
                logger.info(f"Cluster {cluster}: {count} samples")
            
            # Ensure cluster assignments are within valid range
            if (cluster_assignments < 0).any() or (cluster_assignments >= self.config.num_experts).any():
                raise ValueError(
                    f"Invalid cluster assignments found. Min: {cluster_assignments.min()}, "
                    f"Max: {cluster_assignments.max()}, Expected range: [0, {self.config.num_experts-1}]"
                )
            
            return cluster_assignments.cuda()
            
        except Exception as e:
            logger.error(f"Failed to load cluster assignments from {cluster_path}: {str(e)}")
            raise

    def _load_sample(self, idx):
        """Load a single sample with error handling"""
        try:
            base_name = self.base_names[idx]
            rank_suffix = f"_rank{self.rank}"
            
            data = {
                'latent': torch.load(
                    os.path.join(self.latent_dir, f"{base_name}{rank_suffix}.pt"),
                    weights_only=False
                ).to(self.device),
                'clip_embedding': torch.load(
                    os.path.join(self.clip_dir, f"{base_name}{rank_suffix}.pt"),
                    weights_only=False
                ).to(self.device),
                'dims': torch.load(
                    os.path.join(self.dim_dir, f"{base_name}{rank_suffix}.pt"),
                    weights_only=False
                ).to(self.device)
            }
            
            return idx, data
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            return idx, None

    def _load_bucket_assignments(self):
        """Load bucket assignments efficiently"""
        assignments = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for base in self.base_names:
                futures.append(executor.submit(
                    torch.load,
                    os.path.join(self.bucket_dir, f"{base}_rank{self.rank}.pt"),
                    weights_only=False
                ))
            
            for future in futures:
                try:
                    assignments.append(future.result())
                except Exception as e:
                    logger.error(f"Error loading bucket assignment: {str(e)}")
                    assignments.append(torch.tensor(0))  # Default bucket
                    
        return torch.stack(assignments).to(self.device)

    def __getitem__(self, idx):
        """Get item with cache"""
        if idx in self.cache:
            data = self.cache[idx]
        else:
            idx_data = self._load_sample(idx)
            if idx_data[1] is None:
                raise RuntimeError(f"Failed to load sample {idx}")
            data = idx_data[1]
            
            # Update cache with LRU policy
            if len(self.cache) >= self.cache_size:
                oldest_idx = min(self.cache.keys())
                del self.cache[oldest_idx]
            self.cache[idx] = data
        
        # Add assignments
        data['expert'] = self.expert_assignments[idx]
        data['bucket'] = self.bucket_assignments[idx]
        
        return data

    def __len__(self):
        return self.num_samples
        
    def __del__(self):
        """Cleanup resources"""
        pass

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
    
    def __init__(self, dataset, batch_size, bucket_indices=None, device='cpu', shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        
        # Group indices by bucket if not provided
        if bucket_indices is None:
            self.bucket_indices = defaultdict(list)
            for idx in range(len(dataset)):
                bucket = dataset.bucket_assignments[idx].item()
                self.bucket_indices[bucket].append(idx)
        else:
            self.bucket_indices = bucket_indices
        
        # Convert lists to tensors
        self.bucket_tensors = {
            bucket: torch.tensor(indices, device=self.device)
            for bucket, indices in self.bucket_indices.items()
        }
        
        # Calculate total batches
        self.total_batches = sum(
            len(indices) // batch_size if drop_last 
            else (len(indices) + batch_size - 1) // batch_size
            for indices in self.bucket_indices.values()
        )
        
        logger.info(f"Created BucketBatchSampler with {len(self.bucket_indices)} buckets")
        for bucket, indices in self.bucket_indices.items():
            logger.info(f"Bucket {bucket}: {len(indices)} samples")

    def __iter__(self):
        # Create batches for each bucket
        all_batches = []
        
        for bucket, indices in self.bucket_tensors.items():
            if self.shuffle:
                indices = indices[torch.randperm(len(indices), device=indices.device)]
            
            # Split into batches
            batches = torch.split(indices, self.batch_size)
            
            # Handle partial batches
            if self.drop_last and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]
            
            all_batches.extend(batches)
        
        # Shuffle batches if requested
        if self.shuffle:
            random.shuffle(all_batches)
        
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
    device = torch.device('cpu')
    logger = setup_distributed_logger(name="ExpertLoaders", rank=rank)

    # Debug prints for initialization
    print(f"\n[Rank {rank}] ===== EXPERT LOADER INITIALIZATION =====")
    print(f"[Rank {rank}] Total dataset samples: {len(dataset)}")
    print(f"[Rank {rank}] Dataset latent files: {len(dataset.latent_files)}")
    print(f"[Rank {rank}] First 5 expert assignments: {dataset.expert_assignments[:5]}")

    # Distributed progress tracking
    loader_pbar = tqdm(
        total=len(expert_indices),
        desc=f"[Rank {rank}] Creating Expert Loaders",
        position=rank,
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}",
        disable=not is_main_process()
    )

    loader_start = time.time()
    logger.info(f"Rank {rank}: Starting DataLoader creation for {dataset.num_experts.item()} experts")

    # Get expert assignments directly from GPU tensor
    expert_assignments = dataset.expert_assignments.cpu().numpy()
    expert_indices = defaultdict(list)

    # Use vectorized operations for expert index collection
    for idx in np.nditer(np.where(expert_assignments >= 0)):
        expert_idx = expert_assignments[idx]
        # Add index validation
        if idx >= len(dataset):
            print(f"[Rank {rank}] WARNING: Invalid index {idx} in expert {expert_idx}")
            continue
        expert_indices[expert_idx].append(idx.item())

    # Log expert distribution stats with validation
    total_indices = sum(len(indices) for indices in expert_indices.values())
    print(f"[Rank {rank}] Collected {total_indices} valid indices across {len(expert_indices)} experts")
    print(f"[Rank {rank}] Expert index ranges:")
    for expert_idx, indices in expert_indices.items():
        print(f"  Expert {expert_idx}: {len(indices)} samples")
        if indices:
            print(f"    First index: {indices[0]}, Last index: {indices[-1]}")
            if indices[-1] >= len(dataset):
                print(f"    ERROR: Last index {indices[-1]} exceeds dataset size {len(dataset)}")

    expert_loaders = {}

    for expert_idx, indices in expert_indices.items():
        print(f"\n[Rank {rank}] Processing expert {expert_idx}")
        print(f"[Rank {rank}] Total samples: {len(indices)}")
        
        # GPU-accelerated bucket index creation
        with torch.cuda.stream(torch.cuda.Stream(device=rank % torch.cuda.device_count())):
            bucket_indices = defaultdict(list)
            valid_count = 0
            for idx in indices:
                if idx >= len(dataset.bucket_assignments):
                    print(f"[Rank {rank}] WARNING: Index {idx} out of range for bucket assignments")
                    continue
                bucket_idx = dataset.bucket_assignments[idx].item()
                bucket_indices[bucket_idx].append(idx)
                valid_count += 1
            print(f"[Rank {rank}] Valid indices for bucketing: {valid_count}/{len(indices)}")

        # Distributed logging
        logger.info(f"Rank {rank}: Expert {expert_idx} processing on GPU {rank % torch.cuda.device_count()}")

        # Create GPU-accelerated sampler
        sampler_start = time.time()
        try:
            sampler = BucketBatchSampler(
                dataset=dataset,
                batch_size=config.expert_batch_size,
                shuffle=True,
                drop_last=True
            )
            print(f"[Rank {rank}] Sampler created with {len(sampler)} batches")
        except Exception as e:
            print(f"[Rank {rank}] ERROR creating sampler:")
            print(f"Exception: {str(e)}")
            print(f"Bucket indices: {list(bucket_indices.keys())}")
            raise

        # Configure loader with GPU optimizations
        loader_config_start = time.time()
        try:
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=0,
                pin_memory=True
            )
            print(f"[Rank {rank}] DataLoader created successfully")
        except Exception as e:
            print(f"[Rank {rank}] ERROR creating DataLoader:")
            print(f"Exception: {str(e)}")
            print(f"Sampler indices: {list(sampler)[0] if len(sampler) > 0 else 'empty'}")
            raise

        # Warmup pipeline
        warmup_start = time.time()
        try:
            print(f"[Rank {rank}] Warming up loader...")
            for _ in range(1):
                batch = next(iter(loader))
                print(f"[Rank {rank}] Warmup batch shapes:")
                for k, v in batch.items():
                    print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
            print(f"[Rank {rank}] Warmup successful")
        except Exception as e:
            print(f"[Rank {rank}] Warmup failed:")
            print(f"Exception: {str(e)}")
            print(f"Failing indices: {list(sampler)[0] if len(sampler) > 0 else 'empty'}")
            if hasattr(e, 'args') and len(e.args) > 1:
                print(f"Problematic index: {e.args[1]}")
            print("Skipping warmup, continuing with training...")

        expert_loaders[expert_idx] = loader
        loader_pbar.update(1)

    loader_pbar.close()
    print(f"\n[Rank {rank}] ===== LOADER INITIALIZATION COMPLETE =====")
    print(f"[Rank {rank}] Created {len(expert_loaders)} expert loaders")
    print(f"[Rank {rank}] Total initialization time: {time.time() - loader_start:.2f}s")

    return expert_loaders 
