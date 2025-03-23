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

def chunks(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class DDMDataset(Dataset):
    """Lazy-loading dataset pipeline with minimal memory footprint"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.device = torch.device('cpu')
        
        # 1. Parallel latent file verification with batch processing
        self.latent_dir = os.path.join(config.feature_cache_path, "latents")
        self.latent_files = self._get_valid_latent_files_batched(batch_size=1000)
        self.num_samples = len(self.latent_files)
        
        # 2. Memory-mapped cluster loading with batched processing
        self.cluster_dir = os.path.join(config.feature_cache_path, "clusters")
        self.expert_assignments = self._load_cluster_assignments_mmap()
        
        # 3. Parallel bucket index precomputation with caching
        self.bucket_assignments = self._precompute_bucket_indices_cached()

    def _get_valid_latent_files_batched(self, batch_size=1000):
        """Batch-process file verification with memory mapping"""
        all_files = sorted(os.listdir(self.latent_dir))
        latent_files = []
        
        # Process in batches to balance memory and speed
        for batch in tqdm(chunks(all_files, batch_size), desc="Scanning latent files"):
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        lambda f: (f, os.path.exists(os.path.join(self.latent_dir, f))),
                        f
                    ): f for f in batch if f.endswith('.latent.pt')
                }
                
                for future in as_completed(futures):
                    f, exists = future.result()
                    if exists:
                        latent_files.append(f)
        
        return latent_files

    def _load_cluster_assignments_mmap(self):
        """Memory-mapped cluster loading with batched verification"""
        cluster_paths = [
            os.path.join(self.cluster_dir, f.replace('.latent.pt', '.cluster.pt'))
            for f in self.latent_files
        ]
        
        # Pre-verify all cluster files in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(os.path.exists, p): p for p in cluster_paths}
            valid_paths = []
            
            with tqdm(total=len(futures), desc="Verifying cluster files") as pbar:
                for future in as_completed(futures):
                    path = futures[future]
                    if future.result():
                        valid_paths.append(path)
                    pbar.update(1)
        
        # Memory map all valid cluster files
        assignments = []
        for path in tqdm(valid_paths, desc="Loading clusters"):
            try:
                mmap = torch.load(path, map_location='cpu', mmap=True)
                assignments.append(mmap)
            except:
                continue  # Skip corrupted files
        
        return torch.cat(assignments).long()

    def _precompute_bucket_indices_cached(self):
        """Cached bucket index calculation with parallel processing"""
        cache_path = os.path.join(self.config.feature_cache_path, "bucket_cache.pt")
        
        if os.path.exists(cache_path):
            # Load precomputed bucket indices
            return torch.load(cache_path)
        
        # Compute and cache if not exists
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._parse_bucket_from_filename, f): f
                for f in self.latent_files
            }
            
            bucket_indices = []
            with tqdm(total=len(futures), desc="Calculating buckets") as pbar:
                for future in as_completed(futures):
                    bucket_indices.append(future.result())
                    pbar.update(1)
            
            tensor_indices = torch.tensor(bucket_indices, dtype=torch.long)
            torch.save(tensor_indices, cache_path)
        
        return tensor_indices

    def _parse_bucket_from_filename(self, filename):
        """Optimized filename parsing with dimension pattern cache"""
        try:
            # Extract dimensions from filename pattern: {width}x{height}_{hash}.latent.pt
            dim_part = filename.split('_', 1)[0]
            w, h = map(int, dim_part.split('x', 1))
            return next(i for i, (bw, bh) in enumerate(self.config.buckets) if bw == w and bh == h)
        except:
            return 0  # Fallback to first bucket

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load latent with immediate context cleanup
        latent = self._load_latent(idx)
        
        # Load CLIP embedding with immediate context cleanup
        clip_emb = self._load_clip(idx)
        
        return {
            'latent': latent,
            'clip_embedding': clip_emb,
            'bucket': self.bucket_assignments[idx],
            'expert': self.expert_assignments[idx]
        }

    def _load_latent(self, idx):
        """Load and immediately release file handle"""
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        try:
            latent = torch.load(latent_path)
            # Explicit cleanup
            del locals()['latent_path']  # Release file path reference
            return latent
        except Exception as e:
            raise RuntimeError(f"Failed to load latent at index {idx}: {str(e)}")

    def _load_clip(self, idx):
        """Load and immediately release file handle"""
        clip_path = os.path.join(
            self.config.feature_cache_path,
            "clip_embeddings",
            self.latent_files[idx].replace('.latent.pt', '.clip_emb.pt')
        )
        try:
            clip_emb = torch.load(clip_path)
            # Explicit cleanup
            del locals()['clip_path']  # Release file path reference
            return clip_emb
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP embedding at index {idx}: {str(e)}")

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
