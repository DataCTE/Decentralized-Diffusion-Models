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
    """Distributed-optimized dataset pipeline without Redis"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.device = torch.device('cpu')
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # 1. Distributed file discovery
        self.latent_dir = os.path.join(config.feature_cache_path, "latents")
        self.latent_files = self._get_distributed_latent_files()
        self.num_samples = len(self.latent_files)
        
        # 2. Sharded cluster loading
        self.cluster_dir = os.path.join(config.feature_cache_path, "clusters")
        self.expert_assignments = self._load_sharded_clusters()
        
        # 3. Distributed bucket indices
        self.bucket_assignments = self._distributed_bucket_indices()

    def _get_distributed_latent_files(self):
        """Distributed file discovery with memory-mapped index"""
        # Only rank 0 scans files
        if self.rank == 0:
            latent_path = os.path.join(self.latent_dir, "latents.mmap")
            if os.path.exists(latent_path):
                # Load precomputed index
                with open(latent_path, 'rb') as f:
                    valid_files = [line.strip().decode() for line in f]
                logger.info(f"Loaded precomputed latent index with {len(valid_files)} entries")
            else:
                # Build and cache index
                all_files = sorted(os.listdir(self.latent_dir))
                valid_files = []
                
                # Process in parallel batches
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {
                        executor.submit(
                            lambda f: (f, os.path.exists(os.path.join(self.latent_dir, f))),
                            f
                        ): f for f in all_files if f.endswith('.latent.pt')
                    }
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Global file scan"):
                        f, exists = future.result()
                        if exists:
                            valid_files.append(f)
                
                # Save mmap index as newline-separated bytes
                with open(latent_path, 'wb') as f:
                    f.write(b'\n'.join([f.encode() for f in valid_files]))
                logger.info(f"Cached latent index to {latent_path}")

            # Convert to numpy byte array for broadcasting
            file_bytes = np.array([f.encode() for f in valid_files], dtype=np.bytes_)
        else:
            file_bytes = np.empty(0, dtype=np.bytes_)
        
        # Broadcast file list from rank 0 using numpy arrays
        file_bytes = broadcast_object(file_bytes, src=0)
        return [f.decode() for f in file_bytes.tolist()]

    def _load_sharded_clusters(self):
        """Sharded cluster loading from individual files"""
        # Get all cluster files that match latent files
        cluster_files = [
            f.replace('.latent.pt', '.cluster.pt')
            for f in self.latent_files
            if os.path.exists(os.path.join(self.cluster_dir, f.replace('.latent.pt', '.cluster.pt')))
        ]
        
        # Calculate shard boundaries
        shard_size = len(cluster_files) // self.world_size
        start = self.rank * shard_size
        end = start + shard_size if self.rank != self.world_size -1 else len(cluster_files)
        
        assignments = []
        # Process files in parallel batches
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    lambda f: torch.load(os.path.join(self.cluster_dir, f), map_location='cpu'),
                    f
                ): f for f in cluster_files[start:end]
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Rank {self.rank} loading clusters"):
                try:
                    cluster_data = future.result()
                    assignments.append(cluster_data)
                except Exception as e:
                    logger.warning(f"Failed to load cluster file: {str(e)}")
                    continue
        
        # Gather all shards
        if assignments:
            assignments = torch.cat(assignments)
        else:
            assignments = torch.empty(0, dtype=torch.long)
            
        gathered = [torch.empty_like(assignments) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, assignments)
        return torch.cat(gathered).long()

    def _distributed_bucket_indices(self):
        """Optimized bucket index calculation using precomputed metadata"""
        # Load precomputed dimensions from memory-mapped array
        dim_mmap = np.load(os.path.join(self.latent_dir, "dimensions.npy"), mmap_mode='r')
        bucket_indices = torch.zeros(len(dim_mmap), dtype=torch.long)
        
        # Vectorized bucket assignment
        for i, (w, h) in enumerate(self.config.buckets):
            mask = (dim_mmap[:,0] == w) & (dim_mmap[:,1] == h)
            bucket_indices[mask] = i
        
        # Gather all indices
        gathered = [torch.empty_like(bucket_indices) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, bucket_indices)
        return torch.cat(gathered)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Memory-mapped loading with batched prefetch
        latent = torch.load(os.path.join(self.latent_dir, self.latent_files[idx]), 
                           map_location='cpu', mmap=True)
        
        # CLIP embeddings use pointer-based access
        clip_path = os.path.join(
            self.config.feature_cache_path,
            "clip_embeddings",
            self.latent_files[idx].replace('.latent.pt', '.clip_emb.pt')
        )
        clip_emb = torch.load(clip_path, map_location='cpu', mmap=True)
        
        return {
            'latent': latent,
            'clip_embedding': clip_emb,
            'bucket': self.bucket_assignments[idx],
            'expert': self.expert_assignments[idx]
        }

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
