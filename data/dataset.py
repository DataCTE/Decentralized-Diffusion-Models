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

def chunks(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class DDMDataset(Dataset):
    """Distributed-optimized dataset pipeline without Redis"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.feature_dir = config.feature_cache_path
        self.device = torch.device('cpu')
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # 1. Distributed file discovery
        self.latent_dir = os.path.join(config.feature_cache_path, "latents")
        self.clip_dir = os.path.join(config.feature_cache_path, "clip")
        self.cluster_dir = os.path.join(config.feature_cache_path, "clusters")
        self.dim_dir = os.path.join(config.feature_cache_path, "dims")
        self.bucket_dir = os.path.join(config.feature_cache_path, "buckets")
        
        # Verify all required directories exist
        self._verify_cache_dirs()
        
        # Initialize latent files first
        self.latent_files = sorted([
            f for f in os.listdir(self.latent_dir) 
            if f.endswith('.latent.pt')
        ])
        self.num_samples = len(self.latent_files)
        
        # Now load manifest since latent_files is initialized
        self.manifest = self._load_manifest()
        
        # 2. Sharded cluster loading
        self.expert_assignments = self._load_sharded_clusters()
        
        # 3. Distributed bucket indices
        self.bucket_assignments = self._distributed_bucket_indices()
        
        # Initialize memory maps
        self._init_memory_maps()
        
        # Add validation for expert assignments
        if self.expert_assignments is not None:
            logger.info(f"[Rank {self.rank}] Loaded expert assignments with shape: {self.expert_assignments.shape}")
            if torch.isnan(self.expert_assignments).any():
                raise ValueError(f"[Rank {self.rank}] NaN values found in expert assignments")
            if len(self.expert_assignments) == 0:
                raise ValueError(f"[Rank {self.rank}] Empty expert assignments loaded")
            if self.expert_assignments.device.type != 'cuda':
                logger.warning(f"[Rank {self.rank}] Expert assignments not on CUDA device")
                self.expert_assignments = self.expert_assignments.cuda()
        else:
            raise ValueError(f"[Rank {self.rank}] Failed to load expert assignments")

        # Add validation after initialization
        print(f"[Rank {self.rank}] Final dataset stats:")
        print(f"Total samples: {len(self)}")
        print(f"Latent files count: {len(self.latent_files)}")
        print(f"First latent path: {os.path.join(self.latent_dir, self.latent_files[0])}")
        print(f"Last latent path: {os.path.join(self.latent_dir, self.latent_files[-1])}")

        # Validate indices
        if len(self.latent_files) > 0:
            try:
                _ = self[0]
                _ = self[len(self)-1]
                print(f"[Rank {self.rank}] Initial samples loaded successfully")
            except Exception as e:
                print(f"[Rank {self.rank}] Initial sample loading failed: {str(e)}")
                raise

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _verify_cache_dirs(self):
        """Validate cache directory structure"""
        required_dirs = {
            'latents': self.latent_dir,
            'clip_embeddings': self.clip_dir,
            'clusters': self.cluster_dir,
            'dimensions': self.dim_dir
        }
        
        for name, path in required_dirs.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing required cache directory: {name} ({path})"
                )
            if not any(os.scandir(path)):
                raise ValueError(
                    f"Cache directory {name} ({path}) is empty. "
                    "Please generate features first."
                )

    def _get_distributed_latent_files(self):
        """Distributed file discovery with proper device initialization"""
        # Ensure distributed is initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        device = torch.device(f'cuda:{self.rank}')  # Use explicit device assignment
        torch.cuda.set_device(device) # Ensure device is set for this process

        if self.rank == 0:
            # Use generator to avoid loading all filenames into memory
            count = sum(1 for _ in os.scandir(self.latent_dir)
                       if _.name.endswith('.latent.pt'))
            count_tensor = torch.tensor([count], dtype=torch.long, device=device)
        else:
            count_tensor = torch.zeros(1, dtype=torch.long, device=device)

        # Broadcast using NCCL backend
        torch.distributed.broadcast(count_tensor, src=0)
        
        # Generate virtual filenames to avoid storing actual paths
        return [f"{i:08d}.latent.pt" for i in range(count_tensor.item())]

    def _load_sharded_clusters(self):
        """Load clusters from single file with distributed sharding"""
        cluster_path = os.path.join(self.cluster_dir, "final_clusters.pt")
        
        if self.rank == 0:
            try:
                # Explicitly set weights_only=False for numpy array loading
                all_clusters = torch.load(cluster_path, weights_only=False)
                if not isinstance(all_clusters, torch.Tensor):
                    all_clusters = torch.tensor(all_clusters)
                
                # Convert to CUDA tensor for efficient operations
                all_clusters = all_clusters.cuda()
                
                # Calculate chunk size and create list of chunks
                chunk_size = len(all_clusters) // self.world_size
                chunks = [all_clusters[i:i + chunk_size] for i in range(0, len(all_clusters), chunk_size)]
                
                # Handle any remaining elements
                if len(chunks) < self.world_size:
                    chunks.extend([chunks[-1].clone() for _ in range(self.world_size - len(chunks))])
                elif len(chunks) > self.world_size:
                    chunks = chunks[:self.world_size]
                    
                # Get size of first chunk to broadcast
                chunk_size = chunks[0].size()
                
                # Ensure all chunks are the same size
                for i in range(1, len(chunks)):
                    if chunks[i].size(0) < chunks[0].size(0):
                        chunks[i] = torch.cat([chunks[i], chunks[i][-1].unsqueeze(0).repeat(chunks[0].size(0) - chunks[i].size(0), 1)])
                    elif chunks[i].size(0) > chunks[0].size(0):
                        chunks[i] = chunks[i][:chunks[0].size(0)]
                    
            except Exception as e:
                logger.error(f"Error loading clusters on rank 0: {str(e)}")
                chunk_size = None
                chunks = None
        else:
            chunk_size = None
            chunks = None

        # Broadcast chunk size from rank 0 to all ranks
        if torch.distributed.is_initialized():
            if self.rank == 0:
                chunk_size = torch.tensor(list(chunk_size), dtype=torch.long, device='cuda')
            else:
                chunk_size = torch.zeros(2, dtype=torch.long, device='cuda')
            torch.distributed.broadcast(chunk_size, src=0)
            
            # Initialize local_clusters with proper size on all ranks
            local_clusters = torch.zeros(chunk_size.tolist(), device='cuda')
            
            # Scatter the chunks
            try:
                # Convert tuple to list for scatter operation
                scatter_list = chunks if self.rank == 0 else None
                dist.scatter(local_clusters, scatter_list, src=0)
                logger.info(f"[Rank {self.rank}] Successfully received cluster chunk of shape {local_clusters.shape}")
                return local_clusters
                
            except Exception as e:
                logger.error(f"Error in scatter on rank {self.rank}: {str(e)}")
                raise

        else:
            logger.warning("Distributed not initialized, returning empty clusters")
            return torch.tensor([], device='cuda')

    def _distributed_bucket_indices(self):
        """Load precomputed bucket indices with rank suffixes"""
        bucket_files = sorted([
            f for f in os.listdir(self.bucket_dir)
            if f.endswith('.pt') and '_rank' in f
        ])
        
        # Load and concatenate all bucket indices
        indices = []
        for fname in bucket_files:
            indices.append(torch.load(os.path.join(self.bucket_dir, fname)))
        
        return torch.cat(indices)

    def __len__(self):
        return self.num_samples

    def _init_memory_maps(self):
        """Initialize memory maps for all data types"""
        # Create unified memory map index
        index_path = os.path.join(self.config.feature_cache_path, "mmap_index.bin")
        self.mmap_handles = self._create_memory_map(index_path)
        
        # Preload first 10% of data for each type
        self._warmup_cache()

    def _create_memory_map(self, index_path):
        """Create memory map index with validation"""
        # Get file lists with consistent sorting - no truncation needed
        self.latent_files = sorted([f.name for f in Path(self.latent_dir).glob("*.latent.pt")])
        self.clip_files = sorted([f.name for f in Path(self.clip_dir).glob("*.clip_emb.pt")])
        self.cluster_files = sorted([f.name for f in Path(self.cluster_dir).glob("*.cluster.pt")])
        self.dim_files = sorted([f.name for f in Path(self.dim_dir).glob("*.pt")])

        # All features guaranteed aligned by preprocessor
        self.num_samples = len(self.latent_files)
        
        logger.info(f"Loaded aligned features: {self.num_samples} samples")

        # Build index only if needed - use first feature type as reference
        if not os.path.exists(index_path) and is_main_process():
            with open(index_path, 'wb') as f:
                for fname in self.latent_files:
                    base = Path(fname).stem
                    # Write placeholder values since sizes are known
                    f.write(struct.pack('Q', 0)) 

        # Load memory map with additional validation
        mmap = np.memmap(index_path, mode='r', dtype=np.uint64)
        return mmap

    def _warmup_cache(self):
        """Thread-safe cache warming with proper synchronization"""
        # Synchronize processes before starting
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        warmup_pbar = tqdm(
            total=len(self)//10,
            desc="Warming cache" + (f" (Rank {self.rank})" if self.rank != 0 else ""),
            leave=False,
            disable=not is_main_process(),
            position=0  # Unified position for all ranks
        )

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for idx in range(0, len(self), len(self)//10):
                futures.append(executor.submit(self._load_item, idx))
            
            for future in as_completed(futures):
                future.result()
                warmup_pbar.update(1)

        warmup_pbar.close()

    def _load_item(self, idx):
        """Load a single item for cache warming"""
        try:
            # Just access the item to load it into cache
            latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
            if os.path.exists(latent_path):
                # Load with mmap but don't keep in memory
                _ = torch.load(latent_path, map_location='cpu', mmap=True)
            return True
        except Exception as e:
            logger.warning(f"Failed to warm cache for index {idx}: {str(e)}")
            return False

    def _load_manifest(self):
        """Load manifest from precomputed features"""
        # All features guaranteed aligned by preprocessor
        return [Path(f).stem for f in self.latent_files]

    def __getitem__(self, idx):
        base = self.manifest[idx]
        
        # Direct loading without existence checks
        return {
            'latent': torch.load(f"{self.latent_dir}/{base}.pt"),
            'clip_embedding': torch.load(f"{self.clip_dir}/{base}.pt"),
            'expert': torch.load(f"{self.cluster_dir}/final_clusters.pt")[idx],
            'dims': torch.load(f"{self.dim_dir}/{base}.pt")
        }

    def _load_latent(self, idx):
        """Direct memory access with pointer arithmetic"""
        ptr = self.mmap_handles[idx * 4]
        mmap = np.memmap(ptr[0], mode='r', offset=ptr[1], shape=(ptr[2],))
        return torch.from_numpy(np.frombuffer(mmap, dtype=np.float32))

    def _load_clip(self, idx):
        """CLIP embedding with async prefetch"""
        ptr = self.mmap_handles[idx * 4 + 1]
        mmap = np.memmap(ptr[0], mode='r', offset=ptr[1], shape=(ptr[2],))
        return torch.from_numpy(np.frombuffer(mmap, dtype=np.float32))

    def _prefetch_next_batch(self):
        """Background prefetch of next anticipated batch"""
        if not hasattr(self, '_prefetch_executor'):
            self._prefetch_executor = ThreadPoolExecutor(max_workers=4)
            
        # Predict next access pattern
        next_indices = range(self.last_idx, self.last_idx + self.config.batch_size)
        self._prefetch_executor.submit(self._prefetch_indices, next_indices)

    def _prefetch_indices(self, indices):
        """Prefetch specific indices using POSIX_FADV_WILLNEED"""
        for idx in indices:
            if idx >= len(self):
                continue
            for ptr in self.mmap_handles[idx*4:(idx+1)*4]:
                os.posix_fadvise(
                    ptr[0].fileno(), 
                    ptr[1], 
                    ptr[2], 
                    os.POSIX_FADV_WILLNEED
                )

    def validate_dataset(self):
        for i in range(len(self)):
            try:
                _ = self[i]
            except Exception as e:
                logger.error(f"Bad sample at index {i}: {str(e)}")
                raise

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
        print(f"[Rank {torch.distributed.get_rank()}] Initializing BucketBatchSampler with:")
        print(f"Total buckets: {len(self.bucket_tensors)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Shuffle: {self.shuffle}")
        
        # GPU-accelerated shuffling and batching
        all_batches = []
        for bucket_idx, indices in self.bucket_tensors.items():
            print(f"Processing bucket {bucket_idx} with {len(indices)} samples")
            
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
                bucket_indices=bucket_indices,
                batch_size=config.expert_batch_size,
                device=device,
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
