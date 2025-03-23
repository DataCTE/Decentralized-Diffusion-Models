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
        
        # Initialize memory maps
        self._init_memory_maps()

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
        """Distributed cluster loading with proper tensor dimensions"""
        shard_size = len(self.latent_files) // self.world_size
        start = self.rank * shard_size
        end = start + shard_size if self.rank != self.world_size - 1 else len(self.latent_files)

        # Unified progress bar only on main process
        pbar = tqdm(total=end-start,
                   desc="Loading clusters",
                   leave=False,
                   bar_format="{l_bar}{bar:20}{r_bar}",
                   disable=not is_main_process())

        assignments = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._load_cluster_or_default, 
                os.path.join(self.cluster_dir, fname.replace('.latent.pt', '.cluster.pt'))): fname 
                for fname in self.latent_files[start:end]}
            
            for future in as_completed(futures):
                tensor = future.result()
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                assignments.append(tensor)
                pbar.update(1)

        pbar.close()

        if not assignments:
            assignments = [torch.zeros(0, dtype=torch.long)]
        
        assignments = torch.cat(assignments).cuda()
        
        # Distributed sync
        gathered = [torch.empty_like(assignments) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, assignments)
        
        return torch.cat(gathered).cpu()

    def _load_cluster_or_default(self, path):
        """Load cluster with dimension enforcement"""
        try:
            cluster = torch.load(path, map_location='cpu', mmap=True)
            return cluster.view(-1)  # Ensure 1D tensor
        except (FileNotFoundError, IOError, RuntimeError):
            return torch.tensor([0], dtype=torch.long)  # 1D tensor

    def _distributed_bucket_indices(self):
        """Filename-parsed dimensions with proper device placement"""
        # Ensure tensors are on GPU for NCCL backend
        bucket_indices = torch.zeros(len(self.latent_files), 
                                   dtype=torch.long,
                                   device=f'cuda:{self.rank}')
        
        # Extract dimensions from virtual filenames
        for i, fname in enumerate(self.latent_files):
            try:
                # Parse dimensions from virtual filename format: {W}x{H}_{index}.latent.pt
                dim_part = fname.split('_', 1)[0]
                w, h = map(int, dim_part.split('x', 1))
                bucket_idx = next(i for i, (bw, bh) in enumerate(self.config.buckets) 
                                if bw == w and bh == h)
                bucket_indices[i] = bucket_idx
            except:
                bucket_indices[i] = 0  # Fallback to first bucket

        # Create GPU tensors for gathering
        gathered = [torch.empty_like(bucket_indices, device=bucket_indices.device) 
                  for _ in range(self.world_size)]
        
        # Distributed sync with NCCL
        torch.distributed.all_gather(gathered, bucket_indices)
        
        # Move to CPU for dataset operations
        return torch.cat(gathered).cpu()

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
        """Create memory map index with pointer arithmetic"""
        if not os.path.exists(index_path):
            # Build pointer index
            ptrs = []
            with open(index_path, 'wb') as f:
                for ftype in ['latent', 'clip', 'cluster', 'dim']:
                    files = sorted(glob.glob(os.path.join(
                        self.config.feature_cache_path,
                        f"{ftype}s/*.pt"
                    )))
                    for file in files:
                        size = os.path.getsize(file)
                        f.write(struct.pack('Q', size))
                        ptrs.append((file, 0, size))
            
            # Memory map the index
            return np.memmap(index_path, mode='r', dtype=np.uint64)
        
        return np.memmap(index_path, mode='r', dtype=np.uint64)

    def _warmup_cache(self):
        """Preload initial data segments using parallel prefetch"""
        # Unified progress bar only on main process
        warmup_pbar = tqdm(total=len(self)//10,
                         desc="Warming cache",
                         leave=False,
                         disable=not is_main_process())

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

    def __getitem__(self, idx):
        # Get device for current process
        device_id = torch.cuda.current_device()
        
        # Lazy load latent with direct GPU transfer
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = torch.load(latent_path, 
                          map_location=lambda storage, loc: storage.cuda(device_id, non_blocking=True),
                          mmap=True,
                          pin_memory=True)
        
        # Async CLIP loading with stream-aware prefetch
        clip_path = os.path.join(
            self.config.feature_cache_path,
            "clip_embeddings",
            self.latent_files[idx].replace('.latent.pt', '.clip_emb.pt')
        )
        with torch.cuda.stream(torch.cuda.Stream(device_id)):
            clip_emb = torch.load(clip_path, 
                                map_location='cuda', 
                                mmap=True,
                                pin_memory=True) if os.path.exists(clip_path) \
                    else torch.zeros(self.config.clip_embedding_dim, 
                                   device='cuda',
                                   pin_memory=True)

        return {
            'latent': latent,
            'clip_embedding': clip_emb,
            'bucket': self.bucket_assignments[idx].cuda(non_blocking=True),
            'expert': torch.tensor(self.expert_assignments[idx], 
                                 device='cuda', 
                                 non_blocking=True)
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
    device = torch.device('cpu')
    logger = setup_distributed_logger(name="ExpertLoaders", rank=rank)

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
        expert_indices[expert_idx].append(idx.item())

    # Log expert distribution stats
    total_indices = sum(len(indices) for indices in expert_indices.values())
    logger.info(f"Rank {rank}: Collected {total_indices} indices across {len(expert_indices)} active experts")

    expert_loaders = {}

    for expert_idx, indices in expert_indices.items():
        # GPU-accelerated bucket index creation
        with torch.cuda.stream(torch.cuda.Stream(device=rank % torch.cuda.device_count())):
            bucket_indices = defaultdict(list)
            for idx in indices:
                bucket_idx = dataset.bucket_assignments[idx].item()
                bucket_indices[bucket_idx].append(idx)

        # Distributed logging
        logger.info(f"Rank {rank}: Expert {expert_idx} processing on GPU {rank % torch.cuda.device_count()}")

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

        loader_pbar.update(1)
        loader_pbar.set_postfix_str(f"Experts: {len(expert_loaders)}")

    loader_pbar.close()
    total_loader_time = time.time() - loader_start
    logger.info(f"Rank {rank}: DataLoader creation complete in {total_loader_time:.2f}s - {len(expert_loaders)} expert loaders created")

    return expert_loaders 
