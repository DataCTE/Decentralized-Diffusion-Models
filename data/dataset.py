"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from collections import defaultdict
import logging
import time  
import glob
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from torch.serialization import safe_globals
from numpy._core.multiarray import _reconstruct
from types import SimpleNamespace

# Import centralized utilities
from utils.distributed import is_main_process, get_rank, get_world_size
from utils.logging import setup_distributed_logger

# Setup logging
logger = logging.getLogger(__name__)


def chunks(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def simple_collate(batch):
    """Standalone collate function that doesn't capture class instances"""
    try:
        return {
            'latent': torch.stack([item['latent'] for item in batch]),
            'clip_embedding': torch.stack([item['clip_embedding'] for item in batch]),
            'bucket': torch.stack([item['bucket'] for item in batch]),
            'expert': torch.stack([item['expert'] for item in batch])
        }
    except Exception as e:
        print(f"Collation error: {str(e)}")
        return None

class DDMDataset(Dataset):
    """Distributed-optimized dataset pipeline with lazy loading and prefetching"""
    
    def __init__(self, config_dict, split='train'):
        # Convert config dict to SimpleNamespace
        self.config = SimpleNamespace(**config_dict)
        self.feature_dir = self.config.feature_cache_path
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # Set device based on availability and rank
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        
        # Cache directories
        self.latent_dir = os.path.join(self.config.feature_cache_path, "latents")
        self.clip_dir = os.path.join(self.config.feature_cache_path, "clip")
        self.cluster_dir = os.path.join(self.config.feature_cache_path, "clusters")
        self.dim_dir = os.path.join(self.config.feature_cache_path, "dims")
        self.bucket_dir = os.path.join(self.config.feature_cache_path, "buckets")
        
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
        
        # Load cluster assignments and compute statistics
        self.cluster_assignments = self._lazy_load_clusters()
        self._compute_cluster_statistics()
        
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
        """Load precomputed cluster assignments with proper device placement"""
        cluster_path = os.path.join(self.cluster_dir, "final_clusters.pt")
        try:
            # Add safe_globals context for numpy compatibility
            with torch.serialization.safe_globals([_reconstruct]):
                cluster_assignments = torch.load(
                    cluster_path,
                    map_location=self.device,
                    weights_only=False  # Required for numpy compatibility
                )
            
            if isinstance(cluster_assignments, np.ndarray):
                cluster_assignments = torch.from_numpy(cluster_assignments).to(self.device)
            else:
                cluster_assignments = cluster_assignments.to(self.device)
            
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
            
            return cluster_assignments
            
        except Exception as e:
            logger.error(f"Failed to load cluster assignments: {str(e)}")
            raise

    def _load_sample(self, idx):
        """Load a sample by index, supporting both CLIP and T5 embeddings"""
        try:
            # Get filename from index mapping
            filename = self.base_names[idx]
            
            # Load latent feature
            latent_path = os.path.join(self.latent_dir, f"{filename}_rank{self.rank}.pt")
            latent = torch.load(latent_path, map_location='cpu')
            
            # Load text embedding (either CLIP, T5, or both depending on config)
            text_embeddings = {}
            
            # CLIP embedding (always try to load if path exists)
            if hasattr(self.config, 'use_clip') and self.config.use_clip:
                clip_path = os.path.join(self.clip_dir, f"{filename}_rank{self.rank}.pt")
                if os.path.exists(clip_path):
                    text_embeddings['clip'] = torch.load(clip_path, map_location='cpu')
            
            # T5 embedding (load if configured)
            if hasattr(self.config, 'use_t5') and self.config.use_t5:
                t5_path = os.path.join(self.config.feature_cache_path, "t5", f"{filename}_rank{self.rank}.pt")
                if os.path.exists(t5_path):
                    text_embeddings['t5'] = torch.load(t5_path, map_location='cpu')
            
            # Get bucket and expert assignments
            bucket = self.bucket_assignments[idx] if self.bucket_assignments is not None else torch.tensor(0)
            expert = self.expert_assignments[idx] if self.expert_assignments is not None else torch.tensor(0)
            
            # Construct sample dictionary
            sample = {
                'latent': latent,
                'bucket': bucket,
                'expert': expert,
                **text_embeddings  # Add all available text embeddings
            }
            
            return idx, sample
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading sample {idx}: {str(e)}")
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

    def _compute_cluster_statistics(self):
        """Compute and cache cluster statistics"""
        if self.cluster_assignments is None:
            logger.warning("No cluster assignments found - using uniform distribution")
            self.cluster_counts = torch.ones(self.config.num_experts, dtype=torch.long, device=self.device)
            return

        # Count samples per cluster - ensure long dtype
        unique_clusters, counts = torch.unique(
            self.cluster_assignments, 
            return_counts=True
        )
        
        # Initialize counts tensor with matching dtype
        self.cluster_counts = torch.zeros(
            self.config.num_experts, 
            dtype=counts.dtype,  # Match the dtype of counts
            device=self.device
        )
        
        # Fill in actual counts
        self.cluster_counts[unique_clusters] = counts
        
        # Log distribution
        total = self.cluster_counts.sum().item()
        for cluster, count in enumerate(self.cluster_counts.tolist()):
            logger.info(f"Cluster {cluster}: {count} samples ({100 * count/total:.2f}%)")

    def get_cluster_sizes(self):
        """Return number of samples per cluster"""
        if not hasattr(self, 'cluster_counts'):
            self._compute_cluster_statistics()
        return self.cluster_counts

    def get_cluster_distribution(self):
        """Return normalized cluster distribution"""
        counts = self.get_cluster_sizes()
        return counts / counts.sum()

    @staticmethod
    def collate_fn(batch):
        """Modified collate that handles variable-length sequences and multiple embedding types"""
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        result = {
            'latent': torch.stack([item['latent'] for item in batch]),
            'bucket': torch.stack([item['bucket'] for item in batch]),
            'expert': torch.stack([item['expert'] for item in batch])
        }
        
        # Handle CLIP embeddings if present
        if 'clip' in batch[0]:
            clip_embeddings = [item['clip'] for item in batch]
            max_len = max(e.size(1) for e in clip_embeddings)
            
            padded_embeddings = []
            for emb in clip_embeddings:
                pad_size = max_len - emb.size(1)
                padded = torch.nn.functional.pad(emb, (0,0,0,pad_size), value=0)
                padded_embeddings.append(padded)
            
            result['clip_embedding'] = torch.stack(padded_embeddings)
        
        # Handle T5 embeddings if present
        if 't5' in batch[0]:
            t5_embeddings = [item['t5'] for item in batch]
            max_len = max(e.size(1) for e in t5_embeddings)
            
            padded_embeddings = []
            for emb in t5_embeddings:
                pad_size = max_len - emb.size(1)
                padded = torch.nn.functional.pad(emb, (0,0,0,pad_size), value=0)
                padded_embeddings.append(padded)
            
            result['t5_embedding'] = torch.stack(padded_embeddings)
        
        return result

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

def _init_data_loaders(self):
    # Replace with distributed sampler
    sampler = DistributedSampler(
        self.train_dataset,
        num_replicas=self.world_size,
        rank=self.rank,
        shuffle=True
    )
    
    self.train_loader = DataLoader(
        self.train_dataset,
        batch_size=self.config.batch_size,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True
    ) 
