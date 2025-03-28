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
from multiprocessing import Pool

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
    """Distributed-optimized dataset with on-demand loading"""
    
    def __init__(self, config_dict, split='train'):
        # Convert config dict to SimpleNamespace
        self.config = SimpleNamespace(**config_dict)
        self.feature_dir = self.config.feature_cache_path
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = torch.device('cpu')
        
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
        
        # Remove all preloading logic
        logger.info(f"[Rank {self.rank}] Initialized dataset with {self.num_samples} samples (lazy loading enabled)")
        
        # Initialize bucket assignments as a property
        self._bucket_assignments = None  # Initialize cache

    def _verify_cache_dirs(self):
        """Research-grade validation using parallel file checking"""
        # Get all file paths using parallel glob
        with Pool() as pool:
            clip_files = set(pool.map(str, Path(self.clip_dir).glob("anime-*_rank*.pt")))
            latent_files = set(pool.map(str, Path(self.latent_dir).glob("anime-*_rank*.pt")))

        # Extract base names without rank/extension
        clip_bases = {f.split("/")[-1].split("_rank")[0] for f in clip_files}
        latent_bases = {f.split("/")[-1].split("_rank")[0] for f in latent_files}

        # Find valid pairs using set intersection
        valid_bases = clip_bases & latent_bases
        
        # Sort to ensure deterministic ordering
        self.base_names = sorted(valid_bases)
        self.num_samples = len(self.base_names)
        
        # Create full file paths using first available rank
        self.latent_files = [
            f for base in self.base_names
            for f in glob.glob(os.path.join(self.latent_dir, f"{base}_rank*.pt"))
        ]
        
        logger.info(f"Verified {self.num_samples} valid sample pairs")

    def __getitem__(self, idx):
        """Minimal check variant for research efficiency"""
        base_name = self.base_names[idx]
        rank_suffix = f"_rank{self.rank}.pt"
        
        try:
            return {
                'latent': torch.load(f"{self.latent_dir}/{base_name}{rank_suffix}", map_location='cpu'),
                'clip_embedding': torch.load(f"{self.clip_dir}/{base_name}{rank_suffix}", map_location='cpu'),
                'expert': torch.load(f"{self.cluster_dir}/{base_name}{rank_suffix}", map_location='cpu'),
                'dims': torch.load(f"{self.dim_dir}/{base_name}{rank_suffix}", map_location='cpu')
            }
        except FileNotFoundError as e:
            logger.debug(f"Missing file for {base_name}: {str(e)}")
            return None

    def __len__(self):
        return self.num_samples

    # Remove all preloading-related methods
    def _lazy_load_clusters(self):
        """Load cluster assignments on first access"""
        if not hasattr(self, '_cluster_assignments'):
            cluster_file = os.path.join(self.cluster_dir, "final_clusters.pt")
            self._cluster_assignments = torch.load(cluster_file, map_location='cpu')
        return self._cluster_assignments

    @property
    def bucket_assignments(self):
        """Lazy-loaded bucket assignments tensor"""
        if self._bucket_assignments is None:
            self._bucket_assignments = self._load_bucket_assignments()
        return self._bucket_assignments

    def _load_bucket_assignments(self):
        """Load and cache bucket assignments as tensor"""
        assignments = []
        for base_name in self.base_names:
            path = os.path.join(self.bucket_dir, f"{base_name}_rank{self.rank}.pt")
            assignments.append(torch.load(path, map_location='cpu'))  # Load to CPU first
        
        # Stack and move to dataset's device
        return torch.stack(assignments).to(self.device)

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
        # Filter out None items resulting from loading errors in __getitem__
        original_batch_size = len(batch)
        batch = [b for b in batch if b is not None]
        filtered_batch_size = len(batch)

        if filtered_batch_size == 0:
            # Return None or an empty dictionary if the entire batch failed
            if original_batch_size > 0:
                 logger.warning(f"Entire batch of size {original_batch_size} failed to load. Skipping batch.")
            return None # Or perhaps {} depending on how the training loop handles it

        if filtered_batch_size < original_batch_size:
             logger.warning(f"Filtered out {original_batch_size - filtered_batch_size} failed samples from batch.")

        # Initialize result dictionary with keys that are always present
        result = {
            'latent': torch.stack([item['latent'] for item in batch]),
            'bucket': torch.stack([item['bucket'] for item in batch]),
            'expert': torch.stack([item['expert'] for item in batch])
        }

        # Handle CLIP embeddings if present - Use the correct key 'clip_embedding'
        if 'clip_embedding' in batch[0]:
            clip_embeddings = [item['clip_embedding'] for item in batch]
            # Check if padding is necessary (all sequences might have the same length)
            if all(e.size(1) == clip_embeddings[0].size(1) for e in clip_embeddings):
                 result['clip_embedding'] = torch.stack(clip_embeddings)
            else:
                 # Pad only if lengths differ
                 max_len = max(e.size(1) for e in clip_embeddings)
                 padded_embeddings = []
                 for emb in clip_embeddings:
                     pad_size = max_len - emb.size(1)
                     # Pad sequence length dimension (dim=1), value=0
                     padded = torch.nn.functional.pad(emb, (0, 0, 0, pad_size), value=0)
                     padded_embeddings.append(padded)
                 result['clip_embedding'] = torch.stack(padded_embeddings)
        # else: # Optional: Log if clip_embedding is expected but missing
        #     print("Warning: 'clip_embedding' key not found in the first item of the batch.")


        # Handle T5 embeddings if present - Use the correct key 't5_embedding'
        if 't5_embedding' in batch[0]:
            t5_embeddings = [item['t5_embedding'] for item in batch]
            # Check if padding is necessary
            if all(e.size(1) == t5_embeddings[0].size(1) for e in t5_embeddings):
                 result['t5_embedding'] = torch.stack(t5_embeddings)
            else:
                 # Pad only if lengths differ
                 max_len = max(e.size(1) for e in t5_embeddings)
                 padded_embeddings = []
                 for emb in t5_embeddings:
                     pad_size = max_len - emb.size(1)
                     padded = torch.nn.functional.pad(emb, (0, 0, 0, pad_size), value=0)
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
    
    def __init__(self, dataset, batch_size, device='cpu', shuffle=True, drop_last=True, bucket_indices=None):
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
        
        # Convert lists to tensors - FIXED VERSION
        self.bucket_tensors = {
            bucket: torch.tensor(indices, device=self.device)  # Create tensor first
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

    # Get expert assignments directly from GPU tensor
    expert_assignments = dataset.expert_assignments.cpu().numpy()
    expert_indices = defaultdict(list)

    # Debug prints for initialization
    print(f"\n[Rank {rank}] ===== EXPERT LOADER INITIALIZATION =====")
    print(f"[Rank {rank}] Total dataset samples: {len(dataset)}")
    print(f"[Rank {rank}] Dataset latent files: {len(dataset.latent_files)}")
    print(f"[Rank {rank}] First 5 expert assignments: {dataset.expert_assignments[:5]}")
    
    # Use vectorized operations for expert index collection
    for idx in range(len(dataset)):
        if idx >= len(expert_assignments):
            print(f"[Rank {rank}] WARNING: Index {idx} out of range for expert assignments")
            continue
        expert_idx = expert_assignments[idx]
        expert_indices[expert_idx].append(idx)

    # Distributed progress tracking
    loader_pbar = tqdm(
        total=len(expert_indices),
        desc=f"[Rank {rank}] Creating Expert Loaders",
        position=rank,
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}",
        disable=not is_main_process()
    )

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

    loader_start = time.time()
    logger.info(f"Rank {rank}: Starting DataLoader creation for {len(expert_indices)} experts")

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
                bucket_indices=bucket_indices,  # Pass pre-computed bucket indices
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
                pin_memory=True,
                collate_fn=dataset.collate_fn
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
