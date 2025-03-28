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
        
        # New validation step - check file existence
        self.valid_indices = self._validate_file_existence()
        self.num_samples = len(self.valid_indices)
        
        logger.info(f"[Rank {self.rank}] Final dataset size after validation: {self.num_samples}")
        
        # Initialize bucket assignments as a property
        self._bucket_assignments = None  # Initialize cache

    def _verify_cache_dirs(self):
        """Verify cache directories and initialize sample IDs"""
        # Make sure these directories exist
        self.clip_dir = os.path.join(self.config.feature_cache_path, 'clip')
        self.latent_dir = os.path.join(self.config.feature_cache_path, 'latents')
        
        # Check directory existence
        if not os.path.exists(self.clip_dir) or not os.path.exists(self.latent_dir):
            raise ValueError(f"Cache directories not found: {self.clip_dir}, {self.latent_dir}")
        
        # Get sample IDs from actual files
        sample_ids = []
        # Iterate through clip files to extract actual sample IDs
        for file_path in Path(self.clip_dir).glob("anime-*_rank*.pt"):
            file_name = file_path.stem  # Get filename without extension
            # Parse the ID from the filename pattern "anime-{ID}_rank{N}"
            try:
                sample_id = int(file_name.split("-")[1].split("_rank")[0])
                sample_ids.append(sample_id)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse sample ID from {file_name}: {e}")
        
        # Remove duplicates and sort
        self.samples = sorted(set(sample_ids))
        self.num_samples = len(self.samples)
        print(f"Found {self.num_samples} valid samples with both latent and CLIP embeddings")

    def _validate_file_existence(self):
        """Check that all required files exist for each sample"""
        valid_indices = []
        
        for idx in tqdm(range(len(self.latent_files)), desc=f"[Rank {self.rank}] Validating files"):
            base_name = self.base_names[idx]
            
            # CORRECTED: Use consistent naming across features
            required_files = [
                self.latent_files[idx],  # Already contains rank suffix
                os.path.join(self.clip_dir, f"{base_name}_rank{self.rank}.pt"),
                os.path.join(self.cluster_dir, f"{base_name}_rank{self.rank}.pt"),
                os.path.join(self.bucket_dir, f"{base_name}_rank{self.rank}.pt")
            ]
            
            if all(os.path.exists(f) for f in required_files):
                valid_indices.append(idx)
            else:
                missing = [f for f in required_files if not os.path.exists(f)]
                logger.warning(f"Skipping sample {base_name} - missing files: {[Path(f).name for f in missing]}")
                
        return valid_indices

    def __getitem__(self, idx):
        """Load individual sample with error handling"""
        try:
            actual_idx = self.valid_indices[idx]
            base_name = self.base_names[actual_idx]
            
            # Load latent
            latent = torch.load(self.latent_files[actual_idx], map_location='cpu')
            
            # Load CLIP embeddings
            clip_path = os.path.join(self.clip_dir, f"{base_name}.pt")
            clip_embed = torch.load(clip_path, map_location='cpu')
            
            # Load cluster assignment
            cluster_path = os.path.join(self.cluster_dir, f"{base_name}.pt")
            cluster_id = torch.load(cluster_path, map_location='cpu')
            
            # Load bucket dimensions
            dim_path = os.path.join(self.dim_dir, f"{base_name}.pt")
            bucket_dims = torch.load(dim_path, map_location='cpu')
            
            return {
                'latent': latent,
                'clip_embedding': clip_embed,
                'expert': cluster_id,
                'dims': bucket_dims
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} on rank {self.rank}: {str(e)}")
            return None  # Return None to be filtered later

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
        """Modified collate that handles None results"""
        batch = [b for b in batch if b is not None]
        
        if not batch:
            return None
        
        return {
            'latent': torch.stack([item['latent'] for item in batch]),
            'clip_embedding': torch.stack([item['clip_embedding'] for item in batch]),
            'bucket': torch.stack([item['bucket'] for item in batch]),
            'expert': torch.stack([item['expert'] for item in batch])
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
