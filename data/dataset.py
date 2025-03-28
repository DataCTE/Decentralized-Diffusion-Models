"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from collections import defaultdict
import logging
import time  
from tqdm.auto import tqdm
import torch.distributed as dist # Assuming dist is initialized elsewhere or handle initialization
from types import SimpleNamespace
# Import centralized utilities
from utils.distributed import is_main_process, get_rank, get_world_size
from utils.logging import setup_distributed_logger
import re  # Add regex module


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
    
    def __init__(self, config_dict, split='train', logger=None):
        # Convert config dict to SimpleNamespace
        self.config = SimpleNamespace(**config_dict)
        self.feature_dir = self.config.feature_cache_path
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = torch.device('cpu')
        # Store the passed logger
        self.logger = logger if logger else logging.getLogger("DDMDataset_fallback")
        
        # Cache directories
        self.latent_dir = os.path.join(self.config.feature_cache_path, "latents")
        self.clip_dir = os.path.join(self.config.feature_cache_path, "clip")
        self.cluster_dir = os.path.join(self.config.feature_cache_path, "clusters")
        self.dim_dir = os.path.join(self.config.feature_cache_path, "dims")
        self.bucket_dir = os.path.join(self.config.feature_cache_path, "buckets")
        
        # Verify directories exist and find valid base names
        self._verify_cache_dirs()
        
        # Use self.logger for logging
        self.logger.info(f"[Rank {self.rank}] Initialized dataset with {self.num_samples} samples (lazy loading enabled)")
        
        # Initialize bucket assignments as a property
        self._bucket_assignments = None  # Initialize cache

    def _verify_cache_dirs(self):
        """Verify cache directories with strict filename validation."""
        required_dirs = {
            "latents": self.latent_dir,
            "clip": self.clip_dir,
            "clusters": self.cluster_dir,
            "dims": self.dim_dir,
            "buckets": self.bucket_dir
        }

        # Pattern to match 'basename_rankN.pt' where N is a digit
        filename_pattern = re.compile(r"^(.*)_rank(\d+)\.pt$")
        
        all_base_sets = []
        try:
            # Add progress bar for directory verification
            dir_progress = tqdm(
                required_dirs.items(),
                desc=f"Rank {self.rank} Verifying cache",
                disable=not is_main_process(self.rank),
                leave=False
            )
            
            for feature_type, dir_path in dir_progress:
                dir_progress.set_description(f"Checking {feature_type} directory")
                if not os.path.isdir(dir_path):
                    raise FileNotFoundError(f"Missing directory: {dir_path}")
                    
                valid_files = []
                base_names = set()
                
                # Update progress bar description
                dir_progress.set_postfix({"current": feature_type, "status": "scanning"})
                
                # Use regex to validate filenames
                for f in os.listdir(dir_path):
                    match = filename_pattern.match(f)
                    if match:
                        base_name, rank = match.groups()
                        if rank == str(self.rank):  # Only consider files for current rank
                            base_names.add(base_name)
                            valid_files.append(f)
                
                if not base_names:
                    self.logger.error(f"No valid files found in {dir_path} for rank {self.rank}")
                    
                all_base_sets.append(base_names)
                dir_progress.set_postfix({
                    "current": feature_type,
                    "files": len(valid_files),
                    "bases": len(base_names)
                })
                
                if self.rank == 0:
                    self.logger.info(f"{feature_type.upper()} | Valid files: {len(valid_files)} | Unique bases: {len(base_names)}")

            # Find intersection of base names
            valid_bases = set.intersection(*[set(s) for s in all_base_sets])
            dir_progress.set_postfix({"status": f"found {len(valid_bases)} common bases"})
            
            if not valid_bases:
                # Detailed error reporting
                missing_in = []
                for i, (feature, path) in enumerate(required_dirs.items()):
                    if not all_base_sets[i]:
                        missing_in.append(f"{feature} ({path})")
                        
                error_msg = "No common base names found. Missing/invalid files in:\n" + "\n".join(missing_in)
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Sort to ensure deterministic ordering
            self.base_names = sorted(valid_bases)
            self.num_samples = len(self.base_names)

            if self.rank == 0:
                self.logger.info(f"Verified {self.num_samples} samples with complete features")

        except Exception as e:
            self.logger.error("Cache verification failed:\n" + "\n".join([
                f"- {k}: {v} ({len(all_base_sets[i])} bases)" 
                for i, (k, v) in enumerate(required_dirs.items())
            ]))
            raise

    def __getitem__(self, idx):
        """Minimal check variant for research efficiency"""
        base_name = self.base_names[idx]
        rank_suffix = f"_rank{self.rank}.pt"

        # Construct all paths first
        paths = {
            'latent': os.path.join(self.latent_dir, f"{base_name}{rank_suffix}"),
            'clip_embedding': os.path.join(self.clip_dir, f"{base_name}{rank_suffix}"),
            'expert': os.path.join(self.cluster_dir, f"{base_name}{rank_suffix}"),
            'dims': os.path.join(self.dim_dir, f"{base_name}{rank_suffix}"),
            # Include bucket path for completeness, though not directly returned here
            # 'bucket': os.path.join(self.bucket_dir, f"{base_name}{rank_suffix}")
        }

        try:
            # Optional: Add an explicit check here if paranoia is high,
            # but _verify_cache_dirs should handle it.
            # for feature, path in paths.items():
            #     if not os.path.exists(path):
            #         raise FileNotFoundError(f"Verified base name '{base_name}' but file missing at {path}")

            return {
                'latent': torch.load(paths['latent'], map_location='cpu'),
                'clip_embedding': torch.load(paths['clip_embedding'], map_location='cpu'),
                'expert': torch.load(paths['expert'], map_location='cpu'),
                'dims': torch.load(paths['dims'], map_location='cpu'),
                # Bucket data is loaded separately via the bucket_assignments property
            }
        except FileNotFoundError as e:
            # This should ideally not happen now, but log if it does
            self.logger.error(f"[Rank {self.rank}] CRITICAL: File missing for verified base_name '{base_name}' at index {idx}: {str(e)}. Check cache integrity.", exc_info=True)
            # Decide how to handle this - raising might be better than returning None now
            raise e # Re-raise the critical error
        except Exception as e:
             self.logger.error(f"[Rank {self.rank}] Error loading data for base_name '{base_name}' at index {idx}: {str(e)}", exc_info=True)
             # Decide on error handling: skip sample (return None), or raise error?
             return None # Returning None might still be necessary for corrupt files, handle in collate_fn

    def __len__(self):
        return self.num_samples

    @property
    def bucket_assignments(self) -> torch.Tensor:
        """
        Lazy-loaded tensor containing the bucket index for each sample.
        Loads from precomputed files on first access.
        Shape: [num_samples]
        """
        # Check if already loaded or if loading is in progress to prevent race conditions
        # (though less likely with standard DataLoader workers)
        if hasattr(self, '_bucket_assignments_loading') and self._bucket_assignments_loading:
             # Simple spin-wait or use a proper lock if threading issues are observed
             while self._bucket_assignments is None:
                 time.sleep(0.1)
             return self._bucket_assignments

        if self._bucket_assignments is None:
            self._bucket_assignments_loading = True # Mark as loading
            try:
                if self.rank == 0:
                    self.logger.info("Loading bucket assignments for the first time...")
                start_time = time.time()
                # Ensure result is stored on the correct device ('cpu')
                self._bucket_assignments = self._load_bucket_assignments().to(self.device)
                if self.rank == 0:
                    load_time = time.time() - start_time
                    self.logger.info(f"Finished loading {len(self._bucket_assignments)} bucket assignments in {load_time:.2f}s.")
                if len(self._bucket_assignments) != self.num_samples:
                     self.logger.error(f"Mismatch in number of loaded bucket assignments ({len(self._bucket_assignments)}) and dataset samples ({self.num_samples}).")
                     # Decide on error handling - raise or try to recover? Raising is safer.
                     raise RuntimeError("Bucket assignment count mismatch.")
            except Exception as e:
                 self.logger.error(f"Failed to load bucket assignments: {e}", exc_info=True)
                 self._bucket_assignments = None # Ensure it remains None on failure
                 raise # Re-raise the exception
            finally:
                 self._bucket_assignments_loading = False # Mark loading as complete/failed


        return self._bucket_assignments

    def _load_bucket_assignments(self) -> torch.Tensor:
        """Load bucket assignments for all verified samples from cache files."""
        assignments = []
        missing_files = []
        # Use tqdm for loading assignments, only display on rank 0
        pbar_desc = f"Rank {self.rank} loading bucket files"
        # Iterate through the verified base names
        for base_name in tqdm(self.base_names, desc=pbar_desc, disable=(self.rank != 0), leave=False):
            # Construct the expected path for the current rank
            path = os.path.join(self.bucket_dir, f"{base_name}_rank{self.rank}.pt")
            try:
                # Load directly to CPU, assumes assignment is a single tensor/value
                assignment_tensor = torch.load(path, map_location='cpu')
                # Ensure it's a scalar or has the expected format (e.g., single element tensor)
                if not isinstance(assignment_tensor, torch.Tensor):
                     assignment_tensor = torch.tensor(assignment_tensor) # Convert if not tensor
                # Ensure it's treated as a bucket index (long)
                assignments.append(assignment_tensor.long().squeeze()) # Use squeeze to remove extra dims if any
            except FileNotFoundError:
                 # This error should NOT happen if _verify_cache_dirs worked correctly
                 self.logger.error(f"CRITICAL: Bucket file missing for verified base name '{base_name}' at {path}. Cache verification failed or file deleted during run.")
                 missing_files.append(path)
                 # Continue collecting all missing files before raising
            except Exception as e:
                 self.logger.error(f"Error loading bucket file {path}: {e}", exc_info=True)
                 raise RuntimeError(f"Failed to load or process bucket file {path}") from e

        if missing_files:
             raise FileNotFoundError(f"Found {len(missing_files)} missing bucket files for verified samples. Example: {missing_files[0]}")

        if not assignments:
             # This case should also be prevented by _verify_cache_dirs
             self.logger.error("No bucket assignments loaded, despite having verified base names. Check bucket cache contents and _verify_cache_dirs logic.")
             raise RuntimeError("Failed to load any bucket assignments.")

        # Stack the loaded tensors (should be scalars or 1-element tensors)
        try:
             stacked_assignments = torch.stack(assignments)
        except Exception as e:
             self.logger.error(f"Failed to stack loaded bucket assignments. Check individual file contents. Error: {e}", exc_info=True)
             # Log shapes for debugging
             if assignments:
                 self.logger.error(f"Assignment shapes example: {[a.shape for a in assignments[:5]]}")
             raise RuntimeError("Failed to stack bucket assignments.") from e

        return stacked_assignments # Return on CPU, property moves to self.device

    def _compute_cluster_statistics(self):
        """Compute and cache cluster statistics"""
        if self.cluster_assignments is None:
            self.logger.warning("No cluster assignments found - using uniform distribution")
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
            if self.rank == 0:
                self.logger.info(f"Cluster {cluster}: {count} samples ({100 * count/total:.2f}%)")

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
        """
        Collate function that handles potential None values in the batch
        (e.g., due to loading errors other than FileNotFoundError).
        """
        # Filter out None items first
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the whole batch was invalid

        try:
            # Proceed with stacking the valid items
            return {
                'latent': torch.stack([item['latent'] for item in batch]),
                'clip_embedding': torch.stack([item['clip_embedding'] for item in batch]),
                # Bucket info is implicitly handled by BucketBatchSampler forming the batch
                'expert': torch.stack([item['expert'] for item in batch]),
                'dims': torch.stack([item['dims'] for item in batch])
            }
        except Exception as e:
            logging.error(f"Collation error after filtering Nones: {str(e)}", exc_info=True)
            # You might want to log the shapes here for debugging
            # for i, item in enumerate(batch):
            #    logging.error(f"Item {i} shapes: latent={item['latent'].shape}, clip={item['clip_embedding'].shape}, ...")
            return None # Return None if collation fails even after filtering

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

class BucketBatchSampler(Sampler):
    """
    Groups samples by bucket ID and yields batches of indices from the same bucket.
    Ensures samples within a batch have compatible dimensions (handled by bucketing).
    """
    def __init__(self, dataset: DDMDataset, batch_size: int, device=None, shuffle=True, drop_last=True, logger=None):
        """
        Args:
            dataset: The DDMDataset instance. Must have `bucket_assignments` property.
            batch_size: Size of batches to yield.
            device: Unused in this implementation (operates on indices). Kept for compatibility.
            shuffle: If True, shuffle indices within buckets and the order of batches.
            drop_last: If True, drop the last incomplete batch from each bucket.
            logger: Optional logger instance.
        """
        super().__init__(dataset) # Pass dataset to parent Sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.logger = logger if logger else logging.getLogger(__name__)

        # --- Group indices by bucket ---
        self.indices_by_bucket = defaultdict(list)
        try:
            # Access the bucket assignments tensor (triggers loading if needed)
            all_bucket_assignments = self.dataset.bucket_assignments.cpu().numpy() # Work with numpy on CPU
            if len(all_bucket_assignments) != len(self.dataset):
                 raise ValueError(f"Length mismatch: bucket_assignments ({len(all_bucket_assignments)}) vs dataset ({len(self.dataset)})")

            for idx, bucket_id in enumerate(all_bucket_assignments):
                self.indices_by_bucket[bucket_id].append(idx)

        except Exception as e:
             self.logger.error(f"Failed to group indices by bucket: {e}", exc_info=True)
             raise

        self.num_buckets = len(self.indices_by_bucket)
        if self.dataset.rank == 0: # Log only on rank 0
             self.logger.info(f"Created {self.num_buckets} buckets.")
             for bucket_id, indices in self.indices_by_bucket.items():
                 self.logger.info(f"  Bucket {bucket_id}: {len(indices)} samples")

        # --- Create batches ---
        self.batches = []
        for bucket_id in self.indices_by_bucket:
            indices = self.indices_by_bucket[bucket_id]
            if self.shuffle:
                # Shuffle indices within the bucket
                np.random.shuffle(indices)

            # Create batches for this bucket
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                # Handle drop_last
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    self.batches.append(batch_indices)

        if not self.batches:
             self.logger.warning("BucketBatchSampler created no batches. Check dataset size, batch size, and drop_last setting.")

        if self.shuffle:
            # Shuffle the order of the generated batches
            np.random.shuffle(self.batches)

        self.num_batches = len(self.batches)
        if self.dataset.rank == 0:
            self.logger.info(f"BucketBatchSampler initialized with {self.num_batches} batches.")

    def __iter__(self):
        # If shuffling epoch-to-epoch is desired, re-shuffle here
        if self.shuffle:
            # Re-shuffle indices within buckets AND the order of batches for each epoch
            self.batches = []
            for bucket_id in self.indices_by_bucket:
                indices = self.indices_by_bucket[bucket_id]
                np.random.shuffle(indices) # Shuffle within bucket
                for i in range(0, len(indices), self.batch_size):
                    batch_indices = indices[i : i + self.batch_size]
                    if len(batch_indices) == self.batch_size or not self.drop_last:
                        self.batches.append(batch_indices)
            np.random.shuffle(self.batches) # Shuffle batch order

        # Yield the pre-computed batches
        yield from self.batches

    def __len__(self):
        """Number of batches per epoch."""
        return self.num_batches

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

        # Use the refined BucketBatchSampler
        batch_sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=config.batch_size,
            # device=device, # device not used by sampler
            shuffle=True,
            drop_last=True,
            logger=dataset.logger # Pass dataset's logger
        )

        # Configure loader with GPU optimizations
        loader_config_start = time.time()
        try:
            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=DDMDataset.collate_fn # Use static collate_fn
            )
            print(f"[Rank {rank}] DataLoader created successfully")
        except Exception as e:
            print(f"[Rank {rank}] ERROR creating DataLoader:")
            print(f"Exception: {str(e)}")
            print(f"Sampler indices: {list(batch_sampler)[0] if len(batch_sampler) > 0 else 'empty'}")
            raise

        # Warmup pipeline
        warmup_start = time.time()
        try:
            print(f"[Rank {rank}] Warming up loader...")
            for _ in range(1):
                batch = next(iter(data_loader))
                print(f"[Rank {rank}] Warmup batch shapes:")
                for k, v in batch.items():
                    print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
            print(f"[Rank {rank}] Warmup successful")
        except Exception as e:
            print(f"[Rank {rank}] Warmup failed:")
            print(f"Exception: {str(e)}")
            print(f"Failing indices: {list(batch_sampler)[0] if len(batch_sampler) > 0 else 'empty'}")
            if hasattr(e, 'args') and len(e.args) > 1:
                print(f"Problematic index: {e.args[1]}")
            print("Skipping warmup, continuing with training...")

        expert_loaders[expert_idx] = data_loader
        loader_pbar.update(1)

    loader_pbar.close()
    print(f"\n[Rank {rank}] ===== LOADER INITIALIZATION COMPLETE =====")
    print(f"[Rank {rank}] Created {len(expert_loaders)} expert loaders")
    print(f"[Rank {rank}] Total initialization time: {time.time() - loader_start:.2f}s")

    return expert_loaders
