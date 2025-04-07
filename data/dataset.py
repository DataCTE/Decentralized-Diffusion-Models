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
import glob
import json
import math
from pathlib import Path # Add Path import


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
    """
    Dataset for Decentralized Diffusion Models.
    Loads precomputed features (latents, text embeddings, cluster assignments, etc.)
    from a cache directory structured by feature type. Handles bucketing.
    """
    # Map feature types to directory names and expected file extensions
    FEATURE_INFO = {
        'latents': {'dir': 'latents', 'ext': '.pt'},
        'clip':    {'dir': 'clip', 'ext': '.pt'},
        't5':      {'dir': 't5', 'ext': '.pt'},
        'dino':    {'dir': 'dino', 'ext': '.pt'},
        'dims':    {'dir': 'dims', 'ext': '.pt'},
        'buckets': {'dir': 'buckets', 'ext': '.pt'},
        'clusters':{'dir': 'clusters', 'ext': '.cluster.pt'} # Use distinct extension if needed
    }
    # Features that are mandatory for the dataset to function
    MANDATORY_FEATURES = {'latents', 'clip', 't5', 'dims', 'buckets', 'clusters'}
    
    def __init__(self, config_dict, split='train', logger=None):
        """
        Initializes the dataset.

        Args:
            config_dict (dict): Configuration dictionary containing data parameters like
                                'feature_cache_path', 'latent_channels', 'num_experts',
                                bucket settings, etc.
            split (str): Data split ('train', 'val', etc.). Currently unused, assumes train.
            logger (logging.Logger, optional): Logger instance. Defaults to None.
        """
        self.config = dict_to_sns(config_dict) # Convert dict to SimpleNamespace
        self.feature_cache_path = Path(self.config.feature_cache_path)
        self.split = split
        self.logger = logger if logger else logging.getLogger(__name__)

        # Validate mandatory config keys
        if not self.feature_cache_path.exists():
            raise FileNotFoundError(f"Feature cache path does not exist: {self.feature_cache_path}")

        # Check for mandatory feature directories (only check, files discovered later)
        for feat in self.MANDATORY_FEATURES:
            info = self.FEATURE_INFO.get(feat)
            if not info or not (self.feature_cache_path / info['dir']).exists():
                 self.logger.warning(f"Mandatory feature directory '{info['dir'] if info else feat}' not found in {self.feature_cache_path}. Dataset loading might fail.")
                 # Could raise an error here if strictly required

        # Discover feature files and determine dataset size
        self.file_list, self.cumulative_sizes, self.total_samples = self._discover_files()
        if self.total_samples == 0:
            raise ValueError(f"No data samples found in feature cache: {self.feature_cache_path}")
        self.logger.info(f"Discovered {len(self.file_list)} batch files, total samples: {self.total_samples}")

        # Load bucket assignments (essential for BucketBatchSampler)
        self._bucket_assignments = self._load_bucket_assignments()

        # Precompute cluster statistics if needed (optional)
        self._cluster_stats = None
        # self._compute_cluster_statistics() # Uncomment if needed

        # Feature cache (in-memory) - Use with caution for large datasets
        self.feature_cache = {} # Cache loaded batch files {('feature_type', file_idx): tensor}


    def _discover_files(self):
        """
        Discovers precomputed feature files (rank_batchidx.pt) and calculates dataset size.

        Assumes all feature types have corresponding files for each batch index across ranks.
        Uses 'dims' feature as the reference for determining batch structure and size.
        """
        dims_dir = self.feature_cache_path / self.FEATURE_INFO['dims']['dir']
        if not dims_dir.exists():
            raise FileNotFoundError(f"Mandatory 'dims' feature directory not found at {dims_dir}")

        # Find all dims files: rank_batchidx.pt
        dims_files = sorted(list(dims_dir.glob("*_*.pt"))) # Glob for rank_batchidx structure
        if not dims_files:
            self.logger.error(f"No dimension files ('*_*.pt') found in {dims_dir}")
            return [], {}, 0

        # Group files by batch index
        file_groups = {} # {batch_idx: [path_rank0, path_rank1, ...]}
        batch_sizes = {} # {batch_idx: num_samples}

        self.logger.info(f"Discovering files based on {len(dims_files)} dims files in {dims_dir}...")

        total_samples = 0
        processed_batch_indices = set()

        for file_path in dims_files:
            try:
                filename = file_path.name
                parts = filename.split('_')
                if len(parts) != 2 or not parts[0].isdigit() or not parts[1].endswith('.pt'):
                     self.logger.warning(f"Skipping unexpected filename format: {filename} in {dims_dir}")
                     continue

                # rank_str = parts[0] # Rank is part of the filename now
                batch_idx_str = parts[1].replace('.pt', '')
                batch_idx = int(batch_idx_str)

                if batch_idx not in processed_batch_indices:
                    # This is the first time we see this batch_idx, load it to get size
                    try:
                        dims_tensor = torch.load(file_path, map_location='cpu')
                        num_samples = dims_tensor.shape[0]
                        batch_sizes[batch_idx] = num_samples
                        total_samples += num_samples
                        processed_batch_indices.add(batch_idx)
                        file_groups[batch_idx] = [file_path] # Initialize group
                    except Exception as e:
                        self.logger.error(f"Failed to load dims file {file_path} to get size: {e}")
                        continue # Skip this batch if dims file fails
                else:
                    # We've already processed this batch_idx (from another rank's file)
                    if batch_idx in file_groups:
                         file_groups[batch_idx].append(file_path)
                    else:
                         # This case shouldn't happen if logic is correct, but log if it does
                         self.logger.warning(f"Found file for already processed batch_idx {batch_idx} but group not initialized: {file_path}")

            except ValueError:
                self.logger.warning(f"Could not parse batch index from filename: {filename}")
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")

        # --- Verification ---
        # Check if all mandatory features exist for the discovered batch indices
        required_features = self.MANDATORY_FEATURES # Use the defined set
        verified_batch_indices = sorted(list(processed_batch_indices))
        final_file_list = [] # List of tuples: (batch_idx, num_samples)
        final_total_samples = 0

        self.logger.info(f"Verifying presence of mandatory features for {len(verified_batch_indices)} discovered batch indices...")
        for batch_idx in verified_batch_indices:
            batch_complete = True
            for feature_type in required_features:
                info = self.FEATURE_INFO[feature_type]
                expected_filename_pattern = f"*_{batch_idx:06d}{info['ext']}" # Assumes 6-digit padding
                feature_dir = self.feature_cache_path / info['dir']
                # Check if at least one rank's file exists for this batch_idx and feature
                matches = list(feature_dir.glob(expected_filename_pattern))
                if not matches:
                    self.logger.warning(f"Missing mandatory feature '{feature_type}' for batch index {batch_idx} (Pattern: {feature_dir / expected_filename_pattern}). Skipping batch.")
                    batch_complete = False
                    break # Stop checking features for this batch

            if batch_complete:
                num_samples = batch_sizes[batch_idx]
                final_file_list.append((batch_idx, num_samples))
                final_total_samples += num_samples
            else:
                 # Decrement total samples if batch was skipped
                 original_size = batch_sizes.get(batch_idx, 0)
                 # This adjustment might be complex if total_samples wasn't built correctly above
                 # Safer to recalculate total_samples based on final_file_list
                 pass # Recalculate below

        # Recalculate total samples accurately based on verified batches
        final_total_samples = sum(num_samples for _, num_samples in final_file_list)

        # Calculate cumulative sizes for index mapping
        cumulative_sizes = {}
        current_pos = 0
        for batch_idx, num_samples in final_file_list:
            cumulative_sizes[batch_idx] = current_pos
            current_pos += num_samples

        if not final_file_list:
             self.logger.error("No complete batches found after verifying mandatory features.")

        return final_file_list, cumulative_sizes, final_total_samples


    def _get_file_path(self, feature_type: str, batch_idx: int):
         """
         Gets the path(s) for a given feature type and batch index.
         It returns the first available file found across potential ranks.
         """
         info = self.FEATURE_INFO[feature_type]
         feature_dir = self.feature_cache_path / info['dir']
         # Use the precise extension defined in FEATURE_INFO
         filename_pattern = f"*_{batch_idx:06d}{info['ext']}" # Assumes 6 digits padding from precompute
         
         potential_files = sorted(list(feature_dir.glob(filename_pattern)))
         
         if not potential_files:
              # self.logger.warning(f"No file found for feature '{feature_type}', batch {batch_idx} with pattern {filename_pattern}")
              return None # Indicate file not found
              
         # Return the first found file path (e.g., rank 0's if available)
         return potential_files[0]


    def _load_feature_file(self, feature_type: str, file_idx: int):
        """
        Loads data for a specific feature type and original batch file index.
        Uses an in-memory cache.
        """
        cache_key = (feature_type, file_idx)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        file_path = self._get_file_path(feature_type, file_idx)
        if file_path is None:
             self.logger.error(f"Failed to find file path for feature '{feature_type}', batch index {file_idx}. Cannot load.")
             # Return None or raise error depending on how critical this feature is
             # If mandatory features are checked in discover, this might indicate a deeper issue
             return None

        try:
            data = torch.load(file_path, map_location='cpu') # Load to CPU
            self.feature_cache[cache_key] = data # Store in cache
            return data
        except Exception as e:
            self.logger.error(f"Error loading feature file {file_path}: {e}")
            # Decide how to handle load errors (return None, placeholder, raise)
            return None # Return None to indicate loading failure

    def _find_batch_for_sample(self, idx):
        """Find the batch_idx and index within that batch for a global sample index."""
        target_batch_idx = -1
        index_in_batch = -1

        # Iterate through discovered batches to find where idx falls
        for batch_idx, num_samples in self.file_list:
             batch_start_offset = self.cumulative_sizes[batch_idx]
             if batch_start_offset <= idx < batch_start_offset + num_samples:
                  target_batch_idx = batch_idx
                  index_in_batch = idx - batch_start_offset
                  break # Found the correct batch

        if target_batch_idx == -1:
            raise IndexError(f"Sample index {idx} out of range (total samples: {self.total_samples})")

        return target_batch_idx, index_in_batch

    def __getitem__(self, idx):
        """
        Retrieves precomputed features for a single data sample.

        Args:
            idx (int): The global index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the requested features for the sample.
                  Keys are feature types (e.g., 'latents', 'clip'), values are tensors.
                  Returns None for features that failed to load.
        """
        if not (0 <= idx < self.total_samples):
             raise IndexError(f"Index {idx} out of bounds for dataset with size {self.total_samples}")

        # 1. Find which batch file this index belongs to
        batch_idx, index_in_batch = self._find_batch_for_sample(idx)

        # 2. Load all necessary feature files for this batch_idx (use cache)
        # Define all features potentially needed by trainers/models
        all_feature_types = list(self.FEATURE_INFO.keys())
        batch_data_cache = {}
        for feature_type in all_feature_types:
             # Check if directory exists before trying to load (optimization)
             info = self.FEATURE_INFO[feature_type]
             feature_dir = self.feature_cache_path / info['dir']
             if feature_dir.exists():
                 batch_data_cache[feature_type] = self._load_feature_file(feature_type, batch_idx)
             else:
                 # Don't try to load if dir doesn't exist
                 batch_data_cache[feature_type] = None


        # 3. Extract the specific sample's data from the loaded batch tensors
        sample_data = {}
        load_successful = True
        for feature_type, batch_tensor in batch_data_cache.items():
             if batch_tensor is not None:
                 try:
                      sample_data[feature_type] = batch_tensor[index_in_batch]
                 except IndexError:
                      self.logger.error(f"IndexError accessing {feature_type} data: index_in_batch={index_in_batch}, batch_tensor shape={batch_tensor.shape}, batch_idx={batch_idx}")
                      sample_data[feature_type] = None # Mark as failed
                      if feature_type in self.MANDATORY_FEATURES:
                           load_successful = False
                 except Exception as e:
                      self.logger.error(f"Error extracting sample {idx} (batch {batch_idx}, index {index_in_batch}) for feature {feature_type}: {e}")
                      sample_data[feature_type] = None
                      if feature_type in self.MANDATORY_FEATURES:
                           load_successful = False
             else:
                 # Feature file failed to load or dir didn't exist
                 sample_data[feature_type] = None
                 # Check if this was a mandatory feature that failed
                 if feature_type in self.MANDATORY_FEATURES and (self.feature_cache_path / self.FEATURE_INFO[feature_type]['dir']).exists():
                     # If dir exists but load failed for mandatory feature
                     load_successful = False


        # Add index and potentially other metadata if needed
        sample_data['index'] = idx

        # --- Generate img_ids and txt_ids ---
        # Based on sampling_flux.py logic
        latents = sample_data.get('latents')
        t5_embeddings = sample_data.get('t5')

        if latents is not None:
            # Assuming latents are [C, H, W]
            # Flux expects patches of 2x2, so effective grid size is H/2 x W/2
            latent_h, latent_w = latents.shape[-2], latents.shape[-1]
            grid_h, grid_w = math.ceil(latent_h / 2), math.ceil(latent_w / 2) # Use ceil for robustness
            num_img_patches = grid_h * grid_w

            # Create coordinate grid (y, x, 0) - matches flux.math.rope input format potentially
            img_ids = torch.zeros(grid_h, grid_w, 3, dtype=torch.float32)
            img_ids[..., 0] = torch.arange(grid_h, dtype=torch.float32)[:, None] # y-coordinates
            img_ids[..., 1] = torch.arange(grid_w, dtype=torch.float32)[None, :] # x-coordinates
            # Third dimension is often kept 0 for images in RoPE implementations
            sample_data['img_ids'] = img_ids.view(num_img_patches, 3) # Reshape to [N_patches, 3]
        else:
            # Handle missing latents - create dummy or raise error
            self.logger.warning(f"Latents missing for sample {idx}, creating dummy img_ids.")
            sample_data['img_ids'] = torch.zeros(1, 3, dtype=torch.float32) # Minimal placeholder
            if 'latents' in self.MANDATORY_FEATURES: load_successful = False


        if t5_embeddings is not None:
            # Assuming t5_embeddings are [SeqLen, Dim]
            num_txt_tokens = t5_embeddings.shape[0]
            # Flux uses zeros for text IDs, RoPE applied differently? Check Flux model.
            # Creating sequence indices (0, 1, 2...) might be more standard for text RoPE.
            txt_ids = torch.zeros(num_txt_tokens, 3, dtype=torch.float32)
            txt_ids[..., 0] = torch.arange(num_txt_tokens, dtype=torch.float32) # Sequence position
            sample_data['txt_ids'] = txt_ids
        else:
             # Handle missing T5 - create dummy or raise error
            self.logger.warning(f"T5 embeddings missing for sample {idx}, creating dummy txt_ids.")
            sample_data['txt_ids'] = torch.zeros(1, 3, dtype=torch.float32) # Minimal placeholder
            if 't5' in self.MANDATORY_FEATURES: load_successful = False


        # Handle mandatory feature load failures
        if not load_successful:
             self.logger.error(f"Failed to load one or more mandatory features for sample index {idx}. Returning None or partial data.")
             # Option 1: Return None to signal the collate_fn to skip this sample
             # return None
             # Option 2: Return partial data (might cause issues downstream)
             # return sample_data
             # Option 3: Raise an exception
             raise RuntimeError(f"Failed to load mandatory features for sample index {idx}. Check logs.")

        return sample_data

    def __len__(self):
        return self.total_samples

    @property
    def bucket_assignments(self) -> torch.Tensor:
        """Returns the loaded bucket assignments."""
        if self._bucket_assignments is None:
            self.logger.error("Bucket assignments accessed before loading.")
            # Handle appropriately: raise error or return default
            raise ValueError("Bucket assignments not loaded.")
        return self._bucket_assignments

    def _load_bucket_assignments(self) -> torch.Tensor:
        """Loads bucket assignments for the entire dataset."""
        assignments_dir = self.feature_cache_path / self.FEATURE_INFO['buckets']['dir']
        if not assignments_dir.exists():
            raise FileNotFoundError(f"Bucket assignments directory not found: {assignments_dir}")

        # Load assignments from all rank_*_*.pt files and concatenate
        all_assignments = []
        # Use self.file_list which contains verified batch indices and sizes
        self.logger.info(f"Loading bucket assignments for {len(self.file_list)} verified batches...")
        for batch_idx, num_samples in tqdm(self.file_list, desc="Loading bucket assignments"):
            file_path = self._get_file_path('buckets', batch_idx) # Use helper to find the file
            if file_path:
                try:
                    batch_assignments = torch.load(file_path, map_location='cpu')
                    if batch_assignments.shape[0] != num_samples:
                         self.logger.warning(f"Bucket assignment count mismatch for batch {batch_idx}. Expected {num_samples}, got {batch_assignments.shape[0]}. File: {file_path}")
                         # Handle mismatch: skip, truncate, error? For now, use loaded count.
                    all_assignments.append(batch_assignments)
                except Exception as e:
                    self.logger.error(f"Error loading bucket assignment file {file_path}: {e}")
                    # Handle error: skip batch, raise error? Requires consistent handling.
                    raise RuntimeError(f"Failed to load bucket assignments for batch {batch_idx}")
            else:
                self.logger.error(f"Could not find bucket assignment file for batch {batch_idx}. Cannot proceed.")
                raise FileNotFoundError(f"Missing bucket assignment file for batch {batch_idx}")

        if not all_assignments:
             raise ValueError("Failed to load any bucket assignments.")

        full_assignments = torch.cat(all_assignments, dim=0)

        # Validate size
        if full_assignments.shape[0] != self.total_samples:
            self.logger.warning(f"Total loaded bucket assignments ({full_assignments.shape[0]}) does not match expected dataset size ({self.total_samples}).")
            # This might indicate issues in discovery or loading.

        self.logger.info(f"Successfully loaded bucket assignments for {full_assignments.shape[0]} samples.")
        return full_assignments.short() # Use short tensor for memory efficiency


    @staticmethod
    def collate_fn(batch):
        """
        Collates a list of samples (dictionaries) into a single batch dictionary.
        Filters out None samples resulting from loading errors.
        """
        # Filter out None samples if __getitem__ returns None on error
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return None # Return None if the entire batch failed

        # Get keys from the first sample (assume all samples have the same structure)
        elem_keys = batch[0].keys()
        collated = {}

        for key in elem_keys:
            # Get the list of tensors/values for this key
            values = [sample[key] for sample in batch]

            # Stack tensors, handle other types appropriately
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], (int, float, str)):
                # Convert numerical types to tensors if desired, keep strings as list
                try:
                    # Attempt to convert to tensor, fallback to list
                    collated[key] = torch.tensor(values)
                except TypeError:
                    collated[key] = values # Keep as list if not numerical
            elif isinstance(values[0], list): # e.g. list of strings for captions before tokenization
                 collated[key] = values # Keep as list of lists/strings
            else:
                 # Handle other potential types if necessary
                 collated[key] = values

        return collated

class CombinedBatchSampler(Sampler):
    """Combines multiple batch samplers into one epoch."""
    def __init__(self, batch_samplers):
        self.batch_samplers = batch_samplers
        self._len = sum(len(bs) for bs in self.batch_samplers)
    
    def __iter__(self):
        iterators = [iter(bs) for bs in self.batch_samplers]
        while iterators:
            # Randomly pick a sampler to yield from
            idx = random.randrange(len(iterators))
            try:
                yield next(iterators[idx])
            except StopIteration:
                iterators.pop(idx) # Remove exhausted iterator
    
    def __len__(self):
        return self._len

class BucketBatchSampler(Sampler):
    """
    Yields batches of indices, ensuring each batch comes from the same bucket.
    Handles distributed training by ensuring each rank gets unique batches.
    """
    def __init__(self, dataset: DDMDataset, batch_size: int, shuffle=True, drop_last=True, logger=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.logger = logger if logger else logging.getLogger(__name__)

        # Distributed training setup
        self.rank = 0
        self.world_size = 1
        if torch.distributed.is_initialized():
             self.rank = torch.distributed.get_rank()
             self.world_size = torch.distributed.get_world_size()

        # Get bucket assignments (this triggers loading if not already done)
        try:
            assignments = self.dataset.bucket_assignments
        except FileNotFoundError:
            self.logger.error("Bucket assignments not found. Cannot use BucketBatchSampler.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading bucket assignments: {e}")
            raise

        self.num_buckets = assignments.max().item() + 1
        self.logger.info(f"Sampler Rank {self.rank}: Found {self.num_buckets} buckets.")

        # Group indices by bucket
        self.indices_by_bucket = defaultdict(list)
        for idx, bucket_id in enumerate(assignments.tolist()):
             self.indices_by_bucket[bucket_id].append(idx)

        # Prepare batches for each bucket
        self.batches_by_bucket = defaultdict(list)
        self.num_batches = 0
        for bucket_id in range(self.num_buckets):
            indices = self.indices_by_bucket[bucket_id]
            if self.shuffle:
                random.shuffle(indices) # Shuffle within bucket

            bucket_batches = list(chunks(indices, self.batch_size))

            # Handle drop_last for the bucket
            if self.drop_last and len(indices) % self.batch_size != 0:
                bucket_batches = bucket_batches[:-1]

            # Distribute batches across ranks
            # Each rank processes roughly 1/world_size of the batches per bucket
            num_bucket_batches = len(bucket_batches)
            batches_for_this_rank = bucket_batches[self.rank : num_bucket_batches : self.world_size]

            self.batches_by_bucket[bucket_id] = batches_for_this_rank
            self.num_batches += len(batches_for_this_rank)

            if self.rank == 0: # Log bucket info from rank 0
                 self.logger.info(f"Bucket {bucket_id}: {len(indices)} samples -> {num_bucket_batches} total batches -> {len(batches_for_this_rank)} batches for rank 0.")

        # Create the final list of all batches for this rank for the epoch
        self.epoch_batches = []
        for bucket_id in range(self.num_buckets):
            self.epoch_batches.extend(self.batches_by_bucket[bucket_id])

        # Shuffle the order of batches across buckets for the epoch
        if self.shuffle:
            random.shuffle(self.epoch_batches)

        self.logger.info(f"Sampler Rank {self.rank}: Total batches for this rank per epoch: {self.num_batches}")


    def __iter__(self):
        # If shuffling epoch-to-epoch is desired, re-shuffle here
        if self.shuffle:
            # Regenerate self.epoch_batches by reshuffling within buckets and then across buckets
            self.epoch_batches = []
            for bucket_id in range(self.num_buckets):
                indices = self.indices_by_bucket[bucket_id]
                random.shuffle(indices) # Reshuffle indices within bucket
                bucket_batches = list(chunks(indices, self.batch_size))
                if self.drop_last and len(indices) % self.batch_size != 0:
                    bucket_batches = bucket_batches[:-1]
                batches_for_this_rank = bucket_batches[self.rank : len(bucket_batches) : self.world_size]
                self.epoch_batches.extend(batches_for_this_rank)
            # Reshuffle the order of all batches for this rank
            random.shuffle(self.epoch_batches)

        return iter(self.epoch_batches)


    def __len__(self):
        # Total number of batches yielded by this rank per epoch
        return self.num_batches

    def _load_bucket_assignments(self) -> torch.Tensor:
        # ... existing code ...
        try:
            assignments = self.dataset.bucket_assignments
        except FileNotFoundError:
            self.logger.error("Bucket assignments not found. Cannot use BucketBatchSampler.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading bucket assignments: {e}")
            raise # Correct indentation for raise

        self.num_buckets = assignments.max().item() + 1
        self.logger.info(f"Sampler Rank {self.rank}: Found {self.num_buckets} buckets.")
        # ... rest of the method ...

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
            train_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=DDMDataset.collate_fn,
                pin_memory=True,  # Ensure this is set
                num_workers=config.num_workers,
                persistent_workers=config.num_workers > 0
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
                batch = next(iter(train_loader))
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

        expert_loaders[expert_idx] = train_loader
        loader_pbar.update(1)

    loader_pbar.close()
    print(f"\n[Rank {rank}] ===== LOADER INITIALIZATION COMPLETE =====")
    print(f"[Rank {rank}] Created {len(expert_loaders)} expert loaders")
    print(f"[Rank {rank}] Total initialization time: {time.time() - loader_start:.2f}s")

    return expert_loaders
