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
    Loads precomputed features (latents, conditions, cluster assignments)
    from a specified cache directory.
    """
    def __init__(self, config_dict, split='train', logger=None):
        self.config = SimpleNamespace(**config_dict) # Convert dict to namespace
        self.split = split # Currently unused, assuming one large dataset
        self.logger = logger if logger else logging.getLogger(__name__)

        self.feature_cache_path = self.config.feature_cache_path
        self.num_experts = self.config.num_experts

        # --- File Discovery and Metadata Loading ---
        self._file_map = {} # Maps global index to (file_idx, index_within_file)
        self._file_paths = {} # Stores paths for different feature types
        self._file_lengths = [] # Stores number of items per file part
        self.total_samples = 0

        self._discover_files()
        if self.total_samples == 0:
            raise FileNotFoundError(f"No data files found or metadata invalid in {self.feature_cache_path}")

        self.logger.info(f"Found {self.total_samples} total samples across {len(self._file_lengths)} file parts.")

        # --- Lazy Load Placeholders ---
        self._loaded_files = {} # Cache for currently loaded file parts {feature_type: {file_idx: tensor}}
        self._current_file_idx = { 'latent': -1, 'clip': -1, 'cluster': -1, 't5': -1 } # Track loaded file index per feature

        # --- Bucket Assignments (Loaded on demand) ---
        self._bucket_assignments = None # Tensor of bucket indices for all samples

    def _discover_files(self):
        """Discovers feature files and loads metadata."""
        metadata_path = os.path.join(self.feature_cache_path, "metadata.json")
        latent_dir = os.path.join(self.feature_cache_path, "latents")
        clip_dir = os.path.join(self.feature_cache_path, "clip")
        cluster_dir = os.path.join(self.feature_cache_path, "clusters")
        t5_dir = os.path.join(self.feature_cache_path, "t5") # Add T5 directory

        if not os.path.exists(latent_dir):
            raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

        # Find latent files to establish the base structure
        latent_files = sorted(glob.glob(os.path.join(latent_dir, "*.pt")))
        if not latent_files:
            raise FileNotFoundError(f"No latent files (.pt) found in {latent_dir}")

        self._file_paths['latent'] = latent_files
        num_files = len(latent_files)

        # Try loading metadata if it exists
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.total_samples = metadata['total_samples']
                self._file_lengths = metadata['file_lengths']
                if len(self._file_lengths) != num_files:
                     self.logger.warning("Metadata file count mismatch. Re-calculating file lengths.")
                     self._file_lengths = [] # Force recalculation below
                else:
                     self.logger.info("Loaded file structure from metadata.json")

            except Exception as e:
                self.logger.warning(f"Could not load or parse metadata.json: {e}. Will infer structure.")
                self.total_samples = 0
                self._file_lengths = []

        # Infer structure if metadata failed or wasn't present
        if not self._file_lengths:
            self.logger.info("Inferring file structure by loading latent file headers...")
            running_total = 0
            for i, f_path in enumerate(latent_files):
                try:
                     # Quick load just to get shape (less memory intensive)
                     # Note: This still reads the file header. Faster alternatives might exist.
                     tensor_info = torch.load(f_path, map_location='cpu', weights_only=True if hasattr(torch, 'weights_only') else False) # weights_only if >=1.13
                     # If weights_only isn't available or fails, load the whole tensor to get len
                     if not hasattr(tensor_info, '__len__'):
                           tensor_info = torch.load(f_path, map_location='cpu')
                     
                     length = len(tensor_info)
                     self._file_lengths.append(length)
                     running_total += length
                except Exception as e:
                     raise IOError(f"Failed to read header/length of latent file {f_path}: {e}")
            self.total_samples = running_total
            # Optionally save the inferred metadata here if desired

        # Map global index to file index and index within file
        current_offset = 0
        for file_idx, length in enumerate(self._file_lengths):
            for i in range(length):
                self._file_map[current_offset + i] = (file_idx, i)
            current_offset += length

        # Verify and map other feature files (CLIP, Cluster, T5)
        for feature_type, feature_dir in [('clip', clip_dir), ('cluster', cluster_dir), ('t5', t5_dir)]:
             if os.path.exists(feature_dir):
                  pattern = "*.cluster.pt" if feature_type == 'cluster' else "*.pt"
                  feature_files = sorted(glob.glob(os.path.join(feature_dir, pattern)))
                  if len(feature_files) == num_files:
                       self._file_paths[feature_type] = feature_files
                       self.logger.info(f"Found matching {feature_type} files.")
                  else:
                       self.logger.warning(f"Mismatch in number of {feature_type} files ({len(feature_files)}) vs latent files ({num_files}). {feature_type.capitalize()} features might be unavailable.")
                       self._file_paths[feature_type] = []
             else:
                  self.logger.warning(f"{feature_type.capitalize()} directory not found: {feature_dir}. Features unavailable.")
                  self._file_paths[feature_type] = []


    def _load_feature_file(self, feature_type: str, file_idx: int):
        """Loads a specific feature file into the cache if not already loaded."""
        if file_idx == self._current_file_idx.get(feature_type, -1):
            return # Already loaded

        if feature_type not in self._file_paths or not self._file_paths[feature_type]:
            # Feature type is not available
            self._loaded_files[feature_type] = {file_idx: None} # Mark as unavailable
            self._current_file_idx[feature_type] = file_idx
            return

        try:
            filepath = self._file_paths[feature_type][file_idx]
            # Clear previous file cache for this feature type to save memory
            if feature_type in self._loaded_files:
                 self._loaded_files[feature_type].clear()

            # Load the new file (move to CPU to avoid holding GPU memory)
            loaded_tensor = torch.load(filepath, map_location='cpu')
            self._loaded_files[feature_type] = {file_idx: loaded_tensor}
            self._current_file_idx[feature_type] = file_idx
            # self.logger.debug(f"Loaded {feature_type} file index {file_idx}")
        except IndexError:
            self.logger.error(f"File index {file_idx} out of range for {feature_type}")
            self._loaded_files[feature_type] = {file_idx: None} # Mark as unavailable
        except Exception as e:
            self.logger.error(f"Failed to load {feature_type} file {filepath}: {e}")
            self._loaded_files[feature_type] = {file_idx: None} # Mark as unavailable


    def __getitem__(self, idx):
        """
        Loads and returns data for a given global index.
        Handles lazy loading of file parts.
        """
        if not 0 <= idx < self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset size {self.total_samples}")

        file_idx, index_in_file = self._file_map[idx]

        # Prepare the item dictionary
        item = {'index': idx} # Include original index

        # Load required features (Latent is mandatory)
        for feature_type in ['latent', 'clip', 'cluster', 't5']:
             self._load_feature_file(feature_type, file_idx)
             
             loaded_data = self._loaded_files.get(feature_type, {}).get(file_idx, None)

             if loaded_data is not None:
                 try:
                     # Note: .clone() is important if multiple workers might access the same cache
                     item[feature_type] = loaded_data[index_in_file].clone()
                 except IndexError:
                     self.logger.error(f"Index {index_in_file} out of range within {feature_type} file {file_idx} (size {len(loaded_data)}).")
                     item[feature_type] = None # Or handle error appropriately
                 except Exception as e:
                     self.logger.error(f"Error accessing index {index_in_file} in loaded {feature_type} file {file_idx}: {e}")
                     item[feature_type] = None
             else:
                  # Feature file wasn't loaded or available
                  item[feature_type] = None
                  if feature_type == 'latent': # Latent is critical
                       raise RuntimeError(f"Failed to load mandatory latent feature for index {idx} (file {file_idx})")
                  elif feature_type == 'cluster': # Cluster is critical for RouterTrainer
                       # Raise error only if router is being trained? Or always require clusters?
                       # Let's warn for now, trainer can raise error if needed.
                       self.logger.warning(f"Cluster assignment not available for index {idx} (file {file_idx})")


        # Rename keys and generate IDs
        final_item = {}
        # Note: We clone tensors fetched from cache to avoid modification issues if using multiple workers
        if item.get('latent') is not None:
             final_item['image'] = item['latent'].clone() # Use 'image' for latent tensor
        if item.get('clip') is not None:
             final_item['y'] = item['clip'].clone() # Use 'y' for CLIP condition
        if item.get('cluster') is not None:
             # Ensure cluster_idx is a scalar tensor if loaded as single element tensor
             cluster_idx_tensor = item['cluster'].clone()
             final_item['cluster_idx'] = cluster_idx_tensor.item() if cluster_idx_tensor.numel() == 1 else cluster_idx_tensor
        if item.get('t5') is not None:
             final_item['txt'] = item['t5'].clone() # Use 'txt' for T5 condition

        # --- Generate img_ids and txt_ids (Crucial for Flux/ExpertModel) ---
        if final_item.get('image') is not None:
            # Based on models/flux/sampling_flux.py: prepare()
            # Assumes latent is [C, H, W]
            _, latent_h, latent_w = final_item['image'].shape
            # Flux expects image input reshaped and IDs based on halved dimensions before reshaping
            img_h, img_w = math.ceil(latent_h / 2), math.ceil(latent_w / 2) # Match sampling_flux unpack H/W calculation? Needs verification with Flux model structure. Or simply use H//2, W//2 if exact patch division is guaranteed.
            
            # Create img_ids tensor [H * W, 3]
            img_ids_shape = (img_h * img_w, 3)
            img_ids = torch.zeros(img_ids_shape, dtype=torch.float32) # Match dtype potentially expected by flux.math.rope
            
            # Create coordinate grid
            rows = torch.arange(img_h, dtype=torch.float32)
            cols = torch.arange(img_w, dtype=torch.float32)
            grid_h, grid_w = torch.meshgrid(rows, cols, indexing='ij') # Use 'ij' indexing
            
            # Populate img_ids: [id_type=1, row, col] (id_type might be different, check Flux usage)
            img_ids[:, 0] = 1.0 # Assuming 1 signifies image token type, adjust if needed
            img_ids[:, 1] = grid_h.flatten()
            img_ids[:, 2] = grid_w.flatten()
            final_item['img_ids'] = img_ids

        if final_item.get('txt') is not None:
             # Flux sampling_flux.py uses zeros for txt_ids: [id_type=0, 0, 0]
             # Assumes txt is [L, D] where L is sequence length
             txt_len = final_item['txt'].shape[0]
             txt_ids_shape = (txt_len, 3)
             txt_ids = torch.zeros(txt_ids_shape, dtype=torch.float32) # Match dtype
             # txt_ids[:, 0] = 0.0 # Assuming 0 signifies text token type
             final_item['txt_ids'] = txt_ids


        # Filter out None values unless None is valid input (e.g., optional condition)
        # Ensure mandatory keys are present
        if 'image' not in final_item:
            raise RuntimeError(f"Mandatory 'image' (latent) feature missing for index {idx}")
        if 'cluster_idx' not in final_item:
             # Cluster index is needed for router training, maybe optional for expert?
             # Let's keep the warning from before for now.
             self.logger.warning(f"Cluster assignment ('cluster_idx') not available for index {idx}")


        # Return only necessary items for the trainer
        # Filter based on expected keys for the models/trainers
        # This filtering might be better handled in _get_batch_data if different trainers need different subsets
        return final_item


    def __len__(self):
        return self.total_samples

    @property
    def bucket_assignments(self) -> torch.Tensor:
        """Loads and returns the bucket assignments for all samples."""
        if self._bucket_assignments is None:
            self.logger.info("Loading bucket assignments...")
            self._bucket_assignments = self._load_bucket_assignments()
        return self._bucket_assignments

    def _load_bucket_assignments(self) -> torch.Tensor:
        """Loads bucket assignments from cached files."""
        bucket_dir = os.path.join(self.feature_cache_path, "buckets")
        if not os.path.exists(bucket_dir):
             raise FileNotFoundError(f"Bucket assignment directory not found: {bucket_dir}")

        bucket_files = sorted(glob.glob(os.path.join(bucket_dir, "*.pt")))
        if len(bucket_files) != len(self._file_paths['latent']):
             raise FileNotFoundError("Mismatch between number of bucket files and latent files.")

        all_assignments = []
        total_loaded = 0
        for i, f_path in enumerate(bucket_files):
            try:
                assignments = torch.load(f_path, map_location='cpu')
                if len(assignments) != self._file_lengths[i]:
                     raise ValueError(f"Length mismatch in bucket file {f_path}: expected {self._file_lengths[i]}, got {len(assignments)}")
                all_assignments.append(assignments)
                total_loaded += len(assignments)
            except Exception as e:
                raise IOError(f"Failed to load bucket file {f_path}: {e}")

        if total_loaded != self.total_samples:
             raise ValueError(f"Total samples in bucket files ({total_loaded}) does not match dataset size ({self.total_samples}).")

        return torch.cat(all_assignments, dim=0)


    # --- Methods below might be less critical for basic operation ---
    def _compute_cluster_statistics(self):
        """Compute and cache cluster statistics"""
        if self._bucket_assignments is None:
            self.logger.warning("No bucket assignments found - using uniform distribution")
            self.cluster_counts = torch.ones(self.config.num_experts, dtype=torch.long, device=self.device)
            return

        # Count samples per cluster - ensure long dtype
        unique_clusters, counts = torch.unique(
            self._bucket_assignments, 
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
        Collates a list of item dictionaries into a single batch dictionary.
        Handles padding if sequences have variable lengths (e.g., T5 embeddings).
        Uses default_collate for efficient stacking of uniform tensors.
        """
        # Use default collate which handles stacking tensors of the same shape
        collated_batch = torch.utils.data.default_collate(batch)

        # --- Padding for variable length sequences (if needed) ---
        # Check if 'txt' or 'txt_ids' are present and need padding
        # Example for 'txt' (T5 embeddings):
        if 'txt' in collated_batch:
             # Check if tensors in the batch had different lengths originally
             # default_collate might raise error or pad incorrectly if lengths vary.
             # We need the original list to pad correctly.
             txt_sequences = [item['txt'] for item in batch if 'txt' in item]
             if txt_sequences:
                  # Pad to the max length in the batch
                  padded_txt = torch.nn.utils.rnn.pad_sequence(txt_sequences, batch_first=True, padding_value=0.0) # Use 0.0 for padding
                  collated_batch['txt'] = padded_txt

                  # If 'txt_ids' exist, pad them similarly
                  if 'txt_ids' in collated_batch:
                       txt_ids_sequences = [item['txt_ids'] for item in batch if 'txt_ids' in item]
                       # Pad the IDs tensor. Shape is [L, 3], padding should add [0, 0, 0] vectors.
                       padded_txt_ids = torch.nn.utils.rnn.pad_sequence(txt_ids_sequences, batch_first=True, padding_value=0.0)
                       collated_batch['txt_ids'] = padded_txt_ids
                  # TODO: Generate attention mask for 'txt' if model requires it
                  # attention_mask = (padded_txt != 0).any(dim=-1).long() # Example: mask based on padding token (0)
                  # collated_batch['txt_attention_mask'] = attention_mask


        return collated_batch

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
