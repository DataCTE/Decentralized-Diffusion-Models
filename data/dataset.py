"""Dataset classes for Decentralized Diffusion Models."""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from collections import defaultdict
import logging
import time  
from tqdm.auto import tqdm
# Import centralized utilities and logging setup
from utils import is_main_process, get_rank, get_world_size, dict_to_sns # Keep dict_to_sns if needed here
from utils.logging import setup_distributed_logger # Correct import
import math
from pathlib import Path # Add Path import
import re
from typing import Optional
import os
import gc # Import garbage collector


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
        'dims':    {'dir': 'dims', 'ext': '.pt'},
        'buckets': {'dir': 'buckets', 'ext': '.pt'},
        'clusters':{'dir': 'clusters', 'ext': '.pt'}
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
                                This dictionary is expected to be FLAT, not nested like the main config.toml.
            split (str): Data split ('train', 'val', etc.). Currently unused, assumes train.
            logger (logging.Logger, optional): Logger instance. Defaults to None.
        """
        # Convert the FLAT config_dict to SimpleNamespace
        self.config = dict_to_sns(config_dict) 
        
        # Setup logger using the provided function if not passed
        # Pass the rank explicitly if available, otherwise setup_distributed_logger will try to get it
        current_rank = get_rank() if torch.distributed.is_initialized() else 0
        self.logger = logger if logger else setup_distributed_logger(__name__, rank=current_rank)

        # Access feature_cache_path directly from the flat self.config
        feature_cache_path_str = getattr(self.config, 'feature_cache_path', None)
        if not feature_cache_path_str:
             raise ValueError("Missing 'feature_cache_path' in the config_dict passed to DDMDataset.")
        self.feature_cache_path = Path(feature_cache_path_str)

        self.split = split

        # Validate mandatory config keys
        if not self.feature_cache_path.exists():
            raise FileNotFoundError(f"Feature cache path does not exist: {self.feature_cache_path}")

        # Check for mandatory feature directories upfront
        for feat in self.MANDATORY_FEATURES:
            info = self.FEATURE_INFO.get(feat)
            if not info:
                 self.logger.error(f"Configuration for mandatory feature '{feat}' missing in FEATURE_INFO.")
                 raise ValueError(f"Missing config for mandatory feature: {feat}")
            feature_dir_path = self.feature_cache_path / info['dir']
            if not feature_dir_path.exists():
                 self.logger.error(f"Mandatory feature directory '{info['dir']}' not found in {self.feature_cache_path}.")
                 raise FileNotFoundError(f"Mandatory feature directory '{info['dir']}' not found: {feature_dir_path}")

        # --- Retrieve expected batch size --- START EDIT ---
        self.expected_batch_size = getattr(self.config, 'precompute_batch_size', None)
        if self.expected_batch_size is None or not isinstance(self.expected_batch_size, int) or self.expected_batch_size <= 0:
            raise ValueError(f"Invalid or missing 'precompute_batch_size' ({self.expected_batch_size}) in config_dict passed to DDMDataset.")
        self.logger.info(f"Using expected precomputation batch size: {self.expected_batch_size}")
        # --- Retrieve expected batch size --- END EDIT ---

        # Discover feature files and determine dataset size (Optimized v2)
        discovery_start_time = time.time()
        try:
             self.file_list, self.cumulative_sizes, self.total_samples = self._discover_files_optimized_v2()
        except FileNotFoundError as e:
             self.logger.error(f"Failed to initialize dataset due to missing directory: {e}")
             raise # Re-raise after logging
        except Exception as e:
             self.logger.exception(f"Unexpected error during optimized file discovery.") # Log traceback
             raise # Re-raise other unexpected errors
        finally:
            # Clean up potential large intermediate sets if discovery fails midway
            gc.collect()
        discovery_time = time.time() - discovery_start_time
        self.logger.info(f"Optimized file discovery v2 took {discovery_time:.2f} seconds.")

        if self.total_samples == 0:
            # This is more likely an error condition if mandatory directories existed
            self.logger.error(f"No complete data samples found across mandatory features in {self.feature_cache_path}. Check if all features were generated correctly for all batches.")
            raise ValueError(f"No complete data samples found in feature cache: {self.feature_cache_path}")
        self.logger.info(f"Discovered {len(self.file_list)} verified batch file indices, total samples: {self.total_samples}")

        # Load bucket assignments (essential for BucketBatchSampler)
        # These load methods use the verified self.file_list
        loading_start_time = time.time()
        try:
            self._bucket_assignments = self._load_bucket_assignments()
            self._cluster_assignments = self._load_cluster_assignments()
        except Exception as e:
             self.logger.exception("Error during assignment loading.")
             raise
        finally:
            # Collect garbage after potentially loading large assignment tensors
            gc.collect()
        loading_time = time.time() - loading_start_time
        self.logger.info(f"Loading assignments took {loading_time:.2f} seconds.")

        # Feature cache (in-memory) - Use with caution for large datasets
        self.feature_cache = {} # Cache loaded batch files {('feature_type', file_idx): tensor}

    def _discover_files_optimized_v2(self):
        """
        Optimized discovery v2: Assumes reliable precomputation.
        Finds intersection of batch indices, assumes expected batch size for all
        except the last batch index, loads only the last 'dims' file for exact size.
        Requires 'precompute_batch_size' to be set in self.config.
        """
        self.logger.info("Starting optimized file discovery v2 (assuming reliable precomputation)...")
        batch_indices_per_feature = {}
        parse_errors = 0
        dims_tensor = None # Define variable for finally block

        try: # Added try block for potential cleanup
             # 1. Collect batch indices for each mandatory feature type (same as before)
             #    Uses os.scandir for efficiency.
             for feature_type in tqdm(self.MANDATORY_FEATURES, desc="Scanning features"):
                 info = self.FEATURE_INFO[feature_type]
                 feature_dir = self.feature_cache_path / info['dir']
                 feature_ext = info['ext']
                 current_indices = set()
                 try:
                     for entry in os.scandir(feature_dir):
                         if entry.is_file() and entry.name.endswith(feature_ext):
                             match = re.match(r"(\d+)_(\d+)" + re.escape(feature_ext), entry.name)
                             if match:
                                 try:
                                     batch_idx = int(match.group(2))
                                     current_indices.add(batch_idx)
                                 except ValueError: # pragma: no cover
                                     if parse_errors < 10: self.logger.warning(f"Could not parse batch index from filename: {entry.name} in {feature_dir}")
                                     parse_errors += 1
                 except FileNotFoundError: # pragma: no cover
                     self.logger.error(f"Directory not found during scan: {feature_dir}")
                     raise
                 except Exception as e: # pragma: no cover
                     self.logger.error(f"Error scanning directory {feature_dir}: {e}")

                 if not current_indices:
                     self.logger.error(f"No batch files found for mandatory feature '{feature_type}' in {feature_dir}. Dataset cannot be formed.")
                     raise FileNotFoundError(f"No files found for mandatory feature: {feature_type}")
                 batch_indices_per_feature[feature_type] = current_indices
                 del current_indices # Delete intermediate set

             # Log findings from scan
             if parse_errors > 0: self.logger.warning(f"Encountered {parse_errors} filename parsing errors during scan.")
             for ft, idx_set in batch_indices_per_feature.items():
                 self.logger.info(f"Found {len(idx_set)} batch indices for feature '{ft}'.")

             # 2. Find the intersection of batch indices (same as before)
             if not batch_indices_per_feature: # Should not happen if checks above are done
                  self.logger.error("No features scanned, cannot determine valid batches.")
                  return [], {}, 0 # pragma: no cover
             valid_batch_indices = set.intersection(*batch_indices_per_feature.values())
             self.logger.info(f"Found {len(valid_batch_indices)} batch indices present across ALL mandatory features.")
             del batch_indices_per_feature # Delete intermediate dictionary of sets
             gc.collect() # Collect after deleting potentially large structures

             if not valid_batch_indices:
                 self.logger.error("No common batch indices found across all mandatory features. Check precomputation outputs.")
                 return [], {}, 0

             # 3. Determine Batch Sizes (Optimized: Load only last batch dims)
             final_file_list = [] # List of tuples: (batch_idx, num_samples)
             dims_info = self.FEATURE_INFO.get('dims')
             if not dims_info: raise ValueError("Internal Error: 'dims' feature info missing.")

             sorted_indices = sorted(list(valid_batch_indices))
             del valid_batch_indices # Delete set after sorting
             max_batch_idx = sorted_indices[-1]
             last_batch_size = 0

             # Load ONLY the dims file for the maximum batch index found
             self.logger.info(f"Loading 'dims' file only for the last batch index: {max_batch_idx}...")
             dims_file_path = self._get_file_path('dims', max_batch_idx)
             if dims_file_path is None:
                  # This is critical - if the last batch dims file is missing, we can't determine size accurately.
                 self.logger.error(f"Could not find 'dims' file for the MAX validated batch index {max_batch_idx}. Cannot determine dataset size.")
                 raise FileNotFoundError(f"Missing dims file for max batch index: {max_batch_idx}")

             try:
                 dims_tensor = torch.load(dims_file_path, map_location='cpu')
                 if dims_tensor.ndim == 0 or dims_tensor.shape[0] == 0: # Still a minimal safety check
                     self.logger.error(f"Last dims file {dims_file_path} loaded empty/scalar tensor. Cannot determine dataset size.")
                     raise ValueError(f"Last dims file empty/scalar: {dims_file_path}")
                 last_batch_size = dims_tensor.shape[0]
                 self.logger.info(f"Last batch ({max_batch_idx}) has {last_batch_size} samples.")
                 # --- Explicitly delete the loaded tensor --- START EDIT ---
                 del dims_tensor
                 dims_tensor = None # Ensure it's None if deletion happens before assignment
                 # --- Explicitly delete the loaded tensor --- END EDIT ---

             except Exception as e:
                 self.logger.error(f"Failed to load/process the last dims file {dims_file_path}: {e}")
                 raise # Failure to load the last batch size is fatal for this method

             # Construct the file list assuming expected size for all but the last
             for batch_idx in sorted_indices:
                 size = self.expected_batch_size if batch_idx != max_batch_idx else last_batch_size
                 final_file_list.append((batch_idx, size))
             del sorted_indices # Delete sorted list

             # 4. Calculate cumulative sizes and total samples (same as before)
             cumulative_sizes = {}
             current_pos = 0
             final_total_samples = 0
             for batch_idx, num_samples in final_file_list: # final_file_list is already sorted
                 cumulative_sizes[batch_idx] = current_pos
                 current_pos += num_samples
             final_total_samples = current_pos

             self.logger.info(f"Successfully verified {len(final_file_list)} batches using optimized discovery v2.")
             return final_file_list, cumulative_sizes, final_total_samples

        finally:
             # Ensure intermediate large objects are cleaned up even if an error occurs
             del batch_indices_per_feature
             del valid_batch_indices
             del sorted_indices
             if dims_tensor is not None:
                 del dims_tensor
             gc.collect()

    def _get_file_path(self, feature_type: str, batch_idx: int):
         """
         Gets the path(s) for a given feature type and batch index.
         It returns the first available file found across potential ranks.
         """
         info = self.FEATURE_INFO.get(feature_type)
         if not info:
             self.logger.error(f"Feature info not found for type '{feature_type}'")
             return None

         feature_dir = self.feature_cache_path / info['dir']
         # Use the precise extension defined in FEATURE_INFO
         filename_pattern = f"*_{batch_idx:06d}{info['ext']}" # Assumes 6 digits padding from precompute
         
         # Use glob to find matching files
         potential_files = sorted(list(feature_dir.glob(filename_pattern)))
         
         if not potential_files:
              # Reduce log spam: Maybe log only once per feature type or less frequently
              # self.logger.debug(f"No file found for feature '{feature_type}', batch {batch_idx} with pattern {filename_pattern}")
              return None # Indicate file not found
              
         # Return the first found file path (e.g., rank 0's if available and exists)
         # Add an existence check for robustness, though glob should only return existing files
         if potential_files[0].exists():
            return potential_files[0]
         else:
            # This case is unlikely if glob works correctly, but handle defensively
            self.logger.warning(f"Glob found {potential_files[0]} but it does not exist. Trying next...")
            for pfile in potential_files[1:]:
                if pfile.exists():
                    return pfile
            # If none exist after globbing, return None
            self.logger.error(f"Glob found files for {feature_type} batch {batch_idx} but none actually exist.")
            return None


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

        # Define variables for finally block
        batch_data_cache = None
        batch_tensor = None
        sample_data = {}
        latents = None
        t5_embeddings = None
        img_ids = None
        txt_ids = None

        try: # Add try for finally block
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
            load_successful = True
            for feature_type, batch_tensor in batch_data_cache.items():
                 if batch_tensor is not None:
                     try:
                          sample_data[feature_type] = batch_tensor[index_in_batch]
                          # --- Delete reference to the full batch tensor after use --- START EDIT
                          # We only need the single sample's data now.
                          # Note: This assumes the cache (`self.feature_cache`) holds the main reference
                          # if needed again. If the cache wasn't used or got evicted,
                          # deleting here would mean reloading next time.
                          del batch_tensor
                          batch_tensor = None # Reset for next loop iteration or exit
                          # --- Delete reference to the full batch tensor after use --- END EDIT
                     except IndexError:
                          self.logger.error(f"IndexError accessing {feature_type} data: index_in_batch={index_in_batch}, batch_tensor shape={batch_tensor.shape if batch_tensor is not None else 'None'}, batch_idx={batch_idx}")
                          sample_data[feature_type] = None # Mark as failed
                          if feature_type in self.MANDATORY_FEATURES:
                               load_successful = False
                          # Delete tensor even if index error occurred
                          if batch_tensor is not None:
                               del batch_tensor
                               batch_tensor = None
                     except Exception as e:
                          self.logger.error(f"Error extracting sample {idx} (batch {batch_idx}, index {index_in_batch}) for feature {feature_type}: {e}")
                          sample_data[feature_type] = None
                          if feature_type in self.MANDATORY_FEATURES:
                               load_successful = False
                          # Delete tensor even if other error occurred
                          if batch_tensor is not None:
                               del batch_tensor
                               batch_tensor = None
                 else:
                     # Feature file failed to load or dir didn't exist
                     sample_data[feature_type] = None
                     # Check if this was a mandatory feature that failed
                     if feature_type in self.MANDATORY_FEATURES and (self.feature_cache_path / self.FEATURE_INFO[feature_type]['dir']).exists():
                         # If dir exists but load failed for mandatory feature
                         load_successful = False
                 # Ensure batch_tensor is None if it was None initially or after deletion
                 batch_tensor = None

            # --- Explicitly delete the batch data cache dict --- START EDIT ---
            del batch_data_cache
            batch_data_cache = None
            # --- Explicitly delete the batch data cache dict --- END EDIT ---


            # Add index and potentially other metadata if needed
            sample_data['index'] = idx

            # --- Generate img_ids and txt_ids ---
            latents = sample_data.get('latents')
            t5_embeddings = sample_data.get('t5')

            if latents is not None:
                latent_h, latent_w = latents.shape[-2], latents.shape[-1]
                grid_h, grid_w = math.ceil(latent_h / 2), math.ceil(latent_w / 2)
                num_img_patches = grid_h * grid_w
                img_ids = torch.zeros(grid_h, grid_w, 3, dtype=torch.float32)
                img_ids[..., 0] = torch.arange(grid_h, dtype=torch.float32)[:, None]
                img_ids[..., 1] = torch.arange(grid_w, dtype=torch.float32)[None, :]
                sample_data['img_ids'] = img_ids.view(num_img_patches, 3)
                # --- Delete intermediate tensors --- START EDIT ---
                del img_ids
                img_ids = None
                del latents # Delete reference to latents tensor after use
                latents = None
                # --- Delete intermediate tensors --- END EDIT ---
            else:
                self.logger.warning(f"Latents missing for sample {idx}, creating dummy img_ids.")
                sample_data['img_ids'] = torch.zeros(1, 3, dtype=torch.float32) # Minimal placeholder
                if 'latents' in self.MANDATORY_FEATURES: load_successful = False

            if t5_embeddings is not None:
                num_txt_tokens = t5_embeddings.shape[0]
                txt_ids = torch.zeros(num_txt_tokens, 3, dtype=torch.float32)
                txt_ids[..., 0] = torch.arange(num_txt_tokens, dtype=torch.float32)
                sample_data['txt_ids'] = txt_ids
                # --- Delete intermediate tensors --- START EDIT ---
                del txt_ids
                txt_ids = None
                del t5_embeddings # Delete reference to t5 tensor after use
                t5_embeddings = None
                # --- Delete intermediate tensors --- END EDIT ---
            else:
                self.logger.warning(f"T5 embeddings missing for sample {idx}, creating dummy txt_ids.")
                sample_data['txt_ids'] = torch.zeros(1, 3, dtype=torch.float32) # Minimal placeholder
                if 't5' in self.MANDATORY_FEATURES: load_successful = False

            # Handle mandatory feature load failures
            if not load_successful:
                 self.logger.error(f"Failed to load one or more mandatory features for sample index {idx}. Returning None or partial data.")
                 raise RuntimeError(f"Failed to load mandatory features for sample index {idx}. Check logs.")

            return sample_data

        finally:
            # --- Clean up local variables from __getitem__ --- START EDIT ---
            # Although Python's GC usually handles this, explicit deletion
            # can sometimes help, especially if complex objects or cycles exist,
            # or just for clarity that we don't need them anymore.
            del batch_data_cache
            del batch_tensor
            # sample_data is returned, so don't delete it here
            del latents
            del t5_embeddings
            del img_ids
            del txt_ids
            # gc.collect() here might be too frequent and slow down data loading.
            # Rely on Python's GC unless severe issues persist.
            # --- Clean up local variables from __getitem__ --- END EDIT ---

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
        """Loads bucket assignments for the entire dataset using pre-allocation."""
        assignments_dir = self.feature_cache_path / self.FEATURE_INFO['buckets']['dir']
        if not assignments_dir.exists(): # pragma: no cover
            raise FileNotFoundError(f"Bucket assignments directory not found: {assignments_dir}")

        if not self.file_list: # pragma: no cover
            self.logger.error("File list is empty, cannot load bucket assignments.")
            raise ValueError("Cannot load assignments, file list is empty.")
        if self.total_samples <= 0: # pragma: no cover
             self.logger.error(f"Total samples is {self.total_samples}, cannot pre-allocate tensor.")
             raise ValueError("Cannot pre-allocate assignments tensor, total samples is not positive.")


        self.logger.info(f"Loading bucket assignments for {len(self.file_list)} verified batches into pre-allocated tensor ({self.total_samples} samples)...")
        # --- Pre-allocate the full tensor --- START EDIT ---
        full_assignments = torch.empty(self.total_samples, dtype=torch.short, device='cpu')
        # --- Pre-allocate the full tensor --- END EDIT ---

        load_errors = 0
        batch_assignments = None # Define for finally block

        try: # Added try for finally block
            # Use self.file_list which contains verified batch indices and sizes
            for batch_idx, num_samples_expected in tqdm(self.file_list, desc="Loading bucket assignments"):
                file_path = self._get_file_path('buckets', batch_idx) # Use helper to find the file
                if file_path:
                    try:
                        batch_assignments = torch.load(file_path, map_location='cpu')
                        start_idx = self.cumulative_sizes[batch_idx]
                        end_idx = start_idx + batch_assignments.shape[0] # Use actual loaded shape

                        if batch_assignments.shape[0] != num_samples_expected:
                             self.logger.warning(f"Bucket assignment count mismatch for batch {batch_idx}. Expected {num_samples_expected}, got {batch_assignments.shape[0]}. File: {file_path}. Adjusting slice.")
                             end_idx = start_idx + batch_assignments.shape[0]
                             if end_idx > self.total_samples:
                                  self.logger.error(f"Slice end index {end_idx} exceeds total samples {self.total_samples} for batch {batch_idx}. Skipping assignment.")
                                  load_errors += 1
                                  # --- Delete tensor before continuing --- START EDIT
                                  del batch_assignments
                                  batch_assignments = None
                                  # --- Delete tensor before continuing --- END EDIT
                                  continue

                        if batch_assignments.dtype != torch.short:
                             batch_assignments = batch_assignments.short() # Ensure correct dtype

                        full_assignments[start_idx:end_idx] = batch_assignments
                        # --- Explicitly delete intermediate tensor --- START EDIT ---
                        del batch_assignments
                        batch_assignments = None # Reset for next iteration
                        # --- Explicitly delete intermediate tensor --- END EDIT ---

                    except Exception as e:
                        self.logger.error(f"Error loading or assigning bucket assignment file {file_path}: {e}")
                        load_errors += 1
                        # Clean up if tensor was loaded before error
                        if batch_assignments is not None:
                            del batch_assignments
                            batch_assignments = None
                else: # pragma: no cover
                    self.logger.error(f"Could not find bucket assignment file for batch {batch_idx}. Cannot proceed.")
                    raise FileNotFoundError(f"Missing bucket assignment file for batch {batch_idx}")

            if load_errors > 0: # pragma: no cover
                 self.logger.warning(f"Encountered {load_errors} errors during bucket assignment loading.")

            self.logger.info(f"Successfully loaded bucket assignments for {full_assignments.shape[0]} samples into pre-allocated tensor.")
            return full_assignments
        finally:
            # --- Ensure cleanup and garbage collect after loop --- START EDIT ---
            del batch_assignments # Delete final reference if loop exited early
            gc.collect()
            # --- Ensure cleanup and garbage collect after loop --- END EDIT ---


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

    def _load_cluster_assignments(self) -> torch.Tensor:
         """Loads cluster assignments for the entire dataset using pre-allocation."""
         assignments_dir = self.feature_cache_path / self.FEATURE_INFO['clusters']['dir']
         cluster_ext = self.FEATURE_INFO['clusters']['ext'] # Use correct extension
         if not assignments_dir.exists(): # pragma: no cover
             raise FileNotFoundError(f"Cluster assignments directory not found: {assignments_dir}")

         if not self.file_list: # pragma: no cover
            self.logger.error("File list is empty, cannot load cluster assignments.")
            raise ValueError("Cannot load assignments, file list is empty.")
         if self.total_samples <= 0: # pragma: no cover
             self.logger.error(f"Total samples is {self.total_samples}, cannot pre-allocate tensor.")
             raise ValueError("Cannot pre-allocate assignments tensor, total samples is not positive.")


         self.logger.info(f"Loading cluster assignments for {len(self.file_list)} verified batches into pre-allocated tensor ({self.total_samples} samples)...")
         # --- Pre-allocate the full tensor --- START EDIT ---
         full_assignments = torch.empty(self.total_samples, dtype=torch.long, device='cpu')
         # --- Pre-allocate the full tensor --- END EDIT ---

         load_errors = 0
         batch_assignments = None # Define for finally block

         try: # Added try for finally block
             for batch_idx, num_samples_expected in tqdm(self.file_list, desc="Loading cluster assignments"):
                 file_path = self._get_file_path('clusters', batch_idx)
                 if file_path:
                     try:
                         batch_assignments = torch.load(file_path, map_location='cpu')
                         start_idx = self.cumulative_sizes[batch_idx]
                         end_idx = start_idx + batch_assignments.shape[0] # Use actual loaded shape

                         if batch_assignments.shape[0] != num_samples_expected:
                              self.logger.warning(f"Cluster assignment count mismatch for batch {batch_idx}. Expected {num_samples_expected}, got {batch_assignments.shape[0]}. File: {file_path}. Adjusting slice.")
                              end_idx = start_idx + batch_assignments.shape[0]
                              if end_idx > self.total_samples:
                                   self.logger.error(f"Slice end index {end_idx} exceeds total samples {self.total_samples} for batch {batch_idx}. Skipping assignment.")
                                   load_errors += 1
                                   # --- Delete tensor before continuing --- START EDIT
                                   del batch_assignments
                                   batch_assignments = None
                                   # --- Delete tensor before continuing --- END EDIT
                                   continue

                         if batch_assignments.dtype != torch.long:
                              batch_assignments = batch_assignments.long() # Ensure correct dtype

                         full_assignments[start_idx:end_idx] = batch_assignments
                         # --- Explicitly delete intermediate tensor --- START EDIT ---
                         del batch_assignments
                         batch_assignments = None # Reset for next iteration
                         # --- Explicitly delete intermediate tensor --- END EDIT ---

                     except Exception as e:
                         self.logger.error(f"Error loading or assigning cluster assignment file {file_path}: {e}")
                         load_errors += 1
                         # Clean up if tensor was loaded before error
                         if batch_assignments is not None:
                             del batch_assignments
                             batch_assignments = None
                 else: # pragma: no cover
                     self.logger.error(f"Could not find cluster assignment file for batch {batch_idx} (expected ext: {cluster_ext}). Cannot proceed.")
                     raise FileNotFoundError(f"Missing cluster assignment file for batch {batch_idx}")

             if load_errors > 0: # pragma: no cover
                 self.logger.warning(f"Encountered {load_errors} errors during cluster assignment loading.")

             self.logger.info(f"Successfully loaded cluster assignments for {full_assignments.shape[0]} samples into pre-allocated tensor.")
             return full_assignments
         finally:
             # --- Ensure cleanup and garbage collect after loop --- START EDIT ---
             del batch_assignments # Delete final reference if loop exited early
             gc.collect()
             # --- Ensure cleanup and garbage collect after loop --- END EDIT ---

    @property
    def cluster_assignments(self) -> torch.Tensor:
        """Returns the loaded cluster assignments."""
        if self._cluster_assignments is None:
             self.logger.error("Cluster assignments accessed before loading.")
             raise ValueError("Cluster assignments not loaded.")
        return self._cluster_assignments


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
    Can optionally filter indices for a specific target expert ID.
    """
    def __init__(self, dataset: DDMDataset, batch_size: int, shuffle=True, drop_last=True, logger=None, target_expert_id: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.logger = logger if logger else logging.getLogger(__name__)
        self.target_expert_id = target_expert_id

        # Distributed training setup
        self.rank = 0
        self.world_size = 1
        if torch.distributed.is_initialized():
             self.rank = torch.distributed.get_rank()
             self.world_size = torch.distributed.get_world_size()

        # Get bucket assignments (this triggers loading if not already done)
        try:
            bucket_assignments = self.dataset.bucket_assignments
        except FileNotFoundError:
            self.logger.error("Bucket assignments not found. Cannot use BucketBatchSampler.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading bucket assignments: {e}")
            raise

        self.num_buckets = bucket_assignments.max().item() + 1
        self.logger.info(f"Sampler Rank {self.rank}: Found {self.num_buckets} buckets.")

        # Group indices by bucket
        self.indices_by_bucket = defaultdict(list)
        for idx, bucket_id in enumerate(bucket_assignments.tolist()):
             self.indices_by_bucket[bucket_id].append(idx)

        # Filter indices by target_expert_id if provided
        if self.target_expert_id is not None:
             self.logger.info(f"Sampler Rank {self.rank}: Filtering indices for target expert ID: {self.target_expert_id}")
             try:
                  cluster_assignments = self.dataset.cluster_assignments # Load cluster assignments
                  filtered_indices_by_bucket = defaultdict(list)
                  original_count = 0
                  filtered_count = 0
                  for bucket_id, indices in self.indices_by_bucket.items():
                       original_count += len(indices)
                       # Ensure cluster_assignments covers all indices
                       max_idx_in_bucket = max(indices) if indices else -1
                       if max_idx_in_bucket >= len(cluster_assignments):
                            self.logger.error(f"Index {max_idx_in_bucket} in bucket {bucket_id} is out of bounds for cluster assignments (size: {len(cluster_assignments)}). Cannot filter.")
                            # Decide how to handle: skip bucket, raise error? Let's skip for now.
                            continue # Skip this bucket if indices are invalid

                       # Filter indices based on cluster assignment
                       expert_indices = [
                            idx for idx in indices if cluster_assignments[idx].item() == self.target_expert_id
                       ]
                       filtered_indices_by_bucket[bucket_id] = expert_indices
                       filtered_count += len(expert_indices)
                  self.indices_by_bucket = filtered_indices_by_bucket # Overwrite with filtered indices
                  self.logger.info(f"Sampler Rank {self.rank}: Filtering complete. Kept {filtered_count}/{original_count} samples for expert {self.target_expert_id}.")
             except FileNotFoundError:
                  self.logger.error("Cluster assignments not found. Cannot filter for expert.")
                  raise
             except Exception as e:
                  self.logger.error(f"Error loading or filtering cluster assignments: {e}")
                  raise

        # Prepare batches for each bucket using the (potentially filtered) indices
        self.batches_by_bucket = defaultdict(list)
        self.num_batches = 0
        total_samples_this_rank = 0 # Track samples assigned to this rank
        for bucket_id in range(self.num_buckets):
            indices = self.indices_by_bucket[bucket_id] # Now uses filtered indices if applicable
            if not indices: # Skip empty buckets (especially after filtering)
                 continue

            if self.shuffle:
                random.shuffle(indices) # Shuffle within bucket

            bucket_batches = list(chunks(indices, self.batch_size))

            # Handle drop_last for the bucket
            if self.drop_last and len(indices) % self.batch_size != 0 and len(bucket_batches) > 0:
                bucket_batches = bucket_batches[:-1]

            # Distribute batches across ranks
            num_bucket_batches = len(bucket_batches)
            batches_for_this_rank = bucket_batches[self.rank : num_bucket_batches : self.world_size]

            self.batches_by_bucket[bucket_id] = batches_for_this_rank
            self.num_batches += len(batches_for_this_rank)
            samples_in_batches_this_rank = sum(len(b) for b in batches_for_this_rank)
            total_samples_this_rank += samples_in_batches_this_rank

            # Adjusted Logging
            if self.rank == 0: # Log bucket info from rank 0
                 log_expert_id = f" (Expert {self.target_expert_id})" if self.target_expert_id is not None else ""
                 self.logger.info(f"Bucket {bucket_id}{log_expert_id}: {len(indices)} samples -> {num_bucket_batches} total batches -> {len(batches_for_this_rank)} batches ({samples_in_batches_this_rank} samples) for rank 0.")

        # Create the final list of all batches for this rank for the epoch
        self.epoch_batches = []
        for bucket_id in range(self.num_buckets):
            self.epoch_batches.extend(self.batches_by_bucket[bucket_id])

        # Shuffle the order of batches across buckets for the epoch
        if self.shuffle:
            random.shuffle(self.epoch_batches)

        self.logger.info(f"Sampler Rank {self.rank}: Total batches: {self.num_batches}. Total samples assigned: {total_samples_this_rank}.")
        # Sanity check: if num_batches is 0 but total_samples > 0, batch_size might be too large or drop_last issue.
        if self.num_batches == 0 and total_samples_this_rank > 0:
             self.logger.warning(f"Sampler Rank {self.rank}: No batches generated, but {total_samples_this_rank} samples were assigned. Check batch_size ({self.batch_size}) vs samples per bucket and drop_last setting.")
        elif self.num_batches == 0 and total_samples_this_rank == 0:
             self.logger.warning(f"Sampler Rank {self.rank}: No batches generated and no samples assigned. This rank might not have data for this expert or dataset is empty.")

    def __iter__(self):
        # Re-generates batches for the epoch, shuffling within buckets if self.shuffle is True
        # Uses self.indices_by_bucket which might have been filtered in __init__
        self.epoch_batches = []
        for bucket_id in range(self.num_buckets):
            indices = self.indices_by_bucket[bucket_id] # Use potentially filtered indices
            if not indices: continue # Skip empty bucket

            if self.shuffle:
                random.shuffle(indices) # Reshuffle indices within bucket
            bucket_batches = list(chunks(indices, self.batch_size))
            if self.drop_last and len(indices) % self.batch_size != 0 and len(bucket_batches) > 0:
                bucket_batches = bucket_batches[:-1]

            # Distribute batches across ranks for this epoch
            num_bucket_batches = len(bucket_batches)
            batches_for_this_rank = bucket_batches[self.rank : num_bucket_batches : self.world_size]
            self.epoch_batches.extend(batches_for_this_rank)

        # Reshuffle the order of all batches for this rank
        if self.shuffle:
            random.shuffle(self.epoch_batches)

        # Log if the iterator is empty for this rank
        if not self.epoch_batches and self.rank == 0: # Only log warning once from rank 0
             self.logger.warning(f"Sampler Rank {self.rank} __iter__: No batches to yield for this epoch.")
        elif not self.epoch_batches and self.target_expert_id is not None:
             self.logger.warning(f"Sampler Rank {self.rank} __iter__: No batches to yield for expert {self.target_expert_id} on this rank.")

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
    device = torch.device('cpu') # Loaders primarily work with CPU data before transfer
    logger = setup_distributed_logger(name="ExpertLoaders", rank=rank) # Use the imported function

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
            logger=dataset.logger, # Pass dataset's logger
            target_expert_id=expert_idx
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
