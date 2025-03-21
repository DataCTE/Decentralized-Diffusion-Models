"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from PIL import Image
import random
from collections import defaultdict
import logging
import time  
import glob

import io
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized utilities
from utils.distributed import is_main_process, broadcast_object, get_rank, get_local_rank, get_world_size
from utils.logging import setup_distributed_logger
from data.transforms import resize_image, normalize


# Setup logging
logger = logging.getLogger(__name__)

import math  # For BucketBatchSampler

# Add global cache to avoid duplicate validation
_GLOBAL_DATASET_CACHE = {
    "initialized": False,
    "image_files": [],
    "caption_files": [],
    "dim_cache": None,
}

def chunks(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class DDMDataset(Dataset):
    """GPU-optimized dataset pipeline for decentralized diffusion models"""
    
    def __init__(self, config, split='train', transforms=None, hf_split=None):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.split = split
        
        # Validate required parameters
        if not hasattr(config, 'min_size'):
            raise ValueError("Configuration missing required 'min_size' parameter")
        if not hasattr(config, 'num_experts'):
            raise ValueError("Configuration missing required 'num_experts' parameter")

        # Get the local rank for proper device assignment
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        
        # Always use CPU for dataset processing and communication
        self.device = torch.device('cpu')
        self.logger.info(f"Rank {self.rank}: Using CPU for dataset processing and distributed operations")
        
        # Define file extensions
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        self.caption_ext = '.txt'
        
        # Convert config values to tensors with proper validation
        self.min_size = torch.tensor(
            getattr(config, 'min_size', 256),  # Default fallback
            device=self.device
        )
        self.num_experts = torch.tensor(
            getattr(config, 'num_experts', 8),  # Default fallback
            device=self.device
        )
        
        # Store bucket sizes as tensors
        self.bucket_dims = torch.tensor(
            config.buckets, 
            device=self.device,
            dtype=torch.int32
        )
        
        # Load precomputed features and clusters (paper section 4.1)
        feature_path = os.path.join(config.feature_cache_path, "features")
        cluster_path = os.path.join(config.feature_cache_path, "clusters")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Features not found at {feature_path}. Run feature extraction first.")
        
        
        cluster_files = sorted([f for f in os.listdir(cluster_path) if f.endswith(".cluster.pt")])

        if not cluster_files:
            self.logger.warning("No cluster files found. Running clustering automatically...")
            raise FileNotFoundError(
                f"No cluster files found at {cluster_path}. "
                "Please run `run_clustering.py` first to generate cluster assignments."
            )

        # No longer load all cluster assignments into CPU memory
        # Cluster assignments will be loaded on-demand in _load_cluster_assignment

        self.cluster_path = cluster_path  # Store cluster path for later loading
        self.cluster_files = cluster_files  # Store cluster file names
        self.num_cluster_files = len(cluster_files)  # Store number of cluster files
        self.cluster_assignments_cache = {}  # Initialize a cache for loaded cluster assignments
        self.feature_path = feature_path # Store feature path
        self.feature_files = sorted([f for f in os.listdir(feature_path) if f.endswith(".pt")]) # Store feature file names
        self.num_feature_files = len(self.feature_files) # Store number of feature files
        self.features_cache = {} # Initialize cache for features

        self._discover_and_process_files()
        self._init_buckets()
        self._distribute_samples()

    def __getstate__(self):
        """Control what gets pickled to ensure we don't include unpicklable objects"""
        state = self.__dict__.copy()
        # Remove logger as it can't be pickled properly
        state['logger'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling"""
        self.__dict__.update(state)
        # Restore logger
        self.logger = logging.getLogger(__name__)

    def _discover_and_process_files(self):
        """Find valid image-caption pairs in a single efficient pass"""
        global _GLOBAL_DATASET_CACHE
        
        # If cache is already initialized by another dataset instance, use it
        if _GLOBAL_DATASET_CACHE["initialized"] and is_main_process():
            self.logger.info(f"Rank {self.rank}: Using cached dataset with {len(_GLOBAL_DATASET_CACHE['image_files'])} files")
            self.image_files = _GLOBAL_DATASET_CACHE["image_files"]
            self.caption_files = _GLOBAL_DATASET_CACHE["caption_files"]
            self.dim_cache = _GLOBAL_DATASET_CACHE["dim_cache"]
            
            # If we're using the validation split, we'll need to filter later
            return
        elif not is_main_process() and _GLOBAL_DATASET_CACHE["initialized"]:
            # Non-main processes also use the cache if it exists
            self.image_files = _GLOBAL_DATASET_CACHE["image_files"]
            self.caption_files = _GLOBAL_DATASET_CACHE["caption_files"]
            self.dim_cache = _GLOBAL_DATASET_CACHE["dim_cache"]
            return
        
        if not self.config.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        # Process files synchronously across nodes
        if not is_main_process():
            # Initialize storage
            self.image_files = []
            self.caption_files = []
            all_dims = []
            
            # First receive total number of images
            try:
                # Use CPU-only operations for distributed communication
                total_images = broadcast_object(None, src=0)
                self.logger.info(f"Rank {self.rank}: Expecting to receive {total_images} images from rank 0")
                
                # Process in smaller batches to avoid timeouts
                batch_size = 50  # Smaller batch size to prevent timeouts and memory issues
                num_batches = (total_images + batch_size - 1) // batch_size
                
                for i in range(num_batches):
                    self.logger.info(f"Rank {self.rank}: Receiving batch {i+1}/{num_batches}")
                    try:
                        batch_data = broadcast_object(None, src=0)
                        
                        # Check if we received a valid batch or a signal to end
                        if batch_data is None:
                            self.logger.warning(f"Rank {self.rank}: Received None batch, skipping")
                            continue
                            
                        if batch_data.get('done', False):
                            self.logger.info(f"Rank {self.rank}: Received end signal after {len(self.image_files)} images")
                            break
                        
                        batch_images = batch_data.get('images', [])
                        batch_captions = batch_data.get('captions', [])
                        batch_dims = batch_data.get('dimensions', [])
                        
                        self.image_files.extend(batch_images)
                        self.caption_files.extend(batch_captions)
                        all_dims.extend(batch_dims)
                        
                        # Simple acknowledgment without distributed communication
                        self.logger.info(f"Rank {self.rank}: Received batch {i+1}/{num_batches} with {len(batch_images)} images (total: {len(self.image_files)})")
                    except Exception as e:
                        self.logger.error(f"Rank {self.rank}: Error receiving batch {i+1}: {str(e)}")
                
                # Convert dimensions to tensor
                self.dim_cache = torch.tensor(all_dims, device=self.device, dtype=torch.int32)
                self.logger.info(f"Rank {self.rank}: Successfully received {len(self.image_files)} valid image-caption pairs")
                
                # Update the global cache
                _GLOBAL_DATASET_CACHE["image_files"] = self.image_files
                _GLOBAL_DATASET_CACHE["caption_files"] = self.caption_files
                _GLOBAL_DATASET_CACHE["dim_cache"] = self.dim_cache
                _GLOBAL_DATASET_CACHE["initialized"] = True
                
            except Exception as e:
                self.logger.error(f"Rank {self.rank}: Critical error during file reception: {str(e)}")
                raise
            return
        
        self.logger.info(f"Rank {self.rank}: Finding image-caption pairs in {self.config.dataset_path}")
        
        try:
            # Initialize storage for valid filesdebug
            valid_files = []
            caption_files = []
            valid_dims = []
            
            # Discover all image files more efficiently
            all_images = []
            for ext in self.image_extensions:
                pattern = os.path.join(self.config.dataset_path, f'**/*{ext}')
                all_images.extend(glob.glob(pattern, recursive=True))
            
            # Limit dataset size for testing if needed
            max_files = getattr(self.config, 'max_files', len(all_images))
            if max_files < len(all_images):
                all_images = all_images[:max_files]
                self.logger.info(f"Rank {self.rank}: Limiting to {max_files} files for testing")
            
            # Add progress bar for file discovery
            pbar_discover = tqdm(
                all_images,
                desc=f"Rank {self.rank}: Discovering files",
                unit="file",
                dynamic_ncols=True
            )
            
            # First broadcast total number of images we'll process
            total_images = len(all_images)
            self.logger.info(f"Rank {self.rank}: Broadcasting total count of {total_images} images")
            broadcast_object(total_images, src=0)
            
            # Process files with progress tracking
            pbar = tqdm(
                total=len(all_images),
                desc="Finding Valid Pairs",
                unit="pair",
                dynamic_ncols=True
            )
            
            # Use extremely small batch sizes for broadcasting to prevent timeouts
            broadcast_batch_size = 50  # Very small size to ensure reliable transmission
            
            # Process images first to find all valid ones
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
                futures = []
                # Use smaller processing batch size too
                process_batch_size = 500
                
                for i in range(0, len(pbar_discover), process_batch_size): # Iterate over pbar
                    batch = list(pbar_discover)[i:min(i + process_batch_size, len(pbar_discover))] # Get batch from pbar
                    futures.append(executor.submit(self._find_valid_pairs, batch))
                
                # Collect results 
                for future in as_completed(futures):
                    try:
                        batch_images, batch_captions, batch_dims = future.result()
                        valid_files.extend(batch_images)
                        caption_files.extend(batch_captions)
                        valid_dims.extend(batch_dims)
                        pbar.update(process_batch_size)
                    except Exception as e:
                        self.logger.error(f"Rank {self.rank}: Error processing image batch: {str(e)}")
            
            pbar.close()
            
            # Now broadcast the valid files in small batches to prevent timeouts
            self.logger.info(f"Rank {self.rank}: Broadcasting {len(valid_files)} valid files in batches of {broadcast_batch_size}")
            
            # Add progress tracking for broadcasting phase
            broadcast_start = time.time()
            broadcast_pbar = tqdm(
                total=len(valid_files),
                desc="Broadcasting Data",
                unit="file",
                dynamic_ncols=True
            )
            
            # Broadcast in small chunks
            for i in range(0, len(valid_files), broadcast_batch_size):
                end_idx = min(i + broadcast_batch_size, len(valid_files))
                batch_images = valid_files[i:end_idx]
                batch_captions = caption_files[i:end_idx]
                batch_dims = valid_dims[i:end_idx]
                
                batch_data = {
                    'images': batch_images,
                    'captions': batch_captions,
                    'dimensions': batch_dims,
                    'done': False
                }
                
                try:
                    batch_num = i//broadcast_batch_size + 1
                    total_batches = (len(valid_files) + broadcast_batch_size - 1) // broadcast_batch_size
                    self.logger.info(f"Rank {self.rank}: Broadcasting batch {batch_num}/{total_batches} ({end_idx-i} files, {i} processed so far)")
                    broadcast_object(batch_data, src=0)
                    time.sleep(0.1)  # Short sleep to prevent overwhelming receivers
                    broadcast_pbar.update(len(batch_images))
                except Exception as e:
                    self.logger.error(f"Rank {self.rank}: Error broadcasting batch {i//broadcast_batch_size + 1}: {str(e)}")
                    # Try again with a smaller batch if this fails
                    if broadcast_batch_size > 10:
                        smaller_batch_size = max(10, broadcast_batch_size // 2)
                        self.logger.info(f"Rank {self.rank}: Reducing batch size to {smaller_batch_size} and retrying")
                        # Adjust i to retry this batch with smaller size
                        i -= broadcast_batch_size
                        broadcast_batch_size = smaller_batch_size
                        continue
            
            broadcast_pbar.close()
            broadcast_time = time.time() - broadcast_start
            self.logger.info(f"Rank {self.rank}: Broadcasting complete in {broadcast_time:.2f}s")
            
            # Send final "done" signal
            self.logger.info(f"Rank {self.rank}: Sending completion signal after {len(valid_files)} valid pairs")
            broadcast_object({'done': True}, src=0)
            
            # Store results
            self.image_files = valid_files
            self.caption_files = caption_files
            self.dim_cache = torch.tensor(valid_dims, device=self.device, dtype=torch.int32)
            
            # Update the global cache
            _GLOBAL_DATASET_CACHE["image_files"] = self.image_files
            _GLOBAL_DATASET_CACHE["caption_files"] = self.caption_files
            _GLOBAL_DATASET_CACHE["dim_cache"] = self.dim_cache
            _GLOBAL_DATASET_CACHE["initialized"] = True
            
            # Log summary
            self.logger.info(f"Rank {self.rank}: Completed with {len(valid_files)} valid image-caption pairs")
        except Exception as e:
            self.logger.error(f"Rank {self.rank}: Critical error in main process: {str(e)}")
            # Try to send error signal to other processes
            try:
                broadcast_object({'error': True, 'message': str(e)}, src=0)
            except:
                pass
            raise
        
    def _find_valid_pairs(self, image_files):
        """Find valid image-caption pairs from a batch of image files"""
        valid_images = []
        valid_captions = []
        valid_dims = []
        min_size = self.min_size.item()
        
        for img_path in image_files:
            # Check if matching caption exists first - skip early if no caption
            caption_path = os.path.splitext(img_path)[0] + self.caption_ext
            if not os.path.exists(caption_path):
                continue
                
            # Only validate images that have captions
            try:
                # Use faster image opening method - just get dimensions without loading full pixel data
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width >= min_size and height >= min_size:
                        valid_images.append(img_path)
                        valid_captions.append(caption_path)
                        valid_dims.append([width, height])
            except Exception:
                # Skip any images that can't be opened
                continue
                
        return valid_images, valid_captions, valid_dims
        
    def _load_image_tensor(self, idx, target_size):
        """Load image efficiently for training - GPU usage only for final tensor"""
        with open(self.image_files[idx], 'rb') as f:
            img = Image.open(io.BytesIO(f.read())).convert('RGB')
            if target_size:
                img = img.resize(target_size, Image.BILINEAR)
                
            # Process on CPU then only transfer final tensor to GPU if needed
            tensor = transforms.ToTensor()(img)
            normalized = normalize(tensor)
            
            # Return the CPU tensor - moving to device will happen in the model forward pass instead
            return normalized
            
    def _load_caption(self, idx):
        """Load caption for the given image index"""
        with open(self.caption_files[idx], 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _init_buckets(self):
        """CPU-based bucket initialization"""
        # Before bucket assignment, handle train/val split if using cache
        if self.split == 'val' and _GLOBAL_DATASET_CACHE["initialized"]:
            # If this is the validation dataset and we're using cached data, 
            # select only a subset for validation
            val_size = getattr(self.config, 'val_size', 1000)
            if val_size < len(self.image_files):
                # Use deterministic selection to ensure consistency
                all_indices = np.arange(len(self.image_files))
                np.random.seed(42)  # Fixed seed for reproducibility
                val_indices = np.random.choice(all_indices, size=val_size, replace=False)
                
                # Filter files for validation
                self.image_files = [self.image_files[i] for i in val_indices]
                self.caption_files = [self.caption_files[i] for i in val_indices]
                self.dim_cache = self.dim_cache[val_indices]
                self.logger.info(f"Rank {self.rank}: Selected {len(self.image_files)} files for validation split")
        elif self.split == 'train' and _GLOBAL_DATASET_CACHE["initialized"]:
            # For training, exclude validation samples if specified
            val_size = getattr(self.config, 'val_size', 1000)
            if val_size > 0 and val_size < len(self.image_files):
                # Use same deterministic selection as above
                all_indices = np.arange(len(self.image_files))
                np.random.seed(42)  # Fixed seed for reproducibility
                val_indices = np.random.choice(all_indices, size=val_size, replace=False)
                train_indices = np.setdiff1d(all_indices, val_indices)
                
                # Filter files for training
                self.image_files = [self.image_files[i] for i in train_indices]
                self.caption_files = [self.caption_files[i] for i in train_indices]
                self.dim_cache = self.dim_cache[train_indices]
                self.logger.info(f"Rank {self.rank}: Selected {len(self.image_files)} files for training split (excluded {val_size} validation files)")
        
        self.logger.info(f"Rank {self.rank}: Starting bucket assignment for {len(self.image_files)} images...")
        bucket_start = time.time()
        
        # Calculate aspect ratios
        bucket_aspects = self.bucket_dims[:,0] / self.bucket_dims[:,1]
        print(f"Shape of self.dim_cache: {self.dim_cache.shape}")
        image_aspects = self.dim_cache[:,0] / self.dim_cache[:,1]

        # Add progress bar for bucket assignment
        pbar_bucket_assign = tqdm(
            range(len(image_aspects)),
            desc=f"Rank {self.rank}: Assigning buckets",
            unit="image",
            dynamic_ncols=True
        )

        # Find closest bucket using matrix ops
        diffs = torch.abs(image_aspects.unsqueeze(1) - bucket_aspects)
        self.bucket_assignments = torch.argmin(diffs, dim=1)

        pbar_bucket_assign.update(len(image_aspects)) # Complete progress bar
        pbar_bucket_assign.close()
        
        # Count images per bucket for logging
        bucket_counts = {}
        for i in range(self.bucket_dims.shape[0]):
            count = torch.sum(self.bucket_assignments == i).item()
            if count > 0:
                bucket_counts[i] = count
        
        bucket_time = time.time() - bucket_start
        self.logger.info(f"Rank {self.rank}: Bucket assignment completed in {bucket_time:.2f}s - {len(bucket_counts)} buckets used")
        
        # Log distribution stats
        top_buckets = sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for bucket_idx, count in top_buckets:
            bucket_size = tuple(self.bucket_dims[bucket_idx].tolist())
            self.logger.info(f"Rank {self.rank}: Bucket {bucket_idx} ({bucket_size}): {count} images")

    def _distribute_samples(self):
        """CPU-based expert distribution using paper's clustering"""
        self.logger.info(f"Rank {self.rank}: Starting expert assignment for {len(self.image_files)} images...")
        expert_start = time.time()

        # No clustering is performed here anymore, assuming pre-computed clusters are loaded
        # We are only loading pre-computed cluster assignments, so no clustering here
        pass # No clustering here, assuming pre-computed clusters are loaded

        # Store expert assignments from loaded cluster files
        # We are no longer running clustering in dataset.py, so expert_assignments are not set here.
        # Expert assignments are loaded on-demand in __getitem__ and _load_cluster_assignment
        # self.expert_assignments = cluster_assignments.to(self.device) - Removed

        # Validate cluster distribution - this part is no longer relevant as we are not clustering here
        # unique_clusters = torch.unique(cluster_assignments)
        # if unique_clusters.max() >= self.config.num_experts:
        #     raise ValueError(f"Cluster IDs exceed number of experts ({self.config.num_experts})")
        
        # Count images per expert for logging - this part is no longer relevant here
        # expert_counts = {}
        # for i in range(self.num_experts.item()):
        #     count = torch.sum(self.expert_assignments == i).item()
        #     expert_counts[i] = count
        
        expert_time = time.time() - expert_start
        self.logger.info(f"Rank {self.rank}: Expert distribution completed in {expert_time:.2f}s") # Keep this log message, but it's misleading now

        # Log distribution for this rank - this part is no longer relevant here
        # this_rank_count = torch.sum(self.expert_assignments == self.rank).item()
        # self.logger.info(f"Rank {self.rank}: Will process {this_rank_count} images ({this_rank_count/len(self.image_files)*100:.1f}% of dataset)")
        
        # Log total info - keep this
        dataset_prep_complete = time.time()
        self.logger.info(f"Rank {self.rank}: Dataset preparation complete - ready to start training")

        # Paper-mandated cluster validation - this part is no longer relevant here
        # cluster_sizes = [(self.expert_assignments == i).sum().item()
        #                 for i in range(self.config.num_experts)]
        # min_cluster_size = min_cluster_sizesize
        # max_cluster_size = max(cluster_sizes)

        # if getattr(self.config, 'bypass_cluster_validation', False):
        #     self.logger.info("Bypassing cluster size validation as requested by config flag")
        # elif min_cluster_size < 1000:  # Paper's minimum cluster size
        #     raise ValueError(
        #         f"Cluster too small (min_size={min_cluster_size}). "
        #         "Consider reducing num_experts or increasing dataset size."
        #     )

        # size_ratio = max_cluster_size / max_cluster_size
        # if size_ratio > 10:  # Paper's max imbalance
        #     self.logger.warning(
        #         f"Cluster size imbalance {size_ratio:.1f}x exceeds "
        #         "paper's recommended 10x limit"
        #     )

    def __getitem__(self, idx):
        """Get a training sample with image and caption"""
        # Get target size from bucket assignment
        bucket_idx = self.bucket_assignments[idx]
        target_h = self.bucket_dims[bucket_idx, 1]
        target_w = self.bucket_dims[bucket_idx, 0]
        
        # Load cluster assignment for this index
        cluster_assignment = self._load_cluster_assignment(idx)

        # Load feature for this index
        features = self._load_feature(idx)

        # Load image and caption
        tensor = self._load_image_tensor(idx, (target_w, target_h))
        caption = self._load_caption(idx)
        
        return {
            'image': tensor,
            'caption': caption,
            'expert': cluster_assignment, # Use loaded cluster assignment
            'features': features, # Use loaded features
            'bucket': bucket_idx
        }

    def _load_cluster_assignment(self, index):
        """Load cluster assignment for a given index from file, using cache"""
        file_index = index // (len(self.image_files) // self.num_cluster_files) # Calculate file index
        cluster_file_name = self.cluster_files[file_index]
        
        if cluster_file_name not in self.cluster_assignments_cache: # Check cache
            cluster_file_path = os.path.join(self.cluster_path, cluster_file_name)
            self.cluster_assignments_cache[cluster_file_name] = torch.load(cluster_file_path, map_location='cpu') # Load and cache

        file_assignments = self.cluster_assignments_cache[cluster_file_name]
        index_within_file = index % (len(self.image_files) // self.num_cluster_files) # Calculate index within file
        return file_assignments[index_within_file] # Return assignment for index within file

    def _load_feature(self, index):
        """Load feature for a given index from file, using cache"""
        file_index = index // (len(self.features) // self.num_feature_files) # Calculate file index
        feature_file_name = self.feature_files[file_index]

        if feature_file_name not in self.features_cache: # Check cache
            feature_file_path = os.path.join(self.feature_path, feature_file_name)
            self.features_cache[feature_file_name] = torch.load(feature_file_path, map_location='cpu') # Load and cache

        file_features = self.features_cache[feature_file_name]
        index_within_file = index % (len(self.features) // self.num_feature_files) # Calculate index within file
        return file_features[index_within_file] # Return feature for index within file

    def __len__(self):
        """Get dataset length"""
        return len(self.image_files)
        
    def get_status_summary(self):
        """Generate a user-friendly status summary of dataset processing"""
        if not hasattr(self, 'image_files') or len(self.image_files) == 0:
            return {
                "status": "incomplete",
                "message": "Dataset processing has not completed or failed",
                "images_found": 0
            }
            
        # Count buckets actually used
        bucket_counts = {}
        if hasattr(self, 'bucket_assignments'):
            for i in range(self.bucket_dims.shape[0]):
                count = torch.sum(self.bucket_assignments == i).item()
                if count > 0:
                    bucket_counts[i] = count
        
        # Count images per expert
        expert_counts = {}
        if hasattr(self, 'expert_assignments'):
            for i in range(self.num_experts.item()):
                count = torch.sum(self.expert_assignments == i).item()
                if count > 0:
                    expert_counts[i] = count
        
        # Images for this rank
        this_rank_count = 0
        if hasattr(self, 'expert_assignments'):
            this_rank_count = torch.sum(self.expert_assignments == self.rank).item()
        
        return {
            "status": "complete",
            "rank": self.rank,
            "total_images": len(self.image_files),
            "total_buckets": len(bucket_counts),
            "total_experts": len(expert_counts),
            "images_for_this_rank": this_rank_count,
            "percent_for_this_rank": f"{this_rank_count/len(self.image_files)*100:.1f}%",
            "top_buckets": sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "expert_distribution": expert_counts
        }

    def _default_transform(self, img, bucket_idx=None):
        """Apply default transformations based on bucket dimensions"""
        # Get target dimensions from bucket if provided
        if bucket_idx is not None and 0 <= bucket_idx < len(self.buckets):
            width, height = self.buckets[bucket_idx]
        else:
            # Fallback to default image size
            _, height, width = self.config.image_size
        
        # Resize image to target dimensions
        if isinstance(img, Image.Image):
            # PIL Image
            img = resize_image(img, (width, height))
            img = transforms.ToTensor()(img)
        elif isinstance(img, torch.Tensor):
            # Already a tensor, resize with torch functions
            if img.shape[-2] != height or img.shape[-1] != width:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), 
                    size=(height, width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
        
        # Normalize
        img = normalize(img)
        return img
        
    def _create_bucket_samplers(self):
        """Create bucket-specific samplers to ensure consistent shapes in each batch"""
        self.bucket_samplers = {}
        for bucket_idx, _ in enumerate(self.bucket_dims):
            # Get indices of samples in this bucket
            bucket_indices = [i for i, sample in enumerate(self.samples) 
                             if sample.get('bucket_idx', 0) == bucket_idx]
            
            if bucket_indices:
                self.bucket_samplers[bucket_idx] = SubsetRandomSampler(bucket_indices)
        
        logger.info(f"Created {len(self.bucket_samplers)} bucket-specific samplers")

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
    """GPU-optimized bucket batch sampler with tensor-based operations"""
    
    def __init__(self, bucket_indices, batch_size, device, shuffle=True, drop_last=True):
        """
        Args:
            bucket_indices: Dictionary of {bucket_idx: list of indices}
            batch_size: Target batch size
            device: Target device for tensor operations
            shuffle: Whether to shuffle batches
            drop_last: Whether to drop last incomplete batch
        """
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Convert indices to GPU tensors
        self.bucket_tensors = {
            bucket: torch.tensor(indices, device=device, dtype=torch.long)
            for bucket, indices in bucket_indices.items()
        }
        
        # Precompute batch counts using GPU ops
        self.batch_counts = torch.zeros(len(bucket_indices), device=device, dtype=torch.long)
        for i, (bucket, indices) in enumerate(bucket_indices.items()):
            count = len(indices) // batch_size if drop_last else math.ceil(len(indices) / batch_size)
            self.batch_counts[i] = count
            
        self.total_batches = torch.sum(self.batch_counts).item()
        
    def __iter__(self):
        # Generate batches using GPU-accelerated operations
        all_batches = []
        
        for bucket_idx, indices in self.bucket_tensors.items():
            # Shuffle on GPU if needed
            if self.shuffle:
                indices = indices[torch.randperm(len(indices), device=self.device)]
            
            # Split into batches using tensor operations
            batches = torch.split(indices, self.batch_size)
            
            if self.drop_last and len(indices) % self.batch_size != 0:
                batches = batches[:-1]
                
            all_batches.extend(batches)
        
        # Shuffle across buckets if needed
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
