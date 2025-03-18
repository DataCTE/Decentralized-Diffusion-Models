"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
import logging
import time  # Add missing time module import
import glob
import io
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import torch.distributed as dist
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.distributed import is_main_process, get_rank, broadcast_object, get_local_rank, get_world_size
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
    """Pickle-safe dataset pipeline with CPU-only operations"""
    
    def __init__(self, config, split='train'):
        # Simplified initialization without any distributed references
        self.config = config
        self.split = split
        self._init_parameters()
        self._load_dataset()

    def _init_parameters(self):
        """Initialize all parameters as basic Python types"""
        self.min_size = self.config.min_size
        self.num_experts = self.config.num_experts
        self.bucket_dims = [tuple(b) for b in self.config.buckets]
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        self.caption_ext = '.txt'
        self.device = 'cpu'

    def _load_dataset(self):
        """Multi-threaded dataset loading with integrated validation"""
        self.image_files = []
        self.caption_files = []
        self.dim_cache = []
        invalid_entries = []
        
        if is_main_process():
            # First: Discover all potential files quickly
            all_image_paths = []
            for ext in self.image_extensions:
                all_image_paths.extend(
                    glob.glob(os.path.join(self.config.dataset_path, f'**/*{ext}'), recursive=True)
                )
            
            # Configure threading (hardcoded values)
            num_workers = 16  # Default 16 workers
            chunk_size = 50  # Fixed chunk size of 50 files per thread
            
            # Shared progress counter
            manager = multiprocessing.Manager()
            counter = manager.Value('i', 0)
            lock = manager.Lock()
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                pbar = tqdm(total=len(all_image_paths), desc="Validating files")
                
                # Split work into chunks
                for chunk in chunks(all_image_paths, chunk_size):
                    futures.append(
                        executor.submit(
                            self._process_file_chunk,
                            chunk,
                            counter,
                            lock
                        )
                    )
                
                # Collect results as they complete
                for future in as_completed(futures):
                    chunk_images, chunk_captions, chunk_dims, chunk_invalid = future.result()
                    self.image_files.extend(chunk_images)
                    self.caption_files.extend(chunk_captions)
                    self.dim_cache.extend(chunk_dims)
                    invalid_entries.extend(chunk_invalid)
                    pbar.update(chunk_size)
                
                pbar.close()

            logger.info(f"Found {len(self.image_files)} valid pairs, skipped {len(invalid_entries)} invalid files")

        # Distributed synchronization
        if dist.is_initialized():
            self._distributed_sync()

        if len(self.image_files) == 0:
            raise RuntimeError("No valid training samples found in dataset directory")

    def _process_file_chunk(self, file_paths, counter, lock):
        """Process a chunk of files in a thread-safe manner"""
        chunk_images = []
        chunk_captions = []
        chunk_dims = []
        chunk_invalid = []
        
        for img_path in file_paths:
            caption_path = os.path.splitext(img_path)[0] + self.caption_ext
            
            try:
                if not os.path.exists(caption_path):
                    continue
                
                with Image.open(img_path) as img:
                    # Quick format check
                    img.getdata()[0]  # Force partial decode
                    width, height = img.size
                    
                    if width >= self.min_size and height >= self.min_size:
                        chunk_images.append(img_path)
                        chunk_captions.append(caption_path)
                        chunk_dims.append((width, height))
                        
                        # Update progress safely
                        with lock:
                            counter.value += 1
            
            except (IOError, OSError, Image.DecompressionBombError, Image.UnidentifiedImageError):
                chunk_invalid.append(img_path)
            except Exception:
                chunk_invalid.append(img_path)
        
        return chunk_images, chunk_captions, chunk_dims, chunk_invalid

    def _distributed_sync(self):
        """Efficient distributed synchronization of validated files"""
        if is_main_process():
            # Broadcast compressed dataset info
            data_to_sync = {
                'image_files': self.image_files,
                'caption_files': self.caption_files,
                'dim_cache': np.array(self.dim_cache)
            }
            broadcast_object(data_to_sync, src=0)
        else:
            # Receive validated dataset from main
            synced_data = broadcast_object(None, src=0)
            self.image_files = synced_data['image_files']
            self.caption_files = synced_data['caption_files']
            self.dim_cache = synced_data['dim_cache'].tolist()
        
        logger.info(f"Rank {get_rank()} received {len(self.image_files)} validated files")

    def _process_buckets(self):
        """CPU-based bucket processing with numpy"""
        # Convert to numpy arrays for pickle safety
        self.dim_cache = np.array(self.dim_cache, dtype=np.int32)
        bucket_dims = np.array(self.bucket_dims, dtype=np.int32)
        
        # Calculate aspect ratios
        image_ar = self.dim_cache[:,0] / self.dim_cache[:,1]
        bucket_ar = bucket_dims[:,0] / bucket_dims[:,1]
        
        # Find closest bucket for each image
        self.bucket_assignments = np.argmin(
            np.abs(image_ar[:,None] - bucket_ar[None,:]),
            axis=1
        )
        
        # Expert distribution
        self.expert_assignments = np.arange(len(self.image_files)) % self.num_experts

    def __getitem__(self, idx):
        """Load data with on-the-fly tensor conversion"""
        img_path = self.image_files[idx]
        caption_path = self.caption_files[idx]
        
        # Load image
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            target_size = self.bucket_dims[self.bucket_assignments[idx]]
            img = img.resize(target_size)
            tensor = transforms.ToTensor()(img)
        
        # Load caption
        with open(caption_path, 'r') as f:
            caption = f.read().strip()
            
        return {
            'image': tensor,
            'caption': caption,
            'expert': self.expert_assignments[idx],
            'bucket': self.bucket_assignments[idx]
        }

    def __len__(self):
        return len(self.image_files)

    def __getstate__(self):
        return {
            'config': self.config,
            'split': self.split,
            'image_files': self.image_files,
            'caption_files': self.caption_files,
            'dim_cache': self.dim_cache,
            'bucket_assignments': self.bucket_assignments,
            'expert_assignments': self.expert_assignments
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_parameters()
        # No need to reprocess buckets as assignments are stored

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
                logger.info(f"Rank {self.rank}: Selected {len(self.image_files)} files for validation split")
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
                logger.info(f"Rank {self.rank}: Selected {len(self.image_files)} files for training split (excluded {val_size} validation files)")
        
        logger.info(f"Rank {self.rank}: Starting bucket assignment for {len(self.image_files)} images...")
        bucket_start = time.time()
        
        # Calculate aspect ratios
        bucket_aspects = self.bucket_dims[:,0] / self.bucket_dims[:,1]
        image_aspects = self.dim_cache[:,0] / self.dim_cache[:,1]

        # Find closest bucket using matrix ops
        diffs = torch.abs(image_aspects.unsqueeze(1) - bucket_aspects)
        self.bucket_assignments = torch.argmin(diffs, dim=1)
        
        # Count images per bucket for logging
        bucket_counts = {}
        for i in range(self.bucket_dims.shape[0]):
            count = torch.sum(self.bucket_assignments == i).item()
            if count > 0:
                bucket_counts[i] = count
        
        bucket_time = time.time() - bucket_start
        logger.info(f"Rank {self.rank}: Bucket assignment completed in {bucket_time:.2f}s - {len(bucket_counts)} buckets used")
        
        # Log distribution stats
        top_buckets = sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for bucket_idx, count in top_buckets:
            bucket_size = tuple(self.bucket_dims[bucket_idx])
            logger.info(f"Rank {self.rank}: Bucket {bucket_idx} ({bucket_size}): {count} images")

    def _distribute_samples(self):
        """CPU-based expert distribution"""
        logger.info(f"Rank {self.rank}: Starting expert assignment for {len(self.image_files)} images...")
        expert_start = time.time()
        
        # Create expert assignments using modulo
        indices = torch.arange(len(self.image_files), device=self.device)
        self.expert_assignments = indices % self.num_experts
        
        # Count images per expert for logging
        expert_counts = {}
        for i in range(self.num_experts.item()):
            count = torch.sum(self.expert_assignments == i).item()
            expert_counts[i] = count
            
        expert_time = time.time() - expert_start
        logger.info(f"Rank {self.rank}: Expert distribution completed in {expert_time:.2f}s")
        
        # Log distribution for this rank
        this_rank_count = torch.sum(self.expert_assignments == self.rank).item()
        logger.info(f"Rank {self.rank}: Will process {this_rank_count} images ({this_rank_count/len(self.image_files)*100:.1f}% of dataset)")
        
        # Log total info
        logger.info(f"Rank {self.rank}: Dataset preparation complete - ready to start training")

    def get_status_summary(self):
        """Generate a user-friendly status summary of dataset processing"""
        if not hasattr(self, 'image_files') or len(self.image_files) == 0:
            logger.warning("Dataset status requested before initialization")
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
        
    def get_invalid_files(self):
        """Retrieve list of files that failed validation"""
        return getattr(self, '_invalid_files', [])

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
            num_workers=config.num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=config.prefetch_factor if hasattr(config, 'prefetch_factor') else 2,
            multiprocessing_context='spawn' if config.num_workers > 0 else None,
            generator=torch.Generator(device='cpu')  # Always use CPU generator for consistency
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