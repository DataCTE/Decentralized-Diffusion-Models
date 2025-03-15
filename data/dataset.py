"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
import logging

import glob

import io
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized utilities
from utils.distributed import is_main_process, get_rank, broadcast_object
from utils.logging import setup_logger, setup_distributed_logger
from data.transforms import resize_image, normalize
from utils.distributed import synchronize

# Setup logging
logger = logging.getLogger(__name__)

import math  # For BucketBatchSampler

class DataValidator:
    """GPU-accelerated image validation with distributed caching"""
    
    _valid_files_cache = {}
    _invalid_files_cache = set()
    _lock = threading.Lock()
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def validate_images(cls, root_dir, extensions=None, min_size=256, batch_size=1000):
        """Main entry point for image validation with batched processing"""
        extensions = extensions or ['.jpg', '.jpeg', '.png', '.webp']
        cache_key = (root_dir, tuple(extensions), min_size)
        
        with cls._lock:
            if cache_key in cls._valid_files_cache:
                return cls._valid_files_cache[cache_key]
            
            if not is_main_process():
                return cls._broadcast_result(cache_key)

            all_files = cls._discover_files(root_dir, extensions)
            valid_files = cls._process_batches(all_files, min_size, batch_size)
            
            cls._cache_results(cache_key, valid_files, len(all_files))
            return cls._broadcast_result(cache_key)
    
    @staticmethod
    def _discover_files(root_dir, extensions):
        """Find image files with GPU-optimized path discovery"""
        files = []
        for ext in extensions:
            # Use glob with case-insensitive pattern matching
            pattern = os.path.join(root_dir, f'*[{ext.lower()}{ext.upper()}]')
            files.extend(glob.glob(pattern, recursive=True))
        return sorted(files)

    @classmethod
    def _process_batches(cls, all_files, min_size, batch_size):
        """Process files with enhanced GPU progress tracking"""
        valid_files = []
        invalid_files = []
        pbar = None
        
        try:
            if is_main_process():
                pbar = tqdm(
                    total=len(all_files),
                    desc="Validating (GPU)",
                    unit="img",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    postfix={
                        'valid': 0,
                        'invalid': 0,
                        'rate': '0 img/s'
                    },
                    position=0
                )

            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
                futures = [executor.submit(cls._process_batch, batch, min_size)
                         for batch in chunks(all_files, batch_size)]
                
                for future in as_completed(futures):
                    batch_valid, batch_invalid = future.result()
                    valid_files.extend(batch_valid)
                    invalid_files.extend(batch_invalid)
                    cls._invalid_files_cache.update(batch_invalid)
                    
                    if pbar:
                        # Update metrics
                        pbar.update(len(batch_valid) + len(batch_invalid))
                        pbar.set_postfix({
                            'valid': len(valid_files),
                            'invalid': len(invalid_files),
                            'rate': f"{pbar.format_dict['rate']} img/s"
                        })
                        
        finally:
            if pbar:
                pbar.close()
                # Log final validation stats
                valid_pct = (len(valid_files)/len(all_files))*100
                logger.info(f"Validation complete: {len(valid_files)} valid ({valid_pct:.1f}%)"
                          f" | {len(invalid_files)} invalid")
                
        return valid_files

    @classmethod
    def _process_batch(cls, file_batch, min_size):
        """Pure GPU validation pipeline"""
        valid_files = []
        invalid_files = []
        
        # Convert min_size to GPU tensor
        min_size_tensor = torch.tensor(min_size, device=cls._device, dtype=torch.int32)
        
        try:
            # Batch load with GPU-only pipeline
            tensors = [cls._load_image_tensor(fpath).to(cls._device) for fpath in file_batch]
            
            # Vectorized GPU checks
            valid_mask = torch.stack([
                (t.shape[-2] >= min_size_tensor) &
                (t.shape[-1] >= min_size_tensor) &
                (t.min() >= 0) &
                (t.max() <= 1) &
                ~torch.isnan(t).any() &
                ~torch.isinf(t).any()
                for t in tensors
            ])
            
            # Filter using GPU mask
            valid_files = [f for f, m in zip(file_batch, valid_mask.cpu().numpy()) if m]
            invalid_files = [f for f, m in zip(file_batch, valid_mask.cpu().numpy()) if not m]
            
        except RuntimeError as e:  # Catch GPU memory errors
            logger.warning(f"GPU batch failed: {str(e)}")
            invalid_files.extend(file_batch)
            
        return valid_files, invalid_files

    @classmethod
    def _load_image_tensor(cls, fpath):
        """Load image directly to tensor with GPU optimization"""
        try:
            # Use PyTorch's GPU-optimized image loader
            with open(fpath, 'rb') as f:
                img = Image.open(io.BytesIO(f.read()))
                return transforms.ToTensor()(img).unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"GPU loading failed: {str(e)}")
        
    @classmethod
    def _gpu_integrity_check(cls, tensor):
        """GPU-accelerated image validation checks"""
        try:
            # Check for invalid values using GPU ops
            valid = torch.all(tensor >= 0) and torch.all(tensor <= 1)
            
            # Check for NaN/Inf values
            valid &= not torch.isnan(tensor).any()
            valid &= not torch.isinf(tensor).any()
            
            # Check minimum size in tensor form
            _, _, h, w = tensor.shape
            valid &= h >= cls.config.min_size and w >= cls.config.min_size
            
            return valid.item()
        except Exception as e:
            return False

    @classmethod
    def _cache_results(cls, cache_key, valid_files, total_files):
        """Update caches and log results"""
        with cls._lock:
            cls._valid_files_cache[cache_key] = valid_files
            logger.info(f"Validation complete: {len(valid_files)}/{total_files} valid images")

    @classmethod
    def _broadcast_result(cls, cache_key):
        """Handle distributed synchronization"""
        result = broadcast_object(cls._valid_files_cache.get(cache_key, []), src=0)
        with cls._lock:
            cls._valid_files_cache.setdefault(cache_key, result)
        return result

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
        
        # Validate required parameters
        if not hasattr(config, 'min_size'):
            raise ValueError("Configuration missing required 'min_size' parameter")
        if not hasattr(config, 'num_experts'):
            raise ValueError("Configuration missing required 'num_experts' parameter")

        # Then initialize GPU device reference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Initializing dataset on {self.device}")
        
        # Convert config values to GPU tensors with proper validation
        self.min_size = torch.tensor(
            getattr(config, 'min_size', 256),  # Default fallback
            device=self.device
        )
        self.num_experts = torch.tensor(
            getattr(config, 'num_experts', 8),  # Default fallback
            device=self.device
        )
        
        # Load dataset with GPU-accelerated validation
        self._load_dataset()
        self._init_gpu_buckets()
        self._distribute_samples_gpu()

        # Store bucket sizes as GPU tensors
        self.bucket_dims = torch.tensor(
            config.buckets, 
            device=self.device,
            dtype=torch.int32
        )

    def _load_dataset(self):
        """GPU-accelerated dataset loading pipeline"""
        if not self.config.dataset_path:
            raise ValueError("Dataset path must be provided")
            
        # Use GPU-optimized file discovery
        self.image_files = DataValidator.validate_images(
            self.config.dataset_path,
            min_size=self.config.min_size,
            batch_size=self.config.validation_batch_size
        )

        # Move metadata to GPU
        self._load_metadata_gpu()
        self.logger.info(f"Loaded {len(self.image_files)} images on {self.device}")

    def _load_metadata_gpu(self):
        """GPU metadata loading with progress tracking"""
        self.dim_cache = torch.zeros((len(self.image_files), 2), 
                                   dtype=torch.int32, 
                                   device=self.device)
        
        pbar = None
        try:
            if is_main_process():
                pbar = tqdm(
                    total=len(self.image_files),
                    desc="Caching Metadata",
                    unit="img",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    position=1  # Below validation bar
                )

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(self._cache_image_dims, i)
                         for i in range(len(self.image_files))]
                
                for future in as_completed(futures):
                    _ = future.result()
                    if pbar:
                        pbar.update(1)
                        
        finally:
            if pbar:
                pbar.close()

    def _cache_image_dims(self, idx):
        """Cache image dimensions with GPU fallback"""
        try:
            with Image.open(self.image_files[idx]) as img:
                return idx, (img.width, img.height)
        except:
            return idx, (0, 0)

    def _init_gpu_buckets(self):
        """GPU-accelerated bucket initialization"""
        # Convert buckets to GPU tensor
        self.buckets = torch.tensor(self.config.buckets, 
                                  device=self.device,
                                  dtype=torch.float32)
        
        # Calculate aspect ratios on GPU
        bucket_aspects = self.buckets[:,0] / self.buckets[:,1]
        image_aspects = self.dim_cache[:,0] / self.dim_cache[:,1]

        # Find closest bucket using GPU matrix ops
        diffs = torch.abs(image_aspects.unsqueeze(1) - bucket_aspects)
        self.bucket_assignments = torch.argmin(diffs, dim=1)
        
        self.logger.info(f"GPU bucket assignment completed: {self.buckets.shape[0]} buckets")

    def _distribute_samples_gpu(self):
        """GPU-accelerated expert distribution"""
        # Create expert assignments using modulo on GPU
        indices = torch.arange(len(self.image_files), device=self.device)
        self.expert_assignments = indices % self.num_experts
        
        self.logger.info(f"GPU expert distribution completed: {self.num_experts} experts")

    def __getitem__(self, idx):
        """Add periodic loading progress tracking"""
        if not hasattr(self, "_loader_pbar"):
            if is_main_process() and self.split == 'train':
                self._loader_pbar = tqdm(
                    total=len(self),
                    desc="Loading Batches",
                    unit="batch",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    position=2  # Below other bars
                )
            else:
                self._loader_pbar = None

        # Get target size directly from GPU tensor
        bucket_idx = self.bucket_assignments[idx]
        target_h = self.bucket_dims[bucket_idx, 1]
        target_w = self.bucket_dims[bucket_idx, 0]
        
        # Keep all tensors on GPU
        try:
            tensor = self._load_image_tensor(idx, (target_w, target_h))
            return {
                'image': tensor,
                'expert': self.expert_assignments[idx],
                'bucket': bucket_idx
            }
        except Exception as e:
            return self._handle_error_case((target_w, target_h))

    def _load_image_tensor(self, idx, target_size):
        """Pure GPU image pipeline"""
        # Direct GPU load with tensor-based resizing
        tensor = DataValidator._load_image_tensor(self.image_files[idx]).to(self.device)
        
        # GPU-native size check
        if tensor.shape[-2:] != torch.Size(target_size):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=tuple(target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
        return normalize(tensor)

    def _handle_error_case(self, target_size):
        """Generate error placeholder on GPU"""
        return {
            'image': torch.zeros((3, *target_size), device=self.device),
            'expert': torch.tensor(-1, device=self.device),
            'bucket': torch.tensor(-1, device=self.device)
        }
    
    def __len__(self):
        """Get dataset length"""
        return len(self.image_files)
    
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
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    logger = setup_distributed_logger(name="ExpertLoaders", rank=rank)
    
    # Get expert assignments directly from GPU tensor
    expert_assignments = dataset.expert_assignments.cpu().numpy()
    expert_indices = defaultdict(list)
    
    # Use vectorized operations for expert index collection
    for idx in np.nditer(np.where(expert_assignments >= 0)):
        expert_idx = expert_assignments[idx]
        expert_indices[expert_idx].append(idx.item())

    expert_loaders = {}
    for expert_idx, indices in expert_indices.items():
        # Create GPU-optimized bucket indices
        bucket_indices = defaultdict(list)
        for idx in indices:
            bucket_idx = dataset.bucket_assignments[idx].item()
            bucket_indices[bucket_idx].append(idx)
        
        # Create GPU-accelerated sampler
        sampler = BucketBatchSampler(
            bucket_indices=bucket_indices,
            batch_size=config.expert_batch_size,
            device=device,
            shuffle=True,
            drop_last=True
        )
        
        # Configure loader with GPU optimizations
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=config.prefetch_factor if hasattr(config, 'prefetch_factor') else 2,
            multiprocessing_context='spawn' if config.num_workers > 0 else None,
            generator=torch.Generator(device=device)
        )
        
        # Warmup GPU pipeline
        if torch.cuda.is_available():
            for _ in loader:
                break
        
        expert_loaders[expert_idx] = loader
        logger.info(f"Created GPU-optimized loader for expert {expert_idx} "
                   f"with {len(sampler)} batches on {device}")
        
    return expert_loaders 