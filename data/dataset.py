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

from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized utilities
from utils.distributed import is_main_process, broadcast_object
from utils.logging import setup_distributed_logger
from data.transforms import resize_image, normalize


# Setup logging
logger = logging.getLogger(__name__)

import math  # For BucketBatchSampler



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

        # Then initialize GPU device reference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Initializing dataset on {self.device}")
        
        # Define file extensions
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        self.caption_ext = '.txt'
        
        # Convert config values to GPU tensors with proper validation
        self.min_size = torch.tensor(
            getattr(config, 'min_size', 256),  # Default fallback
            device=self.device
        )
        self.num_experts = torch.tensor(
            getattr(config, 'num_experts', 8),  # Default fallback
            device=self.device
        )
        
        # Store bucket sizes as GPU tensors
        self.bucket_dims = torch.tensor(
            config.buckets, 
            device=self.device,
            dtype=torch.int32
        )
        
        # Load dataset with direct file discovery (no separate validation pass)
        self._discover_and_process_files()
        self._init_gpu_buckets()
        self._distribute_samples_gpu()

    def _discover_and_process_files(self):
        """Find valid image-caption pairs in a single efficient pass"""
        if not self.config.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        # Process files synchronously across nodes
        if not is_main_process():
            # Initialize storage
            self.image_files = []
            self.caption_files = []
            all_dims = []
            
            # First receive total number of images
            total_images = broadcast_object(None, src=0)
            self.logger.info(f"Expecting to receive {total_images} images from rank 0")
            
            # Process in smaller batches to avoid timeouts
            batch_size = 5000  # Smaller batch size to prevent timeouts
            num_batches = (total_images + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                self.logger.info(f"Receiving batch {i+1}/{num_batches}")
                batch_data = broadcast_object(None, src=0)
                
                # Check if we received a valid batch or a signal to end
                if batch_data is None or batch_data.get('done', False):
                    self.logger.info(f"Received end signal after {len(self.image_files)} images")
                    break
                
                self.image_files.extend(batch_data['images'])
                self.caption_files.extend(batch_data['captions'])
                all_dims.extend(batch_data['dimensions'])
                
                # Periodically log progress
                if (i+1) % 5 == 0 or (i+1) == num_batches:
                    self.logger.info(f"Received {len(self.image_files)}/{total_images} images so far")
            
            # Convert dimensions to tensor
            self.dim_cache = torch.tensor(all_dims, device=self.device, dtype=torch.int32)
            self.logger.info(f"Received {len(self.image_files)} valid image-caption pairs")
            return
        
        self.logger.info(f"Finding image-caption pairs in {self.config.dataset_path}")
        
        # Initialize storage for valid files
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
            self.logger.info(f"Limiting to {max_files} files for testing")
        
        # First broadcast total number of images to process
        total_images = len(all_images)
        broadcast_object(total_images, src=0)
        
        # Process files with progress tracking
        pbar = tqdm(
            total=len(all_images),
            desc="Finding Valid Pairs",
            unit="pair",
            dynamic_ncols=True
        )
        
        # Process in smaller batches to prevent timeouts
        batch_size = 5000  # Smaller batch size to improve responsiveness
        
        # Process in parallel but broadcast results in smaller chunks
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
            # Process images in chunks and broadcast after each chunk
            for chunk_idx, image_chunk in enumerate(chunks(all_images, batch_size)):
                self.logger.info(f"Processing chunk {chunk_idx+1}/{(len(all_images) + batch_size - 1) // batch_size}")
                
                chunk_valid_files = []
                chunk_caption_files = []
                chunk_valid_dims = []
                
                # Process current batch of images
                futures = []
                for i in range(0, len(image_chunk), 1000):  # Process in sub-batches for parallel processing
                    batch = image_chunk[i:min(i + 1000, len(image_chunk))]
                    futures.append(executor.submit(self._find_valid_pairs, batch))
                
                # Collect results from this batch
                for future in as_completed(futures):
                    batch_images, batch_captions, batch_dims = future.result()
                    chunk_valid_files.extend(batch_images)
                    chunk_caption_files.extend(batch_captions)
                    chunk_valid_dims.extend(batch_dims)
                    pbar.update(len(batch))
                
                # Add to our total collection
                valid_files.extend(chunk_valid_files)
                caption_files.extend(chunk_caption_files)
                valid_dims.extend(chunk_valid_dims)
                
                # Broadcast this chunk to all other processes
                self.logger.info(f"Broadcasting chunk {chunk_idx+1} with {len(chunk_valid_files)} valid pairs")
                batch_data = {
                    'images': chunk_valid_files,
                    'captions': chunk_caption_files,
                    'dimensions': chunk_valid_dims,
                    'done': False
                }
                broadcast_object(batch_data, src=0)
        
        pbar.close()
        
        # Broadcast final done signal
        self.logger.info(f"Sending completion signal after {len(valid_files)} total valid pairs")
        broadcast_object({'done': True}, src=0)
        
        # Store results
        self.image_files = valid_files
        self.caption_files = caption_files
        self.dim_cache = torch.tensor(valid_dims, device=self.device, dtype=torch.int32)
        
        # Log summary
        self.logger.info(f"Found {len(valid_files)} valid image-caption pairs")
    
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
        """Load image efficiently for training"""
        with open(self.image_files[idx], 'rb') as f:
            img = Image.open(io.BytesIO(f.read())).convert('RGB')
            if target_size:
                img = img.resize(target_size, Image.BILINEAR)
            tensor = transforms.ToTensor()(img).to(self.device)
            return normalize(tensor)
            
    def _load_caption(self, idx):
        """Load caption for the given image index"""
        with open(self.caption_files[idx], 'r', encoding='utf-8') as f:
            return f.read().strip()

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
        """Get a training sample with image and caption"""
        # Get target size from bucket assignment
        bucket_idx = self.bucket_assignments[idx]
        target_h = self.bucket_dims[bucket_idx, 1]
        target_w = self.bucket_dims[bucket_idx, 0]
        
        # Load image and caption
        tensor = self._load_image_tensor(idx, (target_w, target_h))
        caption = self._load_caption(idx)
        
        return {
            'image': tensor,
            'caption': caption,
            'expert': self.expert_assignments[idx],
            'bucket': bucket_idx
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