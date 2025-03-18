"""Dataset classes for Decentralized Diffusion Models."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import logging
import time
import torchvision.transforms as transforms

import torch.distributed as dist

from utils.distributed import is_main_process, get_rank, broadcast_object, get_local_rank, get_world_size
from data.transforms import resize_image, normalize

import glob
import os
import multiprocessing as mp

# Setup logging
logger = logging.getLogger(__name__)

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
    """Dataset pipeline with GPU operations"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = torch.device(f'cuda:{get_local_rank()}')
        self._init_parameters()
        self._load_dataset()

    def _init_parameters(self):
        """Initialize parameters on GPU"""
        self.min_size = self.config.min_size
        self.num_experts = self.config.num_experts
        self.bucket_dims = torch.tensor(self.config.buckets, device=self.device)
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        self.caption_ext = '.txt'

    def _load_dataset(self):
        """Load dataset with minimized CPU-GPU transfers"""
        start_time = time.time()
        logger.info(f"Rank {self.rank}: Starting dataset loading...")
        
        # Initial file discovery must happen on CPU
        all_image_paths = []
        with mp.Pool(8) as pool:
            for ext in self.image_extensions:
                pattern = os.path.join(self.config.dataset_path, f'**/*{ext}')
                results = pool.apply(glob.glob, (pattern,))
                all_image_paths.extend(results)
        all_image_paths = list(set(all_image_paths))  # Deduplicate
        
        # Distribute paths across ranks
        num_paths = len(all_image_paths)
        paths_per_rank = num_paths // self.world_size
        start_idx = self.rank * paths_per_rank
        end_idx = start_idx + paths_per_rank if self.rank < self.world_size - 1 else num_paths
        my_paths = all_image_paths[start_idx:end_idx]
        
        logger.info(f"Rank {self.rank}: Validating {len(my_paths)} files...")
        
        # Initial validation still needs CPU for file operations
        valid_paths = []
        valid_dims = []
        
        # Process in batches to avoid overwhelming the CPU
        for path_batch in chunks(my_paths, 256):
            for img_path in path_batch:
                caption_path = os.path.splitext(img_path)[0] + self.caption_ext
                
                # Check if caption exists
                if not os.path.exists(caption_path):
                    continue
                
                # Get image dimensions - temporarily on CPU
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        
                    # Check minimum size
                    if width >= self.min_size and height >= self.min_size:
                        valid_paths.append(img_path)
                        valid_dims.append((width, height))
                except Exception as e:
                    continue
        
        # Move valid dimensions to GPU
        if valid_dims:
            self.dim_cache = torch.tensor(valid_dims, device=self.device)
        else:
            self.dim_cache = torch.empty((0, 2), device=self.device)
        
        # Gather all valid paths and dimensions across GPUs
        valid_count = torch.tensor([len(valid_paths)], device=self.device)
        all_counts = [torch.zeros_like(valid_count) for _ in range(self.world_size)]
        dist.all_gather(all_counts, valid_count)
        
        # Collect paths and dimensions
        self.image_files = valid_paths
        self.caption_files = [p.replace(os.path.splitext(p)[1], self.caption_ext) for p in valid_paths]
        
        logger.info(f"Rank {self.rank}: Found {len(valid_paths)} valid image-caption pairs")
        
        # Precompute bucket assignments on GPU
        self._process_buckets()
        
        # Distribute samples across experts
        self._distribute_samples()
        
        # Calculate dataset size
        total_images = sum(count.item() for count in all_counts)
        logger.info(f"Rank {self.rank}: Dataset loaded with {total_images} total images in {time.time() - start_time:.2f}s")

    def _process_buckets(self):
        """Process bucket assignments on GPU"""
        if len(self.dim_cache) == 0:
            self.bucket_assignments = torch.empty(0, dtype=torch.long, device=self.device)
            return
            
        # Calculate aspect ratios on GPU
        image_dims = self.dim_cache.float()
        image_ar = image_dims[:, 0] / image_dims[:, 1]
        bucket_ar = self.bucket_dims[:, 0].float() / self.bucket_dims[:, 1].float()
        
        # Find closest bucket using matrix ops on GPU
        ar_diff = torch.abs(image_ar.unsqueeze(1) - bucket_ar.unsqueeze(0))
        self.bucket_assignments = torch.argmin(ar_diff, dim=1)
        
        # Log bucket statistics without moving to CPU
        for bucket_idx in range(self.bucket_dims.size(0)):
            count = torch.sum(self.bucket_assignments == bucket_idx).item()
            if count > 0:
                bucket_size = tuple(self.bucket_dims[bucket_idx].cpu().numpy())
                logger.info(f"Rank {self.rank}: Bucket {bucket_idx} {bucket_size}: {count} images")

    def _distribute_samples(self):
        """Distribute samples across experts on GPU"""
        if len(self.image_files) == 0:
            self.expert_assignments = torch.empty(0, dtype=torch.long, device=self.device)
            return
            
        # Assign experts using modulo on GPU
        indices = torch.arange(len(self.image_files), device=self.device)
        self.expert_assignments = indices % self.num_experts
        
        # Log expert distribution
        for expert_idx in range(self.num_experts):
            count = torch.sum(self.expert_assignments == expert_idx).item()
            logger.info(f"Rank {self.rank}: Expert {expert_idx}: {count} images")

    def _gpu_group_by_bucket(self, indices):
        """Group indices by bucket on GPU"""
        bucket_indices = {}
        for bucket_idx in range(self.bucket_dims.size(0)):
            # Create mask on GPU
            mask = self.bucket_assignments[indices] == bucket_idx
            bucket_indices[bucket_idx] = indices[mask]
        return bucket_indices

    def __getitem__(self, idx):
        """Load data with minimal CPU-GPU transfers"""
        # Get image path and caption path
        img_path = self.image_files[idx]
        caption_path = self.caption_files[idx]
        bucket_idx = self.bucket_assignments[idx].item()
        expert_idx = self.expert_assignments[idx].item()
        
        # Get target bucket dimensions
        bucket_dims = self.bucket_dims[bucket_idx].cpu().numpy()
        
        # Load and process image
        try:
            # Load image (must be on CPU initially)
            with Image.open(img_path) as img:
                # Resize according to bucket dimensions
                img = resize_image(img, (int(bucket_dims[0]), int(bucket_dims[1])))
                
                # Convert to tensor and move to GPU
                img_tensor = transforms.ToTensor()(img).to(self.device)
                
                # Normalize on GPU
                img_tensor = normalize(img_tensor)
        except Exception as e:
            # Fallback to random tensor if image loading fails
            logger.warning(f"Failed to load image {img_path}: {e}")
            img_tensor = torch.rand(3, int(bucket_dims[1]), int(bucket_dims[0]), device=self.device)
        
        # Load caption text
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        except Exception as e:
            caption = ""
            
        # Create and return batch dict
        return {
            'image': img_tensor,
            'caption': caption,
            'expert': torch.tensor(expert_idx, device=self.device),
            'bucket': torch.tensor(bucket_idx, device=self.device)
        }

    def __len__(self):
        return len(self.image_files)

    def get_status_summary(self):
        """Generate status summary"""
        if not hasattr(self, 'image_files') or len(self.image_files) == 0:
            return {
                "status": "incomplete",
                "message": "Dataset processing has not completed or failed",
                "images_found": 0
            }
            
        # Count buckets and experts
        bucket_counts = {}
        for bucket_idx in range(self.bucket_dims.size(0)):
            count = torch.sum(self.bucket_assignments == bucket_idx).item()
            if count > 0:
                bucket_counts[bucket_idx] = count
                
        expert_counts = {}
        for expert_idx in range(self.num_experts):
            count = torch.sum(self.expert_assignments == expert_idx).item()
            if count > 0:
                expert_counts[expert_idx] = count
                
        # Images for this rank's expert
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

class BucketBatchSampler(torch.utils.data.Sampler):
    """GPU-optimized bucket batch sampler"""
    
    def __init__(self, bucket_indices, batch_size, shuffle=True, drop_last=True):
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = torch.device(f'cuda:{get_local_rank()}')

    def __iter__(self):
        batches = []
        
        # Process each bucket
        for bucket_idx, indices in self.bucket_indices.items():
            # Skip empty buckets
            if len(indices) == 0:
                continue
                
            # Shuffle indices on GPU if needed
            if self.shuffle:
                perm = torch.randperm(len(indices), device=self.device)
                shuffled_indices = indices[perm]
            else:
                shuffled_indices = indices
                
            # Create batches directly on GPU
            for i in range(0, len(shuffled_indices), self.batch_size):
                if i + self.batch_size <= len(shuffled_indices) or not self.drop_last:
                    end_idx = min(i + self.batch_size, len(shuffled_indices))
                    batches.append(shuffled_indices[i:end_idx])
        
        # Shuffle batches if needed
        if self.shuffle and batches:
            batch_idxs = torch.randperm(len(batches), device=self.device)
            batches = [batches[i] for i in batch_idxs]
            
        return iter(batches)
        
    def __len__(self):
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.bucket_indices.values())
        else:
            return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.bucket_indices.values())

def create_expert_bucket_loaders(dataset, config):
    """Create data loaders for experts with bucket sampling"""
    expert_loaders = {}
    
    # Find unique expert indices
    for expert_idx in range(dataset.num_experts):
        # Get indices for this expert
        expert_mask = dataset.expert_assignments == expert_idx
        if not torch.any(expert_mask):
            continue
            
        indices = torch.nonzero(expert_mask, as_tuple=True)[0]
        
        # Group by bucket
        bucket_indices = dataset._gpu_group_by_bucket(indices)
        
        # Create sampler
        sampler = BucketBatchSampler(
            bucket_indices=bucket_indices,
            batch_size=config.expert_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,  # No CPU workers to avoid transfers
            pin_memory=False  # Don't use pinned memory since we're staying on GPU
        )
        
        expert_loaders[expert_idx] = loader
    
    return expert_loaders