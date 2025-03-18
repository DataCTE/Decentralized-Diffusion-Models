"""Dataset classes for Decentralized Diffusion Models."""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import logging
import time  # Add missing time module import
import torchvision.transforms as transforms

import torch.distributed as dist


from utils.distributed import is_main_process, get_rank, broadcast_object, get_local_rank, get_world_size
from data.transforms import resize_image, normalize

import glob
import os


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
    """Pickle-safe dataset pipeline with CPU-only operations"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.device = torch.device(f'cuda:{get_local_rank()}')
        self._init_parameters()
        self._load_dataset()

    def _init_parameters(self):
        """Initialize parameters on GPU"""
        self.min_size = torch.tensor(self.config.min_size, device=self.device)
        self.num_experts = torch.tensor(self.config.num_experts, device=self.device)
        self.bucket_dims = torch.tensor(self.config.buckets, device=self.device)
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        self.caption_ext = '.txt'

    def _load_dataset(self):
        """GPU-accelerated dataset loading without custom ops"""
        # Discover files using parallel glob
        all_image_paths = []
        for ext in self.image_extensions:
            all_image_paths.extend(
                glob.glob(os.path.join(self.config.dataset_path, f'**/*{ext}'), recursive=True)
            )

        # Split paths across GPUs
        world_size = get_world_size()
        rank = get_rank()
        chunks = np.array_split(all_image_paths, world_size)
        my_paths = chunks[rank].tolist()

        # Parallel validation with GPU acceleration
        valid_paths, dims = self._gpu_validate_paths(my_paths)

        # Gather results from all GPUs
        all_paths = [None] * world_size
        all_dims = [None] * world_size
        dist.all_gather_object(all_paths, valid_paths)
        dist.all_gather_object(all_dims, dims)

        # Combine results
        self.image_files = [p for sublist in all_paths for p in sublist]
        self.dim_cache = torch.cat([d.to(self.device) for d in all_dims])
        self.caption_files = [p.replace(ext, self.caption_ext) 
                            for p, ext in zip(self.image_files, self.image_extensions)]

        # Process buckets and experts
        self._process_buckets_gpu()
        self._distribute_samples_gpu()

    def _gpu_validate_paths(self, paths):
        """Validate paths using mixed CPU/GPU processing"""
        valid_paths = []
        valid_dims = []
        
        for path in paths:
            caption_path = os.path.splitext(path)[0] + self.caption_ext
            if not os.path.exists(caption_path):
                continue
            
            try:
                # Load image dimensions using PIL
                with Image.open(path) as img:
                    width, height = img.size
                    if width >= self.config.min_size and height >= self.config.min_size:
                        valid_paths.append(path)
                        valid_dims.append([width, height])
            except Exception:
                continue

        # Convert to GPU tensor
        dim_tensor = torch.tensor(valid_dims, device=self.device) if valid_dims else torch.empty((0, 2), device=self.device)
        return valid_paths, dim_tensor

    def _process_buckets_gpu(self):
        """GPU-based bucket processing"""
        image_dims = self.dim_cache.float()
        image_ar = image_dims[:, 0] / image_dims[:, 1]
        bucket_ar = self.bucket_dims[:, 0] / self.bucket_dims[:, 1]
        
        self.bucket_assignments = torch.argmin(
            torch.abs(image_ar.unsqueeze(1) - bucket_ar.unsqueeze(0)), 
            dim=1
        )

    def _distribute_samples_gpu(self):
        """GPU-based expert distribution"""
        indices = torch.arange(len(self.image_files), device=self.device)
        self.expert_assignments = indices % self.num_experts

    def __getitem__(self, idx):
        """GPU-accelerated data loading"""
        # Load image to GPU directly
        img_path = self.image_files[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            target_size = tuple(self.bucket_dims[self.bucket_assignments[idx]].cpu().numpy())
            img_tensor = transforms.functional.to_tensor(img).to(self.device)
            img_tensor = transforms.functional.resize(img_tensor, target_size[::-1])
        
        # Load caption
        with open(self.caption_files[idx], 'r') as f:
            caption = f.read().strip()
            
        return {
            'image': img_tensor,
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

    def _find_valid_pairs(self, *args, **kwargs):
        raise NotImplementedError("Replaced by GPU validation")
        
    def _load_image_tensor(self, *args, **kwargs):
        raise NotImplementedError("Replaced by GPU mmap")
            
    def _load_caption(self, *args, **kwargs):
        raise NotImplementedError("Replaced by GPU text loading")

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
    """Optimized for GPU tensor operations"""
    
    def __init__(self, bucket_indices, batch_size, shuffle=True, drop_last=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # GPU-native index storage
        self.bucket_tensors = {
            bucket: torch.tensor(indices, device=self.device)
            for bucket, indices in bucket_indices.items()
        }

    def __iter__(self):
        batches = []
        for bucket, indices in self.bucket_tensors.items():
            if self.shuffle:
                indices = indices[torch.randperm(len(indices), device=self.device)]
            
            bucket_batches = torch.split(indices, self.batch_size)
            if self.drop_last and len(indices) % self.batch_size != 0:
                bucket_batches = bucket_batches[:-1]
                
            batches.extend(bucket_batches)
        
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), device=self.device)]
            
        return iter(batches)

def create_expert_bucket_loaders(dataset, config, world_size=1, rank=0):
    """GPU-native data loader creation"""
    expert_loaders = {}
    
    # Get expert assignments directly from GPU
    expert_assignments = dataset.expert_assignments.cpu().unique().tolist()
    
    for expert_idx in expert_assignments:
        expert_mask = dataset.expert_assignments == expert_idx
        indices = torch.where(expert_mask)[0]
        
        # GPU-optimized sampler
        sampler = BucketBatchSampler(
            bucket_indices=dataset._gpu_group_by_bucket(indices),
            batch_size=config.expert_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # GPU-direct DataLoader
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,  # No CPU workers
            pin_memory=False,  # Never use pinned memory
            persistent_workers=False,
            generator=torch.Generator(device='cuda')
        )
        
        expert_loaders[expert_idx] = loader
    
    return expert_loaders 