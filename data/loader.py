"""Data loading utilities for Decentralized Diffusion Models."""


import logging
import os
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor


# Import centralized utilities
from utils.distributed import get_rank, get_world_size 
from data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

def create_loader(dataset, config, is_train=True, distributed=False, rank=0, world_size=1):
    """
    Optimized data loader with paper-recommended defaults and distributed enhancements
    """
    # Extract core parameters with safe defaults
    batch_size = getattr(config, 'batch_size', 1)
    num_workers = getattr(config, 'num_workers', max(1, os.cpu_count()//2))
    pin_memory = getattr(config, 'pin_memory', True)
    persistent_workers = getattr(config, 'persistent_workers', num_workers > 0)
    prefetch_factor = getattr(config, 'prefetch_factor', 2)
    
    # Paper-recommended settings for DDM
    if is_train:
        num_workers = min(num_workers, 8)  # Prevent over-subscription
        drop_last = True
        shuffle = not distributed  # Let DistributedSampler handle shuffling
    else:
        num_workers = min(num_workers, 4)  # Validation needs less workers
        drop_last = False
        shuffle = False

    # Distributed sampler logic
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling

    # Optimized collation function
    def fast_collate(batch):
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        elif isinstance(batch[0], dict):
            return {key: fast_collate([d[key] for d in batch]) for key in batch[0]}
        else:
            return torch.utils.data.default_collate(batch)

    # Configure loader with optimized settings
    loader_config = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'sampler': sampler,
        'drop_last': drop_last,
        'collate_fn': fast_collate
    }

    # Handle different dataset types
    try:
        # Hugging Face dataset optimization
        if hasattr(dataset, '_indices') and hasattr(dataset, '_formatting_type'):
            loader_config['batch_size'] = None
            loader_config['sampler'] = None
            loader_config['shuffle'] = False
            loader_config['collate_fn'] = None
            
        loader = DataLoader(**loader_config)
        
        # Warmup loader (reduces first batch latency)
        if num_workers > 0:
            for _ in loader:
                break
            
        return loader
    except Exception as e:
        logger.error(f"Loader creation failed: {str(e)}")
        raise
    
    # Additional optimizations for file-based datasets
    if isinstance(dataset, DDMDataset):
        # Pre-fetch first batch during initialization
        with ThreadPoolExecutor(max_workers=1) as executor:
            _ = executor.submit(lambda: next(iter(loader)))
            
    return loader

def create_expert_bucket_loaders(dataset, config, world_size=1, rank=0):
    """
    Create per-expert data loaders with bucket sampling
    
    Args:
        dataset: Dataset with bucket assignments
        config: Configuration object
        world_size: World size for distributed training
        rank: Process rank
        
    Returns:
        Dictionary of {expert_idx: dataloader}
    """
    # Use the centralized bucket loader creation logic
    from utils.loader import create_expert_bucket_loaders as create_centralized_bucket_loaders
    
    return create_centralized_bucket_loaders(
        dataset=dataset,
        config=config,
        world_size=world_size,
        rank=rank
    )

def get_image_files(root_dir, extensions=None):
    """
    Get all image files in a directory
    
    Args:
        root_dir: Root directory to search
        extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png', '.webp'])
        
    Returns:
        List of image file paths
    """
    # Use the centralized image file collection logic
    from utils.loader import get_image_files as get_centralized_image_files
    
    return get_centralized_image_files(
        root_dir=root_dir,
        extensions=extensions
    )

def create_validation_loader(dataset, config, distributed=False, rank=0, world_size=1):
    """
    Create validation data loader
    
    Args:
        dataset: Dataset to create loader for
        config: Configuration object
        distributed: Whether to use DistributedSampler
        rank: Process rank (for distributed training)
        world_size: World size (for distributed training)
        
    Returns:
        Validation DataLoader
    """
    # Use the centralized validation loader creation logic
    from utils.loader import create_validation_loader as create_centralized_validation_loader
    
    return create_centralized_validation_loader(
        dataset=dataset,
        config=config,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )

def get_transform_for_dataset(config, is_train=True):
    """
    Get appropriate transforms for a dataset
    
    Args:
        config: Configuration object
        is_train: Whether this is for training
        
    Returns:
        Appropriate transforms
    """
    if is_train:
        return get_train_transforms(config)
    else:
        return get_val_transforms(config)

def create_dataloaders(dataset_train, dataset_val, config, distributed=False):
    """
    Create both training and validation dataloaders
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        config: Configuration object
        distributed: Whether to use distributed training
        
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    rank = get_rank() if distributed else 0
    world_size = get_world_size() if distributed else 1
    
    # Create training loader
    train_loader = create_loader(
        dataset=dataset_train,
        config=config,
        is_train=True,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    # Create validation loader
    val_loader = create_validation_loader(
        dataset=dataset_val,
        config=config,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }