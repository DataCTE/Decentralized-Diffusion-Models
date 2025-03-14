"""Data loading utilities for Decentralized Diffusion Models."""


import logging


# Import centralized utilities
from utils.distributed import get_rank, get_world_size 
from data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

def create_loader(dataset, config, is_train=True, distributed=False, rank=0, world_size=1):
    """
    Paper's data loading defaults from Appendix A.1
    
    Args:
        dataset: Dataset to create loader for
        config: Configuration object
        is_train: Whether this is a training loader
        distributed: Whether to use DistributedSampler
        rank: Process rank (for distributed training)
        world_size: World size (for distributed training)
        
    Returns:
        DataLoader instance
    """
    # Use the centralized sampler creation logic
    from utils.loader import create_loader as create_centralized_loader
    
    return create_centralized_loader(
        dataset=dataset,
        config=config,
        is_train=is_train,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )

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