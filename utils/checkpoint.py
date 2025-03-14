import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import os
import logging

logger = logging.getLogger(__name__)

def save_sharded(checkpoint, path, model=None):
    """Paper's recommended checkpoint format from Appendix A.2"""
    if model is not None:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state = model.state_dict()
        checkpoint = {**checkpoint, 'model': state}
    
    if dist.is_initialized() and dist.get_rank() != 0:
        # Only save from rank 0
        return None
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
        return None

def save_model_checkpoint(model, optimizer=None, scheduler=None, path=None, metadata=None, is_fsdp=True):
    """
    Save model checkpoint with optimizers and metadata
    
    Args:
        model: Model to save
        optimizer: Optimizer to save (optional)
        scheduler: Learning rate scheduler to save (optional)
        path: Path to save checkpoint to
        metadata: Additional metadata to save
        is_fsdp: Whether the model is using FSDP
        
    Returns:
        Path to saved checkpoint (None if not rank 0)
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        # Only save from rank 0
        return None
        
    # Create checkpoint
    checkpoint = {'metadata': metadata or {}}
    
    # Add model state
    if is_fsdp and isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            checkpoint['model'] = model.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
    
    # Add optimizer and scheduler states if provided
    if optimizer is not None:
        if is_fsdp and isinstance(optimizer, torch.optim.Optimizer):
            optim_state = FSDP.optim_state_dict(model, optimizer)
            checkpoint['optimizer'] = optim_state
        else:
            checkpoint['optimizer'] = optimizer.state_dict()
            
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
        
    return save_sharded(checkpoint, path)

def load_model_checkpoint(model, path, optimizer=None, scheduler=None, is_fsdp=True, device=None):
    """
    Load model checkpoint with optimizers and metadata
    
    Args:
        model: Model to load weights into
        path: Path to load checkpoint from
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        is_fsdp: Whether the model is using FSDP
        device: Device to load checkpoint to (if not using FSDP)
        
    Returns:
        Metadata from checkpoint
    """
    if not os.path.exists(path):
        logger.error(f"Checkpoint not found at {path}")
        return None
        
    try:
        # Load checkpoint
        if device is not None:
            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(path)
            
        # Load model state
        if is_fsdp and isinstance(model, FSDP):
            load_policy = FullStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
                model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
            
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer' in checkpoint:
            if is_fsdp and isinstance(optimizer, torch.optim.Optimizer):
                optim_state = checkpoint['optimizer']
                FSDP.optim_state_dict_to_load(optim_state, model, optimizer)
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
                
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('metadata', {})
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {path}: {str(e)}")
        return None