import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import os
import logging
import time
import json
from utils.distributed import is_main_process, is_dist_initialized

logger = logging.getLogger(__name__)

def safe_synchronize(timeout_seconds=30):
    """Safely synchronize without timeout for compatibility"""
    if not is_dist_initialized():
        return True
        
    try:
        # Simple barrier call without timeout
        dist.barrier()
        return True
    except Exception as e:
        logger.error(f"Synchronization error: {str(e)}")
        return False

def save_sharded(checkpoint, path, model=None):
    """Paper's recommended checkpoint format from Appendix A.2"""
    # First, let non-main processes wait a short time
    if not is_main_process() and is_dist_initialized():
        time.sleep(0.1)  # Short delay to allow main process to start first
    
    if model is not None:
        try:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state = model.state_dict()
            checkpoint = {**checkpoint, 'model': state}
        except Exception as e:
            logger.error(f"Failed to get model state_dict: {str(e)}")
            if is_main_process():
                # Don't include model state if it failed
                checkpoint = {**checkpoint, 'model': {}}
    
    if not is_main_process():
        # Only save from main process
        safe_synchronize()  # Ensure all processes wait for main to complete save
        return None
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        # Notify other processes that save is complete
        if is_dist_initialized():
            # Use a simple broadcast as a signal
            signal = torch.tensor([1], device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(signal, src=0)
        return path
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
        # Signal error to other processes
        if is_dist_initialized():
            # Use a simple broadcast as a signal
            signal = torch.tensor([0], device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(signal, src=0)
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
        Path to saved checkpoint (None if not main process)
    """
    # First, let non-main processes wait a short time
    if not is_main_process() and is_dist_initialized():
        time.sleep(0.1)  # Short delay to allow main process to start first
    
    # Create checkpoint
    checkpoint = {'metadata': metadata or {}}
    
    # Add model state
    if model is not None:
        try:
            if is_fsdp and isinstance(model, FSDP):
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    checkpoint['model'] = model.state_dict()
            else:
                checkpoint['model'] = model.state_dict()
        except Exception as e:
            logger.error(f"Failed to get model state_dict: {str(e)}")
            if is_main_process():
                # Don't include model state if it failed
                checkpoint['model'] = {}
    
    # Add optimizer and scheduler states if provided
    if optimizer is not None:
        try:
            if is_fsdp and isinstance(optimizer, torch.optim.Optimizer):
                optim_state = FSDP.optim_state_dict(model, optimizer)
                checkpoint['optimizer'] = optim_state
            else:
                checkpoint['optimizer'] = optimizer.state_dict()
        except Exception as e:
            logger.error(f"Failed to get optimizer state_dict: {str(e)}")
            checkpoint['optimizer'] = {}
            
    if scheduler is not None:
        try:
            checkpoint['scheduler'] = scheduler.state_dict()
        except Exception as e:
            logger.error(f"Failed to get scheduler state_dict: {str(e)}")
            checkpoint['scheduler'] = {}
    
    if not is_main_process():
        # Only save from main process
        safe_synchronize()  # Wait for main process to finish saving
        return None
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Actual saving (only on main process)
    try:
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        # Notify other processes that save is complete
        if is_dist_initialized():
            # Use a simple broadcast as a signal
            signal = torch.tensor([1], device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(signal, src=0)
        return path
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
        # Signal error to other processes
        if is_dist_initialized():
            # Use a simple broadcast as a signal
            signal = torch.tensor([0], device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(signal, src=0)
        return None
        
def load_model_checkpoint(model, optimizer=None, scheduler=None, path=None, is_fsdp=True, device=None):
    """
    Load a model checkpoint
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        path: Path to checkpoint
        is_fsdp: Whether the model is using FSDP
        device: Device to load to
        
    Returns:
        Metadata from checkpoint
    """
    # Wait for main process to start loading first
    if not is_main_process() and is_dist_initialized():
        time.sleep(0.1)
    
    if not os.path.exists(path):
        logger.error(f"Checkpoint not found at {path}")
        safe_synchronize()
        return None
    
    try:
        # Load checkpoint
        checkpoint_data = None
        
        # For FSDP, only rank 0 needs to read the file unless we need optimizer state
        if is_main_process() or not is_fsdp or optimizer is not None:
            checkpoint_data = torch.load(path, map_location='cpu')
        
        # Broadcast checkpoint from rank 0 if using distributed
        if is_dist_initialized():
            # Indicate whether loading was successful on rank 0
            success = torch.tensor([checkpoint_data is not None], dtype=torch.bool, 
                                  device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(success, src=0)
            
            if not success.item():
                logger.error(f"Loading checkpoint failed on rank 0")
                return None
                
            # If not main process and using FSDP and don't need optimizer, 
            # no need to load the full checkpoint
            if not is_main_process() and is_fsdp and optimizer is None:
                logger.info(f"Using FSDP state_dict loading for checkpoint {path}")
                # We'll use FSDP's state_dict loading below
            elif not is_main_process():
                # Other ranks need to receive the checkpoint data
                checkpoint_bytes = torch.empty([1], dtype=torch.uint8, 
                                             device='cuda' if torch.cuda.is_available() else 'cpu')
                dist.broadcast(checkpoint_bytes, src=0)
                checkpoint_data = torch.load(checkpoint_bytes, map_location='cpu')
        
        # Load model state
        if 'model' in checkpoint_data and model is not None:
            # Load model state with appropriate method
            if is_fsdp and isinstance(model, FSDP):
                load_policy = FullStateDictConfig(offload_to_cpu=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
                    model.load_state_dict(checkpoint_data['model'])
            else:
                model.load_state_dict(checkpoint_data['model'])
                
            logger.info(f"Loaded model from {path}")
                
        # Load optimizer state if available and requested
        if 'optimizer' in checkpoint_data and optimizer is not None:
            if is_fsdp and isinstance(optimizer, torch.optim.Optimizer) and isinstance(model, FSDP):
                FSDP.optim_state_dict_to_load(checkpoint_data['optimizer'], model, optimizer)
            else:
                optimizer.load_state_dict(checkpoint_data['optimizer'])
                
            logger.info(f"Loaded optimizer from {path}")
                
        # Load scheduler state if available and requested
        if 'scheduler' in checkpoint_data and scheduler is not None:
            scheduler.load_state_dict(checkpoint_data['scheduler'])
            logger.info(f"Loaded scheduler from {path}")
                
        # Return metadata
        metadata = checkpoint_data.get('metadata', {})
        
        # Synchronize all processes before continuing
        safe_synchronize()
        
        return metadata
            
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {path}: {str(e)}")
        # Ensure other processes are notified of failure
        if is_dist_initialized() and is_main_process():
            # Use a simple broadcast as a signal
            signal = torch.tensor([0], device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(signal, src=0)
        
        safe_synchronize()
        return None

def save_coordinator_checkpoint(save_dir, state_dict):
    """
    Save coordinator state separately from model checkpoints
    
    Args:
        save_dir: Directory to save checkpoint
        state_dict: Dictionary of state to save
    """
    if not is_main_process():
        return
        
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "coordinator_state.json")
    
    # Convert any non-serializable objects
    serializable_dict = {}
    for k, v in state_dict.items():
        if k == 'config':
            # Convert config object to dict 
            if hasattr(v, '__dict__'):
                serializable_dict[k] = v.__dict__
            else:
                serializable_dict[k] = str(v)  # Fall back to string representation
        elif isinstance(v, torch.Tensor):
            # Convert tensors to lists
            serializable_dict[k] = v.cpu().numpy().tolist()
        else:
            # Keep other values as is if JSON serializable
            try:
                json.dumps({k: v})
                serializable_dict[k] = v
            except (TypeError, OverflowError):
                serializable_dict[k] = str(v)  # Fall back to string
    
    # Save as JSON
    with open(checkpoint_path, 'w') as f:
        json.dump(serializable_dict, f, indent=4)
        
    logger.info(f"Saved coordinator state to {checkpoint_path}")
    
def load_coordinator_checkpoint(checkpoint_dir):
    """
    Load coordinator state from checkpoint
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        
    Returns:
        Dictionary of loaded state or None if not found
    """
    checkpoint_path = os.path.join(checkpoint_dir, "coordinator_state.json")
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Coordinator checkpoint not found at {checkpoint_path}")
        return None
        
    try:
        with open(checkpoint_path, 'r') as f:
            state_dict = json.load(f)
            
        logger.info(f"Loaded coordinator state from {checkpoint_path}")
        return state_dict
    except Exception as e:
        logger.error(f"Error loading coordinator checkpoint: {e}")
        return None

# Create a new utility function for debugging
def debug_checkpoint_hook(module, input, output):
    if torch.distributed.is_initialized() and torch.rand(1).item() < 0.005:
        rank = torch.distributed.get_rank()
        print(f"[Rank {rank}] Checkpoint hook - input shapes: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}")
        print(f"[Rank {rank}] Checkpoint hook - output shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
    return None

# Register this hook in appropriate places