"""Distributed utilities for Decentralized Diffusion Models."""

import torch
import torch.distributed as dist
import pickle
import logging
import io
from datetime import timedelta
import os
import time

logger = logging.getLogger(__name__)

def is_dist_initialized():
    """Check if distributed training is initialized"""
    return dist.is_initialized()

def get_rank():
    """Get the rank of the current process."""
    return dist.get_rank() if is_dist_initialized() else 0

def get_local_rank():
    """Get the local rank of the current process from the LOCAL_RANK environment variable."""
    return int(os.environ.get('LOCAL_RANK', '0'))

def get_world_size():
    """Get the world size (number of processes)."""
    return dist.get_world_size() if is_dist_initialized() else 1

def is_main_process():
    """Determines if the current process is the main process."""
    return get_rank() == 0

def synchronize():
    """Synchronize all processes"""
    if not is_dist_initialized():
        return
    dist.barrier()

def broadcast_object(obj, src=0, group=None, device=None, timeout=None):
    """
    Broadcast an arbitrary Python object from src to all processes in the group.
    
    Args:
        obj: Object to broadcast (any picklable Python object)
        src: Source rank for broadcast
        group: Process group for communication
        device: Device to use for tensor communication
        timeout: Timeout in seconds for the operation (ignored in this version of PyTorch)
    
    Returns:
        The broadcast object on all ranks
    """
    if not is_dist_initialized():
        return obj

    # Get rank info for logging
    rank = get_rank()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Maximum number of retries for robustness
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # If we are the source process, pickle the object
            if rank == src:
                buffer = io.BytesIO()
                torch.save(obj, buffer)
                data = bytearray(buffer.getvalue())
                size = torch.tensor([len(data)], dtype=torch.long, device=device)
            else:
                size = torch.tensor([0], dtype=torch.long, device=device)
            
            # Broadcast the size of the pickled object
            # Note: timeout parameter is ignored in this version of PyTorch
            dist.broadcast(size, src=src, group=group)
            
            # Broadcast the pickled object
            if rank == src:
                tensor = torch.tensor(list(data), dtype=torch.uint8, device=device)
            else:
                tensor = torch.empty(size.item(), dtype=torch.uint8, device=device)
            
            dist.broadcast(tensor, src=src, group=group)
            
            # If we're not the source, unpickle the object
            if rank != src:
                buffer = io.BytesIO(tensor.cpu().numpy().tobytes())
                obj = torch.load(buffer)
            
            # Successful broadcast
            return obj
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger = logging.getLogger(__name__)
                logger.warning(f"Rank {rank}: Broadcast attempt {attempt+1} failed: {str(e)}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger = logging.getLogger(__name__)
                logger.error(f"Rank {rank}: All broadcast attempts failed. Last error: {str(e)}")
                raise

def broadcast_tensor(tensor, src=0):
    """
    Broadcast a tensor from source rank to all processes
    
    Args:
        tensor: Tensor to broadcast (must exist on source rank)
        src: Source rank
    
    Returns:
        Broadcasted tensor on all ranks
    """
    if not is_dist_initialized():
        return tensor
    
    # Create empty tensor on non-source ranks
    if get_rank() != src:
        if tensor is None:
            shape = broadcast_object((0,), src)
            tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
        elif not tensor.is_cuda:
            tensor = tensor.cuda()
    else:
        if not tensor.is_cuda:
            tensor = tensor.cuda()
    
    # Broadcast shape and metadata
    shape = broadcast_object(tensor.shape, src)
    dtype = broadcast_object(tensor.dtype, src)
    
    # Ensure tensor is on correct device with correct shape
    if get_rank() != src:
        tensor = torch.empty(shape, dtype=dtype, device='cuda')
    
    # Broadcast tensor
    dist.broadcast(tensor, src=src)
    
    return tensor

def broadcast_numpy_array(array, src=0):
    """
    Broadcast a numpy array from source rank to all processes
    
    Args:
        array: Numpy array to broadcast
        src: Source rank
    
    Returns:
        Broadcasted numpy array on all ranks
    """
    if not is_dist_initialized():
        return array
    
    # Convert to tensor, broadcast, then convert back to numpy
    if get_rank() == src:
        tensor = torch.from_numpy(array).cuda()
    else:
        tensor = None
    
    tensor = broadcast_tensor(tensor, src)
    
    # Convert back to numpy
    return tensor.cpu().numpy()

def reduce_dict(input_dict, average=True):
    """
    Average or sum the values in a dictionary across processes
    
    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average or sum
    
    Returns:
        Reduced dictionary
    """
    if not is_dist_initialized():
        return input_dict
    
    # Create world-size copies of the dictionary
    world_size = get_world_size()
    
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        # Convert dict to tensor for reduction
        names = []
        values = []
        
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
            
        values = torch.stack(values, dim=0)
        
        # All-reduce
        dist.all_reduce(values)
        
        # Average if needed
        if average:
            values /= world_size
            
        # Reconstruct dictionary
        reduced_dict = {k: v for k, v in zip(names, values)}
        
    return reduced_dict

def gather_tensor(tensor, dst=0):
    """
    Gather tensors from all ranks to a specific rank
    
    Args:
        tensor: Tensor to gather (must be same shape on all ranks)
        dst: Destination rank
        
    Returns:
        List of tensors on destination rank, None on others
    """
    if not is_dist_initialized():
        return [tensor]
    
    world_size = get_world_size()
    
    if world_size < 2:
        return [tensor]
    
    # Ensure tensor is on CUDA
    if not tensor.is_cuda:
        tensor = tensor.cuda()
    
    # Create output list on destination
    if get_rank() == dst:
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gathered = None
    
    # Gather
    dist.gather(tensor, gathered, dst=dst)
    
    return gathered

def all_gather_tensor(tensor):
    """
    Gather tensors from all ranks to all ranks
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all ranks
    """
    if not is_dist_initialized():
        return [tensor]
    
    world_size = get_world_size()
    
    if world_size < 2:
        return [tensor]
    
    # Ensure tensor is on CUDA
    if not tensor.is_cuda:
        tensor = tensor.cuda()
        
    # Create output list
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    
    # All-gather
    dist.all_gather(gathered, tensor)
    
    return gathered

def setup_distributed_printer(rank):
    """Set up a printer that only prints on the specified rank"""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        if rank == 0:
            builtin_print(*args, **kwargs)
            
    __builtin__.print = print 