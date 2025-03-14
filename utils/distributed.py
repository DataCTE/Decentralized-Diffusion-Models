"""Utility functions for distributed training"""

import torch
import torch.distributed as dist
import logging
import numpy as np

logger = logging.getLogger(__name__)

def broadcast_tensor(tensor, src_rank=0):
    """
    Broadcast a tensor from source rank to all processes
    
    Args:
        tensor: Tensor to broadcast (only needs to be defined on src_rank)
        src_rank: Source rank for broadcasting
        
    Returns:
        Broadcasted tensor on all ranks
    """
    if not dist.is_initialized():
        return tensor
        
    # Create empty tensor on non-source ranks
    if dist.get_rank() != src_rank:
        if tensor is not None:
            logger.warning(f"Tensor already exists on rank {dist.get_rank()} but will be overwritten")
        tensor = torch.empty(0, device='cuda')
    
    # Broadcast the shape first
    if dist.get_rank() == src_rank:
        shape_tensor = torch.LongTensor(list(tensor.shape)).cuda()
    else:
        shape_tensor = torch.LongTensor([0] * 4).cuda()  # Assuming max 4D tensors
    
    dist.broadcast(shape_tensor, src_rank)
    shape = shape_tensor.tolist()
    shape = [s for s in shape if s > 0]  # Filter out padding
    
    # Allocate correctly sized tensor on non-source ranks
    if dist.get_rank() != src_rank:
        tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
    
    # Broadcast actual data
    dist.broadcast(tensor, src_rank)
    
    return tensor

def broadcast_object(obj, src_rank=0):
    """
    Broadcast a Python object (via pickle) from source rank to all processes
    
    Args:
        obj: Python object to broadcast (only needs to be defined on src_rank)
        src_rank: Source rank for broadcasting
        
    Returns:
        Broadcasted object on all ranks
    """
    if not dist.is_initialized():
        return obj
    
    import pickle
    
    if dist.get_rank() == src_rank:
        # Serialize the object
        buffer = pickle.dumps(obj)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to('cuda')
    else:
        # Create dummy tensor
        tensor = torch.empty(0, dtype=torch.uint8, device='cuda')
    
    # Broadcast tensor size
    local_size = torch.LongTensor([tensor.numel()]).to('cuda')
    dist.broadcast(local_size, src_rank)
    
    # Resize tensor on non-source ranks
    if dist.get_rank() != src_rank:
        tensor = torch.empty(local_size.item(), dtype=torch.uint8, device='cuda')
    
    # Broadcast actual data
    dist.broadcast(tensor, src_rank)
    
    # Deserialize on non-source ranks
    if dist.get_rank() != src_rank:
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    
    return obj

def broadcast_numpy_array(array, src_rank=0):
    """
    Broadcast a numpy array from source rank to all processes
    
    Args:
        array: Numpy array to broadcast (only needs to be defined on src_rank)
        src_rank: Source rank for broadcasting
        
    Returns:
        Broadcasted numpy array on all ranks
    """
    if not dist.is_initialized():
        return array
    
    if dist.get_rank() == src_rank and array is not None:
        # Convert numpy array to tensor
        tensor = torch.from_numpy(array).cuda()
    else:
        # Create dummy tensor 
        tensor = torch.empty(0, device='cuda')
    
    # Broadcast the tensor
    tensor = broadcast_tensor(tensor, src_rank)
    
    # Convert back to numpy array
    return tensor.cpu().numpy()

def setup_distributed_printer(rank):
    """Set up a printer that only prints on the specified rank"""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        if rank == 0:
            builtin_print(*args, **kwargs)
            
    __builtin__.print = print 