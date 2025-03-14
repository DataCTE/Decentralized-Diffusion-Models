"""Distributed utilities for Decentralized Diffusion Models."""

import torch
import torch.distributed as dist
import numpy as np
import pickle
import logging
import os

logger = logging.getLogger(__name__)

def is_dist_initialized():
    """Check if distributed training is initialized"""
    return dist.is_initialized()

def get_rank():
    """Get current process rank"""
    return dist.get_rank() if is_dist_initialized() else 0

def get_world_size():
    """Get total number of processes"""
    return dist.get_world_size() if is_dist_initialized() else 1

def is_main_process():
    """Check if current process is the main process (rank 0)"""
    return get_rank() == 0

def synchronize():
    """Synchronize all processes"""
    if not is_dist_initialized():
        return
    dist.barrier()

def broadcast_object(obj, src=0):
    """
    Broadcast a Python object from a source rank to all other processes
    
    Args:
        obj: Python object to broadcast
        src: Source rank
    
    Returns:
        Object on all ranks
    """
    if not is_dist_initialized():
        return obj
    
    # Main process serializes object
    if get_rank() == src:
        buffer = pickle.dumps(obj)
        size = torch.tensor(len(buffer), dtype=torch.long, device='cuda')
    else:
        size = torch.tensor(0, dtype=torch.long, device='cuda')
    
    # Broadcast size
    dist.broadcast(size, src=src)
    
    # Create tensor of appropriate size
    if get_rank() == src:
        buffer_tensor = torch.ByteTensor(list(buffer)).cuda()
    else:
        buffer_tensor = torch.empty(size.item(), dtype=torch.uint8, device='cuda')
    
    # Broadcast data
    dist.broadcast(buffer_tensor, src=src)
    
    # Deserialize on non-source ranks
    if get_rank() != src:
        buffer = buffer_tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    
    return obj

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