#!/usr/bin/env python3
"""Main training script for Decentralized Diffusion Models (DDM)"""


import torch
import logging
import multiprocessing

import torch.distributed as dist

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Import core components
from trainers.coordinator import DDMTrainingCoordinator
from config import get_config
from utils.logging import setup_logger, log_training_start
from utils.expert_cache import ExpertCacheManager
from utils.distributed import is_dist_initialized

def setup_distributed():
    """Initialize distributed training environment using our centralized utilities"""
    # Use our centralized distributed setup
    from utils.distributed import setup_distributed as dist_setup, is_dist_initialized
    
    # Only set up if not already initialized
    if not is_dist_initialized():
        # Get rank and world size
        rank, world_size = dist_setup()
    else:
        # Get rank and world size if already initialized
        from utils.distributed import get_rank, get_world_size
        rank = get_rank()
        world_size = get_world_size()
    
    # Verify device assignment
    device = torch.device(f"cuda:{rank}")
    with torch.no_grad():
        test_tensor = torch.tensor([rank], device=device)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        del test_tensor  # Explicitly delete test tensor
        torch.cuda.synchronize(device)
    
    # Force memory cleanup before proceeding
    torch.cuda.empty_cache()
    
    # Log success message
    print(f"[Rank {rank}] Distributed setup complete with {world_size} processes")
    
    return rank, world_size

def main():
    # Add early synchronization
    if is_dist_initialized():
        dist.barrier()
    
    # Load configuration
    config = get_config("config.py")
    
    try:
        rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{rank}")
        
        # Force FSDP configuration to FULL_SHARD for maximum parameter distribution
        config.fsdp_sharding_strategy = "FULL_SHARD"
        
        # Force proper device placement
        torch.cuda.set_device(rank)  # Explicitly set device
        
        # Ensure all processes wait for model size prints to complete
        dist.barrier()
        
        # Clear GPU memory before proceeding
        torch.cuda.empty_cache()
        
        # Initialize logging on all processes for debugging
        # Modified logging setup with proper parameters
        if rank == 0:
            logger = setup_logger("DDMCoordinator", config.output_dir, log_to_console=True)
            log_training_start(logger, config, rank)
            
            print("="*50)
            print(" Initializing dataset - this may take a few minutes")
            print(" Progress logs will be shown during the process")
            print("="*50)
        else:
            setup_logger("DDMCoordinator", config.output_dir, log_to_console=False)
        
        # Modified memory check with higher tolerance
        allocated = torch.cuda.memory_allocated(device)
        if allocated > 10 * 1024 * 1024:  # Allow 10MB for framework overhead
            raise RuntimeError(f"Rank {rank} has {allocated/1024**2:.2f}MB allocated before model creation")
        
        # Create expert cache manager with proper memory constraints
        max_experts_per_rank = max(1, config.max_experts_in_memory // world_size)
        cache_manager = ExpertCacheManager(
            config=config,
            device=device,
            max_experts=max_experts_per_rank,  # Per-rank expert limit
            cpu_offload=config.expert_offload_to_cpu
        )
        
        # Initialize coordinator
        coordinator = DDMTrainingCoordinator(
            config=config,
            rank=rank,
            world_size=world_size,
            cache_manager=cache_manager
        )
        
        # Add verification after initialization
        if rank == 0:
            print("\nVerifying FSDP model distribution...")
            print("="*50)
            
        # Verify FSDP for all ranks
        if hasattr(coordinator, '_verify_sharding'):
            coordinator._verify_sharding()
            
        # Wait for verification to complete
        dist.barrier()
        
        # Train
        coordinator.train(config.num_steps)
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 