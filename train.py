#!/usr/bin/env python3
"""Main training script for Decentralized Diffusion Models (DDM)"""

import os
import torch
import logging
import multiprocessing
from datetime import datetime, timedelta

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Import core components
from trainers.coordinator import DDMTrainingCoordinator
from config import get_config, estimate_model_size
from utils.logging import setup_logger, log_training_start
from utils.checkpoint import load_coordinator_checkpoint
from utils.expert_cache import ExpertCacheManager

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
    test_tensor = torch.tensor([rank], device=device)
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    
    # Log success message
    print(f"[Rank {rank}] Distributed setup complete with {world_size} processes")
    
    return rank, world_size

def main():
    # Load configuration
    config = get_config("config.py")
    
    try:
        rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{rank}")
        
        # Force FSDP configuration to FULL_SHARD for maximum parameter distribution
        # This ensures models are fully sharded across all GPUs
        config.fsdp_sharding_strategy = "FULL_SHARD"
        
        # Add model size estimation here, AFTER distributed setup
        if rank == 0:
            print("Estimating ExpertMMDiT model size:")
            estimate_model_size(config, model_type="expert")
            print("\nEstimating RouterModel size:")
            estimate_model_size(config, model_type="router")
        
        # Ensure all processes wait for model size prints to complete
        dist.barrier()
        torch.cuda.empty_cache()
        
        # Initialize logging only on main process
        if rank == 0:
            setup_logger(config.output_dir)
            log_training_start(logging.getLogger(), config, rank)
            
            print("="*50)
            print(" Initializing dataset - this may take a few minutes")
            print(" Progress logs will be shown during the process")
            print("="*50)
        
        # Create expert cache manager
        cache_manager = ExpertCacheManager(
            config=config,
            device=device,
            max_experts=config.max_experts_in_memory,
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
