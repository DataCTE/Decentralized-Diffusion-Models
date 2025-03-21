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
    """Initialize distributed training environment"""
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(minutes=90)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Simple barrier sync
    dist.barrier()
    
    return rank, world_size

def main():
    # Load configuration
    config = get_config("config.py")
    
    try:
        rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{rank}")
        
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
