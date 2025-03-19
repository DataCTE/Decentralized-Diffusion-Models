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
from config import get_config
from utils.logging import setup_logger, log_training_start
from utils.checkpoint import load_coordinator_checkpoint
from utils.expert_cache import ExpertCacheManager

def setup_distributed():
    """Initialize distributed training environment"""
    # Set NCCL environment variables
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Adjust if needed
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # Initialize process group with longer timeout
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(minutes=90)  # 90 minute timeout
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device and ensure it's properly initialized
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Synchronize all processes before proceeding
    torch.cuda.synchronize(device)
    dist.barrier()
    
    return rank, world_size

def main():
    # Load configuration
    config = get_config("config.py")
    
    # Basic setup with error handling
    try:
        rank, world_size = setup_distributed()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        # Initialize logging only on main process
        if rank == 0:
            setup_logger(config.output_dir)
            log_training_start(logging.getLogger(), config, rank)
            
            print("="*50)
            print(" Initializing dataset - this may take a few minutes")
            print(" Progress logs will be shown during the process")
            print("="*50)
        
        # Synchronize before dataset initialization
        dist.barrier()
        
        # Create expert cache manager with proper error handling
        cache_manager = ExpertCacheManager(
            config=config,
            device=device,
            max_experts=config.max_experts_in_memory,
            cpu_offload=config.expert_offload_to_cpu
        )
        
        # Initialize coordinator with progress tracking
        coordinator = DDMTrainingCoordinator(
            config=config,
            rank=rank,
            world_size=world_size,
            cache_manager=cache_manager
        )
        
        # Train with proper error handling
        coordinator.train(config.num_steps)
        
    except Exception as e:
        logging.error(f"Training failed on rank {rank}: {str(e)}")
        raise
    finally:
        # Ensure cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
