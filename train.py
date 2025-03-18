#!/usr/bin/env python3
"""Main training script for Decentralized Diffusion Models (DDM)"""

import os
import torch
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import core components
from trainers.coordinator import DDMTrainingCoordinator
from config import get_config
from utils.logging import setup_logger, log_training_start
from utils.checkpoint import load_coordinator_checkpoint
from utils.expert_cache import ExpertCacheManager

def setup_distributed():
    """Initialize distributed training environment"""
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(minutes=15)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def main():
    # Basic setup
    rank, world_size = setup_distributed()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Load configuration
    config = get_config("config.py")
    
    # Initialize logging only on main process
    if rank == 0:
        setup_logger(config.output_dir)
        log_training_start(logging.getLogger(), config, rank)
        
        # Show dataset initialization message
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

    try:
        # Initialize coordinator with progress tracking
        start_time = datetime.now()
        
        # Initialize training coordinator
        coordinator = DDMTrainingCoordinator(
            config=config,
            rank=rank,
            world_size=world_size,
            cache_manager=cache_manager
        )
        
        # Log dataset initialization completion time
        init_time = datetime.now() - start_time
        if rank == 0:
            print(f"Dataset initialization completed in {init_time.total_seconds():.2f} seconds")

        # Load checkpoint if available
        if config.resume_checkpoint:
            checkpoint_dir = os.path.join(config.output_dir, 'checkpoints', config.resume_checkpoint)
            load_coordinator_checkpoint(coordinator, checkpoint_dir)

        # Call the train method directly, which already contains the main training loop
        coordinator.train(config.num_steps)

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
