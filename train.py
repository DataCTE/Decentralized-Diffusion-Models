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
from utils.distributed import setup_distributed

def main():
    # Load configuration
    config = get_config("config.py")
    
    try:
        # Initialize logging only on main process
        if dist.get_rank() == 0:
            setup_logger(config.output_dir)
            log_training_start(logging.getLogger(), config, dist.get_rank())
            
            print("="*50)
            print(" Initializing dataset - this may take a few minutes")
            print(" Progress logs will be shown during the process")
            print("="*50)
        
        # Create expert cache manager
        cache_manager = ExpertCacheManager(
            config=config,
            device=torch.device(f"cuda:{dist.get_rank()}"),
            max_experts=config.max_experts_in_memory,
            cpu_offload=config.expert_offload_to_cpu
        )
        
        # Initialize coordinator
        coordinator = DDMTrainingCoordinator(
            config=config,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
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
