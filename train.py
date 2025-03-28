#!/usr/bin/env python3
"""Main training script for Decentralized Diffusion Models (DDM)"""


import torch
import logging
import multiprocessing
import os

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
    """Improved distributed setup with error handling"""
    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {str(e)}") from e

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Validate NCCL version
    if dist.get_backend() == "nccl" and torch.cuda.nccl.version() < (2, 10):
        print("Recommend NCCL 2.10+ for best FSDP performance")
        
    return rank, world_size

def main():
    # Load configuration first
    config = get_config("config.py")

    logger = None # Initialize logger to None

    try:
        rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{rank}")
        
        # === Centralized Logger Initialization ===
        # Setup logger AFTER getting rank and world_size
        # Use a consistent name, e.g., "DDM"
        logger = setup_logger(
            name="DDM",
            output_dir=config.output_dir,
            log_to_console=True, # Control console logging
            level=logging.DEBUG if config.verbose_training else logging.INFO, # Control level via config
            rank=rank,
            world_size=world_size
        )
        # =========================================

        # Force FSDP configuration to FULL_SHARD for maximum parameter distribution
        config.fsdp_sharding_strategy = "FULL_SHARD"
        
        # Force proper device placement
        torch.cuda.set_device(rank)  # Explicitly set device
        
        # Use the initialized logger
        if logger: # Check if logger was successfully initialized
             log_training_start(logger, config, rank) # Pass logger instance
        else:
             # Fallback print if logger setup failed (shouldn't happen ideally)
             if rank == 0: print("Warning: Logger setup failed.")

        if rank == 0:
            # Print statements can be replaced by logger.info if desired
            logger.info("="*50)
            logger.info(" Initializing dataset - this may take a few minutes")
            logger.info(" Progress logs will be shown during the process")
            logger.info("="*50)
        # else: # No need for explicit setup_logger call here anymore
            # setup_logger("DDMCoordinator", config.output_dir, log_to_console=False) # REMOVE

        # Modified memory check with higher tolerance
        allocated = torch.cuda.memory_allocated(device)
        if allocated > 10 * 1024 * 1024:  # Allow 10MB for framework overhead
            # Use the logger for errors
            if logger: logger.error(f"Rank {rank} has {allocated/1024**2:.2f}MB allocated before model creation")
            raise RuntimeError(f"Rank {rank} has {allocated/1024**2:.2f}MB allocated before model creation")
        
        # Create expert cache manager with proper memory constraints
        max_experts_per_rank = max(1, config.max_experts_in_memory // world_size)
        cache_manager = ExpertCacheManager(
            config=config,
            device=device,
            max_experts=max_experts_per_rank,  # Per-rank expert limit
            cpu_offload=config.expert_offload_to_cpu,
            logger=logger # Pass logger to CacheManager if it needs it
        )
        
        # Initialize coordinator
        coordinator = DDMTrainingCoordinator(
            config=config,
            rank=rank,
            world_size=world_size,
            cache_manager=cache_manager,
            logger=logger # Pass the initialized logger
        )
        
        # Train
        coordinator.train(config.num_steps)
        
        # Enable memory efficient SDP and flash SDP
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        
    except Exception as e:
        # Use logger for top-level exception handling
        if logger:
            logger.exception(f"Training failed: {str(e)}") # Use logger.exception for stack trace
        else:
            print(f"Training failed (logger unavailable): {str(e)}") # Fallback print
        raise # Re-raise the exception
    finally:
        if dist.is_initialized():
            if logger: logger.info(f"Rank {rank}: Destroying process group.")
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 