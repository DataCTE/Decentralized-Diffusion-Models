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
    
    # Create expert cache manager
    cache_manager = ExpertCacheManager(
        config=config,
        device=device,
        max_experts=config.max_experts_in_memory,
        cpu_offload=config.expert_offload_to_cpu
    )

    try:
        # Initialize training coordinator
        coordinator = DDMTrainingCoordinator(
            config=config,
            rank=rank,
            world_size=world_size,
            cache_manager=cache_manager
        )

        # Load checkpoint if available
        if config.resume_checkpoint:
            checkpoint_dir = os.path.join(config.output_dir, 'checkpoints', config.resume_checkpoint)
            load_coordinator_checkpoint(coordinator, checkpoint_dir)

        # Main training loop
        if rank == 0:
            pbar = tqdm(total=config.num_steps, desc="Training DDM")
            
        for step in range(config.num_steps):
            # Training step handled by coordinator
            expert_loss, router_loss = coordinator.train_step()
            
            # Logging and checkpointing (only on main process)
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    "expert_loss": f"{expert_loss:.4f}",
                    "router_loss": f"{router_loss:.4f}"
                })

                # Save checkpoint periodically
                if step % config.checkpoint_interval == 0 and step > 0:
                    coordinator.save_checkpoint(step)

                # Generate validation samples
                if step % config.validation_interval == 0:
                    coordinator.validate(step)

        # Final save
        if rank == 0:
            coordinator.save_checkpoint(config.num_steps, final=True)
            pbar.close()

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
