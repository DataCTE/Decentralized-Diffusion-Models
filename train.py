"""Main training script for Decentralized Diffusion Models."""

import os
import torch
import torch.distributed
import datetime
import logging
import wandb

from config import DDMConfig
from trainers.coordinator import DDMTrainingCoordinator

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def setup_distributed():
    """Initialize distributed training across nodes"""
    # Get rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", 
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30)
        )
    
    # Verify initialization
    assert torch.distributed.is_initialized()
    
    return rank, local_rank, world_size

def train_ddm():
    """
    Main training function for Decentralized Diffusion Models
    
    Implements the distributed training approach described in Section 4.1
    of the paper, with multiple experts trained in parallel.
    """
    try:
        # Initialize distributed training (Section 4.1)
        rank, local_rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        
        logger.info(f"Starting DDM training with {world_size} GPUs")
        logger.info(f"Process rank: {rank}, local rank: {local_rank}")
        
        # Initialize training coordinator with paper-recommended settings
        coordinator = DDMTrainingCoordinator(
            DDMConfig(), rank, world_size
        )
        
        # Run training cycle (Algorithm 1 in the paper)
        coordinator.run_training_cycle()

        # Finalize training
        if rank == 0:
            logger.info("Training complete")
            wandb.finish()

    except KeyboardInterrupt:
        logger.warning("Training interrupted - saving state...")
        if rank == 0:
            coordinator.save_checkpoints()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train_ddm() 