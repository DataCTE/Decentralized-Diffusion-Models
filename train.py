"""Main training script for Decentralized Diffusion Models (Paper Section 4)."""

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
    """Initialize distributed training following paper Appendix A.4"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=1)
    )
    return rank, local_rank, world_size

def main():
    # Paper-mandated initialization sequence
    rank, local_rank, world_size = setup_distributed()
    torch.device(f"cuda:{local_rank}")
    
    # Initialize with paper-recommended config
    config = DDMConfig()
    
    # WandB setup only on rank 0 (paper Section 4.3)
    if rank == 0:
        wandb.init(
            project="decentralized-diffusion",
            config=config.__dict__,
            name=f"ddm-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    try:
        # Create training coordinator (paper Algorithm 2)
        coordinator = DDMTrainingCoordinator(config, rank, world_size)
        
        # Paper's recommended training cycle
        logger.info(f"Starting training for {config.num_steps} steps")
        for step in range(config.num_steps):
            # Expert training phase (Section 3.2)
            expert_loss = coordinator.train_experts(step)
            
            # Router training phase (Section 3.3)
            router_loss = coordinator.train_router(step)
            
            # Paper-mandated synchronization points
            if coordinator.needs_reclustering(step):
                coordinator.perform_reclustering()
                coordinator.run_validation(step)
            
            # Distributed metric logging (Section 4.3)
            coordinator.log_sharded_metrics(step, expert_loss, router_loss)
            
            # Checkpointing
            if step % config.save_interval == 0:
                coordinator.save_sharded_checkpoints(step)

        # Final validation and distillation (Section 3.6)
        if rank == 0:
            coordinator.run_validation(config.num_steps)
            coordinator.train_distilled_model()
            
    except KeyboardInterrupt:
        logger.info("Interrupted - saving final model")
        coordinator.save_sharded_checkpoints(config.num_steps)
    finally:
        torch.distributed.destroy_process_group()
        if rank == 0:
            wandb.finish()

if __name__ == "__main__":
    # Paper-recommended settings for logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main() 