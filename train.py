"""Main training script for Decentralized Diffusion Models (Paper Section 4)."""

import os
import torch
import torch.distributed
import datetime
import logging
import wandb
import time

from config import DDMConfig
from trainers.coordinator import DDMTrainingCoordinator

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
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
    # Log startup information
    start_time = time.time()
    logger.info("Starting Decentralized Diffusion Models training")
    
    # Paper-mandated initialization sequence
    rank, local_rank, world_size = setup_distributed()
    if rank == 0:
        logger.info(f"Distributed setup complete: {world_size} processes (rank {rank}, local_rank {local_rank})")
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize with paper-recommended config
    config = DDMConfig()
    
    # WandB setup only on rank 0 (paper Section 4.3)
    if rank == 0:
        logger.info("Initializing Weights & Biases logging")
        wandb.init(
            project="decentralized-diffusion",
            config=config.__dict__,
            name=f"ddm-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    try:
        # Create training coordinator (paper Algorithm 2)
        if rank == 0:
            logger.info("Creating DDM Training Coordinator")
            logger.info("This will initialize clustering, models, and data loaders")
            logger.info("Feature extraction and clustering may take 20-30 minutes")
            
        coordinator = DDMTrainingCoordinator(config, rank, world_size)
        
        # Paper's recommended training cycle
        if rank == 0:
            logger.info(f"Starting training for {config.num_steps} steps")
            setup_time = time.time() - start_time
            logger.info(f"Setup completed in {setup_time/60:.1f} minutes")
            
        for step in range(config.num_steps):
            # Log step information on main process
            if rank == 0 and step % 10 == 0:
                logger.info(f"Starting training step {step}/{config.num_steps}")
                
            # Expert training phase (Section 3.2)
            expert_loss = coordinator.train_experts(step)
            
            # Router training phase (Section 3.3)
            router_loss = coordinator.train_router(step)
            
            # Paper-mandated synchronization points
            if coordinator.needs_reclustering(step):
                if rank == 0:
                    logger.info(f"Step {step}: Performing scheduled reclustering")
                coordinator.perform_reclustering()
                
                if rank == 0:
                    logger.info(f"Step {step}: Running validation after reclustering")
                coordinator.run_validation(step)
            
            # Distributed metric logging (Section 4.3)
            coordinator.log_sharded_metrics(step, expert_loss, router_loss)
            
            # Checkpointing
            if step % config.save_interval == 0:
                if rank == 0:
                    logger.info(f"Step {step}: Saving checkpoint")
                coordinator.save_sharded_checkpoints(step)

        # Final validation and distillation (Section 3.6)
        if rank == 0:
            logger.info("Training complete, running final validation")
            coordinator.run_validation(config.num_steps)
            
            logger.info("Starting model distillation")
            coordinator.train_distilled_model()
            
            total_time = time.time() - start_time
            logger.info(f"Total training time: {total_time/3600:.1f} hours")
            
    except KeyboardInterrupt:
        logger.info("Interrupted - saving final model")
        coordinator.save_sharded_checkpoints(config.num_steps)
    finally:
        torch.distributed.destroy_process_group()
        if rank == 0:
            wandb.finish()
            logger.info("Training finished")

if __name__ == "__main__":
    # Paper-recommended settings for logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main() 