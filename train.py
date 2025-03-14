"""Main training script for Decentralized Diffusion Models (Paper Section 4)."""

import os
import torch
import torch.distributed
import datetime
import time


from config import DDMConfig
from trainers.coordinator import DDMTrainingCoordinator

# Import centralized utilities
from utils.logging import setup_logger, init_wandb
from utils.distributed import is_main_process, get_rank, synchronize
from utils.expert_cache import ExpertCacheManager

# Setup root logger
logger = None

def setup_distributed(config):
    """Initialize distributed training following paper Appendix A.4"""
    # Check if distributed training is enabled in config
    use_distributed = getattr(config, 'use_distributed', torch.cuda.device_count() > 1)
    if not use_distributed:
        logger.info("Running in single-GPU mode")
        return 0, 0, 1
    
    # Initialize distributed process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Use pre-existing environment variables (e.g. from slurm, torchrun)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        # Configure from config directly
        rank = getattr(config, 'rank', 0)
        local_rank = getattr(config, 'local_rank', 0)
        world_size = getattr(config, 'world_size', torch.cuda.device_count())
        
        # Set environment variables for subprocess compatibility
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    
    # Initialize process group
    init_method = getattr(config, 'dist_url', "env://")
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=1)
    )
    
    logger.info(f"Initialized distributed training: rank {rank}/{world_size}, local_rank: {local_rank}")
    return rank, local_rank, world_size

def create_directories(config):
    """Create necessary directories from config"""
    if is_main_process():
        # Create directories for logs, checkpoints, and samples
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        logger.info(f"Created directories: {config.log_dir}, {config.checkpoint_dir}, {config.sample_dir}")
    
    # Wait for main process to create directories
    synchronize()

def main():
    global logger
    
    start_time = time.time()
    
    # Load configuration
    config = DDMConfig()
    
    # Setup logging
    log_file = None
    if is_main_process():
        # Only create log files on main process
        os.makedirs(getattr(config, 'log_dir', 'logs'), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(getattr(config, 'log_dir', 'logs'), f"train-{timestamp}.log")
    
    # Setup root logger
    logger = setup_logger("DDMTraining", rank=get_rank(), log_file=log_file)
    logger.info("Starting Decentralized Diffusion Models training")
    
    # Initialize distributed training if enabled
    rank, local_rank, world_size = setup_distributed(config)
    
    # Create directories
    create_directories(config)
    
    # Initialize wandb logging if configured
    if is_main_process() and getattr(config, 'use_wandb', False):
        run_name = getattr(config, 'wandb_run_name', None) or f"ddm-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run = init_wandb(
            config=config,
            project=getattr(config, 'wandb_project', "decentralized-diffusion"),
            name=run_name
        )
        logger.info(f"Initialized WandB logging: {run_name}")
    
    try:
        # Initialize expert cache manager for memory-efficient expert loading
        logger.info("Initializing Expert Cache Manager")
        cache_manager = ExpertCacheManager(
            config=config,
            device=torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"),
            max_experts_in_memory=config.max_experts_in_memory,
            swap_strategy=config.expert_swap_strategy,
            cpu_offload=config.expert_offload_to_cpu,
            prefetch=config.expert_prefetch_next
        )
        
        # Create training coordinator
        logger.info("Creating DDM Training Coordinator")
        logger.info("This will initialize clustering, models, and data loaders")
        
        if is_main_process():
            logger.info("Feature extraction and clustering may take 20-30 minutes")
        
        coordinator = DDMTrainingCoordinator(
            config=config,
            rank=rank,
            world_size=world_size,
            cache_manager=cache_manager
        )
        
        # Load existing checkpoint if configured
        if hasattr(config, 'resume_from') and config.resume_from:
            start_step = coordinator.load_checkpoint(config.resume_from)
            logger.info(f"Resuming from checkpoint at step {start_step}")
        else:
            start_step = 0
        
        # Training loop
        logger.info(f"Starting training for {config.num_steps} steps")
        
        for step in range(start_step, config.num_steps):
            # Log progress periodically
            if is_main_process() and step % getattr(config, 'log_every_n_steps', 10) == 0:
                logger.info(f"Starting training step {step}/{config.num_steps}")
            
            # Expert training phase (Section 3.2)
            expert_loss = coordinator.train_experts(step)
            
            # Router training phase (Section 3.3)
            router_loss = coordinator.train_router(step)
            
            # Log metrics
            if step % getattr(config, 'log_every_n_steps', 10) == 0:
                coordinator.log_sharded_metrics(step, expert_loss, router_loss)
            
            # Reclustering if needed
            if coordinator.needs_reclustering(step):
                if is_main_process():
                    logger.info(f"Step {step}: Performing scheduled reclustering")
                coordinator.perform_reclustering()
                
                # Run validation after reclustering
                if is_main_process():
                    logger.info(f"Step {step}: Running validation after reclustering")
                coordinator.run_validation(step)
            
            # Regular validation
            if step % getattr(config, 'validation_interval', 1000) == 0:
                if is_main_process():
                    logger.info(f"Step {step}: Running scheduled validation")
                coordinator.run_validation(step)
                
                # Run ensemble validation to validate the DDM objective
                if is_main_process():
                    logger.info(f"Step {step}: Running ensemble validation")
                coordinator.run_ensemble_validation(step)
            
            # Checkpointing
            if step % getattr(config, 'save_interval', 5000) == 0 or step == config.num_steps - 1:
                if is_main_process():
                    logger.info(f"Step {step}: Saving checkpoint")
                coordinator.save_sharded_checkpoints(step)
        
        # Final validation and distillation
        if is_main_process():
            logger.info("Training complete, running final validation")
            coordinator.run_validation(config.num_steps)
            
            # Run distillation if configured
            if getattr(config, 'do_distillation', True):
                logger.info("Starting model distillation")
                coordinator.train_distilled_model()
            
            total_time = time.time() - start_time
            logger.info(f"Total training time: {total_time/3600:.1f} hours")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted - saving final checkpoint")
        try:
            coordinator.save_sharded_checkpoints(step)
        except Exception as e:
            logger.error(f"Error saving checkpoint after interruption: {str(e)}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
    finally:
        # Clean up cache manager
        if locals().get('cache_manager') is not None:
            try:
                cache_manager.shutdown()
                logger.info("Expert cache manager shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down cache manager: {str(e)}")
        
        # Clean up distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        if is_main_process() and getattr(config, 'use_wandb', False):
            import wandb
            wandb.finish()
            
        logger.info("Training finished")

if __name__ == "__main__":
    main() 