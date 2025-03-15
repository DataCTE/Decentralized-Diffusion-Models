"""Main training script for Decentralized Diffusion Models (Paper Section 4)."""

import os
import torch
import torch.distributed
import datetime
import time
import sys
import logging


from config import DDMConfig
from trainers.coordinator import DDMTrainingCoordinator

# Import centralized utilities
from utils.logging import setup_logger, init_wandb
from utils.distributed import is_main_process, get_rank, synchronize
from utils.expert_cache import ExpertCacheManager

# Setup a basic logger early to prevent null reference errors
# This will be replaced with a properly configured logger in main()
logger = logging.getLogger("DDMTraining")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Add a direct console print function for immediate feedback regardless of logger
def console_print(message, force=False):
    """Print directly to console with timestamp regardless of rank unless force=False and not rank 0"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if force or is_main_process():
        print(f"[{timestamp}] {message}", flush=True)

def setup_distributed(config):
    """Initialize distributed training following paper Appendix A.4"""
    console_print("Setting up distributed training...")
    
    # Check if distributed training is enabled in config
    use_distributed = getattr(config, 'use_distributed', torch.cuda.device_count() > 1)
    if not use_distributed:
        console_print("Running in single-GPU mode")
        logger.info("Running in single-GPU mode")
        return 0, 0, 1
    
    # Initialize distributed process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Use pre-existing environment variables (e.g. from slurm, torchrun)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
        console_print(f"Using environment variables: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}", rank == 0)
    else:
        # Configure from config directly
        rank = getattr(config, 'rank', 0)
        local_rank = getattr(config, 'local_rank', 0)
        world_size = getattr(config, 'world_size', torch.cuda.device_count())
        
        # Set environment variables for subprocess compatibility
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        console_print(f"Setting environment variables: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}", rank == 0)
    
    # Set device
    console_print(f"Setting device for rank {rank}, local_rank {local_rank}", rank == 0)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        console_print(f"Rank {rank}: Using CUDA device {local_rank}, backend=nccl", rank == 0)
    else:
        backend = "gloo"
        console_print(f"Rank {rank}: CUDA not available, using gloo backend", rank == 0)
    
    # Initialize process group
    console_print(f"Rank {rank}: Initializing process group", rank == 0)
    init_method = getattr(config, 'dist_url', "env://")
    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(hours=1)
        )
        console_print(f"Rank {rank}: Process group initialized successfully", rank == 0)
    except Exception as e:
        console_print(f"ERROR: Rank {rank}: Failed to initialize process group: {str(e)}", True)
        raise
    
    console_print(f"Rank {rank}: Distributed training initialized: {rank}/{world_size}, local_rank: {local_rank}")
    logger.info(f"Initialized distributed training: rank {rank}/{world_size}, local_rank: {local_rank}")
    return rank, local_rank, world_size

def create_directories(config):
    """Create necessary directories from config"""
    console_print("Creating necessary directories...")
    if is_main_process():
        # Create directories for logs, checkpoints, and samples
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        console_print(f"Created directories: {config.log_dir}, {config.checkpoint_dir}, {config.sample_dir}")
        logger.info(f"Created directories: {config.log_dir}, {config.checkpoint_dir}, {config.sample_dir}")
    
    # Wait for main process to create directories
    console_print(f"Rank {get_rank()}: Waiting for directory creation to complete...")
    synchronize()
    console_print(f"Rank {get_rank()}: Directory synchronization complete")

def main():
    global logger
    
    # Print startup message immediately to console
    print("\n" + "="*80)
    print(f"Starting Decentralized Diffusion Models training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Load configuration
    print("Loading configuration...")
    config = DDMConfig()
    print("Configuration loaded")
    
    # Setup logging
    print("Setting up logging...")
    log_file = None
    if not 'RANK' in os.environ or int(os.environ.get('RANK', '0')) == 0:
        # Only create log files on main process
        os.makedirs(getattr(config, 'log_dir', 'logs'), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(getattr(config, 'log_dir', 'logs'), f"train-{timestamp}.log")
        print(f"Log file will be saved at: {log_file}")
    
    # Replace the early global logger with the properly configured one
    # Remove existing handlers to avoid duplicate messages
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Setup new logger with file and console output
    logger = setup_logger("DDMTraining", log_file=log_file)
    logger.info("Starting Decentralized Diffusion Models training")
    
    # Initialize distributed training if enabled
    try:
        print("Setting up distributed training...")
        rank, local_rank, world_size = setup_distributed(config)
        print(f"Distributed training setup complete: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    except Exception as e:
        print(f"ERROR: Failed to initialize distributed training: {str(e)}")
        logger.error(f"Failed to initialize distributed training: {str(e)}", exc_info=True)
        return
    
    # Create directories
    try:
        create_directories(config)
    except Exception as e:
        console_print(f"ERROR: Failed to create directories: {str(e)}", True)
        logger.error(f"Failed to create directories: {str(e)}", exc_info=True)
        return
    
    # Initialize wandb logging if configured
    if is_main_process() and getattr(config, 'use_wandb', False):
        try:
            console_print("Initializing WandB logging...")
            run_name = getattr(config, 'wandb_run_name', None) or f"ddm-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            run = init_wandb(
                config=config,
                project=getattr(config, 'wandb_project', "decentralized-diffusion"),
                name=run_name
            )
            console_print(f"WandB logging initialized: {run_name}")
            logger.info(f"Initialized WandB logging: {run_name}")
        except Exception as e:
            console_print(f"WARNING: Failed to initialize WandB: {str(e)}")
            logger.warning(f"Failed to initialize WandB: {str(e)}")
    
    try:
        # Initialize expert cache manager for memory-efficient expert loading
        console_print(f"Rank {rank}: Initializing Expert Cache Manager...")
        logger.info("Initializing Expert Cache Manager")
        cache_manager = ExpertCacheManager(
            config=config,
            device=torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"),
            max_experts=config.max_experts_in_memory,
            cpu_offload=config.expert_offload_to_cpu
        )
        console_print(f"Rank {rank}: Expert Cache Manager initialized")
        
        # Create training coordinator
        console_print(f"Rank {rank}: Creating DDM Training Coordinator - this may take several minutes...")
        console_print(f"Rank {rank}: Will initialize clustering, models, and data loaders")
        logger.info("Creating DDM Training Coordinator")
        logger.info("This will initialize clustering, models, and data loaders")
        
        if is_main_process():
            console_print("Feature extraction and clustering may take 20-30 minutes, please be patient...")
            logger.info("Feature extraction and clustering may take 20-30 minutes")
        
        # Add progress indicators during coordinator initialization
        init_start_time = time.time()
        def progress_callback(stage, progress=None):
            elapsed = time.time() - init_start_time
            if progress is not None:
                console_print(f"Initialization progress: {stage} - {progress:.1f}% complete ({elapsed:.1f}s elapsed)")
            else:
                console_print(f"Initialization stage: {stage} ({elapsed:.1f}s elapsed)")
        
        # Print progress every 30 seconds during initialization
        progress_thread_active = True
        def progress_thread():
            last_print = time.time()
            # Different stages depending on whether clustering is skipped
            if getattr(config, 'skip_clustering', False):
                console_print(f"Rank {rank}: Clustering will be skipped (skip_clustering=True)")
                console_print(f"Rank {rank}: Using uniform distribution across {config.num_experts} experts")
                stages = [
                    "fast path cluster manager initialization", 
                    "fast path data loader initialization", 
                    "router initialization", 
                    "expert initialization"
                ]
                # Fast initialization should be much quicker
                wait_time = 10  # Report progress every 10 seconds for fast path
            else:
                console_print(f"Rank {rank}: Clustering will be performed (skip_clustering=False)")
                stages = ["clustering", "router initialization", "expert initialization", "data loading"]
                wait_time = 30  # Report progress every 30 seconds for normal path
                
            stage_idx = 0
            while progress_thread_active:
                current_time = time.time()
                if current_time - last_print > wait_time:
                    elapsed = current_time - init_start_time
                    stage = stages[min(stage_idx, len(stages)-1)]
                    console_print(f"Still initializing: {stage} (elapsed: {elapsed:.1f}s)")
                    last_print = current_time
                    stage_idx = (stage_idx + 1) % len(stages)
                time.sleep(1)
        
        import threading
        progress_monitor = threading.Thread(target=progress_thread)
        progress_monitor.daemon = True
        progress_monitor.start()
        
        try:
            coordinator = DDMTrainingCoordinator(
                config=config,
                rank=rank,
                world_size=world_size,
                cache_manager=cache_manager,
                progress_callback=progress_callback
            )
        finally:
            # Stop the progress thread
            progress_thread_active = False
            progress_monitor.join(timeout=1.0)
        
        console_print(f"Rank {rank}: DDM Training Coordinator created successfully!")
        
        # Load existing checkpoint if configured
        if hasattr(config, 'resume_from') and config.resume_from:
            console_print(f"Rank {rank}: Loading checkpoint from {config.resume_from}...")
            start_step = coordinator.load_checkpoint(config.resume_from)
            console_print(f"Rank {rank}: Resuming from checkpoint at step {start_step}")
            logger.info(f"Resuming from checkpoint at step {start_step}")
        else:
            start_step = 0
        
        # Training loop
        console_print(f"Rank {rank}: Starting training for {config.num_steps} steps")
        logger.info(f"Starting training for {config.num_steps} steps")
        
        for step in range(start_step, config.num_steps):
            step_start_time = time.time()
            
            # Log progress periodically
            if is_main_process() and step % getattr(config, 'log_every_n_steps', 10) == 0:
                console_print(f"Starting training step {step}/{config.num_steps}")
                logger.info(f"Starting training step {step}/{config.num_steps}")
            
            # Expert training phase (Section 3.2)
            console_print(f"Rank {rank}: Step {step}: Training experts...", rank == 0)
            expert_loss = coordinator.train_experts(step)
            console_print(f"Rank {rank}: Step {step}: Expert training complete, loss: {expert_loss}", rank == 0)
            
            # Router training phase (Section 3.3)
            console_print(f"Rank {rank}: Step {step}: Training router...", rank == 0)
            router_loss = coordinator.train_router(step)
            console_print(f"Rank {rank}: Step {step}: Router training complete, loss: {router_loss}", rank == 0)
            
            # Log metrics
            if step % getattr(config, 'log_every_n_steps', 10) == 0:
                console_print(f"Rank {rank}: Step {step}: Logging metrics...", rank == 0)
                coordinator.log_sharded_metrics(step, expert_loss, router_loss)
                console_print(f"Rank {rank}: Step {step}: Metrics logged", rank == 0)
            
            # Reclustering if needed
            if coordinator.needs_reclustering(step):
                if is_main_process():
                    console_print(f"Step {step}: Performing scheduled reclustering...")
                    logger.info(f"Step {step}: Performing scheduled reclustering")
                coordinator.perform_reclustering()
                console_print(f"Rank {rank}: Step {step}: Reclustering complete", rank == 0)
                
                # Run validation after reclustering
                if is_main_process():
                    console_print(f"Step {step}: Running validation after reclustering...")
                    logger.info(f"Step {step}: Running validation after reclustering")
                coordinator.run_validation(step)
                console_print(f"Rank {rank}: Step {step}: Post-reclustering validation complete", rank == 0)
            
            # Regular validation
            if step % getattr(config, 'validation_interval', 1000) == 0:
                if is_main_process():
                    console_print(f"Step {step}: Running scheduled validation...")
                    logger.info(f"Step {step}: Running scheduled validation")
                coordinator.run_validation(step)
                console_print(f"Rank {rank}: Step {step}: Validation complete", rank == 0)
                
                # Run ensemble validation to validate the DDM objective
                if is_main_process():
                    console_print(f"Step {step}: Running ensemble validation...")
                    logger.info(f"Step {step}: Running ensemble validation")
                coordinator.run_ensemble_validation(step)
                console_print(f"Rank {rank}: Step {step}: Ensemble validation complete", rank == 0)
            
            # Checkpointing
            if step % getattr(config, 'save_interval', 5000) == 0 or step == config.num_steps - 1:
                if is_main_process():
                    console_print(f"Step {step}: Saving checkpoint...")
                    logger.info(f"Step {step}: Saving checkpoint")
                coordinator.save_sharded_checkpoints(step)
                console_print(f"Rank {rank}: Step {step}: Checkpoint saved", rank == 0)
            
            # Calculate and log step time
            step_time = time.time() - step_start_time
            if is_main_process() and step % getattr(config, 'log_every_n_steps', 10) == 0:
                console_print(f"Step {step} completed in {step_time:.2f}s")
                logger.info(f"Step {step} completed in {step_time:.2f}s")
        
        # Final validation and distillation
        if is_main_process():
            console_print("Training complete, running final validation...")
            logger.info("Training complete, running final validation")
            coordinator.run_validation(config.num_steps)
            
            # Run distillation if configured
            if getattr(config, 'do_distillation', True):
                console_print("Starting model distillation...")
                logger.info("Starting model distillation")
                coordinator.train_distilled_model()
            
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            console_print(f"Total training time: {time_str}")
            logger.info(f"Total training time: {total_time/3600:.1f} hours ({time_str})")
            
    except KeyboardInterrupt:
        console_print("Training interrupted - saving final checkpoint...")
        logger.info("Training interrupted - saving final checkpoint")
        try:
            coordinator.save_sharded_checkpoints(step)
            console_print("Checkpoint saved after interruption")
        except Exception as e:
            console_print(f"ERROR: Failed to save checkpoint after interruption: {str(e)}")
            logger.error(f"Error saving checkpoint after interruption: {str(e)}")
    except Exception as e:
        console_print(f"ERROR: Training failed: {str(e)}")
        logger.error(f"Error during training: {str(e)}", exc_info=True)
    finally:
        # Clean up cache manager
        if locals().get('cache_manager') is not None:
            try:
                console_print("Shutting down expert cache manager...")
                cache_manager.shutdown()
                console_print("Expert cache manager shutdown completed")
                logger.info("Expert cache manager shutdown completed")
            except Exception as e:
                console_print(f"ERROR: Failed to shut down cache manager: {str(e)}")
                logger.error(f"Error shutting down cache manager: {str(e)}")
        
        # Clean up distributed
        if torch.distributed.is_initialized():
            console_print("Destroying distributed process group...")
            torch.distributed.destroy_process_group()
            console_print("Distributed process group destroyed")
        
        if is_main_process() and getattr(config, 'use_wandb', False):
            import wandb
            console_print("Finalizing WandB logging...")
            wandb.finish()
            console_print("WandB logging finalized")
            
        console_print("Training finished")
        logger.info("Training finished")

if __name__ == "__main__":
    main() 