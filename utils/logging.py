"""Logging utilities for Decentralized Diffusion Models."""

import logging
import os
import sys
import time
from datetime import datetime
import json
import torch.distributed as dist
import torch
import wandb

def setup_logger(name, output_dir=None, log_to_console=True, level=logging.INFO, rank=0, world_size=1):
    """
    Initialize logger with proper handlers, considering distributed rank.
    Only rank 0 will have console and file handlers by default.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level) # Set base level for the logger

    # Clear existing handlers to prevent duplicates during re-runs or testing
    logger.handlers = []

    # Determine if this rank should handle logging output
    is_logging_rank = (rank == 0)

    # Prevent propagation to avoid duplicate messages if using root logger elsewhere
    logger.propagate = False

    # Console handler (only for the designated logging rank)
    if log_to_console and is_logging_rank:
        console_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
        # Set formatter for clearer logs (optional)
        formatter = logging.Formatter(f'%(asctime)s - Rank {rank} - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        # Set the level for the handler (e.g., INFO for console)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    # File handler (only for the designated logging rank)
    if output_dir and is_logging_rank:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"rank_{rank}_{name}.log") # Include rank in filename
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        # Set formatter (optional, can be more detailed for file)
        file_formatter = logging.Formatter(f'%(asctime)s - Rank {rank} - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        # Set the level for the file handler (e.g., DEBUG for file)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    # For non-logging ranks, add a NullHandler to prevent "No handler found" warnings
    if not is_logging_rank:
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
        # Optionally set a higher level to completely silence non-logging ranks
        # logger.setLevel(logging.CRITICAL + 1)


    return logger

def setup_distributed_logger(name=None, level=logging.INFO, log_file=None, rank=0):
    """
    Set up a logger that only logs on the specified rank
    
    Args:
        name: Logger name (None for root logger)
        level: Logging level (default: INFO)
        log_file: Optional file path to save logs
        rank: Process rank (only logs on rank 0 by default)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure logging on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        logger.setLevel(logging.WARNING)  # Set high threshold for non-rank-0 processes
        return logger
    
    return setup_logger(name, level, log_file)

def init_wandb(config, project_name="decentralized-diffusion", run_name=None, rank=0):
    """
    Initialize Weights & Biases logging
    
    Args:
        config: Configuration object
        project_name: W&B project name
        run_name: Run name (defaults to timestamp)
        rank: Process rank (only initializes on rank 0)
        
    Returns:
        wandb run object if initialized, None otherwise
    """
    # Only initialize on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return None
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"ddm-{timestamp}"
    
    # Convert config to dict if it's an object
    if hasattr(config, '__dict__'):
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    else:
        config_dict = config
    
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        reinit=True
    )
    
    return run

def log_metrics(metrics, step=None, prefix=None, rank=0):
    """
    Log metrics to wandb
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
        prefix: Optional prefix for metric names
        rank: Process rank (only logs on rank 0 by default)
    """
    # Only log on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return
    
    # Add prefix to metric names if specified
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log(metrics, step=step)

def log_images(images, captions=None, step=None, prefix="samples", rank=0):
    """
    Log images to wandb
    
    Args:
        images: Tensor or list of images
        captions: Optional list of captions
        step: Optional step number
        prefix: Prefix for image names
        rank: Process rank (only logs on rank 0 by default)
    """
    # Only log on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return
    
    # Check if wandb is initialized
    if wandb.run is None:
        return
    
    # Convert tensor to list of images
    if torch.is_tensor(images):
        # Ensure images are in range [0, 1]
        if images.min() < 0:
            images = (images + 1) / 2
        
        # Convert to numpy
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype('uint8')
    
    # Create wandb image objects
    wandb_images = [wandb.Image(img, caption=cap) for img, cap in 
                    zip(images, captions if captions else [None] * len(images))]
    
    # Log images
    wandb.log({prefix: wandb_images}, step=step)

def log_model_summary(model, input_size, batch_size=1, device='cuda', rank=0):
    """
    Log model summary to wandb
    
    Args:
        model: PyTorch model
        input_size: Input size (excluding batch dimension)
        batch_size: Batch size
        device: Device to run summary on
        rank: Process rank (only logs on rank 0 by default)
    """
    # Only log on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return
    
    # Check if wandb is initialized
    if wandb.run is None:
        return
    
    try:
        from torchinfo import summary
        
        # Generate model summary
        model_summary = summary(
            model,
            input_size=(batch_size, *input_size),
            device=device,
            verbose=0
        )
        
        # Log summary to wandb
        wandb.run.summary["model_summary"] = str(model_summary)
    except ImportError:
        print("torchinfo not installed. Install with: pip install torchinfo")

def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create a directory for the experiment with timestamp
    
    Args:
        base_dir: Base directory
        experiment_name: Optional experiment name
        
    Returns:
        Path to created directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create experiment name if not provided
    if experiment_name is None:
        experiment_name = f"experiment_{timestamp}"
    else:
        experiment_name = f"{experiment_name}_{timestamp}"
    
    # Create directory
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir

def setup_distributed_printer(rank=0):
    """
    Set up a printer that only prints on the specified rank
    
    Args:
        rank: Process rank to print on (default: 0)
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == rank:
            builtin_print(*args, **kwargs)
            
    __builtin__.print = print

def save_config(config, save_path):
    """
    Save configuration to a JSON file
    
    Args:
        config: Configuration object or dictionary
        save_path: Path to save config to
    """
    # Convert config to dict if it's an object
    if hasattr(config, '__dict__'):
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    else:
        config_dict = config
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save config
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_config(load_path):
    """
    Load configuration from a JSON file
    
    Args:
        load_path: Path to load config from
        
    Returns:
        Loaded configuration as dictionary
    """
    # Check if file exists
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Config file not found at {load_path}")
    
    # Load config
    with open(load_path, 'r') as f:
        config = json.load(f)
    
    return config

def log_training_start(logger, config, rank=0):
    """
    Log training start information
    
    Args:
        logger: Logger to use
        config: Configuration object
        rank: Process rank (only logs on rank 0 by default)
    """
    # Only log on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return
    
    # Log start message
    logger.info("=" * 80)
    logger.info("Starting Decentralized Diffusion Models training")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            logger.info(f"  {key}: {value}")
    
    # Log CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Log distributed information
    if dist.is_initialized():
        logger.info(f"Distributed setup: {dist.get_world_size()} processes")
        logger.info(f"Distributed backend: {dist.get_backend()}")
    
    logger.info("=" * 80)

def log_training_end(logger, start_time, rank=0):
    """
    Log training end information
    
    Args:
        logger: Logger to use
        start_time: Training start time
        rank: Process rank (only logs on rank 0 by default)
    """
    # Only log on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Log end message
    logger.info("=" * 80)
    logger.info("Training complete")
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info("=" * 80)

# Remove the global logger instance creation
# logger = setup_logger("DDM") # <--- REMOVE THIS LINE

# ... (rest of the functions like init_wandb, log_metrics, etc. remain the same) ...
# Note: Functions like log_training_start already accept a logger instance, which aligns with the new approach.
# Functions using wandb directly are fine as wandb handles rank internally. 