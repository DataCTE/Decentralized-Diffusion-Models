"""Configuration utilities for Decentralized Diffusion Models."""

import os
import sys
import importlib.util
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)

# Add this at the top under imports
DEFAULT_CONFIG = {
    # ===== Training parameters =====
    'num_steps': 1000000,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0,
    'use_mixed_precision': True,
    'resume_checkpoint': None,  # Add resume checkpoint path (None = don't resume)
    
    # ===== DDM Model parameters =====
    'num_experts': 8,
    'ffn_dim': 3072,    # 4x hidden_dim
    'hidden_size': 768,
    'hidden_dim': 768,  # ExpertMMDiT uses hidden_dim instead of hidden_size
    'num_heads': 12,
    'num_layers': 12,
    'patch_size': 16,

    # ===== VAE and CLIP parameters =====
    'vae_model': "AuraDiffusion/16ch-vae",
    'latent_channels': 16,  # Specific to 16ch-VAE
    'vae_scaling_factor': 0.18215,  # Seen in data/vae.py
    'clip_model': "openai/clip-vit-large-patch14",
    'max_token_length': 77,  # Standard CLIP token length
    'clip_embedding_dim': 768, # Assuming CLIP embedding dim is 768 for ViT-Large-Patch14
    
    
    # ===== Router parameters =====
    'router_hidden_size': 512,
    'router_num_heads': 8,
    'router_num_layers': 4,
    'router': {
        'input_dim': 768,  # Should match your text encoder output dimension
        'hidden_dim': 512,
        'output_dim': 8,   # Should match number of experts
        'num_layers': 3
    },
    
    # ===== Dataset parameters =====
    'dataset_path': '/home/alex/workspace/datasets/danbooru2025',
    'dataset_size': 380000,
    'min_size': 256,  # Minimum image dimension
    'max_size': 1024,  # Maximum image dimension
    'val_size': 1000,
    'buckets': [
        (512, 512),   # Square format
        (576, 448),   # Landscape
        (448, 576),   # Portrait
        (640, 384),   # Wide
        (384, 640),   # Tall
    ],
    'validation_batch_size': 4,  # Reduced from 1000 to prevent OOM
    'num_workers': 8,  # Reduced from 8 to prevent I/O bottlenecks
    'pin_memory': False,  # Enable pin_memory for faster data transfer
    'persistent_workers': True,
    'prefetch_factor': 2,
    'broadcast_batch_size': 1000,  # Control broadcast chunk size
    
    # ===== Training parameters =====
    'adam_betas': (0.9, 0.999),
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    
    # ===== Flow Matching parameters =====
    'diffusion_steps': 1000,
    'sigma': 0.5,
    'loss_type': 'huber',
    'beta_schedule': 'cosine',
    'flow_matching_delta': 0.1,  # Delta parameter for Huber loss
    
    # ===== Sampling parameters =====
    'sampling_steps': 50,
    'cfg_scale': 7.5,
    'top_k': 1,
    'temperature': 1.0,
    'eta': 0.0,
    
    # ===== Output paths =====
    'output_dir': './outputs',
    
    # ===== Logging parameters =====
    'log_every': 1,
    'save_every': 5000,
    'validate_every': 1000,
    'generate_every': 1000,
    'sample_count': 4,
    'checkpoint_interval': 5000,  # How often to save checkpoints
    'validation_interval': 1000,  # How often to run validation
    
    # ===== Distributed training =====
    'save_from_all_ranks': False,
    
    # Changed from cluster-based parameters
    'expert_batch_size': 1,
    
    # ===== Expert Cache parameters =====
    'max_experts_in_memory': 2,  # Number of experts to keep in GPU memory
    'expert_offload_to_cpu': True,  # Whether to offload unused experts to CPU
    
    # Add to DEFAULT_CONFIG
    'use_gradient_checkpointing': True,
    'fsdp_sharding_strategy': "FULL_SHARD",
    'fsdp_cpu_offload': False,
    'fsdp_backward_prefetch': "BACKWARD_PRE",
    'fsdp_auto_wrap_policy': "LAMBDA",
    'fsdp_min_num_params': 1e6,
    'router_learning_rate': 1e-4,
    'fsdp_use_orig_params': True,
    'fsdp_limit_all_gathers': True,
    
    # ===== Distillation parameters =====
    'ema_decay': 0.9999,  # Exponential moving average decay factor for model weights
    
    # ===== Distributed Training ===== (simplified)
    'batch_size_per_gpu': 1,  # Batch size per GPU
    'gradient_accumulation_steps': 1,  # Accumulate gradients
    
    # ===== W&B Logging Parameters =====
    'wandb_enabled': True,                    # Whether to use wandb for logging
    'wandb_project': 'decentralized-diffusion', # Project name
    'wandb_entity': None,                     # Username or team name, None for default
    'wandb_group': None,                      # Group related runs together
    'wandb_name': None,                       # Run name, None for auto-generated name
    'wandb_id': None,                         # Run ID for resuming, None for new run
    'wandb_dir': './wandb',                   # Directory for local files
    'wandb_tags': [],                         # List of tags for the run
    'wandb_mode': 'online',                   # Options: online, offline, disabled
    'wandb_save_code': False,                  # Save code snapshot with run
    'wandb_watch_model': 'gradients',         # Options: gradients, parameters, all, None
    'wandb_log_every': 1,                     # Log every step (not every N steps)
    'wandb_log_artifacts': False,              # Save model checkpoints as artifacts
    'wandb_log_batch_metrics': False,         # Log per-batch metrics (more overhead)
    'wandb_log_memory': False,                 # Log memory usage
    'wandb_commit_frequency': 1,   # Frequency to commit logs (N steps)
    'max_sampling_experts': 4,  # Maximum experts to use in sampling
    'fast_validation': True,    # Use fewer steps for validation
    'sampling_steps': 20,       # Default number of sampling steps for validation
    
    # ===== Training controls =====
    'enable_validation': False,
    'enable_sampling': False,
    'enable_checkpointing': True,
    
    # Add to DEFAULT_CONFIG
    'expert_specialization': 'timestep',  # or 'text' for conditional models
    'dynamic_expert_count': 4,
    'expert_selection_strategy': 'top_k',
    
    # ===== Shape Test Controls =====
    'bypass_cluster_validation': True, # Flag to bypass cluster size validation, set to True in shape_test.py
}

def get_config(config_path):
    """Load config from a Python file."""
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the module dynamically
    config_name = os.path.basename(config_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config_module
    spec.loader.exec_module(config_module)
    
    # Create a namespace for the config
    config = SimpleNamespace()
    
    # Add all non-hidden attributes from the config module
    for key in dir(config_module):
        if not key.startswith('_'):
            setattr(config, key, getattr(config_module, key))
    
    # Add default values for any missing required fields
    defaults = DEFAULT_CONFIG
    
    # Add defaults only if not already set
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    # Create required directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
    
    # Derive image_size from the first bucket if not explicitly set
    if not hasattr(config, 'image_size') and hasattr(config, 'buckets') and config.buckets:
        w, h = config.buckets[0]
        config.image_size = (3, h, w)
        logger.info(f"Derived image_size {config.image_size} from first bucket")
    
    return config

def create_default_config():
    """Create a default configuration."""
    # Start with an empty config
    config = SimpleNamespace()
    
    # Use the same defaults dictionary from get_config
    defaults = DEFAULT_CONFIG
    
    # Add all default values
    for key, value in defaults.items():
        setattr(config, key, value)
    
    # Derive image_size from the first bucket
    if hasattr(config, 'buckets') and config.buckets:
        w, h = config.buckets[0]
        config.image_size = (3, h, w)
    
    # Create required directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
    
    return config

def save_config(config, path):
    """Save configuration to a Python file."""
    with open(path, 'w') as f:
        f.write('"""Generated DDM configuration."""\n\n')
        
        for key in sorted(dir(config)):
            if not key.startswith('_'):
                value = getattr(config, key)
                # Handle different types appropriately
                if isinstance(value, str):
                    f.write(f'{key} = "{value}"\n')
                else:
                    f.write(f'{key} = {value}\n')
    
    logger.info(f"Saved configuration to {path}")