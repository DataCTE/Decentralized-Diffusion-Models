"""Configuration utilities for Decentralized Diffusion Models."""

import os
import sys
import importlib.util
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)

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
    defaults = {
        # ===== Training parameters =====
        'num_steps': 100000,              # Total number of training steps
        'batch_size': 1,                 # Batch size per GPU
        'learning_rate': 1e-4,            # Learning rate for both experts and router
        'weight_decay': 1e-2,             # Weight decay for regularization
        'warmup_steps': 1000,             # Learning rate warmup steps
        'max_grad_norm': 1.0,             # Gradient clipping norm
        'use_mixed_precision': True,      # Whether to use mixed precision training
        
        # ===== DDM Model parameters =====
        'num_experts': 8,                 # Should match number of GPUs
        'hidden_size': 768,               # Hidden dimension for transformer models
        'num_heads': 12,                  # Number of attention heads
        'num_layers': 12,                 # Number of transformer layers
        
        # ===== Router parameters =====
        'router_hidden_size': 512,        # Hidden dimension for router
        'router_num_heads': 8,            # Number of attention heads for router
        'router_num_layers': 4,           # Number of transformer layers for router
        
        # ===== Dataset parameters =====
        'dataset_path': '/home/alex/workspace/datasets/danbooru2025',         # Path to dataset
        'dataset_size': 100000,           # Size of dataset for uniform distribution
        'val_size': 1000,                 # Size of validation dataset
        'buckets': [                      # Multiple aspect ratio buckets (W, H)
            (512, 512),                   # Square 1:1
            (576, 448),                   # Landscape 4:3
            (448, 576),                   # Portrait 3:4
            (640, 384),                   # Landscape 16:9
            (384, 640),                   # Portrait 9:16
        ],
        'num_workers': 4,                 # Number of workers for data loading
        
        # ===== Flow Matching parameters =====
        'diffusion_steps': 1000,          # Number of diffusion timesteps
        'sigma': 0.5,                     # Flow matching sigma parameter
        'loss_type': 'huber',             # Loss function ('mse', 'huber', 'l1')
        'beta_schedule': 'cosine',        # Noise schedule ('cosine', 'linear')
        
        # ===== Sampling parameters =====
        'sampling_steps': 50,             # Diffusion sampling steps
        'cfg_scale': 7.5,                 # Classifier-free guidance scale
        'top_k': 1,                       # Number of experts to use per sample
        'temperature': 1.0,               # Temperature for router softmax
        'eta': 0.0,                       # DDIM eta parameter (0.0 = deterministic)
        
        # ===== Output paths =====
        'output_dir': './outputs',        # Main output directory
        
        # ===== Logging parameters =====
        'log_every': 100,                 # Log training metrics every N steps
        'save_every': 5000,               # Save checkpoint every N steps
        'validate_every': 1000,           # Run validation every N steps
        'generate_every': 1000,           # Generate samples every N steps
        'sample_count': 4,                # Number of samples to generate during training
        
        # ===== Distributed training =====
        'save_from_all_ranks': False,     # Whether to save checkpoints from all ranks
        
        # Changed from cluster-based parameters
        'expert_batch_size': 1,           # Batch size per expert
    }
    
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
    
    # Add all default values
    defaults = get_config.__defaults__[0]
    for key, value in defaults.items():
        setattr(config, key, value)
    
    # Derive image_size from the first bucket
    if hasattr(config, 'buckets') and config.buckets:
        w, h = config.buckets[0]
        config.image_size = (3, h, w)
    
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