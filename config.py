"""Configuration utilities for Decentralized Diffusion Models."""

import os
import sys
import importlib.util
from types import SimpleNamespace
import logging
import types  # Add this import


logger = logging.getLogger(__name__)

# Add this at the top under imports
DEFAULT_CONFIG = {
    # ===== Core Architecture =====
    'hidden_size': 1152,
    'num_heads': 16,
    'depth': 30,
    'mlp_ratio': 4.0,
    'qkv_bias': True,
    'vec_in_dim': 768,
    'context_in_dim': 768,
    
    # ===== Expert Configuration =====
    'num_experts': 8,
    'num_clusters': 8,
    'cluster_embed_dim': 512,
    'expert_capacity_factor': 1.25,
    
    # ===== Training Parameters ===== 
    'batch_size': 1,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_steps': 1000,
    'use_mixed_precision': True,
    'gradient_accumulation_steps': 2,
    
    # ===== Distributed Training =====
    'use_gradient_checkpointing': True,
    'fsdp_sharding_strategy': "FULL_SHARD",
    'fsdp_auto_wrap_policy': "LAMBDA",
    
    # ===== Dataset & Bucketing =====
    'buckets': [
        (512, 512), (576, 448), 
        (448, 576), (640, 384),
        (384, 640)
    ],
    'latent_channels': 16,
    'vae_scaling_factor': 0.18215,
    
    # ===== Router Configuration =====
    'router_learning_rate': 1e-4,
    'router_temperature': 2.0,
    'router_min_temp': 0.5,
    'router_temperature_decay': 0.9997,
    
    # ===== Sampling & Validation =====
    'sampling_steps': 50,
    'cfg_scale': 7.5,
    'enable_validation': False,
    'validation_interval': 1000,
    
    # ===== Paths & Logging =====
    'output_dir': './outputs',
    'feature_cache_path': './cache',
    'wandb_enabled': True,
    'wandb_project': 'decentralized-diffusion',
    
    # ===== Positional Embeddings =====
    'position_embed_type': 'rope_2d',
    'theta': 10000,
    'axes_dim': [32, 32],
    
    # ===== Debug/Test Flags =====
    'bypass_cluster_validation': False
}

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    from types import SimpleNamespace
    
    # Handle the case where d might be None
    if d is None:
        return None
        
    # Convert dict to namespace
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def get_config(config_path=None):
    """Load config from a Python file or create default config"""
    if config_path:
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
        
        # Add this filter when loading config attributes
        for key in dir(config_module):
            val = getattr(config_module, key)
            if not key.startswith('_') and not isinstance(val, types.ModuleType):
                setattr(config, key, val)
        
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
        
        # Convert all nested dictionaries to SimpleNamespace
        for key in dir(config_module):
            if not key.startswith('_'):
                val = getattr(config_module, key)
                if isinstance(val, dict):
                    setattr(config, key, dict_to_namespace(val))
        
        return config
    else:
        # For default config, convert the distributed dict
        defaults = DEFAULT_CONFIG
        defaults['distributed'] = dict_to_namespace(defaults['distributed'])
        
        # Add all default values
        config = SimpleNamespace()
        for key, value in defaults.items():
            setattr(config, key, dict_to_namespace(value) if isinstance(value, dict) else value)
        
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

