"""Configuration utilities for Decentralized Diffusion Models."""

import os
import sys
import importlib.util
from types import SimpleNamespace
import logging
import torch
# Add import for torchinfo and check availability
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

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
    
    # ===== CLIP Configuration =====
    'clip_model': "openai/clip-vit-large-patch14",
    'clip_embedding_dim': 768,  # ViT-L/14 output dim
    'max_token_length': 77,     # CLIP standard

    # ===== Flux MMDiT Architecture =====
    'hidden_size': 768,          # Transformer hidden size
    'in_channels': 3,
    'out_channels': 16,
    'num_heads': 12,            # Attention heads per layer
    'num_layers': 24,           # Total transformer blocks
    'patch_size': 2,            # Input patch dimension 
    'mlp_ratio': 4.0,           # FFN expansion factor
    'qkv_bias': True,           # Enable QKV projection biases
    'qk_rmsnorm': True,         # Use RMSNorm for Q/K projections

    # ===== Expert Configuration =====
    'num_experts': 8,           # Number of data clusters
    'cluster_embed_dim': 256,   # Expert specialization dimension
    'expert_specialization': 'timestep',  # Cluster conditioning type
    
    # ===== VAE Configuration =====
    'vae_model': "AuraDiffusion/16ch-vae",
    'latent_channels': 16,      # VAE output channels
    'vae_scaling_factor': 0.18215,

    # ===== Router Network =====
    'router_hidden_size': 512,
    'router_num_heads': 8,
    'router_num_layers': 4,
    
    # ===== Dataset parameters =====
    'dataset_path': '/home/alex/workspace/datasets/danbooru2025',
    'feature_cache_path': '/home/alex/workspace/Decentralized-Diffusion-Models/cache',
    'dataset_size': 380000,
    'min_size': 256,  # Minimum image dimension
    'max_size': 1024,  # Maximum image dimension
    'val_size': 1000,
    'buckets': [
        (512, 512),   # Square
        (576, 448),   # Landscape 
        (448, 576),   # Portrait
        (640, 384),   # Wide landscape
        (384, 640),   # Tall portrait
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

    'fast_validation': True,    # Use fewer steps for validation
    'sampling_steps': 20,       # Default number of sampling steps for validation
    
    # ===== Training controls =====
    'enable_validation': False,
    'enable_sampling': False,
    'enable_checkpointing': True,
    
    # Add to DEFAULT_CONFIG
    'dynamic_expert_count': 2,
    'expert_selection_strategy': 'top_k',
    'max_sampling_experts': 2,  # Maximum experts to use in sampling
    'max_experts_in_memory': 2,  # Number of experts to keep in GPU memory
    'num_fine_clusters': 1024,  # For initial KMeans clustering
    'min_cluster_samples': 50000,  # Minimum samples per expert cluster
    'kmeans_restarts': 3,  # Number of KMeans restarts
    'cluster_linkage': 'average',  # Hierarchical clustering method
    
    # ===== Shape Test Controls =====
    'bypass_cluster_validation': False, # Flag to bypass cluster size validation, set to True in shape_test.py
    'use_cuda_graphs': False,

    # Add to DEFAULT_CONFIG
    'bucket_scale': 64,  # Base scaling factor for buckets
    'min_bucket_dim': 256,  # Minimum dimension for any bucket
    'bucket_thresholds': {  # Aspect ratio groupings
        'square': (0.9, 1.1),
        'portrait': (0.4, 0.9), 
        'landscape': (1.1, 2.5)
    },
}

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
    else:
        return create_default_config()

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

# Modified function to estimate and print model size - now accepts model_type
def estimate_model_size(config, model_type="expert"):
    """
    Estimates and prints the size of the specified model type, then clears memory.

    Args:
        config: Configuration object.
        model_type: "expert" or "router" to specify which model size to estimate.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_model = None
    dummy_input = None  # Initialize dummy_input here

    if model_type == "expert":
        from models.mmdit import ExpertMMDiT
        dummy_model = ExpertMMDiT(config).to(device)
        model_name = "ExpertMMDiT"
        # Add expert-specific dummy input
        dummy_input = [
            torch.randn(
                config.batch_size,
                config.latent_channels,
                config.image_size[1] // config.patch_size,
                config.image_size[2] // config.patch_size,
                device=device
            ),
            torch.randint(0, 1000, (config.batch_size,), device=device),
            torch.randn(
                config.batch_size, 
                config.max_token_length,
                config.clip_embedding_dim,
                device=device
            )
        ]
    elif model_type == "router":
        from models.router import RouterModel
        dummy_model = RouterModel(config).to(device)
        model_name = "RouterModel"
        dummy_input = [
            torch.randn(
                config.batch_size,
                config.latent_channels,
                config.image_size[1] // config.patch_size,
                config.image_size[2] // config.patch_size,
                device=device
            ),
            torch.randint(0, 1000, (config.batch_size,), device=device),
            torch.randn(
                config.batch_size, 
                config.max_token_length,
                config.clip_embedding_dim,
                device=device
            )
        ]
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'expert' or 'router'.")

    if TORCHINFO_AVAILABLE:
        model_summary = summary(
            dummy_model,
            dtypes=[torch.float32],
            device=device,
            verbose=0,
            input_data=dummy_input,  # Now properly defined for both cases
        )
        total_params = model_summary.total_params
        trainable_params = model_summary.trainable_params
        # Conditional check for non_trainable_params
        if hasattr(model_summary, 'non_trainable_params'):
            non_trainable_params = model_summary.non_trainable_params
        else:
            non_trainable_params = total_params - trainable_params # Calculate manually if not available
        param_size_mb = model_summary.total_param_bytes / (1024**2) # Use total_param_bytes

        print(f"===================== {model_name} Size Summary =====================") # Model name in summary
        print(f"Total params:        {total_params:,}")
        print(f"Trainable params:    {trainable_params:,}")
        print(f"Non-trainable params:{non_trainable_params:,}")
        print(f"Model size (MB):     {param_size_mb:.2f}")
        print("===============================================================")

    else:
        total_params = sum(p.numel() for p in dummy_model.parameters())
        print(f"torchinfo not available. Estimating parameter count manually for {model_name}.") # Model name in message
        print(f"Total parameters (approximate): {total_params:,}")
        print("Please install torchinfo for a detailed model summary.")

    # Cleanup: Move model to CPU and delete to free memory
    dummy_model.to('cpu') # Move model to CPU
    del dummy_model # Delete model object
    if device == 'cuda':
        torch.cuda.empty_cache() # Clear CUDA cache

    print("Model unloaded and memory cleared.") # Inform user about cleanup

