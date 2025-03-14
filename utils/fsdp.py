"""FSDP utilities for Decentralized Diffusion Models."""

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, 
    CPUOffload, 
    MixedPrecision,
    BackwardPrefetch
)
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
)
import logging

logger = logging.getLogger(__name__)

def get_fsdp_defaults():
    """Paper-specified isolation defaults"""
    return {
        "process_group": torch.distributed.new_group(backend="nccl"),  # Separate group per expert
        "sync_module_states": False,  # Don't sync initial weights
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,
        "use_orig_params": True
    }

def create_fsdp_config(config, sharding_strategy="FULL_SHARD"):
    """Paper's sharding strategies with config overrides"""
    defaults = get_fsdp_defaults()
    return {
        **defaults,
        "sharding_strategy": {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
            "NO_SHARD": ShardingStrategy.NO_SHARD
        }[sharding_strategy],
        "device_id": torch.cuda.current_device()
    }

def create_mixed_precision_config(config):
    """Create mixed precision config for FSDP"""
    # Default to bfloat16 if available, otherwise fall back to float16
    bf16_ready = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    if config.use_mixed_precision:
        if bf16_ready and getattr(config, 'use_bfloat16', False):
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )
        else:
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
    return None

def get_auto_wrap_policy(config):
    """Get auto wrap policy based on config"""
    if hasattr(config, 'fsdp_auto_wrap_policy'):
        if config.fsdp_auto_wrap_policy == "SIZE_BASED":
            min_params = getattr(config, 'fsdp_min_num_params', 1e6)
            return size_based_auto_wrap_policy(min_num_params=min_params)
        elif config.fsdp_auto_wrap_policy == "TRANSFORMER":
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block
            from transformers.models.bert.modeling_bert import BertLayer
            # Add any other transformer layers that might be used
            transformer_layer_cls = [GPT2Block, BertLayer]
            return transformer_auto_wrap_policy(transformer_layer_cls=transformer_layer_cls)
    
    # Default to size-based policy
    return default_auto_wrap_policy

def get_backward_prefetch(config):
    """Get backward prefetch setting based on config"""
    if hasattr(config, 'fsdp_backward_prefetch'):
        if config.fsdp_backward_prefetch == "BACKWARD_PRE":
            return BackwardPrefetch.BACKWARD_PRE
        elif config.fsdp_backward_prefetch == "BACKWARD_POST":
            return BackwardPrefetch.BACKWARD_POST
    
    # Default prefetch setting
    return BackwardPrefetch.BACKWARD_PRE

def wrap_model_with_fsdp(model, config, param_init_fn=None):
    """
    Wrap model with FSDP using configuration from config object
    
    Args:
        model: Base model to wrap
        config: Configuration object with FSDP settings
        param_init_fn: Optional function to initialize parameters (for expert isolation)
        
    Returns:
        FSDP-wrapped model
    """
    # Get FSDP configuration settings from config
    sharding_strategy = getattr(config, 'fsdp_sharding_strategy', "FULL_SHARD")
    cpu_offload = getattr(config, 'fsdp_cpu_offload', False)
    mixed_precision = getattr(config, 'use_mixed_precision', False)
    
    # Create FSDP configuration
    fsdp_config = create_fsdp_config(config, sharding_strategy)
    
    # Add CPU offload if enabled
    if cpu_offload:
        fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)
    
    # Add mixed precision if enabled
    if mixed_precision:
        fsdp_config["mixed_precision"] = create_mixed_precision_config(config)
    
    # Add backward prefetch
    fsdp_config["backward_prefetch"] = get_backward_prefetch(config)
    
    # Add auto wrap policy
    fsdp_config["auto_wrap_policy"] = get_auto_wrap_policy(config)
    
    # Add parameter initialization function if provided
    if param_init_fn is not None:
        fsdp_config["param_init_fn"] = param_init_fn
    
    # Wrap model with FSDP
    wrapped_model = FSDP(model, **fsdp_config)
    
    logger.info(f"Model wrapped with FSDP using {sharding_strategy} strategy")
    
    return wrapped_model

def save_fsdp_model(model, save_path, optim=None, scheduler=None, metadata=None):
    """
    Save FSDP model state dict
    
    Args:
        model: FSDP-wrapped model
        save_path: Path to save checkpoint
        optim: Optional optimizer
        scheduler: Optional scheduler
        metadata: Optional metadata dictionary
    """
    from utils.checkpoint import save_model_checkpoint
    return save_model_checkpoint(
        model=model,
        optimizer=optim,
        scheduler=scheduler, 
        path=save_path,
        metadata=metadata,
        is_fsdp=True
    )
    
def load_fsdp_model(model, load_path, optim=None, scheduler=None, device=None):
    """
    Load FSDP model state dict
    
    Args:
        model: FSDP-wrapped model
        load_path: Path to load checkpoint from
        optim: Optional optimizer
        scheduler: Optional scheduler
        device: Optional device to load to
        
    Returns:
        Loaded metadata
    """
    from utils.checkpoint import load_model_checkpoint
    return load_model_checkpoint(
        model=model,
        path=load_path,
        optimizer=optim,
        scheduler=scheduler,
        is_fsdp=True,
        device=device
    )