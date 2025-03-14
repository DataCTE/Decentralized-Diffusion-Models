"""FSDP utilities for Decentralized Diffusion Models."""

import torch
import torch.distributed as dist
import logging
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    lambda_auto_wrap_policy
)
from torch.distributed.fsdp import StateDictType

logger = logging.getLogger(__name__)

def get_sharding_strategy(strategy_name):
    """Get sharding strategy from name"""
    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    
    if strategy_name not in strategies:
        logger.warning(f"Unknown sharding strategy: {strategy_name}, using FULL_SHARD")
        return ShardingStrategy.FULL_SHARD
        
    return strategies[strategy_name]

def get_backward_prefetch(prefetch_name):
    """Get backward prefetch strategy from name"""
    prefetches = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    
    if prefetch_name not in prefetches:
        logger.warning(f"Unknown backward prefetch: {prefetch_name}, using BACKWARD_PRE")
        return BackwardPrefetch.BACKWARD_PRE
        
    return prefetches[prefetch_name]

def get_mixed_precision_config(config):
    """Create mixed precision config based on configuration"""
    if not hasattr(config, 'use_mixed_precision') or not config.use_mixed_precision:
        return None
        
    # Define param, buffer, and reduce dtypes
    param_dtype = torch.float16 if not hasattr(config, 'mp_param_dtype') else getattr(torch, config.mp_param_dtype)
    reduce_dtype = torch.float16 if not hasattr(config, 'mp_reduce_dtype') else getattr(torch, config.mp_reduce_dtype)
    buffer_dtype = torch.float16 if not hasattr(config, 'mp_buffer_dtype') else getattr(torch, config.mp_buffer_dtype)
    
    # Create mixed precision config
    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype
    )

def get_auto_wrap_policy(config):
    """Create auto wrap policy based on configuration"""
    policy_name = getattr(config, 'fsdp_auto_wrap_policy', 'SIZE_BASED')
    
    if policy_name == 'SIZE_BASED':
        min_params = getattr(config, 'fsdp_min_num_params', 1e6)
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_params
        )
    elif policy_name == 'TRANSFORMER':
        # Import model-specific layers for transformer wrapping
        from models.dit import DiTBlock
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                DiTBlock,  # Wrap DiT blocks
            }
        )
    else:
        logger.warning(f"Unknown auto wrap policy: {policy_name}, using SIZE_BASED")
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=getattr(config, 'fsdp_min_num_params', 1e6)
        )

def get_param_init_fn(device):
    """Create parameter initialization function for FSDP"""
    def init_fn(module):
        # Move module to specified device but without recursing
        # This avoids excessive memory usage during initialization
        module.to_empty(device=device, recurse=False)
        return module
    return init_fn

def create_fsdp_model(model, config, rank=0):
    """Create FSDP-wrapped model from configuration"""
    # Check if distributed is initialized
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning unwrapped model")
        return model
        
    # Create sharding strategy
    sharding_strategy = get_sharding_strategy(
        getattr(config, 'fsdp_sharding_strategy', 'FULL_SHARD')
    )
    
    # Create backward prefetch
    backward_prefetch = get_backward_prefetch(
        getattr(config, 'fsdp_backward_prefetch', 'BACKWARD_PRE')
    )
    
    # Create CPU offload
    cpu_offload = CPUOffload(
        offload_params=getattr(config, 'fsdp_cpu_offload', False)
    )
    
    # Create mixed precision config
    mixed_precision = get_mixed_precision_config(config)
    
    # Create auto wrap policy
    auto_wrap_policy = get_auto_wrap_policy(config)
    
    # Create parameter initialization function
    device = torch.device(f"cuda:{rank}")
    param_init_fn = get_param_init_fn(device)
    
    # Create FSDP model
    fsdp_model = FSDP(
        model,
        device_id=rank,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=backward_prefetch,
        cpu_offload=cpu_offload,
        mixed_precision=mixed_precision,
        param_init_fn=param_init_fn,
        use_orig_params=getattr(config, 'fsdp_use_orig_params', True),
        limit_all_gathers=getattr(config, 'fsdp_limit_all_gathers', True)
    )
    
    logger.info(f"Created FSDP model with strategy={sharding_strategy}")
    
    return fsdp_model

def apply_activation_checkpointing(model, config):
    """Apply activation checkpointing to model"""
    if not hasattr(config, 'fsdp_activation_checkpointing') or not config.fsdp_activation_checkpointing:
        return model
        
    try:
        # Import model-specific modules for checkpointing
        from models.dit import DiTBlock
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing as apply_ac
        )
        
        # Use REENTRANT checkpointing which is more memory efficient
        checkpoint_impl = CheckpointImpl.REENTRANT
        
        # Define a custom checkpoint_wrapper function with more efficient memory handling
        def custom_checkpoint_wrapper(module, **kwargs):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=checkpoint_impl,
                # Use torch.utils.checkpoint configs for better memory management
                checkpoint_kwargs={
                    "preserve_rng_state": False,  # More memory efficient
                    "use_reentrant": True,       # Prevents memory leaks
                },
                **kwargs
            )
        
        # Identify the largest modules to checkpoint for better memory savings
        def check_fn(submodule):
            return isinstance(submodule, DiTBlock)
        
        # Apply checkpointing with improved memory management
        apply_ac(
            model,
            checkpoint_wrapper_fn=custom_checkpoint_wrapper,
            check_fn=check_fn
        )
        
        logger.info(f"Applied activation checkpointing to FSDP model with {checkpoint_impl} implementation")
    except Exception as e:
        logger.error(f"Failed to apply activation checkpointing: {str(e)}")
        
    return model

def fsdp_sync_module_states(model):
    """Sync module states across ranks"""
    # Only relevant for FSDP models
    if not isinstance(model, FSDP):
        return
        
    try:
        model._sync_module_states()
        torch.cuda.synchronize()
        logger.info("Synchronized FSDP module states")
    except Exception as e:
        logger.error(f"Failed to sync module states: {str(e)}")

def fsdp_summon_full_params(model, writeback=True, recurse=True):
    """Context manager for accessing full parameters"""
    # Only relevant for FSDP models
    if not isinstance(model, FSDP):
        return model.parameters()
        
    try:
        return model.summon_full_params(writeback=writeback, recurse=recurse)
    except Exception as e:
        logger.error(f"Failed to summon full params: {str(e)}")
        # Return regular parameters as fallback
        return model.parameters()

def fsdp_get_rank_zero_params(model):
    """Get parameters from rank zero for saving"""
    # Only relevant for FSDP models
    if not isinstance(model, FSDP):
        return model.state_dict()
        
    try:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, StateDictConfig(offload_to_cpu=True)):
            state_dict = model.state_dict()
        return state_dict
    except Exception as e:
        logger.error(f"Failed to get rank zero params: {str(e)}")
        # Return regular state dict as fallback
        return model.state_dict()

def fsdp_broadcast_params(model, src_rank=0):
    """Broadcast parameters from source rank to all other ranks"""
    # Only relevant for FSDP models
    if not isinstance(model, FSDP):
        return
        
    try:
        FSDP.broadcast_params(model, src_rank=src_rank)
        logger.info(f"Broadcast parameters from rank {src_rank}")
    except Exception as e:
        logger.error(f"Failed to broadcast params: {str(e)}")

def configure_optimizer_for_fsdp(model, optimizer_class, **kwargs):
    """Configure optimizer for FSDP model to avoid duplicate parameters"""
    # Only relevant for FSDP models
    if not isinstance(model, FSDP):
        return optimizer_class(model.parameters(), **kwargs)
        
    try:
        # Use no_weight_decay parameter from kwargs if specified
        if 'no_weight_decay' in kwargs:
            no_weight_decay = kwargs.pop('no_weight_decay')
            
            # Filter parameters based on name
            parameters = []
            for name, param in model.named_parameters():
                if any(nd in name for nd in no_weight_decay):
                    parameters.append({'params': param, 'weight_decay': 0.0})
                else:
                    parameters.append({'params': param})
                    
            return optimizer_class(parameters, **kwargs)
        else:
            # Simple case - just pass all parameters
            return optimizer_class(model.parameters(), **kwargs)
    except Exception as e:
        logger.error(f"Failed to configure optimizer: {str(e)}")
        # Fallback to standard optimizer
        return optimizer_class(model.parameters(), **kwargs)

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
    return lambda_auto_wrap_policy

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