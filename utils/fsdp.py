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
    CPUOffload,
    StateDictType,
    StateDictConfig
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    lambda_auto_wrap_policy
)


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
        from models.expert import DiTBlock
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DiTBlock}
        )
    elif policy_name == 'LAMBDA':
        from models.router import SelfAttentionBlock
        return functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: isinstance(m, SelfAttentionBlock)
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
        
    # Set up local GPU device
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Get all visible devices
    device_count = torch.cuda.device_count()
    
    logger.info(f"Rank {rank}: Found {device_count} visible devices")
    
    if device_count == 0:
        device = torch.device("cpu")
        logger.warning(f"Rank {rank}: No CUDA devices found, using CPU")
    elif device_count < world_size:
        # If we have fewer devices than processes, use CPU tensor communication
        # The model may still use GPU but communication will be on CPU
        logger.warning(f"Rank {rank}: Fewer CUDA devices ({device_count}) than processes ({world_size})")
        logger.warning(f"Rank {rank}: Using CPU for communication to avoid conflicts")
        device = torch.device("cpu")
    else:
        # Use modulo to assign devices if we have enough
        device_idx = rank % device_count
        device = torch.device(f"cuda:{device_idx}")
        logger.info(f"Rank {rank}: Using device cuda:{device_idx}")
    
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
        from models.expert import DiTBlock
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing as apply_ac
        )
        
        # Define checkpoint wrapper with explicit use_reentrant=False
        def custom_checkpoint_wrapper(module, **kwargs):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,  # Change to NO_REENTRANT 
                checkpoint_kwargs={
                    "preserve_rng_state": False,
                    "use_reentrant": False,  # Explicitly set to False
                },
                **kwargs
            )
        
        # Apply checkpointing
        apply_ac(
            model,
            checkpoint_wrapper_fn=custom_checkpoint_wrapper,
            check_fn=lambda submodule: isinstance(submodule, DiTBlock)
        )
        
        logger.info(f"Applied activation checkpointing with NO_REENTRANT implementation")
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
    """Get paper-recommended FSDP defaults with process group"""
    return {
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "process_group": dist.group.WORLD if dist.is_initialized() else None,
        "use_orig_params": True,
        "limit_all_gathers": True
    }

def create_fsdp_config(config, sharding_strategy="FULL_SHARD", rank=0):
    """Paper's sharding strategies with config overrides"""
    defaults = get_fsdp_defaults()
    
    # Get process group if available
    process_group = defaults.get("process_group", None)
    if process_group is None and dist.is_initialized():
        process_group = dist.group.WORLD

    fsdp_config = {
        **defaults,
        "sharding_strategy": {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
            "NO_SHARD": ShardingStrategy.NO_SHARD
        }[sharding_strategy],
        "device_id": torch.device(f"cuda:{rank}"),
        "process_group": process_group  # Add explicit process group
    }
    if defaults["process_group"] is not None:
        fsdp_config["process_group"] = defaults["process_group"]
        logger.info("FSDP: Process group included in FSDP config.")
    else:
        logger.info("FSDP: Process group is None, FSDP config will not include it.")
    return fsdp_config

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

def get_backward_prefetch(config):
    """Get backward prefetch setting based on config"""
    if hasattr(config, 'fsdp_backward_prefetch'):
        if config.fsdp_backward_prefetch == "BACKWARD_PRE":
            return BackwardPrefetch.BACKWARD_PRE
        elif config.fsdp_backward_prefetch == "BACKWARD_POST":
            return BackwardPrefetch.BACKWARD_POST
    
    # Default prefetch setting
    return BackwardPrefetch.BACKWARD_PRE

def wrap_model_with_fsdp(model, config, param_init_fn=None, rank=0):
    """Wrap model with FSDP using paper's recommended settings"""
    sharding_strategy = getattr(config, 'fsdp_sharding_strategy', 'FULL_SHARD')
    fsdp_config = create_fsdp_config(config, sharding_strategy, rank=rank)
    
    return FSDP(
        model,
        **fsdp_config,
        auto_wrap_policy=get_auto_wrap_policy(config),
        param_init_fn=param_init_fn
    )

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

def check_fsdp_wrapping(model, name="model"):
    """Check if model is properly wrapped with FSDP"""
    if not dist.is_initialized():
        return False, f"{name} not using FSDP (distributed not initialized)"
    
    # Check if model is FSDP wrapped
    is_fsdp_wrapped = isinstance(model, FSDP)
    message = f"{name} {'is' if is_fsdp_wrapped else 'is NOT'} FSDP wrapped"
    
    # If wrapped, check sharding strategy
    if is_fsdp_wrapped:
        strategy = model.sharding_strategy
        strategy_name = {
            ShardingStrategy.FULL_SHARD: "FULL_SHARD",
            ShardingStrategy.SHARD_GRAD_OP: "SHARD_GRAD_OP",
            ShardingStrategy.NO_SHARD: "NO_SHARD",
            ShardingStrategy.HYBRID_SHARD: "HYBRID_SHARD",
        }.get(strategy, "Unknown")
        
        message += f" with {strategy_name} strategy"
    
    return is_fsdp_wrapped, message