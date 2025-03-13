from torch.distributed.fsdp import ShardingStrategy, CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import torch

def get_fsdp_defaults():
    """Paper-specified isolation defaults"""
    return {
        "process_group": torch.distributed.new_group(backend="nccl"),  # Separate group per expert
        "sync_module_states": False,  # Don't sync initial weights
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,
        "use_origin_params": True
    }

def create_fsdp_config(config, sharding_strategy="FULL_SHARD"):
    """Paper's sharding strategies with config overrides"""
    defaults = get_fsdp_defaults()
    return {
        **defaults,
        "sharding_strategy": {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP
        }[sharding_strategy],
        "device_id": torch.cuda.current_device()
    }