import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

def save_sharded(checkpoint, path, model):
    """Paper's recommended checkpoint format from Appendix A.2"""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL, save_policy):
        state = model.state_dict()
    
    if dist.get_rank() == 0:
        torch.save({**checkpoint, 'model': state}, path)
        print(f"Saved sharded checkpoint to {path}")