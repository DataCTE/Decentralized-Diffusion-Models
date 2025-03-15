"""Router trainer for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import math
from bitsandbytes.optim import AdamW8bit
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    lambda_auto_wrap_policy
)



from models.router import RouterModel, SelfAttentionBlock
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint

def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Convert sharding strategy name to enum"""
    return {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }[name.upper()]

def get_backward_prefetch(name: str) -> BackwardPrefetch:
    """Convert backward prefetch name to enum"""
    return {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }[name.upper()]

def get_auto_wrap_policy(config):
    """Get FSDP auto wrap policy based on config"""
    if getattr(config, 'fsdp_auto_wrap_policy', 'DEFAULT') == "SIZE_BASED":
        return size_based_auto_wrap_policy(
            min_num_params=getattr(config, 'fsdp_min_num_params', 1e6)
        )
    # Wrap router's attention blocks
    return lambda_auto_wrap_policy(
        lambda_fn=lambda m: isinstance(m, SelfAttentionBlock)
    )

class RouterTrainer:
    """Trainer for the router model in DDM"""
    def __init__(self, config, device, rank, world_size=None):
        # Initialize parameters
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size or 1
        
        # Create base router model with safe config access
        base_router = RouterModel(config).to(device)
        
        # Get FSDP parameters with defaults
        sharding_strategy = get_sharding_strategy(
            getattr(config, 'fsdp_sharding_strategy', 'FULL_SHARD')
        )
        cpu_offload = CPUOffload(
            offload_params=getattr(config, 'fsdp_cpu_offload', False)
        )
        backward_prefetch = get_backward_prefetch(
            getattr(config, 'fsdp_backward_prefetch', 'BACKWARD_PRE')
        )
        auto_wrap_policy = get_auto_wrap_policy(config)
        
        # Apply FSDP wrapping
        self.router = FSDP(
            base_router,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True
        )
        
        if rank == 0:
            print(f"Initialized SHARDED Router across {self.world_size} GPUs")
        
        # Paper-recommended optimizer settings
        self.optimizer = AdamW8bit(
            self.router.parameters(),
            lr=config.router_learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Paper-recommended learning schedule
        # Add warmup steps if not in config
        self.warmup_steps = getattr(config, 'warmup_steps', int(0.05 * config.num_steps))
        self.total_steps = getattr(config, 'num_steps', 400000)
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/self.warmup_steps, 1.0) 
            if step < self.warmup_steps 
            else 0.5*(1 + math.cos(math.pi*(step - self.warmup_steps)/self.total_steps))
        )

    def train_step(self, batch):
        """
        Implements Algorithm 1 from paper (router training)
        
        This trains the router to predict which expert should handle
        each sample, as described in Section 3.3 of the paper.
        """
        # Set router in training mode
        self.router.train()
        
        # Get images and cluster assignments (Section 3.3)
        images = batch["image"].to(self.device)
        clusters = batch["cluster"].to(self.device)  # k in Algorithm 1
        
        # Sample random timesteps t ∈ [0, 1] (Algorithm 1)
        t = torch.rand(images.size(0), device=self.device)
        
        # Sample random noise ε ~ N(0, I) (Algorithm 1)
        ε = torch.randn_like(images)
        
        # Forward process using cosine schedule (Algorithm 1)
        # xt = αt x0 + σt ε
        αt = torch.cos(t * math.pi/2)[:,None,None,None]
        σt = torch.sin(t * math.pi/2)[:,None,None,None]
        xt = αt * images + σt * ε
        
        # Router prediction (Equation 5 in the paper)
        # z = rθ(xt, t) ∈ R^|K|
        z = self.router(xt, t)
        
        # Cross-entropy loss for router (Section 3.3)
        # LCE(z, OneHot(k))
        loss = self.criterion(z, clusters)
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        # Paper-recommended gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.router.parameters(), 
            max_norm=self.config.max_grad_norm,
            norm_type=2.0
        )
        self.optimizer.step()
        self.lr_scheduler.step()  # Update learning rate
        
        return loss.item()

    def train_epoch(self, loader):
        """
        Trains the router for one epoch
        
        Args:
            loader: DataLoader with image and cluster data
            
        Returns:
            Average loss over the epoch
        """
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
        
        # Return average loss over the epoch
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, save_dir, step):
        """Save router checkpoint using centralized utility"""
        # Create checkpoint path
        checkpoint_path = f"{save_dir}/router_step{step}.pt"
        
        # Create metadata
        metadata = {
            'step': step,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        
        # Save using the centralized utility
        return save_model_checkpoint(
            model=self.router,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            metadata=metadata,
            is_fsdp=isinstance(self.router, FSDP)
        )
        
    def load_checkpoint(self, checkpoint_path):
        """Load router checkpoint using centralized utility"""
        # Load using the centralized utility
        metadata = load_model_checkpoint(
            model=self.router,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            is_fsdp=isinstance(self.router, FSDP),
            device=self.device
        )
        
        return metadata

    def forward(self, x_t, timesteps, text_embeddings=None):
        """
        Forward pass for router model (test time)
        
        Args:
            x_t: Noisy data at timestep t
            timesteps: Timestep values
            text_embeddings: Optional text embeddings for conditional generation
            
        Returns:
            Logits for expert selection
        """
        # Forward pass through the router
        return self.router(x_t, timesteps, text_embeddings)
            