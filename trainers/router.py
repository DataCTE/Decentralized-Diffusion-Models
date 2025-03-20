"""Router trainer for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import math
from bitsandbytes.optim import AdamW8bit
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch



from models.router import RouterModel, SelfAttentionBlock
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
from utils.fsdp import get_auto_wrap_policy as get_fsdp_policy
from utils.fsdp import wrap_model_with_fsdp

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
    """Get FSDP auto wrap policy using centralized utility"""
    return get_fsdp_policy(config)

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
        
        # Apply FSDP wrapping
        self.router = wrap_model_with_fsdp(
            base_router,
            config,
            param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
            rank=rank
        )
        
        if rank == 0:
            print(f"Initialized SHARDED Router across {self.world_size} GPUs")
        
        # Add VAE for latent encoding
        from data.vae import VAEWrapper
        self.vae = VAEWrapper(device, config)
        
        # Paper-recommended optimizer settings
        self.optimizer = AdamW8bit(
            self.router.parameters(),
            lr=getattr(config, 'router_learning_rate', 1e-4),
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
        """Trains router with uniform distribution instead of clustering"""
        images = batch["image"].to(self.device)
        
        # Use mixed precision training if configured
        scaler = torch.amp.GradScaler('cuda', enabled=getattr(self.config, 'use_mixed_precision', False))
        with torch.amp.autocast('cuda', enabled=self.config.use_mixed_precision):
            # VAE encoding (match expert trainer flow)
            with torch.no_grad():
                latents = self.vae.encode(images)
                
            # Sample random timesteps t ∈ [0, 1] (match expert)
            t_indices = torch.randint(0, 1000, (latents.size(0),), device=self.device)
            t = t_indices.float() / 1000.0  # Normalize to [0, 1]
            
            # Forward process using cosine schedule (match expert)
            alpha_t = torch.cos((t + 0.008)/1.008 * math.pi/2).pow(2)[:,None,None,None]
            sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
            latent_t = alpha_t * latents + sigma_t * torch.randn_like(latents)
            
            # Get actual cluster assignments from dataset
            targets = batch["expert"].to(self.device)
            
            # Generate dummy text embeddings for router training
            batch_size = images.shape[0]
            dummy_text_embeds = torch.randn(batch_size, self.config.clip_embedding_dim, device=self.device)

            # Get router predictions
            logits = self.router(latent_t, t_indices, dummy_text_embeds)  # Pass dummy text embeddings
            
            # Compute loss
            loss = self.criterion(logits, targets)
        
        # Optimize with gradient isolation
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        
        # Apply gradient clipping if configured
        if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.router.parameters(), 
                self.config.max_grad_norm
            )
        
        # Finish optimization
        scaler.step(self.optimizer)
        scaler.update()
        
        # Update learning rate
        self.lr_scheduler.step()
        
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
            