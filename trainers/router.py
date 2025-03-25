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

    def train_step(self, batch, true_clusters=None, temperature=1.0):
        """Train router with cross-entropy loss per paper Section 3.3"""
        # Get inputs and ensure proper shapes
        latents = batch["latent"].to(self.device)  # [B, C, H, W] or [B, 1, C, H, W]
        if latents.dim() == 5:
            latents = latents.squeeze(1)  # Remove extra dimension if present
        
        if true_clusters is None:
            true_clusters = batch["expert"].to(self.device)
        
        text_embeds = batch["clip_embedding"].to(self.device)
        if text_embeds.dim() == 4:  # [B, 1, seq_len, dim]
            text_embeds = text_embeds.squeeze(1)
        
        # Use mixed precision training with updated syntax
        scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_mixed_precision)
        
        with torch.amp.autocast('cuda', enabled=self.config.use_mixed_precision):
            # Sample timestep uniformly as in paper
            t = torch.rand(latents.size(0), device=self.device)
            
            # Forward diffusion process (Section 3.1)
            alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
            sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
            noise = torch.randn_like(latents)
            x_t = alpha_t * latents + sigma_t * noise
            
            # Ensure x_t has correct shape [B, C, H, W]
            if x_t.dim() != 4:
                raise ValueError(f"Expected x_t to have 4 dimensions [B,C,H,W], got shape {x_t.shape}")
            
            # Get router predictions with temperature annealing
            logits = self.router(x_t, t * 1000, text_embeds)
            logits = logits / temperature  # Apply temperature scaling
            
            # Compute cross-entropy loss
            loss = self.criterion(logits, true_clusters)
        
        # Optimize
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.router.parameters(), 
                self.config.max_grad_norm
            )
        
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

    def validate_router(self, val_loader):
        """Validate router accuracy on validation set"""
        self.router.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                latents = batch["latent"].to(self.device)
                targets = batch["expert"].to(self.device)
                text_embeds = batch["clip_embedding"].to(self.device)
                
                # Sample random timestep
                t = torch.rand(latents.size(0), device=self.device)
                
                # Forward diffusion
                alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
                sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
                noise = torch.randn_like(latents)
                x_t = alpha_t * latents + sigma_t * noise
                
                # Get predictions
                logits = self.router(x_t, t * 1000, text_embeds)
                predictions = torch.argmax(logits, dim=1)
                
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        return accuracy

    def get_expert_weights(self, x_t, t, text_embeddings=None, strategy='top_k', k=1):
        """
        Get expert combination weights for inference (Section 3.4)
        
        Args:
            x_t: Noisy input at timestep t
            t: Timestep values
            text_embeddings: Optional text conditioning
            strategy: Sampling strategy ('top_k', 'nucleus', etc.)
            k: Number of experts to select for top-k
        """
        self.router.eval()
        with torch.no_grad():
            # Get logits
            logits = self.router(x_t, t, text_embeddings)
            probs = torch.softmax(logits, dim=-1)
            
            if strategy == 'top_k':
                # Zero out all but top-k probabilities
                topk_probs, indices = torch.topk(probs, k=k, dim=-1)
                zeros = torch.zeros_like(probs)
                zeros.scatter_(-1, indices, topk_probs)
                weights = zeros / zeros.sum(dim=-1, keepdim=True)
                
            elif strategy == 'nucleus':
                # Nucleus (top-p) sampling
                sorted_probs, indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum <= self.config.top_p
                mask[..., 0] = True  # Always keep top probability
                
                # Zero out filtered probabilities
                filtered_probs = sorted_probs * mask
                weights = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                
                # Restore original ordering
                reverse_indices = torch.argsort(indices)
                weights = torch.gather(weights, -1, reverse_indices)
                
            else:
                weights = probs  # Use raw probabilities
                
            return weights
            