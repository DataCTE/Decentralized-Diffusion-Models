"""Router trainer for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import math
from bitsandbytes.optim import AdamW8bit
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
import torch.distributed as dist
import logging
import torch.nn.functional as F
from typing import Dict

# Import centralized utilities for consistent implementation

from utils.fsdp import wrap_model_with_fsdp
from models.router import RouterModel
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
from utils.fsdp import get_auto_wrap_policy as get_fsdp_policy

# Initialize module-level logger
logger = logging.getLogger(__name__)

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
        
        # Initialize logger for this class
        self.logger = logging.getLogger(__name__)
        
        # Create base router model with safe config access
        base_router = RouterModel(config)
        
        # Apply FSDP wrapping - note we don't call .to(device) before FSDP wrapping
        # IMPORTANT: Let FSDP handle device placement to ensure proper sharding
        self.router = wrap_model_with_fsdp(
            base_router,
            config,
            param_init_fn=lambda m: m.to_empty(recurse=False),
            rank=rank
        )
        
        # Print initialization message from all ranks
        self.logger.info(f"[Rank {rank}] Created base Router model")
        
        # Add VAE for latent encoding - load VAE on demand to save memory
        self._vae = None
        
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

    def train(self):
        """Set router model to training mode"""
        self.router.train()
        
    def eval(self):
        """Set router model to evaluation mode"""
        self.router.eval()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Implements paper's router training with proper step tracking and type safety
        
        Args:
            batch: Dictionary containing:
                - 'latent': Tensor of shape [B, C, H, W]
                - 'clip_embedding': Tensor of shape [B, seq_len, D]
                - 'cluster_labels': LongTensor of shape [B]
            
        Returns:
            Computed cross-entropy loss with temperature scaling
        """
        # Validate input types and shapes
        self._validate_batch(batch)
        
        # Update step counter and calculate temperature
        self.step += 1  # Maintain internal step counter
        current_step = max(1, self.step)
        temp = self._calculate_temperature(current_step)
        
        # Forward pass with mixed precision
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            logits = self.router(
                self._add_diffusion_noise(batch['latent']),
                self._generate_timesteps(batch['latent'].size(0)),
                batch['clip_embedding']
            ) / temp
            
            return F.cross_entropy(logits, batch['cluster_labels'])

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Type and shape validation for training batches"""
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be dict, got {type(batch)}")
        
        required_keys = {'latent', 'clip_embedding', 'cluster_labels'}
        missing = required_keys - set(batch.keys())
        if missing:
            raise ValueError(f"Missing required batch keys: {missing}")
        
        if batch['latent'].dim() != 4:
            raise ValueError(f"Latent must be 4D tensor, got {batch['latent'].dim()}D")
        
        if batch['clip_embedding'].dim() != 3:
            raise ValueError(f"CLIP embeddings must be 3D tensor, got {batch['clip_embedding'].dim()}D")

    def _calculate_temperature(self, current_step: int) -> float:
        """Compute temperature with exponential decay and minimum floor"""
        return max(
            self.config.router_min_temp,
            self.config.router_temperature * 
            (self.config.router_temperature_decay ** current_step)
        )

    def _add_diffusion_noise(self, x0: torch.Tensor) -> torch.Tensor:
        """Add analytical diffusion noise per paper's cosine schedule"""
        B = x0.size(0)
        device = x0.device
        
        t = torch.rand(B, device=device)
        alpha_t = torch.cos(t * math.pi/2)[:, None, None, None]
        sigma_t = torch.sin(t * math.pi/2)[:, None, None, None]
        noise = torch.randn_like(x0)
        
        return alpha_t * x0 + sigma_t * noise

    def _generate_timesteps(self, batch_size: int) -> torch.Tensor:
        """Generate timesteps scaled to [0, 1000) range"""
        return torch.rand(batch_size, device=self.device) * 1000

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

    def _init_weights(self):
        """Paper's initialization scheme (Section 4.1)"""
        # Initialize embedder with scaled normal distribution
        nn.init.normal_(self.embedder.weight, mean=0.0, std=0.02/math.sqrt(3))
        nn.init.zeros_(self.embedder.bias)
        
        # Initialize CLS token with smaller variance
        nn.init.normal_(self.cls_token, std=0.01)
        
        # Final classifier initialization
        nn.init.normal_(self.classifier[-1].weight, std=0.01)
        nn.init.zeros_(self.classifier[-1].bias)
        
        # Initialize text projection layers
        for layer in self.text_embed_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
            