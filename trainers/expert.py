"""Expert trainer for Decentralized Diffusion Models."""

import torch
from bitsandbytes.optim import AdamW8bit
import math
import os
import logging
from typing import Dict
import torch.nn as nn
from einops import rearrange

from models.mmdit import ExpertMMDiT
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from trainers.base import BaseTrainer
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
from types import SimpleNamespace



class ExpertTrainer(BaseTrainer):
    """
    Trainer for expert DiT models in DDM.
    Each expert trains in complete isolation on its assigned data cluster,
    with no cross-communication between experts as described in paper Section 3.2
    """
    def __init__(self, expert_idx, config, device, rank, world_size, router=None, logger=None):
        # Paper-recommended initialization (section 4.1)
        super().__init__(config, device, rank)
        
        # Use the passed logger or create a fallback
        self.logger = logger if logger else logging.getLogger(f"ExpertTrainer_{expert_idx}_fallback")
        
        # Validate router existence before initializing components
        if router is None:
            # Use self.logger for error messages
            self.logger.error(f"ExpertTrainer requires router reference. Missing router for expert {expert_idx} (rank {rank})")
            raise ValueError(
                f"ExpertTrainer requires router reference. "
                f"Missing router for expert {expert_idx} (rank {rank})"
            )
        
        self.router = router
        self.expert_idx = expert_idx
        self.world_size = world_size
        
        # Calculate patched dimensions correctly
        patch_size = config.patch_size
        if hasattr(config, 'latent_channels'):
            model_config_dict = vars(config).copy()
            expected_channels = config.latent_channels * (patch_size ** 2)
            model_config_dict['in_channels'] = expected_channels
            model_config_dict['out_channels'] = expected_channels
            model_config_dict['latent_channels'] = config.latent_channels
            
            # Log the dimension configuration for debugging
            self.logger.info(f"Expert {expert_idx} configured with: in_channels={expected_channels}, "
                             f"latent_channels={config.latent_channels}, patch_size={patch_size}")
        else:
            self.logger.warning(f"Using default in_channels {config.in_channels}")
        
        # Ensure patch_size propagates correctly to model
        model_config_dict['patch_size'] = patch_size
        
        model_config = SimpleNamespace(**model_config_dict)
        self.expert = ExpertMMDiT(model_config)
        
        # After creating the expert:
        self.expert = self.expert.to(device)
        
        # Don't wrap with FSDP here - let cache manager handle it
        
        # Use expert-specific optimizer parameters
        self.optimizer = AdamW8bit(
            self.expert.parameters(),
            lr=config.expert_learning_rate,  # Changed from general learning_rate
            betas=config.expert_adam_betas,
            weight_decay=config.expert_weight_decay
        )
        
        # Paper-defined components
        self.flow_matcher = DecentralizedFlowMatcher(
            sigma=config.sigma, 
            loss_type=config.loss_type
        )
        self.vae = None
        self.clip = None
        
        # Precompute diffusion schedule as in paper appendix
        self.alphas, self.alpha_bar, _ = get_alphas_and_betas()
        
        # Update warmup and scheduler to use expert-specific values
        self.warmup_steps = config.expert_warmup_steps
        self.total_steps = config.num_steps
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step / self.warmup_steps, 1.0)
        )

        # Add step counter initialization
        self.step = 0  # Track training steps for gradient accumulation
        
        # Initialize scaler once in __init__
        self.scaler = torch.amp.GradScaler(enabled=config.use_mixed_precision)

        # Add paper-recommended initialization
        for p in self.expert.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1/math.sqrt(config.num_experts))
        nn.init.constant_(self.expert.cluster_embed.weight, 0.01)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Implements paper's per-expert loss (Section 3.2, Equation 6)
        
        Args:
            batch: Contains 'latent' (B, C, H, W) and 'clip_embedding' (B, L, D)
            
        Returns:
            Dictionary with 'raw_loss', 'weighted_loss', 'router_confidence', and 'cluster_alignment'
        """
        # Paper's cosine schedule handling
        x0 = batch["latent"]
        B = x0.shape[0]
        
        # Sample timesteps and noise following paper appendix A
        t = torch.rand(B, device=x0.device)
        alpha_t = torch.cos(t * math.pi/2).view(-1, 1, 1, 1)
        sigma_t = torch.sin(t * math.pi/2).view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = alpha_t * x0 + sigma_t * noise
        
        # Reshape for transformer input (B, L, C)
        xt_seq = xt.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Get cluster predictions from router (Section 3.3)
        with torch.no_grad():
            cluster_ids = self.router(
                img=xt,
                timesteps=t * 1000,  # Scale to [0,1000)
                txt=batch['clip_embedding']
            ).argmax(dim=-1)
        
        # Get router confidence scores for ALL experts (paper Eq.4)
        with torch.no_grad():
            router_logits = self.router(
                img=xt,
                timesteps=t * 1000,
                txt=batch['clip_embedding']
            )
            router_probs = torch.softmax(router_logits, dim=-1)
        
        # Forward pass with cluster conditioning (Equation 5)
        pred_flow = self.expert(
            img=xt_seq,
            img_ids=self._get_position_ids(xt),
            txt=batch['clip_embedding'],
            txt_ids=self._get_text_position_ids(batch['clip_embedding']),
            timesteps=t * 1000,
            y=self._get_conditioning(B),
            cluster_ids=cluster_ids
        )
        
        # Pass actual diffused latent (xt) to flow matcher
        raw_loss = self.flow_matcher.compute_loss(pred_flow, x0, xt, t)
        
        # New ensemble-weighted loss (paper Section 3.3)
        expert_weight = router_probs[:, self.expert_idx]
        weighted_loss = raw_loss * expert_weight.mean()
        
        # Track specialization metrics
        cluster_match = (cluster_ids == self.expert_idx).float().mean()
        
        return {
            'raw_loss': raw_loss,
            'weighted_loss': weighted_loss,
            'router_confidence': expert_weight.mean(),
            'cluster_alignment': cluster_match
        }

    def train_step(self, batch):
        """Capacity-aware training with dynamic shape handling"""
        # Capacity enforcement remains unchanged
        expert_capacity = int(self.config.batch_size * self.config.expert_capacity_factor)
        if len(batch['latent']) > expert_capacity:
            indices = torch.randperm(len(batch['latent']))[:expert_capacity]
            batch = {k: v[indices] for k, v in batch.items()}

        # Get actual latent dimensions
        latent = batch["latent"]
        B, C, H, W = latent.shape  # Now dynamic based on input
        
        # Generate proper diffusion parameters
        t = torch.rand(B, device=self.device)  # [0, 1) range
        alpha_t = torch.cos(t * math.pi/2).view(-1, 1, 1, 1)
        sigma_t = torch.sin(t * math.pi/2).view(-1, 1, 1, 1)
        noise = torch.randn_like(latent)
        xt = alpha_t * latent + sigma_t * noise  # Diffused latent

        # Router uses scaled timesteps
        with torch.no_grad():
            batch['cluster_pred'] = self.router(
                img=xt,  # Use diffused latent
                timesteps=t * 1000,
                txt=batch['clip_embedding']
            ).argmax(dim=-1)

        # Process through expert model
        img_seq = rearrange(
            xt,  # Use diffused latent for patching
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=4, p2=4
        )
        
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            pred_flow = self.expert(
                img=img_seq,
                img_ids=self._get_position_ids(xt),
                txt=batch["clip_embedding"],
                txt_ids=self._get_text_position_ids(batch["clip_embedding"]),
                timesteps=t * 1000,
                y=self._get_conditioning(B),
                cluster_ids=batch['cluster_pred']
            )
            
            # Calculate loss with proper dimensions
            loss = self.flow_matcher.compute_loss(
                pred_flow,  # Model predictions
                latent,     # Original x0
                xt,         # Diffused latent
                t           # Original timesteps
            ) * self.config.expert_loss_weight

        # Proper mixed precision handling
        self.scaler.scale(loss).backward()

        # Gradient accumulation logic
        if (self.step + 1) % self.config.expert_gradient_accumulation_steps == 0:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            if self.config.expert_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.expert.parameters(),
                    self.config.expert_max_grad_norm
                )
            
            # Update parameters and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        self.step += 1  # Increment step counter after optimization
        
        # Update scheduler every step regardless of gradient accumulation
        self.lr_scheduler.step()
        
        # Modified return with metrics
        metrics = self.compute_loss(batch)
        return {
            'total_loss': loss.item(),
            'raw_loss': loss.item() / self.config.expert_loss_weight,
            **metrics  # From compute_loss()
        }
    
    def save_checkpoint(self, save_dir, step):
        """Save a checkpoint for the expert model using consolidated checkpoint utility"""
        # Create checkpoint path
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = f"{save_dir}/expert_{self.expert_idx}_step{step}.pt"
        
        # Create metadata
        metadata = {
            'expert_idx': self.expert_idx,
            'step': step,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        
        # Save using the centralized utility, passing the logger if needed by the utility
        return save_model_checkpoint(
            model=self.expert,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            metadata=metadata,
            is_fsdp=True
        )
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint for the expert model using consolidated checkpoint utility"""
        # Example: Make sure logging uses self.logger
        self.logger.info(f"Loading checkpoint for expert {self.expert_idx} from {checkpoint_path}")

        # Load using the centralized utility
        metadata = load_model_checkpoint(
            model=self.expert,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            is_fsdp=True
        )
        if metadata:
            self.logger.info(f"Loaded checkpoint for expert {self.expert_idx}, step {metadata.get('step', 'N/A')}")
        else:
            self.logger.warning(f"Failed to load checkpoint for expert {self.expert_idx} from {checkpoint_path}")

        return metadata
    
    def forward_diffuse(self, x0, t, noise=None):
        """
        Forward diffusion process with paper's cosine schedule
        Section 3.1 and Appendix A
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Paper's cosine schedule
        alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
        sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
        
        # Forward process: xt = αt·x0 + σt·ε
        return alpha_t * x0 + sigma_t * noise

    def reset_parameters(self):
        """Reset expert parameters when assigned to a new cluster"""
        if self.rank == 0:
            print(f"Resetting parameters for expert {self.expert_idx}")
            
        # Reinitialize the model
        for module in self.expert.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                
        # Reset optimizer state
        self.optimizer = AdamW8bit(
            self.expert.parameters(),
            lr=self.config.expert_learning_rate,
            betas=self.config.expert_adam_betas,
            weight_decay=self.config.expert_weight_decay
        )
        
        # Reset scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/self.warmup_steps, 1.0)
        )

    def isolate_optimizer_state(self):
        """Prevent contamination of optimizer state across experts"""
        # Get current optimizer state
        state = self.optimizer.state_dict()
        
        # Identify parameters specific to this expert
        expert_params = {id(p): p for p in self.expert.parameters()}
        
        # Filter optimizer state to only include this expert's parameters
        if 'state' in state and state['state']:
            filtered_state = {}
            for param_id, param_state in state['state'].items():
                if param_id in expert_params:
                    filtered_state[param_id] = param_state
            
            # Replace with filtered state
            state['state'] = filtered_state
            
            # Load filtered state back
            self.optimizer.load_state_dict(state)
            
            if self.rank == 0:
                self.logger.info(f"Isolated optimizer state for expert {self.expert_idx}") 

    def _get_position_ids(self, x):
        """Generates position IDs for PATCHED latent tensor"""
        _, _, H, W = x.shape
        patch_size = self.config.patch_size
        
        # Ensure divisible patch dimensions
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"Input size {H}x{W} must be divisible by patch_size {patch_size}")
        
        H_patch = H // patch_size
        W_patch = W // patch_size
        
        device = x.device
        dtype = x.dtype
        
        # Generate grid for PATCHED dimensions
        pos_h = torch.arange(H_patch, device=device, dtype=dtype)
        pos_w = torch.arange(W_patch, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Stack and flatten spatial positions
        pos_ids = torch.stack([grid_h, grid_w], dim=-1).flatten(0, 1)[None]  # [1, L, 2]
        pos_ids = pos_ids.expand(x.shape[0], -1, -1)  # [B, L, 2]
        
        return pos_ids

    def _get_text_position_ids(self, text_emb):
        """Handle 4D CLIP embeddings with proper squeezing"""
        # Handle both 3D and 4D CLIP embeddings
        if text_emb.dim() == 4:
            # Verify sequence dimension is singleton before squeezing
            if text_emb.size(1) == 1:
                text_emb = text_emb.squeeze(1)
            else:
                self.logger.warning(f"Unexpected text embedding shape {text_emb.shape}, taking first sequence element")
                text_emb = text_emb[:, 0]
        
        # Validate final dimensions
        if text_emb.dim() != 3:
            raise ValueError(f"Invalid text embedding dimensions after processing: {text_emb.shape}")
        
        # Generate position IDs for validated 3D tensor
        B, L, _ = text_emb.shape
        pos_ids = torch.arange(L, device=self.device)[None].repeat(B, 1)
        return pos_ids[:, :, None].repeat(1, 1, 2)  # Add 2D position dim

    def _get_conditioning(self, batch_size):
        """Get conditioning vector for expert"""
        # Return zero conditioning vector of correct shape
        return torch.zeros(batch_size, self.config.vec_in_dim, device=self.device) 