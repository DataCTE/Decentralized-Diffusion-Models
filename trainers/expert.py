"""Expert trainer for Decentralized Diffusion Models."""

import torch
from bitsandbytes.optim import AdamW8bit
import math
import os
import logging
from typing import Dict
import torch.nn as nn

from models.mmdit import ExpertMMDiT
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from trainers.base import BaseTrainer
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint




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
        
        # --- FIX: Ensure model config uses latent_channels for in_channels ---
        from types import SimpleNamespace
        model_config_dict = vars(config).copy() # Create a mutable copy
        if hasattr(config, 'latent_channels'):
             model_config_dict['in_channels'] = config.latent_channels
             #logger.info(f"Expert {expert_idx}: Setting model in_channels to latent_channels ({config.latent_channels})")
        else:
             self.logger.warning(f"Expert {expert_idx}: config.latent_channels not found, using config.in_channels ({config.in_channels}) for model.")
        # Use a SimpleNamespace or a dedicated dataclass if ExpertMMDiT expects one
        model_config = SimpleNamespace(**model_config_dict)
        # --- END FIX ---

        # Initialize base model first using the corrected config
        self.expert = ExpertMMDiT(model_config) # Pass the modified config
        
        # After creating the expert:
        self.expert = self.expert.to(device)
        
        # Don't wrap with FSDP here - let cache manager handle it
        
        # Use standard optimizer initially - will be reconfigured when FSDP is applied
        self.optimizer = AdamW8bit(
            self.expert.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
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
        
        # Learning rate scheduler
        warmup_steps = int(0.05 * config.num_steps)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/warmup_steps, 1.0) 
            if step < warmup_steps
            else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (config.num_steps - warmup_steps)))
        )

        # Initialize scaler once in __init__
        self.scaler = torch.amp.GradScaler(enabled=config.use_mixed_precision)

        # Add paper-recommended initialization
        for p in self.expert.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1/math.sqrt(config.num_experts))
        nn.init.constant_(self.expert.cluster_embed.weight, 0.01)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Implements paper's per-expert loss (Section 3.2, Equation 6)
        
        Args:
            batch: Contains 'latent' (B, C, H, W) and 'clip_embedding' (B, L, D)
            
        Returns:
            Scalar loss value
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
        
        # Flow matching loss (Equation 6)
        return self.flow_matcher.compute_loss(pred_flow, x0, t)

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
        
        # Router logic remains unchanged
        with torch.no_grad():
            t = torch.rand(B, device=self.device) * 1000
            batch['cluster_pred'] = self.router(
                img=latent,
                timesteps=t,
                txt=batch['clip_embedding']
            ).argmax(dim=-1)

        # Dynamic sequence reshaping
        img_seq = latent.view(B, C, H * W).permute(0, 2, 1)  # [B, L, C]
        
        # Position IDs generation using actual H/W
        pos_ids = self._get_position_ids(latent)  # Ensure this uses H/W
        
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            pred_flow = self.expert(
                img=img_seq,
                img_ids=pos_ids,
                txt=batch["clip_embedding"],
                txt_ids=self._get_text_position_ids(batch["clip_embedding"]),
                timesteps=torch.rand(B, device=self.device) * 1000,
                y=self._get_conditioning(B),
                cluster_ids=batch['cluster_pred']
            )
            
            # Dynamic loss reshaping
            pred_flow = pred_flow.permute(0, 2, 1).view(B, C, H, W)
            loss = self.flow_matcher.compute_loss(
                pred_flow, 
                latent,
                torch.rand(B, device=self.device)
            ) * self.config.expert_loss_weight

        # Original optimization steps
        self.scaler.scale(loss).backward()
        if self.step % self.config.gradient_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return loss.item()
    
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
            lr=self.config.learning_rate,
            betas=self.config.adam_betas,
            weight_decay=self.config.weight_decay
        )
        
        # Reset scheduler
        warmup_steps = int(0.05 * self.config.num_steps)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/warmup_steps, 1.0) 
            if step < warmup_steps
            else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (self.config.num_steps - warmup_steps)))
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
        """Generates position IDs from 4D latent tensor"""
        _, _, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        # Generate grid for actual H/W dimensions
        pos_h = torch.arange(H, device=device, dtype=dtype)
        pos_w = torch.arange(W, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Stack and flatten spatial positions
        pos_ids = torch.stack([grid_h, grid_w], dim=-1).flatten(0, 1)[None]  # [1, L, 2]
        pos_ids = pos_ids.expand(x.shape[0], -1, -1)  # [B, L, 2]
        
        return pos_ids

    def _get_text_position_ids(self, text_emb):
        """Simple sequence position IDs for text with improved shape handling"""
        print(f"[DEBUG Expert {self.expert_idx}] Text embedding shape for position IDs: {text_emb.shape}")
        
        if text_emb.dim() == 4:
            # Handle [B, S, L, D] → [B, L, D]
            B, S, L, D = text_emb.shape
            if S == 1:
                # If sequence dimension is 1, we can simply remove it
                print(f"[DEBUG Expert {self.expert_idx}] Reshaping 4D text embedding for position IDs")
                text_emb = text_emb.squeeze(1)
            else:
                print(f"[WARNING Expert {self.expert_idx}] Multiple text sequences ({S}), using first")
                text_emb = text_emb[:, 0]
        
        # Now process the standard 3D case
        B, L, _ = text_emb.shape
        pos_ids = torch.arange(L, device=self.device)[None].repeat(B, 1)
        pos_ids = pos_ids[:, :, None].repeat(1, 1, 2)  # Add 2D position dim for consistency
        
        print(f"[DEBUG Expert {self.expert_idx}] Text position ID shape: {pos_ids.shape}")
        return pos_ids

    def _get_conditioning(self, batch_size):
        """Get conditioning vector for expert"""
        # Return zero conditioning vector of correct shape
        return torch.zeros(batch_size, self.config.vec_in_dim, device=self.device) 