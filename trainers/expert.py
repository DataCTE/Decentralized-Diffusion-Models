"""Expert trainer for Decentralized Diffusion Models."""

import torch
from bitsandbytes.optim import AdamW8bit
import math
import os
import torch.nn.functional as F
from utils.distributed import is_dist_initialized, synchronize
from utils.fsdp import wrap_model_with_fsdp, configure_optimizer_for_fsdp

from models.mmdit import ExpertMMDiT
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from trainers.base import BaseTrainer
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
from utils.logging import logger

# Import FSDP explicitly for type checking
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP



class ExpertTrainer(BaseTrainer):
    """
    Trainer for expert DiT models in DDM.
    Each expert trains in complete isolation on its assigned data cluster,
    with no cross-communication between experts as described in paper Section 3.2
    """
    def __init__(self, expert_idx, config, device, rank, world_size):
        # Paper-recommended initialization (section 4.1)
        super().__init__(config, device, rank)
        self.expert_idx = expert_idx  # Store the expert index for identification
        self.world_size = world_size
        
        # Initialize base model first
        self.expert = ExpertMMDiT(config)
        
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

    def compute_loss(self, batch):
        """Paper's per-expert loss calculation (Equation 6)"""
        # FSDP handles device placement - no .to(device) needed
        x0 = batch["latent"]  # Already on correct device
        
        # Sample timesteps uniformly as in paper
        t = torch.rand(x0.size(0), device=x0.device)  # Use tensor's device
        
        # Forward process using paper's flow matching formulation
        alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
        sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
        noise = torch.randn_like(x0)
        xt = alpha_t * x0 + sigma_t * noise
        
        # Expert forward pass with cluster conditioning
        pred_flow = self.expert(
            img=xt,
            img_ids=self._get_position_ids(xt),
            txt=batch['clip_embedding'],
            txt_ids=self._get_text_position_ids(batch['clip_embedding']),
            timesteps=t,
            y=self._get_conditioning(x0.shape[0]),
            cluster_ids=batch['expert']
        )
        
        # Flow matching target calculation from paper
        target_flow = (x0 - alpha_t * xt) / (sigma_t**2 + 1e-7)
        
        return F.mse_loss(pred_flow, target_flow)

    def train_step(self, batch):
        """Train expert with flow matching loss per paper Section 3.2"""
        # Get data - ensure proper device placement
        latents = batch["latent"].to(self.device, non_blocking=True)
        text_embeds = batch["clip_embedding"].to(self.device, non_blocking=True)
        cluster_ids = batch["expert"].to(self.device, non_blocking=True)
        
        # Mixed precision context
        with torch.amp.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            # Random timesteps
            t = torch.rand(latents.size(0), device=self.device)
            
            # Forward pass through expert model
            pred_flow = self.expert(
                img=latents,
                img_ids=self._get_position_ids(latents),
                txt=text_embeds,
                txt_ids=self._get_text_position_ids(text_embeds),
                timesteps=t * 1000,  # Scale timesteps
                y=self._get_conditioning(latents.shape[0]),
                cluster_ids=cluster_ids
            )
            
            # Calculate loss
            loss = self.flow_matcher.compute_loss(pred_flow, latents, t)
        
        # Backpropagation
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
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
        
        # Save using the centralized utility
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
        # Load using the centralized utility
        metadata = load_model_checkpoint(
            model=self.expert,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            is_fsdp=True
        )
        
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
                logger.info(f"Isolated optimizer state for expert {self.expert_idx}") 

    def _get_position_ids(self, x):
        """Simple patch-based position IDs as described in paper"""
        B, C, H, W = x.shape
        h = H // self.config.patch_size
        w = W // self.config.patch_size
        
        # Create grid of position indices
        pos_h = torch.arange(h, device=self.device)
        pos_w = torch.arange(w, device=self.device)
        pos_grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing='ij'), dim=-1)
        pos_grid = pos_grid.reshape(-1, 2)[None].repeat(B, 1, 1)
        
        return pos_grid

    def _get_text_position_ids(self, text_emb):
        """Simple sequence position IDs for text"""
        B, L, _ = text_emb.shape
        pos_ids = torch.arange(L, device=self.device)[None].repeat(B, 1)
        return pos_ids[:, :, None].repeat(1, 1, 2)  # Add 2D position dim for consistency 

    def _get_conditioning(self, batch_size):
        """Get conditioning vector for expert"""
        # Return zero conditioning vector of correct shape
        return torch.zeros(batch_size, self.config.vec_in_dim, device=self.device) 