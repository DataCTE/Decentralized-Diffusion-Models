"""Expert trainer for Decentralized Diffusion Models."""

import torch
from bitsandbytes.optim import AdamW8bit
import math
import os
import torch.nn.functional as F

from models.mmdit import ExpertMMDiT
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from trainers.base import BaseTrainer
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
from utils.logging import logger
from utils.fsdp import wrap_model_with_fsdp



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
        
        # Create base expert model
        base_expert = ExpertMMDiT(config).to(device)
        
        # Apply FSDP wrapping using the same method as router
        self.expert = wrap_model_with_fsdp(
            base_expert,
            config,
            param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
            rank=rank
        )
        
        # Paper-specified optimizer settings - use Adam with FSDP
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
        self.vae = VAEWrapper(device, config)
        self.clip = CLIPTextEncoder(device, config)
        
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
        
        # Update log message to match router style
        if rank == 0:
            print(f"Initialized SHARDED Expert {expert_idx} across {world_size} GPUs")

    def compute_loss(self, batch):
        """Paper's per-expert loss calculation (Equation 6)"""
        # Extract latents and move to device
        x0 = batch["latent"].to(self.device)
        
        # Sample timesteps uniformly as in paper
        t = torch.rand(x0.size(0), device=self.device)
        
        # Forward process using paper's flow matching formulation
        # ut(xt|x0) = (x0 - αt·xt)/(σt^2)
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
        """Execute a single training step"""
        self.expert.train()
        self.optimizer.zero_grad()
        
        # Extract and reshape inputs from batch
        latents = batch['latent']  # [B, 1, C, H, W]
        B, _, C, H, W = latents.shape
        latents = latents.squeeze(1)  # [B, C, H, W]
        
        # Get CLIP embeddings
        clip_emb = batch['clip_embedding']  # [B, 1, seq_len, dim]
        clip_emb = clip_emb.squeeze(1)  # [B, seq_len, dim]
        
        # Calculate patch dimensions
        h = H // self.config.patch_size
        w = W // self.config.patch_size
        
        # Reshape image to sequence format as expected by Flux
        img_seq = latents.view(B, C, h * w).permute(0, 2, 1)  # [B, L, C]
        
        # Create position IDs for patches
        pos_h = torch.arange(h, device=self.device)
        pos_w = torch.arange(w, device=self.device)
        img_ids = torch.stack(torch.meshgrid(pos_h, pos_w, indexing='ij'), dim=-1)
        img_ids = img_ids.reshape(-1, 2)  # [h*w, 2]
        img_ids = img_ids[None].expand(B, -1, -1)  # [B, h*w, 2]
        
        # Create position IDs for text
        txt_ids = torch.arange(clip_emb.size(1), device=self.device)
        txt_ids = torch.stack([
            txt_ids // self.config.max_token_length,
            txt_ids % self.config.max_token_length
        ], dim=-1)
        txt_ids = txt_ids[None].expand(B, -1, -1)  # [B, seq_len, 2]
        
        # Sample timesteps uniformly as in paper
        timesteps = torch.rand(B, device=self.device)
        
        # Generate conditioning vector
        y = torch.randn(B, self.config.vec_in_dim, device=self.device)
        
        # Forward pass through expert with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            pred_flow = self.expert(
                img=img_seq,           # [B, h*w, C]
                img_ids=img_ids,       # [B, h*w, 2]
                txt=clip_emb,          # [B, seq_len, dim]
                txt_ids=txt_ids,       # [B, seq_len, 2]
                timesteps=timesteps,   # [B]
                y=y,                   # [B, vec_dim]
                cluster_ids=batch['expert']  # [B]
            )
            
            # Reshape predicted flow back to image format
            pred_flow = pred_flow.view(B, h * w, -1)
            pred_flow = pred_flow.permute(0, 2, 1).view(B, -1, h, w)
            
            # Compute flow matching loss
            target_flow = self.flow_matcher.compute_target_flow(latents, timesteps)
            loss = self.flow_matcher.compute_flow_matching_loss(pred_flow, target_flow)
        
        # Optimization step
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.expert.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        
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