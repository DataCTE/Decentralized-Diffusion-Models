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
    """Trainer for expert DiT models in DDM"""
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
        # Paper's per-expert loss calculation
        x0 = batch["latent"].to(self.device)
        t = torch.rand(x0.size(0), device=self.device)  # t ~ U[0,1]
        
        # Forward process using paper's cosine schedule
        alpha_t = torch.cos((t + 0.008)/1.008 * math.pi/2).pow(2)[:,None,None,None]
        sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
        noise = torch.randn_like(x0)
        xt = alpha_t * x0 + sigma_t * noise
        
        # Expert forward pass with cluster conditioning
        pred = self.expert(xt, (t * 1000).long(), text_embeds=None, cluster_ids=batch["expert"])
        
        # Flow matching target calculation
        target = (x0 - alpha_t * xt) / (sigma_t**2 + 1e-7)
        return F.mse_loss(pred, target)

    def train_step(self, batch):
        """Execute a single training step"""
        self.expert.train()
        self.optimizer.zero_grad()
        
        # Extract and reshape inputs from batch
        latents = batch['latent']  # [B, 1, C, H, W]
        B, _, C, H, W = latents.shape
        latents = latents.squeeze(1)  # Remove sequence dim for latents
        
        # Get CLIP embeddings and reshape
        clip_emb = batch['clip_embedding']  # [B, 1, seq_len, dim]
        clip_emb = clip_emb.squeeze(1)  # [B, seq_len, dim]
        
        # Create position IDs for text and image
        txt_ids = torch.arange(clip_emb.size(1), device=self.device)[None].repeat(B, 1)
        txt_ids = torch.stack([txt_ids // self.config.max_token_length, txt_ids % self.config.max_token_length], dim=-1)
        
        img_size = H // self.config.patch_size
        img_ids = torch.arange(img_size * img_size, device=self.device)[None].repeat(B, 1)
        img_ids = torch.stack([img_ids // img_size, img_ids % img_size], dim=-1)
        
        # Sample random timesteps
        timesteps = torch.rand(B, device=self.device)
        
        # Generate random noise for conditioning
        y = torch.randn(B, self.config.vec_in_dim, device=self.device)
        
        # Reshape latents to sequence format
        img_seq = latents.view(B, C, -1).permute(0, 2, 1)  # [B, L, C]
        
        # Forward pass through expert
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            pred_flow = self.expert(
                img=img_seq,
                img_ids=img_ids,
                txt=clip_emb,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                cluster_ids=batch['expert']
            )
            
            # Compute target flow
            target_flow = self.flow_matcher.compute_target_flow(latents, timesteps)
            
            # Compute loss
            loss = self.flow_matcher.compute_flow_matching_loss(pred_flow, target_flow)
        
        # Backward pass and optimization
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
    
    def forward_diffuse(self, x0, t, noise):
        """Forward diffusion process with precomputed alpha_bar"""
        alpha_bar = self.alpha_bar.to(device=self.device, dtype=x0.dtype)
        
        # Extract alpha_bar for the specific timesteps
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
        
        # Apply forward diffusion: x_t = sqrt(α_t)·x_0 + sqrt(1-α_t)·ε
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

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