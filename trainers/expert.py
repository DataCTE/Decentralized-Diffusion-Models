"""Expert trainer for Decentralized Diffusion Models."""

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from bitsandbytes.optim import AdamW8bit
import math
import os

from models.dit import ExpertDiT
from utils.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from utils.vae import VAEWrapper
from utils.clip import CLIPTextEncoder

class ExpertTrainer:
    """Trainer for expert DiT models in DDM"""
    def __init__(self, expert_idx, config, device, rank):
        # Paper-recommended initialization (section 4.1)
        self.expert_idx = expert_idx  # Store the expert index for identification
        self.expert = ExpertDiT(config).to(device)
        self.config = config
        self.device = device
        self.rank = rank
        
        # Paper-specified optimizer settings
        self.optimizer = ZeroRedundancyOptimizer(
            self.expert.parameters(),
            optimizer_class=AdamW8bit,
            parameters_as_bucket_view=True,
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
        
        # Log initialization of this expert
        if rank == 0:
            print(f"Initialized Expert {expert_idx} on device {device}")

    def train_step(self, batch):
        """
        Implements Algorithm 1 from paper (expert training)
        
        This trains an expert model using the flow matching objective
        as described in Section 3.2 of the paper.
        """
        images = batch["image"].to(self.device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # VAE encoding (paper section 4.1)
            # The paper uses a VAE to encode images into latent space
            latents = self.vae.encode(images)
            
            # Sample random timesteps t ∈ [0, 1] (Section 3.2)
            # Note: We sample from [0, 1000) and normalize to [0, 1]
            # The paper uses uniform sampling of t in [0, 1]
            t_indices = torch.randint(0, 1000, (latents.size(0),), device=self.device)
            t = t_indices.float() / 1000.0  # Normalize to [0, 1]
            
            # Sample random noise (Section 3.2)
            # ε ~ N(0, I) as in Algorithm 1
            noise = torch.randn_like(latents)
            
            # Forward process using cosine schedule (Section 3.2)
            # x_t = alpha_t * x_0 + sigma_t * noise
            # This follows the cosine schedule in the paper:
            # alpha_t = cos(t * pi/2), sigma_t = sin(t * pi/2)
            alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
            sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
            latent_t = alpha_t * latents + sigma_t * noise
            
            # Text conditioning (paper section 4.1)
            # The paper uses CLIP text embeddings for conditioning
            text_embeds = self.clip.encode(batch["caption"])
            
            # Expert prediction of flow field u_t(x_t) (Equation 6)
            # The expert predicts the flow field at the current timestep
            pred_flow = self.expert(latent_t, t_indices, text_embeds)
            
            # Flow matching loss (Equation 6 from paper)
            # L(θ) = E_{t,x_0,ε}[||u_θ(x_t, t) - u_t(x_t|x_0)||²]
            # This computes the MSE between predicted flow and target flow
            loss = self.flow_matcher.expert_loss(pred_flow, latents, t, noise)

        # Optimization following paper training details (Section 4.1)
        # The paper uses AdamW with weight decay and gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.expert.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, save_dir, step):
        """Save a checkpoint for this expert"""
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = f"{save_dir}/expert_{self.expert_idx}_step{step}.pt"
            torch.save(self.expert.state_dict(), checkpoint_path)
            return checkpoint_path
        return None
    
    def forward_diffuse(self, x0, t, noise):
        """Forward diffusion process with precomputed alpha_bar"""
        alpha_bar = self.alpha_bar.to(device=self.device, dtype=x0.dtype)
        
        # Extract alpha_bar for the specific timesteps
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
        
        # Apply forward diffusion: x_t = sqrt(α_t)·x_0 + sqrt(1-α_t)·ε
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise 