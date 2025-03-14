"""Diffusion utilities implementing paper's equations"""

import math
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    
    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
    return embedding

def get_alphas_and_betas(num_timesteps=1000, schedule_type='cosine'):
    """Paper's noise schedules from Section 3.2 and Appendix B.1"""
    if schedule_type == 'cosine':
        # Equation 4: Cosine schedule
        ts = torch.linspace(0, 1, num_timesteps + 1)
        alphas = torch.cos((ts + 0.008) / 1.008 * math.pi / 2).pow(2)  # Paper Eq.4
        alphas = alphas / alphas[0]  # Ensure alpha_0 = 1 (Paper Eq.4)
        betas = 1 - (alphas[1:] / alphas[:-1])
    elif schedule_type == 'linear':
        # Linear schedule (paper baseline comparison)
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_type == 'squared_linear':
        # Squared linear schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")
    
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar, betas

def forward_diffuse(x0, t, alpha_bar, noise=None):
    """Paper Equation 4: Forward diffusion process"""
    if noise is None:
        noise = torch.randn_like(x0)
        
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

def get_noise_schedule(timesteps, schedule_type='cosine'):
    """Get noise schedule for given timesteps"""
    if schedule_type == 'cosine':
        return torch.cos(timesteps * math.pi / 2)
    else:
        return 1 - timesteps

def get_variance_schedule(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='cosine'):
    """Get variance schedule for given parameters"""
    if schedule_type == 'cosine':
        # Cosine schedule
        steps = torch.linspace(0, 1, num_timesteps + 1, dtype=torch.float32)
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2).pow(2)
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, min=0, max=0.999)
    elif schedule_type == 'linear':
        # Linear schedule
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    elif schedule_type == 'squared_linear':
        # Squared linear schedule
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule as used in the paper.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)

def ddim_step(x, predicted_noise, t, t_prev, alpha_bar=None):
    """
    DDIM step for deterministic sampling.
    
    Args:
        x: Current noisy sample
        predicted_noise: Predicted noise
        t: Current timestep
        t_prev: Previous timestep
        alpha_bar: Optional precomputed alpha_bar values
    """
    if alpha_bar is None:
        # Default to cosine schedule if not provided
        steps = 1000
        alpha_bar = torch.cos(((torch.arange(steps + 1) / steps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
    
    # Get alpha values
    t_idx = int(t.item() * 999)
    t_prev_idx = int(t_prev.item() * 999)
    
    # Clamp indices
    t_idx = min(999, max(0, t_idx))
    t_prev_idx = min(999, max(0, t_prev_idx))
    
    alpha_bar_t = alpha_bar[t_idx]
    alpha_bar_prev = alpha_bar[t_prev_idx]
    
    # Scale predicted noise
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    
    # Predict x0
    pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / torch.sqrt(alpha_bar_t)
    
    # Sample
    pred_coef = torch.sqrt(alpha_bar_prev)
    pred_noise_coef = torch.sqrt(1 - alpha_bar_prev)
    x_prev = pred_coef * pred_x0 + pred_noise_coef * predicted_noise
    
    return x_prev

def update_sample(x_t, pred_noise, t, alphas=None, alpha_bar=None, betas=None, steps=1000):
    """
    Update sample with predicted noise using DDIM sampler (deterministic)
    
    Args:
        x_t: Current noisy sample [B, C, H, W]
        pred_noise: Predicted noise [B, C, H, W]
        t: Current timestep [B,]
        alphas, alpha_bar, betas: Precomputed diffusion coefficients
    """
    if alphas is None or alpha_bar is None or betas is None:
        alphas, alpha_bar, betas = get_alphas_and_betas(steps)
    
    # Move tensors to correct device and dtype
    device = x_t.device
    dtype = x_t.dtype
    alphas = alphas.to(device=device, dtype=dtype)
    alpha_bar = alpha_bar.to(device=device, dtype=dtype)
    betas = betas.to(device=device, dtype=dtype)
    
    # Previous timestep
    prev_t = (t - 1).clamp(min=0)
    
    # Get alpha values for current and previous timestep
    alpha_t = alphas[t]
    alpha_prev = alphas[prev_t]
    alpha_bar_t = alpha_bar[t]
    alpha_bar_prev = alpha_bar[prev_t]
    
    # Reshaping for broadcasting
    alpha_t = alpha_t[:, None, None, None]
    alpha_prev = alpha_prev[:, None, None, None]
    alpha_bar_t = alpha_bar_t[:, None, None, None]
    alpha_bar_prev = alpha_bar_prev[:, None, None, None]
    
    # Predict x0 from xt and predicted noise
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
    
    # DDIM update (deterministic sampling)
    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
    sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev)
    
    # Compute "direction pointing to xt"
    dir_xt = torch.sqrt(1. - alpha_bar_prev - sqrt_one_minus_alpha_bar_prev**2) * pred_noise
    
    # Update xt to xt-1
    x_prev = sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_alpha_bar_prev * pred_noise + dir_xt
    
    # Return properly thresholded results
    return x_prev.clamp(-1., 1.)

def get_schedule(timesteps, schedule_type='cosine'):
    """Paper's noise schedules from Section 3.2"""
    if schedule_type == 'cosine':
        return torch.cos(timesteps * math.pi/2)
    else:
        return 1 - timesteps 

class DecentralizedFlowMatcher:
    """Implements paper's Equations 6-8 for DFM"""
    
    def __init__(self, sigma=0.8, loss_type='l2'):
        """
        Args:
            sigma: Noise scale (paper Appendix B.1)
            loss_type: 'l2' or 'huber' (paper Section 4.1)
        """
        self.sigma = sigma
        self.loss_fn = {
            'l2': torch.nn.functional.mse_loss,
            'huber': torch.nn.functional.smooth_l1_loss
        }[loss_type]

    def compute_loss(self, pred_flows, router_probs, x0, t, alpha_bar):
        """
        Paper Equation 7: Decentralized flow matching loss
        Args:
            pred_flows: List of expert predictions [E][B, C, H, W]
            router_probs: Router probabilities [B, E]
            x0: Clean samples [B, C, H, W]
            t: Timesteps [B,]
            alpha_bar: Precomputed alpha_bar schedule
        """
        # Sample noise and create x_t
        noise = torch.randn_like(x0)
        x_t = forward_diffuse(x0, t, alpha_bar, noise)
        
        # Compute target flow (Equation 4 derivative)
        sigma_t = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
        target_flow = (x0 - x_t) / sigma_t
        
        # Calculate per-expert losses
        expert_losses = torch.stack([
            self.loss_fn(pred, target_flow, reduction='none')
            for pred in pred_flows
        ], dim=1)  # [B, E, ...]
        
        # Apply router weights and reduce
        weighted_losses = router_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * expert_losses
        return weighted_losses.sum(dim=1).mean()

    def compute_flow_matching_target(self, x0, x_t, t):
        """
        Compute the flow matching target v_t(x_t)
        
        Args:
            x0: Original clean data [B, C, H, W]
            x_t: Noisy data at timestep t [B, C, H, W]
            t: Timestep values [B,]
            
        Returns:
            Target flow field [B, C, H, W]
        """
        # Compute sigma_t
        sigma_t = torch.sin(t * math.pi/2)[:, None, None, None]
        
        # Compute target flow
        target_flow = (x0 - x_t) / sigma_t
        return target_flow
        
    def compute_flow_matching_loss(self, pred_flow, target_flow):
        """
        Compute flow matching loss between predicted and target flows
        
        Args:
            pred_flow: Predicted flow field [B, C, H, W]
            target_flow: Target flow field [B, C, H, W]
            
        Returns:
            Loss value
        """
        return self.loss_fn(pred_flow, target_flow)

    def sample(self, router, experts, shape, alpha_bar, steps=50, top_k=1):
        """
        Paper Algorithm 2: Decentralized sampling
        Args:
            shape: Output shape [B, C, H, W]
            alpha_bar: Precomputed schedule
            steps: Number of sampling steps
            top_k: Experts to use per step
        """
        device = next(router.parameters()).device
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps
        
        for step in reversed(range(steps)):
            t = torch.full((shape[0],), step/steps, device=device)
            
            # Get router probabilities
            with torch.no_grad():
                logits = router(x, t)
                probs = torch.softmax(logits, dim=-1)
            
            # Select top-k experts
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            
            # Compute and combine expert predictions
            combined = torch.zeros_like(x)
            for expert_idx, expert in enumerate(experts):
                mask = (top_indices == expert_idx).any(dim=-1)
                if mask.any():
                    pred = expert(x[mask], t[mask])
                    combined[mask] += pred * top_probs[mask][..., None, None, None]
            
            # Update sample (Equation 8 Euler step)
            x = x + combined * dt
            
        return x 