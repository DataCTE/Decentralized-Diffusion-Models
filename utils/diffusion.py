"""Diffusion utilities implementing paper's equations"""

import math
import torch

def get_alphas_and_betas(num_timesteps=1000, schedule_type='cosine'):
    """Paper's noise schedules from Section 3.2 and Appendix B.1"""
    if schedule_type == 'cosine':
        # Equation 4: Cosine schedule
        ts = torch.linspace(0, 1, num_timesteps + 1)
        alphas = torch.cos((ts + 0.008) / 1.008 * math.pi / 2).pow(2)  # Paper Eq.4
        alphas = alphas / alphas[0]  # Ensure alpha_0 = 1 (Paper Eq.4)
        betas = 1 - (alphas[1:] / alphas[:-1])
    else:  # Linear schedule (paper baseline comparison)
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
    
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