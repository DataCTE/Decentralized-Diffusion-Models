"""Diffusion process utilities for Decentralized Diffusion Models."""

import torch
import math

def get_alphas_and_betas(num_timesteps=1000, schedule_type='cosine'):
    """Compute noise schedule coefficients for diffusion"""
    if schedule_type == 'cosine':
        # Improved DDPM cosine schedule
        max_beta = 0.999
        ts = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((ts / num_timesteps + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(max_beta))
    else:  # linear schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    
    alphas = 1. - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar, betas

def forward_diffuse(x0, t, noise, alpha_bar=None):
    """
    Forward diffusion process - adds noise to images
    
    Args:
        x0: Original images [B, C, H, W]
        t: Timestep indices [B,]
        noise: Noise to add [B, C, H, W]
        alpha_bar: Precomputed cumulative product of alphas [T,]
    """
    if alpha_bar is None:
        _, alpha_bar, _ = get_alphas_and_betas()
        
    alpha_bar = alpha_bar.to(device=x0.device, dtype=x0.dtype)
    
    # Extract alpha_bar for the specific timesteps
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
    
    # Apply forward diffusion: x_t = sqrt(α_t)·x_0 + sqrt(1-α_t)·ε
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

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

class DecentralizedFlowMatcher:
    """Implements DFM objective from paper section 3.2"""
    def __init__(self, sigma=0.8, loss_type='l2'):
        """
        Initialize the Decentralized Flow Matcher
        
        Args:
            sigma: Flow matching noise scale (paper section 3.2)
            loss_type: Loss type for flow matching ('l2' or 'huber')
        """
        self.sigma = sigma
        self.loss_type = loss_type
        
    def compute_conditional_flow(self, x0, t, noise):
        """
        Implements Equation 3 from paper - computes the conditional flow field
        
        Args:
            x0: Original clean data [B, C, H, W]
            t: Normalized timesteps in [0, 1] [B,]
            noise: Random noise [B, C, H, W]
        """
        # Compute alpha_t and sigma_t using cosine schedule as in paper Section 3.2
        # alpha_t = cos(t * pi/2), sigma_t = sin(t * pi/2)
        alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
        sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
        
        # Compute noisy sample x_t = alpha_t * x_0 + sigma_t * noise (Forward process)
        x_t = alpha_t * x0 + sigma_t * noise
        
        # Compute conditional flow field (Equation 3 from paper)
        # u_t(x_t|x_0) = (x_0 - x_t) / sigma_t
        # This represents the direction from x_t toward x_0, scaled by 1/sigma_t
        return (x0 - x_t) / sigma_t
    
    def expert_loss(self, pred_flow, x0, t, noise):
        """
        Equation 6 from paper - computes the expert loss
        
        Args:
            pred_flow: Predicted flow from expert [B, C, H, W]
            x0: Original clean data [B, C, H, W]
            t: Normalized timesteps in [0, 1] [B,]
            noise: Random noise [B, C, H, W]
        """
        # Compute target flow field
        target_flow = self.compute_conditional_flow(x0, t, noise)
        
        # Apply loss function (L2 or Huber as specified in paper)
        if self.loss_type == 'l2':
            return torch.nn.functional.mse_loss(pred_flow, target_flow)
        elif self.loss_type == 'huber':
            return torch.nn.functional.smooth_l1_loss(pred_flow, target_flow)
        else:
            return torch.nn.functional.mse_loss(pred_flow, target_flow)

    def ensemble_flows(self, router_probs, expert_flows):
        """
        Implements Equation 4 from paper - combines expert flows using router probabilities
        
        Args:
            router_probs: Router probabilities [B, num_experts]
            expert_flows: List of expert flow predictions [num_experts, B, C, H, W]
        """
        # Stack flows [B, num_experts, C, H, W]
        stacked = torch.stack(expert_flows, dim=1)
        
        # Weighted sum using router probabilities (Equation 4)
        # u_t(x_t) = sum_k p_k(x_t, t) * u_t^k(x_t)
        return (router_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1) 