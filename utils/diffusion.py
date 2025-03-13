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
    """Implements DFM objective from Paper Equation 6 & 7"""
    
    def __init__(self, sigma=0.8, loss_type='l2'):
        """
        Args:
            sigma: Noise scale from paper Appendix B.1
            loss_type: 'l2' or 'huber' as per paper recommendations
        """
        self.sigma = sigma
        self.loss_type = loss_type
        self._loss_fn = {
            'l2': torch.nn.functional.mse_loss,
            'huber': torch.nn.functional.smooth_l1_loss
        }[loss_type]

    def compute_loss(self, pred_flows, router_probs, x0, t):
        """
        Paper Eq. 7: Decentralized flow matching loss
        Args:
            pred_flows: List of expert flow predictions [E][B, C, H, W]
            router_probs: Router probability distribution [B, E]
            x0: Clean samples [B, C, H, W]
            t: Timesteps [B,]
        """
        # Paper's noise schedule (Eq. 4)
        alpha_t = torch.cos(t * math.pi/2)[:, None, None, None]
        sigma_t = torch.sin(t * math.pi/2)[:, None, None, None]
        
        # Target flow calculation (Eq. 4)
        noise = torch.randn_like(x0)
        xt = alpha_t * x0 + sigma_t * noise
        target_flow = (x0 - xt) / sigma_t
        
        # Compute expert losses (Eq. 6)
        expert_losses = torch.stack([
            self._loss_fn(pred, target_flow, reduction='none')
            for pred in pred_flows
        ], dim=1)  # [B, E, ...]
        
        # Apply router weights and reduce (Eq. 7)
        weighted_losses = router_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * expert_losses
        return weighted_losses.sum(dim=1).mean()

    def sample(self, router, experts, shape, num_steps=50, top_k=1):
        """
        Paper Algorithm 2: Decentralized sampling
        Args:
            router: Trained router model
            experts: List of expert models
            shape: Output shape [B, C, H, W]
            num_steps: Number of sampling steps
            top_k: Number of experts to use per step
        """
        device = next(router.parameters()).device
        x = torch.randn(shape, device=device)
        
        for step in reversed(range(num_steps)):
            t = torch.ones(shape[0], device=device) * step / num_steps
            
            # Get router probabilities
            with torch.no_grad():
                logits = router(x, t)
                probs = torch.softmax(logits, dim=-1)
            
            # Select top-k experts
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            
            # Compute expert predictions
            preds = []
            for expert_idx in range(len(experts)):
                mask = (top_indices == expert_idx).any(dim=-1)
                if mask.any():
                    expert_pred = experts[expert_idx](x[mask], t[mask])
                    preds.append((mask, expert_pred))
            
            # Combine predictions
            combined = torch.zeros_like(x)
            for mask, pred in preds:
                combined[mask] += pred * top_probs[mask][..., None, None, None]
            
            # Update sample (Eq. 8)
            dt = 1.0 / num_steps
            x = x + combined * dt
            
        return x 