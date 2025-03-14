"""Diffusion utilities implementing paper's equations"""

import math
import torch
import logging
import torch.nn.functional as F

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
    """
    Apply forward diffusion to data at time t
    
    Args:
        x0: Initial data [B, C, H, W]
        t: Timestep tensor [B]
        alpha_bar: Cumulative product of alphas 
        noise: Optional pre-generated noise (if None, will sample Gaussian)
        
    Returns:
        x_t: Diffused data
        noise: Generated noise
    """
    # Extract batch size
    batch_size = x0.shape[0]
    
    # Get alpha_bar at specified timesteps
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    
    # Sample Gaussian noise if not provided
    if noise is None:
        noise = torch.randn_like(x0)
        
    # Apply forward diffusion: x_t = √(α_t)·x_0 + √(1-α_t)·ε
    mean = a_bar.sqrt() * x0
    std = (1 - a_bar).sqrt()
    x_t = mean + std * noise
    
    return x_t, noise

def ddim_step(model, x_t, t, t_next, alphas, alpha_bar, eta=0.0, text_embeddings=None):
    """
    DDIM step for deterministic diffusion sampling
    
    Args:
        model: Diffusion model
        x_t: Current latent [B, C, H, W]
        t: Current timestep [B]
        t_next: Next timestep [B]
        alphas: Alpha schedule
        alpha_bar: Cumulative product of alphas
        eta: Stochasticity parameter (0 for deterministic, 1 for stochastic)
        text_embeddings: Optional text conditioning
        
    Returns:
        x_{t-1}: Next latent
    """
    try:
        # Get model prediction
        with torch.no_grad():
            noise_pred = model(x_t, t, text_embeddings)
            
        # Get alphas for current and next timestep
        a_t = alphas[t]
        a_prev = alphas[t_next]
        a_bar_t = alpha_bar[t]
        a_bar_prev = alpha_bar[t_next]
        
        # Reshape for broadcasting
        a_t = a_t.view(-1, 1, 1, 1)
        a_prev = a_prev.view(-1, 1, 1, 1)
        a_bar_t = a_bar_t.view(-1, 1, 1, 1)
        a_bar_prev = a_bar_prev.view(-1, 1, 1, 1)
        
        # Compute predicted x0
        x0_pred = (x_t - torch.sqrt(1 - a_bar_t) * noise_pred) / torch.sqrt(a_bar_t)
        
        # Prevent extreme values for stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Compute variance
        var = eta * torch.sqrt(
            (1 - a_bar_prev) / (1 - a_bar_t) * (1 - a_t / a_bar_t)
        )
        
        # Compute direction
        dir_xt = torch.sqrt(1 - a_bar_prev - var**2) * noise_pred
        
        # Sample random noise for stochastic component
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        
        # Compute next latent
        x_prev = torch.sqrt(a_bar_prev) * x0_pred + dir_xt + var * noise
        
        return x_prev
    except Exception as e:
        logger.error(f"Error in DDIM step: {str(e)}")
        # Return input as fallback for stability
        return x_t

class DecentralizedFlowMatcher:
    """
    Implements the flow matching objective for Decentralized Diffusion Models (Paper Section 3.1)
    
    This class implements the loss function for training experts in the DDM framework.
    """
    
    def __init__(self, sigma=0.5, loss_type='huber'):
        """
        Initialize flow matcher
        
        Args:
            sigma: Flow matching sigma parameter
            loss_type: Loss function type ('mse', 'huber', or 'l1')
        """
        self.sigma = sigma
        self.loss_type = loss_type
        
    def compute_flow_matching_target(self, x0, xt, t):
        """
        Compute flow matching target according to paper Section 3.1
        
        Args:
            x0: Original data [B, C, H, W]
            xt: Noisy data at time t [B, C, H, W]
            t: Timestep tensor [B]
            
        Returns:
            Flow matching target ut(xt|x0)
        """
        # Ensure t is properly shaped for broadcasting
        t = t.reshape(-1, 1, 1, 1)
        
        # Improved numerical stability - use a smooth transition for very small t
        # This prevents division by zero and excessive magnification of noise
        eps = 1e-5
        min_t_value = 1e-4
        
        # Compute a smooth blend factor
        blend = torch.sigmoid((t - min_t_value) / eps)
        
        # For very small t, use a stable approximation
        # For t ≈ 0, the flow should approach zero (no transport needed)
        safe_t = torch.maximum(t, torch.tensor(min_t_value, device=t.device))
        denom = torch.sqrt(safe_t)
        
        # Compute standard target: u_t(x_t|x_0) = (x_0 - x_t) / sqrt(t)
        # This is the conditional flow from paper Equation 1
        standard_target = (x0 - xt) / denom
        
        # For very small t, use zero flow
        stable_target = torch.zeros_like(standard_target)
        
        # Smoothly blend between the two based on t value
        target = blend * standard_target + (1 - blend) * stable_target
        
        return target

    def compute_ensemble_flow(self, router_outputs, expert_flows):
        """
        Compute ensemble flow according to paper's Equation 4
        
        Args:
            router_outputs: Router predictions [B, K] (probabilities for each expert)
            expert_flows: List of K expert flow predictions [K, B, C, H, W]
            
        Returns:
            Ensemble flow [B, C, H, W]
        """
        # Initialize ensemble flow with zeros
        ensemble_flow = torch.zeros_like(expert_flows[0]) if expert_flows else None
        
        # Implement Equation 4 from the paper:
        # ut(xt) = sum_k (pt,Sk(xt)/pt(xt)) * [expert flow for cluster k]
        # where router_outputs represent pt,Sk(xt)/pt(xt)
        for k in range(len(expert_flows)):
            # Get router weight for expert k (reshape for broadcasting)
            router_weight = router_outputs[:, k].view(-1, 1, 1, 1)
            
            # Add weighted expert flow to ensemble
            ensemble_flow += router_weight * expert_flows[k]
            
        return ensemble_flow
    
    def compute_flow_matching_loss(self, pred, target):
        """
        Compute flow matching loss following paper Section 3.4
        
        Args:
            pred: Model prediction [B, C, H, W]
            target: Target flow [B, C, H, W]
            
        Returns:
            Loss value
        """
        # Calculate MSE, Huber, or L1 loss based on config
        if self.loss_type == 'mse':
            # MSE loss (Equation 6 in the paper)
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'huber':
            # Huber loss for robustness
            loss = F.huber_loss(pred, target, reduction='none', delta=0.1)
        elif self.loss_type == 'l1':
            # L1 loss
            loss = F.l1_loss(pred, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
            
        # Reduce along spatial and channel dimensions
        return loss.mean(dim=[1, 2, 3]).mean()

    def compute_loss(self, predictions, x0, t):
        """
        Compute full flow matching loss according to paper Section 3.2 and 3.4
        
        Args:
            predictions: Model predictions [B, C, H, W]
            x0: Original data [B, C, H, W]
            t: Timestep tensor [B]
            
        Returns:
            Loss value
        """
        # Forward process to get x_t
        alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
        sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
        noise = torch.randn_like(x0)
        xt = alpha_t * x0 + sigma_t * noise
        
        # Compute target flow field
        target = self.compute_flow_matching_target(x0, xt, t)
        
        # Compute loss
        return self.compute_flow_matching_loss(predictions, target) 