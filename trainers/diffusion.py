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
        # Get model prediction first
        with torch.no_grad():
            noise_pred = model(x_t, t, text_embeddings)
            
        # Now safe to print noise_pred details
        #print("Shape of t:", t.shape)
        #print("Values of t:", t)
        #if t_next is not None:
            #print("Shape of t_next:", t_next.shape)
            #print("Values of t_next:", t_next)

        #print("dtype of x_t:", x_t.dtype)
        #print("dtype of noise_pred:", noise_pred.dtype)
        #print("Shape of noise_pred:", noise_pred.shape)
        
        # Get alphas for current and next timestep
        a_t = alphas[t]
        a_prev = alphas[t_next] if t_next is not None else alphas[t]
        a_bar_t = alpha_bar[t]
        a_bar_prev = alpha_bar[t_next] if t_next is not None else alpha_bar[t]
        
        # Reshape for broadcasting
        a_t = a_t.view(-1, 1, 1, 1)
        a_prev = a_prev.view(-1, 1, 1, 1)
        a_bar_t = a_bar_t.view(-1, 1, 1, 1)
        a_bar_prev = a_bar_prev.view(-1, 1, 1, 1)

        #print("Shape of x_t:", x_t.shape)
        #print("Shape of noise_pred:", noise_pred.shape)
        #print("Shape of a_bar_t:", a_bar_t.shape)
        #print("Shape of (1 - a_bar_t).sqrt():", (1 - a_bar_t).sqrt().shape)

        # Compute predicted x0
        try:
            x0_pred = (x_t - torch.sqrt(1 - a_bar_t) * noise_pred) / torch.sqrt(a_bar_t)
        except RuntimeError as e:
            print("Error in x0_pred calculation:", e)
            return x_t
        
        # Prevent extreme values for stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Compute variance
        try:
            #print("Shape of a_bar_prev:", a_bar_prev.shape)
            #print("Shape of a_bar_t:", a_bar_t.shape)
            #print("Shape of a_t:", a_t.shape)
            var = eta * torch.sqrt(
                (1 - a_bar_prev) / (1 - a_bar_t) * (1 - a_t / a_bar_t)
            )
        except RuntimeError as e:
            print("Error in var calculation:", e)
            return x_t
        
        # Compute direction
        try:
            dir_xt = torch.sqrt(1 - a_bar_prev - var**2) * noise_pred
        except RuntimeError as e:
            print("Error in dir_xt calculation:", e)
            return x_t
        
        # Sample random noise for stochastic component
        try:
            noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        except RuntimeError as e:
            print("Error in noise calculation:", e)
            return x_t
        
        # Compute next latent
        try:
            x_prev = torch.sqrt(a_bar_prev) * x0_pred + dir_xt + var * noise
        except RuntimeError as e:
            print("Error in x_prev calculation:", e)
            return x_t
        
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
        self.temperature = 2.0  # Start with higher temperature
        self.temp_decay = 0.99995
        
    def compute_flow_matching_target(self, x0, xt, t):
        """Implements paper Eq.1 with numerical stability"""
        # Expand timestep tensors to match input dimensions
        alpha_t = torch.cos(t * math.pi/2).view(-1, 1, 1, 1)
        sigma_t = torch.sin(t * math.pi/2).view(-1, 1, 1, 1)
        
        return (x0 - alpha_t * xt) / (sigma_t ** 2 + 1e-7)

    def compute_ensemble_flow(self, router_outputs, expert_flows):
        """Non-clustered expert combination from paper Eq. 4"""
        # Implements paper's Equation 4
        router_weights = F.softmax(router_outputs / self.temperature, dim=-1)
        return torch.einsum('bk,kbchw->bchw', router_weights, torch.stack(expert_flows))
    
    def compute_flow_matching_loss(self, pred, target):
        """Handle dynamic spatial dimensions through resizing"""
        # Get target dimensions from x0
        _, _, H_target, W_target = target.shape
        
        # Resize prediction to match target spatial dimensions
        pred_resized = torch.nn.functional.interpolate(
            pred, 
            size=(H_target, W_target), 
            mode='bilinear', 
            align_corners=False
        )
        
        if self.loss_type == 'huber':
            return F.huber_loss(pred_resized, target, delta=0.1, reduction='mean')
        elif self.loss_type == 'mse':
            return F.mse_loss(pred_resized, target, reduction='mean')
        elif self.loss_type == 'l1':
            return F.l1_loss(pred_resized, target, reduction='mean')
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

    def compute_loss(self, pred_flow, x0, xt, t):
        """Only keep this version"""
        target = self.compute_flow_matching_target(x0, xt, t)
        return self.compute_flow_matching_loss(pred_flow, target) 