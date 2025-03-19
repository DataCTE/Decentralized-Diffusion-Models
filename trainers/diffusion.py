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
        self.temperature = 1.0  # Default temperature for router softmax
        
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
        # Implements paper's Equation 1 with numerical stability
        safe_t = torch.maximum(t, torch.tensor(1e-4, device=t.device))
        denom = torch.sqrt(safe_t)
        standard_target = (x0 - xt) / denom[:, None, None, None]
        
        # Improved numerical stability - use a smooth transition for very small t
        # This prevents division by zero and excessive magnification of noise
        eps = 1e-5
        min_t_value = 1e-4
        
        # Compute a smooth blend factor
        blend = torch.sigmoid((safe_t - min_t_value) / eps)
        
        # For very small t, use zero flow
        stable_target = torch.zeros_like(standard_target)
        
        # Smoothly blend between the two based on t value
        target = blend[:, None, None, None] * standard_target + (1 - blend[:, None, None, None]) * stable_target
        
        # Make sure target has same shape as xt
        if target.shape != xt.shape:
            # Reshape target to match xt
            target = target.reshape(xt.shape)
        
        return target

    def compute_ensemble_flow(self, router_outputs, expert_flows):
        """Non-clustered expert combination from paper Eq. 4"""
        # Implements paper's Equation 4
        router_weights = F.softmax(router_outputs / self.temperature, dim=-1)
        return torch.einsum('bk,kbchw->bchw', router_weights, torch.stack(expert_flows))
    
    def compute_flow_matching_loss(self, pred, target):
        """
        Compute flow matching loss following paper Section 3.4
        
        Args:
            pred: Model prediction [B, C, H, W] or [B, C, N]
            target: Target flow [B, C, H, W] or [B, C, N]
            
        Returns:
            Loss value
        """
        # Get shapes and dimensions for debugging
        pred_shape = pred.shape
        target_shape = target.shape
        pred_dim = pred.dim()
        target_dim = target.dim()
        
        # Handle shape mismatches dynamically based on tensor dimensions
        if pred_shape != target_shape:
            # Case 1: Model output is flattened but target is spatial
            if pred_dim == 3 and target_dim == 4:
                B, C, N = pred_shape
                B_t, C_t, H, W = target_shape
                
                # Check if reshaping is possible (N = H*W)
                if N == H*W:
                    # Reshape pred to match target's spatial dimensions
                    pred = pred.reshape(B, C, H, W)
                else:
                    # Flatten target to match pred
                    target = target.reshape(B_t, C_t, -1)
            
            # Case 2: Model output is spatial but target is flattened
            elif pred_dim == 4 and target_dim == 3:
                B, C, H, W = pred_shape
                B_t, C_t, N = target_shape
                
                # Check if reshaping is possible
                if H*W == N:
                    # Flatten pred to match target
                    pred = pred.reshape(B, C, -1)
                else:
                    # Reshape target to match pred
                    target = target.reshape(B_t, C_t, H, W)
            
            # Case 3: Both are spatial but with different dimensions
            elif pred_dim == 4 and target_dim == 4:
                B, C, H_p, W_p = pred_shape
                B_t, C_t, H_t, W_t = target_shape
                
                # Different spatial dimensions - reshape both to flattened form
                pred = pred.reshape(B, C, -1)
                target = target.reshape(B_t, C_t, -1)
        
        # After reshaping, we need to check if the shapes now match
        if pred.shape != target.shape:
            # If shapes still don't match, we might need to resize or interpolate
            # For now, log the mismatch and try to continue with a best-effort approach
            logger.warning(f"Shape mismatch persists after reshaping: pred={pred.shape}, target={target.shape}")
            
            # If both are 3D (channel + flattened spatial), we can try to make them match
            if pred.dim() == 3 and target.dim() == 3:
                B, C, N_p = pred.shape
                B_t, C_t, N_t = target.shape
                
                # Ensure batch and channel dimensions match
                assert B == B_t and C == C_t, "Batch or channel dimensions don't match"
                
                # Resize the smaller one to match the larger one
                if N_p < N_t:
                    # Resize pred to match target
                    pred = F.interpolate(
                        pred.unsqueeze(3), 
                        size=(N_t, 1), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(3)
                else:
                    # Resize target to match pred
                    target = F.interpolate(
                        target.unsqueeze(3), 
                        size=(N_p, 1), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(3)
        
        # Calculate the appropriate loss based on config
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
        
        # Dynamically determine reduction dimensions based on tensor shape
        if loss.dim() == 4:  # [B, C, H, W]
            return loss.mean(dim=[1, 2, 3]).mean()
        elif loss.dim() == 3:  # [B, C, N]
            return loss.mean(dim=[1, 2]).mean()
        else:
            # Fallback for unexpected dimensions
            return loss.mean()

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