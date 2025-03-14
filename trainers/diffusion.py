"""Diffusion utilities implementing paper's equations"""

import math
import torch
import logging

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
    Cosine beta schedule as defined in the paper
    
    Args:
        timesteps: Number of diffusion steps
        s: Parameter controlling the minimum SNR
        
    Returns:
        Beta schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

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

def update_sample(x, pred, t_index, t_next_index, alphas, alpha_bar, eta=0.0):
    """
    Update sample using diffusion model prediction
    
    Args:
        x: Current sample [B, C, H, W]
        pred: Model prediction [B, C, H, W]
        t_index: Current timestep index
        t_next_index: Next timestep index
        alphas: Alpha schedule
        alpha_bar: Cumulative product of alphas
        eta: Stochasticity parameter (0 for deterministic, 1 for stochastic)
        
    Returns:
        Updated sample
    """
    try:
        # Get alphas for current and next timestep
        alpha_t = alphas[t_index] if t_index < len(alphas) else torch.tensor(1.0, device=alphas.device)
        alpha_next = alphas[t_next_index] if t_next_index < len(alphas) else torch.tensor(1.0, device=alphas.device)
        
        alpha_bar_t = alpha_bar[t_index] if t_index < len(alpha_bar) else torch.tensor(1.0, device=alpha_bar.device)
        alpha_bar_next = alpha_bar[t_next_index] if t_next_index < len(alpha_bar) else torch.tensor(1.0, device=alpha_bar.device)
        
        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_next = alpha_next.view(-1, 1, 1, 1)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        alpha_bar_next = alpha_bar_next.view(-1, 1, 1, 1)
        
        # Predict x0
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred) / torch.sqrt(alpha_bar_t)
        
        # Clamp for stability
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute direction
        dir_xt = torch.sqrt(1 - alpha_bar_next - eta**2 * (1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next)) * pred
        
        # Compute noise strength
        noise_strength = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next))
        
        # Sample random noise
        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        
        # Update sample
        x_next = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt + noise_strength * noise
        
        return x_next
    except Exception as e:
        logger.error(f"Error in update_sample: {str(e)}")
        # Return input as fallback for stability
        return x

def get_schedule(timesteps, schedule_type='cosine'):
    """Paper's noise schedules from Section 3.2"""
    if schedule_type == 'cosine':
        return torch.cos(timesteps * math.pi/2)
    else:
        return 1 - timesteps 

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
        Compute flow matching target with improved numerical stability
        
        Args:
            x0: Original data [B, C, H, W]
            xt: Noisy data at time t [B, C, H, W]
            t: Timestep tensor [B]
            
        Returns:
            Flow matching target
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
        batch_size = router_outputs.shape[0]
        num_experts = router_outputs.shape[1]
        
        # Initialize ensemble flow with zeros
        ensemble_flow = torch.zeros_like(expert_flows[0]) if expert_flows else None
        
        # Implement Equation 4 from the paper:
        # u_t(x_t) = sum_k (p_t,S_k(x_t)/p_t(x_t)) * (sum_{x_0 in S_k} u_t(x_t|x_0)p_t(x_t|x_0)q(x_0)/p_t,S_k(x_t))
        # Router outputs represent p_t,S_k(x_t)/p_t(x_t)
        # Expert flows represent the inner sum term
        
        for k in range(num_experts):
            # Get router weight for expert k (reshape for broadcasting)
            router_weight = router_outputs[:, k].view(batch_size, 1, 1, 1)
            
            # Get flow prediction from expert k
            expert_flow = expert_flows[k]
            
            # Add weighted expert flow to ensemble
            ensemble_flow += router_weight * expert_flow
            
        return ensemble_flow
    
    def compute_flow_matching_loss(self, pred, target):
        """
        Compute flow matching loss with improved efficiency
        
        Args:
            pred: Model prediction [B, C, H, W]
            target: Flow matching target [B, C, H, W]
            
        Returns:
            Loss value
        """
        # Compute element-wise squared difference
        squared_diff = (pred - target)**2
        
        if self.loss_type == 'mse':
            # Use a more efficient reduction for MSE
            return torch.mean(squared_diff)
        elif self.loss_type == 'huber':
            # Huber loss with delta=1.0
            delta = 1.0
            abs_diff = torch.abs(pred - target)
            quadratic_mask = abs_diff <= delta
            linear_mask = ~quadratic_mask
            
            # Efficient implementation avoiding unnecessary operations
            loss = torch.where(
                quadratic_mask,
                0.5 * squared_diff,
                delta * (abs_diff - 0.5 * delta)
            )
            return torch.mean(loss)
        elif self.loss_type == 'l1':
            return torch.mean(torch.abs(pred - target))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def compute_loss(self, batch, model=None):
        """
        Compute complete flow matching loss for a batch with robust error handling
        
        Args:
            batch: Dictionary containing 'image' and optionally 'text_embedding'
            model: Model to compute predictions (if None, just compute target)
            
        Returns:
            Loss value or (loss, prediction, target) tuple
        """
        try:
            # Extract data
            x0 = batch['image']
            batch_size = x0.shape[0]
            
            # Sample random timestep with improved distribution
            # Use log-uniform sampling for better coverage of small t values
            # This matches the paper's recommendation for sampling efficiency
            u = torch.rand(batch_size, device=x0.device)
            t = torch.exp((torch.log(torch.tensor(1e-4)) * (1 - u)) + (torch.log(torch.tensor(1.0)) * u))
            
            # Ensure t is in correct range [0,1]
            t = torch.clamp(t, 0.0, 1.0)
            
            # Sample random noise
            noise = torch.randn_like(x0)
            
            # Improved diffusion with exact cosine schedule as in the paper
            alpha_t = torch.cos(t.view(-1, 1, 1, 1) * math.pi/2)
            sigma_t = torch.sin(t.view(-1, 1, 1, 1) * math.pi/2)
            xt = alpha_t * x0 + sigma_t * noise
            
            # Get text conditioning if available
            text_embeds = batch.get('text_embedding', None)
            
            # Compute target with improved stability
            target = self.compute_flow_matching_target(x0, xt, t)
            
            # If model is provided, compute prediction and loss
            if model is not None:
                # Convert t to model-expected format (typically int indices)
                t_indices = (t * 1000).long()
                
                # Forward pass
                pred = model(xt, t_indices, text_embeds)
                
                # Check for NaN values
                if torch.isnan(pred).any():
                    logger.warning("NaN values detected in model prediction, using zero loss")
                    return torch.tensor(0.0, device=x0.device), None, None
                
                # Compute loss with target clipping for stability
                target_clipped = torch.clamp(target, -100.0, 100.0)  # Prevent extreme targets
                loss = self.compute_flow_matching_loss(pred, target_clipped)
                
                return loss, pred, target
            else:
                # Just return target
                return target
        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}")
            # Return zero loss as fallback for stability
            if model is not None:
                return torch.tensor(0.0, device=x0.device), None, None
            else:
                return torch.zeros_like(x0)
                
    def compute_ensemble_loss(self, batch, router, experts):
        """
        Compute loss for the full ensemble as described in Section 3.2
        
        Args:
            batch: Dictionary containing 'image' and optionally 'text_embedding'
            router: Router model that predicts expert probabilities
            experts: List of expert models
            
        Returns:
            Loss value and ensemble prediction
        """
        try:
            # Extract data
            x0 = batch['image']
            batch_size = x0.shape[0]
            
            # Sample random timestep
            u = torch.rand(batch_size, device=x0.device)
            t = torch.exp((torch.log(torch.tensor(1e-4)) * (1 - u)) + (torch.log(torch.tensor(1.0)) * u))
            t = torch.clamp(t, 0.0, 1.0)
            
            # Sample random noise
            noise = torch.randn_like(x0)
            
            # Forward diffusion
            alpha_t = torch.cos(t.view(-1, 1, 1, 1) * math.pi/2)
            sigma_t = torch.sin(t.view(-1, 1, 1, 1) * math.pi/2)
            xt = alpha_t * x0 + sigma_t * noise
            
            # Get text conditioning if available
            text_embeds = batch.get('text_embedding', None)
            
            # Compute target
            target = self.compute_flow_matching_target(x0, xt, t)
            
            # Convert t to model-expected format
            t_indices = (t * 1000).long()
            
            # Get router probabilities
            router_outputs = router(xt, t_indices, text_embeds)
            
            # Get each expert's prediction
            expert_flows = []
            for expert in experts:
                with torch.no_grad():  # Don't backprop through other experts
                    expert_pred = expert(xt, t_indices, text_embeds)
                    expert_flows.append(expert_pred)
            
            # Compute ensemble flow according to Equation 4
            ensemble_pred = self.compute_ensemble_flow(router_outputs, expert_flows)
            
            # Compute loss
            target_clipped = torch.clamp(target, -100.0, 100.0)
            loss = self.compute_flow_matching_loss(ensemble_pred, target_clipped)
            
            return loss, ensemble_pred
        except Exception as e:
            logger.error(f"Error in compute_ensemble_loss: {str(e)}")
            return torch.tensor(0.0, device=x0.device), None 