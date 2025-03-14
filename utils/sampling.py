"""Sampling utilities for Decentralized Diffusion Models."""

import math
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np
from utils.diffusion import (
    get_alphas_and_betas, 
    get_timestep_embedding, 
    ddim_step,
    cosine_beta_schedule,
    update_sample
)

logger = logging.getLogger(__name__)

def ddm_sample(
    router, 
    experts, 
    shape, 
    steps=50, 
    top_k=1, 
    device="cuda", 
    cfg_scale=7.5,
    text_embeddings=None,
    uncond_embeddings=None,
    eta=0.0,  # 0.0 = deterministic (DDIM), 1.0 = stochastic (DDPM)
    scheduler="cosine",
    verbose=True,
    callback=None,
):
    """
    Sample from Decentralized Diffusion Models
    
    Args:
        router: Router model
        experts: List of expert models
        shape: Output shape [B, C, H, W]
        steps: Number of sampling steps
        top_k: Number of experts to use per step
        device: Device to sample on
        cfg_scale: Classifier-free guidance scale (if text_embeddings provided)
        text_embeddings: Optional text embeddings for conditioning
        uncond_embeddings: Optional unconditional embeddings for CFG
        eta: Controls the amount of noise added (0.0 = DDIM, 1.0 = DDPM)
        scheduler: Noise schedule ("cosine", "linear", etc.)
        verbose: Whether to show progress bar
        callback: Optional callback function called after each step
        
    Returns:
        Sampled tensor
    """
    # Initialize with random noise
    x = torch.randn(shape, device=device)
    
    # Precompute diffusion parameters
    alphas, alpha_bar, betas = get_alphas_and_betas(steps, scheduler)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)
    betas = betas.to(device)
    
    # Calculate timestep sequence
    timesteps = torch.linspace(0, 1, steps + 1, device=device)[:-1]
    
    # Create progress bar
    pbar = tqdm(reversed(range(steps)), total=steps, disable=not verbose)
    pbar.set_description("Sampling")
    
    # Sampling loop
    for i in pbar:
        # Current timestep
        t = torch.full((shape[0],), i/steps, device=device)
        
        # Get router probabilities
        with torch.no_grad():
            logits = router(x, t)
            probs = torch.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(1)), dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        
        # Compute and combine expert predictions
        combined = torch.zeros_like(x)
        
        for expert_idx, expert in enumerate(experts):
            # Get samples that should use this expert
            mask = (top_indices == expert_idx).any(dim=-1)
            
            if mask.any():
                # Get conditional prediction
                if text_embeddings is not None:
                    # Split batch based on mask
                    masked_x = x[mask]
                    masked_t = t[mask]
                    
                    # Get conditional and unconditional predictions for CFG
                    with torch.no_grad():
                        cond_pred = expert(masked_x, masked_t, text_embeddings[mask])
                        uncond_pred = expert(masked_x, masked_t, uncond_embeddings[mask])
                    
                    # Apply classifier-free guidance
                    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                else:
                    # Standard unconditional prediction
                    with torch.no_grad():
                        pred = expert(x[mask], t[mask])
                
                # Weight prediction by router probability and add to combined prediction
                expert_probs = top_probs[mask][top_indices[mask] == expert_idx]
                combined[mask] += pred * expert_probs.view(-1, 1, 1, 1)
        
        # Update sample with combined prediction
        if i > 0:
            # Calculate parameters for current and previous timestep
            alpha = alphas[i]
            alpha_prev = alphas[i-1]
            alpha_bar_t = alpha_bar[i]
            alpha_bar_prev = alpha_bar[i-1]
            beta = betas[i]
            
            # DDIM update step
            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * combined) / \
                torch.sqrt(alpha_bar_t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # Clip predicted x0 to prevent extreme values
            pred_x0 = pred_x0.clamp(-1, 1)
            
            # Calculate direction to xt
            dir_xt = torch.sqrt(1 - alpha_bar_prev - eta**2 * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * combined
            
            # Calculate random noise for stochastic sampling
            noise = torch.randn_like(x)
            noise_strength = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # Update xt to xt-1
            x = torch.sqrt(alpha_bar_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * pred_x0 + \
                dir_xt + \
                noise_strength * noise
        else:
            # Last step - just predict x0 directly
            x = combined
        
        # Run callback if provided
        if callback is not None:
            callback(x, i)
    
    return x

def euler_sample(
    model,
    shape,
    steps=50,
    device="cuda",
    cfg_scale=7.5,
    text_embeddings=None,
    uncond_embeddings=None,
    scheduler="cosine",
    verbose=True,
    callback=None,
):
    """
    Sample from diffusion model using Euler method
    
    Args:
        model: Diffusion model (must accept x, t, text_embeddings)
        shape: Output shape [B, C, H, W]
        steps: Number of sampling steps
        device: Device to sample on
        cfg_scale: Classifier-free guidance scale (if text_embeddings provided)
        text_embeddings: Optional text embeddings for conditioning
        uncond_embeddings: Optional unconditional embeddings for CFG
        scheduler: Noise schedule ("cosine", "linear", etc.)
        verbose: Whether to show progress bar
        callback: Optional callback function called after each step
        
    Returns:
        Sampled tensor
    """
    # Initialize with random noise
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    
    # Create progress bar
    pbar = tqdm(reversed(range(steps)), total=steps, disable=not verbose)
    pbar.set_description("Sampling")
    
    # Sampling loop
    for i in pbar:
        # Current timestep
        t = torch.full((shape[0],), i/steps, device=device)
        
        # Get model prediction
        if text_embeddings is not None:
            # Get conditional and unconditional predictions for CFG
            with torch.no_grad():
                cond_pred = model(x, t, text_embeddings)
                uncond_pred = model(x, t, uncond_embeddings)
            
            # Apply classifier-free guidance
            pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        else:
            # Standard unconditional prediction
            with torch.no_grad():
                pred = model(x, t)
        
        # Update sample with prediction using Euler method
        x = x + pred * dt
        
        # Run callback if provided
        if callback is not None:
            callback(x, i)
    
    return x

def ddim_sample(model, shape, num_steps=50, clip=None, guidance_scale=7.5, device=None):
    """
    DDIM sampling for a single diffusion model.
    
    Args:
        model: Diffusion model
        shape: Output shape
        num_steps: Number of sampling steps
        clip: CLIP encoder (optional)
        guidance_scale: Classifier-free guidance scale
        device: Device to use
    """
    # Default device
    if device is None:
        device = next(model.parameters()).device
        
    # Create noise
    x = torch.randn(shape, device=device)
    
    # Steps
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)[:-1]
    
    # Sampling loop
    for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
        # Current timestep
        t_batch = torch.full((shape[0],), t, device=device)
        
        with torch.no_grad():
            # Predict noise
            pred = model(x, t_batch)
            
            # Previous timestep
            if i == len(timesteps) - 1:
                prev_t = torch.tensor(0., device=device)
            else:
                prev_t = timesteps[i + 1]
                
            # Denoise
            x = ddim_step(x, pred, t, prev_t)
        
    return x

def flow_matching_sample(
    flow_model,
    shape,
    steps=50,
    device="cuda",
    cfg_scale=7.5,
    text_embeddings=None,
    uncond_embeddings=None,
    verbose=True,
    callback=None,
):
    """
    Sample from flow matching model
    
    Args:
        flow_model: Flow model (must accept x, t, text_embeddings)
        shape: Output shape [B, C, H, W]
        steps: Number of sampling steps
        device: Device to sample on
        cfg_scale: Classifier-free guidance scale (if text_embeddings provided)
        text_embeddings: Optional text embeddings for conditioning
        uncond_embeddings: Optional unconditional embeddings for CFG
        verbose: Whether to show progress bar
        callback: Optional callback function called after each step
        
    Returns:
        Sampled tensor
    """
    # Initialize with random noise
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    
    # Create progress bar
    pbar = tqdm(reversed(range(steps)), total=steps, disable=not verbose)
    pbar.set_description("Sampling")
    
    # Sampling loop (backward ODE integration)
    for i in pbar:
        # Current timestep
        t = torch.full((shape[0],), i/steps, device=device)
        
        # Get flow vector field
        if text_embeddings is not None:
            # Get conditional and unconditional predictions for CFG
            with torch.no_grad():
                cond_flow = flow_model(x, t, text_embeddings)
                uncond_flow = flow_model(x, t, uncond_embeddings)
            
            # Apply classifier-free guidance
            flow = uncond_flow + cfg_scale * (cond_flow - uncond_flow)
        else:
            # Standard unconditional prediction
            with torch.no_grad():
                flow = flow_model(x, t)
        
        # Update sample with flow using Euler method
        x = x + flow * dt
        
        # Run callback if provided
        if callback is not None:
            callback(x, i)
    
    return x

def classifier_guided_sample(
    model,
    classifier,
    shape,
    class_labels,
    steps=50,
    device="cuda",
    guidance_scale=7.5,
    scheduler="cosine",
    verbose=True,
    callback=None,
):
    """
    Sample from diffusion model with classifier guidance
    
    Args:
        model: Diffusion model (must accept x, t)
        classifier: Classifier model that outputs gradients for class labels
        shape: Output shape [B, C, H, W]
        class_labels: Class labels to guide sampling towards
        steps: Number of sampling steps
        device: Device to sample on
        guidance_scale: Classifier guidance scale
        scheduler: Noise schedule ("cosine", "linear", etc.)
        verbose: Whether to show progress bar
        callback: Optional callback function called after each step
        
    Returns:
        Sampled tensor
    """
    # Initialize with random noise
    x = torch.randn(shape, device=device)
    
    # Precompute diffusion parameters
    alphas, alpha_bar, betas = get_alphas_and_betas(steps, scheduler)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)
    betas = betas.to(device)
    
    # Create progress bar
    pbar = tqdm(reversed(range(steps)), total=steps, disable=not verbose)
    pbar.set_description("Sampling")
    
    # Sampling loop
    for i in pbar:
        # Current timestep
        t = torch.full((shape[0],), i/steps, device=device)
        
        # Get model prediction
        with torch.no_grad():
            model_pred = model(x, t)
        
        # Get classifier gradient
        x_in = x.detach().requires_grad_(True)
        class_pred = classifier(x_in, t, class_labels)
        
        # Calculate gradient of log probability with respect to x
        grad = torch.autograd.grad(class_pred.sum(), x_in)[0]
        
        # Apply classifier guidance
        guided_pred = model_pred + guidance_scale * grad
        
        # DDIM update step
        if i > 0:
            # Calculate parameters for current and previous timestep
            alpha_bar_t = alpha_bar[i]
            alpha_bar_prev = alpha_bar[i-1]
            
            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * guided_pred) / \
                torch.sqrt(alpha_bar_t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # Clip predicted x0 to prevent extreme values
            pred_x0 = pred_x0.clamp(-1, 1)
            
            # Calculate direction to xt
            dir_xt = torch.sqrt(1 - alpha_bar_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * guided_pred
            
            # Update xt to xt-1
            x = torch.sqrt(alpha_bar_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * pred_x0 + dir_xt
        else:
            # Last step - just predict x0 directly
            x = guided_pred
        
        # Run callback if provided
        if callback is not None:
            callback(x, i)
    
    return x

def combined_expert_sample(
    router,
    experts,
    shape,
    steps=50,
    device="cuda",
    cfg_scale=7.5,
    text_embeddings=None,
    uncond_embeddings=None,
    combine_method="weighted",
    verbose=True,
    callback=None,
):
    """
    Sample using different expert combination methods
    
    Args:
        router: Router model
        experts: List of expert models
        shape: Output shape [B, C, H, W]
        steps: Number of sampling steps
        device: Device to sample on
        cfg_scale: Classifier-free guidance scale
        text_embeddings: Optional text embeddings for conditioning
        uncond_embeddings: Optional unconditional embeddings for CFG
        combine_method: How to combine expert predictions ("weighted", "top1", "ensemble")
        verbose: Whether to show progress bar
        callback: Optional callback function called after each step
        
    Returns:
        Sampled tensor
    """
    # Initialize with random noise
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    
    # Create progress bar
    pbar = tqdm(reversed(range(steps)), total=steps, disable=not verbose)
    pbar.set_description("Sampling")
    
    # Sampling loop
    for i in pbar:
        # Current timestep
        t = torch.full((shape[0],), i/steps, device=device)
        
        # Get router probabilities
        with torch.no_grad():
            logits = router(x, t)
            probs = torch.softmax(logits, dim=-1)
        
        combined = torch.zeros_like(x)
        
        # Different combination methods
        if combine_method == "top1":
            # Use only the top expert for each sample
            top_idx = probs.argmax(dim=-1)
            
            for expert_idx, expert in enumerate(experts):
                # Get samples that should use this expert
                mask = (top_idx == expert_idx)
                
                if mask.any():
                    # Get predictions with classifier-free guidance if needed
                    if text_embeddings is not None:
                        with torch.no_grad():
                            cond_pred = expert(x[mask], t[mask], text_embeddings[mask])
                            uncond_pred = expert(x[mask], t[mask], uncond_embeddings[mask])
                        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                    else:
                        with torch.no_grad():
                            pred = expert(x[mask], t[mask])
                    
                    combined[mask] = pred
        
        elif combine_method == "ensemble":
            # Simple average of all expert predictions
            expert_preds = []
            
            for expert in experts:
                if text_embeddings is not None:
                    with torch.no_grad():
                        cond_pred = expert(x, t, text_embeddings)
                        uncond_pred = expert(x, t, uncond_embeddings)
                    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                else:
                    with torch.no_grad():
                        pred = expert(x, t)
                
                expert_preds.append(pred)
            
            # Average predictions
            combined = torch.stack(expert_preds).mean(dim=0)
        
        else:  # weighted (default)
            # Weight predictions by router probabilities
            for expert_idx, expert in enumerate(experts):
                if text_embeddings is not None:
                    with torch.no_grad():
                        cond_pred = expert(x, t, text_embeddings)
                        uncond_pred = expert(x, t, uncond_embeddings)
                    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                else:
                    with torch.no_grad():
                        pred = expert(x, t)
                
                # Weight by probability
                weight = probs[:, expert_idx].view(-1, 1, 1, 1)
                combined += weight * pred
        
        # Update sample using Euler method
        x = x + combined * dt
        
        # Run callback if provided
        if callback is not None:
            callback(x, i)
    
    return x

def get_distilled_model_prediction(distilled_model, x, t, prompt_embeds=None, cfg_scale=7.5):
    """
    Get prediction from a distilled model with classifier-free guidance
    
    Args:
        distilled_model: Distilled model
        x: Input tensor
        t: Timestep
        prompt_embeds: Text prompt embeddings (optional)
        cfg_scale: Classifier-free guidance scale
    """
    # Prediction without prompt
    uncond_pred = distilled_model(x, t)
    
    if prompt_embeds is not None and cfg_scale > 1.0:
        # Prediction with prompt
        cond_pred = distilled_model(x, t, prompt_embeds)
        
        # Combine with classifier-free guidance
        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    else:
        pred = uncond_pred
        
    return pred

def distilled_sample(distilled_model, shape, num_steps=50, prompt_embeds=None, 
                    cfg_scale=7.5, device=None, scheduler_type='ddim', 
                    noise=None, callback=None):
    """
    DDIM sampling using a distilled diffusion model.
    
    Args:
        distilled_model: Distilled diffusion model
        shape: Output shape
        num_steps: Number of sampling steps
        prompt_embeds: Text prompt embeddings (optional)
        cfg_scale: Classifier-free guidance scale
        device: Device to use
        scheduler_type: Scheduler type ('ddim', 'euler', 'dpm')
        noise: Initial noise (optional)
        callback: Callback function for intermediate steps (optional)
    """
    # Default device
    if device is None:
        device = next(distilled_model.parameters()).device
    
    # Create initial noise if not provided
    if noise is None:
        x = torch.randn(shape, device=device)
    else:
        x = noise.to(device)
    
    batch_size = shape[0]
    has_cond = prompt_embeds is not None
    
    # Calculate timesteps
    if scheduler_type == 'ddim':
        timesteps = torch.linspace(1, 0, num_steps + 1, device=device)[:-1]
    else:  # Default linear space
        timesteps = torch.linspace(1, 0, num_steps + 1, device=device)[:-1]
    
    # Progress bar
    progress = tqdm(timesteps, desc="Distilled Sampling")
    
    # Store intermediate samples if needed for callback
    intermediate_samples = []
    
    # Sampling loop
    for step_idx, t in enumerate(progress):
        # Scale timestep to model input range
        t_input = (t * 999).long()  # Scale to [0, 999]
        t_batch = torch.full((batch_size,), t_input, device=device)
        
        with torch.no_grad():
            # Get model prediction
            pred = get_distilled_model_prediction(
                distilled_model, x, t_batch, 
                prompt_embeds=prompt_embeds,
                cfg_scale=cfg_scale
            )
            
            # Previous timestep
            if step_idx == len(timesteps) - 1:
                prev_t = torch.tensor(0., device=device)
            else:
                prev_t = timesteps[step_idx + 1]
            
            # Denoise
            x = ddim_step(x, pred, t, prev_t)
            
            # Store intermediate result if callback provided
            if callback is not None:
                if step_idx % max(num_steps // 10, 1) == 0 or step_idx == len(timesteps) - 1:
                    intermediate_samples.append(x.detach().clone())
                    callback(step_idx, x)
    
    return x, intermediate_samples if callback is not None else x 