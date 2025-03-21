"""Sampling utilities for Decentralized Diffusion Models."""

import torch
import logging
from tqdm import tqdm
from trainers.diffusion import (
    get_alphas_and_betas, 
    ddim_step
)
import torch.nn.functional as F
from collections import defaultdict
from config import get_config
import torch.distributed as dist

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
    expert_cache_manager=None,
    temperature=1.0,
    config=None,
):
    """
    Sample from Decentralized Diffusion Models as described in paper Section 3.5
    
    Args:
        router: Router model
        experts: Dict of expert models {expert_idx: model}
        shape: Output shape [B, C, H, W]
        steps: Number of sampling steps
        top_k: Number of experts to use per step (0 for full ensemble)
        device: Device to sample on
        cfg_scale: Classifier-free guidance scale
        text_embeddings: Optional text embeddings for conditioning
        uncond_embeddings: Optional unconditional embeddings for CFG
        eta: Controls the amount of noise added (0.0 = DDIM, 1.0 = DDPM)
        scheduler: Noise schedule ("cosine", "linear", etc.)
        verbose: Whether to show progress bar
        callback: Optional callback function called after each step
        expert_cache_manager: Optional ExpertCacheManager for efficient expert loading
        temperature: Temperature for router softmax
        config: Configuration object
        
    Returns:
        Sampled batch of images
    """
    # Paper-recommended optimizations
    if config is None:
        raise ValueError("Config must be provided for DDM sampling")
        
    # Initialize from random noise
    x = torch.randn(shape, device=device)
    
    # Get batch size from shape
    batch_size = shape[0]
    
    # Paper's recommended noise schedule
    alphas, alpha_bar, betas = get_alphas_and_betas(steps, "cosine")
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)
    
    # Track expert usage as described in paper Appendix C.2
    expert_usage = defaultdict(int)
    
    # Sampling loop with paper's recommended modifications
    for t in tqdm(range(steps), disable=not verbose):
        timestep = torch.full((batch_size,), t, device=device)
        
        # Get router predictions (paper Eq. 7)
        router_logits = router(x, timestep, text_embeddings)
        router_weights = F.softmax(router_logits / temperature, dim=-1)
        
        # Select top-k experts per sample (paper Section 3.5)
        expert_weights, expert_indices = torch.topk(router_weights, top_k, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Get unique experts needed for this step
        selected_experts = torch.unique(expert_indices)
        
        # Get predictions from required experts
        expert_preds = []
        for expert_idx in selected_experts:
            expert = experts[expert_idx]
            mask = (expert_indices == expert_idx).any(dim=1)
            
            # Get conditional and unconditional predictions
            if text_embeddings is not None and cfg_scale > 1.0:
                uncond_pred = expert(x[mask], timestep[mask], uncond_embeddings[mask])
                cond_pred = expert(x[mask], timestep[mask], text_embeddings[mask])
                pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            else:
                pred = expert(x[mask], timestep[mask], text_embeddings[mask] if text_embeddings else None)
            
            expert_preds.append(pred)
            expert_usage[expert_idx.item()] += mask.sum().item()

        # Combine predictions using router weights (paper Eq. 4)
        combined_pred = torch.zeros_like(x)
        for batch_idx in range(batch_size):
            for k in range(top_k):
                expert_idx = expert_indices[batch_idx, k]
                weight = expert_weights[batch_idx, k]
                pred = expert_preds[selected_experts.tolist().index(expert_idx)][batch_idx]
                combined_pred[batch_idx] += weight * pred

        # Paper's recommended DDIM update step
        x = ddim_step(
            lambda x_t, t, c: combined_pred,
            x,
            timestep,
            torch.full((batch_size,), t+1, device=device) if t < steps-1 else None,
            alphas,
            alpha_bar,
            eta=eta
        )

    return x

def distilled_sample(distilled_model, shape, num_steps=50, prompt_embeds=None, 
                    cfg_scale=7.5, device=None, eta=0.0, 
                    noise=None, callback=None):
    """
    Sampling using a distilled model (paper Section 3.6)
    
    Args:
        distilled_model: Distilled diffusion model
        shape: Output shape
        num_steps: Number of sampling steps
        prompt_embeds: Text prompt embeddings (optional)
        cfg_scale: Classifier-free guidance scale
        device: Device to use
        eta: Controls the amount of noise added
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
    
    # Setup noise schedule
    alphas, alpha_bar, betas = get_alphas_and_betas(num_steps)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)
    
    # Progress bar
    progress = tqdm(range(num_steps), desc="Distilled Sampling")
    
    # Sampling loop
    for t in progress:
        # Current timestep
        timestep = torch.full((batch_size,), t, device=device)
        
        with torch.no_grad():
            # Get model prediction with conditional guidance if provided
            if prompt_embeds is not None and cfg_scale > 1.0:
                # Run both conditional and unconditional forward passes
                unconditional_latents = x.clone()
                conditional_latents = x.clone()
                
                # Get unconditional prediction
                uncond_pred = distilled_model(unconditional_latents, timestep)
                
                # Get conditional prediction
                cond_pred = distilled_model(conditional_latents, timestep, prompt_embeds)
                
                # Apply classifier-free guidance
                pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            else:
                # Simple forward pass without guidance
                pred = distilled_model(x, timestep)
            
            # DDIM step
            next_timestep = torch.full((batch_size,), t+1, device=device) if t < num_steps-1 else None
            
            if next_timestep is not None:
                # Update sample using DDIM step
                x = ddim_step(
                    lambda x_t, t, c: pred,  # Use precomputed prediction
                    x, 
                    timestep,
                    next_timestep,
                    alphas,
                    alpha_bar,
                    eta=eta
                )
            
            # Optional callback for visualization
            if callback and t % 5 == 0:
                callback(x, t)
    
    return x

def quantize_model_for_inference(model, dtype=torch.float16):
    """
    Quantize a model for memory-efficient inference
    
    Args:
        model: The model to quantize
        dtype: Target dtype for quantization (default: float16)
        
    Returns:
        Quantized model
    """
    # Simple quantization by converting to half precision
    model = model.to(dtype)
    logger.info(f"Converted model to {dtype} for efficient inference")
    return model

def ddm_sample_optimized(router, experts, shape, steps=50, top_k=1, **kwargs):
    """
    Memory-optimized version of ddm_sample that quantizes models
    
    Args:
        Same as ddm_sample but applies quantization
        
    Returns:
        Sampled tensor
    """
    # Quantize router for efficient inference
    quantized_router = quantize_model_for_inference(router)
    
    # Quantize experts
    quantized_experts = {}
    for idx, expert in experts.items():
        if expert is not None:
            quantized_experts[idx] = quantize_model_for_inference(expert)
    
    # Call regular sampling with quantized models
    return ddm_sample(
        router=quantized_router,
        experts=quantized_experts,
        shape=shape,
        steps=steps,
        top_k=top_k,
        **kwargs
    ) 