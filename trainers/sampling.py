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
    num_steps=50,  # Renamed from steps for consistency
    device=None,    # Get device from router instead of parameter
    cfg_scale=7.5,
    text_embeddings=None,
    uncond_embeddings=None,
    eta=0.0,
    verbose=True,
    temperature=1.0,
    top_k=1,  # Add top_k parameter from paper
):
    """Naive DDM sampling following paper's simple example"""
    # Auto-detect device if not specified
    if device is None:
        device = next(router.parameters()).device
    
    # Initialize from random noise with proper batch dimension
    x = torch.randn(shape, device=device)
    num_clusters = len(experts)
    
    # Get paper-recommended cosine schedule (remove scheduler parameter)
    alphas, alpha_bar, _ = get_alphas_and_betas(num_steps, "cosine")
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    # Pre-process embeddings for consistent batch dimensions
    batch_size = shape[0]
    if text_embeddings is not None:
        text_embeddings = text_embeddings[:batch_size]
        uncond_embeddings = uncond_embeddings[:batch_size] if uncond_embeddings is not None else None

    for t in tqdm(range(num_steps), disable=not verbose):
        timestep = torch.full((x.size(0),), t, device=device)
        
        # Get router predictions and select top-k
        router_logits = router(x, timestep, text_embeddings)
        router_weights = F.softmax(router_logits / temperature, dim=-1)
        
        # Paper's efficient top-k selection (Section 3.5)
        topk_weights, topk_indices = router_weights.topk(top_k, dim=-1)
        topk_weights = F.softmax(topk_weights / temperature, dim=-1)

        combined_pred = torch.zeros_like(x)
        for i in range(top_k):
            # Process each expert in current top-k position
            cluster_indices = topk_indices[:, i]
            unique_experts = torch.unique(cluster_indices)
            
            for expert_idx in unique_experts:
                # Mask for samples using this expert
                mask = cluster_indices == expert_idx
                if not mask.any():
                    continue
                
                expert = experts[expert_idx.item()]
                x_masked = x[mask]
                t_masked = timestep[mask]
                
                # Paper's efficient CFG implementation (Appendix B)
                if text_embeddings is not None and cfg_scale > 1.0:
                    emb_masked = text_embeddings[mask] if text_embeddings else None
                    uncond_masked = uncond_embeddings[mask] if uncond_embeddings else None
                    
                    # Batch-efficient CFG with mask-aware concatenation
                    x_in = torch.cat([x_masked, x_masked])
                    t_in = torch.cat([t_masked, t_masked])
                    emb_in = torch.cat([uncond_masked, emb_masked])
                    
                    preds = expert(x_in, t_in, emb_in).chunk(2)
                    pred = preds[0] + cfg_scale * (preds[1] - preds[0])
                else:
                    pred = expert(x_masked, t_masked, text_embeddings[mask] if text_embeddings else None)
                
                # Accumulate weighted predictions
                weight = topk_weights[mask, i].view(-1, 1, 1, 1)
                combined_pred[mask] += pred * weight

        # DDIM update with proper dimension handling
        x = ddim_step(
            lambda x_t, t, c: combined_pred,
            x,
            timestep,
            torch.full_like(timestep, t+1) if t < num_steps-1 else None,
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
        batch_size = x.shape[0]
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

def ddm_sample_optimized(router, experts, shape, **kwargs):
    """
    Memory-optimized version that quantizes models
    """
    # Quantize models
    quantized_router = quantize_model_for_inference(router)
    quantized_experts = {
        idx: quantize_model_for_inference(expert)
        for idx, expert in experts.items()
    }
    
    return ddm_sample(
        router=quantized_router,
        experts=quantized_experts,
        shape=shape,
        **kwargs
    ) 