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
    # Performance optimizations:
    # config = get_config()  # Remove this line
    # Use the config passed to ddm_sample instead
    
    # 1. Reduce sampling steps for validation
    if steps > 20 and getattr(config, 'fast_validation', True):
        steps = 20  # Use fewer steps during validation
    
    # 2. Force evaluation mode
    router.eval()
    # Add device synchronization
    torch.cuda.synchronize(device=device)
    for expert in experts.values():
        if hasattr(expert, 'eval'):
            expert.eval()
    
    # Add expert count validation
    if len(experts) == 0:
        raise ValueError("No experts available for sampling")
    
    # Add expert availability check
    available_experts = [idx for idx in experts if experts[idx] is not None]
    if len(available_experts) == 0:
        raise RuntimeError("No valid experts found for sampling")
    
    # Add distributed barrier
    if dist.is_initialized():
        dist.barrier()
    
    # 3. Use no_grad context for the entire sampling
    with torch.no_grad():
        # Setup noise schedule
        alphas, alpha_bar, betas = get_alphas_and_betas(steps, scheduler)
        alphas = alphas.to(device)
        alpha_bar = alpha_bar.to(device)
        betas = betas.to(device)
        
        # Get batch size
        batch_size = shape[0]
        
        # Setup for conditional generation
        use_cfg = text_embeddings is not None and uncond_embeddings is not None and cfg_scale > 1.0
        
        # Initialize from random noise
        x = torch.randn(shape, device=device)
        
        # Setup progress bar
        progress = tqdm(range(steps), disable=not verbose)
        
        # Track expert usage for stats
        expert_usage = defaultdict(int)
        
        # For each timestep t
        for t in progress:
            if verbose and t == 0:
                logger.debug(f"Rank {torch.distributed.get_rank()} entered sampling loop")
            # Get timestep
            timestep = torch.tensor([t] * batch_size, device=device)
            
            # For classifier-free guidance, we need to do two forward passes
            if use_cfg:
                # Unconditional forward pass
                uncond_input = x
                uncond_timestep = timestep
                
                # Get router predictions for unconditional
                uncond_router_logits = router(uncond_input, uncond_timestep, text_embeddings=uncond_embeddings)
                uncond_router_probs = F.softmax(uncond_router_logits / temperature, dim=-1)
                
                # Get top-k experts for unconditional
                uncond_weights, uncond_indices = torch.topk(uncond_router_probs, top_k, dim=-1)
                
                # Normalize weights to sum to 1
                uncond_weights = uncond_weights / uncond_weights.sum(dim=-1, keepdim=True)
                
                # Get conditional input
                cond_input = x
                cond_timestep = timestep
                
                # Get router predictions for conditional
                cond_router_logits = router(cond_input, cond_timestep, text_embeddings)
                cond_router_probs = F.softmax(cond_router_logits / temperature, dim=-1)
                
                # Get top-k experts for conditional
                cond_weights, cond_indices = torch.topk(cond_router_probs, top_k, dim=-1)
                
                # Normalize weights to sum to 1
                cond_weights = cond_weights / cond_weights.sum(dim=-1, keepdim=True)
                
                # Combine the indices from both sets to get all unique experts we need
                if top_k > 0:
                    # Combine and deduplicate indices
                    all_indices = torch.cat([uncond_indices.flatten(), cond_indices.flatten()])
                    selected_experts = torch.unique(all_indices).cpu().tolist()
                else:
                    # Use all experts
                    selected_experts = list(experts.keys())
            else:
                # For unconditional generation, just use router directly
                # Create zero text embeddings for unconditional case
                batch_size = shape[0]
                zero_text_emb_router = torch.zeros((batch_size, config.clip_embedding_dim), dtype=torch.float32, device=device) # Zero text embeddings for router
                router_logits = router(x, timestep, text_embeddings=zero_text_emb_router)
                router_probs = F.softmax(router_logits / temperature, dim=-1)
                
                if top_k > 0:
                    # Get top-k experts
                    expert_weights, expert_indices = torch.topk(router_probs, top_k, dim=-1)
                    
                    # Normalize weights to sum to 1
                    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
                    
                    # Get unique experts to run
                    selected_experts = torch.unique(expert_indices.flatten()).cpu().tolist()
                else:
                    # Use all experts
                    selected_experts = list(experts.keys())
                    expert_weights = router_probs
            
            # Record expert usage
            for expert_idx in selected_experts:
                expert_usage[expert_idx] += 1
                
            # Get predictions from selected experts
            expert_predictions = {}
            
            # Apply experts in batches rather than one by one (more efficient)
            for expert_idx in selected_experts:
                # Skip if expert isn't available
                if expert_idx not in experts:
                    continue
                    
                # Get expert (through cache manager if provided)
                if expert_cache_manager is not None and callable(experts[expert_idx]):
                    expert = expert_cache_manager.get_expert(expert_idx, experts[expert_idx])
                else:
                    expert = experts[expert_idx]
                
                # Perform forward pass through expert
                if use_cfg:
                    # Conditional generation
                    uncond_pred = expert(uncond_input, uncond_timestep, uncond_embeddings)
                    cond_pred = expert(cond_input, cond_timestep, text_embeddings)
                    
                    # Combine with classifier-free guidance
                    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                else:
                    # Unconditional generation
                    zero_text_emb_expert = torch.zeros((batch_size, config.clip_embedding_dim), dtype=torch.float32, device=device) # Zero text embeddings for expert
                    pred = expert(x, timestep, zero_text_emb_expert) # Pass zero text embeddings to expert
                
                expert_predictions[expert_idx] = pred
            
            # Combine expert predictions according to router weights (paper Equation 7)
            combined_pred = torch.zeros_like(x)
            if top_k > 0:
                for k_idx in range(top_k): # Iterate up to top_k
                    expert_indices_k = expert_indices[:, k_idx] # Experts indices for k-th position in top-k for all batches
                    expert_weights_k = expert_weights[:, k_idx] # Weights for k-th position in top-k for all batches

                    for batch_index in range(expert_weights.shape[0]): # Iterate over batch dimension
                        expert_idx = expert_indices_k[batch_index].item() # Get expert index for this batch and top_k position
                        if expert_idx in expert_predictions: # Check if expert prediction exists
                            expert_pred = expert_predictions[expert_idx] # Get prediction for this expert
                            weight = expert_weights_k[batch_index].view(-1, 1, 1, 1) # Get weight for this batch and top_k position
                            combined_pred[batch_index] += weight * expert_pred[batch_index] # Accumulate weighted prediction
            else:
                for expert_idx, expert_pred in expert_predictions.items():
                    weights = expert_weights[:, expert_idx].view(-1, 1, 1, 1)
                    combined_pred += weights * expert_pred
            
            # Release experts back to cache if needed
            if expert_cache_manager is not None:
                for expert_idx in selected_experts:
                    expert_cache_manager.release_expert(expert_idx)
            
            # Update progress with active expert info
            if verbose and t % 5 == 0:
                active_str = ", ".join([f"E{selected_experts[i]}:{expert_weights[:,i].mean().item():.2f}" for i in range(len(selected_experts))])
                progress.set_postfix({"Active": active_str})
            
            # Sample step (DDIM)
            next_timestep = torch.full((batch_size,), t+1, device=device) if t < steps-1 else None
            
            if next_timestep is not None:
                # Update x using DDIM step
                x = ddim_step(
                    lambda x_t, t, c: combined_pred,  # Use precomputed prediction
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
    
    # Log expert usage stats
    if verbose:
        usage_str = ", ".join([f"E{idx}:{count}" for idx, count in sorted(expert_usage.items())])
        print(f"Expert usage: {usage_str}")
    
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