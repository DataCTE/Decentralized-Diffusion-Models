"""Sampling utilities for Decentralized Diffusion Models."""

import torch
import logging
from tqdm import tqdm
from trainers.diffusion import (
    get_alphas_and_betas, 
    ddim_step
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
    expert_cache_manager=None,
    temperature=1.0,
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
        
    Returns:
        Sampled batch of images
    """
    # Setup noise schedule
    alphas, alpha_bar, betas = get_alphas_and_betas(steps, scheduler)
    
    alpha_bar = alpha_bar.to(device)
    betas = betas.to(device)
    
    # Get batch size
    batch_size = shape[0]
    
    # Setup for conditional generation
    use_cfg = text_embeddings is not None and uncond_embeddings is not None and cfg_scale > 1.0
    
    # Initialize x with Gaussian noise (paper Section 3.5)
    x = torch.randn(shape, device=device)
    
    # Initialize expert usage tracking
    expert_usage = {k: 0 for k in experts.keys()}
    
    # Setup progress bar
    progress = tqdm(range(steps), desc="Sampling", disable=not verbose)
    
    # For each sampling step
    for t in progress:
        # Current timestep
        timestep = torch.full((batch_size,), t, device=device)
        
        # Get router predictions for which experts to use
        with torch.no_grad():
            # Get router probabilities (paper Equation 7)
            router_logits = router(x, timestep, text_embeddings)
            router_probs = torch.softmax(router_logits / temperature, dim=1)
            
            # Apply different expert selection strategies (paper Section 3.5)
            if top_k == 0:  # Full ensemble
                # Use all experts with their router probabilities
                selected_experts = list(experts.keys())
                expert_weights = router_probs
                # No expert fetching needed since all experts are used
            
            elif top_k > 0:  # Top-k selection
                # Get top-k expert indices and their probabilities
                topk_probs, topk_indices = router_probs.topk(min(top_k, router_probs.shape[1]), dim=1)
                
                # Normalize the top-k probabilities to sum to 1
                topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
                
                # Find unique experts across all batches
                selected_experts = set()
                for batch_idx in range(batch_size):
                    for k in range(topk_indices.shape[1]):
                        expert_idx = topk_indices[batch_idx, k].item()
                        selected_experts.add(expert_idx)
                selected_experts = sorted(list(selected_experts))
                
                # Create expert weights with zeros for non-selected experts
                expert_weights = torch.zeros_like(router_probs)
                for batch_idx in range(batch_size):
                    for k in range(topk_indices.shape[1]):
                        expert_idx = topk_indices[batch_idx, k].item()
                        expert_pos = selected_experts.index(expert_idx)
                        weight = topk_probs[batch_idx, k].item()
                        expert_weights[batch_idx, expert_idx] = weight
            
            # Fetch experts from cache if using cache manager
            if expert_cache_manager is not None:
                active_experts = {}
                for expert_idx in selected_experts:
                    expert = expert_cache_manager.get_expert(expert_idx, lambda: experts.get(expert_idx))
                    active_experts[expert_idx] = expert
                    expert_usage[expert_idx] += 1
            else:
                active_experts = {idx: experts[idx] for idx in selected_experts}
                for expert_idx in selected_experts:
                    expert_usage[expert_idx] += 1
            
            # Run conditional and unconditional forward passes for all selected experts
            expert_predictions = {}
            
            # For each selected expert
            for expert_idx, expert in active_experts.items():
                expert.eval()  # Ensure model is in eval mode
                
                if use_cfg:
                    # Concatenate conditional and unconditional embeddings
                    combined_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
                    combined_x = torch.cat([x] * 2, dim=0)
                    combined_timestep = torch.cat([timestep] * 2, dim=0)
                    
                    # Combined forward pass
                    combined_pred = expert(combined_x, combined_timestep, combined_embeddings)
                    
                    # Split predictions
                    uncond_pred, cond_pred = combined_pred.chunk(2, dim=0)
                    
                    # Apply classifier-free guidance
                    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                else:
                    # Simple forward pass without guidance
                    pred = expert(x, timestep, text_embeddings)
                
                expert_predictions[expert_idx] = pred
            
            # Create a list of predictions in the order of all experts
            all_expert_preds = []
            for expert_idx in range(router_probs.shape[1]):
                if expert_idx in expert_predictions:
                    all_expert_preds.append(expert_predictions[expert_idx])
                else:
                    # For inactive experts, create zero tensor (won't affect output due to zero weight)
                    all_expert_preds.append(torch.zeros_like(x))
            
            # Combine expert predictions according to router weights (paper Equation 7)
            combined_pred = torch.zeros_like(x)
            for i, expert_pred in enumerate(all_expert_preds):
                # Extract weights for this expert across all batches
                weights = expert_weights[:, i].view(-1, 1, 1, 1)
                combined_pred += weights * expert_pred
            
            # Release experts back to cache if needed
            if expert_cache_manager is not None:
                for expert_idx in selected_experts:
                    expert_cache_manager.release_expert(expert_idx)
            
            # Update progress with active expert info
            if verbose and t % 5 == 0:
                active_str = ", ".join([f"E{idx}:{expert_weights[:,idx].mean().item():.2f}" for idx in selected_experts])
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