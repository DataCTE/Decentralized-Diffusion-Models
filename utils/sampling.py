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
        experts: Dict of expert models {expert_idx: model}
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
    # Check inputs
    if not isinstance(experts, dict):
        # Convert list to dict for backward compatibility
        experts = {i: expert for i, expert in enumerate(experts) if expert is not None}
    
    # Initialize with random noise
    x = torch.randn(shape, device=device)
    
    # Precompute diffusion parameters
    alphas, alpha_bar, betas = get_alphas_and_betas(steps, scheduler)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)
    betas = betas.to(device)
    
    # Convert to the timesteps we'll use for sampling
    timesteps = torch.linspace(1, 0, steps + 1).to(device) * 1000
    timesteps = timesteps.round().long()
    
    # Track active experts per step for metrics
    expert_usage_counts = torch.zeros(len(experts), device=device)
    
    # Setup progress bar
    progress = tqdm(range(steps), desc="Sampling", disable=not verbose)
    
    # Create caching structures for improved memory efficiency
    expert_cache = {}  # Cache for currently loaded experts
    expert_latents = {}  # Cache for expert outputs
    
    # Get batch size
    batch_size = shape[0]
    
    # Initialize memory tracker for adaptive expert loading
    max_experts_per_batch = min(len(experts), top_k * batch_size)
    available_memory = torch.cuda.get_device_properties(device).total_memory
    used_memory_baseline = torch.cuda.memory_allocated(device)
    
    # Setup for conditional generation
    use_cfg = text_embeddings is not None and uncond_embeddings is not None and cfg_scale > 1.0
    
    # For each sampling step
    for step_idx, step in enumerate(progress):
        # Current timestep
        t_idx = step
        t_next_idx = step + 1 if step + 1 < len(timesteps) else step
        
        # Calculate current and next timestep values
        t = timesteps[t_idx].unsqueeze(0).repeat(batch_size)
        t_next = timesteps[t_next_idx].unsqueeze(0).repeat(batch_size)
        
        # Scale t to [0, 1]
        t_scaled = t.float() / 1000.0
        
        with torch.no_grad():
            # Get router predictions for each sample in the batch
            router_preds = router(x, t_scaled)
            
            # Apply softmax to get probabilities
            router_probs = F.softmax(router_preds, dim=-1)  # [B, num_experts]
            
            # Get top-k expert indices and probabilities for each sample
            # This is done efficiently to minimize redundant expert loads
            all_selected_experts = []
            all_weights = []
            
            # Router selection with nucleus sampling option
            if getattr(router, 'nucleus_threshold', None) is not None and router.nucleus_threshold > 0:
                # Nucleus sampling - only use experts that cumulatively exceed a probability threshold
                for batch_idx in range(batch_size):
                    # Sort probabilities for this sample
                    probs, indices = router_probs[batch_idx].sort(descending=True)
                    
                    # Find experts that exceed threshold
                    cumsum = torch.cumsum(probs, dim=0)
                    mask = cumsum <= router.nucleus_threshold
                    
                    # Always include at least one expert
                    if not mask.any():
                        mask[0] = True
                    
                    # Get selected experts and their weights
                    selected_experts = indices[mask]
                    weights = probs[mask]
                    
                    # Normalize weights to sum to 1
                    weights = weights / weights.sum()
                    
                    all_selected_experts.append(selected_experts)
                    all_weights.append(weights)
            else:
                # Standard top-k selection
                for batch_idx in range(batch_size):
                    # Get top-k expert indices and probabilities for this sample
                    weights, indices = router_probs[batch_idx].topk(min(top_k, router_probs.shape[1]))
                    
                    # Normalize weights to sum to 1
                    weights = weights / weights.sum()
                    
                    all_selected_experts.append(indices)
                    all_weights.append(weights)
            
            # Get unique expert indices across all samples
            unique_experts = set()
            for experts_indices in all_selected_experts:
                unique_experts.update(experts_indices.tolist())
            
            # Check if we need to manage memory
            if len(unique_experts) > max_experts_per_batch:
                # Clear cache to free memory
                expert_cache.clear()
                torch.cuda.empty_cache()
                
                # Process in smaller batches
                batch_size_adjust = max(1, max_experts_per_batch // len(unique_experts))
                logger.warning(f"Too many unique experts ({len(unique_experts)}), splitting batch into {batch_size // batch_size_adjust} sub-batches")
                
                # Process in smaller sub-batches
                sub_batches = []
                for i in range(0, batch_size, batch_size_adjust):
                    end_idx = min(i + batch_size_adjust, batch_size)
                    sub_x = x[i:end_idx]
                    sub_t = t[i:end_idx]
                    if use_cfg:
                        sub_text_embeddings = text_embeddings[i:end_idx] if text_embeddings is not None else None
                        sub_uncond_embeddings = uncond_embeddings[i:end_idx] if uncond_embeddings is not None else None
                    
                    # Recursive call with smaller batch
                    sub_result = ddm_sample(
                        router=router,
                        experts=experts,
                        shape=sub_x.shape,
                        steps=1,  # Just do one step
                        top_k=top_k,
                        device=device,
                        cfg_scale=cfg_scale,
                        text_embeddings=sub_text_embeddings if use_cfg else None,
                        uncond_embeddings=sub_uncond_embeddings if use_cfg else None,
                        eta=eta,
                        scheduler=scheduler,
                        verbose=False
                    )
                    sub_batches.append(sub_result)
                
                # Combine sub-batches
                x = torch.cat(sub_batches, dim=0)
                continue
            
            # Load all unique experts needed for this step
            for expert_idx in unique_experts:
                if expert_idx not in expert_cache and expert_idx in experts and experts[expert_idx] is not None:
                    try:
                        # Track current memory usage for monitoring
                        mem_before = torch.cuda.memory_allocated(device)
                        
                        # Load expert (might already be on device)
                        expert = experts[expert_idx]
                        expert_cache[expert_idx] = expert
                        
                        # Track memory impact of loading this expert
                        mem_after = torch.cuda.memory_allocated(device)
                        expert_memory = mem_after - mem_before
                        
                        # Update max experts per batch based on actual memory usage
                        if expert_memory > 0:
                            available_memory_per_expert = (available_memory - used_memory_baseline) * 0.8 / expert_memory
                            max_experts_per_batch = min(max_experts_per_batch, int(available_memory_per_expert))
                            
                        logger.debug(f"Loaded expert {expert_idx} for sampling")
                    except Exception as e:
                        logger.error(f"Failed to load expert {expert_idx}: {e}")
                        expert_cache[expert_idx] = None
            
            # Process with conditional guidance if text embeddings are provided
            if use_cfg:
                # Do CFG with two forward passes (conditional and unconditional)
                
                # For memory efficiency, process each expert separately
                pred_noise_latent = torch.zeros_like(x)
                uncond_pred_noise_latent = torch.zeros_like(x)
                
                # Process each expert for all samples that need it
                for expert_idx in unique_experts:
                    if expert_idx not in expert_cache or expert_cache[expert_idx] is None:
                        continue
                    
                    # Find which samples use this expert
                    samples_using_expert = []
                    expert_weights = []
                    
                    for batch_idx in range(batch_size):
                        # Check if this sample uses this expert
                        if expert_idx in all_selected_experts[batch_idx]:
                            # Find weight for this expert
                            expert_pos = (all_selected_experts[batch_idx] == expert_idx).nonzero(as_tuple=True)[0]
                            weight = all_weights[batch_idx][expert_pos]
                            
                            samples_using_expert.append(batch_idx)
                            expert_weights.append(weight.item())
                    
                    if not samples_using_expert:
                        continue
                    
                    # Prepare batch for this expert
                    expert_batch_x = x[samples_using_expert]
                    expert_batch_t = t[samples_using_expert]
                    expert_batch_text = text_embeddings[samples_using_expert]
                    expert_batch_uncond = uncond_embeddings[samples_using_expert]
                    expert_weights = torch.tensor(expert_weights, device=device).view(-1, 1, 1, 1)
                    
                    # Forward pass with text condition
                    expert_output_cond = expert_cache[expert_idx](expert_batch_x, expert_batch_t, expert_batch_text)
                    
                    # Forward pass with unconditional embeddings
                    expert_output_uncond = expert_cache[expert_idx](expert_batch_x, expert_batch_t, expert_batch_uncond)
                    
                    # Apply CFG formula: pred = uncond + cfg_scale * (cond - uncond)
                    expert_output = expert_output_uncond + cfg_scale * (expert_output_cond - expert_output_uncond)
                    
                    # Apply expert weights and add to output latent
                    for idx, batch_idx in enumerate(samples_using_expert):
                        weight = expert_weights[idx]
                        pred_noise_latent[batch_idx] += expert_output[idx] * weight
                        
                        # Update usage stats
                        expert_usage_counts[expert_idx] += 1
                
                # Apply diffusion update
                x = update_sample(
                    x=x,
                    pred=pred_noise_latent,
                    t_index=t_idx,
                    t_next_index=t_next_idx,
                    alphas=alphas,
                    alpha_bar=alpha_bar,
                    eta=eta
                )
                
            else:
                # Standard non-guided generation
                # Process each expert for all samples that need it
                pred_noise_latent = torch.zeros_like(x)
                
                for expert_idx in unique_experts:
                    if expert_idx not in expert_cache or expert_cache[expert_idx] is None:
                        continue
                    
                    # Find which samples use this expert
                    samples_using_expert = []
                    expert_weights = []
                    
                    for batch_idx in range(batch_size):
                        # Check if this sample uses this expert
                        if expert_idx in all_selected_experts[batch_idx]:
                            # Find weight for this expert
                            expert_pos = (all_selected_experts[batch_idx] == expert_idx).nonzero(as_tuple=True)[0]
                            weight = all_weights[batch_idx][expert_pos]
                            
                            samples_using_expert.append(batch_idx)
                            expert_weights.append(weight.item())
                    
                    if not samples_using_expert:
                        continue
                    
                    # Process this batch with the expert
                    expert_batch_x = x[samples_using_expert]
                    expert_batch_t = t[samples_using_expert]
                    expert_weights = torch.tensor(expert_weights, device=device).view(-1, 1, 1, 1)
                    
                    # Forward pass with this expert
                    expert_output = expert_cache[expert_idx](
                        expert_batch_x, expert_batch_t, 
                        text_embeddings[samples_using_expert] if text_embeddings is not None else None
                    )
                    
                    # Apply expert weights and add to output latent
                    for idx, batch_idx in enumerate(samples_using_expert):
                        weight = expert_weights[idx]
                        pred_noise_latent[batch_idx] += expert_output[idx] * weight
                        
                        # Update usage stats
                        expert_usage_counts[expert_idx] += 1
                
                # Apply diffusion update
                x = update_sample(
                    x=x,
                    pred=pred_noise_latent,
                    t_index=t_idx,
                    t_next_index=t_next_idx,
                    alphas=alphas,
                    alpha_bar=alpha_bar,
                    eta=eta
                )
                
        # Clear expert cache after each step to save memory
        expert_cache.clear()
        torch.cuda.empty_cache()
        
        # Update progress bar
        if verbose:
            progress.set_postfix({
                "step": step_idx,
                "active_experts": len(unique_experts),
                "mem_used": f"{torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
            })
            
        # Call callback if provided
        if callback is not None:
            callback(x, step_idx)
    
    # Log expert usage statistics
    if verbose:
        # Calculate expert usage percentage
        expert_usage_pct = expert_usage_counts / (steps * batch_size) * 100
        top_experts = torch.topk(expert_usage_pct, min(5, len(expert_usage_pct)))
        
        logger.info("Expert usage statistics:")
        for i, (idx, pct) in enumerate(zip(top_experts.indices.tolist(), top_experts.values.tolist())):
            logger.info(f"  Top-{i+1}: Expert {idx} - {pct:.2f}%")
    
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