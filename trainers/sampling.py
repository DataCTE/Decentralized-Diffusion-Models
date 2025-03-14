"""Sampling utilities for Decentralized Diffusion Models."""

import torch
import logging
from tqdm import tqdm
from trainers.diffusion import (
    get_alphas_and_betas, 
    ddim_step,
    update_sample
)
import torch.nn.functional as F

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
    Sample from Decentralized Diffusion Models with improved expert selection
    
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
        expert_cache_manager: Optional ExpertCacheManager for memory-efficient expert loading
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
    
    # Initialize x
    x = torch.randn(shape, device=device)
    
    # Initialize expert usage tracking
    expert_usage_counts = torch.zeros(len(experts), device=device)
    
    # Setup progress bar
    progress = tqdm(range(steps), desc="Sampling", disable=not verbose)
    
    # For each sampling step
    for t in progress:
        # Current timestep
        timestep = torch.full((batch_size,), t, device=device)
        
        # Get router predictions for which experts to use
        with torch.no_grad():
            # Get router probabilities
            router_outputs = router(x, timestep, text_embeddings, temperature=temperature)
            
            # Apply softmax to get probability distribution
            router_probs = torch.softmax(router_outputs, dim=1)
            
            # Paper Section 3.5: Ensemble predictions weighted by router probabilities
            if top_k == 0:  # Use all experts (full ensemble)
                # Use all experts with their router probabilities
                expert_weights = router_probs
                selected_experts = list(range(len(experts)))
                
                # Prepare for weighted combination
                combined_pred = torch.zeros_like(x)
                
                # Get predictions from all experts
                for expert_idx in selected_experts:
                    # Get expert
                    if expert_cache_manager:
                        expert = expert_cache_manager.get_expert(
                            expert_idx, 
                            lambda idx: experts[idx]
                        )
                    else:
                        expert = experts[expert_idx]
                        
                    # Get prediction
                    if use_cfg:
                        # Double the batch for conditional and unconditional inputs
                        x_in = torch.cat([x] * 2)
                        timestep_in = torch.cat([timestep] * 2)
                        emb_in = torch.cat([uncond_embeddings, text_embeddings])
                        
                        # Get prediction
                        pred = expert(x_in, timestep_in, emb_in)
                        
                        # Split for classifier-free guidance
                        uncond_pred, cond_pred = pred.chunk(2)
                        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                    else:
                        # Single prediction
                        pred = expert(x, timestep, text_embeddings)
                        
                    # Weight by router probability and accumulate
                    for i in range(batch_size):
                        weight = expert_weights[i, expert_idx]
                        combined_pred[i] += weight * pred[i]
                
                # Use combined prediction
                denoised = combined_pred
                
            elif top_k == 1:  # Top-1 selection (most efficient)
                # Get top expert for each sample
                selected_experts = router_probs.argmax(dim=1).tolist()
                selected_weights = [1.0] * batch_size
                
                # Efficient single expert selection
                denoised = torch.zeros_like(x)
                
                # Group samples by expert for efficiency
                expert_to_samples = {}
                for sample_idx, expert_idx in enumerate(selected_experts):
                    if expert_idx not in expert_to_samples:
                        expert_to_samples[expert_idx] = []
                    expert_to_samples[expert_idx].append(sample_idx)
                
                # Process each expert once with all its samples
                for expert_idx, sample_indices in expert_to_samples.items():
                    # Get expert
                    if expert_cache_manager:
                        expert = expert_cache_manager.get_expert(
                            expert_idx,
                            lambda idx: experts[idx]
                        )
                    else:
                        expert = experts[expert_idx]
                        
                    if not sample_indices:
                        continue
                        
                    # Get predictions for this expert's samples
                    if use_cfg:
                        # Prepare conditional and unconditional inputs
                        uncond_indices = sample_indices
                        cond_indices = [idx + batch_size for idx in sample_indices]
                        
                        # Double the batch
                        x_double = torch.cat([x] * 2)
                        timestep_double = torch.cat([timestep] * 2)
                        emb_double = torch.cat([uncond_embeddings, text_embeddings])
                        
                        # Gather the relevant samples
                        x_gather = torch.cat([x_double[uncond_indices], x_double[cond_indices]])
                        t_gather = torch.cat([timestep_double[uncond_indices], timestep_double[cond_indices]])
                        emb_gather = torch.cat([emb_double[uncond_indices], emb_double[cond_indices]])
                        
                        # Get predictions
                        expert_preds = expert(x_gather, t_gather, emb_gather)
                        
                        # Split into conditional and unconditional
                        half = len(sample_indices)
                        uncond_preds = expert_preds[:half]
                        cond_preds = expert_preds[half:]
                        
                        # Apply classifier-free guidance
                        guided_preds = uncond_preds + cfg_scale * (cond_preds - uncond_preds)
                        
                        # Place in final result
                        for i, idx in enumerate(sample_indices):
                            denoised[idx] = guided_preds[i]
                    else:
                        # Get the relevant samples
                        x_gather = x[sample_indices]
                        t_gather = timestep[sample_indices]
                        emb_gather = text_embeddings[sample_indices] if text_embeddings is not None else None
                        
                        # Get predictions
                        expert_preds = expert(x_gather, t_gather, emb_gather)
                        
                        # Place in final result
                        for i, idx in enumerate(sample_indices):
                            denoised[idx] = expert_preds[i]
            
            else:  # Top-k selection
                # Get top-k experts and their weights
                top_values, top_indices = torch.topk(router_probs, k=min(top_k, router_probs.size(1)), dim=1)
                
                # Renormalize weights to sum to 1
                top_weights = top_values / top_values.sum(dim=1, keepdim=True)
                
                # Convert to lists for easier handling
                selected_experts = [top_indices[i].tolist() for i in range(batch_size)]
                selected_weights = [top_weights[i].tolist() for i in range(batch_size)]
                
                # Initialize output tensor
                denoised = torch.zeros_like(x)
                
                # Process each sample individually with its top-k experts
                for i in range(batch_size):
                    experts_i = selected_experts[i]
                    weights_i = selected_weights[i]
                    
                    # Get weighted predictions from top-k experts
                    sample_pred = torch.zeros_like(x[i:i+1])
                    
                    for j, (expert_idx, weight) in enumerate(zip(experts_i, weights_i)):
                        # Get expert
                        if expert_cache_manager:
                            expert = expert_cache_manager.get_expert(
                                expert_idx,
                                lambda idx: experts[idx]
                            )
                        else:
                            expert = experts[expert_idx]
                            
                        # Get prediction
                        if use_cfg:
                            x_in = torch.cat([x[i:i+1]] * 2)
                            t_in = torch.cat([timestep[i:i+1]] * 2)
                            emb_in = torch.cat([uncond_embeddings[i:i+1], text_embeddings[i:i+1]])
                            
                            pred = expert(x_in, t_in, emb_in)
                            
                            # Apply CFG
                            uncond_pred, cond_pred = pred.chunk(2)
                            pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                        else:
                            pred = expert(x[i:i+1], timestep[i:i+1], 
                                          text_embeddings[i:i+1] if text_embeddings is not None else None)
                            
                        # Add weighted prediction
                        sample_pred += weight * pred
                        
                        # Update expert usage counts
                        expert_usage_counts[expert_idx] += 1
                    
                    # Store weighted prediction for this sample
                    denoised[i:i+1] = sample_pred
            
            # Update expert usage statistics for logging
            if verbose and t % 10 == 0:
                sorted_usage, sorted_indices = torch.sort(expert_usage_counts, descending=True)
                top_experts = [(idx.item(), count.item()) for idx, count in 
                            zip(sorted_indices[:3], sorted_usage[:3])]
                progress.set_postfix({"top_experts": str(top_experts)})
        
        # Apply classifier-free guidance
        if use_cfg:
            unconditional, conditional = denoised.chunk(2)
            denoised = unconditional + cfg_scale * (conditional - unconditional)
        
        # Update sample with new prediction
        x = update_sample(x, denoised, t, steps, eta, betas, alphas, alpha_bar)
        
        # Call callback if provided
        if callback is not None:
            callback(x, t)
    
    # Log expert usage statistics
    if verbose:
        expert_usage_pct = 100 * expert_usage_counts / steps / batch_size
        logger.info("Expert usage percentages:")
        for expert_idx, pct in enumerate(expert_usage_pct):
            logger.info(f"  Expert {expert_idx}: {pct:.1f}%")
    
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

def quantize_model_for_inference(model, dtype=torch.float16):
    """
    Quantize a model for more memory-efficient inference
    
    Args:
        model: The model to quantize
        dtype: Target dtype for quantization (default: float16)
        
    Returns:
        Quantized model
    """
    try:
        # Check if torch.ao.quantization is available (PyTorch ≥ 1.13)
        has_quantization = hasattr(torch, 'ao') and hasattr(torch.ao, 'quantization')
        
        if has_quantization:
            from torch.ao.quantization import quantize_dynamic
            try:
                # Try to use int8 dynamic quantization for maximum memory savings
                quantized_model = quantize_dynamic(
                    model, 
                    qconfig_spec={torch.nn.Linear},  # Quantize linear layers
                    dtype=torch.qint8
                )
                logger.info("Successfully applied int8 dynamic quantization")
                return quantized_model
            except Exception as e:
                logger.warning(f"Int8 quantization failed, falling back to {dtype}: {e}")
        
        # If int8 quantization is not available or failed, fall back to float16
        # This is a simpler approach but still saves memory
        model = model.to(dtype)
        logger.info(f"Converted model to {dtype} for efficient inference")
        return model
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        # Return original model
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