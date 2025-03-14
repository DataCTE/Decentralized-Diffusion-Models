"""Sampling utilities for Decentralized Diffusion Models."""

import torch
import logging
from tqdm import tqdm
from trainers.diffusion import (
    get_alphas_and_betas, 
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
    expert_cache_manager=None,
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
        expert_cache_manager: Optional ExpertCacheManager for memory-efficient expert loading
        
    Returns:
        Sampled batch of images
    """
    # Setup noise schedule
    if scheduler == "cosine":
        alpha_bar = cosine_beta_schedule(steps)  # alpha_cumprod
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]  # beta
    else:
        # Default linear schedule
        betas = torch.linspace(0.0001, 0.02, steps)
        alpha = 1 - betas
        alpha_bar = alpha.cumprod(0)
        
    alpha_bar = alpha_bar.to(device)
    betas = betas.to(device)
    
    # Convert to the timesteps we'll use for sampling
    timesteps = torch.linspace(1, 0, steps + 1).to(device) * 1000
    timesteps = timesteps.round().long()
    
    # Track active experts per step for metrics
    expert_usage_counts = torch.zeros(len(experts), device=device)
    
    # Setup progress bar
    progress = tqdm(range(steps), desc="Sampling", disable=not verbose)
    
    # Create memory-efficient expert management
    managed_experts = expert_cache_manager is not None
    
    # Get batch size
    batch_size = shape[0]
    
    # Setup for conditional generation
    use_cfg = text_embeddings is not None and uncond_embeddings is not None and cfg_scale > 1.0
    
    # Initialize x
    x = torch.randn(shape, device=device)
    
    # Initialize expert usage tracking
    expert_selections = {}  # Maps timestep to list of selected experts
    
    # For each sampling step
    for step_idx, step in enumerate(progress):
        # Current and next timesteps
        t = timesteps[step]
        t_next = timesteps[step + 1]
        
        # Create timesteps for the batch
        ts = t.expand(batch_size)
        
        # Process inputs in manageable sub-batches if needed
        # Determine if sub-batching is needed
        max_batch_size = getattr(router.config, 'max_inference_batch_size', batch_size)
        use_sub_batches = batch_size > max_batch_size
        
        # Process in sub-batches if needed
        if use_sub_batches:
            logger.info(f"Processing batch of size {batch_size} in sub-batches of max size {max_batch_size}")
            # Initialize storage for results
            sub_batch_results = torch.zeros_like(x)
            
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                sub_batch_size = end_idx - i
                
                # Extract sub-batch
                sub_x = x[i:end_idx]
                sub_ts = ts[i:end_idx]
                sub_text_emb = text_embeddings[i:end_idx] if use_cfg and text_embeddings is not None else None
                sub_uncond_emb = uncond_embeddings[i:end_idx] if use_cfg and uncond_embeddings is not None else None
                
                # Process sub-batch
                sub_result = process_ddm_batch(
                    router=router,
                    experts=experts,
                    x=sub_x,
                    t=sub_ts,
                    t_next=t_next,
                    text_embeddings=sub_text_emb,
                    uncond_embeddings=sub_uncond_emb,
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                    top_k=top_k,
                    step=step,
                    device=device,
                    expert_usage_counts=expert_usage_counts,
                    expert_selections=expert_selections,
                    expert_cache_manager=expert_cache_manager
                )
                
                # Store sub-batch results
                sub_batch_results[i:end_idx] = sub_result
                
            # Use combined results for this step
            x = sub_batch_results
        else:
            # Process entire batch at once
            x = process_ddm_batch(
                router=router,
                experts=experts,
                x=x,
                t=ts,
                t_next=t_next,
                text_embeddings=text_embeddings,
                uncond_embeddings=uncond_embeddings,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
                top_k=top_k,
                step=step,
                device=device,
                expert_usage_counts=expert_usage_counts,
                expert_selections=expert_selections,
                expert_cache_manager=expert_cache_manager
            )
        
        # Call the callback if provided
        if callback is not None:
            callback(x, step_idx)
            
        # Update progress
        if verbose:
            progress.set_postfix({"t": t.item()})
    
    # Log active expert usage
    if verbose:
        total_activations = expert_usage_counts.sum().item()
        expert_usage_pct = expert_usage_counts / max(1, total_activations) * 100
        
        logger.info(f"Expert usage distribution:")
        for expert_idx, usage_pct in enumerate(expert_usage_pct.cpu().numpy()):
            logger.info(f"  Expert {expert_idx}: {usage_pct:.1f}%")
    
    return x

def process_ddm_batch(
    router,
    experts,
    x,
    t,
    t_next,
    text_embeddings=None,
    uncond_embeddings=None,
    use_cfg=False,
    cfg_scale=7.5,
    top_k=1,
    step=0,
    device="cuda",
    expert_usage_counts=None,
    expert_selections=None,
    expert_cache_manager=None
):
    """
    Process a single batch for DDM sampling
    
    Args:
        router: Router model
        experts: Dict of expert models {expert_idx: model}
        x: Current noisy samples [B, C, H, W]
        t: Current timesteps [B]
        t_next: Next timestep (scalar)
        text_embeddings: Optional text embeddings for conditioning [B, seq_len, dim]
        uncond_embeddings: Optional unconditional embeddings [B, seq_len, dim]
        use_cfg: Whether to use classifier-free guidance
        cfg_scale: CFG scale
        top_k: Number of experts to use
        step: Current sampling step
        device: Device to use
        expert_usage_counts: Tensor to track expert usage
        expert_selections: Dict to track expert selections
        expert_cache_manager: Optional ExpertCacheManager
    
    Returns:
        Updated samples
    """
    batch_size = x.shape[0]
    managed_experts = expert_cache_manager is not None
    
    # Forward pass through router
    with torch.no_grad():
        # For cond + uncond if using CFG
        if use_cfg:
            # Process conditional
            cond_router_output = router(x, t, text_embeddings)
            # Process unconditional (empty/null embedding)
            uncond_router_output = router(x, t, uncond_embeddings)
            # Get expert weights
            expert_weights = (1 + cfg_scale) * cond_router_output - cfg_scale * uncond_router_output
        else:
            # Process without CFG
            expert_weights = router(x, t, text_embeddings)
        
        # Get top experts for each sample
        if top_k > 1:
            # Select top-k experts per sample
            topk_weights, topk_indices = torch.topk(expert_weights, min(top_k, expert_weights.size(1)), dim=1)
            # Normalize weights
            topk_weights = torch.softmax(topk_weights, dim=1)
        else:
            # Select single best expert per sample
            topk_weights, topk_indices = torch.max(expert_weights, dim=1)
            # Reshape to match expected dimensions
            topk_weights = topk_weights.view(-1, 1)
            topk_indices = topk_indices.view(-1, 1)
        
        # Track which experts are used
        unique_experts = torch.unique(topk_indices).cpu().numpy()
        
        # Update tracking
        if expert_usage_counts is not None:
            for idx in unique_experts:
                expert_usage_counts[idx] += torch.sum(topk_indices == idx).item()
        
        if expert_selections is not None:
            expert_selections[step] = unique_experts.tolist()
        
        # Prepare memory-efficient expert loading
        expert_predictions = {}
        
        # Load experts and get predictions
        for expert_idx in unique_experts:
            # Get expert model (from cache manager if available)
            if managed_experts:
                expert = expert_cache_manager.get_expert(
                    expert_idx=expert_idx, 
                    expert_builder=lambda idx: experts[idx] if idx in experts else None
                )
            else:
                expert = experts[expert_idx] if expert_idx in experts else None
            
            if expert is None:
                logger.warning(f"Expert {expert_idx} not found")
                continue
            
            # Run expert with conditioning
            if use_cfg:
                # Process conditional
                with torch.no_grad():
                    cond_flow = expert(x, t, text_embeddings)
                    
                # Process unconditional
                with torch.no_grad():
                    uncond_flow = expert(x, t, uncond_embeddings)
                    
                # Apply classifier-free guidance
                expert_flow = uncond_flow + cfg_scale * (cond_flow - uncond_flow)
            else:
                # Process without CFG
                with torch.no_grad():
                    expert_flow = expert(x, t, text_embeddings)
            
            # Save prediction
            expert_predictions[expert_idx] = expert_flow
        
        # Combine expert predictions based on router weights
        combined_flow = torch.zeros_like(x)
        
        for sample_idx in range(batch_size):
            sample_weights = topk_weights[sample_idx]
            sample_experts = topk_indices[sample_idx]
            
            # Only use available predictions
            valid_mask = torch.tensor([idx.item() in expert_predictions for idx in sample_experts], 
                                     device=device)
            
            if not valid_mask.any():
                logger.warning(f"No valid experts for sample {sample_idx}")
                continue
                
            # Renormalize weights for available experts
            if valid_mask.any():
                valid_weights = sample_weights[valid_mask]
                valid_experts = sample_experts[valid_mask]
                
                if len(valid_weights) > 0:
                    valid_weights = valid_weights / valid_weights.sum()
                    
                    # Weighted sum of expert predictions
                    for i, (weight, expert_idx) in enumerate(zip(valid_weights, valid_experts)):
                        expert_idx = expert_idx.item()
                        if expert_idx in expert_predictions:
                            combined_flow[sample_idx] += weight * expert_predictions[expert_idx][sample_idx]
        
        # Update samples with combined flow
        updated_x = update_sample(x, combined_flow, t, t_next)
        
        return updated_x

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