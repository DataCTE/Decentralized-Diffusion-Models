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

# test add
def ddm_sample(
    router,
    experts,
    shape,
    num_steps=50,
    device=None,
    cfg_scale=7.5,
    text_embeddings=None,
    uncond_embeddings=None,
    eta=0.0,
    verbose=True,
    temperature=1.0,
    inference_strategy="top_k",  # Add strategy parameter from paper
    top_k=1,                     # For top-k selection
    top_p=0.9,                   # For nucleus sampling
    true_clusters=None,          # For oracle evaluation
):
    """DDM sampling with multiple inference strategies"""
    if device is None:
        device = next(router.parameters()).device
    
    x = torch.randn(shape, device=device)
    num_clusters = len(experts)
    alphas, alpha_bar, _ = get_alphas_and_betas(num_steps, "cosine")
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    batch_size = shape[0]
    if text_embeddings is not None:
        text_embeddings = text_embeddings[:batch_size]
        uncond_embeddings = uncond_embeddings[:batch_size] if uncond_embeddings is not None else None

    for t in tqdm(range(num_steps), disable=not verbose):
        timestep = torch.full((x.size(0),), t, device=device)
        
        # Get router predictions
        router_logits = router(x, timestep, text_embeddings)
        router_weights = F.softmax(router_logits / temperature, dim=-1)
        
        # Paper's inference strategies (Section 3.5)
        if inference_strategy == "full":
            # Full ensemble - use all experts
            selected_weights = router_weights
            selected_indices = torch.arange(num_clusters, device=device).expand(batch_size, -1)
        elif inference_strategy == "top_k":
            # Top-k experts
            selected_weights, selected_indices = router_weights.topk(top_k, dim=-1)
            selected_weights = F.softmax(selected_weights / temperature, dim=-1)
        elif inference_strategy == "sample":
            # Stochastic sampling
            if torch.isnan(router_weights).any() or torch.isinf(router_weights).any() or (router_weights < 0).any():
                #print("Warning: Invalid values detected in router_weights before multinomial sampling!")
                #print("router_weights min:", router_weights.min())
                #print("router_weights max:", router_weights.max())
                router_weights = torch.clamp(router_weights, min=0, max=1) # Clamp to valid probability range

            # Ensure router_weights are valid probabilities for multinomial sampling
            router_weights = torch.where(torch.isnan(router_weights) | torch.isinf(router_weights) | (router_weights < 0), torch.zeros_like(router_weights), router_weights)
            router_weights = torch.clamp(router_weights, min=0, max=1) # Double clamp to be safe

            if inference_strategy == "sample":
                #print("Inference strategy is sample")
                #print("Sum of router_weights:", router_weights.sum())
                #print("Min of router_weights:", router_weights.min())
                #print("Max of router_weights:", router_weights.max())
                #print("Shape of router_weights:", router_weights.shape)
                #print("Sample values of router_weights:", router_weights[:2])

                selected_indices = torch.multinomial(router_weights, 1).squeeze(-1)
                #print("Shape of selected_indices:", selected_indices.shape)
                #print("Type of selected_indices:", selected_indices.dtype)
                #print("Min value of selected_indices:", selected_indices.min())
                #print("Max value of selected_indices:", selected_indices.max())
                #print("Sample values of selected_indices:", selected_indices[:10])

            selected_weights = torch.ones_like(selected_indices, dtype=torch.float32)
        elif inference_strategy == "nucleus":
            # Nucleus sampling (top-p)
            sorted_weights, sorted_indices = torch.sort(router_weights, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_weights, dim=-1)
            mask = cum_probs <= top_p
            mask[..., 0] = True  # Ensure at least one expert
            selected_weights = sorted_weights[mask]
            selected_indices = sorted_indices[mask]
        elif inference_strategy == "oracle" and true_clusters is not None:
            # Oracle selection (for evaluation only)
            selected_indices = true_clusters
            selected_weights = torch.ones_like(selected_indices, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid inference strategy: {inference_strategy}")

        combined_pred = torch.zeros_like(x)
        active_experts = set()

        # Process selected experts - Batch-first approach
        for batch_idx in range(batch_size): # Iterate over batch dimension
            sample_pred = torch.zeros_like(x[batch_idx:batch_idx+1]) # Initialize prediction for this sample
            active_experts_sample = set() # Track active experts for this sample

            num_experts_per_sample = 0
            if inference_strategy == "sample":
                num_experts_per_sample = 1
            elif inference_strategy in ["top_k", "full", "nucleus"]:
                if selected_indices.ndim == 1: # Handle case where selected_indices is 1D for nucleus in some cases
                    num_experts_per_sample = 1
                elif selected_indices.ndim == 2:
                    num_experts_per_sample = selected_indices.size(1)
                else:
                    raise ValueError(f"Unexpected dimensions for selected_indices in {inference_strategy}: {selected_indices.ndim}")
            else:
                raise ValueError(f"Invalid inference strategy: {inference_strategy}")


            for i in range(num_experts_per_sample):
                expert_idx_sample = None
                if inference_strategy == "sample":
                    cluster_index_sample = selected_indices[batch_idx:batch_idx+1] # Get cluster index for this sample
                    expert_idx_sample = cluster_index_sample.item()
                elif inference_strategy in ["top_k", "full", "nucleus"]:
                    if selected_indices.ndim == 1:
                        cluster_index_sample = selected_indices[batch_idx:batch_idx+1] # Handle 1D case for nucleus
                        expert_idx_sample = cluster_index_sample[i].item() # Still need index 'i' even if 1D to align with loop
                    elif selected_indices.ndim == 2:
                        cluster_index_sample = selected_indices[batch_idx, i]
                        expert_idx_sample = cluster_index_sample.item()
                    else:
                        raise ValueError(f"Unexpected dimensions for selected_indices in {inference_strategy}: {selected_indices.ndim}")


                expert_idx = expert_idx_sample # Get the expert index for the current sample and expert iteration
                mask = torch.tensor([True], device=device) # Mask is now just for this sample (size 1)

                expert = experts[expert_idx]
                active_experts_sample.add(expert_idx)


                # Classifier-free guidance
                sample_x = x[batch_idx:batch_idx+1] # Get sample x
                if text_embeddings is not None and cfg_scale > 1.0:
                    x_in = torch.cat([sample_x, sample_x]) # Use sample_x here
                    t_in = timestep[batch_idx:batch_idx+1].repeat(2)
                    emb_in = torch.cat([uncond_embeddings[batch_idx:batch_idx+1], text_embeddings[batch_idx:batch_idx+1]])

                    preds = expert(x_in, t_in, emb_in).chunk(2)
                    pred = preds[0] + cfg_scale * (preds[1] - preds[0])
                else:
                    pred = expert(sample_x, timestep[batch_idx:batch_idx+1], text_embeddings[batch_idx:batch_idx+1])

                # Apply strategy-specific weighting
                weight = 1.0 # Weight is 1.0 per expert for now, adjust if needed for strategies other than sample
                if inference_strategy == "sample":
                    weight_sample = selected_weights[batch_idx:batch_idx+1].view(-1, 1, 1, 1) # Get weight for sample
                    weight = weight_sample # Assign sample-specific weight
                elif inference_strategy in ["top_k", "full", "nucleus"]:
                    if selected_weights.ndim == 1: # Handle 1D weights for nucleus if needed
                         weight_sample = selected_weights[batch_idx:batch_idx+1].view(-1, 1, 1, 1) # Get weight for sample
                         weight = weight_sample
                    elif selected_weights.ndim == 2:
                        weight_sample = selected_weights[batch_idx, i].view(-1, 1, 1, 1) # Get weight for sample and expert
                        weight = weight_sample
                    else:
                        raise ValueError(f"Unexpected dimensions for selected_weights in {inference_strategy}: {selected_weights.ndim}")


                sample_pred += pred * weight # Accumulate prediction for this sample
            combined_pred[batch_idx:batch_idx+1] = sample_pred # Assign sample prediction to combined prediction

        # DDIM update step
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